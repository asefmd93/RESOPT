import math
import numpy as np
from typing import List, Tuple, Dict
ANNEAL_STEPS = 200_000
_global_step = 0
# System & Architectural Constants

INPUT_SEQ_LEN = 256

FACC            = 1.98e9
NUM_PE          = 16896
DEFAULT_PE_UTIL = 0.8
DEFAULT_VOLTAGE = 0.8

# Memory limits & bandwidths
MEM_CAP = {
    'OnChipBuffer': 100e6,      # bytes
    'DRAM':         180e9,       # bytes
}
ONCHIP_BW_PER_BANK = 40e9       # bytes/s per bank
MEM_BW = {'DRAM_CHIPLET_INDIVIDUAL': 1.0e12}  # bytes/s per channel

# Network latencies
LAT = {
    'PointToPoint': {
        'NVSwitch':  (1.5e-6, 96e9),
        'PCIe_Gen5': (4.5e-6, 28e9),
    }
}
HOPS = {'NVSwitch': 1, 'PCIe_Gen5': 2}

# --- B200 memory structure & host spill (capacity-aware) ---
BASE_HBM_STACKS          = 8
DRAM_CAP_PER_STACK_BYTES = MEM_CAP['DRAM'] / BASE_HBM_STACKS  # 22.5 GB/stack
HBM_PER_STACK_BW_BYTES   = MEM_BW['DRAM_CHIPLET_INDIVIDUAL']  # 1.0 TB/s per stack
PCIE_HOST_BW_BYTES       = 28e9  # PCIe Gen5 x16 ~128 GB/s; conservative BW for spill

# Energy coefficients
E_OP_COEFF      = 0.99e-12        # J per flop
E_MEM_COEFF     = {'OnChipBuffer': 1.22e-12, 'DRAM': 4e-11}
ENERGY_PER_BYTE = {'NVSwitch': 0.5e-12, 'PCIe_Gen5': 1.5e-12}

# Leakage & transition
P_LEAK_COEFF = {
    'Compute':          5.0,
    'OnChipBank':       0.1,
    'DRAMChipletBank':  0.5,
    'DeepSleep':        0.05,
}
P_MEM_ACTIVE_WATT          = 15.0    # W when memory is active
E_TRANSITION_GPU_STATE     = 1e-5    # J per GPU state transition

# Diminishingâ€returns model
ONCHIP_DIMINISHING_RETURNS_BANK_THRESHOLD = 32
ONCHIP_DIMINISHING_RETURNS_FACTOR         = 0.6
DRAM_BW_DIMINISHING_RETURNS_THRESHOLD     = 8
DRAM_BW_DIMINISHING_RETURNS_FACTOR        = 0.75

COMPUTE_MEM_OVERLAP_FACTOR = 0.2
COMM_COMPUTE_OVERLAP_BY_TOPO = {'NVSwitch': 0.3, 'PCIe_Gen5': 0.9}
WORKLOAD_FIXED_OVERHEAD    = 0.05

LEAKAGE_VOLTAGE_SENSITIVITY = 1.0
LEAKAGE_FREQ_SENSITIVITY    = 0.5


# Reward & Exploration Penalty Constants

TARGET_LATENCY = 0.01      # seconds
ENERGY_SCALE   = 100.0
LATENCY_WEIGHT = 0.4
ENERGY_WEIGHT  = 0.6
LATENCY_PENALTY = 5.0

# Small penalties to encourage exploration
PENALTY_DVFS        = 0.005
PENALTY_ONCHIP      = 0.005
PENALTY_HBM         = 0.005
PENALTY_GPUS        = 0.02
PENALTY_DP          = 0.0
PENALTY_MEMCLK      = 0.01
PENALTY_ACCUM_STEPS = 0.005
PENALTY_PREC_BITS   = 0.02

# Normalization bounds for exploration penalties
DVFS_MIN, DVFS_MAX         = 0.6, 1.0
ONCHIP_MIN, ONCHIP_MAX     = 8, 64
HBM_MIN, HBM_MAX           = 2, 8
GPUS_MIN, GPUS_MAX         = 1, 8
DP_MIN, DP_MAX             = 1, 8
MEMCLK_MIN, MEMCLK_MAX     = 0.8, 1.0
ACCUM_MIN, ACCUM_MAX       = 1, 8
PREC_MIN, PREC_MAX         = 16, 32


#   Communication helpers (allâ€reduce / ring)

def get_comm_cost(msg_bytes: float, dp_degree: int, topo: str) -> Tuple[float, float]:

    if msg_bytes <= 0 or dp_degree <= 0:
        return 0.0, 0.0
    alpha, beta = LAT['PointToPoint'].get(topo, (5e-6, 1e9))
    hops = HOPS.get(topo, 1)
    # time: Î±Â·hops + Î²Â·bytes / dp_degree
    t = alpha * hops + msg_bytes / (beta * dp_degree)
    # energy = Î²_eÂ·bytes * hops
    e = ENERGY_PER_BYTE.get(topo, 1e-12) * msg_bytes * hops
    return t, e

def ring_allreduce_cost(msg_bytes: float, dp_degree: int, topo: str) -> Tuple[float, float]:
    if msg_bytes <= 0 or dp_degree <= 1:
        return 0.0, 0.0
    alpha, beta = LAT['PointToPoint'].get(topo, (5e-6, 1e9))
    hops = HOPS.get(topo, 1)
    P = dp_degree
    t = 2 * (P - 1) * alpha * hops + 2 * ((P - 1) / P) * (msg_bytes / beta)
    e = ENERGY_PER_BYTE.get(topo, 1e-12) * msg_bytes * (2 * (P - 1))
    return t, e


#  Memory bandwidth & time

def calculate_onchip_bw(cfg: Dict) -> float:
    return cfg['onchip_banks'] * ONCHIP_BW_PER_BANK

def calculate_dram_bw(cfg: Dict) -> float:
    base = MEM_BW['DRAM_CHIPLET_INDIVIDUAL']
    ch = cfg['hbm_channels']
    # diminishing returns after threshold
    if ch <= DRAM_BW_DIMINISHING_RETURNS_THRESHOLD:
        eff = base * ch
    else:
        extra = ch - DRAM_BW_DIMINISHING_RETURNS_THRESHOLD
        eff = base * (DRAM_BW_DIMINISHING_RETURNS_THRESHOLD + extra * DRAM_BW_DIMINISHING_RETURNS_FACTOR)
    return eff * cfg['mem_clk_mul']

def calculate_memory_time(dram_bytes: float, onchip_bytes: float, cfg: Dict) -> float:
    t_on = onchip_bytes / calculate_onchip_bw(cfg) if onchip_bytes > 0 else 0.0
    if dram_bytes <= 0.0:
        return t_on
    t_dr = dram_bytes / calculate_dram_bw(cfg)
    return max(t_on, t_dr)


#   Dynamic energy per layer

def calculate_dynamic_energy(
    ops: float,
    onchip_traffic: float,
    dram_traffic: float,
    t_mem: float,
    cfg: Dict
) -> float:
    # decode DVFS knob
    v = cfg['core_dvfs']['voltage']
    f = cfg['core_dvfs']['freq']
    vr = v / DEFAULT_VOLTAGE
    fr = f / FACC

    # precision scaling
    precision_factor = cfg.get('precision_bits', 32) / 32.0

    # compute energy
    e_comp = ops * E_OP_COEFF * (vr ** 2) * precision_factor
    memclk = max(0.1, float(cfg.get('mem_clk_mul', 1.0)))
    # memory access energy
    e_mem  = (
        onchip_traffic * E_MEM_COEFF['OnChipBuffer'] +
        dram_traffic   * E_MEM_COEFF['DRAM']
    ) * memclk
    # static memory energy while active
    stacks = max(1, int(cfg.get('hbm_channels', BASE_HBM_STACKS)))
    e_static = (P_MEM_ACTIVE_WATT * (stacks / BASE_HBM_STACKS) * memclk) * t_mem

    return e_comp + e_mem + e_static


#   Latency & energy prediction per layer

def predict_layer_perf(
    ops: float,
    bytes_moved: float,
    working_set: float,
    cfg: Dict
) -> Tuple[float, float]:
    # split working set
    on_ws   = min(working_set, MEM_CAP['OnChipBuffer'])
    on_tr   = bytes_moved * (on_ws / working_set) if working_set > 0 else 0.0
    dr_tr   = bytes_moved - on_tr

    # compute time
    comp_speed = cfg['core_dvfs']['freq'] * NUM_PE * DEFAULT_PE_UTIL
    t_comp = ops / comp_speed

    # memory time
    t_mem  = calculate_memory_time(dr_tr, on_tr, cfg)
    stacks = max(1, int(cfg.get('hbm_channels', BASE_HBM_STACKS)))
    eff_cap = DRAM_CAP_PER_STACK_BYTES * stacks
    if working_set > eff_cap:
        spill = working_set - eff_cap
        t_mem += spill / PCIE_HOST_BW_BYTES

    # overlap model
    latency = max(t_comp, t_mem) + COMPUTE_MEM_OVERLAP_FACTOR * min(t_comp, t_mem)
    # dynamic energy
    energy = calculate_dynamic_energy(ops, on_tr, dr_tr, t_mem, cfg)
    return latency, energy


#   Workload scaling with sequence length

def scale_workload(base_val: float, seq_len: int, power: float) -> float:
    if base_val <= 0.0:
        return 0.0
    ratio = seq_len / INPUT_SEQ_LEN
    return base_val * (WORKLOAD_FIXED_OVERHEAD + (1 - WORKLOAD_FIXED_OVERHEAD) * (ratio ** power))


#   Core simulation: one training step (pipeline + all-reduce + leakage + reward)

def simulate_training_step(
    partitions: List[List[Dict]],
    gpu_configs: List[Dict],
    dp_degree: int,
    seq_len: int,
    topo: str = 'NVSwitch'
) -> Tuple[float, float, float]:
    # select only active GPUs
    active = [(layers, cfg) for layers, cfg in zip(partitions, gpu_configs)
              if layers and cfg is not None]
    if not active:
        return 0.0, 0.0, -LATENCY_PENALTY

    # per-stage latency & dyn energy
    stage_lats = []
    dyn_energy = 0.0
    for layers, cfg in active:
        # forward
        t_stage = 0.0
        for direction in ('fwd', 'bwd'):
            ops_sum   = sum(scale_workload(layer[f'ops_{direction}'], seq_len, 1.0)
                            for layer in layers)
            bm_sum    = sum(scale_workload(layer[f'bytes_mv_{direction}'], seq_len, 1.0)
                            for layer in layers)
            # working set calc
            ws_list   = [
                layer['param_bytes'] + 2 * layer[f'bytes_mv_{direction}']
                for layer in layers
            ]
            ws_max    = max(ws_list) if ws_list else 0.0
            per_gpu_ops = ops_sum / max(1, dp_degree)
            per_gpu_bytes = bm_sum / max(1, dp_degree)
            per_gpu_ws = ws_max if ws_max <= MEM_CAP['OnChipBuffer'] else max(
                ws_max * (per_gpu_bytes / (bm_sum + 1e-9)), MEM_CAP['OnChipBuffer'])

            lat, en = predict_layer_perf(per_gpu_ops, per_gpu_bytes, per_gpu_ws, cfg)
            t_stage += lat
            dyn_energy += en
        stage_lats.append(t_stage)

    S = max(1, len(active))
    try:
        M = int(gpu_configs[0].get('accum_steps', 1))
    except Exception:
        M = 1
    bubble_frac = 0.0 if S <= 1 else (S - 1) / (M + S - 1)
    pipe_lat = max(stage_lats) + bubble_frac * float(np.mean(stage_lats))

    # gradientâ€sync cost
    grad_per_stage = []
    for layers, _ in active:
        grad_per_stage.append(sum(layer['grad_bytes'] for layer in layers))
    grad_bytes = max(grad_per_stage) if grad_per_stage else 0.0

    t_sync_raw, e_sync_raw = ring_allreduce_cost(grad_bytes, dp_degree, topo)
    # if using gradient accumulation:
    accum = gpu_configs[0].get('accum_steps', 1)
    t_sync = t_sync_raw / accum
    e_sync = e_sync_raw / accum

    overlap = COMM_COMPUTE_OVERLAP_BY_TOPO.get(topo, 0.9)
    total_time   = pipe_lat + t_sync * overlap
    total_dyn_e  = dyn_energy * dp_degree + e_sync

    # leakage & transition
    leak_p = 0.0
    for _, cfg in active:
        vr = cfg['core_dvfs']['voltage'] / DEFAULT_VOLTAGE
        fr = cfg['core_dvfs']['freq']    / FACC
        leak_p += (
            P_LEAK_COEFF['Compute'] +
            cfg['onchip_banks'] * P_LEAK_COEFF['OnChipBank'] +
            cfg['hbm_channels'] * P_LEAK_COEFF['DRAMChipletBank']
        ) * (vr ** LEAKAGE_VOLTAGE_SENSITIVITY) * (fr ** LEAKAGE_FREQ_SENSITIVITY)
    leak_e = leak_p * total_time * dp_degree
    trans_e = len(active) * E_TRANSITION_GPU_STATE * dp_degree

    total_energy = total_dyn_e + leak_e + trans_e



    return total_time, total_energy, None

