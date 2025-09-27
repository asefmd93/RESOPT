import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pickle
import os
from simulator import simulate_training_step, FACC, DEFAULT_VOLTAGE


class HeliosGymEnv(gym.Env):

    def __init__(self,
                 summary_paths: list[str],
                 max_steps: int = 100,
                 unequal_splits: bool = False,
                 target_latency=0.01,
                 k_values=None,
                 ):
        super().__init__()

        self.summary_paths   = summary_paths
        self.n_models = len(self.summary_paths)
        self._norm_cache = {}

        self.k_values = np.array(k_values if k_values is not None else [0.90, 1.00, 1.10], dtype=float)
        self.k_weights = None
        self.max_steps       = max_steps
        self.unequal_splits  = unequal_splits
        self.target_latency = target_latency
        self._profile_idx    = 0
        self.step_count      = 0
        # Load the very first profile
        self._load_profiles(self._profile_idx)


        # Define discrete action bins
        self.dvfs_bins       = np.linspace(0.6, 1.0, 5)    # DVFS voltage/freq points
        self.onchip_bins     = np.array([8, 16, 32, 64])      # On-chip banks
        self.hbm_bins        = np.array([1, 2, 4, 6, 8])       # HBM channels
        self.partition_bins  = np.array([1,2,3,4,5,6,7,8])            # Pipeline stages
        self.dp_bins         = np.array([1,2,4])            # Data-parallel degree
        self.mem_clk_bins    = np.array([0.8, 0.9, 1.0, 1.5])  # Memory clock multiplier
        self.accum_bins      = np.array([1,2,4,8,16])         # Accumulation steps
        self.precision_bins  = np.array([8, 16, 32])             # Precision bits
        self.seq_bins       = np.array([128,256,512,1024])

        self.action_bins = [
            self.dvfs_bins,
            self.onchip_bins,
            self.hbm_bins,
            self.partition_bins,
            self.dp_bins,
            self.mem_clk_bins,
            self.accum_bins,
            self.precision_bins,
            self.seq_bins,
        ]
        # MultiDiscrete
        self.action_space = spaces.MultiDiscrete(
            [len(b) for b in self.action_bins]
        )


        low = np.concatenate([
            np.zeros(4, dtype=np.float32),
            np.array([0.0, 0.0, -2.0], np.float32),
            np.zeros(self.n_models, dtype=np.float32),
            np.array([-2.0], np.float32),
            np.zeros(9, dtype=np.float32),
        ])
        high = np.concatenate([
            np.ones(4, dtype=np.float32),
            np.array([1.0, 1.0, 2.0], np.float32),
            np.ones(self.n_models, dtype=np.float32),
            np.array([2.0], np.float32),
            np.ones(9, dtype=np.float32),
        ])
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)


        self.ema_beta = 0.7
        self._ema_lat = None
        self._ema_eng = None

    def _load_profiles(self, idx: int):

        summary_path = self.summary_paths[idx]
        with open(summary_path, 'rb') as f:
            data = pickle.load(f)
        # Extract the 4-D summary vector
        self.profile_summary = np.array(data['summary'], dtype=np.float32)


        raw_path = summary_path.replace('_summary.pkl', '_blocks.pkl')
        with open(raw_path, 'rb') as f:
            raw = pickle.load(f)
        # Extract the list of per‐block stats
        self.layer_profiles = raw['blocks']

    def _equal_partitions(self, n_stages: int):

        blocks = self.layer_profiles
        base, rem = divmod(len(blocks), n_stages)
        parts, i = [], 0
        for s in range(n_stages):
            take = base + (1 if s < rem else 0)
            parts.append(blocks[i:i + take])
            i += take
        return parts

    def _stage_cfgs_from_indices(self, a: np.ndarray):

        volt = float(self.dvfs_bins[a[0]])
        freq = float(FACC * volt)
        n = int(self.partition_bins[a[3]])
        cfgs = []
        for _ in range(n):
            cfgs.append({
                'core_dvfs': {'voltage': volt, 'freq': freq},
                'onchip_banks': int(self.onchip_bins[a[1]]),
                'hbm_channels': int(self.hbm_bins[a[2]]),
                'mem_clk_mul': float(self.mem_clk_bins[a[5]]),
                'accum_steps': int(self.accum_bins[a[6]]),
                'precision_bits': int(self.precision_bins[a[7]]),
            })
        return cfgs


    def _model_one_hot(self, idx: int) -> np.ndarray:
        v = np.zeros(self.n_models, dtype=np.float32)
        v[idx] = 1.0
        return v

    def _norm(self, x, lo, hi):
        return float(x - lo) / float(hi - lo + 1e-12)

    def _last_action_feats(self, a_idx):

        return np.array([
            self._norm(self.dvfs_bins[a_idx[0]], self.dvfs_bins[0], self.dvfs_bins[-1]),
            self._norm(self.onchip_bins[a_idx[1]], self.onchip_bins[0], self.onchip_bins[-1]),
            self._norm(self.hbm_bins[a_idx[2]], self.hbm_bins[0], self.hbm_bins[-1]),
            self._norm(self.partition_bins[a_idx[3]], self.partition_bins[0], self.partition_bins[-1]),
            self._norm(self.dp_bins[a_idx[4]], self.dp_bins[0], self.dp_bins[-1]),
            self._norm(self.mem_clk_bins[a_idx[5]], self.mem_clk_bins[0], self.mem_clk_bins[-1]),
            self._norm(self.accum_bins[a_idx[6]], self.accum_bins[0], self.accum_bins[-1]),
            self._norm(self.precision_bins[a_idx[7]], self.precision_bins[0], self.precision_bins[-1]),
            self._norm(self.seq_bins[a_idx[8]], self.seq_bins[0], self.seq_bins[-1]),
        ], dtype=np.float32)

    def _bootstrap_norm_bounds(self, n_samples: int = 256):

        import numpy as _np
        rng = _np.random.default_rng(0)
        lats, engs = [], []
        for _ in range(n_samples):
            a = _np.array([rng.integers(0, n) for n in self.action_space.nvec])
            n_stage = int(self.partition_bins[a[3]])
            parts = self._equal_partitions(n_stage)
            cfgs = self._stage_cfgs_from_indices(a)
            dp = int(self.dp_bins[a[4]])
            seq = int(self.seq_bins[a[8]])
            lat, eng, _ = simulate_training_step(parts, cfgs, dp, seq)
            if lat > 0 and eng > 0:
                lats.append(lat);
                engs.append(eng)
        lats = _np.array(lats);
        engs = _np.array(engs)
        lat_min, lat_max = _np.percentile(lats, [5, 95])
        en_min, en_max = _np.percentile(engs, [5, 95])
        p50 = float(_np.percentile(lats, 50))
        return float(lat_min), float(lat_max), float(en_min), float(en_max), float(p50)

    def _ensure_profile_stats(self):

        key = self.summary_paths[self._profile_idx]
        if key not in self._norm_cache:
            self._norm_cache[key] = self._bootstrap_norm_bounds(n_samples=256)
        return self._norm_cache[key]

    def reset(self, *, seed=None, **kwargs):

        # round-robin over models; (np.random.randint(self.n_models)) is also fine
        self._profile_idx = (self._profile_idx + 1) % len(self.summary_paths)
        self._load_profiles(self._profile_idx)
        self.step_count = 0

        # Get per-profile stats
        lat_min, lat_max, en_min, en_max, p50 = self._ensure_profile_stats()


        allowed = np.asarray(self.k_values, dtype=float)
        K_min = float(lat_min / (p50 + 1e-12))
        # keep only K >= K_min; no arbitrary 0.85 floor
        feasible = allowed[allowed >= K_min - 1e-9]
        if feasible.size == 0:

            feasible = np.array([allowed.max()], dtype=float)


        w = self.k_weights
        if w is None or len(w) != len(allowed):
            probs = np.ones_like(allowed, dtype=float) / len(allowed)
        else:
            probs = np.asarray(w, dtype=float) / (np.sum(w) + 1e-12)

        mask = np.isin(allowed, feasible)
        probs = (probs * mask) / (np.sum(probs * mask) + 1e-12)

        self._K = float(np.random.choice(feasible, p=probs[mask]))
        self.target_latency = self._K * p50
        self._goal_feat = float(np.clip(np.log(self.target_latency / (p50 + 1e-12)), -2.0, 2.0))


        self._ema_lat = None
        self._ema_eng = None


        self._last_action_idx = np.array([len(b) // 2 for b in self.action_bins], dtype=int)
        last_feats = self._last_action_feats(self._last_action_idx)


        model_1h = self._model_one_hot(self._profile_idx)


        obs = np.concatenate([
            self.profile_summary,  # (4)
            np.array([0.0, 0.0, 0.0], np.float32),  # lat_n, eng_n, rel_log_lat
            model_1h,  # (n_models)
            np.array([self._goal_feat], np.float32),  # goal
            last_feats,  # (9)
        ]).astype(np.float32)

        return obs, {}

    def step(self, action_idx):

        # Map each discrete choice to its real value
        config = {
            'dvfs'     : float(self.dvfs_bins[action_idx[0]]),
            'onchip'   : int(self.onchip_bins[action_idx[1]]),
            'hbm'      : int(self.hbm_bins[action_idx[2]]),
            'partition': int(self.partition_bins[action_idx[3]]),
            'dp'       : int(self.dp_bins[action_idx[4]]),
            'mem_clk'  : float(self.mem_clk_bins[action_idx[5]]),
            'accum'    : int(self.accum_bins[action_idx[6]]),
            'precision': int(self.precision_bins[action_idx[7]]),
            'seq' : int(self.seq_bins[action_idx[8]]),
        }
        #volt = config['dvfs']  # e.g. 0.8
        #freq = FACC * volt  # e.g. 1.5e9 * 0.8 = 1.2e9


        n = config['partition']
        partitions = self._equal_partitions(n)
        gpu_cfgs = self._stage_cfgs_from_indices(action_idx)

        latency, energy, _ = simulate_training_step(
            partitions,
            gpu_cfgs,
            config['dp'],
            config['seq']
        )
        # Per-profile bounds for normalization
        lat_min, lat_max, en_min, en_max, p50 = self._ensure_profile_stats()

        # Guard bad simulator returns
        if not (np.isfinite(latency) and np.isfinite(energy)) or latency <= 0 or energy <= 0:
            latency = max(float(latency), 1e-9)
            energy = max(float(energy), 1e-9)


        if self._ema_lat is None:
            self._ema_lat = float(latency)
            self._ema_eng = float(energy)
        else:
            b = float(self.ema_beta)
            self._ema_lat = b * self._ema_lat + (1.0 - b) * float(latency)
            self._ema_eng = b * self._ema_eng + (1.0 - b) * float(energy)

        lat_avg = self._ema_lat
        eng_avg = self._ema_eng

        # Instantaneous features for the agent’s observation
        lat_n_clip = np.clip((latency - lat_min) / (lat_max - lat_min + 1e-9), 0.0, 1.0)
        eng_n_clip = np.clip((energy - en_min) / (en_max - en_min + 1e-9), 0.0, 1.0)
        rel_log_lat = float(np.clip(np.log((lat_avg / self.target_latency) + 1e-9), -2.0, 2.0))

        throughput = config['dp'] * config['seq'] / max(latency, 1e-12)


        lat_n_u = (lat_avg - lat_min) / (lat_max - lat_min + 1e-9)
        eng_n_u = (eng_avg - en_min) / (en_max - en_min + 1e-9)

        ratio = lat_avg / (self.target_latency + 1e-12)
        band = 0.03  # ±3% SLO band

        def _nrm(val, arr):
            return float(val - float(arr.min())) / float(float(arr.max()) - float(arr.min()) + 1e-12)


        pen = (
                0.010 * _nrm(int(self.partition_bins[action_idx[3]]), self.partition_bins) +
                0.010 * _nrm(int(self.dp_bins[action_idx[4]]), self.dp_bins) +
                0.010 * _nrm(float(self.mem_clk_bins[action_idx[5]]), self.mem_clk_bins) +
                0.010 * _nrm(float(self.dvfs_bins[action_idx[0]]), self.dvfs_bins)
        )


        dev = abs(ratio - 1.0) / (band + 1e-12)
        dev = min(1.0, dev)

        # weights inside vs outside the band
        WE_IN = 0.70  # energy weight when within/under band
        WT_IN = 0.30  # target-hug weight when within/under band
        WE_MISS = 0.25  # energy weight when overshooting

        if ratio <= 1.0 + band:

            reward = 1.0 - (WE_IN * np.tanh(max(0.0, eng_n_u)) + WT_IN * dev) - pen
        else:

            miss = (ratio - 1.0) / (band + 1e-12)
            miss = min(1.0, miss)
            reward = 1.0 - (miss + WE_MISS * max(0.0, eng_n_u)) - pen

        reward = float(np.clip(reward, 0.0, 1.0))

        # update last action and build new obs
        self._last_action_idx = np.array(action_idx, dtype=int)
        last_feats = self._last_action_feats(self._last_action_idx)
        model_1h = self._model_one_hot(self._profile_idx)
        obs = np.concatenate([
            self.profile_summary,
            np.array([lat_n_clip, eng_n_clip, rel_log_lat], np.float32),
            model_1h,
            np.array([self._goal_feat], np.float32),
            last_feats,
        ]).astype(np.float32)


        self.step_count += 1
        #done = (self._slo_success >= self.slo_window) or (self.step_count >= self.max_steps)
        #terminated = (self._slo_success >= self.slo_window)
        terminated = False
        truncated = (self.step_count >= self.max_steps)
        info = {
            "latency": float(latency),  # <— add
            "energy": float(energy),
            "lat_avg": float(lat_avg),
            "eng_avg": float(eng_avg),
            'throughput': throughput,
            "violated_avg": bool(lat_avg > self.target_latency),
            'target_latency': float(self.target_latency),  # <— add
            'K_used': float(getattr(self, "_K", np.nan)),  # <— add
        }

        info["model"] = os.path.basename(self.summary_paths[self._profile_idx]).replace("_summary.pkl", "")
        info["K"] = float(getattr(self, "_K", np.nan))

        return obs, float(reward), terminated, truncated, info


    def render(self, mode='human'):

        pass
