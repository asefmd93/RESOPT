import os, re, glob, json, argparse
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment import HeliosGymEnv

try:
    from skopt import gp_minimize
    from skopt.space import Real, Integer
except Exception:
    gp_minimize = None
    class Real:
        def __init__(self, low, high): self.low, self.high = low, high
    class Integer:
        def __init__(self, low, high): self.low, self.high = low, high


try:
    import gym.spaces as spaces
except Exception:
    from gymnasium import spaces


SLO_BAND = 0.03
K_COLORS = {0.80: "#4C78A8", 1.00: "#F58518", 1.20: "#54A24B"}
MARKER_SIZE_RL = 260
MARKER_SIZE_BASE = 120


FOCUS_LIMITS = {
    "gpt2-large":           dict(xmax=0.10, ymax=15.0),
    "mistralai_Mistral-7B": dict(xmax=0.60, ymax=80.0),
    "meta-llama_Llama-13b": dict(xmax=1.30, ymax=220.0),
}



def base_key(name: str) -> str:

    return re.split(r"_bs\d+|_sl\d+", name)[0]

def pareto_front_2d(lat, en):

    lat = np.asarray(lat); en = np.asarray(en)
    order = np.argsort(lat)
    keep, best = [], np.inf
    for i in order:
        if en[i] < best - 1e-12:
            keep.append(i); best = en[i]
    return np.array(keep, dtype=int)

def baseline_in_band(rand_lat, rand_en, rand_tput, target, band=SLO_BAND, strategy="p25"):

    lat = np.asarray(rand_lat); en = np.asarray(rand_en)
    mask = (lat >= target*(1-band)) & (lat <= target*(1+band))
    if not np.any(mask): return None
    lat = lat[mask]; en = en[mask]
    tput = (np.asarray(rand_tput)[mask] if rand_tput is not None else None)
    order = np.argsort(en)
    if strategy == "min": i = order[0]
    elif strategy == "median": i = order[len(order)//2]
    else: i = order[max(0, int(0.25*len(order))-1)]  # p25
    return dict(lat=float(lat[i]), en=float(en[i]),
                tput=(None if tput is None else float(tput[i])))

def focus_limits(name, pf_lat, pf_en, hlats, hens):

    key = base_key(name)
    lim = FOCUS_LIMITS.get(name) or FOCUS_LIMITS.get(key) or {}
    def q98(a): return float(np.quantile(a, 0.98)) if len(a) else 0.0
    xmax_auto = max(q98(pf_lat), (max(hlats) if hlats else 0.0)) * 1.05
    ymax_auto = max(q98(pf_en),  (max(hens)  if hens  else 0.0)) * 1.05
    xmax = max(xmax_auto, float(lim.get("xmax", 0.0)))
    ymax = max(ymax_auto, float(lim.get("ymax", 0.0)))
    return xmax, ymax



def make_env(summary_paths: List[str], max_steps: int):
    def _init():
        return HeliosGymEnv(summary_paths=summary_paths,
                            max_steps=max_steps,
                            unequal_splits=False,
                            target_latency=1.0)   # overridden per-episode
    return _init

def _unwrap_dummy(venv):
    env = venv
    for _ in range(6):
        if hasattr(env, "envs"): return env
        env = getattr(env, "venv", None)
        if env is None: break
    raise RuntimeError("Could not unwrap to DummyVecEnv")

def force_profile_and_k(venv, model_idx: int, K: float):
    base = _unwrap_dummy(venv).envs[0]
    base._profile_idx = int(model_idx) - 1
    base.k_values = np.array([float(K)], dtype=float)

def venv_reset(venv):
    return venv.reset()

def step_policy_episode(venv, model, deterministic=True):
    obs = venv_reset(venv)
    fixed_action, _ = model.predict(obs, deterministic=deterministic)
    lat_sum = eng_sum = thr_sum = 0.0
    steps = 0; last_info = {}
    while True:
        obs, _, dones, infos = venv.step(fixed_action)
        info = infos[0]; last_info = info
        l = float(info.get("lat_avg", info.get("latency", 0.0)))
        e = float(info.get("eng_avg", info.get("energy", 0.0)))
        t = float(info.get("throughput", 0.0))
        lat_sum += l; eng_sum += e; thr_sum += t
        steps += 1
        if dones[0]: break
    return lat_sum/steps, eng_sum/steps, thr_sum/steps, np.array(fixed_action).reshape(-1), last_info

def step_random_episode(venv):
    base = _unwrap_dummy(venv).envs[0]
    fixed_action = base.action_space.sample()
    obs = venv_reset(venv)
    lat_sum = eng_sum = thr_sum = 0.0
    steps = 0; last_info = {}
    while True:
        obs, _, dones, infos = venv.step(np.array([fixed_action]))
        info = infos[0]; last_info = info
        l = float(info.get("lat_avg", info.get("latency", 0.0)))
        e = float(info.get("eng_avg", info.get("energy", 0.0)))
        t = float(info.get("throughput", 0.0))
        lat_sum += l; eng_sum += e; thr_sum += t
        steps += 1
        if dones[0]: break
    return lat_sum/steps, eng_sum/steps, thr_sum/steps, np.array(fixed_action, dtype=float), last_info

def _action_bounds_from_env(venv):
    env0 = _unwrap_dummy(venv).envs[0]
    space = env0.action_space

    if isinstance(space, spaces.Box):
        return [Real(float(l), float(h)) for l, h in zip(space.low, space.high)]
    if isinstance(space, spaces.MultiDiscrete):
        return [Integer(0, int(n) - 1) for n in space.nvec]
    if isinstance(space, spaces.Discrete):
        return [Integer(0, int(space.n) - 1)]
    if isinstance(space, spaces.MultiBinary):
        return [Integer(0, 1) for _ in range(int(space.n))]
    raise NotImplementedError(f"Unsupported action space: {type(space)}")

def _coerce_for_vecenv_action(venv, x):

    env0 = _unwrap_dummy(venv).envs[0]
    space = env0.action_space

    if isinstance(space, spaces.Discrete):
        a = int(np.clip(round(float(x[0])), 0, space.n - 1))
        return np.array([a], dtype=np.int64)

    if isinstance(space, spaces.MultiDiscrete):
        a = np.asarray(x, dtype=np.int64)
        a = np.clip(a, 0, space.nvec - 1)
        return a[None, :]

    if isinstance(space, spaces.MultiBinary):
        a = (np.asarray(x) > 0.5).astype(np.int8)
        a = np.clip(a, 0, 1)
        return a[None, :]

    if isinstance(space, spaces.Box):
        a = np.asarray(x, dtype=np.float32)
        a = np.clip(a, space.low, space.high)
        return a[None, :]

    raise NotImplementedError(f"Unsupported action space: {type(space)}")


def decode_action_from_indices(env, action_idx):

    a = np.asarray(action_idx, dtype=int).ravel()
    bins = [
        ("dvfs",           env.dvfs_bins),
        ("onchip_banks",   env.onchip_bins),
        ("hbm_stacks",     env.hbm_bins),
        ("partition",      env.partition_bins),
        ("dp_degree",      env.dp_bins),
        ("mem_clk_mul",    env.mem_clk_bins),
        ("accum_steps",    env.accum_bins),
        ("precision_bits", env.precision_bins),
        ("seq_len",        env.seq_bins),
    ]
    out = {}
    for i, (k, arr) in enumerate(bins):
        if i < len(a):
            val = arr[int(a[i])]
            out[k] = float(val) if np.asarray(val).dtype.kind in ("f",) else int(val)
    return out


def export_rl_actions_selected(df: pd.DataFrame, env, p50_map: dict, K_LIST: list, out_dir: str, band: float = SLO_BAND):

    os.makedirs(out_dir, exist_ok=True)
    rows = []
    models = sorted(df["model"].unique().tolist())
    for name in models:
        if name not in p50_map:
            continue
        p50 = float(p50_map[name])
        P_all = df[(df["model"] == name) & (df["tag"] == "policy")].copy()
        if P_all.empty:
            continue

        for K in K_LIST:
            tau = p50 * float(K)
            Pk = P_all[P_all["K_req"].eq(float(K))]
            if Pk.empty:
                continue

            inb = Pk[(Pk["avg_latency"] >= tau * (1 - band)) & (Pk["avg_latency"] <= tau * (1 + band))]
            if len(inb):
                best = inb.loc[inb["avg_energy"].idxmin()]
            else:
                # closest to τ by latency, then by min energy if needed
                i = (Pk["avg_latency"] - tau).abs().idxmin()
                best = Pk.loc[i]

            act = best.get("action_mean", None)
            dec = decode_action_from_indices(env, act) if isinstance(act, (list, tuple, np.ndarray)) else {}

            rows.append({
                "model": name,
                "K": float(K),
                "target_latency_tau_s": float(tau),
                "avg_latency_s": float(best["avg_latency"]),
                "avg_energy_J": float(best["avg_energy"]),
                "avg_throughput_sps": float(best.get("avg_throughput", np.nan)),
                # decoded knobs (columns start with knob_)
                **{f"knob_{k}": v for k, v in dec.items()},
                # raw indices for reproducibility
                "action_indices": json.dumps(list(map(int, act))) if isinstance(act, (list, tuple, np.ndarray)) else "",
            })

    out_csv = os.path.join(out_dir, "rl_actions_selected.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")



def _pick_best_row_at_K(sub: pd.DataFrame, tau: float) -> pd.Series | None:

    if sub.empty:
        return None
    inb = sub[(sub["avg_latency"] >= tau * (1 - SLO_BAND)) &
              (sub["avg_latency"] <= tau * (1 + SLO_BAND))]
    if len(inb):
        return inb.loc[inb["avg_energy"].idxmin()]
    # closest-to-τ
    return sub.loc[(sub["avg_latency"] - tau).abs().idxmin()]

def _decode_action_knobs_eff(env, action_idx):

    a = np.asarray(action_idx, dtype=int).ravel()
    # Env bin order: [dvfs, onchip, hbm, partition, dp, mem_clk, accum, precision, seq]
    dp    = int(env.dp_bins[int(a[4])])    if a.size > 4 else 1
    accum = int(env.accum_bins[int(a[6])]) if a.size > 6 else 1
    seq   = int(env.seq_bins[int(a[8])])   if a.size > 8 else 128
    return dp, accum, seq

def _tokens_metrics_from_row(row: pd.Series, env) -> tuple[float, float]:

    lat = float(row["avg_latency"])
    eng = float(row["avg_energy"])
    act = row.get("action_mean", None)
    dp, accum, seq = _decode_action_knobs_eff(env, act if isinstance(act, (list, tuple, np.ndarray)) else [])
    tokens_per_step = max(1, dp * accum * seq)
    tps = tokens_per_step / max(1e-9, lat)
    tpj = tokens_per_step / max(1e-12, eng)
    return tps, tpj

def export_efficiency_plots_rl_bo(df: pd.DataFrame, env, p50_map: dict, K_LIST: list, out_dir: str):

    os.makedirs(out_dir, exist_ok=True)
    models = sorted(df["model"].unique().tolist())
    for name in models:
        if name not in p50_map:
            continue
        tau_base = float(p50_map[name])
        sub = df[df["model"] == name]
        P = sub[sub["tag"] == "policy"]
        B = sub[sub["tag"] == "bo"]
        if P.empty or B.empty:
            continue

        xs = np.arange(len(K_LIST))
        width = 0.35
        rl_tps = []; bo_tps = []
        rl_tpj = []; bo_tpj = []

        for K in K_LIST:
            tau = tau_base * float(K)
            Pk = P[P["K_req"].eq(float(K))]
            Bk = B[B["K_req"].eq(float(K))]
            r = _pick_best_row_at_K(Pk, tau)
            b = _pick_best_row_at_K(Bk, tau)
            if r is None or b is None:
                rl_tps.append(np.nan); rl_tpj.append(np.nan)
                bo_tps.append(np.nan); bo_tpj.append(np.nan)
                continue
            r_tps, r_tpj = _tokens_metrics_from_row(r, env)
            b_tps, b_tpj = _tokens_metrics_from_row(b, env)
            rl_tps.append(r_tps); rl_tpj.append(r_tpj)
            bo_tps.append(b_tps); bo_tpj.append(b_tpj)
        # --- Write CSVs for Excel (exact values used in the bar plots) ---
        tau_base = float(p50_map[name])
        rows_tps = []
        rows_tpj = []
        for K, r_v, b_v in zip(K_LIST, rl_tps, bo_tps):
            rows_tps.append(dict(model=name, series="RL", K=float(K), value=float(r_v), tau_s=tau_base * float(K)))
            rows_tps.append(dict(model=name, series="BO", K=float(K), value=float(b_v), tau_s=tau_base * float(K)))
        for K, r_v, b_v in zip(K_LIST, rl_tpj, bo_tpj):
            rows_tpj.append(dict(model=name, series="RL", K=float(K), value=float(r_v), tau_s=tau_base * float(K)))
            rows_tpj.append(dict(model=name, series="BO", K=float(K), value=float(b_v), tau_s=tau_base * float(K)))

        csv_tps = os.path.join(out_dir, f"plot_tokens_per_sec_{name}.csv")
        csv_tpj = os.path.join(out_dir, f"plot_tokens_per_joule_{name}.csv")
        pd.DataFrame(rows_tps).to_csv(csv_tps, index=False)
        pd.DataFrame(rows_tpj).to_csv(csv_tpj, index=False)
        print(f"[OK] Saved {csv_tps}")
        print(f"[OK] Saved {csv_tpj}")

        def _draw(vals_rl, vals_bo, ylab, title, fname):
            fig, ax = plt.subplots(figsize=(8, 4.5))
            # RL (left) and BO (right) bars per K
            ax.bar(xs - width/2, vals_rl, width=width, label="RL",  color="#4C78A8", edgecolor="black", linewidth=0.7)
            ax.bar(xs + width/2, vals_bo, width=width, label="BO",  color="#F58518", edgecolor="black", linewidth=0.7)
            # Annotate RL vs BO %
            for xi, r, b in zip(xs, vals_rl, vals_bo):
                if not (np.isnan(r) or np.isnan(b)) and b != 0:
                    pct = (r/b - 1.0) * 100.0
                    ax.text(xi - width/2, r, f"{pct:+.0f}%", ha="center", va="bottom", fontsize=9)
            ax.set_xticks(xs); ax.set_xticklabels([f"K={k}" for k in K_LIST])
            ax.set_ylabel(ylab); ax.set_title(f"{title} — {name}")
            ax.legend(loc="best", framealpha=0.95)
            ax.grid(True, axis="y", alpha=0.3)
            fig.tight_layout()
            out_png = os.path.join(out_dir, fname)
            fig.savefig(out_png, dpi=300)
            plt.close(fig)
            print(f"[OK] Saved {out_png}")

        _draw(rl_tps, bo_tps, "Tokens/sec", "Tokens/sec (RL vs BO)", f"eff_tps_rl_bo_{name}.png")
        _draw(rl_tpj, bo_tpj, "Tokens/J",   "Tokens/J (RL vs BO)",  f"eff_tpj_rl_bo_{name}.png")



def _eval_once(venv, action):

    env0 = _unwrap_dummy(venv).envs[0]
    space = env0.action_space
    if isinstance(space, (spaces.MultiDiscrete, spaces.Discrete, spaces.MultiBinary)):
        act_to_env = np.asarray([np.array(action, dtype=int)])
    else:
        act_to_env = np.asarray([np.array(action, dtype=float)])
    obs = venv_reset(venv)
    L = E = T = 0.0
    n = 0
    done = False
    a_vec = _coerce_for_vecenv_action(venv, action)  # <-- important

    while not done:
        obs, _, dones, infos = venv.step(a_vec)
        info = infos[0]; done = dones[0]
        L += float(info.get("lat_avg",  info.get("latency",   0.0)))
        E += float(info.get("eng_avg",  info.get("energy",    0.0)))
        T += float(info.get("throughput", 0.0))
        n += 1

    n = max(n, 1)
    return L/n, E/n, T/n
def gather_bo_seeds(rows, model_name, K, tau, band=SLO_BAND, max_random=5, include_rl=True):

    seeds = []


    rnd = [r for r in rows
           if r.get("tag") == "random" and r.get("model") == model_name
           and r.get("avg_latency") is not None and r.get("avg_energy") is not None
           and (tau*(1-band) <= r["avg_latency"] <= tau*(1+band))
           and isinstance(r.get("action_mean"), (list, tuple))]
    rnd.sort(key=lambda r: r["avg_energy"])
    for r in rnd[:max_random]:
        seeds.append(list(r["action_mean"]))


    if include_rl:
        rl = [r for r in rows
              if r.get("tag") == "policy" and r.get("model") == model_name
              and float(r.get("K_req", float("nan"))) == float(K)
              and r.get("avg_latency") is not None
              and (tau*(1-band) <= r["avg_latency"] <= tau*(1+band))
              and isinstance(r.get("action_mean"), (list, tuple))]
        if rl:
            rl_best = min(rl, key=lambda r: r["avg_energy"])
            seeds.append(list(rl_best["action_mean"]))


    uniq = []
    seen = set()
    for a in seeds:
        key = tuple(int(x) if float(x).is_integer() else float(x) for x in a)
        if key not in seen:
            uniq.append(a); seen.add(key)
    return uniq


def run_bo_point(venv, model_idx, K, tau, band=0.03, n_init=10, n_steps=20, seeds_x=None):

    force_profile_and_k(venv, model_idx, float(K))
    dims = _action_bounds_from_env(venv)
    evals = []  # (x, L, E, T)

    def objective(x):
        L, E, T = _eval_once(venv, x)
        slack = max(0.0, abs(L - tau) - band * tau)
        penalty = 1e6 * (slack / max(tau, 1e-9))**2
        evals.append((x, L, E, T))
        return E + penalty

    seeds_x = seeds_x or []

    n_seed = len(seeds_x)
    n_init_eff = max(0, n_init - n_seed)
    n_calls = n_seed + n_init_eff + n_steps

    if gp_minimize is None:

        for x0 in seeds_x:
            objective(x0)

        def sample(d):
            lo, hi = d.low, d.high
            return int(np.random.randint(lo, hi + 1)) if isinstance(d, Integer) else float(np.random.uniform(lo, hi))

        for _ in range(n_init_eff + n_steps):
            objective([sample(d) for d in dims])
    else:
        gp_minimize(
            objective, dims,
            n_calls=n_calls,
            n_initial_points=n_init_eff,
            x0=seeds_x if n_seed > 0 else None,
            y0=None,
            random_state=0
        )


    feas = [(x,L,E,T) for (x,L,E,T) in evals if tau*(1-band) <= L <= tau*(1+band)]
    cand = feas if feas else sorted(evals, key=lambda r: abs(r[1] - tau))[:1]
    x, L, E, T = min(cand, key=lambda r: r[2])
    return dict(lat=L, eng=E, thr=T, act=x)
# ============================================================================
def _write_pareto_plot_csv(name: str, pf_lat, pf_en, point_rows: list, out_dir: str, band: float = SLO_BAND):

    rows = []
    # Pareto front
    for x, y in zip(pf_lat, pf_en):
        rows.append(dict(model=name, series="front", K="", latency_s=float(x), energy_J=float(y),
                         tau_s="", band_low_s="", band_high_s=""))

    # Points & targets
    for r in point_rows:
        typ = str(r["type"]).lower()
        if typ in ("baseline", "rl", "bo"):
            rows.append(dict(
                model=name, series=typ, K=float(r["K"]),
                latency_s=float(r["latency_plot"]),
                energy_J=float(r["energy"]),
                tau_s=float(r["target_latency"]),
                band_low_s=float(r["target_latency"])*(1.0-band),
                band_high_s=float(r["target_latency"])*(1.0+band),
            ))
        elif typ == "target":
            tau = float(r["target_latency"])
            rows.append(dict(
                model=name, series="target", K=float(r["K"]),
                latency_s=tau, energy_J="",
                tau_s=tau, band_low_s=tau*(1.0-band), band_high_s=tau*(1.0+band)
            ))

    out_csv = os.path.join(out_dir, f"plot_pareto_{name}.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[OK] Saved {out_csv}")

def export_paper_csvs(df: pd.DataFrame, out_dir: str) -> Dict[str, Dict[str, float]]:

    os.makedirs(out_dir, exist_ok=True)
    models = sorted(df["model"].unique().tolist())
    all_pf_rows, all_point_rows = [], []


    p50_map = {}
    for name in models:
        subp = df[(df["model"] == name) & (df["tag"] == "policy")]
        if subp.empty: continue
        p50_map[name] = float(np.median(subp["target_latency"] / subp["K_req"]))

    for name in models:
        sub = df[df["model"] == name]
        R_all = sub[sub["tag"] == "random"]
        P_all = sub[sub["tag"] == "policy"]

        Ks = sorted(P_all["K_req"].dropna().unique().tolist())
        B_all = sub[sub["tag"] == "bo"]

        # Compute front from ALL points
        lat_all = sub["avg_latency"].to_numpy()
        en_all  = sub["avg_energy"].to_numpy()
        pf_idx  = pareto_front_2d(lat_all, en_all)
        pf_lat, pf_en = lat_all[pf_idx], en_all[pf_idx]
        order = np.argsort(pf_lat); pf_lat, pf_en = pf_lat[order], pf_en[order]

        # Save per-model front CSV
        front_csv = os.path.join(out_dir, f"paper_pareto_front_{name}.csv")
        pd.DataFrame({"model": name, "latency_s": pf_lat, "energy_J": pf_en}).to_csv(front_csv, index=False)


        rows = []; H_LAT, H_EN = [], []
        for K in Ks:
            tau = float(p50_map[name]) * float(K)

            Pk = P_all[P_all["K_req"].eq(float(K))]
            feas = Pk[Pk["avg_latency"] <= tau + 1e-12]
            if len(feas):
                star = feas.loc[feas["avg_energy"].idxmin()]
                lat_plot = tau                          # snap to τ for the figure
            else:
                star = Pk.loc[(Pk["avg_latency"] - tau).abs().idxmin()]
                lat_plot = float(star["avg_latency"])   # no snap if infeasible

            rl = dict(lat_plot=float(lat_plot),
                      lat_meas=float(star["avg_latency"]),
                      en=float(star["avg_energy"]),
                      tput=float(star.get("avg_throughput", np.nan)),
                      tau=float(tau))

            base = baseline_in_band(
                R_all["avg_latency"].to_numpy(),
                R_all["avg_energy"].to_numpy(),
                R_all["avg_throughput"].to_numpy() if "avg_throughput" in R_all else None,
                tau, band=SLO_BAND, strategy="p25"
            )

            dE = (100.0*(1.0 - rl["en"]/base["en"])) if (base and base["en"]>0) else np.nan

            # baseline row
            rows.append(dict(model=name, K=float(K), type="baseline",
                             latency_plot=(base["lat"] if base else np.nan),
                             latency_measured=(base["lat"] if base else np.nan),
                             energy=(base["en"] if base else np.nan),
                             target_latency=tau, deltaE_pct=np.nan))
            # RL row
            rows.append(dict(model=name, K=float(K), type="rl",
                             latency_plot=rl["lat_plot"], latency_measured=rl["lat_meas"],
                             energy=rl["en"], target_latency=tau, deltaE_pct=dE))

            Bk = B_all[B_all["K_req"].eq(float(K))]
            if not Bk.empty:
                inb = Bk[(Bk["avg_latency"] >= tau * (1 - SLO_BAND)) &
                         (Bk["avg_latency"] <= tau * (1 + SLO_BAND))]
                if len(inb):
                    bo = inb.loc[inb["avg_energy"].idxmin()]  # best energy in-band
                else:
                    bo = Bk.loc[(Bk["avg_latency"] - tau).abs().idxmin()]  # closest-to-τ

                rows.append(dict(
                    model=name, K=float(K), type="bo",
                    latency_plot=float(tau),  # snap to τ for the figure
                    latency_measured=float(bo["avg_latency"]),
                    energy=float(bo["avg_energy"]),
                    target_latency=tau, deltaE_pct=np.nan
                ))

            rows.append(dict(model=name, K=float(K), type="target",
                             latency_plot=tau, latency_measured=tau,
                             energy=np.nan, target_latency=tau, deltaE_pct=np.nan))


            if base: H_LAT.append(base["lat"]); H_EN.append(base["en"])
            H_LAT.extend([rl["lat_meas"], tau]); H_EN.extend([rl["en"], 0.0])

        points_csv = os.path.join(out_dir, f"paper_points_{name}.csv")
        pd.DataFrame(rows).to_csv(points_csv, index=False)
        # NEW: one CSV with exactly what the Pareto figure uses (front + points + targets)
        _write_pareto_plot_csv(name, pf_lat, pf_en, rows, out_dir, band=SLO_BAND)


        all_pf_rows.extend([{"model": name, "latency_s": float(x), "energy_J": float(y)}
                            for x, y in zip(pf_lat, pf_en)])
        all_point_rows.extend(rows)


        make_zoom_plot_from_csv(front_csv, points_csv,
                                os.path.join(out_dir, f"paper_{name}.png"))


    if all_pf_rows:
        pd.DataFrame(all_pf_rows).to_csv(os.path.join(out_dir, "paper_all_pareto_front.csv"), index=False)
    if all_point_rows:
        pd.DataFrame(all_point_rows).to_csv(os.path.join(out_dir, "paper_all_points.csv"), index=False)

    return p50_map


def make_zoom_plot_from_csv(front_csv: str, points_csv: str, out_png: str):
    pf  = pd.read_csv(front_csv)
    pts = pd.read_csv(points_csv)
    name = pf["model"].iloc[0] if "model" in pf.columns else os.path.basename(front_csv)[len("paper_pareto_front_"):-4]

    x = pf["latency_s"].to_numpy()
    y = pf["energy_J"].to_numpy()

    lat_col = "latency_plot" if "latency_plot" in pts.columns else "latency"
    H_LAT, H_EN = [], []

    plt.figure(figsize=(7.2, 4.8))
    ax = plt.gca()
    ax.plot(x, y, lw=3.0, color="#111111", label="Pareto front")

    shown = {"band": False, "target": False, "baseline": False, "rl": False, "bo": False}
    for K in sorted(pts["K"].dropna().unique().tolist()):
        sub = pts[pts["K"].eq(K)]
        tau = float(sub["target_latency"].iloc[0])
        c = K_COLORS.get(float(K), "#888888")

        ax.axvspan(tau*(1-SLO_BAND), tau*(1+SLO_BAND), color=c, alpha=0.09, lw=0,
                   label=None if shown["band"] else "SLO band (±3%)")
        shown["band"] = True
        ax.axvline(tau, color=c, ls="--", lw=1.8,
                   label=None if shown["target"] else "SLO target")
        shown["target"] = True

        b = sub[sub["type"]=="baseline"]
        if not b.empty and pd.notna(b["energy"].iloc[0]):
            bx = float(b[lat_col].iloc[0]); by = float(b["energy"].iloc[0])
            ax.scatter([bx], [by], marker="D", s=MARKER_SIZE_BASE, color=c,
                       edgecolors="black", linewidths=1.0,
                       label=None if shown["baseline"] else "Baseline (in-band, p25)")
            shown["baseline"] = True
            H_LAT.append(bx); H_EN.append(by)

        r = sub[sub["type"]=="rl"]
        if not r.empty and pd.notna(r["energy"].iloc[0]):
            rx = float(r[lat_col].iloc[0]); ry = float(r["energy"].iloc[0])
            ax.scatter([rx], [ry], marker="*", s=MARKER_SIZE_RL, color=c,
                       edgecolors="black", linewidths=1.0,
                       label=None if shown["rl"] else "RL @ K")
            shown["rl"] = True
            H_LAT.append(rx); H_EN.append(ry)

            if "deltaE_pct" in r.columns and pd.notna(r["deltaE_pct"].iloc[0]):
                ax.text(rx*1.01, ry*1.02, f"-{float(r['deltaE_pct'].iloc[0]):.0f}%",
                        fontsize=10, color=c, ha="left", va="bottom", fontweight="bold")


        bopt = sub[sub["type"] == "bo"]
        if not bopt.empty and pd.notna(bopt["energy"].iloc[0]):
            bx = float(bopt[lat_col].iloc[0])
            by = float(bopt["energy"].iloc[0])
            ax.scatter([bx], [by], marker="^", s=MARKER_SIZE_BASE, color=c,
                       edgecolors="black", linewidths=1.0,
                       label=None if shown["bo"] else "BayesOpt @ K")
            shown["bo"] = True
            H_LAT.append(bx); H_EN.append(by)

        H_LAT.append(tau); H_EN.append(0.0)

    ax.set_title(f"Latency–Energy — {name}", pad=10, fontsize=14)
    ax.set_xlabel("Latency (s)"); ax.set_ylabel("Energy (J)")
    ax.grid(True, alpha=0.25)
    ax.xaxis.set_major_formatter(ScalarFormatter(useMathText=True))
    ax.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))

    xmax, ymax = focus_limits(name, x, y, H_LAT, H_EN)
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)

    handles, labels = ax.get_legend_handles_labels()
    seen = {}; H=[]; L=[]
    for h, l in zip(handles, labels):
        if l and l not in seen:
            seen[l] = 1; H.append(h); L.append(l)
    ax.legend(H, L, loc="upper right", framealpha=0.95)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=400); plt.close()
    print(f"[OK] Saved {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model",    default="logs/ppo_helios.zip", help="Path to saved PPO .zip")
    ap.add_argument("--vecnorm",  default="logs/vecnormalize.pkl", help="(Optional) VecNormalize pickle")
    ap.add_argument("--profiles", default="profiles", help="Folder with *_summary.pkl (+ *_blocks.pkl)")
    ap.add_argument("--model-keys", default="gpt2-large,mistral,llama-13b",
                    help="Comma-separated substrings to pick 3 summaries, in order")
    ap.add_argument("--out",      default="eval_mm_results", help="Output dir for CSVs/plots")
    ap.add_argument("--episodes", type=int, default=40, help="Policy episodes per (model,K)")
    ap.add_argument("--random",   type=int, default=800, help="Random baseline episodes per model")
    ap.add_argument("--k",        default="0.80,1.00,1.20", help="Comma-separated K values")
    ap.add_argument("--max-steps",type=int, default=100, help="Episode length (must match training)")
    args = ap.parse_args()

    K_LIST = [float(x) for x in args.k.split(",")]

    os.makedirs(args.out, exist_ok=True)


    all_summ = sorted(glob.glob(os.path.join(args.profiles, "*_summary.pkl")))
    keys = [k.strip() for k in args.model_keys.split(",")]
    picked = []
    for key in keys:
        cand = [p for p in all_summ if key.lower() in os.path.basename(p).lower()]
        if not cand: raise FileNotFoundError(f"No summary for key='{key}' in {args.profiles}")
        picked.append(cand[0])
    if len(picked) != 3: raise RuntimeError(f"Expected 3 models, got {len(picked)}")
    model_names = [os.path.basename(p).replace("_summary.pkl","") for p in picked]
    print("[Eval] Models:", model_names)


    venv = DummyVecEnv([make_env(picked, args.max_steps)])
    if os.path.exists(args.vecnorm):
        venv = VecNormalize.load(args.vecnorm, venv)
        venv.training = False; venv.norm_reward = False
        print(f"[Eval] Loaded VecNormalize: {args.vecnorm}")
    else:
        print("[Eval] VecNormalize not found; continuing without.")


    model = PPO.load(args.model, env=venv, device="cpu")
    print(f"[Eval] Loaded policy: {args.model}")


    base_env = _unwrap_dummy(venv).envs[0]
    p50_map = {}
    for i, name in enumerate(model_names):
        force_profile_and_k(venv, i, 1.0)
        venv_reset(venv)  # load profile
        _, _, _, _, p50 = base_env._ensure_profile_stats()
        p50_map[name] = float(p50)
        print(f"[Eval] {name}: P50 latency = {p50_map[name]:.6f} s")


    rows = []

    for i, name in enumerate(model_names):
        for _ in range(args.random):
            force_profile_and_k(venv, i, 1.0)
            lat, en, thr, act, info = step_random_episode(venv)
            rows.append(dict(tag="random", model=name,
                             K=np.nan, K_req=np.nan, target_latency=np.nan,
                             avg_latency=lat, avg_energy=en, avg_throughput=thr,
                             thr_per_joule=thr/max(en,1e-12), action_mean=act.tolist(),
                             violated=bool(info.get("violated_avg", False))))

    for i, name in enumerate(model_names):
        p50 = p50_map[name]
        for K in K_LIST:
            req_target = float(K)*p50
            for _ in range(args.episodes):
                force_profile_and_k(venv, i, float(K))
                lat, en, thr, act, info = step_policy_episode(venv, model, deterministic=True)
                used_K = float(info.get("K_used", K))
                used_target = float(info.get("target_latency", req_target))
                rows.append(
                    dict(
                        tag="policy",
                        model=name,
                        K=used_K,
                        K_req=float(K),
                        target_latency=used_target,
                        avg_latency=lat,
                        avg_energy=en,
                        avg_throughput=thr,
                        thr_per_joule=thr / max(en, 1e-12),
                        action_mean=act.tolist(),
                        violated=bool(info.get("violated_avg", (lat > used_target))),
                    )
                )


    BO_INIT, BO_STEPS = 10, 20  # small, fair budget per K
    for i, name in enumerate(model_names):
        p50 = p50_map[name]
        for K in K_LIST:
            tau = float(K) * p50


            seeds_x = gather_bo_seeds(rows, name, K, tau, band=SLO_BAND,
                                      max_random=5, include_rl=True)

            bo = run_bo_point(
                venv, i, float(K), tau,
                band=SLO_BAND,
                n_init=BO_INIT, n_steps=BO_STEPS,
                seeds_x=seeds_x
            )

            rows.append(dict(
                tag="bo",
                model=name,
                K=float(K),
                K_req=float(K),
                target_latency=tau,
                avg_latency=bo["lat"],
                avg_energy=bo["eng"],
                avg_throughput=bo["thr"],
                thr_per_joule=bo["thr"] / max(bo["eng"], 1e-12),
                action_mean=list(map(float, bo["act"])),
                violated=bool(bo["lat"] < tau * (1.0 - SLO_BAND) or bo["lat"] > tau * (1.0 + SLO_BAND))
            ))

    df = pd.DataFrame(rows)
    raw_csv = os.path.join(args.out, "mm_points_raw.csv")
    df.to_csv(raw_csv, index=False)

    base_env = _unwrap_dummy(venv).envs[0]
    export_rl_actions_selected(df, base_env, p50_map, K_LIST, args.out, band=SLO_BAND)

    print(f"[OK] Saved {raw_csv} (rows={len(df)})")


    export_paper_csvs(df, args.out)

    export_efficiency_plots_rl_bo(df, base_env, p50_map, K_LIST, args.out)

    print("Done.")

if __name__ == "__main__":
    main()
