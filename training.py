import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback
#from tensorboard.backend.event_processing import event_accumulator
from environment import HeliosGymEnv
import torch.nn as nn
# training.py (top, after imports)
import torch
SEED = 42
np.random.seed(SEED); torch.manual_seed(SEED)
from stable_baselines3.common.utils import set_random_seed
set_random_seed(SEED)

TRAIN_K_VALUES = [0.80, 1.00, 1.20]

#  Hyperparameters & Paths
TOTAL_TIMESTEPS = 500000
LEARNING_RATE   = 1e-4
GAMMA           = 0.995
ENT_COEF        = 0.05
VF_COEF         = 1.0
CLIP_RANGE      = 0.2
N_STEPS         = 2048
BATCH_SIZE      = 128
NOPTEPOCHS      = 20


RETRAIN_LEARNING_RATE = None
RETRAIN_GAMMA         = None
RETRAIN_ENT_COEF      = None
RETRAIN_CLIP_RANGE    = None
RETRAIN_N_STEPS       = None
RETRAIN_BATCH_SIZE    = None
RETRAIN_NOPTEPOCHS    = None
RETRAIN_TARGET_LATENCY = None
LOG_DIR      = "logs"
RESULTS_DIR  = "results"
MODEL_NAME   = "ppo_helios"
MODEL_PATH   = os.path.join(LOG_DIR, MODEL_NAME)
TB_LOG_DIR = "results/tb"
MONITOR_PATH = os.path.join(RESULTS_DIR, "monitor.csv")



POLICY_KWARGS = dict(
    activation_fn=nn.Tanh,
    net_arch=[dict(pi=[128, 128], vf=[256, 256, 256])]
)


os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(TB_LOG_DIR, exist_ok=True)

#Environment setup
SUMMARY_DIR   = "profiles"
all_summaries = sorted(glob.glob(os.path.join(SUMMARY_DIR, "*_summary.pkl")))
# keep exactly the 3 models we care about (case-insensitive)
summary_paths = [
    p for p in all_summaries
    if any(k in os.path.basename(p).lower() for k in ["gpt2-large", "mistral", "llama-13b"])
]
assert len(summary_paths) == 3, (
    f"Expected 3 model summaries (gpt2-large, mistral, llama-13b) in {SUMMARY_DIR}, "
    f"found {len(summary_paths)}: {summary_paths}"
)
def make_env(worker_id):
    def _init():
        env = HeliosGymEnv(
            summary_paths=summary_paths,
            max_steps=100,
            unequal_splits=False,
            k_values=TRAIN_K_VALUES,
            target_latency=1.0
        )

        if hasattr(env, "summary_paths"):
            import numpy as _np
            env._profile_idx = int(_np.random.randint(len(env.summary_paths))) - 1
            env.k_weights = np.array([1.0, 3.0, 1.0], dtype=float)
        return env
    return _init


num_envs = 4
venv = DummyVecEnv([make_env(i) for i in range(num_envs)])

venv = VecMonitor(venv, filename=MONITOR_PATH)
venv = VecNormalize(venv, norm_obs=True, norm_reward=True, clip_obs=10.0)

# Custom callback to collect latency, energy, and actions
class MetricsCallback(BaseCallback):

    def __init__(self, results_dir: str, verbose=0):
        super().__init__(verbose)
        self.results_dir = results_dir
        self.latencies = []
        self.energies  = []
        self.episodes  = []
        self.nvec = None
        self.num_envs = None
        self.hist_counts = {}

    # Called once training starts; unwrap to DummyVecEnv to inspect action_space and envs
    def _on_training_start(self) -> None:
        # unwrap: VecNormalize -> VecMonitor -> DummyVecEnv
        vecnorm = self.training_env
        vecmon  = vecnorm.venv
        dummy   = vecmon.venv
        self.num_envs = dummy.num_envs
        base_env0 = dummy.envs[0]
        # action bins per dimension
        self.nvec = np.array(base_env0.action_space.nvec, dtype=int)

    def _get_underlying_env(self, i):
        # unwrap chain for env i
        vecnorm = self.training_env
        vecmon  = vecnorm.venv
        dummy   = vecmon.venv
        return dummy.envs[i]

    def _on_step(self) -> bool:
        infos   = self.locals.get("infos", [])
        dones   = self.locals.get("dones", [])
        actions = self.locals.get("actions", [])

        for i, (info, done, act) in enumerate(zip(infos, dones, actions)):
            if not done:
                continue

            # record latency/energy
            lat = float(info.get("latency", np.nan))
            en  = float(info.get("energy",  np.nan))
            self.latencies.append(lat)
            self.energies.append(en)


            model_name = info.get("model", None)
            K = info.get("K", None)
            if model_name is None or K is None:
                uenv = self._get_underlying_env(i)
                model_name = os.path.basename(uenv.summary_paths[uenv._profile_idx]).replace("_summary.pkl", "")
                K = float(getattr(uenv, "_K", np.nan))

            # store episode record
            self.episodes.append({
                "model": model_name,
                "K": K,
                "latency": lat,
                "energy": en,
                "action": np.array(act, dtype=int).tolist(),
            })

            # update per-(model,K) hist counts for each action dimension
            key = (model_name, K)
            if key not in self.hist_counts:
                self.hist_counts[key] = [np.zeros(n, dtype=int) for n in self.nvec]
            counts = self.hist_counts[key]
            act_arr = np.array(act, dtype=int)
            for d, aidx in enumerate(act_arr):
                # clip just in case
                if aidx < 0 or aidx >= self.nvec[d]:
                    continue
                counts[d][aidx] += 1

        return True

    def _on_training_end(self) -> None:
        # Write a CSV of per-episode records
        import csv
        csv_path = os.path.join(self.results_dir, "episodes_model_K_actions.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["model", "K", "latency", "energy", "action_json"])
            for r in self.episodes:
                w.writerow([r["model"], f"{r['K']:.3f}", r["latency"], r["energy"], str(r["action"])])


        for (model_name, K), counts in self.hist_counts.items():
            for d, arr in enumerate(counts):
                plt.figure(figsize=(6,4))
                xs = np.arange(len(arr))
                plt.bar(xs, arr)
                plt.xlabel(f"Action dim {d} index")
                plt.ylabel("Count")
                plt.title(f"Histogram dim {d} — {model_name}, K={K:.3f}")
                plt.grid(axis="y", alpha=0.3)
                out = os.path.join(self.results_dir, f"hist_dim{d}_{model_name}_K{K:.3f}.png")
                plt.tight_layout()
                plt.savefig(out)
                plt.close()

        print(f"[MetricsCallback] Wrote {csv_path} and {len(self.hist_counts)}×{len(self.nvec)} hist PNGs.")

metrics_cb = MetricsCallback(RESULTS_DIR)


# Initialize or load PPO
if os.path.exists(MODEL_PATH + ".zip"):
    model = PPO.load(
        MODEL_PATH,
        env=venv,
        tensorboard_log=TB_LOG_DIR,
        device="auto"
    )
    # Override hyperparameters for fine-tuning
    if RETRAIN_LEARNING_RATE is not None:
        model.learning_rate = RETRAIN_LEARNING_RATE
    if RETRAIN_GAMMA is not None:
        model.gamma = RETRAIN_GAMMA
    if RETRAIN_ENT_COEF is not None:
        model.ent_coef = RETRAIN_ENT_COEF
    if RETRAIN_CLIP_RANGE is not None:
        model.clip_range = RETRAIN_CLIP_RANGE
    if RETRAIN_N_STEPS is not None:
        model.n_steps = RETRAIN_N_STEPS
    if RETRAIN_BATCH_SIZE is not None:
        model.batch_size = RETRAIN_BATCH_SIZE
    if RETRAIN_NOPTEPOCHS is not None:
        model.n_epochs = RETRAIN_NOPTEPOCHS

    reset_flag = False
else:
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=LEARNING_RATE,
        gamma=GAMMA,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        clip_range=CLIP_RANGE,
        clip_range_vf=0.2,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        n_epochs=NOPTEPOCHS,
        policy_kwargs=POLICY_KWARGS,
        verbose=1,
        tensorboard_log=TB_LOG_DIR
    )

# Training
#model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=metrics_cb, tb_log_name="PPO_run", reset_num_timesteps=False)
model.set_env(venv)
model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=metrics_cb, reset_num_timesteps=False)
model.save(MODEL_PATH)
# Save VecNormalize statistics so evaluation can reproduce obs/reward scaling
venv.save(os.path.join(LOG_DIR, "vecnormalize.pkl"))


df = pd.read_csv(MONITOR_PATH, skiprows=2, header=None, names = ['r', 'l','t'], dtype= {'t':float, 'l':float, 'r':float})
df = df.rename(columns={'t':'timesteps', 'l':'length', 'r':'reward'})
plt.figure(figsize=(8,5))
plt.plot(df["timesteps"], df["reward"], label="Episode Reward")
plt.xlabel("Timesteps")
plt.ylabel("Reward")
plt.title("Episode Reward vs Timesteps")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reward_curve.png"))
plt.show()

# Rolling Mean Reward
window = 20
rolling = df["reward"].rolling(window).mean()
plt.figure(figsize=(8,5))
plt.plot(df["timesteps"], rolling, label=f"{window}-Episode Rolling Mean")
plt.xlabel("Timesteps")
plt.ylabel("Mean Reward")
plt.title("Rolling Mean Reward")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "reward_rolling_mean.png"))
plt.show()


# Plot Latency and Energy over Episodes
episodes = np.arange(len(metrics_cb.latencies))
plt.figure(figsize=(8,5))
plt.plot(episodes, metrics_cb.latencies, label="Latency")
plt.xlabel("Episode")
plt.ylabel("Latency")
plt.title("Episode Latency")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "latency.png"))
plt.show()

plt.figure(figsize=(8,5))
plt.plot(episodes, metrics_cb.energies, label="Energy")
plt.xlabel("Episode")
plt.ylabel("Energy")
plt.title("Episode Energy")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "energy.png"))
plt.show()


