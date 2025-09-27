Required installation before running the codes- pytorch, transformers, gymnasium, matplotlib, stable-baselines3[extra], tensorboard.
To collect LLM workload- run LLM_profiler.py with huggingface token ID.
Save all the workload profiles in folder and pass the folder name to the environment.py
For training- run training.py
Training performance can be observed via tensorboard browser tab.
For deployment configuration for LLM training and observing performance of the configuration run deployment.py using the saved ppo after training.
