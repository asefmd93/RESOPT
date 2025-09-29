Required installation before running the codes- pytorch, transformers, gymnasium, matplotlib, stable-baselines3[extra], tensorboard.
To collect LLM workload- run LLM_profiler.py with huggingface token ID and llm_to_model.py.
Save all the workload profiles of llm_to_model in a folder and pass the folder name to the environment.py
For training- run training.py
Training performance can be observed via tensorboard browser tab.
For deployment configuration for LLM training and observing performance of the configuration run deployment.py using the saved ppo after training.
