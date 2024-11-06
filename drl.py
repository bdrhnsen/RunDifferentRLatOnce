import argparse
import gymnasium as gym
from stable_baselines3 import PPO, A2C, DQN
# Predefined hyperparameter configurations
HYPERPARAMS = {
    "PPO": {
        "default": {"ent_coef": 0.01, "learning_rate": 1e-4},
        "high_lr": {"ent_coef": 0.01, "learning_rate": 5e-4},
        "low_ent_coef": {"ent_coef": 0.001, "learning_rate": 1e-4},
    },
    "A2C": {
        "default": {"learning_rate": 1e-4},
        "high_lr": {"learning_rate": 5e-4},
    },
    "DQN": {
        "default": {"learning_rate": 1e-4},
        "high_lr": {"learning_rate": 5e-4},
    }
}

def train(env, model_name, algorithm, config_name, total_timesteps):
    # Retrieve hyperparameters based on algorithm and config name
    hyperparams = HYPERPARAMS.get(algorithm, {}).get(config_name, {})
    print(f"Training {algorithm} with config '{config_name}': {hyperparams}")
    if algorithm == 'PPO':
        model = PPO('MlpPolicy', env, verbose=1, device="cuda", tensorboard_log=f"rl/{model_name}_PPO", **hyperparams)
    elif algorithm == 'A2C':
        model = A2C('MlpPolicy', env, verbose=1, device="cuda", tensorboard_log=f"rl/{model_name}_A2C", **hyperparams)
    elif algorithm == 'DQN':
        model = DQN('MlpPolicy', env, verbose=1, device="cuda", tensorboard_log=f"rl/{model_name}_DQN", **hyperparams)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    model.learn(total_timesteps=total_timesteps)
    model.save(f"rl/{model_name}_{algorithm}")

def test(env, model_name, algorithm):

    # Load the trained model
    model_path = f"rl/{model_name}_{algorithm}"
    if algorithm == 'PPO':
        model = PPO.load(model_path)
    elif algorithm == 'A2C':
        model = A2C.load(model_path)
    elif algorithm == 'DQN':
        model = DQN.load(model_path)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    # Test the model in the environment
    while True:
        done = truncated = False
        obs, info = env.reset()
        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)            
            obs, reward, done, truncated, info = env.step(action)
            env.render()

def main():
    parser = argparse.ArgumentParser(description="Train or Test a reinforcement learning model gym env.")
    parser.add_argument('--mode', choices=['train', 'test'], required=True, help="Mode to run: 'train' or 'test'")
    parser.add_argument('--algorithm', choices=['PPO', 'A2C', 'DQN'], required=True, help="RL Algorithm to use: PPO, A2C, or DQN")
    parser.add_argument('--model_name', required=True, help="Name of the model for saving/loading")
    parser.add_argument('--config_name', default="default", help="Name of the hyperparameter configuration to use (default: 'default')")
    parser.add_argument('--total_timesteps', type=int, default=int(1e7), help="Total timesteps for training (default: 1e7)")
    args = parser.parse_args()

    # Create the environment
    env = gym.make("env_name")

    # Run training or testing based on mode
    if args.mode == 'train':
        train(env, args.model_name, args.algorithm, args.config_name, args.total_timesteps)
    elif args.mode == 'test':
        test(env, args.model_name, args.algorithm)

if __name__ == "__main__":
    main()
