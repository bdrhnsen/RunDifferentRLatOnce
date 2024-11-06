#!/bin/bash

# Path to your virtual environment
VENV_PATH=".env/"

# Function to run a command in a new terminal tab or window with virtualenv activation
run_in_new_terminal() {
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    # For Linux (e.g., GNOME Terminal)
    gnome-terminal -- bash -c "source $VENV_PATH/bin/activate; $1; exec bash"
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    # For macOS (open a new Terminal window)
    osascript -e "tell application \"Terminal\" to do script \"source $VENV_PATH/bin/activate; $1\""
  elif [[ "$OSTYPE" == "cygwin" ]] || [[ "$OSTYPE" == "msys" ]]; then
    # For Git Bash on Windows
    start cmd /k "$VENV_PATH\Scripts\activate && $1"
  else
    echo "Unsupported OS: $OSTYPE"
  fi
}

# Commands to run each option with virtual environment activated
run_in_new_terminal "python drl.py --mode train --algorithm PPO --model_name ppo_default --config_name default --total_timesteps 1000000"
run_in_new_terminal "python drl.py --mode train --algorithm PPO --model_name ppo_high_lr --config_name high_lr --total_timesteps 1000000"
run_in_new_terminal "python drl.py --mode train --algorithm A2C --model_name a2c_default --config_name default --total_timesteps 1000000"
run_in_new_terminal "python drl.py --mode train --algorithm DQN --model_name dqn_high_lr --config_name high_lr --total_timesteps 1000000"
