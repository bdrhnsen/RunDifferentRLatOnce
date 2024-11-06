# Run Multiple RL algorithms from stablebaselines3 at once 

This will become handy if you do not know which will be a better choice for your environment. With argument parsing and a little bit bash scripting one can achieve this easily


## Command-Line Arguments
The training/testing script allows fast run for different Rl algorithms at once

```bash
python train_test_rl.py --mode <train/test> --algorithm <PPO/A2C/DQN> --model_name <MODEL_NAME> [--config_name <CONFIG_NAME>] [--total_timesteps <TIMESTEPS>]
```

Arguments
* --mode: Required. Choose train to train a new model or test to test an existing model.
* --algorithm: Required. Specify the RL algorithm to use. Options are PPO, A2C, or DQN.
* --model_name: Required. Name to assign to the model. This name is used for saving/loading the model and logging in TensorBoard.
* --config_name: Optional. Name of the hyperparameter configuration to use. Default is default. Other options depend on the configurations defined in the script.
* --total_timesteps: Optional. Number of timesteps to train the model. Default is 1e7

## Running All Configurations at Once

You can run multiple training sessions in parallel using the provided bash script, run_all.sh. This script will open each configuration in a new terminal window or tab, automatically activating the virtual environment.
Make the script executable:

```bash
chmod +x run_all.sh
```
run the script 
```bash
./run_all.sh
```
This will open new terminal windows (or tabs) for each training configuration specified in the script, allowing you to run experiments in parallel.

then you can start a Tensorboard session and watch the values.

```bash
tensorboard --logdir rl/
```
