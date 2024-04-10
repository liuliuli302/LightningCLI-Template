# This is a record file of commonly used script commands
# and is not a sequence of scripts that should be executed.
# Therefore, please do not run this file directly,
# but choose the desired script and manually run it on the command line

# Script that trains the model.
python main.py fit --config config/config.yaml

# Script that open TensorBoard.
# Change "version_0" to the correct dir.
tensorboard --logdir=lightning_logs/version_0

