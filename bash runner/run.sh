#!/bin/bash

#i have been using conda for this- to activate your conda env: follow this-
## Initialize conda for bash
#eval "$('path to conda.exe' 'shell.bash' 'hook')"
#conda activate (env name)

#or use pip to install the packages if required
# Install required packages
echo "Installing required packages..."
pip install torch torchvision numpy argparse

# Run the script
echo "Running model..."

#pass the stride and layers as a list only, stride for each layer
python main.py \
  --data_path "Image/4" \
  --conv_layers 64 128 256 512 512 \
  --fc_layers 1024 512 \
  --stride 1 1 1 1 1 \
  --use_maxpool \
  --epochs 2
