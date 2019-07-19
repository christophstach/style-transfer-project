#!/usr/bin/env bash

echo 'GPU'
# GPU
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=32 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=16 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=8 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=4 --width=640 --height=480 --device_type=cuda

python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=32 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=16 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=8 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=4 --width=640 --height=480 --device_type=cuda

python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=32 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=16 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=8 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=4 --width=640 --height=480 --device_type=cuda

python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=32 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=16 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=8 --width=640 --height=480 --device_type=cuda
python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=4 --width=640 --height=480 --device_type=cuda

echo 'CPU'
# CPU
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=32 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=16 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=8 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=5 --channel_multiplier=4 --width=640 --height=480 --device_type=cpu

python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=32 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=16 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=8 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=4 --channel_multiplier=4 --width=640 --height=480 --device_type=cpu

python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=32 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=16 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=8 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=3 --channel_multiplier=4 --width=640 --height=480 --device_type=cpu

python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=32 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=16 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=8 --width=640 --height=480 --device_type=cpu
python3 scripts/run-experiment-fast.py --bottleneck_size=2 --channel_multiplier=4 --width=640 --height=480 --device_type=cpu