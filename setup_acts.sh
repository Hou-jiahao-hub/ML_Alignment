#!/bin/bash

if [ ! -d "/cvmfs/sft.cern.ch/lcg" ]; then
    echo "检测到 CVMFS 未挂载，正在为您自动挂载..."
    sudo mount -t cvmfs sft.cern.ch /cvmfs/sft.cern.ch
else 
    echo "cvmf 已就绪。"
fi

echo "正在加载LCG_109基础环境..."
source /cvmfs/sft.cern.ch/lcg/views/LCG_109/x86_64-el9-gcc13-opt/setup.sh

export PATH="$HOME/.local/bin:$PATH"

echo "正在注入ACTS本地环境变量"
export ACTS_HOME="$HOME/alignTest"
export LD_LIBRARY_PATH="$ACTS_HOME/install/lib64:$ACTS_HOME/install/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="$ACTS_HOME/build/python:$PYTHONPATH"

export ODD_PATH="$ACTS_HOME/thirdparty/OpenDataDetector"

echo "============================================"
echo "Configuration successful!"
echo "============================================"
