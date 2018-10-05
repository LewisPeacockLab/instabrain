#!/bin/bash

progname="ejaMBRemoteICE"
#pathname="."
pathname="/home/realtime/remoteServer/ejaMBRemoteICE-R007"

# 0.0.0.0 listens on all configured addresses
# can specify a single address to listen on instead
ip="0.0.0.0"
port=10000

# example configuration for 8 CPU cores
cputhreads=1
gpudevices=0
gputhreads=0

LD_LIBRARY_PATH=${pathname}
export LD_LIBRARY_PATH

echo Starting ${progname} "->" messages output to local terminal
${pathname}/${progname} $port --address $ip --cpu-threads $cputhreads --gpu-threads $gputhreads --gpu-devices $gpudevices --verbose
