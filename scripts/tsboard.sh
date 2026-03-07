#!/bin/bash

# image=/home/25/minh/ctc-assisted-asr-llm/containers/ctc-asr-llm.sif
event=$1
image=/home-local/minh/myhome/MLC-SLM-Baseline/MLC-SLM.sif

# Check if event directory is provided
if [ -z "$event" ]; then
    echo "Error: Event directory not provided"
    echo "Usage: $0 <event_directory>"
    exit 1
fi

# Check if event directory exists
if [ ! -d "$event" ]; then
    echo "Error: Event directory does not exist: $event"
    exit 1
fi

# Check if directory has event files
if [ -z "$(find "$event" -name "events.out.tfevents.*" -o -name "*.json" 2>/dev/null)" ]; then
    echo "Warning: No event files found in $event"
fi

find_available_port() {
    local start_port=$1
    local end_port=$((start_port + 1000))

    for ((port=start_port; port<=end_port; port++)); do
        if ! lsof -i:"$port" &>/dev/null; then
            echo $port
            return
        fi
    done

    echo "Error: No available port found in range $start_port-$end_port" >&2
    exit 1
}

port=$(find_available_port 6006)
echo "Using port $port for TensorBoard"

cmd="tensorboard \
--logdir $event \
--port $port \
--bind_all \
--reload_interval 5"
singularity exec --nv --bind /home,/home-local,/CDshare,/kailab,/App,/LabWork,/UserTmp,/local,/Mirror --no-home $image $cmd

