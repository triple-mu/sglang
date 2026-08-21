#!/bin/bash

docker run \
  -it \
  --gpus all \
  --runtime=nvidia \
  --privileged=true \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e SHELL=/bin/bash \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /data/sonlin:/workspace \
  -v /data/cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  --name sglang-diffusion-triplemu \
  nvcr.io/nvidia/pytorch:26.07-py3 \
  bash -lc '
    if ! command -v ffprobe >/dev/null 2>&1; then
      apt-get update
      apt-get install -y --no-install-recommends ffmpeg libopenmpi-dev libopencv-dev
    fi
    exec bash
  '

docker run \
  -it \
  --gpus all \
  --privileged=true \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v /data/sonlin:/workspace:rw \
  -v /data/cache/huggingface:/root/.cache/huggingface:rw \
  -w /workspace \
  --name sglang-diffusion-triplemu \
  nvcr.io/nvidia/pytorch:26.07-py3 \
  bash

docker run \
  -it \
  --gpus all \
  --privileged=true \
  --network=host \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  -v ${PWD}:/workspace:rw \
  -w /workspace \
  --name sglang-diffusion-triplemu \
  nvcr.io/nvidia/pytorch:26.07-py3 \
  bash