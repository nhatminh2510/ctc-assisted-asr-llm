#!/bin/bash

export HF_HOME=/home-local/minh/myhome/.cache/huggingface
export CUDA_VISIBLE_DEVICES=0,1  # Set GPU device
USE_DEVICE=0
# Seoul is fine, 
# Singapore, Toronto is shit

CUDA_VISIBLE_DEVICES=$USE_DEVICE python src/test.py

