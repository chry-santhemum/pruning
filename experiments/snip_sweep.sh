#!/bin/bash

# GPU 0 jobs (sequential)
{
    CUDA_VISIBLE_DEVICES=0 python snip_run.py --safety_p 0.02 --capability_p 0.10 --take_abs --base_model --safety_dataset benign
    CUDA_VISIBLE_DEVICES=0 python snip_run.py --safety_p 0.02 --capability_p 0.10 --safety_dataset benign
} &

# sleep a bit to stagger time
sleep 5

# GPU 1 jobs (sequential)
{
    CUDA_VISIBLE_DEVICES=1 python snip_run.py --safety_p 0.01 --capability_p 0.05 --take_abs --base_model --safety_dataset benign
    CUDA_VISIBLE_DEVICES=1 python snip_run.py --safety_p 0.01 --capability_p 0.05 --safety_dataset benign
} &

# Wait for all GPUs to complete their jobs
wait

echo "All experiments completed!"