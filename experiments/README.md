# SNIP Set Difference Diagonal Sweep Experiments

This directory contains scripts to run SNIP set difference experiments for P=Q values ranging from 1% to 10% (0.01 to 0.10) and evaluate vanilla adversarial attack success rates.

## Overview

- **Model**: `llama2-7b-chat-hf` (via Hugging Face)
- **Method**: `wandg_set_difference` (SNIP with set difference)
- **Experiments**: 10 diagonal cases where P=Q
- **P=Q Values**: 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.10
- **ASR Focus**: Vanilla ASR (ASR_basic)

## Setup

All scripts are ready to run. The model path has been configured to use HuggingFace's `meta-llama/Llama-2-7b-chat-hf`.

## Execution Steps

### Step 1: Dump SNIP Scores (One-time, ~1-2 hours)

This step computes and saves SNIP scores for both utility and safety datasets. These scores are required for the set difference method.

```bash
# Run score dumping in background
nohup bash experiments/dump_scores.sh > experiments/dump_scores.log 2>&1 &

# Save the process ID
echo $! > experiments/dump_scores.pid

# Monitor progress
tail -f experiments/dump_scores.log
```

**Output**: Creates pickle files in `out/experiments/diagonal_sweep/scores/`

### Step 2: Run Diagonal Sweep (10 experiments, ~2.5-5 hours)

After scores are dumped, run the diagonal sweep experiments.

```bash
# Run diagonal sweep in background
nohup bash experiments/run_diagonal_sweep.sh > experiments/diagonal_sweep.log 2>&1 &

# Save the process ID
echo $! > experiments/diagonal_sweep.pid

# Monitor progress
tail -f experiments/diagonal_sweep.log
```

**Output**: Creates 10 directories in `out/experiments/diagonal_sweep/`:
- `p_0.01_q_0.01/`
- `p_0.02_q_0.02/`
- ...
- `p_0.10_q_0.10/`

Each directory contains:
- `log_wandg_set_difference.txt` - Summary metrics
- `attack_0.500000/*.jsonl` - Detailed attack results

### Step 3: Collect Results

Parse and aggregate results from all experiments.

```bash
python experiments/collect_diagonal_results.py
```

**Output**: Creates `out/experiments/diagonal_sweep/results_diagonal.csv`

## Process Management Commands

```bash
# Check if dump_scores is running
ps aux | grep dump_scores
# OR
cat experiments/dump_scores.pid | xargs ps -p

# Kill a running process
kill $(cat experiments/dump_scores.pid)

# View live logs
tail -f experiments/dump_scores.log
tail -f experiments/diagonal_sweep.log

# Check overall progress
bash experiments/monitor_progress.sh
```

## Results Structure

```
out/experiments/diagonal_sweep/
├── scores/
│   ├── utility/
│   │   └── wandg_score_*.pkl
│   └── safety/
│       └── wandg_score_*.pkl
├── p_0.01_q_0.01/
│   ├── log_wandg_set_difference.txt
│   ├── attack_0.500000/
│   │   ├── inst_basic.jsonl
│   │   ├── inst_basic_no_sys.jsonl
│   │   ├── no_inst_basic.jsonl
│   │   └── no_inst_basic_no_sys.jsonl
│   └── pytorch_model.bin
├── p_0.02_q_0.02/
├── ...
├── p_0.10_q_0.10/
└── results_diagonal.csv
```

## Expected Results

The `results_diagonal.csv` will show the relationship between P=Q values and attack success rates:

| P_Q  | inst_ASR_basic | no_inst_ASR_basic | ... |
|------|----------------|-------------------|-----|
| 0.01 | 0.XXXX         | 0.XXXX           | ... |
| 0.02 | 0.XXXX         | 0.XXXX           | ... |
| ...  | ...            | ...              | ... |
| 0.10 | 0.XXXX         | 0.XXXX           | ... |

Higher ASR values indicate the model is more vulnerable to adversarial attacks after pruning safety-critical neurons.

## Troubleshooting

### Check tmux session status
```bash
tmux attach -t dump_scores  # or pq_sweep
# View the log output
```

### Check GPU usage
```bash
nvidia-smi
```

### Manually check individual results
```bash
cat out/experiments/diagonal_sweep/p_0.01_q_0.01/log_wandg_set_difference.txt
```

### Re-run a specific P=Q experiment
```bash
python main.py \
    --model llama2-7b-chat-hf \
    --prune_method wandg_set_difference \
    --sparsity_ratio 0.5 \
    --prune_data align \
    --p 0.05 --q 0.05 \
    --sparsity_type unstructured \
    --save out/experiments/diagonal_sweep/p_0.05_q_0.05 \
    --eval_attack --save_attack_res
```
