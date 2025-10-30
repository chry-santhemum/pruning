# ActSVD Grid Search Experiments

Modular pipeline for running ActSVD (orthogonal projection) experiments with grid search over utility ranks (r_u) and safety ranks (r_s).

## Overview

- **Method**: ActSVD with orthogonal projection (from `main_low_rank_diff.py`)
- **Grid**: 30 (r_u, r_s) pairs with log spacing from 50 to 4000
- **Model**: llama2-7b-chat-hf
- **Datasets**:
  - Utility: `alpaca_cleaned_no_safety` (for r_u projection)
  - Safety: `align` (for r_s projection)
- **Metrics**: ASR (Vanilla, Adv-Suffix, Adv-Decoding), Zero-shot accuracy, PPL

## Quick Start

### Test Mode (1 experiment, ~15-20 min)
```bash
bash experiments/actsvd/test_single.sh
```

### Full Run (30 experiments, ~10-15 hours)
```bash
bash experiments/run_actsvd_pipeline.sh
```

## Modular Components

### 1. Generate Rank Pairs
```bash
python experiments/actsvd/generate_rank_pairs.py
```
- Creates `rank_pairs.json` with 30 (r_u, r_s) pairs
- Log-spaced from 50 to 4000
- Diagonal pattern (r_u = r_s)

### 2. Run Single Experiment
```bash
python experiments/actsvd/run_single_actsvd.py --ru 100 --rs 100 --eval_zero_shot --eval_attack
```
- Runs one ActSVD experiment
- Can be tested independently
- Skips if results exist

### 3. Run Batch Experiments
```bash
bash experiments/actsvd/run_actsvd_sweep.sh
```
- Reads `rank_pairs.json`
- Runs all 30 experiments sequentially
- Progress tracking and error handling

### 4. Collect Results
```bash
python experiments/actsvd/collect_actsvd_results.py
```
- Parses log files from all experiments
- Creates `results_actsvd.csv`
- Displays summary table

### 5. Generate Plots
```bash
python experiments/actsvd/plot_actsvd_results.py
```
- Creates 3 plots:
  - `plot_vanilla_asr.png`
  - `plot_adv_suffix_asr.png`
  - `plot_adv_decoding_asr.png`
- X-axis: Zero-shot Accuracy
- Y-axis: ASR metrics
- Same styling as P,Q plots

## Directory Structure

```
experiments/actsvd/
├── generate_rank_pairs.py       # Generate (r_u, r_s) pairs
├── rank_pairs.json               # Generated pairs
├── run_single_actsvd.py          # Single experiment wrapper
├── run_actsvd_sweep.sh           # Batch runner
├── collect_actsvd_results.py     # Results parser
├── plot_actsvd_results.py        # Plot generator
├── test_single.sh                # Quick test
└── README.md                     # This file

out/experiments/actsvd_sweep/
├── ru_0050_rs_0050/              # Experiment directories
│   ├── log.txt                   # Metrics
│   └── attack_50_50/             # Attack results
├── ru_0058_rs_0058/
├── ...
├── ru_4000_rs_4000/
├── results_actsvd.csv            # Collected results
└── plot_*.png                    # Generated plots
```

## Monitoring

### Tmux Session
```bash
# Attach to running session
tmux attach -t actsvd_experiments

# List sessions
tmux ls

# Kill session
tmux kill-session -t actsvd_experiments
```

### Log File
```bash
tail -f experiments/actsvd_pipeline.log
```

### Check Progress
```bash
# Count completed experiments
ls -d out/experiments/actsvd_sweep/ru_*/log.txt 2>/dev/null | wc -l

# View latest results
cat out/experiments/actsvd_sweep/results_actsvd.csv
```

## Rank Pairs

The 30 log-spaced rank values from 50 to 4000:

```
50, 58, 67, 78, 91, 106, 123, 143, 167, 194,
226, 263, 306, 357, 415, 484, 563, 656, 764, 890,
1037, 1207, 1406, 1638, 1907, 2222, 2588, 3014, 3510, 4000
```

All 30 experiments use diagonal pairs: r_u = r_s

## Expected Outputs

### CSV Columns
- `r_u`, `r_s`: Rank values
- `PPL`: Perplexity
- `inst_ASR_basic`, `inst_ASR_basic_nosys`, `inst_ASR_multiple_nosys`: Instructional ASR
- `no_inst_ASR_basic`, `no_inst_ASR_basic_nosys`, `no_inst_ASR_multiple_nosys`: No-instruction ASR
- `ASR_gcg`: GCG adversarial suffix attack
- `arc_challenge`, `arc_easy`, `boolq`, `hellaswag`, `rte`, `openbookqa`, `winogrande`: Zero-shot benchmarks
- `averaged`: Average zero-shot accuracy

### Plots
Three scatter plots showing Zero-shot Accuracy vs ASR:
1. ASR_Vanilla
2. ASR_Adv-Suffix (GCG)
3. ASR_Adv-Decoding (multiple sampling)

## Runtime Estimates

- **Single experiment**: ~20-30 minutes (depends on GPU)
- **Test script**: ~15-20 minutes (1 experiment)
- **Full sweep**: ~10-15 hours (30 experiments)

## Troubleshooting

### Dataset not found
```bash
# Check if datasets exist
ls data/align/
ls data/alpaca_cleaned_no_safety/
```

### CUDA out of memory
- Reduce batch size in `main_low_rank_diff.py`
- Or use smaller ranks for testing

### Log parsing errors
- Check log format in `out/experiments/actsvd_sweep/ru_XXXX_rs_YYYY/log.txt`
- Expected format: `rank\tINST\tmetric\tscore`

### Resume interrupted sweep
- Just re-run `bash experiments/actsvd/run_actsvd_sweep.sh`
- Already completed experiments are automatically skipped

## Notes

- ActSVD computes SVD on-the-fly (no pre-computation needed like SNIP)
- Each experiment loads both utility and safety datasets
- Experiments are GPU-intensive and run sequentially
- Results are automatically skipped if log file exists
- All scripts are idempotent and resume-safe
