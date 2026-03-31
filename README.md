# characterize_ttlora

Phase 1 now has a clean local training scaffold for establishing a stable baseline before TT-LoRA-specific ablations.

Code layout:

- `train.py`: main CLI entrypoint for Phase 1 runs
- `cttlora/config.py`: experiment configuration dataclasses
- `cttlora/tasks.py`: task registry and dataset normalization
- `cttlora/data.py`: local dataset loading and tokenization
- `cttlora/modeling.py`: model loading and adaptation hooks
- `cttlora/training.py`: training loop, metrics, checkpoints, and summaries
- `test_train.py` and `test_utils.py`: kept as reference code

Example run:

```bash
python train.py \
  --dataset-name mrpc \
  --dataset-root /usr/projects/unsupgan/afia/stack_v2 \
  --model-path /home/pkunwar/characterize_ttlora/roberta-base/checkpoints \
  --adaptation-method full \
  --batch-size 16 \
  --epochs 5 \
  --output-dir runs
```

Outputs are written under `runs/<run-name>/`:

- `config.json`
- `history.csv`
- `summary.json`
- `checkpoints/best/`

Current Phase 1 scope:

- stable RoBERTa sequence-classification baseline
- gradient accumulation support
- train/validation convergence tracking
- gradient norm logging
- parameter counting
- peak GPU memory tracking
- clean adaptation hook for future TT-LoRA integration
