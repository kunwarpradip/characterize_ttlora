# TT-Shape Selection

This package generates Tensor-Train shape candidates for TT-LoRA adapter weights.
It is intentionally separate from the phase-specific experiment scripts so it can
later be packaged or uploaded with examples.

## Input Format

Provide one or more weights with names and matrix dimensions:

```json
{
  "weights": [
    {"name": "c_attn", "shape": [2304, 768]},
    {"name": "c_proj", "shape": [768, 768]},
    {"name": "q_proj", "shape": [2048, 2048]}
  ]
}
```

The shape convention is `[out_features, in_features]`.

## CLI

From this directory:

```bash
python -m ttshape_selection.cli \
  --weights-json examples/gpt2_weights.json \
  --rank 6 \
  --core-counts 2 3 4 \
  --top-k-per-weight 10 \
  --output-json /tmp/ttshape_candidates.json
```

## Python API

```python
from ttshape_selection import WeightSpec, generate_catalog

weights = [
    WeightSpec(name="c_attn", shape=(2304, 768)),
    WeightSpec(name="c_proj", shape=(768, 768)),
]

catalog = generate_catalog(
    weights,
    rank=6,
    core_counts=[2, 3, 4],
    split_strategy="all",
    top_k_per_weight=10,
)

payload = catalog.to_dict()
```

Each candidate includes:

- weight name and weight shape
- input/output factors
- TT shape, using `input_factors + reversed(output_factors)`
- core counts
- TT rank
- trainable TT parameter count for the current TT-LoRA implementation
- selection score and component penalties
- TT-matrix parameter count and compression ratio used by the selection score
