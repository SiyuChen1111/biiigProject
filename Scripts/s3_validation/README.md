# s3_validation — Validation Routing Notes

This folder is now a lightweight routing layer.
The active validation scripts live in `Results/validation/`.

Run the validation steps after `Scripts/s2_training/train.py` has produced
`Data/IntermediateData/latents_full/latents_full.npz`.

## Scripts

| Script | What it does |
|--------|-------------|
| `run_dataset_fixed_stage2_stage3.py` | Legacy combined stage-2/3 run, kept only as old-flow reference |
| `Results/validation/run_neural_validation_audit.py` | Full hidden-state decoding audit |
| `Results/validation/run_fast_representational_checks.py` | Quick CPP correlation and RT-bin checks |
| `Results/validation/run_hidden_cpp_audit.py` | Strict hidden-state vs CPP amplitude audit |
| `Results/validation/make_publication_figures.py` | Publication-style figure refresh |

## Usage

```bash
# From the project root:
python Results/validation/run_neural_validation_audit.py \
    --latent-path Data/IntermediateData/latents_full/latents_full.npz \
    --dataset-dir Data/ProcessedData \
    --output-dir  Results/validation
```
