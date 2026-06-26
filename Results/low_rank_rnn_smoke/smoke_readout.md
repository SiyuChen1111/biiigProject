# Low-Rank RNN Smoke Readout

This is an exploratory smoke test, not a final model comparison.

## Run Settings

- Dataset: `Data/ProcessedData`
- Sampled trials: `1200`
- Ranks: `[2, 3, 5]`
- Epochs per rank: `12`
- Device: `cpu`

## Main Metrics

| Rank | Full signal R2 | CPP average corr | CPP average R2 | Late CPP amp corr | Full-window slope corr |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.180 | 0.902 | 0.488 | 0.788 | 0.880 |
| 3 | 0.460 | 0.948 | 0.267 | 0.835 | 0.932 |
| 5 | 0.520 | 0.944 | 0.813 | 0.830 | 0.922 |

## Interpretation Guide

- Treat a high CPP average correlation as evidence that the model preserved the broad response-locked CPP shape.
- Treat high CPP slope or amplitude correlations as evidence that the low-rank state preserved trial-level CPP features.
- Inspect the trajectory figures before making any claim about mechanism; numerical reconstruction alone is not enough.

## Generated Figures

- `rank_*/cpp_average_reconstruction.png`
- `rank_*/latent_trajectories_by_group.png`
- `rank_*/mean_latent_timecourses.png`
