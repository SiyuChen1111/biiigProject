# Publication Figure Plan

## Main figures

### Main Figure 1. Neural reconstruction in the pre-response window
- **Panel a:** grand-average empirical vs reconstructed CPP waveform, restricted to the main response-locked analysis window (-600 to 0 ms), with the response onset marked and the response-proximal region lightly shaded.
- **Panel b:** condition-wise empirical and model waveforms in a compact paired layout.
- **Panel c:** correctness-wise empirical and model waveforms in the same layout.
- **Panel d:** difficulty-wise empirical and model waveforms in the same layout.
- **Rationale:** this figure puts the strongest and most interpretable evidence first: the model captures the overall response-locked CPP-like shape in the scientifically relevant window while keeping the unstable early edge region out of the main emphasis.

### Main Figure 2. Hidden-state relation to neural features and task variables
- **Panel a:** heatmap of hidden-state prediction quality for empirical CPP amplitude and slope across the four pre-defined time windows.
- **Panel b:** heatmap of hidden-state task coding, expressed as balanced-accuracy gain over the best shuffled or majority control.
- **Rationale:** this figure centers the question of whether hidden states carry meaningful neural information and treats task decoding as supporting evidence rather than the primary claim.

## Supplementary figures

### Supplementary Figure. Behavioral external validation
- Compact model-comparison plot for log RT prediction across behavior-only, CPP-only, hidden-only, combined, and shuffled-hidden baselines.
- **Rationale:** this is useful external validation, but it should remain secondary because it does not directly establish that the latent states are a mechanistic behavioral model.

## Omitted items

- Full-window waveform plots are not used as main panels because the unstable edge region near -1000 ms is visually distracting and not part of the main interpretation.
- CPP AUC is not included because it is not present in the current saved validation tables.
