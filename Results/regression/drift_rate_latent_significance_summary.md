# Drift-rate latent significance summary

- Latent source: `/Users/siyu/Documents/GitHub/biiigProject/tmp/low_rank_r5_no_cpp_prior_notebook_runs/20260629_100551/Data/IntermediateData/latents_low_rank_r5_no_cpp_prior/latents_low_rank_r5_no_cpp_prior.npz`
- Baseline behaviour adjusted R2: `0.599`
- CPP-amplitude baseline adjusted R2: `0.610`
- Number of FDR-significant latent additions: `18`
- Strongest hit: `Slope+Amplitude z3` with delta adjusted R2 `0.041` and q `0.0004`

Interpretation:

- Drift-rate-related latent signal is weakest in the early window and strongest in the response-proximal windows.
- The most stable contributors are z3 first, then z2 and z5.
- This supports using RT as an earlier behavioural validation step, but drift-rate as the more mechanistic follow-up target.