# Rank-5 Low-Rank RNN Drift-Rate Follow-Up Plan

## Purpose

The Rank-5 low-rank RNN notebooks suggest that the learned z variables are not only useful as RT predictors. Compared with the earlier GRU latent representation, the low-rank z variables are more compact, interpretable, and potentially better suited for testing whether CPP-related latent dynamics are connected to drift rate and evidence accumulation.

The next analysis should use two Rank-5 z sources in parallel:

- `rank5_no_cpp_prior`: the no-CPP-shape-prior model, used as the cleaner representation analysis.
- `rank5_cpp_prior`: the original CPP-shape-prior model, retained as a theory-guided robustness comparison.

The no-prior result should be treated as the primary interpretive anchor, but paper-facing claims should emphasize effects that are directionally stable across both versions and survive CPP and shuffled-z controls.

The next analysis should therefore move from prediction to mechanism:

> Do Rank-5 z variables, in both no-prior and CPP-prior versions, explain drift rate or evidence-accumulation dynamics beyond conventional CPP amplitude and CPP slope summaries?

This should be tested with residual regression, partial regression, and nested model comparison rather than simple raw correlation.

## Why Not Raw Correlation Alone

Raw correlations such as `corr(z, drift_rate)` are useful as exploratory checks, but they can be misleading because z variables, drift rate, and RT may all share variance with:

- subject differences
- condition difficulty
- RT differences
- CPP amplitude
- CPP slope
- accuracy
- task block or other session-level effects

The main analysis should therefore ask whether z variables explain drift-rate-related variance after controlling for subject, condition, and conventional CPP summaries.

## Required Data Level

The z variables and drift-rate estimates must be aligned at the same analysis level.

The current processed metadata do not yet contain a drift-rate estimate column. Until drift estimates are added, the completed dual-prior analyses should be treated as RT/CPP representation checks rather than final drift-rate evidence.

If drift rate is available at the subject-condition level, aggregate z variables to subject-condition level before modeling:

```text
subject x condition:
  mean(z1 early), mean(z2 early), ..., mean(z5 late)
  drift_rate
```

If trial-wise drift estimates are available, use trial-level models:

```text
drift_trial ~ z_trial + CPP_trial + condition + subject
```

For trial-level analyses, prefer subject-aware cross-validation or mixed-effects modeling so that subject structure is not treated as independent trial noise.

## Primary Time Windows

Use two primary windows:

```text
early: -600 to -300 ms
late:  -120 to  -50 ms
```

Use two exploratory windows:

```text
mid:  -300 to -120 ms
full: -600 to  -50 ms
```

The early and late windows should be treated as the main planned comparisons because prior Rank-5 analyses suggested that the clearest baseline+z improvement appears in early and late response-locked periods.

Do not include `-1000 to -600 ms` in the main drift-rate or RT/CPP mechanism tests. That interval is a quality-check/background window only. The no-prior prediction-quality check showed weaker far-baseline reconstruction than the planned `-600 to -50 ms` analysis window, so the main interpretation should stay inside the planned pre-response interval.

## Dual-Prior Analysis Rule

Every planned z-space analysis should produce a model-version column:

```text
model_version: no_prior | cpp_prior
```

Run the same analysis for both latent exports using identical windows, subject-aware cross-validation, baseline variables, CPP covariates, and shuffled-z controls.

Core comparisons:

```text
no_prior baseline+z       - no_prior baseline
cpp_prior baseline+z      - cpp_prior baseline
no_prior baseline+CPP+z   - no_prior baseline+CPP
cpp_prior baseline+CPP+z  - cpp_prior baseline+CPP
```

The strongest conclusion is reserved for effects that have the same direction across both versions and do not appear in shuffled-z controls.

## Analysis Step 1: Exploratory Partial Correlation

Create a partial-correlation heatmap:

```text
rows:    z1, z2, z3, z4, z5
columns: early, mid, late, full
target:  drift_rate
controls:
  subject
  condition
  CPP amplitude
  CPP slope
split/facet:
  model_version = no_prior vs cpp_prior
```

This plot should answer:

> Which z variables, and which time windows, are most strongly related to drift rate after removing subject, task, and CPP-summary variance?

Recommended output:

```text
dual_prior_z_drift_partial_correlation_heatmap.pdf
dual_prior_z_drift_partial_correlation_heatmap.svg
dual_prior_z_drift_partial_correlation_heatmap.png
dual_prior_z_drift_partial_correlation_table.csv
```

## Analysis Step 2: Residual Scatter Plot

For the strongest planned or exploratory z-window relation, make a residual scatter plot.

Example:

```text
x-axis: residual z3 early
y-axis: residual drift rate
```

Residualize both variables against the same control set:

```text
residual z3 early  = z3 early  ~ subject + condition + CPP amplitude + CPP slope
residual drift     = drift     ~ subject + condition + CPP amplitude + CPP slope
```

This figure is useful for publication because it directly visualizes the question:

> After removing subject, condition, and CPP summary effects, is z still related to drift rate?

Recommended output:

```text
dual_prior_residual_z_vs_drift_scatter.pdf
dual_prior_residual_z_vs_drift_scatter.svg
dual_prior_residual_z_vs_drift_scatter.png
dual_prior_residual_z_vs_drift_scatter_data.csv
```

## Analysis Step 3: Nested Regression / Delta R2 Forest Plot

The main confirmatory analysis should compare nested models.

Baseline model:

```text
drift_rate ~ subject + condition + task variables
```

CPP model:

```text
drift_rate ~ subject + condition + task variables + CPP amplitude + CPP slope
```

z model:

```text
drift_rate ~ subject + condition + task variables + z1 + z2 + z3 + z4 + z5
```

CPP + z model:

```text
drift_rate ~ subject + condition + task variables + CPP amplitude + CPP slope + z1 + z2 + z3 + z4 + z5
```

Key contrasts:

```text
baseline + z       - baseline
baseline + CPP + z - baseline + CPP
```

If `baseline + CPP + z - baseline + CPP > 0`, this suggests that z variables contain drift-rate-relevant information not fully captured by conventional CPP amplitude and slope.

Interpret the contrast separately for no-prior and CPP-prior z. A positive result in both versions is stronger evidence than a result that appears only in the CPP-prior model.

Recommended output:

```text
dual_prior_drift_delta_r2_forestplot.pdf
dual_prior_drift_delta_r2_forestplot.svg
dual_prior_drift_delta_r2_forestplot.png
dual_prior_drift_nested_model_performance.csv
dual_prior_drift_delta_r2_ci.csv
```

## Analysis Step 4: Standardized Beta Forest Plot

For the nested models, plot standardized coefficients for z1-z5.

Recommended figure:

```text
y-axis: z1, z2, z3, z4, z5
x-axis: standardized coefficient predicting drift rate
reference line: 0
point: mean coefficient
whisker: confidence interval across folds or bootstrap samples
```

Generate this separately for the primary early and late windows.

Recommended output:

```text
dual_prior_drift_z_standardized_beta_forestplot.pdf
dual_prior_drift_z_standardized_beta_forestplot.svg
dual_prior_drift_z_standardized_beta_forestplot.png
dual_prior_drift_z_standardized_beta_table.csv
```

## Multiple Comparison Control

Avoid reporting the largest raw correlation alone. The planned analysis has at least:

```text
5 z variables x 4 time windows = 20 comparisons
```

Use false-discovery-rate correction for exploratory partial correlations, and keep early and late windows as the primary planned comparisons.

For the final conclusion, emphasize multivariate model comparison rather than isolated z-by-window correlations.

Also emphasize cross-version stability. A raw no-prior-only or CPP-prior-only peak should be treated as exploratory unless it is supported by nested model comparison and controls.

## Interpretation Guide

### Result Pattern 1: z predicts drift after CPP controls

Interpretation:

> Low-rank latent states may capture trial-level evidence accumulation dynamics that are not fully captured by conventional CPP amplitude or CPP slope summaries.

Chinese summary:

> z variables 捕捉到了传统 CPP 指标没有完全表达的证据积累速度相关信息。

Strongest version of this result: the same conclusion appears for both no-prior and CPP-prior z.

### Result Pattern 2: z relates to drift, but disappears after CPP controls

Interpretation:

> z variables may largely reflect CPP-related accumulation dynamics, rather than providing independent information beyond CPP.

Chinese summary:

> z variables 可能主要是在重新表达 CPP 相关的积累过程，而不是提供 CPP 之外的新信息。

### Result Pattern 3: z predicts RT but not drift

Interpretation:

> z variables may reflect response preparation, urgency, motor timing, or non-decision components rather than evidence accumulation rate.

Chinese summary:

> z variables 可能不是 drift rate，而是和反应准备、urgency、motor preparation 或 non-decision time 更相关。

### Result Pattern 4: only early z relates to drift

Interpretation:

> Early low-rank dynamics may encode initial evidence quality or trial difficulty before late CPP buildup.

Chinese summary:

> early z 可能捕捉较早出现的证据质量、难度或初始积累状态，而不是单纯晚期 CPP slope。

### Result Pattern 5: only CPP-prior z predicts drift

Interpretation:

> The result should be framed as theory-guided reconstruction rather than autonomous mechanistic discovery.

Chinese summary:

> 如果只有 CPP-prior 版本有效，说明结果可能依赖 CPP 形状引导，机制解释要降级。

### Result Pattern 6: no-prior and CPP-prior z agree

Interpretation:

> The low-rank z result is less likely to be an artifact of CPP-shape-prior training and is more suitable for cautious mechanism-facing interpretation.

Chinese summary:

> 两个版本方向一致时，说明 z 结果不太可能只是 CPP 形状先验塑造出来的。

## Recommended Final Framing

The next paper-facing claim should remain cautious:

> Rank-5 low-rank latent variables may capture evidence-accumulation-related neural dynamics that are not fully summarized by conventional CPP amplitude and slope measures, especially when no-prior and CPP-prior analyses agree.

Avoid claiming:

```text
z variables are CPP
z variables prove drift rate
z variables are direct DDM parameters
```

Preferred phrasing:

```text
z variables provide compact latent-state readouts for testing whether CPP-related neural dynamics track evidence accumulation and drift-rate-like variation. The no-prior model is the cleaner representation analysis; the CPP-prior model is the theory-guided comparison.
```

Short cautious claim:

```text
Low-rank latent states may capture trial-level evidence accumulation dynamics not fully captured by conventional CPP amplitude/slope summaries.
```
