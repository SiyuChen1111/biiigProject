# Low-Rank RNN CPP Latent: Current Status and Next Analysis Plan

## 1. Current status

Full-data rank-5 low-rank RNN pipeline 已完成。模型在 5 个 seeds 上训练，使用 41 名被试、7297 个有效 trials。当前数据是 response-locked，response 位于 0 ms；模型输入通道是 CP1、CP2 和 CPz。

因此，当前的 CPP-like latent variables 应解释为 CPP-related dynamics 的 low-dimensional compression / measurement representation。更谨慎的表述是：low-rank RNN has recovered stable CPP-like latent coordinates。

## 2. What the current result supports

当前结果支持：full-data rank-5 low-rank RNN 学到了稳定的 aligned CPP-like latent。这个 latent 在多个 seeds 中重复出现，大多数 seeds 对齐到 `aligned_z2`，一个 seed 对齐到 `aligned_z3`。

aligned CPP-like latent 与 CPP late mean 和 CPP slope 都有强相关。这说明模型捕捉到了 CPP-like response-proximal build-up dynamics。由于 latent 维度可以旋转、翻转或交换，raw z indices 不应直接解释；正式解释应使用 aligned CPP-like latent，而不是直接说某个原始 z 维度代表某个心理变量。

## 3. What the current result does not yet support

当前结果还不能证明 latent 是直接的 evidence accumulation variable。模型输入本身就是 CP1、CP2、CPz 这些 CPP 相关通道，所以高 CPP-latent 相关更适合解释为模型对 CPP-related activity 的稳定低维测量。

也就是说，我们现在能说的是：模型学习到了 CPP-like response-proximal latent dynamics。我们还不能说：RNN discovered the evidence accumulation variable。

## 4. Why small CPP+z incremental RT prediction is expected

如果 z 是 low-dimensional CPP representation，那么 `CPP + z` 不一定应该比 CPP-only 带来很大的 incremental R2。大的增量会提示 z 含有 CPP 以外的额外信息，但这不是 measurement-representation claim 所必需的。

因此，小的 `CPP + z` improvement over CPP-only 不代表 z 没有价值。它更可能说明 z 不是独立于 CPP 的新信息源，而是一个 compact、denoised、stable 的 CPP measurement representation。真正重要的问题是：这个 CPP measurement representation 是否能映射到 DDM drift rate。

## 5. Why RT regression is still useful as a behavioral bridge

RT regression 仍然有用，但它是 behavioral bridge analysis，不是最终的机制检验。

它可以检验 aligned CPP-like latent 是否与行为有基本关系，latent slope 是否预测更快 RT，这种关系是否能在 subject、difficulty、correctness 控制后保留，是否在 correct-only trials 中存在，是否跨 seeds 稳定，以及这些效应来自 early/build-up windows 还是只来自 response-proximal windows。

但 RT 本身无法识别 drift rate，因为 RT 同时受 drift rate、boundary separation、non-decision time、starting point、motor preparation、response caution 和 trial noise 影响。所以 RT regression 有用，但不充分。

## 6. Next step A: RT / behavioral bridge regression

下一步 A 是把 RT regression 明确作为 behavioral bridge。推荐比较以下模型：

```text
M0: logRT ~ subject + difficulty + correctness

M1: logRT ~ subject + difficulty + correctness
          + CPP_late_mean + CPP_slope

M2: logRT ~ subject + difficulty + correctness
          + CPP_latent_late_mean + CPP_latent_slope

M3: logRT ~ subject + difficulty + correctness
          + CPP_late_mean + CPP_slope
          + CPP_latent_late_mean + CPP_latent_slope
```

还应包括 correct-only RT models、RT residual models、temporal-window RT models、correctness / accuracy models 和 seed-stability checks。

解释标准是：`z-only ≈ CPP-only` 支持 z 是有效 CPP proxy；`CPP + z > CPP-only` 支持 z 含有传统 CPP features 之外的额外信息。如果只有很晚的 response-proximal z 预测 RT，应谨慎解释为 response preparation / decision commitment；如果 slope 或较早 build-up windows 预测 RT，则更符合 accumulation-like dynamics。

## 7. Next step B: DDM drift-rate regression

下一步 B 是主要机制检验。核心问题是：Does the aligned CPP-like latent specifically map onto DDM drift rate?

推荐 DDM 模型如下：

```text
Baseline DDM:
v ~ evidence_strength + condition + subject
a ~ subject
t0 ~ subject

Traditional CPP-to-drift:
v ~ evidence_strength + CPP_late_mean + CPP_slope + subject
a ~ subject
t0 ~ subject

RNN CPP-latent-to-drift:
v ~ evidence_strength + CPP_latent_late_mean + CPP_latent_slope + subject
a ~ subject
t0 ~ subject

Combined CPP + latent:
v ~ evidence_strength + CPP_late_mean + CPP_slope
    + CPP_latent_late_mean + CPP_latent_slope + subject
a ~ subject
t0 ~ subject
```

还需要 parameter-specificity tests：

```text
v  ~ CPP_latent_features
a  ~ CPP_latent_features
t0 ~ CPP_latent_features
v + a + t0 ~ CPP_latent_features
```

如果 CPP-like latent 最好地改善 drift-rate model，这支持 drift-like accumulation interpretation。如果它更好预测 non-decision time，可能反映 response preparation 或 motor execution。如果它更好预测 boundary，可能反映 caution 或 threshold-related processes。如果它不预测任何 DDM 参数，它仍然是 CPP-like neural coordinate，但暂时没有强 DDM parameter relevance。

## 8. Why the DDM analysis is necessary

DDM analysis 是必要的，因为 RT 是多个过程混合后的结果。只看 RT，无法知道 neural feature 关联的是证据积累速度、反应谨慎程度，还是非决策时间。

DDM 把行为分解为 drift rate、boundary separation 和 non-decision time 等参数。只有当 aligned CPP-like latent 特异性映射到 drift rate，而不是同样或更强地映射到 boundary 或 non-decision time，才更有理由把它解释为 evidence-accumulation-related representation。

## 9. Why this analysis route is valid

这条路线是有效的，因为 CPP 在理论上与 centro-parietal evidence accumulation 有关。low-rank RNN 给出了稳定的 low-dimensional representation of CPP-like dynamics。RT 太间接，无法单独识别 drift rate；DDM 可以把行为拆成 latent decision parameters。

因此，检验 CPP-like latent 是否特异性映射到 drift rate，正好回答 CPP 是否能作为 neural accumulation representation 的关键问题。Parameter specificity 可以避免把普通 RT 相关误解释成 evidence accumulation。比较 traditional CPP features 与 RNN latent features，则可以检验 RNN latent 是否是一个有用的 measurement model。

## 10. Formal interpretation boundary

Current result supports:

> The full-data rank-5 low-rank RNN learns stable CPP-like response-proximal latent dynamics.

Current result does not yet prove:

> The latent is a direct evidence accumulation variable.

Next analyses will test:

> Whether the CPP-like latent maps selectively to DDM drift rate and can therefore support a stronger evidence-accumulation interpretation.

## 11. Immediate action items

1. Finalize the formal rank-5 result directory as the reference result.
2. Keep exploratory smoke/audit outputs outside the formal result story.
3. Run RT / behavioral bridge regression with baseline, CPP-only, z-only, and CPP+z models.
4. Add correct-only, residual, temporal-window, correctness, and seed-stability checks.
5. Build DDM drift-rate regression models and parameter-specificity tests.
6. Compare traditional CPP features against aligned CPP-like latent features.
7. Maintain conservative wording until DDM parameter specificity is established.
