
# Module 20 – Week 9: Bayesian Optimisation

## Methodology

Week 9 continued the Bayesian optimisation of eight unknown functions, incorporating the observations returned from Week 8. The approach remained based on Gaussian Process (GP) regression, Upper Confidence Bound (UCB), Expected Improvement (EI), and neural-network (NN) ensembles.

The workflow comprised eight stages:

1. Incorporate Week 8 observations and assess improvements and diminishing returns.
2. Refit GP and NN surrogates and evaluate model reliability.
3. Generate candidates using UCB and EI acquisition functions.
4. Generate NN candidates and compare surrogate recommendations.
5. Quantify GP–NN agreement.
6. Evaluate neighbourhood evidence and investigate Functions 1 and 2 separately.
7. Select function-specific candidates, including additional local refinement.
8. Validate and format the final submissions.

As in previous weeks, **100,000 candidate points per function** were evaluated. UCB was assessed at κ = 0.5, 1.0, 1.5 and 2.5, while EI used ξ = 0%, 1%, 5% and 10% of the observed output standard deviation. This expanded the acquisition comparison used in Week 8. Five-model NN ensembles provided alternative recommendations, with reliability assessed through leave-one-out cross-validation (LOOCV) against a mean-prediction baseline.

Candidate selection considered predicted performance, uncertainty, model reliability, agreement between surrogates, proximity to existing observations and recent optimisation results. Additional local searches were performed where appropriate, comparing search radii of 0.03, 0.05 and 0.10.

---

## 1. Week 8 Results

The eight Week 8 observations were incorporated and compared with the previous best output for each function.

| **Function** | **Previous best** | **Week 8 output** | **Improved?** |
| ------------ | ----------------- | ----------------- | ------------- |
| F1 | 0.000000 | 0.000000 | No |
| F2 | 0.746962 | 0.657120 | No |
| F3 | -0.010190 | -0.000863 | Yes |
| F4 | 0.186912 | 0.387262 | Yes |
| F5 | 8662.482500 | 8115.434723 | No |
| F6 | -0.379740 | -0.274388 | Yes |
| F7 | 2.929370 | 2.922059 | No |
| F8 | 9.930811 | 9.957032 | Yes |

**Four of the eight functions achieved new best outputs.**

F4 and F6 demonstrated particularly encouraging improvements. F4 increased from 0.186912 to 0.387262, while F6 improved from -0.379740 to -0.274388.

F3 also improved, moving considerably closer to zero, while F8 reached a new best of 9.957032.

F5 remained relatively close to its existing best of 8662.482500, returning 8115.434723 following the previous week's boundary refinement. F7 also remained close to its best, returning 2.922059 compared with 2.929370.

F2 produced a weaker result, while F1 again returned effectively zero despite continued exploration.

These results supported continued refinement for F3, F4 and F6, while suggesting that F7 and F8 might be experiencing smaller incremental gains.

### 1.1 Optimisation Progress and Diminishing Returns

Running-best outputs were examined to determine whether additional observations were continuing to produce improvements.

| **Function** | **Observations** | **Current best** | **New bests recorded** |
| ------------ | ---------------- | ---------------- | ---------------------- |
| F1 | 18 | ~0.000000 | 3 |
| F2 | 18 | 0.746962 | 3 |
| F3 | 23 | -0.000863 | 8 |
| F4 | 38 | 0.387262 | 11 |
| F5 | 28 | 8662.482500 | 11 |
| F6 | 28 | -0.274388 | 5 |
| F7 | 38 | 2.929370 | 6 |
| F8 | 48 | 9.957032 | 9 |

F3 and F4 demonstrated the strongest evidence of continued optimisation progress. F3 had achieved new bests across its four most recent observations, while F4 continued producing substantial improvements.

F5 had also shown strong progress, particularly during its more recent observations, although its latest query did not improve on the existing best.

F6's recent improvements increased from approximately 0.001637 to 0.077134 and 0.105352, suggesting that its current region remained promising.

F7 had not improved in its two latest observations, although its most recent result remained very close to its best.

F8 showed the clearest evidence of diminishing returns. Recent improvements decreased from approximately 0.621928 to 0.223504, 0.089263 and 0.019562, followed by a slightly larger improvement of 0.026220.

F2 showed intermittent progress rather than a consistent reduction in improvement size, while F1 remained effectively flat despite additional observations.

Overall, the results did not demonstrate a uniform relationship between observation count and optimisation performance. Some functions continued improving substantially, while others appeared to be approaching regions of diminishing returns.

---

## 2. Surrogate Diagnostics

### 2.1 Gaussian Process Models

The GP models were refitted using the expanded datasets. Fitted RBF length scales and log marginal likelihoods were examined to identify potential modelling problems.

| **Function** | **RBF length scale** | **Log marginal likelihood** |
| ------------ | -------------------- | --------------------------- |
| F1 | 0.00001 | -25.540894 |
| F2 | 0.00001 | -25.540894 |
| F3 | 0.201 | -25.852938 |
| F4 | 0.313 | -23.818726 |
| F5 | 0.303 | -23.970400 |
| F6 | 0.527 | -22.245983 |
| F7 | 0.323 | -34.151061 |
| F8 | 1.050 | -21.138922 |

The most important finding was that **F1 and F2 continued to reach the GP length-scale lower bound of 1e-05**.

Both functions had exhibited this behaviour in Week 8. The additional observations therefore did not resolve concerns about the GP models' ability to generalise between observations.

The remaining GP length scales were comparatively stable. F4's length scale decreased from approximately 0.416 to 0.313, while F5 increased from 0.273 to 0.303.

F6 and F8 retained relatively large length scales, supporting continued use of their GP recommendations alongside other evidence.

Overall, the results suggested that F3–F8 remained suitable for GP-based candidate evaluation, while F1 and F2 required greater caution.

### 2.2 Neural-Network Validation

The five-model NN ensembles were retrained and evaluated using LOOCV.

| **Function** | **NN LOOCV MAE** | **Baseline MAE** | **Beats baseline?** |
| ------------ | ---------------- | ---------------- | ------------------- |
| F1 | 0.001837 | 0.000995 | No |
| F2 | 0.236724 | 0.241635 | Yes |
| F3 | 0.056881 | 0.053409 | No |
| F4 | 1.385355 | 7.460801 | Yes |
| F5 | 255.307417 | 2324.264043 | Yes |
| F6 | 0.255144 | 0.530835 | Yes |
| F7 | 0.443789 | 0.712291 | Yes |
| F8 | 0.309128 | 0.981238 | Yes |

Six of the eight NN models outperformed the mean-prediction baseline.

F4, F5, F6, F7 and F8 demonstrated useful validation performance, supporting the use of their NN recommendations as additional evidence during candidate selection.

F2 also passed validation, although narrowly. Its NN LOOCV MAE was 0.236724 compared with a baseline of 0.241635. This was particularly relevant because its GP remained unreliable.

F1 and F3 again failed to outperform their baselines, so their NN predictions were given limited weight.

The NN architecture used two hidden layers of 50 neurons in Week 9, compared with 20 neurons in Weeks 7–8. Consequently, changes in validation performance between weeks cannot be attributed solely to additional observations. The LOOCV procedure also evaluates an individual NN fitting procedure rather than the complete five-model ensemble.

---

## 3. GP Acquisition Candidates

The refitted GP models were used to generate candidate points with UCB and EI.

Four UCB settings (κ = 0.5, 1.0, 1.5 and 2.5) and four EI settings (ξ = 0%, 1%, 5% and 10%) were compared. Each candidate was assessed using predicted output, uncertainty, acquisition behaviour and distance from existing observations.

| **Function** | **Main GP findings** |
| ------------ | -------------------- |
| F1 | All acquisition settings selected [0.094177, 0.975622], although the GP remained unreliable. |
| F2 | Low-κ UCB returned the existing Week 8 observation. Higher-κ UCB and EI selected [0.403694, 0.391789]. |
| F3 | UCB κ=0.5 favoured local refinement, while higher exploration settings moved further from existing observations. |
| F4 | UCB κ=0.5 predicted 0.736288, above the current best. Higher κ values increasingly favoured uncertainty. |
| F5 | Low-κ UCB favoured the successful boundary region, while higher-exploration settings selected distant alternatives. |
| F6 | Low-κ UCB predicted -0.248253, while EI ξ=0% predicted -0.220587, both above the current best. |
| F7 | Low-κ UCB returned the existing best observation, while EI favoured a more exploratory alternative. |
| F8 | UCB κ=0.5 predicted 9.967635, slightly above the current best, while higher κ values selected more exploratory candidates. |

The strongest evidence for local refinement came from F3, F4 and F6, where recent improvements supported continued exploitation of successful regions.

F4 was particularly encouraging because its local UCB recommendation predicted a substantial improvement over the existing best of 0.387262.

F5 again showed a pronounced distinction between exploitation and exploration. The higher-exploration GP candidate was approximately 0.6863 from its nearest observation but had a predicted output of only 1607.436, substantially below the existing best of 8662.482500.

F7's low-κ recommendation duplicated an existing observation, while F8's local recommendation predicted a smaller but potentially worthwhile improvement.

For F1 and F2, acquisition results remained less reliable because their GP length scales had reached the lower bound.

The results reinforced the importance of evaluating GP recommendations alongside surrogate validation and observed neighbourhood performance.

---

## 4. Neural-Network Candidates

The NN ensembles independently selected candidates from the same 100,000-point search spaces.

Candidate selection used the highest ensemble mean prediction, with model disagreement and distance from existing observations recorded as supporting diagnostics.

| **Function** | **NN predicted output** | **NN disagreement** | **Nearest distance** |
| ------------ | ----------------------- | ------------------- | -------------------- |
| F1 | 0.001946 | 0.001331 | 0.182315 |
| F2 | 0.831771 | 0.431009 | 0.106007 |
| F3 | -0.001013 | 0.000717 | 0.018674 |
| F4 | 0.173431 | 0.101814 | 0.032892 |
| F5 | 8132.577652 | 837.663492 | 0.247618 |
| F6 | -0.248980 | 0.042060 | 0.127622 |
| F7 | 2.893229 | 0.055542 | 0.110492 |
| F8 | 10.056825 | 0.027228 | 0.234486 |

F2's NN selected `[0.702517, 0.009487]`, predicting 0.831771 compared with the existing best of 0.746962. However, disagreement between ensemble members was relatively high, limiting confidence in the prediction.

F4 and F7 favoured regions close to their established best observations, although their NN predictions remained below the current best values.

F5 selected a candidate near the upper boundary, but model disagreement remained substantial. Its recommendation was therefore assessed alongside the stronger observed evidence around `[1,1,1,1]`.

F6's NN predicted -0.248980, above the current best of -0.274388, with relatively low disagreement.

F8's NN predicted 10.056825, above the current best of 9.957032, with low disagreement of 0.027228. This suggested that additional refinement might remain worthwhile despite smaller recent improvements.

F1 and F3 predictions were given limited weight because their NNs failed LOOCV validation.

---

## 5. GP–NN Candidate Comparison

Euclidean distances were calculated between each NN recommendation and the GP acquisition candidates.

The closest GP candidate and corresponding distance were recorded for each function.

| **Function** | **Closest GP method** | **GP–NN distance** |
| ------------ | --------------------- | ------------------ |
| F1 | UCB κ=0.5 | 1.063404 |
| F2 | UCB κ=1.5 | 0.485232 |
| F3 | UCB κ=0.5 | 0.058180 |
| F4 | UCB κ=0.5 | 0.129899 |
| F5 | UCB κ=1.5 | 0.221396 |
| F6 | EI ξ=0% | 0.193602 |
| F7 | UCB κ=0.5 | 0.125569 |
| F8 | UCB κ=0.5 | 0.243999 |

F3, F4 and F7 demonstrated the closest agreement between surrogate recommendations.

F3 had the smallest distance, although its NN failed LOOCV, reducing confidence in the agreement.

F4 and F7 showed relatively strong agreement around established high-performing regions, with both models favouring local refinement.

F5 and F6 demonstrated moderate agreement, while F8's NN remained reasonably close to the lower-exploration GP recommendation.

F1 and F2 exhibited substantial disagreement. This was consistent with their GP reliability problems and supported investigating alternative selection approaches.

**Cross-model agreement was treated as supporting evidence rather than proof of candidate quality.** Agreement was less informative where models failed validation or where recommended candidates were already observed.

---

## 6. Neighbourhood Evidence

The three nearest observed outputs to leading GP and NN candidates were examined to assess whether recommendations were supported by historical performance.

| **Function** | **Neighbourhood findings** |
| ------------ | -------------------------- |
| F1 | Both GP and NN candidate regions had nearby outputs approximately equal to zero. |
| F2 | NN region: 0.471844, 0.538996, 0.746962. Exploratory GP region: 0.214965, -0.013858, -0.065624. |
| F3 | Local region: -0.000863, -0.010190, -0.016451. |
| F4 | Local region: 0.387262, 0.186912. Exploratory GP region included -13.527649 and -10.069633. |
| F5 | Boundary region: 8115.434723 and 8662.482500. Exploratory GP region included 109.571876, 233.223610 and 64.443440. |
| F6 | Local region: -0.274388, -0.388687, -0.379740. |
| F7 | Leading region: 2.922059, 2.929370, 2.536939. |
| F8 | Leading region: 9.930811, 9.911249, 9.957032. |

The strongest evidence for continued local refinement was in F3, F4, F5 and F6, where the best-performing regions had produced strong recent observations.

F4's local candidates were surrounded by substantially better observed outputs than its exploratory GP alternatives.

F5 showed an even clearer distinction. The established boundary region contained outputs above 8000, while the exploratory GP recommendation had nearby observations below 250.

F2's neighbourhood evidence favoured the NN recommendation over the exploratory GP candidate. This was particularly relevant because the NN passed LOOCV while the GP remained unreliable.

F7's strong neighbourhood evidence supported local refinement, although the close proximity of observations to its best suggested that improvement might be increasingly difficult.

For F8, both the NN and low-κ GP regions had strong nearby observations, supporting further refinement despite the smaller recent gains.

F1 remained unresolved because neither surrogate's recommended region had meaningful supporting observations.

### 6.1 Additional Function 1 Investigation

F1 continued to present the greatest modelling difficulty.

Its GP length scale again reached the lower bound, its NN failed LOOCV, and the models disagreed substantially about the next query.

Following the previous week's investigation of alternative kernels, a model-independent space-filling approach was retained.

A search across 100,000 points in the two-dimensional `[0,1]` search space identified regions furthest from existing observations.

The selected candidate was:

`[0.682178, 0.001903]`

Its nearest-observation distance was approximately **0.306754**, while the second and third nearest observations were approximately 0.308314 and 0.316118 away.

This was substantially more novel than the recommendations from the GP and NN models.

**[0.682178, 0.001903] was selected primarily because it increased coverage of the search space.**

This represented deliberate exploration rather than an expectation that the selected point would outperform the existing best.

### 6.2 Additional Function 2 Investigation

F2 also required a separate approach because its GP remained unreliable while its NN continued to outperform the LOOCV baseline.

NN-based local searches were performed around the current best using radii of 0.03, 0.05 and 0.10.

| **Radius** | **Candidate** | **NN prediction** | **NN disagreement** | **Nearest distance** |
| ---------- | ------------- | ----------------- | ------------------- | -------------------- |
| 0.03 | [0.711375, 0.108192] | 0.718309 | 0.149730 | 0.018533 |
| 0.05 | [0.710141, 0.088175] | 0.743700 | 0.211086 | 0.030706 |
| 0.10 | [0.704305, 0.038527] | 0.810157 | 0.363055 | 0.077676 |

The larger search radius produced the strongest predicted output, although NN disagreement also increased.

The 0.03 candidate had relatively low disagreement but was predicted below the existing best. The 0.05 candidate was predicted approximately equal to the best.

The 0.10 candidate predicted 0.810157, above the existing best of 0.746962, while providing greater novelty.

**[0.704305, 0.038527] was selected as the most promising NN-supported refinement candidate**, acknowledging the increased model disagreement.

---

## 7. Final Candidate Selection

The final selections combined recent optimisation performance, GP and NN reliability, acquisition results, model agreement, neighbourhood evidence and candidate novelty.

Additional local searches were performed for F3–F8 because their strongest recommendations were concentrated around existing high-performing observations.

For each function, 50,000 candidates were evaluated at radii of 0.03, 0.05 and 0.10 around the current best. GP and NN predictions were standardised and combined to compare local alternatives.

| **Function** | **Selected approach** | **Reasoning** |
| ------------ | --------------------- | ------------- |
| F1 | Space-filling exploration | Both surrogates unreliable; continued model-independent exploration. |
| F2 | NN-supported refinement | GP unreliable; NN passed LOOCV and had stronger neighbourhood evidence. |
| F3 | Local refinement | Recent improvement and strong GP support; NN failed validation. |
| F4 | Local refinement | Substantial recent improvement and strong predictions around the successful region. |
| F5 | Boundary refinement | Strong observed performance near [1,1,1,1], requiring a novelty constraint. |
| F6 | Broader local refinement | GP and validated NN both predicted improvement. |
| F7 | Conservative local refinement | Strong nearby observations but increasing disagreement further from the best. |
| F8 | Broader local refinement | Both surrogates predicted improvement despite diminishing recent gains. |

### 7.1 Additional Local Refinement

For F3, radius 0.03 identified:

`[0.473523, 0.500295, 0.456235]`

The GP predicted approximately -0.000166, slightly above the current best of -0.000863. The candidate remained close to the strongest observed region, with a nearest-observation distance of approximately 0.010539.

For F4, radius 0.05 identified:

`[0.388825, 0.366595, 0.350533, 0.420635]`

The GP predicted 0.563957, above the current best of 0.387262. The local searches consistently favoured the same region, supporting continued refinement.

For F6, radius 0.10 identified:

`[0.440497, 0.372759, 0.525613, 0.791828, 0.000000]`

The GP and NN predicted approximately -0.195871 and -0.229762 respectively, both above the current best of -0.274388. The candidate was approximately 0.110780 from its nearest observation.

For F7, radius 0.03 identified:

`[0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374]`

The NN predicted approximately 2.985400, above the current best of 2.929370. Larger-radius searches increased NN optimism but also disagreement, supporting the more conservative selection.

For F8, radius 0.10 identified:

`[0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246]`

The GP and NN predicted approximately 9.976275 and 10.009696, both above the current best of 9.957032.

The candidate was approximately 0.190714 from its nearest observation, providing greater novelty while remaining in a promising region.

### 7.2 Function 5 Novelty Constraint

For F5, the initial local searches repeatedly selected points almost identical to the existing best observation `[1,1,1,1]`.

Although technically novel, these candidates were approximately 0.0001 from an existing observation and therefore offered limited additional information.

Minimum nearest-observation distances of 0.03, 0.05 and 0.10 were compared.

| **Minimum novelty** | **GP prediction** | **NN prediction** | **NN disagreement** |
| ------------------- | ----------------- | ----------------- | ------------------- |
| 0.03 | 8287.623 | 8688.604 | 118.483 |
| 0.05 | 8001.755 | 8710.650 | 197.197 |
| 0.10 | 7187.507 | 8743.780 | 369.571 |

The NN became increasingly optimistic at larger distances, but GP predictions deteriorated while uncertainty and NN disagreement increased substantially.

The minimum novelty distance of **0.03** provided the best compromise between maintaining the successful boundary strategy and gathering meaningful additional information.

The resulting candidate was:

`[1.000000, 0.969998, 1.000000, 1.000000]`

### 7.3 Final Selections

| **Function** | **Final candidate** | **Selection rationale** |
| ------------ | ------------------- | ----------------------- |
| F1 | [0.682178, 0.001903] | Space-filling exploration where surrogates remain unreliable. |
| F2 | [0.704305, 0.038527] | NN-supported local refinement with greater novelty. |
| F3 | [0.473523, 0.500295, 0.456235] | Conservative refinement following a new best. |
| F4 | [0.388825, 0.366595, 0.350533, 0.420635] | Strong GP prediction in the successful local region. |
| F5 | [1.000000, 0.969998, 1.000000, 1.000000] | Boundary refinement with a 0.03 novelty constraint. |
| F6 | [0.440497, 0.372759, 0.525613, 0.791828, 0.000000] | Both surrogates predicted improvement at radius 0.10. |
| F7 | [0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374] | Conservative local refinement with controlled disagreement. |
| F8 | [0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246] | Broader local refinement supported by both surrogates. |

---

## 8. Validation and Submission

The final candidates were checked for:

- Correct dimensionality for each function.
- Input values within the permitted `[0,1]` bounds.
- No duplicate observations.
- Distance from existing observations.
- Consistent formatting to six decimal places.

All eight candidates passed the validation checks.

Nearest-observation distances ranged from approximately 0.010539 for F3 to 0.306755 for F1, reflecting the different exploitation and exploration strategies.

### Final Week 9 Submission Points

    Function 1: 0.682178-0.001903
    Function 2: 0.704305-0.038527
    Function 3: 0.473523-0.500295-0.456235
    Function 4: 0.388825-0.366595-0.350533-0.420635
    Function 5: 1.000000-0.969998-1.000000-1.000000
    Function 6: 0.440497-0.372759-0.525613-0.791828-0.000000
    Function 7: 0.199451-0.330731-0.460263-0.255214-0.346569-0.603374
    Function 8: 0.031565-0.175185-0.160043-0.130670-0.742356-0.487463-0.194164-0.622246

These points match the final selections recorded in the original Week 9 draft.

**Reproducibility limitation:** A separate execution of the NN refinement produced a slightly different F2 candidate, `[0.701892, 0.038659]`, rather than `[0.704305, 0.038527]`. The cause of this difference has not been established. The original submission has been retained, but its exact reproduction remains unresolved.

---

## Reflection

Week 9 demonstrated the importance of assessing optimisation progress alongside surrogate reliability and the performance of previous observations.

The additional running-best and improvement-size analysis showed that increasing the number of observations did not produce consistent results across all functions. F3, F4 and F6 continued improving, while F8 showed clearer evidence of diminishing returns as its best output approached 10. F7 also remained near its best without achieving a further improvement.

The strongest evidence supported continued local refinement for F3, F4, F6 and F8, where recent improvements, strong nearby observations and model predictions suggested that further progress remained possible.

F5 again illustrated the importance of novelty constraints. The initial local searches repeatedly returned candidates almost identical to the existing best. Imposing a minimum distance of 0.03 provided a more informative query while maintaining the successful boundary-focused strategy.

F1 continued to require a different approach. Both surrogates remained unreliable, and the earlier investigation of alternative GP kernels had not resolved the problem. Space-filling exploration therefore remained preferable to relying on uncertain predictions.

F2 also illustrated the value and limitations of NN-supported selection. Its GP remained unreliable while its NN narrowly outperformed the validation baseline. The NN identified a potentially promising region, although disagreement increased as the local search moved further from the existing best. The difference observed when reproducing the F2 refinement also highlights the importance of checking computational reproducibility.

More generally, the results reinforced that cross-model agreement can strengthen confidence in a region but does not guarantee improvement. Model reliability, neighbourhood evidence and candidate novelty remained important considerations.

One methodological limitation is that the NN architecture changed from two hidden layers of 20 neurons in Weeks 7–8 to 50 neurons in Week 9. Consequently, differences in NN performance cannot be attributed solely to the increased number of observations.

Overall, Week 9 strengthened the function-specific optimisation process by incorporating diminishing-returns analysis and comparing multiple local-search radii. The final selections balanced predicted improvement, model validation, neighbourhood evidence and novelty, while retaining alternative exploration strategies where surrogate predictions were least reliable.

