
# Module 21 – Week 10: Bayesian Optimisation

## Methodology

Week 10 continued the Bayesian optimisation of eight unknown functions, incorporating the observations returned from Week 9. The approach remained based on Gaussian Process (GP) regression, Upper Confidence Bound (UCB), Expected Improvement (EI), and neural-network (NN) ensembles.

An additional focus this week was assessing how accurately the GP and NN models had predicted the outputs of the previous week's submitted candidates. These realised prediction errors provided further evidence of model reliability alongside the existing validation and neighbourhood analysis.

The workflow comprised eight stages:

1. Incorporate Week 9 observations, assess improvements and compare realised predictions.
2. Refit GP and NN surrogates and evaluate model reliability.
3. Generate candidates using UCB and EI acquisition functions.
4. Generate NN candidates and compare surrogate recommendations.
5. Quantify GP–NN agreement.
6. Evaluate neighbourhood evidence and investigate Functions 1 and 2 separately.
7. Select function-specific candidates, including additional local and boundary refinement.
8. Validate and format the final submissions.

As in previous weeks, **100,000 candidate points per function** were evaluated. UCB was assessed at κ = 0.5, 1.0, 1.5 and 2.5, while EI used ξ = 0%, 1%, 5% and 10% of the observed output standard deviation. Five-model NN ensembles provided alternative recommendations, with reliability assessed using leave-one-out cross-validation (LOOCV) against a mean-prediction baseline.

Candidate selection considered predicted performance, uncertainty, model reliability, realised prediction accuracy, agreement between surrogates, proximity to existing observations and recent optimisation results.

---

## 1. Week 9 Results

The eight observations returned from the previous week's submissions were incorporated and compared with the previous best output for each function.

| **Function** | **Previous best** | **New output** | **Improved?** |
| ------------ | ----------------- | -------------- | ------------- |
| F1 | 0.000000 | 0.000000 | No |
| F2 | 0.746962 | 0.557217 | No |
| F3 | -0.000863 | -0.011198 | No |
| F4 | 0.387262 | 0.583730 | Yes |
| F5 | 8662.482500 | 8115.400447 | No |
| F6 | -0.274388 | -0.239769 | Yes |
| F7 | 2.929370 | 2.839196 | No |
| F8 | 9.957032 | 9.984981 | Yes |

**Three of the eight functions achieved new best outputs.**

F4 demonstrated the largest improvement, increasing from 0.387262 to 0.583730. This continued its strong recent performance and provided further support for local refinement around the established high-performing region.

F6 also improved, increasing from -0.274388 to -0.239769. Although the improvement of approximately 0.034619 was smaller than some previous gains, it supported the broader local refinement strategy used in Week 9.

F8 reached another new best of 9.984981, continuing its recent pattern of smaller improvements as the observed output approaches 10.

The remaining functions provided useful information despite not improving.

F3 returned -0.011198, considerably below its previous best of -0.000863 despite the candidate being extremely close to the established optimum. This suggested that the high-performing region might be particularly narrow.

F5 returned 8115.400447, approximately 547 below its best of 8662.482500. The candidate had moved slightly away from the upper boundary, providing further evidence of the importance of this region.

F7 remained relatively strong at 2.839196 but failed to improve on its best of 2.929370. F2 returned 0.557217, substantially below its existing best of 0.746962, while F1 again returned effectively zero.

These results supported continued refinement for F4, F6 and F8, while suggesting that the models and search strategies used for F2, F3, F5 and F7 required greater caution.

### 1.1 Realised Model Prediction Accuracy

The GP and NN predictions made for the previous week's submitted candidates were compared with the actual outputs received.

Absolute prediction error was calculated for each available prediction.

| **Function** | **Actual output** | **GP absolute error** | **NN absolute error** | **More accurate model** |
| ------------ | ----------------- | --------------------- | --------------------- | ----------------------- |
| F1 | ~0.000000 | — | — | Not applicable |
| F2 | 0.557217 | — | 0.252939 | NN only |
| F3 | -0.011198 | 0.011032 | 0.010468 | NN |
| F4 | 0.583730 | 0.019773 | 0.272725 | GP |
| F5 | 8115.400447 | 172.222571 | 573.203872 | GP |
| F6 | -0.239769 | 0.043899 | 0.010008 | NN |
| F7 | 2.839196 | 0.089696 | 0.146204 | GP |
| F8 | 9.984981 | 0.008706 | 0.024715 | GP |

F1 could not be assessed because the submitted candidate had been generated through model-independent exploration. F2 had only an NN prediction available.

The strongest result was F4. Its GP prediction of 0.563957 was very close to the realised output of 0.583730, producing an absolute error of only 0.019773. This supported both the GP model and the local-refinement strategy.

F8 provided similar evidence, with a GP error of only 0.008706. The GP was also more accurate for F5 and F7, although neither candidate improved on the existing best.

For F6, the NN performed particularly well, predicting the new output to within 0.010008. Combined with the new best achieved, this supported continued use of the NN as additional evidence for this function.

F3 was less encouraging. Both models predicted an output close to zero, but the actual result was -0.011198. Although the NN had the slightly smaller error, neither model captured how quickly performance deteriorated after a small movement from the best observation.

F2 provided an important warning against relying too heavily on individual NN predictions. Its predicted improvement to 0.810157 did not materialise, with the actual output falling to 0.557217.

Overall, the results supported placing greater weight on GP evidence for F4 and F8 and NN evidence for F6. However, a single realised observation was not considered sufficient to establish that one model would always outperform another.

---

## 2. Surrogate Diagnostics

### 2.1 Gaussian Process Models

The GP models were refitted using the expanded datasets. Fitted RBF length scales and log marginal likelihoods were compared with the previous week to identify changes in model behaviour.

| **Function** | **Previous length scale** | **New length scale** | **Log marginal likelihood** |
| ------------ | ------------------------- | -------------------- | --------------------------- |
| F1 | 0.00001 | 0.00001 | -26.959832 |
| F2 | 0.00001 | 0.00001 | -26.959832 |
| F3 | 0.201 | 0.0968 | -29.946737 |
| F4 | 0.313 | 0.316 | -19.214108 |
| F5 | 0.303 | 0.338 | -20.623561 |
| F6 | 0.527 | 0.518 | -20.566977 |
| F7 | 0.323 | 0.320 | -32.767946 |
| F8 | 1.050 | 1.070 | -18.081197 |

The most important finding was that **F1 and F2 continued to reach the GP length-scale lower bound of 1e-05**.

The additional observations therefore did not resolve concerns about the GP models' ability to generalise between observations. Their predictions continued to receive limited weight during candidate selection.

F3 showed the largest change, with its length scale falling from 0.201 to 0.0968. This was consistent with the latest observation, where a relatively small movement away from the best point produced a substantially worse output.

The shorter length scale suggested that F3 might vary over a relatively small distance, supporting tighter local refinement.

F4 remained particularly stable, with its length scale increasing only slightly from 0.313 to 0.316. Its log marginal likelihood also improved substantially. Combined with its accurate realised GP prediction and latest new best, this provided strong evidence for continued GP-led refinement.

F5, F6 and F7 showed comparatively small changes in their length scales, while F8 retained the largest fitted length scale at approximately 1.070.

Overall, the GP diagnostics reinforced the existing function-specific approach. F3 required greater caution because of its shorter fitted length scale, while F4 and F8 continued to demonstrate encouraging GP behaviour.

### 2.2 Neural-Network Validation

The five-model NN ensembles were retrained and evaluated using LOOCV.

| **Function** | **NN LOOCV MAE** | **Baseline MAE** | **Beats baseline?** |
| ------------ | ---------------- | ---------------- | ------------------- |
| F1 | 0.033792 | 0.000946 | No |
| F2 | 0.193346 | 0.238444 | Yes |
| F3 | 0.076948 | 0.054294 | No |
| F4 | 1.221417 | 7.632505 | Yes |
| F5 | 197.316315 | 2556.291575 | Yes |
| F6 | 0.161145 | 0.547373 | Yes |
| F7 | 0.349355 | 0.777671 | Yes |
| F8 | 0.520098 | 0.999338 | Yes |

Six of the eight NN models continued to outperform the mean-prediction baseline.

F4, F5, F6 and F7 all demonstrated improved validation performance compared with Week 9.

F6 was particularly encouraging. Its NN LOOCV MAE decreased from 0.255144 to 0.161145, while the NN had also accurately predicted the latest submitted candidate, which produced a new best.

F2's NN validation also improved, with its MAE decreasing from 0.236724 to 0.193346. However, the substantial realised prediction error from the previous round suggested that its individual predictions should still be treated cautiously.

F8 continued to outperform its baseline, although its NN MAE increased from 0.309128 to 0.520098. Given the stronger realised GP prediction, the GP was considered more reliable for this function.

F1 and F3 again failed to outperform their baselines. F1's validation performance deteriorated considerably, while F3's weaker results were consistent with the increased difficulty of modelling its narrow high-performing region.

Although most NN models continued to validate successfully, LOOCV and realised prediction accuracy were considered together rather than treating either measure as definitive.

---

## 3. GP Acquisition Candidates

The refitted GP models were used to generate candidate points with UCB and EI.

As in Week 9, four UCB settings (κ = 0.5, 1.0, 1.5 and 2.5) and four EI settings (ξ = 0%, 1%, 5% and 10%) were compared.

Each candidate was assessed using predicted output, uncertainty, acquisition behaviour and distance from existing observations.

| **Function** | **Main GP findings** |
| ------------ | -------------------- |
| F1 | All acquisition settings selected [0.374540, 0.950714], although the GP remained unreliable. |
| F2 | All acquisition settings selected [0.282588, 0.458677], predicting 0.349989, well below the current best. |
| F3 | UCB κ=0.5 predicted 0.016134, while EI predicted 0.018737 near the best region. Higher κ favoured greater exploration. |
| F4 | UCB κ=0.5 and all EI settings selected the same candidate, predicting 0.763656, above the current best. |
| F5 | Lower-κ UCB favoured the boundary region, while higher-exploration settings selected distant candidates with weaker predictions. |
| F6 | UCB κ=0.5 predicted -0.222140, slightly above the current best of -0.239769. |
| F7 | No GP acquisition candidate predicted an improvement over the existing best of 2.929370. |
| F8 | UCB κ=0.5 remained relatively local but predicted 9.938809, below the current best of 9.984981. |

F3's acquisition candidates demonstrated a clear exploration–exploitation trade-off. Lower-exploration UCB and EI predicted improvement relatively close to the current best, while higher κ values moved into more uncertain regions.

However, its shorter fitted GP length scale and unsuccessful previous local query suggested that these predictions required careful investigation.

F4 provided the strongest GP evidence for continued local refinement. UCB κ=0.5 and all EI settings selected a candidate predicting 0.763656, substantially above the current best of 0.583730. This was particularly encouraging given the GP's accurate prediction in the previous round.

F5 again showed a pronounced distinction between local and exploratory candidates. Even the lower-exploration candidate predicted only 6049.596207 compared with the current best of 8662.482500. The more exploratory alternatives relied heavily on uncertainty and had much weaker predicted outputs.

F6's UCB κ=0.5 candidate predicted a modest improvement, supporting further local refinement around its recently improved region.

For F7 and F8, the global GP acquisition searches did not predict improvement. However, both functions had strong established regions, suggesting that additional targeted local searches might be more informative.

F1 and F2 remained unreliable because both GP length scales were fixed at the lower bound.

These results reinforced the importance of evaluating acquisition recommendations alongside realised prediction accuracy, model validation and observed neighbourhood performance.

---

## 4. Neural-Network Candidates

The NN ensembles independently selected candidates from the same 100,000-point search spaces.

Candidate selection used the highest ensemble mean prediction, with model disagreement and distance from existing observations recorded as supporting diagnostics.

| **Function** | **NN predicted output** | **Nearest distance** | **Main finding** |
| ------------ | ----------------------- | -------------------- | ---------------- |
| F1 | 0.015695 | 0.265804 | Model failed LOOCV. |
| F2 | 1.004493 | 0.138158 | Optimistic prediction, but previous NN recommendation overpredicted. |
| F3 | -0.005204 | 0.014162 | Local candidate; NN failed LOOCV. |
| F4 | 0.117631 | — | Below the current best despite good validation. |
| F5 | 6187.342810 | — | Boundary-focused region, but below the existing best. |
| F6 | -0.273307 | 0.094106 | Below the current best, with low disagreement. |
| F7 | 2.900738 | — | Close to the best region, with relatively low disagreement. |
| F8 | 10.434491 | 0.677331 | Highly optimistic but exploratory, with greater disagreement. |

F2's NN selected `[0.998959, 0.845004]`, predicting 1.004493 compared with the current best of 0.746962.

Although the NN continued to pass LOOCV, its previous recommendation had substantially overpredicted the realised output. The new candidate was also located in a different region of the search space, limiting confidence in its predicted improvement.

F4's NN predicted only 0.117631, considerably below the current best of 0.583730. This contrasted with the GP's more optimistic recommendation and stronger recent realised accuracy.

F5's NN remained focused on the upper region but predicted 6187.342810, below its best of 8662.482500. This supported continued boundary-focused investigation rather than a broader move.

F6's NN predicted -0.273307, below the current best of -0.239769, although its low disagreement and strong validation suggested that it remained useful as supporting evidence.

F7 selected a candidate near the established high-performing region, predicting 2.900738, slightly below the current best of 2.929370.

F8's NN was the most optimistic, predicting 10.434491. However, the candidate was approximately 0.677331 from its nearest observation and had relatively high disagreement of 0.272532.

Given F8's weaker NN validation and the GP's greater realised accuracy, this prediction was considered too speculative to justify direct selection.

F1 and F3 recommendations continued to receive limited weight because their NNs failed LOOCV validation.

---

## 5. GP–NN Candidate Comparison

Euclidean distances were calculated between each NN recommendation and the GP acquisition candidates.

The closest GP candidate and corresponding distance were recorded for each function.

| **Function** | **Closest GP–NN distance** | **Interpretation** |
| ------------ | -------------------------- | ------------------ |
| F1 | 0.462308 | Limited agreement between unreliable models. |
| F2 | 0.813902 | Substantial disagreement. |
| F3 | 0.051914 | Strong regional agreement, although NN failed validation. |
| F4 | 0.197230 | Moderate agreement around the established region. |
| F5 | 0.000000 | Exact agreement with lower-κ UCB candidates. |
| F6 | 0.170772 | Reasonable agreement with UCB κ=0.5. |
| F7 | 0.246767 | Moderate agreement around the best region. |
| F8 | 0.949228 | Substantial disagreement between GP and NN recommendations. |

F5 showed the strongest numerical agreement, with the NN and UCB κ = 0.5, 1.0 and 1.5 selecting exactly the same candidate.

However, both models predicted this candidate substantially below the existing best. Agreement therefore did not necessarily indicate a promising improvement.

F3 showed strong regional agreement between the NN, EI and lower-exploration UCB candidates. However, the NN continued to fail validation, while the GP length scale had decreased considerably. This supported additional tight local refinement rather than relying directly on the global candidate.

F4 and F6 demonstrated useful agreement around established high-performing regions. This was supported by their recent improvements and encouraging model diagnostics.

F7 showed moderate agreement, with both models favouring the existing successful region rather than broader exploration.

F8 exhibited the greatest disagreement. Its NN recommended a distant region with a highly optimistic prediction, while the GP remained focused closer to existing observations.

Combined with the NN's weaker validation and less accurate realised prediction, this supported favouring GP-led local refinement.

F1 and F2 also showed substantial disagreement, consistent with their continuing GP reliability problems.

**Cross-model agreement was treated as supporting evidence rather than proof of candidate quality.** Agreement was less informative where models failed validation or where predicted performance remained below the existing best.

---

## 6. Neighbourhood Evidence

The three nearest observed outputs to leading GP and NN candidates were examined to assess whether recommendations were supported by historical performance.

| **Function** | **Neighbourhood findings** |
| ------------ | -------------------------- |
| F1 | Both GP and NN regions had nearby outputs approximately equal to zero. |
| F2 | NN region: 0.420586, 0.293993, -0.082041. GP region: -0.065624, 0.214965, -0.013858. |
| F3 | NN and low-κ UCB remained close to the best of -0.000863, while higher-exploration candidates had weaker support. |
| F4 | NN region included 0.583730, 0.387262 and 0.186912. Exploratory GP regions had substantially weaker observations. |
| F5 | Boundary region included 6691.931674, 8115.400447 and 7121.151955. Exploratory alternatives had much weaker support. |
| F6 | NN and low-κ UCB regions included -0.239769, -0.274388 and -0.379740. |
| F7 | NN region included 2.929370, 2.839196 and 2.922059. |
| F8 | Local UCB region included 9.984981, 9.840530 and 9.930811. NN region had much weaker observations. |

The strongest evidence for continued local refinement came from F4 and F6, where recent observations had established progressively better outputs.

F4's NN candidate remained very close to the latest best of 0.583730, while the lower-exploration GP candidate was connected to the same improving region.

F5's neighbourhood evidence again favoured the upper boundary. The established region contained outputs above 8000, while exploratory alternatives had neighbouring outputs below 234.

F7 also showed strong local support, with its NN candidate close to several of the highest observed outputs.

For F8, the difference between model recommendations was particularly important. The optimistic NN candidate was surrounded by much weaker observed outputs, including 9.141639 and 9.183005.

In contrast, UCB κ=0.5 remained close to the strongest observations, including the latest best of 9.984981.

F2's NN region was somewhat better supported than the GP region, but its neighbouring outputs remained substantially below the existing best of 0.746962.

F3's local recommendations were supported by the established best region, although the previous unsuccessful query and shorter fitted GP length scale suggested that small movements could still produce substantially different outputs.

F1 remained unresolved because neither model's recommended region had meaningful supporting observations.

Overall, the neighbourhood evidence favoured controlled local refinement for most functions rather than relying on highly exploratory global recommendations.

### 6.1 Additional Function 1 Investigation

F1 continued to present the greatest modelling difficulty.

Its GP length scale again reached the lower bound, its NN failed LOOCV, and the models disagreed substantially about the next query.

Following the previous week's approach, a model-independent space-filling search was retained.

The selected candidate was:

`[0.002495, 0.694432]`

Its nearest-observation distance was approximately **0.301746**, with the second and third nearest observations at distances of 0.302489 and 0.324234.

This represented a substantially underexplored region of the search space without relying on either predictive model.

**[0.002495, 0.694432] was selected primarily because it increased coverage of the search space.**

As in previous weeks, the aim was information gathering rather than an expectation that the candidate would immediately improve the existing best.

### 6.2 Additional Function 2 Investigation

F2 also required a separate approach because its GP remained unreliable while its NN continued to outperform the LOOCV baseline.

However, the previous week's NN prediction had been substantially more optimistic than the realised output.

NN-supported local searches were therefore performed around the current best of 0.746962 at `[0.713092, 0.138155]`, using radii of 0.03, 0.05 and 0.10.

| **Radius** | **Candidate** | **NN prediction** | **NN disagreement** | **Nearest distance** |
| ---------- | ------------- | ----------------- | ------------------- | -------------------- |
| 0.03 | [0.707291, 0.167589] | 0.631267 | 0.018715 | 0.009635 |
| 0.05 | [0.702093, 0.186930] | 0.622806 | 0.013365 | 0.029174 |
| 0.10 | [0.705183, 0.237842] | 0.606641 | 0.036773 | 0.079817 |

Unlike Week 9, predicted performance decreased as the search radius increased.

The 0.03 candidate had the highest prediction of 0.631267 but was only 0.009635 from an existing observation.

The 0.05 candidate had a slightly lower prediction of 0.622806 but provided more novelty, with a nearest-observation distance of 0.029174 and the lowest ensemble disagreement.

The 0.10 candidate provided considerably more novelty but had the weakest prediction and greater disagreement.

None of the three candidates were predicted to outperform the existing best of 0.746962.

**[0.702093, 0.186930] was selected as a compromise between novelty, model disagreement and proximity to the established best region.**

This represented a more conservative strategy than the previous week's wider NN-supported move.

---

## 7. Final Candidate Selection

The final selections combined recent optimisation performance, GP and NN reliability, realised prediction accuracy, acquisition results, model agreement, neighbourhood evidence and candidate novelty.

Additional local searches were performed for F3, F4, F6, F7 and F8, evaluating 50,000 candidates at radii of 0.03, 0.05 and 0.10 around the current best.

| **Function** | **Selected approach** | **Reasoning** |
| ------------ | --------------------- | ------------- |
| F1 | Space-filling exploration | Both surrogate models remained unreliable. |
| F2 | NN-supported local refinement | GP unreliable; conservative refinement following an unsuccessful NN prediction. |
| F3 | Tight local refinement | Shorter GP length scale and unsuccessful previous query suggested a narrow optimum. |
| F4 | GP-led local refinement | Consecutive improvements and strong realised GP accuracy supported further exploitation. |
| F5 | Boundary refinement | Strongest observations remained concentrated near [1,1,1,1]. |
| F6 | Local GP/NN refinement | Recent improvement and encouraging validation supported continued refinement. |
| F7 | Conservative local refinement | Strong model agreement at smaller radii. |
| F8 | GP-supported local refinement | Recent improvement and stronger GP accuracy favoured local exploration over the distant NN candidate. |

### 7.1 Additional Local Refinement

For F3, radius 0.05 identified:

`[0.440641, 0.482965, 0.435994]`

All three search radii produced very similar candidates, with GP predictions ranging from approximately 0.011428 to 0.012920.

Although the GP predicted improvement over the current best of -0.000863, the NN remained more conservative and failed LOOCV. The radius 0.05 candidate was therefore selected as a cautious local refinement.

For F4, radius 0.05 identified:

`[0.388750, 0.360865, 0.352431, 0.416723]`

The GP predicted approximately 0.649909, above the current best of 0.583730.

The candidate was only 0.007193 from the latest best observation, continuing the fine local-refinement strategy supported by recent improvements and accurate GP predictions.

For F6, radius 0.05 identified:

`[0.444874, 0.347937, 0.533470, 0.742972, 0.000000]`

The GP predicted improvement at all three radii, with the strongest prediction of -0.204221 occurring at radius 0.10.

However, uncertainty and NN disagreement also increased with radius. The 0.05 candidate was selected to retain predicted improvement while avoiding the greater uncertainty associated with the wider search.

For F7, radius 0.05 identified:

`[0.145227, 0.296570, 0.489754, 0.259986, 0.333711, 0.644995]`

Both models predicted improvement at radii 0.03 and 0.05.

Although the GP predicted as high as 3.012495 at radius 0.10, the NN no longer predicted improvement and disagreement increased.

The 0.05 candidate was therefore selected because it maintained strong agreement between the models while providing additional novelty.

For F8, radius 0.05 identified:

`[0.000000, 0.221175, 0.182914, 0.081049, 0.771543, 0.527149, 0.198086, 0.585768]`

The GP predictions across all three radii remained close to, but slightly below, the current best of 9.984981.

The NN predicted improvement throughout, although its optimism and disagreement increased as the radius expanded.

The radius 0.05 candidate provided a compromise between novelty and model reliability, avoiding both the wider 0.10 refinement and the much more speculative global NN recommendation.

### 7.2 Function 5 Novelty Constraint

F5 continued to require separate treatment because its best observation remained at `[1,1,1,1]`, with an output of 8662.482500.

As in previous weeks, minimum nearest-observation distances of 0.03, 0.05 and 0.10 were compared to avoid candidates effectively identical to the current best.

| **Minimum novelty** | **GP prediction** | **NN prediction** | **Main finding** |
| ------------------- | ----------------- | ----------------- | ---------------- |
| 0.03 | 8156.394720 | 8144.043455 | Strongest predictions; close to the boundary. |
| 0.05 | 7793.218387 | 7800.066846 | Lower predictions with greater uncertainty. |
| 0.10 | 6717.968992 | 6887.668776 | Substantial predicted deterioration. |

Unlike the previous week, the GP and NN now showed relatively close agreement about the deterioration in performance as candidates moved further from the upper boundary.

At minimum novelty 0.03, the models selected:

`[1.000000, 1.000000, 1.000000, 0.969985]`

The GP and NN predicted 8156.394720 and 8144.043455 respectively, both below the existing best.

At larger novelty distances, predicted performance declined substantially, while uncertainty and model disagreement increased.

The minimum novelty distance of **0.03** was therefore retained.

This candidate provided a useful comparison with the previous week's query, which had moved away from the boundary in the second dimension. The new candidate instead tested a similar movement in the fourth dimension.

Although neither model predicted an improvement, the selected point could provide further information about sensitivity close to the upper boundary.

### 7.3 Final Selections

| **Function** | **Final candidate** | **Selection rationale** |
| ------------ | ------------------- | ----------------------- |
| F1 | [0.002495, 0.694432] | Space-filling exploration where surrogates remain unreliable. |
| F2 | [0.702093, 0.186930] | Conservative NN-supported refinement with greater novelty than radius 0.03. |
| F3 | [0.440641, 0.482965, 0.435994] | Tight local refinement following an unsuccessful previous query. |
| F4 | [0.388750, 0.360865, 0.352431, 0.416723] | Strong GP prediction close to the newly established best. |
| F5 | [1.000000, 1.000000, 1.000000, 0.969985] | Boundary refinement with a 0.03 novelty constraint. |
| F6 | [0.444874, 0.347937, 0.533470, 0.742972, 0.000000] | Local refinement balancing predicted improvement and uncertainty. |
| F7 | [0.145227, 0.296570, 0.489754, 0.259986, 0.333711, 0.644995] | GP–NN agreement at radius 0.05 with controlled disagreement. |
| F8 | [0.000000, 0.221175, 0.182914, 0.081049, 0.771543, 0.527149, 0.198086, 0.585768] | Local refinement supported by stronger GP evidence and neighbourhood performance. |

---

## 8. Validation and Submission

The final candidates were checked for:

- Correct dimensionality for each function.
- Input values within the permitted `[0,1]` bounds.
- No duplicate observations.
- Distance from existing observations.
- Consistent formatting to six decimal places.

All eight candidates passed the validation checks.

Nearest-observation distances varied deliberately between functions.

F1 remained the most exploratory, with a nearest-observation distance of approximately 0.301746.

F4 and F5 were considerably more exploitative, with distances of approximately 0.007193 and 0.030015 respectively, reflecting the stronger evidence supporting their established regions.

F2, F3, F6 and F7 involved relatively small local adjustments, while F8 made a somewhat broader move at approximately 0.099346 from its nearest observation.

### Final Week 10 Submission Points

    Function 1: 0.002495-0.694432
    Function 2: 0.702093-0.186930
    Function 3: 0.440641-0.482965-0.435994
    Function 4: 0.388750-0.360865-0.352431-0.416723
    Function 5: 1.000000-1.000000-1.000000-0.969985
    Function 6: 0.444874-0.347937-0.533470-0.742972-0.000000
    Function 7: 0.145227-0.296570-0.489754-0.259986-0.333711-0.644995
    Function 8: 0.000000-0.221175-0.182914-0.081049-0.771543-0.527149-0.198086-0.585768

These points match the final selections recorded in the original Week 10 draft.

---

## Reflection

Week 10 demonstrated the value of incorporating realised prediction accuracy into the existing Bayesian optimisation workflow.

The latest results showed that successful local refinement remained possible for several functions, with F4, F6 and F8 achieving new best outputs. However, the unsuccessful queries for F2, F3, F5 and F7 reinforced the importance of evaluating model predictions alongside observed performance.

The realised prediction analysis was particularly useful for F4, F6 and F8. F4's GP accurately predicted another substantial improvement, while F8's GP also closely predicted its new best. For F6, the NN was particularly accurate, supporting its continued use alongside the GP.

F2 illustrated the limitations of relying on a model simply because it passes LOOCV. Although its NN validation improved, the previous candidate had been substantially overpredicted. The updated local searches no longer suggested that moving further away from the best region would improve performance, supporting a more conservative selection.

F3 also required caution. Its latest query performed worse despite being very close to the existing best, while its fitted GP length scale decreased substantially. This suggested that the function may vary sharply over short distances and supported tighter local refinement.

F5 provided further evidence of the importance of the upper boundary. Both models increasingly agreed that meaningful movements away from `[1,1,1,1]` were likely to reduce performance. The novelty-constrained candidate was therefore selected primarily to investigate sensitivity in another dimension rather than because it was expected to improve the best output.

F7 continued to show useful GP–NN agreement around its established high-performing region. However, the wider local search introduced greater disagreement, favouring a conservative refinement strategy.

F8 illustrated why the highest model prediction should not automatically determine candidate selection. Its NN predicted a substantial improvement in a distant region, but this was poorly supported by neighbouring observations and came with relatively high disagreement. The GP's stronger realised accuracy provided greater justification for continued local refinement.

F1 remained the most difficult function. Neither surrogate demonstrated reliable predictive ability, and the latest exploratory query again returned effectively zero. Model-independent space-filling exploration therefore remained the most appropriate strategy.

One methodological limitation remains that NN validation performance reflects the architecture used in the later weeks of the project, with two hidden layers of 50 neurons compared with 20 in Weeks 7–8. Consequently, cross-week changes in NN performance cannot be attributed solely to additional observations.

Overall, Week 10 strengthened the function-specific optimisation process by incorporating realised prediction errors alongside GP diagnostics, NN validation, acquisition behaviour, neighbourhood evidence and novelty. The final selections continued to balance exploitation where model performance and observed results supported it, with more cautious exploration where predictions remained unreliable.

