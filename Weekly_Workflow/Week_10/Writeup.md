
# Week 10 — Evaluating Realised Prediction Accuracy and Multi-Radius Refinement

**Imperial College AI/ML Capstone Project | Module 21**

## Objective

The objective of Week 10 was to incorporate the latest observations, evaluate how accurately the surrogate models had predicted the previous round's results, and refine the selection of new query points.

Previous rounds had increasingly favoured local refinement around established high-performing regions. However, some model predictions had been overly optimistic, while certain functions appeared sensitive to relatively small changes in their inputs.

This week therefore introduced a direct comparison between previously predicted outputs and the actual results received. This provided an additional measure of model reliability alongside Gaussian Process (GP) diagnostics and neural-network (NN) Leave-One-Out Cross-Validation (LOOCV).

A systematic multi-radius local search was also introduced, comparing candidate points generated approximately 0.03, 0.05 and 0.10 from the current best observation.

The aim was to improve the balance between predicted performance, uncertainty, model reliability and candidate novelty without unnecessarily changing the established optimisation methodology.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 9 queries were added to the existing datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | ~0.000000 | 0.000000 | No improvement |
| 2 | 0.746962 | 0.557217 | 0.746962 | No improvement |
| 3 | -0.000863 | -0.011198 | -0.000863 | No improvement |
| 4 | 0.387262 | 0.583730 | 0.583730 | New best |
| 5 | 8662.482500 | 8115.400447 | 8662.482500 | No improvement |
| 6 | -0.274388 | -0.239769 | -0.239769 | New best |
| 7 | 2.929370 | 2.839196 | 2.929370 | No improvement |
| 8 | 9.957032 | 9.984981 | 9.984981 | New best |

Three of the eight queries produced new best observations.

**Function 4** continued its strong improvement, increasing from approximately 0.3873 to 0.5837. This provided further evidence that local refinement around its established high-performing region remained effective.

**Function 6** improved from approximately -0.2744 to -0.2398. Although this gain was smaller than some previous improvements, it supported the decision to investigate a broader local search radius.

**Function 8** achieved another new best of approximately 9.9850, continuing its pattern of smaller improvements as the observed maximum approached 10.

The unsuccessful results also provided important information.

Function 3 deteriorated despite its selected point being extremely close to the previous best. This suggested that its high-performing region might be particularly narrow, with small changes in inputs producing substantial differences in output.

Function 5 returned approximately 8115.40 after moving away from the previously successful upper boundary, further supporting the importance of the region around `[1, 1, 1, 1]`.

Function 2 failed to improve despite a promising NN prediction, while Function 7 remained relatively close to its best without exceeding it.

Function 1 again produced an effectively zero output, leaving its optimisation problem unresolved.

Overall, the results supported continued function-specific optimisation, with greater emphasis on evaluating model reliability before selecting new candidates.

## 2. Evaluate Realised Model Prediction Accuracy

An important development this week was the comparison of previously predicted outputs with the actual results returned from the submitted queries.

For each function, the absolute prediction error was calculated:

Absolute Error = |Predicted Output - Actual Output|

This provided a direct assessment of how accurately the models predicted previously unseen observations.

Unlike training accuracy or cross-validation, this comparison evaluated predictions made before the actual outputs were known.

### 2.1 GP and Neural-Network Prediction Errors

| Function | GP Absolute Error | NN Absolute Error | More Accurate Model |
|---|---:|---:|---|
| 1 | — | — | Not applicable |
| 2 | — | 0.252939 | NN only |
| 3 | 0.011032 | 0.010468 | NN, marginally |
| 4 | 0.019773 | 0.272725 | GP |
| 5 | 172.222571 | 573.203872 | GP |
| 6 | 0.043899 | 0.010008 | NN |
| 7 | 0.089696 | 0.146204 | GP |
| 8 | 0.008706 | 0.024715 | GP |

Function 1 could not be assessed because its previous point was generated using model-independent exploration rather than a surrogate prediction.

Function 2 had only a neural-network prediction available for its submitted candidate.

### 2.2 Interpretation

**Function 4** produced one of the most encouraging results. Its GP predicted approximately 0.563957, compared with the actual output of 0.583730. The absolute error of just 0.019773 provided strong evidence supporting the GP's recent predictions.

**Function 8** showed similarly accurate GP performance, with an absolute error of only 0.008706. This strengthened the case for retaining the GP as the primary source of guidance for this function.

Function 5's GP also performed better than the neural network, although both models overestimated the returned output relative to their predictions.

**Function 6** favoured the neural network, which predicted the result to within approximately 0.010008. This was particularly encouraging because the candidate also produced a new best.

Function 3 showed relatively similar prediction errors from both models. However, both failed to anticipate the deterioration in performance following a very small movement from the previous best.

**Function 2** provided an important warning. Its NN predicted an output of approximately 0.810157, but the actual result was only 0.557217.

This relatively large error demonstrated that strong surrogate predictions could be unreliable even when a model performed reasonably well under cross-validation.

Overall, the results suggested that the GP currently had particularly strong realised accuracy for Functions 4 and 8, while the NN provided useful additional evidence for Function 6.

However, a single realised prediction was insufficient to establish the long-term superiority of either surrogate.

The results were therefore treated as supporting evidence alongside GP diagnostics, LOOCV, historical performance and observed neighbourhoods.

## 3. Refit and Diagnose the Gaussian Process Models

The RBF Gaussian Processes were refitted using the expanded datasets.

Fitted length scales and log marginal likelihoods were compared with the previous round to assess how the additional observations affected the surrogate models.

### 3.1 Updated GP Length Scales

| Function | Previous Length Scale | Updated Length Scale |
|---|---:|---:|
| 1 | 1e-05 | 1e-05 |
| 2 | 1e-05 | 1e-05 |
| 3 | 0.201 | 0.0968 |
| 4 | 0.313 | 0.316 |
| 5 | 0.303 | 0.338 |
| 6 | 0.527 | 0.518 |
| 7 | 0.323 | 0.320 |
| 8 | 1.05 | 1.07 |

Functions 1 and 2 continued to reach the lower RBF length-scale bound of `1e-05`.

This suggested that the additional observations had not resolved their existing modelling difficulties.

**Function 3 showed the largest change**, with its fitted length scale falling from approximately 0.201 to 0.0968.

This was consistent with the latest unsuccessful query, which suggested that performance could change rapidly over small distances.

The remaining functions produced comparatively stable length scales.

Functions 4, 5, 6, 7 and 8 continued to provide more informative GP behaviour.

### 3.2 Log Marginal Likelihood

| Function | Updated Log Marginal Likelihood |
|---|---:|
| 1 | -26.959832 |
| 2 | -26.959832 |
| 3 | -29.946737 |
| 4 | -19.214108 |
| 5 | -20.623561 |
| 6 | -20.566977 |
| 7 | -32.767946 |
| 8 | -18.081197 |

Functions 4, 5, 6, 7 and 8 showed improvements in log marginal likelihood relative to the previous round.

Function 4 improved from approximately -23.82 to -19.21, while Function 8 improved from -21.14 to -18.08.

These results supported the conclusion that their updated models continued to represent the growing observations reasonably well.

Function 3's reduced length scale and deteriorating modelling behaviour justified a more cautious approach.

Functions 1 and 2 remained unsuitable for relying heavily on GP acquisition recommendations.

## 4. Retrain and Validate the Neural Networks

The existing neural-network ensembles were retrained using the additional observations.

The architecture and validation approach remained unchanged to allow comparisons with previous rounds.

LOOCV was repeated, with neural-network Mean Absolute Error (MAE) compared against a mean-prediction baseline.

### 4.1 Updated LOOCV Results

| Function | NN LOOCV MAE | Mean Baseline MAE | Beats Baseline |
|---|---:|---:|---|
| 1 | 0.033792 | 0.000946 | No |
| 2 | 0.193346 | 0.238444 | Yes |
| 3 | 0.076948 | 0.054294 | No |
| 4 | 1.221417 | 7.632505 | Yes |
| 5 | 197.316315 | 2556.291575 | Yes |
| 6 | 0.161145 | 0.547373 | Yes |
| 7 | 0.349355 | 0.777671 | Yes |
| 8 | 0.520098 | 0.999338 | Yes |

The neural networks outperformed the baseline for **six of the eight functions**, maintaining the pattern observed in previous rounds.

Functions 4, 5, 6 and 7 all improved their NN validation performance.

**Function 6** was particularly encouraging. Its MAE decreased from approximately 0.2551 to 0.1611, while the latest realised prediction was also highly accurate.

Function 2's validation improved to 0.1933, comfortably below the mean baseline. However, its substantial overprediction in the previous round meant its candidate recommendations still required caution.

Functions 1 and 3 continued to fail validation.

Function 1's MAE deteriorated considerably, further reducing confidence in using either the GP or NN.

Function 3's MAE also increased, reinforcing the evidence that its output was difficult to predict accurately.

Function 8 remained better than the baseline, but its NN MAE increased from approximately 0.3091 to 0.5201. Combined with the GP's stronger realised prediction accuracy, this supported giving greater weight to GP recommendations.

The results highlighted the distinction between overall validation performance and the accuracy of individual predictions.

Both forms of evidence were retained when assessing the next candidates.

## 5. Generate and Compare Updated Model Candidates

The updated Gaussian Processes were used to evaluate candidate points using UCB and Expected Improvement.

As in previous rounds, different exploration settings were considered to assess how candidate locations changed as uncertainty received greater weight.

The neural-network ensembles also generated alternative recommendations, allowing cross-model comparisons.

### 5.1 Neural-Network Candidate Results

| Function | NN Predicted Output | Key Observation |
|---|---:|---|
| 1 | 0.015695 | NN fails LOOCV |
| 2 | 1.004493 | Optimistic prediction; previous overprediction raises concern |
| 3 | -0.005204 | Very local candidate; NN fails LOOCV |
| 4 | 0.117631 | Below current best of 0.583730 |
| 5 | 6187.342810 | Substantially below boundary best |
| 6 | -0.273307 | Below current best of -0.239769 |
| 7 | 2.900738 | Slightly below current best |
| 8 | 10.434491 | Large predicted improvement but distant and uncertain |

The neural-network recommendations varied considerably in reliability.

**Function 2** produced an optimistic prediction of approximately 1.0045, exceeding the current best of 0.7470. However, the candidate was located in a different region, and the NN had substantially overpredicted the previous submitted point.

The prediction was therefore not considered sufficient evidence for selection.

**Function 4** provided an interesting contrast. Although its neural network performed well in cross-validation, its current global recommendation was predicted substantially below the observed best.

The GP, which had demonstrated strong realised accuracy, remained preferable.

Function 5's NN recommendation was also considerably below the established best of 8662.48, supporting continued boundary-focused refinement.

Function 6's NN was reliable overall but did not identify a clear global improvement. This suggested that local refinement around the new best might be more useful.

**Function 8** produced the most optimistic NN prediction of approximately 10.4345.

However, this candidate was 0.677331 from its nearest observation and showed relatively high ensemble disagreement of 0.272532.

Given the deterioration in NN LOOCV and the GP's superior realised accuracy, this global NN candidate was considered too speculative.

### 5.2 GP–NN Candidate Agreement

The locations of NN candidates were compared with GP acquisition recommendations using Euclidean distance.

The strongest useful agreement occurred for Function 3, where the NN and lower-exploration GP approaches identified a similar region.

However, the NN failed LOOCV and the GP length scale had decreased considerably. Agreement therefore supported the general region rather than any individual prediction.

Function 4 also showed useful agreement between the NN and the more exploitative GP candidates.

Function 5 produced exact agreement between the NN and several lower-κ UCB candidates, although their predicted outputs remained below the established best.

Function 6 showed moderate agreement, with the NN candidate approximately 0.171 from the closest GP recommendation.

Function 7 also showed moderate agreement within the established high-performing region.

**Function 8 showed the greatest disagreement**, with its NN candidate approximately 0.949 from even the closest GP candidate.

Combined with its relatively high uncertainty and limited neighbourhood evidence, this weakened the case for following the NN recommendation.

Overall, agreement between models remained useful but was not treated as sufficient evidence without considering observed performance and validation.

## 6. Assess Candidate Neighbourhoods

The three nearest existing observations to representative NN, low-κ UCB and high-κ UCB candidates were examined.

This provided an empirical check on whether the proposed regions were supported by previously observed outputs.

### 6.1 Function-Specific Findings

**Function 1**

The nearby observations around both GP and NN candidates had outputs effectively equal to zero.

Neither region offered convincing evidence of a superior location.

Model-independent exploration therefore remained appropriate.

**Function 2**

The NN candidate was surrounded by observations producing approximately 0.4206, 0.2940 and -0.0820.

These were all below the current best of 0.7470.

The GP region showed even weaker historical performance.

Neither global recommendation provided sufficient justification to abandon the established best region.

**Function 3**

The NN candidate was only approximately 0.0168 from the current best observation of -0.000863.

The more exploitative GP candidate was also relatively close, while higher-exploration candidates moved into weaker regions.

This supported careful local refinement.

**Function 4**

The NN candidate was approximately 0.0530 from the latest best of 0.583730 and was also close to previous strong observations of 0.387262 and 0.186912.

The lower-exploration GP recommendation remained associated with the same region.

This provided strong evidence for continued local optimisation.

**Function 5**

The strongest neighbourhood remained concentrated near the upper boundary.

The current best of 8662.48 occurred at `[1, 1, 1, 1]`, while nearby deviations produced lower outputs.

The more exploratory GP regions were considerably weaker.

This reinforced the decision to use very small novelty-constrained boundary movements.

**Function 6**

Both NN and lower-exploration GP candidates were associated with the latest strong observations of -0.239769, -0.274388 and -0.379740.

This provided strong support for continued local refinement.

**Function 7**

The NN candidate was surrounded by observations producing 2.929370, 2.839196 and 2.922059.

These were among the strongest outputs recorded for the function.

Local refinement was therefore favoured over greater exploration.

**Function 8**

The global NN candidate was close to relatively weak observations of 9.141639 and 9.183005.

By contrast, the low-κ GP candidate was surrounded by outputs of 9.984981, 9.840530 and 9.930811.

This provided strong empirical support for following the GP-supported local region rather than the optimistic NN prediction.

Overall, the neighbourhood analysis favoured local refinement for Functions 2–8.

Function 1 remained the main exception because neither surrogate model nor observed neighbourhoods provided reliable guidance.

## 7. Multi-Radius Local Refinement

A more systematic local refinement process was introduced to compare candidate points at different distances from the current best observation.

For Functions 3, 4, 6, 7 and 8, local candidates were generated around their strongest observed points using approximate search radii of:

- `0.03` — fine local refinement.
- `0.05` — moderate local refinement.
- `0.10` — broader local refinement.

The candidates were assessed using GP predicted output, GP uncertainty, NN predicted output, ensemble disagreement and distance from existing observations.

Functions 1, 2 and 5 were treated separately because their circumstances required different strategies.

### 7.1 Function 2 — Targeted Local Refinement

Function 2's GP remained unreliable, while its NN had overpredicted the previous submitted point.

A targeted NN-supported local search was therefore performed around the current best observation of 0.746962.

| Radius | NN Predicted Output | Nearest Observation Distance |
|---|---:|---:|
| 0.03 | 0.631267 | 0.009635 |
| 0.05 | 0.622806 | 0.029174 |
| 0.10 | 0.606641 | 0.079817 |

Predicted output decreased as the local search radius increased.

This differed from the previous round, when a larger local move had produced an optimistic NN prediction that failed to materialise.

None of the updated candidates was predicted to exceed the current best.

The radius 0.03 candidate had the strongest prediction but was extremely close to an existing observation.

The radius 0.10 candidate provided more novelty but had weaker predicted performance.

**The radius 0.05 candidate was therefore selected as a compromise**, providing additional information about the established region without making another aggressive model-led move.

Selected point:

`[0.702093, 0.186930]`

### 7.2 Function 3 — Cautious Local Refinement

Function 3 required particular caution following its latest unsuccessful query and the reduction in its GP length scale.

The multi-radius search identified candidates in a similar local region.

Although the GP continued to predict promising outputs, the deterioration in NN validation and the most recent observed result suggested that small changes could materially affect performance.

A moderate radius of 0.05 was retained rather than relying on a larger exploratory movement.

Selected point:

`[0.440641, 0.482965, 0.435994]`

This preserved a local exploitation strategy while providing sufficient novelty to investigate the surrounding region.

### 7.3 Function 4 — GP-Supported Refinement

Function 4 had produced another substantial improvement and its GP demonstrated strong realised prediction accuracy.

The multi-radius comparison supported continued exploration immediately around the latest best observation.

The radius 0.05 candidate provided the strongest GP prediction among the locally evaluated alternatives while remaining close to the established successful region.

Selected point:

`[0.388750, 0.360865, 0.352431, 0.416723]`

This was a deliberately exploitative selection supported by both recent performance and GP reliability.

### 7.4 Function 5 — Boundary Refinement

Function 5 continued to present evidence that the optimum lay close to the upper input boundary.

The current best of 8662.48 had been observed at `[1, 1, 1, 1]`, while recent deviations away from this point had produced weaker outputs.

The additional analysis therefore focused on selecting a novel point close to the boundary rather than moving into a substantially different region.

A small novelty constraint was applied to ensure that the candidate was not effectively identical to a previous observation.

Selected point:

`[1.000000, 1.000000, 1.000000, 0.969985]`

The selection tested another small departure from the established optimum region while preserving the strong exploitation focus.

### 7.5 Function 6 — Balanced Local Refinement

Function 6's latest query produced a new best of -0.239769.

The GP remained informative, while the NN demonstrated strong cross-validation and accurate realised predictions.

This provided evidence for continued local refinement using both surrogates.

The radius 0.05 candidate offered a balance between predicted improvement and uncertainty, without the greater movement associated with radius 0.10.

Selected point:

`[0.444874, 0.347937, 0.533470, 0.742972, 0.000000]`

This allowed the search to continue investigating a productive region while avoiding unnecessarily broad exploration.

### 7.6 Function 7 — Model-Supported Local Refinement

Function 7 remained close to its observed maximum, although the latest query had failed to improve.

Both GP and NN recommendations continued to favour the established successful region.

The radius 0.05 candidate provided relatively strong agreement between the two models without the higher uncertainty associated with radius 0.10.

Selected point:

`[0.145227, 0.296570, 0.489754, 0.259986, 0.333711, 0.644995]`

This maintained a local strategy while allowing a new observation to be collected.

### 7.7 Function 8 — GP-Led Local Refinement

Function 8 achieved another new best of 9.984981.

The GP had predicted the previous output accurately, while the NN's current global recommendation was distant, uncertain and weakly supported by neighbouring observations.

The multi-radius search therefore prioritised a candidate close to the established successful region.

The radius 0.05 candidate offered useful novelty without the larger movement associated with radius 0.10.

Selected point:

`[0.000000, 0.221175, 0.182914, 0.081049, 0.771543, 0.527149, 0.198086, 0.585768]`

This reflected a relatively conservative approach to improving an already strong result.

### 7.8 Function 1 — Continued Space-Filling Exploration

Function 1 continued to experience substantial difficulties with both surrogate models.

Its GP remained at the minimum length scale and its NN failed cross-validation.

Neither model provided sufficiently convincing evidence for a model-driven query.

Maximum-distance exploration was therefore retained.

Selected point:

`[0.002495, 0.694432]`

Its distance to the nearest existing observation was approximately 0.301746, providing additional coverage of a relatively unexplored region.

## 8. Final Week 10 Query Selections

The final candidates were selected using the accumulated evidence from realised prediction accuracy, GP fitting, NN validation, acquisition-function comparisons, candidate neighbourhoods and multi-radius local searches.

Each candidate was checked for dimensionality, bounds and duplication against existing observations.

### Final Proposed Queries

| Function | Selection Strategy | Final Proposed Point | Nearest Observation Distance |
|---|---|---|---:|
| 1 | Space-filling exploration | `[0.002495, 0.694432]` | 0.301746 |
| 2 | Radius 0.05 local NN refinement | `[0.702093, 0.186930]` | 0.029174 |
| 3 | Radius 0.05 local refinement | `[0.440641, 0.482965, 0.435994]` | Approximately 0.03–0.06 |
| 4 | Radius 0.05 GP refinement | `[0.388750, 0.360865, 0.352431, 0.416723]` | 0.007193 |
| 5 | Boundary refinement | `[1.000000, 1.000000, 1.000000, 0.969985]` | 0.030015 |
| 6 | Radius 0.05 GP/NN refinement | `[0.444874, 0.347937, 0.533470, 0.742972, 0.000000]` | Approximately 0.03–0.06 |
| 7 | Radius 0.05 GP/NN refinement | `[0.145227, 0.296570, 0.489754, 0.259986, 0.333711, 0.644995]` | Approximately 0.03–0.06 |
| 8 | Radius 0.05 GP refinement | `[0.000000, 0.221175, 0.182914, 0.081049, 0.771543, 0.527149, 0.198086, 0.585768]` | 0.099346 |

All eight final candidates were reported as valid in the saved validation results.

The different nearest-observation distances reflected deliberate differences in strategy.

Function 1 remained the most exploratory, while Functions 4 and 5 selected points extremely close to established successful regions.

Functions 2, 3, 6 and 7 made relatively small local adjustments, and Function 8 retained a somewhat broader but still locally supported move.

### Submission Format

    Function 1: 0.002495-0.694432
    Function 2: 0.702093-0.186930
    Function 3: 0.440641-0.482965-0.435994
    Function 4: 0.388750-0.360865-0.352431-0.416723
    Function 5: 1.000000-1.000000-1.000000-0.969985
    Function 6: 0.444874-0.347937-0.533470-0.742972-0.000000
    Function 7: 0.145227-0.296570-0.489754-0.259986-0.333711-0.644995
    Function 8: 0.000000-0.221175-0.182914-0.081049-0.771543-0.527149-0.198086-0.585768

## 9. Week 10 Reflection

Week 10 demonstrated the value of assessing surrogate predictions against realised outputs rather than relying solely on model diagnostics and cross-validation.

Three of the previous queries produced new best observations, with particularly encouraging results for Functions 4, 6 and 8.

The realised prediction comparison provided important additional evidence about model reliability.

The Gaussian Process accurately predicted the latest outcomes for Functions 4 and 8, supporting its continued use for local refinement. The neural network was particularly accurate for Function 6, reinforcing its value as a supplementary surrogate.

However, Function 2 highlighted the risks of relying on attractive NN predictions. Its forecast improvement failed to materialise despite the model continuing to outperform the mean baseline under LOOCV.

Function 3 also presented difficulties. Its latest query performed substantially worse despite being very close to the previous best, while its GP length scale decreased considerably. These results suggested that its high-performing region could be particularly narrow.

Function 5 continued to provide evidence that its strongest region lay near `[1, 1, 1, 1]`. The latest unsuccessful query supported very small boundary-focused movements rather than greater exploration.

The introduction of multi-radius local refinement provided a more structured way to evaluate the trade-off between predicted improvement and candidate novelty.

Comparing radii of 0.03, 0.05 and 0.10 allowed the local search to be adapted to each function, rather than applying the same degree of exploration universally.

For Functions 4, 6, 7 and 8, radius 0.05 provided a useful balance between remaining close to successful regions and gathering additional information.

Function 2 also selected the moderate-radius candidate, although no model-supported improvement was identified. This was considered preferable to following another speculative global NN recommendation.

Function 1 remained the principal exception, continuing to use space-filling exploration because neither surrogate model provided reliable guidance.

Overall, Week 10 strengthened the evidence-based approach developed throughout the capstone. Realised prediction accuracy, cross-validation, GP diagnostics and observed neighbourhoods were considered together when selecting candidates.

The following round will evaluate whether these choices produce further improvements and investigate whether additional techniques can help identify useful structure in the accumulated observations.

---

**Associated code:** `Week_10.ipynb`

