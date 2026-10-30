
# Week 7 — Hyperparameter Tuning and Candidate Robustness

**Imperial College AI/ML Capstone Project | Module 18**

## Objective

The objective of Week 7 was to investigate the sensitivity and robustness of the existing Bayesian Optimisation methods by tuning the exploration parameters used in Upper Confidence Bound (UCB) and Expected Improvement (EI).

Rather than introducing another modelling approach, the focus was on understanding how acquisition settings influenced candidate selection and whether different settings identified similar promising regions.

The Gaussian Process (GP) remained the primary surrogate model, while the neural-network (NN) ensembles continued to provide independent predictions where their validation performance supported their use.

Final selections considered acquisition-function stability, GP reliability, NN cross-validation, historical optimisation performance and the observed neighbourhood surrounding each candidate.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 6 queries were added to the accumulated datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | ~0.000000 | 0.000000 | No improvement |
| 2 | 0.746962 | 0.471844 | 0.746962 | No improvement |
| 3 | -0.016451 | -0.015498 | -0.015498 | New best |
| 4 | -0.300645 | -0.200431 | -0.200431 | New best |
| 5 | 6691.931674 | 7121.151955 | 7121.151955 | New best |
| 6 | -0.456874 | -0.379740 | -0.379740 | New best |
| 7 | 2.536939 | 2.929370 | 2.929370 | New best |
| 8 | 9.911249 | 9.840530 | 9.911249 | No improvement |

Five of the eight queries produced new best observations.

**Expected Improvement** was particularly successful, improving Functions 3, 4 and 6. Function 4 increased from approximately -0.301 to -0.200, while Function 6 improved from -0.457 to -0.380.

**Function 5** continued its strong performance, improving from 6691.93 to 7121.15 using the neural-network recommendation. This provided further evidence that the NN could identify productive local refinements within its high-performing region.

**Function 7** also improved substantially, increasing from 2.537 to 2.929 using a candidate supported by both UCB and EI.

Functions 2 and 8 did not improve. Function 8's NN recommendation failed for the second consecutive round, suggesting that its predictions should be treated more cautiously despite reasonable validation performance.

Function 1 again returned an output effectively equal to zero, highlighting the continuing difficulty of modelling this function.

Overall, the results supported retaining a mixed, function-specific approach rather than relying on a single surrogate or acquisition function.

## 2. Refit and Diagnose the Gaussian Process Models

The RBF Gaussian Processes were refitted using all observations available at the beginning of Week 7.

The fitted kernel length scales were examined to identify changes in model behaviour following the latest observations.

| Function | Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0431 |
| 3 | 0.193 |
| 4 | 0.436 |
| 5 | 0.363 |
| 6 | 0.558 |
| 7 | 0.300 |
| 8 | 1.020 |

Most functions continued to produce relatively stable fitted length scales.

Functions 3, 4 and 6 showed modest increases following their successful queries, while Function 7 increased from approximately 0.248 to 0.300.

**Function 1 remained the exception**, continuing to reach the lower RBF length-scale bound of `1e-05`. This suggested that its GP was still unable to identify a sufficiently informative relationship between observations.

Function 2's length scale decreased to approximately 0.0431, suggesting a relatively local response structure. Functions 5 and 8 remained broadly stable.

### 2.1 Prediction and Uncertainty Diagnostics

Each GP was evaluated across 100,000 candidate points to assess its ability to distinguish between different regions of the search space.

| Function | Maximum Predicted Output | Maximum Uncertainty |
|---|---:|---:|
| 1 | Approximately 0 | 0.001626 |
| 2 | 0.7905 | 0.2550 |
| 3 | -0.0118 | 0.0782 |
| 4 | -0.2003 | 7.4108 |
| 5 | 7121.16 | 2074.52 |
| 6 | -0.3656 | 0.5196 |
| 7 | 2.9294 | 0.7721 |
| 8 | 9.9641 | 0.6576 |

Functions 2–8 produced meaningful variation in predictions and uncertainty, allowing acquisition-function tuning to proceed.

Functions 3–7 produced predicted maxima relatively close to their strongest existing observations, indicating that the GPs were identifying the recently successful regions.

Function 1 continued to show limited predictive variation, meaning its acquisition recommendations required particular caution.

## 3. Tune the UCB Exploration Parameter

The UCB acquisition function was evaluated using four exploration settings:

- `κ = 0.5` — greater emphasis on predicted performance.
- `κ = 1.0` — relatively exploitative.
- `κ = 1.5` — intermediate exploration.
- `κ = 2.5` — greater emphasis on uncertainty.

The same 100,000 candidate points were used for every setting to ensure that differences arose from changes in κ rather than candidate generation.

### 3.1 UCB Sensitivity Results

| Function | Sensitivity | Main Observation |
|---|---|---|
| 1 | None | Same candidate at all κ values |
| 2 | Gradual | Distance from best increased from 0.021 to 0.048 |
| 3 | Gradual | Distance increased from 0.289 to 0.474 |
| 4 | Threshold | Same candidate at κ = 0.5–1.5; substantial exploration at 2.5 |
| 5 | Threshold | Existing best selected at low κ; exploratory candidate at 2.5 |
| 6 | Strong gradual | Distance increased from 0.163 to 0.677 |
| 7 | Threshold | Existing best at κ = 0.5; increasingly exploratory thereafter |
| 8 | Gradual | Distance increased from 0.345 to 0.761 |

The results demonstrated that acquisition settings could materially influence candidate selection.

For **Functions 2, 3, 6 and 8**, increasing κ progressively moved recommendations further from the current best observations.

Functions 4, 5 and 7 showed more sudden changes, with larger exploration weightings selecting substantially different candidates.

Function 1 remained unchanged across all four settings. However, because its GP continued to experience convergence difficulties, this consistency was not considered strong evidence of predictive reliability.

Overall, the comparison suggested that a single fixed κ value was unlikely to be appropriate across all eight functions.

## 4. Tune the Expected Improvement Parameter

Expected Improvement was evaluated using ξ values equivalent to 0%, 1%, 5% and 10% of each function's observed output standard deviation.

Scaling ξ relative to output variability allowed comparable exploration settings to be assessed despite the considerable differences in output magnitude between functions.

Increasing ξ requires a greater improvement over the current best and can encourage selection of more uncertain candidates.

### 4.1 EI Sensitivity Results

| Function | Sensitivity | Main Observation |
|---|---|---|
| 1 | None | Identical candidate across all ξ values |
| 2 | Very low | Only small changes in candidate location |
| 3 | Gradual | Distance from best increased from 0.347 to 0.389 |
| 4 | Strong threshold | Local candidate at 0%; highly exploratory from 1% |
| 5 | None | Same candidate at all ξ values |
| 6 | Threshold | Same at 0–1%; more exploratory at 5–10% |
| 7 | None | Same candidate across all ξ values |
| 8 | Strong gradual | Distance from best increased from 0.345 to 0.761 |

The EI analysis demonstrated similar function-specific sensitivity.

**Functions 1, 5 and 7** showed no change in selection across the tested settings. Function 2 also remained relatively stable.

Functions 3 and 8 became progressively more exploratory as ξ increased, while Functions 4 and 6 showed clearer threshold effects.

Function 4 was particularly sensitive. Increasing ξ from 0% to 1% shifted the selected candidate from only 0.0775 from the current best to approximately 0.9250 away.

These results reinforced the importance of assessing acquisition parameters individually rather than automatically applying the same settings across every function.

## 5. Retrain and Validate the Neural Networks

The neural-network ensembles introduced in Week 5 were retrained using the expanded datasets.

The architecture remained unchanged, with five independently initialised models per function and two hidden layers containing 16 neurons each.

Leave-One-Out Cross-Validation (LOOCV) was repeated to determine whether the networks generalised beyond their training observations.

Predictive performance was measured using Mean Absolute Error (MAE) and compared with a simple mean-prediction baseline.

### 5.1 LOOCV Results

| Function | NN LOOCV MAE | Mean Baseline MAE | NN Beats Baseline |
|---|---:|---:|---|
| 1 | 0.001776 | 0.001111 | No |
| 2 | 0.218453 | 0.237181 | Yes |
| 3 | 0.063001 | 0.050129 | No |
| 4 | 1.906639 | 7.081622 | Yes |
| 5 | 228.443505 | 1689.428734 | Yes |
| 6 | 0.235898 | 0.497547 | Yes |
| 7 | 0.376668 | 0.586378 | Yes |
| 8 | 0.358344 | 0.941538 | Yes |

The neural networks outperformed the baseline for **six of the eight functions**.

Functions 4, 5, 6 and 8 showed particularly strong relative improvements.

Function 2 also passed the validation comparison after failing in the previous round, with its MAE decreasing from approximately 0.2674 to 0.2185.

Functions 1 and 3 continued to perform worse than the baseline, meaning their NN recommendations would receive limited weight.

Although training losses remained low across all functions, these validation results demonstrated why model generalisation was more important than training performance alone.

## 6. Compare GP and Neural-Network Candidates

The validated neural-network ensembles generated alternative query recommendations for each function.

These were compared with the candidates identified during UCB and EI tuning.

### 6.1 Neural-Network Recommendations

| Function | NN Predicted Output | Model Disagreement | Distance from Current Best |
|---|---:|---:|---:|
| 1 | 0.000741 | 0.000834 | 0.2664 |
| 2 | 0.706124 | 0.082928 | 0.0210 |
| 3 | -0.014537 | 0.003243 | 0.0725 |
| 4 | -0.001964 | 0.398781 | 0.0775 |
| 5 | 7103.413178 | 20.900199 | 0.0000 |
| 6 | -0.347172 | 0.039748 | 0.1737 |
| 7 | 2.977398 | 0.203612 | 0.1979 |
| 8 | 10.264397 | 0.164985 | 0.3583 |

Several results were particularly informative.

**Function 2** produced a neural-network candidate close to the current best observation, supported by its newly improved validation performance.

**Function 4** showed particularly strong agreement between the models. Its neural-network candidate was identical to the recommendations produced by UCB at κ = 0.5–1.5 and EI at ξ = 0%.

**Function 5** returned the existing best observation rather than a new point. This demonstrated the models' preference for the successful region but required an additional local search to identify a valid new candidate.

Functions 6 and 7 produced plausible NN recommendations near previously successful regions, while Function 8 again predicted an improvement above its current best.

However, Function 8's two previous unsuccessful NN queries meant this prediction required caution.

### 6.2 Quantitative Model Agreement

The Euclidean distance between each neural-network candidate and its closest GP acquisition candidate was calculated.

| Function | Closest NN–GP Distance | Interpretation |
|---|---:|---|
| 1 | 0.3384 | Substantial disagreement |
| 2 | 0.0022 | Very strong agreement |
| 3 | 0.2358 | Limited agreement |
| 4 | 0.0000 | Exact agreement |
| 5 | 0.0000 | Exact agreement, but duplicate observation |
| 6 | 0.1857 | Moderate agreement |
| 7 | 0.1785 | Moderate agreement |
| 8 | 0.2994 | Some agreement with exploitative GP settings |

Functions 2 and 4 provided the strongest cross-model evidence.

For Function 2, the NN recommendation was only 0.0022 from the low-κ UCB and zero-ξ EI candidates.

Function 4 showed exact agreement between the NN and multiple GP acquisition settings.

Function 5 also showed exact agreement in the established high-performing region, although its selected point was already an observation and therefore could not be submitted again.

The weaker agreement for Functions 1 and 3 reinforced the decision to give their neural-network predictions limited weight.

## 7. Assess Candidate Neighbourhoods

The three nearest existing observations to representative UCB, EI and NN candidates were examined.

This allowed model recommendations to be compared with actual historical performance in their surrounding regions.

The neighbourhood analysis produced several useful findings:

- **Functions 2 and 4:** Strong local evidence supported the GP and NN recommendations, with candidates close to previous high-performing observations.
- **Function 3:** The NN selected a locally promising point, but its failure to pass LOOCV reduced confidence in that recommendation.
- **Function 5:** The models consistently favoured the high-performing region near the current best, although several candidates duplicated an existing observation.
- **Functions 6 and 7:** Lower-exploration GP candidates remained close to recently successful regions, while higher-exploration settings moved further away.
- **Function 8:** The NN and more exploitative GP recommendations remained closer to strong observations than the highly exploratory alternatives.
- **Function 1:** Both GP and NN candidates were near observations effectively equal to zero, providing little evidence of a superior region.

Overall, this favoured local refinement for several functions rather than increasing exploration unnecessarily.

## 8. Final Function-Specific Selections

The final decisions combined recent query performance, GP diagnostics, hyperparameter stability, NN validation, cross-model agreement and local observed performance.

| Function | Selected Strategy | Reasoning |
|---|---|---|
| 1 | GP UCB/EI agreement | All GP settings agree; NN fails validation |
| 2 | UCB κ = 0.5 / EI ξ = 0% | Strong GP–NN agreement and local support |
| 3 | UCB κ = 0.5 | Controlled exploitation; NN fails validation |
| 4 | GP/NN agreement | Exact agreement across multiple settings |
| 5 | GP/NN local refinement | Existing best repeatedly selected; new nearby point required |
| 6 | UCB κ = 0.5 | Recent improvement and support for lower exploration |
| 7 | UCB κ = 1.5 / EI agreement | Novel point near successful region |
| 8 | UCB κ = 0.5 / EI ξ = 0% | More conservative than repeatedly unsuccessful NN recommendations |

### Key Decisions

**Function 1** remained the least reliable modelling problem. Its GP continued to reach the lower length-scale bound, and the NN failed LOOCV. Nevertheless, all tested UCB and EI settings selected the same candidate, providing the most consistent available recommendation.

**Function 2** demonstrated particularly strong agreement between the GP and NN, with both identifying a region close to the current best output of 0.746962.

**Function 4** was one of the strongest selections, with exact agreement between the NN, several UCB settings and EI at ξ = 0%.

**Function 5** required a dedicated local-refinement search because the standard NN and low-κ UCB selections returned the existing best observation. A new candidate at the upper boundary `[1, 1, 1, 1]` was selected. The GP predicted approximately 7920.78, while the NN predicted 8059.00, compared with the current best of 7121.15.

**Function 6** retained a relatively exploitative UCB setting following its recent improvement, while Function 7 selected a novel point near the region that had produced its latest best.

**Function 8** returned to a more conservative GP recommendation after two consecutive unsuccessful NN selections.

## 9. Final Week 7 Query Selections

All eight candidates were checked for dimensionality, permitted input bounds and duplication against existing observations.

| Function | Selection Method | Final Proposed Point | Nearest Observation Distance |
|---|---|---|---:|
| 1 | UCB / EI agreement | `[0.858598, 0.697368]` | 0.117860 |
| 2 | UCB κ = 0.5 / EI ξ = 0% | `[0.706079, 0.158030]` | 0.021076 |
| 3 | UCB κ = 0.5 | `[0.450320, 0.561567, 0.490526]` | 0.129453 |
| 4 | GP/NN agreement | `[0.384332, 0.391111, 0.336887, 0.416488]` | 0.066534 |
| 5 | Local refinement | `[1.000000, 1.000000, 1.000000, 1.000000]` | 0.064222 |
| 6 | UCB κ = 0.5 | `[0.387946, 0.435172, 0.345126, 0.743513, 0.076825]` | 0.163215 |
| 7 | UCB κ = 1.5 / EI agreement | `[0.054205, 0.216830, 0.421532, 0.346756, 0.385001, 0.543902]` | 0.208059 |
| 8 | UCB κ = 0.5 / EI ξ = 0% | `[0.030327, 0.231267, 0.050266, 0.208295, 0.818520, 0.414744, 0.307096, 0.583908]` | 0.345195 |

All eight points passed the saved validation checks.

Function 5's boundary point was confirmed to be a valid, novel selection, while the remaining candidates represented varying levels of local refinement and exploration.

### Submission Format

```text
Function 1: 0.858598-0.697368
Function 2: 0.706079-0.158030
Function 3: 0.450320-0.561567-0.490526
Function 4: 0.384332-0.391111-0.336887-0.416488
Function 5: 1.000000-1.000000-1.000000-1.000000
Function 6: 0.387946-0.435172-0.345126-0.743513-0.076825
Function 7: 0.054205-0.216830-0.421532-0.346756-0.385001-0.543902
Function 8: 0.030327-0.231267-0.050266-0.208295-0.818520-0.414744-0.307096-0.583908
```

## 10. Week 7 Reflection

Week 7 demonstrated the importance of testing acquisition-function sensitivity rather than relying on a single fixed exploration parameter.

Five of the eight previous queries produced new best observations, including particularly encouraging results for Functions 5 and 7. This provided further support for the increasingly function-specific optimisation strategy.

Comparing multiple UCB κ and EI ξ settings showed that some functions were relatively stable, while others changed substantially as greater emphasis was placed on exploration. Functions 4, 5 and 7 demonstrated particularly clear threshold effects, while Functions 2, 3, 6 and 8 showed more gradual shifts.

Neural-network validation also improved, with six of the eight functions outperforming the mean baseline. Function 2 passed LOOCV for the first time, strengthening the evidence for its neural-network recommendations. However, Functions 1 and 3 continued to fail validation, reinforcing the need to evaluate model reliability individually.

Cross-model agreement became an increasingly useful diagnostic. Functions 2 and 4 showed particularly strong agreement between GP and NN recommendations, while Function 5's models consistently identified the region surrounding its existing best observation.

The local-refinement strategy adopted for Function 5 was the most distinctive decision this week. Rather than resubmitting an existing point, a new candidate at the upper boundary was selected with support from both surrogate models.

Overall, Week 7 placed greater emphasis on acquisition robustness, predictive validation and observed neighbourhood evidence. The resulting selections were designed to refine promising regions where the evidence was strong while retaining more exploratory behaviour where appropriate.

The following round will assess whether these choices improve the observed maxima and whether the additional data further strengthens the reliability of the surrogate models.

---

**Associated code:** `Week_07.ipynb`

