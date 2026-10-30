
# Week 6 — Incorporating Historical Model Performance into Query Selection

**Imperial College AI/ML Capstone Project | Module 17**

## Objective

The objective of Week 6 was to incorporate the latest observations, reassess the Gaussian Process (GP) and neural-network surrogate models, and introduce a more systematic consideration of previous optimisation performance when selecting new queries.

Week 5 had demonstrated that neural networks could provide useful alternative recommendations, particularly where the Gaussian Processes struggled. However, the latest results also highlighted that attractive model predictions did not necessarily translate into improvements.

The main development this week was therefore to examine how different selection methods had performed in previous rounds and use this information alongside current model predictions, uncertainty, validation performance and neighbouring observations.

Rather than introducing a formal weighting system or additional model complexity, historical results would provide supporting evidence for function-specific decisions.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 5 queries were incorporated into the datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | ~0.000000 | 0.000000 | No improvement |
| 2 | 0.611205 | 0.746962 | 0.746962 | New best |
| 3 | -0.020167 | -0.016451 | -0.016451 | New best |
| 4 | -0.300645 | -0.638343 | -0.300645 | No improvement |
| 5 | 4857.449601 | 6691.931674 | 6691.931674 | New best |
| 6 | -0.458511 | -0.456874 | -0.456874 | New best |
| 7 | 2.536939 | 1.644736 | 2.536939 | No improvement |
| 8 | 9.911249 | 9.742277 | 9.911249 | No improvement |

Four of the eight queries produced new best observations.

**Function 5** showed the largest improvement, increasing from approximately 4857.45 to 6691.93. This continued its strong performance and provided further evidence that the optimisation had identified a particularly productive region.

**Functions 2 and 3** also improved using Expected Improvement (EI), supporting the decision to favour this acquisition function in the previous round.

**Function 6** achieved a small improvement of approximately 0.00164 using its neural-network candidate. Although relatively minor, this provided some evidence that the neural network could identify promising points for a function whose GP had previously struggled.

Functions 4, 7 and 8 failed to improve. Function 1 also returned an output effectively equal to zero.

These mixed results provided a useful basis for assessing the historical reliability of the different selection methods.

## 2. Refit and Diagnose the Gaussian Process Models

The RBF Gaussian Processes were refitted using all observations collected so far. The existing kernel configuration and optimisation settings were retained for consistency with previous rounds.

### 2.1 Fitted RBF Length Scales

| Function | Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0521 |
| 3 | 0.185 |
| 4 | 0.423 |
| 5 | 0.389 |
| 6 | 0.535 |
| 7 | 0.248 |
| 8 | 1.000 |

Most functions continued to produce fitted length scales above the lower bound.

**Function 1** remained problematic, with its length scale continuing to reach `1e-05`, suggesting that the GP was still struggling to represent relationships across its input space.

The most significant change occurred in **Function 6**, where the fitted length scale increased from `1e-05` in Week 5 to approximately `0.535`.

This was particularly encouraging because Function 6 had repeatedly experienced difficulties with GP fitting. The additional observation appeared to have improved the model's ability to identify relationships between different regions of the search space.

As a result, GP-based recommendations for Function 6 could now be reconsidered alongside the neural-network approach.

### 2.2 Prediction and Uncertainty Diagnostics

Each GP was evaluated across 100,000 candidate points to determine whether its predictions and uncertainty meaningfully varied across the search space.

| Function | Predicted Mean Range | Uncertainty Range |
|---|---:|---:|
| 1 | 0.000635 | 0.001654 |
| 2 | 0.831417 | 0.259242 |
| 3 | 0.389101 | 0.078040 |
| 4 | 34.434331 | 7.386272 |
| 5 | 7326.020374 | 1726.249284 |
| 6 | 2.348109 | 0.508036 |
| 7 | 2.575816 | 0.664717 |
| 8 | 4.822485 | 0.659779 |

The results showed meaningful variation in predicted outputs and uncertainty for Functions 2–8.

Function 6 demonstrated a particularly important improvement, with predicted means ranging from approximately -2.662 to -0.313. This contrasted with previous rounds where its GP frequently produced almost identical predictions across candidate points.

Function 1 continued to show limited predictive variation and retained its extremely small length scale.

The remaining functions produced sufficiently varied predictions to proceed with GP-based acquisition methods.

## 3. Generate UCB and Expected Improvement Candidates

The updated Gaussian Processes were used to generate candidate points using two acquisition functions:

- **Upper Confidence Bound (UCB):** Retained with `κ = 1.5`, following the more exploitative approach introduced in Week 4.
- **Expected Improvement (EI):** Used to identify candidates with the greatest expected improvement over the current best observed output.

Both acquisition functions were evaluated across candidate points within the permitted `[0, 1]` search space.

### 3.1 Comparison of GP Candidates

| Function | UCB Predicted Output | EI Predicted Output | Distance Between Candidates |
|---|---:|---:|---:|
| 1 | -0.000635 | -0.000635 | 0.0000 |
| 2 | 0.644023 | 0.679240 | 0.0077 |
| 3 | -0.048421 | -0.010868 | 0.3695 |
| 4 | -0.455899 | -0.278717 | 0.0766 |
| 5 | 6213.849027 | 6213.849027 | 0.0000 |
| 6 | -0.338919 | -0.313445 | 0.1444 |
| 7 | 2.512543 | 2.512543 | 0.0000 |
| 8 | 9.793025 | 9.793025 | 0.0000 |

The updated GP models produced several areas of agreement.

**Functions 1, 5, 7 and 8** selected identical candidates under UCB and EI, while Function 2's recommendations were separated by only 0.0077.

Functions 3, 4 and 6 showed greater differences between the acquisition functions, suggesting that the exploration–exploitation trade-off remained important.

Several findings stood out:

- **Function 2:** Both acquisition functions selected candidates very close to the new best observation of 0.746962, with EI only 0.0309 away.
- **Function 3:** EI predicted an output of approximately -0.0109, exceeding the current best of -0.0165. UCB selected a more exploratory candidate with a weaker predicted output.
- **Function 5:** UCB and EI selected exactly the same point in a region that had produced repeated improvements.
- **Function 6:** Following the GP's recovery, EI predicted approximately -0.3134 against a current best of -0.4569, suggesting a potentially significant improvement.
- **Function 7:** The acquisition functions again agreed, although the previous GP-selected candidate had failed to improve.
- **Function 8:** UCB and EI now agreed despite producing substantially different recommendations in Week 5. However, their candidate remained relatively distant from the best-known observation.

Function 1's agreement required particular caution. Both acquisition functions relied on the same GP, whose fitted length scale continued to indicate modelling difficulties.

Agreement between acquisition functions was therefore treated as supporting evidence rather than independent confirmation of a reliable prediction.

## 4. Retrain and Evaluate the Neural-Network Models

The neural-network ensembles introduced in Week 5 were retained without changing their architecture.

Each function continued to use five independently initialised networks, each containing two hidden layers with 16 neurons per layer and ReLU activation functions.

Keeping the architecture unchanged allowed its behaviour to be compared more consistently as additional observations became available.

### 4.1 Assess Predictive Performance

Leave-One-Out Cross-Validation (LOOCV) was used to evaluate the neural networks' ability to predict observations excluded from training.

The resulting Mean Absolute Error (MAE) was compared with a simple mean-prediction baseline.

| Function | NN LOOCV MAE | Mean Baseline MAE | NN Beats Baseline |
|---|---:|---:|---|
| 1 | 0.001675 | 0.001179 | No |
| 2 | 0.267443 | 0.241496 | No |
| 3 | 0.067227 | 0.048537 | No |
| 4 | 1.827993 | 6.878180 | Yes |
| 5 | 273.005550 | 1375.410321 | Yes |
| 6 | 0.309082 | 0.476405 | Yes |
| 7 | 0.323396 | 0.507559 | Yes |
| 8 | 0.294854 | 0.920553 | Yes |

The neural networks outperformed the baseline for **five of the eight functions**.

Functions 4–8 continued to show evidence of useful predictive relationships.

However, Functions 1–3 failed to outperform the baseline. Function 2 was particularly notable because its neural network had passed this comparison in Week 5 but performed slightly worse than the baseline in Week 6.

This reinforced the importance of assessing generalisation rather than relying on low training losses or attractive predictions.

### 4.2 Generate Neural-Network Candidates

The trained ensembles were used to evaluate candidate points, combining predictions from the five networks to obtain an average predicted output and a measure of disagreement.

| Function | NN Predicted Output | Model Disagreement |
|---|---:|---:|
| 1 | 0.003677 | 0.002351 |
| 2 | 2.724940 | 1.237037 |
| 3 | -0.001139 | 0.032445 |
| 4 | -0.374836 | 0.459252 |
| 5 | 6691.842285 | 47.407719 |
| 6 | -0.406389 | 0.128387 |
| 7 | 2.412130 | 0.419323 |
| 8 | 10.052735 | 0.132650 |

The neural-network recommendations varied considerably.

**Function 5** identified a candidate close to its new best observation, with a predicted output of approximately 6691.84. This provided further support for the established high-performing region.

**Function 6** again selected a candidate near its strongest observations, although the improved GP now provided credible competing recommendations.

**Function 8** produced a predicted output of approximately 10.05, above the existing best of 9.91.

In contrast, Function 2 produced a very high predicted output of 2.72, but this was accompanied by substantial disagreement between the individual networks and poor cross-validation performance.

Functions 1 and 3 also produced potentially attractive predictions but failed the validation comparison.

The neural-network recommendations were therefore considered alongside GP predictions and historical results rather than accepted automatically.

## 5. Compare GP and Neural-Network Recommendations

The UCB, EI and neural-network candidates were compared to determine whether the models identified similar promising regions.

### 5.1 Candidate Agreement

The strongest agreement between acquisition methods occurred in Functions 2, 5, 7 and 8.

However, the neural-network recommendations sometimes differed substantially from those generated by the Gaussian Processes.

**Function 5** showed particularly encouraging agreement in the general region being explored. Both model types continued to target the area that had produced successive improvements.

**Function 6** also showed encouraging convergence. Following the GP's recovery, both the GP and neural-network approaches identified candidates around the strongest existing observations.

Function 8 presented a more difficult decision. The GP acquisition functions agreed exactly, but their candidate was relatively distant from existing data. The neural network proposed a different candidate with a stronger predicted output.

For Functions 1–3, weaker neural-network validation reduced confidence in using their NN recommendations, particularly where the GP candidates were supported by nearby observations.

These comparisons demonstrated that agreement between models could be informative, but predictions still needed to be assessed against observed results.

## 6. Introduce Historical Model-Performance Analysis

The main methodological development in Week 6 was to examine whether previously selected methods had successfully improved the observed maxima.

Rather than assigning fixed numerical weights to different models, this historical evidence was used to inform confidence in their current recommendations.

### 6.1 Performance of Week 5 Selections

| Function | Previous Selection Method | Improvement | Outcome |
|---|---|---:|---|
| 1 | UCB / EI agreement | ~0.000000 | No improvement |
| 2 | Expected Improvement | +0.135757 | New best |
| 3 | Expected Improvement | +0.003716 | New best |
| 4 | Expected Improvement | -0.337698 | No improvement |
| 5 | Expected Improvement | +1834.482072 | New best |
| 6 | Neural Network | +0.001637 | New best |
| 7 | UCB / EI agreement | -0.892203 | No improvement |
| 8 | Neural Network | -0.168972 | No improvement |

The historical comparison provided several useful insights.

**Expected Improvement** had produced successful results for Functions 2, 3 and 5. Function 5's improvement was particularly substantial and strengthened the case for continued exploitation of its high-performing region.

**The neural network** produced a small improvement for Function 6, supporting its continued consideration as a supplementary surrogate.

However, the neural-network recommendation for Function 8 had failed to improve despite an attractive predicted output.

Similarly, the UCB/EI agreement candidates for Functions 1 and 7 had not produced improvements.

This demonstrated that agreement between acquisition functions, or an attractive surrogate prediction, did not guarantee success.

Historical performance was therefore treated as additional evidence rather than a strict selection rule, recognising that an individual unsuccessful query did not necessarily indicate a poor model.

## 7. Compare Candidate Neighbourhoods

The neighbourhood surrounding each proposed candidate was examined using the actual observed outputs.

For each UCB, EI and neural-network recommendation, the three nearest existing observations were identified and considered alongside the model predictions.

This provided a direct assessment of whether candidates were located near previously successful or unsuccessful regions.

### 7.1 Distance to Nearest Existing Observation

| Function | UCB Distance | EI Distance | NN Distance |
|---|---:|---:|---:|
| 1 | 0.1807 | 0.1807 | 0.3077 |
| 2 | 0.0385 | 0.0309 | 0.3164 |
| 3 | 0.2255 | 0.1407 | 0.2033 |
| 4 | 0.1042 | 0.1049 | 0.0619 |
| 5 | 0.1651 | 0.1651 | 0.0286 |
| 6 | 0.2812 | 0.2131 | 0.2223 |
| 7 | 0.1727 | 0.1727 | 0.2813 |
| 8 | 0.6614 | 0.6614 | 0.3972 |

The comparison provided additional evidence for several selections.

**Functions 2 and 3:** The EI candidates were closer to their strongest existing observations than the neural-network alternatives. Combined with successful EI performance in the previous round and poor NN validation, this favoured EI.

**Function 4:** All three candidates were close to the current best, but the neural-network candidate was also close to several weaker recent observations. The GP candidates offered slightly different positions around the promising region.

**Function 5:** Both approaches targeted the successful region, but the NN candidate was only 0.0286 from the new best observation of 6691.93, compared with 0.1651 for the GP candidate. This provided strong support for local refinement using the neural network.

**Function 6:** All three candidates were located around the two strongest observations. EI was particularly promising because it predicted a substantial improvement, while the GP model had also become more informative.

**Function 7:** The identical UCB/EI candidate was closer to the strongest existing observations than the neural-network alternative.

**Function 8:** The neural-network candidate was closer to existing observations than the GP candidate, although its previous recommendation had been unsuccessful.

Function 1 remained difficult to assess because nearby outputs were effectively zero and neither surrogate model provided strong evidence of a promising region.

Overall, the neighbourhood analysis reinforced the importance of comparing model recommendations with actual observations rather than relying solely on predicted outputs.

## 8. Final Function-Specific Selections

The final selections considered all available evidence: GP predictions and uncertainty, UCB/EI agreement, neural-network validation, historical method performance and observed candidate neighbourhoods.

| Function | Selected Method | Reasoning | Confidence |
|---|---|---|---|
| 1 | UCB / EI agreement | More conservative candidate than unreliable NN alternative | Low |
| 2 | Expected Improvement | Successful previous query, strong GP agreement and proximity to new best | High |
| 3 | Expected Improvement | Promising predicted improvement, stronger local evidence and successful previous EI query | High |
| 4 | Expected Improvement | Predicted improvement while remaining near strongest observations | Medium |
| 5 | Neural Network | Strong validation and very close proximity to repeatedly successful region | High |
| 6 | Expected Improvement | Substantial improvement in GP fit and promising predicted output | Medium–High |
| 7 | UCB / EI agreement | Identical GP selections with stronger local support than NN | Medium–High |
| 8 | Neural Network | Strong validation, promising prediction and closer proximity to observations | Medium |

Function 5 was the most notable change, with the neural network preferred over the GP recommendations because it offered a small local refinement close to the latest high-performing observation.

Function 6 also changed strategy. Although the neural network had produced a small improvement in the previous round, the recovery of its Gaussian Process provided sufficient evidence to return to Expected Improvement.

Functions 2 and 3 continued to favour EI, while Function 7 retained the common GP acquisition candidate.

Function 8 remained a more uncertain decision because its previous neural-network query had failed to improve. Nevertheless, validation performance and candidate proximity supported retaining the NN recommendation.

## 9. Final Week 6 Query Selections

The final proposed points were validated to confirm that all coordinates were within the permitted `[0, 1]` range and that no selected point duplicated an existing observation.

| Function | Selection Method | Final Proposed Point | Distance to Nearest Observation |
|---|---|---|---:|
| 1 | UCB / EI agreement | `[0.773956, 0.438878]` | 0.180671 |
| 2 | Expected Improvement | `[0.729513, 0.111999]` | 0.030884 |
| 3 | Expected Improvement | `[0.308260, 0.310001, 0.479247]` | 0.140675 |
| 4 | Expected Improvement | `[0.329887, 0.354827, 0.355151, 0.453774]` | 0.104897 |
| 5 | Neural Network | `[0.939869, 0.992261, 0.978815, 0.999723]` | 0.028637 |
| 6 | Expected Improvement | `[0.397519, 0.393747, 0.471443, 0.711380, 0.165385]` | 0.213058 |
| 7 | UCB / EI agreement | `[0.174885, 0.324058, 0.479146, 0.269144, 0.331135, 0.614490]` | 0.172684 |
| 8 | Neural Network | `[0.042609, 0.051269, 0.314729, 0.127230, 0.965221, 0.369000, 0.150066, 0.389171]` | 0.397197 |

All eight selections passed the validation checks.

Function 5's candidate was particularly close to the current best observation, reflecting the decision to refine an established successful region.

Function 8 remained more exploratory, while the other selections generally balanced proximity to high-performing observations with the potential for further improvement.

### Submission Format

```text
Function 1: 0.773956-0.438878
Function 2: 0.729513-0.111999
Function 3: 0.308260-0.310001-0.479247
Function 4: 0.329887-0.354827-0.355151-0.453774
Function 5: 0.939869-0.992261-0.978815-0.999723
Function 6: 0.397519-0.393747-0.471443-0.711380-0.165385
Function 7: 0.174885-0.324058-0.479146-0.269144-0.331135-0.614490
Function 8: 0.042609-0.051269-0.314729-0.127230-0.965221-0.369000-0.150066-0.389171
```

## 10. Week 6 Reflection

Week 6 demonstrated the value of considering the historical performance of optimisation decisions alongside the latest model predictions.

Four of the eight Week 5 queries produced new best observations, with particularly strong improvements for Functions 2 and 5. These results provided additional support for Expected Improvement in certain functions, while the small improvement achieved by Function 6's neural-network candidate reinforced the potential value of alternative surrogates.

The most important modelling development was the recovery of Function 6's Gaussian Process. Its fitted length scale increased from the minimum bound to approximately 0.535, and its predictions began to meaningfully distinguish between candidate regions. This allowed EI to be reconsidered rather than automatically retaining the neural network.

Function 5 continued to present strong evidence for exploitation of its established high-performing region. The neural-network candidate was particularly close to the latest best observation and was selected as a local refinement.

The historical performance analysis also highlighted the limitations of relying solely on model agreement. Several previously selected candidates had failed to improve despite agreement between acquisition functions or attractive surrogate predictions.

By incorporating actual query outcomes, validation results and neighbouring observations, the selection process became more evidence-based without introducing an unnecessarily complex weighting system.

Overall, Week 6 retained the function-specific approach developed throughout the capstone while placing greater emphasis on learning from previous optimisation decisions.

The next round will assess whether these selections improve the observed maxima and whether historical model performance continues to provide useful guidance.

---

**Associated code:** `Week_06.ipynb`



