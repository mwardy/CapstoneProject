
# Week 4 — Comparing UCB Exploration Parameters and Expected Improvement

**Imperial College AI/ML Capstone Project | Module 15**

## Objective

The objective of Week 4 was to incorporate the latest observations, reassess the Gaussian Process (GP) models and investigate whether adjusting the exploration–exploitation balance could improve query selection.

Previous rounds had used a fixed Upper Confidence Bound (UCB) exploration parameter of κ = 2.5. This week, three values (1.5, 2.5 and 3.5) were compared to understand how different levels of exploration affected candidate selection.

Expected Improvement (EI) was also introduced as an alternative acquisition function, allowing the proposed queries to be assessed using a different approach to balancing predicted performance and uncertainty.

Functions 1 and 6 continued to require separate consideration because their RBF Gaussian Processes had repeatedly struggled to produce informative predictions.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 3 queries were added to the existing datasets and compared with the previous best observations.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | 0.000000 | 0.000000 | No improvement |
| 2 | 0.611205 | 0.548608 | 0.611205 | No improvement |
| 3 | -0.020167 | -0.115141 | -0.020167 | No improvement |
| 4 | -0.300645 | -1.850268 | -0.300645 | No improvement |
| 5 | 2765.819787 | 4595.479318 | 4595.479318 | New best |
| 6 | -0.458511 | -1.182842 | -0.458511 | No improvement |
| 7 | 1.950612 | 2.139551 | 2.139551 | New best |
| 8 | 9.821986 | 9.645301 | 9.821986 | No improvement |

Two of the eight queries produced new best observations.

**Function 5** showed the largest improvement, increasing from approximately 2765.82 to 4595.48. This was its third consecutive improvement and provided further evidence that the optimisation had identified a particularly promising region of its input space.

**Function 7** also improved, increasing from approximately 1.95 to 2.14.

The remaining functions did not improve their previous maxima. However, Function 2 returned an output relatively close to its existing best, suggesting that the search remained in a promising region.

Function 6's SVM-informed query did not improve on the previous best. This suggested that classification could help identify broadly higher-performing regions without necessarily identifying the highest-output points within those regions.

These results provided the basis for reassessing the query-selection strategies.

## 2. Refit and Diagnose the Gaussian Process Models

The RBF Gaussian Process models were refitted using all available observations.

Ten optimiser restarts were retained to improve the search for suitable kernel parameters and maintain consistency with previous rounds.

During the initial fitting, the optimiser restart parameter had accidentally been omitted. This caused Functions 1–7 to reach the minimum permitted RBF length scale. The models were therefore refitted using `n_restarts_optimizer=10` before interpreting their results.

### Fitted RBF Length Scales

| Function | Fitted Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0516 |
| 3 | 0.198 |
| 4 | 0.428 |
| 5 | 0.367 |
| 6 | 1e-05 |
| 7 | 0.296 |
| 8 | 0.949 |

Following the corrected fitting process, Functions 2, 3, 4, 5, 7 and 8 produced length scales above the lower bound.

These models therefore appeared suitable for further GP-based candidate evaluation.

However, **Functions 1 and 6** again reached the minimum length scale of `1e-05`, continuing the difficulties observed in previous rounds.

### 2.1 Diagnostics for Functions 1 and 6

Both models were evaluated across 100,000 candidate points to determine whether they could meaningfully distinguish between different areas of the search space.

| Metric | Function 1 | Function 6 |
|---|---:|---:|
| Minimum predicted mean | -0.0002773894 | -1.4874220585 |
| Maximum predicted mean | -0.0000000154 | -1.4874220585 |
| Predicted mean range | 0.0002773740 | 0.0000000000 |
| Minimum uncertainty | 0.0000101201 | 0.5341917888 |
| Maximum uncertainty | 0.0009609052 | 0.5341917888 |
| Uncertainty range | 0.0009507851 | 0.0000000000 |

Function 1 showed some variation in predictions and uncertainty, but its persistent lower-bound length scale continued to raise concerns about the reliability of its GP.

Function 6 produced effectively identical predictions and uncertainty across all candidate points. This meant that its GP could not meaningfully rank candidates using UCB.

Consequently, Functions 1 and 6 were excluded from the acquisition-function comparison and considered separately.

The remaining six functions proceeded to the UCB parameter analysis.

## 3. Compare Different UCB Exploration Parameters

Previous rounds had used a fixed exploration parameter of `κ = 2.5`.

This week, three values were compared:

- **κ = 1.5:** Greater emphasis on exploitation and predicted performance.
- **κ = 2.5:** The previous baseline, balancing predicted performance and uncertainty.
- **κ = 3.5:** Greater emphasis on exploration and uncertainty.

The UCB acquisition function remained:

UCB(x) = μ(x) + κ × σ(x)

For a fair comparison, the same 100,000 candidate points were evaluated under all three κ values for each function.

The resulting candidates were assessed using predicted output, uncertainty, distance from the current best observation and distance to the nearest existing observation.

### 3.1 Results of the κ Comparison

| Function | κ = 1.5 Behaviour | κ = 2.5 Behaviour | κ = 3.5 Behaviour |
|---|---|---|---|
| 2 | Close to current best; predicted 0.556 | Moves into a different region | Similar exploratory region to κ = 2.5 |
| 3 | Strongest predicted output (-0.034) | More exploratory | Greater emphasis on uncertainty |
| 4 | Close to best; predicted -0.309 | Similar region to κ = 1.5 | Distant point; predicted -18.362 |
| 5 | Near successful region; predicted 4561.79 | More uncertain point | Same point as κ = 2.5 |
| 7 | Selects a local promising candidate | Same point as κ = 1.5 | More exploratory alternative |
| 8 | Predicted 10.059, above current best | More exploratory; predicted 9.551 | Most exploratory; predicted 9.253 |

The results demonstrated that changing κ could materially affect candidate selection.

For **Function 5**, κ = 1.5 selected a candidate close to the high-performing region identified over previous rounds, with a predicted output of approximately 4561.79 compared with the current best of 4595.48. Higher κ values selected a more uncertain point with a lower predicted mean.

**Function 4** demonstrated the potential drawbacks of excessive exploration. At κ = 3.5, the model selected a substantially more distant candidate with a predicted output of approximately -18.36, compared with -0.309 at κ = 1.5.

**Function 8** also showed a clear trade-off. At κ = 1.5, the selected candidate had a predicted output of 10.059, exceeding the current observed best of 9.822. Increasing κ shifted selection towards more uncertain regions with lower predicted performance.

Function 7 was comparatively stable, with κ = 1.5 and 2.5 selecting the same candidate.

### 3.2 Compare Candidate Distances Across κ Values

To understand whether changing κ selected genuinely different regions, the Euclidean distances between the resulting candidate points were calculated.

| Function | Distance: 1.5 → 2.5 | Distance: 1.5 → 3.5 | Distance: 2.5 → 3.5 |
|---|---:|---:|---:|
| 2 | 0.8507 | 0.7421 | 0.1135 |
| 3 | 0.6195 | 0.6647 | 0.0707 |
| 4 | 0.0478 | 0.9597 | 0.9480 |
| 5 | 0.1438 | 0.1438 | 0.0000 |
| 7 | 0.0000 | 0.1986 | 0.1986 |
| 8 | 0.7627 | 0.9229 | 0.9939 |

The distance analysis reinforced the differences observed in predicted performance.

Functions 2 and 3 showed substantial changes between the lower and higher κ settings, indicating that their next query locations were sensitive to the exploration weighting.

Function 4 remained relatively stable between κ = 1.5 and 2.5 but changed dramatically at κ = 3.5.

Functions 5 and 7 were more stable across parts of the tested range.

**Function 8 was the most sensitive**, with all three κ values selecting substantially different regions.

Overall, this suggested that a single fixed exploration parameter might not be appropriate for every function at every stage of the optimisation.

## 4. Select Function-Specific Strategies

The κ comparison was assessed alongside previous query performance and GP reliability.

For Functions 2, 3, 4, 5, 7 and 8, **κ = 1.5 was provisionally preferred**, as it generally selected candidates with stronger predicted outputs while retaining some contribution from uncertainty.

The reasoning differed between functions:

- **Functions 2 and 3:** Recent queries had not improved the maxima, so greater emphasis on promising predicted regions was preferred over continued strong exploration.
- **Function 4:** Higher κ produced a distant candidate with particularly poor predicted performance, supporting a more exploitative approach.
- **Function 5:** Three consecutive improvements justified continuing to investigate the established high-performing region.
- **Function 7:** The latest query improved the maximum, and κ = 1.5 and 2.5 selected the same point.
- **Function 8:** κ = 1.5 identified a candidate predicted to exceed the current best, whereas higher κ values favoured distant, lower-mean points.

Functions 1 and 6 remained unsuitable for standard UCB selection because of their GP modelling difficulties.

Before finalising the queries, Expected Improvement was introduced as an alternative acquisition function to test whether it supported the provisional selections.

## 5. Compare UCB with Expected Improvement

Expected Improvement (EI) was evaluated alongside UCB for Functions 2, 3, 4, 5, 7 and 8.

Unlike UCB, which combines predicted performance and uncertainty through κ, EI evaluates the expected positive improvement over the current best observed output.

This offered a useful comparison because it considered the potential to exceed the current maximum directly rather than relying on a fixed uncertainty weighting.

### 5.1 Expected Improvement Results

| Function | EI Predicted Output | Distance from κ = 1.5 UCB Candidate | Interpretation |
|---|---:|---:|---|
| 2 | 0.590 | 0.0085 | Very similar region; slightly stronger prediction |
| 3 | -0.031 | 0.0228 | Very similar region; slightly stronger prediction |
| 4 | -0.309 | 0.0000 | Identical candidate |
| 5 | 4561.794 | 0.0000 | Identical candidate |
| 7 | 2.054 | 0.1893 | Different candidate, closer to current best |
| 8 | 10.080 | 0.2621 | Different candidate, stronger predicted output |

The EI comparison broadly supported the more exploitative strategy identified through the κ analysis.

For **Functions 2 and 3**, EI selected candidates very close to those identified using κ = 1.5, but with slightly stronger predicted outputs.

For **Functions 4 and 5**, EI and UCB selected exactly the same candidate. This provided additional support for the regions identified by both acquisition functions.

Function 5's result was particularly encouraging because the agreement occurred within an area that had produced three consecutive improvements.

Functions 7 and 8 showed greater differences between EI and UCB.

For Function 7, EI selected a candidate closer to the current best region with a stronger predicted output, favouring further exploitation following the latest improvement.

For Function 8, both methods predicted potential improvements over the current observed maximum. However, EI selected a candidate with a slightly higher predicted output of approximately 10.080 and lower uncertainty.

### 5.2 Alternative Strategies for Functions 1 and 6

Functions 1 and 6 were treated separately because their GP models remained unsuitable for reliable acquisition-based selection.

Function 1 retained maximum-distance exploration, which had been used in previous rounds.

For Function 6, the previous SVM-informed query had failed to improve the observed maximum, while its Gaussian Process continued to produce effectively flat predictions.

Maximum-distance exploration was therefore selected as a model-independent alternative.

For both functions, 100,000 candidate points were generated and the point with the greatest distance to its nearest existing observation was selected.

| Function | Selected Exploration Point | Distance to Nearest Observation | Distance from Current Best |
|---|---|---:|---:|
| 1 | `[0.417430, 0.463246]` | 0.3153 | 0.4137 |
| 6 | `[0.033216, 0.058439, 0.124012, 0.011504, 0.042834]` | 0.8782 | 0.8810 |

Function 1's selection provided further exploration of a relatively poorly sampled region.

Function 6's candidate was substantially more exploratory, reflecting its higher-dimensional input space and the continuing limitations of its model-based selection methods.

Although this increased the uncertainty surrounding its immediate performance, it provided an opportunity to gather information from a genuinely unexplored region.

## 6. Final Week 4 Query Selections

The final queries were selected using the accumulated evidence from recent results, GP diagnostics, the κ comparison and Expected Improvement.

EI was preferred for Functions 2, 3, 7 and 8 because it identified candidates with stronger predicted outputs than their κ = 1.5 UCB alternatives.

Functions 4 and 5 retained candidates selected identically by both EI and UCB.

Functions 1 and 6 used maximum-distance exploration because their Gaussian Processes remained unreliable.

### Final Proposed Queries

| Function | Selection Method | Final Proposed Point | Distance to Nearest Observation |
|---|---|---|---:|
| 1 | Maximum-distance exploration | `[0.417430, 0.463246]` | 0.315336 |
| 2 | Expected Improvement | `[0.730707, 0.915532]` | 0.030161 |
| 3 | Expected Improvement | `[0.433866, 0.716516, 0.000863]` | 0.178778 |
| 4 | EI / UCB agreement | `[0.443740, 0.493644, 0.409790, 0.452219]` | 0.091845 |
| 5 | EI / UCB agreement | `[0.851064, 0.985689, 0.919598, 0.979330]` | 0.193062 |
| 6 | Maximum-distance exploration | `[0.033216, 0.058439, 0.124012, 0.011504, 0.042834]` | 0.878156 |
| 7 | Expected Improvement | `[0.154752, 0.313081, 0.341855, 0.169387, 0.318877, 0.633022]` | 0.149214 |
| 8 | Expected Improvement | `[0.148540, 0.297539, 0.201860, 0.328776, 0.636564, 0.500531, 0.197949, 0.479426]` | 0.518070 |

The final validation confirmed that all eight selections were within the permitted `[0, 1]` input range and did not duplicate any existing observations.

Functions 2, 4, 5 and 7 generally remained close to their existing high-performing regions, consistent with the greater emphasis on exploitation introduced this week.

Functions 3 and 8 retained more exploratory elements despite being selected through EI.

Functions 1 and 6 were deliberately exploratory, with Function 6 representing the highest-risk selection because of its substantial distance from previously observed points.

### Submission Format

```text
Function 1: 0.417430-0.463246
Function 2: 0.730707-0.915532
Function 3: 0.433866-0.716516-0.000863
Function 4: 0.443740-0.493644-0.409790-0.452219
Function 5: 0.851064-0.985689-0.919598-0.979330
Function 6: 0.033216-0.058439-0.124012-0.011504-0.042834
Function 7: 0.154752-0.313081-0.341855-0.169387-0.318877-0.633022
Function 8: 0.148540-0.297539-0.201860-0.328776-0.636564-0.500531-0.197949-0.479426
```

## 7. Week 4 Reflection

Week 4 demonstrated that the exploration–exploitation balance could be refined by comparing multiple acquisition settings rather than relying on a fixed UCB parameter.

The latest observations strengthened the evidence for continued exploitation in Function 5, which achieved its third consecutive improvement, and Function 7, which also produced a new best output.

Comparing κ values of 1.5, 2.5 and 3.5 showed that higher exploration weightings could substantially change candidate selection. This was particularly apparent for Functions 2, 3, 4 and 8, where higher κ values often selected more distant points with lower predicted outputs.

Expected Improvement provided an additional perspective. Its selections closely agreed with lower-κ UCB for Functions 2 and 3, while both methods selected identical candidates for Functions 4 and 5. This supported a more exploitative approach for functions with informative GP models.

However, Functions 1 and 6 continued to highlight the limitations of relying on Gaussian Process predictions when fitted kernel parameters indicated poor generalisation. Maximum-distance exploration was therefore retained to gather additional information without depending on unreliable acquisition scores.

Overall, Week 4 introduced greater flexibility into the optimisation process by combining UCB parameter comparisons, Expected Improvement and model-independent exploration.

The next round will assess whether these more targeted selections improve the observed maxima and whether the additional observations help resolve the continuing modelling difficulties.

---

**Associated code:** `Week_04.ipynb`


