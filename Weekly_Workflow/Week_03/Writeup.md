
# Module 14 — Week 3: Gaussian process optimisation and SVM analysis

## Weekly objective and step-by-step plan

The objective of Week 3 was to incorporate the latest query results, reassess the Gaussian process (GP) models and select a third query point for each of the eight functions.

This week also introduced support vector machines (SVMs) as an additional way of investigating the functions. I wanted to assess whether SVM classification could identify higher-performing regions and provide useful guidance where the GP models were struggling.

Particular attention was given to Functions 1 and 6, which had previously experienced modelling difficulties.

The process followed seven steps:

1. Incorporate the new observations and compare them with the previous best outputs.
2. Refit the RBF Gaussian processes and examine their predictions and uncertainty.
3. Introduce SVM classification and compare linear and RBF kernels.
4. Reassess the exploration–exploitation balance for each function.
5. Apply GP + UCB to functions with usable GP models.
6. Use alternative selection methods for Functions 1 and 6.
7. Validate and format the eight final queries.

## Methodology

The latest observations were incorporated into the cumulative datasets, and the Gaussian processes were refitted using RBF kernels, normalised outputs, ten optimiser restarts and a random state of 42.

As in previous weeks, I used upper confidence bound (UCB) as the main acquisition function:

**UCB(x) = μ(x) + κσ(x)**

Here, μ(x) is the predicted output, σ(x) is predictive uncertainty and κ controls the relative importance of exploration.

I retained κ = 2.5 and generated 100,000 candidate points for each function where GP + UCB was used.

I also introduced SVM classification. Since the functions produce continuous outputs, I divided each function's observations into high- and low-performing classes using its median observed output. I then compared linear and RBF-kernel SVMs using three-fold cross-validation.

The SVM analysis was intended to supplement the existing optimisation approach rather than replace it automatically. Unlike a GP, an SVM classifier does not directly predict the continuous function output or provide the predictive uncertainty required by UCB.

## Results and interpretation by step

### Step 1 — Incorporate new observations

I compared the returned outputs from Week 2's proposed queries with the previous best observations.

| Function | Previous best | New output | Updated best | Result |
|---|---:|---:|---:|---|
| 1 | ≈0 | ≈0 | ≈0 | No improvement |
| 2 | 0.611205 | 0.366010 | 0.611205 | No improvement |
| 3 | -0.020167 | -0.060870 | -0.020167 | No improvement |
| 4 | -0.300645 | -3.743172 | -0.300645 | No improvement |
| 5 | 2329.015854 | 2765.819787 | 2765.819787 | New best |
| 6 | -0.714265 | -0.458511 | -0.458511 | New best |
| 7 | 1.950612 | 0.975233 | 1.950612 | No improvement |
| 8 | 9.821986 | 9.621122 | 9.821986 | No improvement |

Two of the eight queries produced new best outputs.

Function 5 improved from approximately 2329.02 to 2765.82, an increase of 18.8%. This continued the positive trend from previous rounds and suggested that the GP + UCB approach was identifying a promising region.

Function 6 also improved, from -0.7143 to -0.4585. This was particularly encouraging because Function 6 had required maximum-distance exploration in Week 1 before returning to GP + UCB in Week 2.

Although the remaining functions did not improve, their new observations still provided additional information for updating the models.

### Step 2 — Refit and evaluate the Gaussian processes

I refitted the RBF Gaussian processes using the expanded datasets and examined their fitted length scales.

| Function | Week 2 length scale | Week 3 length scale |
|---|---:|---:|
| 1 | 1e-05 | 1e-05 |
| 2 | 0.0453 | 0.0427 |
| 3 | 0.202 | 0.202 |
| 4 | 0.522 | 0.495 |
| 5 | 0.225 | 0.413 |
| 6 | 0.481 | 1e-05 |
| 7 | 0.354 | 0.244 |
| 8 | 0.913 | 0.931 |

Most functions continued to produce usable fitted models. Function 5's length scale increased from approximately 0.225 to 0.413 following another successful observation.

However, Functions 1 and 6 both reached the lower RBF length-scale bound of 1e-05 and generated convergence warnings.

Function 1 had experienced this issue in previous weeks, but Function 6's return to the lower bound was unexpected after its improvement in Week 2.

To investigate further, I examined the predicted means and uncertainty across 100,000 candidate points for both functions.

| Diagnostic | Function 1 | Function 6 |
|---|---:|---:|
| Predicted mean range | 0.0003003282 | 0.0000000000 |
| Uncertainty range | 0.0009624562 | 0.0000000000 |

The results showed different behaviour despite both models reaching the same length-scale bound.

Function 1 produced some variation in its predictions and uncertainty, suggesting that the GP could distinguish between at least some regions. However, the recurring convergence warning meant I remained cautious about relying on its predictions.

Function 6 produced identical predicted means and uncertainty across all candidates. Consequently, UCB would assign effectively identical scores to unexplored points, providing no meaningful basis for choosing between them.

This demonstrated that a new best observation does not necessarily mean the refitted model has improved. Although Function 6's latest query was successful, its updated GP was no longer useful for guiding the next selection.

### Step 3 — Introduce and compare SVM models

I introduced SVMs to investigate whether the existing observations could distinguish between relatively high- and low-performing regions.

For each function, I calculated the median observed output and classified observations above the median as high-performing and those at or below it as low-performing.

This produced approximately balanced classes, which was useful given the relatively small datasets.

I then compared linear and RBF-kernel SVMs using three-fold cross-validation.

| Function | Linear SVM accuracy | RBF SVM accuracy |
|---|---:|---:|
| 1 | 0.167 | 0.500 |
| 2 | 0.667 | 0.833 |
| 3 | 0.344 | 0.522 |
| 4 | 0.627 | 0.782 |
| 5 | 0.649 | 0.458 |
| 6 | 0.821 | 0.768 |
| 7 | 0.785 | 0.815 |
| 8 | 0.857 | 0.833 |

The RBF SVM achieved higher accuracy for Functions 1, 2, 3, 4 and 7, while the linear SVM performed better for Functions 5, 6 and 8.

Functions 2 and 4 showed relatively strong improvements with the RBF kernel, suggesting that nonlinear decision boundaries may be useful for separating their higher- and lower-performing observations.

For Function 6, the linear SVM achieved an accuracy of 0.821, compared with 0.768 for the RBF model. This suggested that its observed outputs contained some structure that a linear classifier could identify, despite the difficulties with its Gaussian process.

Function 1 remained problematic. Although its RBF SVM outperformed the linear alternative, an accuracy of 0.500 was not convincing evidence that the model could reliably identify higher-performing regions.

These results supported considering the SVMs individually rather than assuming that one kernel or selection strategy would be suitable for every function.

The cross-validation results were also treated cautiously because the datasets were small and the median-based classes represented relative observed performance rather than known optimal regions.

### Step 4 — Reassess exploration and exploitation

I used the latest outputs, GP diagnostics and SVM results to reassess the query strategy for each function.

The aim was to exploit promising regions where the models provided useful guidance while continuing to explore where predictions were unreliable or uncertainty suggested that other areas could be valuable.

The main decisions were:

- Function 1: Continue maximum-distance exploration because the GP remained questionable and neither SVM provided convincing guidance.
- Functions 2 and 3: Retain GP + UCB, as the models remained usable without strong evidence for changing strategy.
- Function 4: Retain GP + UCB, with the previously identified promising region remaining relevant despite the latest query not improving the maximum.
- Function 5: Continue searching around the high-performing region responsible for successive improvements.
- Function 6: Investigate linear SVM-informed selection because its GP could no longer meaningfully rank candidates.
- Functions 7 and 8: Retain GP + UCB to balance predicted performance and uncertainty.

For the six functions continuing with GP + UCB, I retained κ = 2.5.

### Step 5 — Apply GP + UCB

I generated 100,000 candidate points for Functions 2, 3, 4, 5, 7 and 8 and selected the candidates with the highest UCB scores.

| Function | Predicted output | Uncertainty | UCB score | Distance from current best |
|---|---:|---:|---:|---:|
| 2 | 0.449335 | 0.185666 | 0.913500 | 0.0452 |
| 3 | -0.076907 | 0.077556 | 0.116983 | 0.8461 |
| 4 | -0.064270 | 0.420976 | 0.988170 | 0.1376 |
| 5 | 2631.682259 | 240.331386 | 3232.510725 | 0.2030 |
| 7 | 1.662117 | 0.258204 | 2.307626 | 0.1612 |
| 8 | 9.635913 | 0.434286 | 10.721628 | 0.8392 |

The selected points showed different balances of exploration and exploitation.

Functions 2, 4 and 7 remained relatively close to their current best observations, suggesting greater exploitation of previously successful regions.

Function 4 was particularly interesting because its predicted output of -0.0643 was better than the current observed maximum of -0.3006. Although uncertainty remained substantial, the candidate appeared promising.

Function 5 also continued to search around its high-performing region. The selected point retained high values for its second, third and fourth input dimensions, consistent with the area responsible for the previous improvements.

In contrast, Functions 3 and 8 selected points much further from their current best observations, with distances of approximately 0.846 and 0.839 respectively. This suggested that uncertainty was encouraging greater exploration.

These results supported retaining κ = 2.5. The same acquisition parameter was producing different exploration–exploitation behaviour depending on each function's fitted predictions and uncertainty.

### Step 6 — Apply alternative strategies to Functions 1 and 6

#### Function 1 — Maximum-distance exploration

Function 1 continued to reach the lower RBF length-scale bound, and its SVM results did not provide convincing evidence of a reliable high-performing region.

I therefore continued using maximum-distance exploration.

For each of 100,000 candidate points, I calculated its distance to the nearest existing observation and selected the point with the greatest minimum distance.

| Measure | Result |
|---|---|
| Selected point | `[0.000045, 0.003062]` |
| Distance to nearest observation | 0.321671 |

This deliberately prioritised a relatively unexplored part of the input space rather than relying on a GP model whose behaviour remained questionable.

Although this approach did not guarantee a better output, it could provide useful information for subsequent model fitting.

#### Function 6 — Linear SVM-informed selection

Function 6 required a different approach because its GP produced effectively constant predictions and uncertainty across the candidate space.

The linear SVM had achieved a cross-validation accuracy of 0.821, so I fitted it to all available Function 6 observations and used its decision function to score 100,000 candidate points.

I selected the candidate with the highest decision score for the higher-performing class.

| Measure | Result |
|---|---|
| Selected point | `[0.013758, 0.147930, 0.445597, 0.994845, 0.013469]` |
| SVM decision score | 5.272052 |
| Distance from current best | 0.402781 |

The selected candidate had a strong positive decision score and was approximately 0.403 from the current best observation.

This suggested that the point was strongly classified within the higher-performing region while still exploring a location some distance from the best-known input.

Importantly, the SVM decision score is not a predicted function output. It measures the candidate's position relative to the learned classification boundary, so the selected point was considered a reasonable alternative rather than a guaranteed improvement.

### Step 7 — Validate the final proposed queries

Before submission, I checked that all eight selected points were within the permitted [0, 1] input range and did not duplicate existing observations.

I also calculated their distances to the nearest existing observations as a final check on the selected strategies.

| Function | Method | Nearest observation distance |
|---|---|---:|
| 1 | Max-distance exploration | 0.321671 |
| 2 | GP + UCB | 0.045156 |
| 3 | GP + UCB | 0.306635 |
| 4 | GP + UCB | 0.137608 |
| 5 | GP + UCB | 0.203041 |
| 6 | Linear SVM | 0.402781 |
| 7 | GP + UCB | 0.161151 |
| 8 | GP + UCB | 0.800486 |

All eight proposed queries passed the validation checks.

The final queries, formatted to six decimal places, were:

| Function | Selection method | Proposed query coordinates |
|---|---|---|
| 1 | Max-distance exploration | `[0.000045, 0.003062]` |
| 2 | GP + UCB | `[0.709627, 0.881953]` |
| 3 | GP + UCB | `[0.732035, 0.997300, 0.015653]` |
| 4 | GP + UCB | `[0.391976, 0.424757, 0.494823, 0.464424]` |
| 5 | GP + UCB | `[0.671629, 0.981118, 0.990458, 0.985189]` |
| 6 | Linear SVM | `[0.013758, 0.147930, 0.445597, 0.994845, 0.013469]` |
| 7 | GP + UCB | `[0.021549, 0.313334, 0.360216, 0.141732, 0.295328, 0.686548]` |
| 8 | GP + UCB | `[0.025529, 0.451238, 0.098209, 0.031391, 0.412421, 0.138573, 0.114934, 0.994361]` |

## Weekly summary and next steps

Week 3 produced new best observations for Functions 5 and 6. Function 5 continued its improvement through GP + UCB, while Function 6 demonstrated that a successful query could still be followed by difficulties when refitting the Gaussian process.

The main methodological development was introducing SVMs to investigate high- and low-performing regions. The results varied across functions, with some favouring nonlinear classification and others performing better with linear models.

For Function 6, the relatively strong linear SVM results provided an alternative way of selecting a query when the GP was unable to distinguish between candidate points. Function 1 continued with maximum-distance exploration because its modelling difficulties remained unresolved.

The next steps were to:

- Submit the eight proposed query points and assess their returned outputs.
- Incorporate the new observations and refit the GP models.
- Reassess whether the SVM-informed selection improved Function 6.
- Investigate whether alternative acquisition functions or adjustments to κ could improve the exploration–exploitation balance in subsequent rounds.

Overall, Week 3 showed the importance of assessing model behaviour rather than relying only on whether the latest query improved the observed maximum. GP + UCB remained the main optimisation approach, while SVM classification and maximum-distance exploration provided alternatives when the Gaussian processes were less informative.


