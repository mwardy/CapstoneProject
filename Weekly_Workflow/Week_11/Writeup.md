
# Week 11 — Introducing K-Means Clustering for Candidate Selection

**Imperial College AI/ML Capstone Project | Module 22**

## Objective

The objective of Week 11 was to investigate whether K-Means clustering could provide additional information about the structure of the eight unknown functions and help identify promising regions for future queries.

Previous rounds had increasingly relied on function-specific optimisation strategies combining Gaussian Process (GP) regression, neural-network (NN) predictions, acquisition functions, historical results and local refinement.

Although these approaches had produced several improvements, some functions remained difficult to model, while others appeared to be concentrating around established high-performing regions.

K-Means clustering was therefore introduced as an additional analytical method rather than a replacement for the existing optimisation framework.

The analysis examined whether observations formed meaningful clusters in their input spaces, whether some clusters consistently produced stronger outputs, and whether this information could improve candidate generation.

The final selections would continue to consider model reliability, predicted performance, uncertainty and novelty alongside the clustering evidence.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 10 queries were compared with the previous highest observed outputs and incorporated into the existing datasets.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | ~0.000000 | 0.000000 | No improvement |
| 2 | 0.746962 | 0.666153 | 0.746962 | No improvement |
| 3 | -0.000863 | -0.010582 | -0.000863 | No improvement |
| 4 | 0.583730 | 0.622950 | 0.622950 | New best |
| 5 | 8662.482500 | 8115.177655 | 8662.482500 | No improvement |
| 6 | -0.239769 | -0.283228 | -0.239769 | No improvement |
| 7 | 2.929370 | 3.024610 | 3.024610 | New best |
| 8 | 9.984981 | Below previous best | 9.984981 | No improvement |

Two of the eight queries produced new best observations.

**Function 4** improved from approximately 0.5837 to 0.6229, continuing its successful sequence of local refinements.

**Function 7** increased from approximately 2.9294 to 3.0246, providing evidence that further improvements remained possible within its established high-performing region.

Function 2 returned 0.6662, below its current best but reasonably close given its continuing GP difficulties.

Functions 3 and 6 failed to improve, despite their previous selections being supported by local model predictions.

**Function 5** returned approximately 8115.18, reinforcing the pattern that movements away from its best-known upper-boundary point tended to reduce performance.

Function 1 remained effectively unchanged, while Function 8 failed to improve despite its previous strong optimisation results.

These outcomes reinforced the need to assess candidate regions using multiple sources of evidence rather than relying exclusively on predicted outputs.

## 2. Evaluate Realised Prediction Accuracy

Building on Week 10, previously recorded GP and NN predictions were compared with the outputs actually returned.

The absolute prediction error was calculated for each available model prediction:

Absolute Error = |Predicted Output - Actual Output|

This provided an out-of-sample assessment of model reliability, complementing cross-validation and training diagnostics.

### 2.1 Comparison of Prediction Errors

| Function | GP Absolute Error | NN Absolute Error | More Accurate Model |
|---|---:|---:|---|
| 1 | — | — | Not applicable |
| 2 | — | 0.0433 | NN only |
| 3 | 0.0221 | 0.0099 | NN |
| 4 | 0.0270 | 0.0889 | GP |
| 5 | 41.22 | 28.87 | NN |
| 6 | 0.0741 | 0.0526 | NN |
| 7 | 0.0440 | 0.0707 | GP |
| 8 | 0.0235 | 0.1246 | GP |

Function 1 was excluded because its previous point was selected through model-independent exploration.

Function 2 had only an NN prediction available for its submitted candidate.

### 2.2 Interpretation

**Function 4** continued to demonstrate strong GP accuracy. The GP predicted approximately 0.6499 compared with the actual output of 0.6229, producing an absolute error of only 0.0270.

**Function 7** also showed encouraging GP performance. Its prediction of approximately 2.9806 was relatively close to the new best of 3.0246.

**Function 8** showed the clearest GP advantage, with an absolute error of 0.0235 compared with 0.1246 for the NN.

These results supported giving greater weight to GP predictions for Functions 4, 7 and 8.

The NN was more accurate for Functions 3, 5 and 6.

Function 3 was particularly interesting because its GP predicted a positive result, while the actual output remained negative. The NN better represented the behaviour of the local region.

For Function 5, both models predicted the result reasonably well relative to its much larger output scale, supporting their representation of the upper-boundary region.

Overall, neither surrogate was consistently more accurate across all functions. Historical realised accuracy therefore continued to support a function-specific approach to model weighting.

## 3. Refit and Diagnose the Surrogate Models

The GP and NN models were refitted using the expanded datasets.

The RBF Gaussian Processes were evaluated using their optimised length scales, log marginal likelihoods and convergence warnings.

### 3.1 Gaussian Process Diagnostics

| Function | Previous Length Scale | Updated Length Scale | Assessment |
|---|---:|---:|---|
| 1 | 1e-05 | 1e-05 | Remains problematic |
| 2 | 1e-05 | 1e-05 | Remains problematic |
| 3 | 0.0968 | 0.0642 | Greater local variability |
| 4 | 0.316 | 0.272 | Informative |
| 5 | 0.338 | 0.359 | Stable |
| 6 | 0.518 | 0.470 | Informative |
| 7 | 0.320 | 0.338 | Stable |
| 8 | 1.07 | 1.08 | Stable |

**Functions 1 and 2** again reached the minimum RBF length-scale bound of `1e-05`, indicating that their GP fitting difficulties remained unresolved.

Function 3's length scale decreased to approximately 0.0642. This suggested that the model was identifying more localised variation in the function's output.

Functions 4–8 remained above the minimum bound, supporting their continued use for acquisition-based and local-refinement decisions.

Function 8 retained the largest length scale, suggesting a comparatively smooth relationship from the GP's perspective.

### 3.2 Neural-Network Validation

Leave-One-Out Cross-Validation (LOOCV) was repeated using the updated observations.

NN Mean Absolute Error (MAE) was compared with a baseline that predicted the mean of the remaining training outputs.

| Function | NN LOOCV MAE | Mean Baseline MAE | Beats Baseline |
|---|---:|---:|---|
| 1 | 0.0278 | 0.0009 | No |
| 2 | 0.2369 | 0.2392 | Yes, narrowly |
| 3 | 0.0460 | Approximately 0.055 | Yes |
| 4 | 1.4783 | 7.7991 | Yes |
| 5 | 160.03 | 2752.03 | Yes |
| 6 | 0.2398 | Above NN MAE | Yes |
| 7 | 0.4733 | Above NN MAE | Yes |
| 8 | Better than baseline | — | Yes |

The most important change occurred in **Function 3**, whose NN MAE decreased from approximately 0.0769 to 0.0460.

This allowed the NN to outperform its mean baseline, providing stronger evidence for incorporating its predictions into candidate selection.

Function 2 continued to outperform its baseline, although only marginally.

Functions 4 and 5 retained strong validation performance, while Functions 6–8 continued to show useful predictive structure.

Function 1 remained the principal exception. Although its NN error improved, it was still substantially worse than the baseline, supporting continued model-independent exploration.

These results suggested that the NN could contribute useful evidence for Functions 2–8, with differing levels of confidence.

## 4. Analyse Recent Optimisation Progress

The running-best histories and recent improvement patterns were examined before introducing clustering.

This provided context for determining whether each function should continue exploiting established promising regions or investigate new areas.

**Functions 4 and 7** had just produced new best observations, supporting further local refinement.

**Function 5** continued to demonstrate exceptionally strong performance near the upper boundary, although recent deviations had failed to improve its maximum.

**Function 8** remained close to an observed maximum of 10, with recent gains becoming smaller.

**Function 6** had improved substantially over earlier rounds but failed to improve in the latest iteration.

Functions 2 and 3 had shown mixed recent performance, while Function 1 continued to exhibit little meaningful progress.

The analysis therefore suggested that clustering might be most useful for identifying previously unexplored promising regions in some functions, while providing additional confirmation of established high-performing regions in others.

## 5. Introduce K-Means Clustering

K-Means clustering was introduced to investigate whether the existing observations formed distinct groups in their input spaces.

Unlike GP regression or neural-network models, K-Means does not directly predict the function's output.

Instead, it groups observations according to the similarity of their input coordinates.

This allowed the analysis to investigate whether particular areas of each search space contained consistently stronger outputs.

### 5.1 Selecting the Number of Clusters

Different values of `k` were evaluated using the silhouette score.

The silhouette score measures how closely observations are grouped within their assigned clusters relative to other clusters.

Higher values generally indicate clearer separation, although a high score does not necessarily mean that the clusters are useful for optimisation.

### 5.2 Clustering Results

| Function | Selected Clusters | Silhouette Score | Interpretation |
|---|---:|---:|---|
| 1 | 4 | 0.4066 | Moderate separation |
| 2 | 3 | 0.6462 | Strongest separation |
| 3 | Selected through silhouette analysis | ~0.30 | Relatively weak |
| 4 | Selected through silhouette analysis | ~0.30 | Relatively weak |
| 5 | 2 | 0.4331 | Reasonable separation |
| 6 | Selected through silhouette analysis | 0.2572 | Weak |
| 7 | Selected through silhouette analysis | ~0.30 | Relatively weak |
| 8 | 4 | 0.1853 | Weakest separation |

**Function 2** demonstrated the clearest clustering structure, achieving a silhouette score of approximately 0.6462 with three clusters.

**Function 5** also showed relatively meaningful separation, with a silhouette score of 0.4331 using two clusters.

Function 1 produced moderate separation, but this did not necessarily indicate useful differences in output performance.

Functions 3, 4 and 7 showed relatively weak separation, while Function 6's score was only 0.2572.

**Function 8 showed the weakest clustering structure**, with a silhouette score of 0.1853.

This suggested that clustering should not influence all eight functions equally.

In particular, the low score for Function 8 indicated that forcing its observations into distinct groups might provide limited useful information.

## 6. Assess Cluster Performance

After selecting the clustering configurations, the outputs associated with each cluster were compared.

For each cluster, the analysis considered:

- Number of observations.
- Mean output.
- Highest observed output.
- Cluster centroid.
- Whether the current best observation belonged to that cluster.

The aim was to determine whether the strongest individual observations were supported by broader regions containing other high-performing results.

### 6.1 Function-Specific Findings

**Function 1**

The best observation belonged to a cluster with a mean output of approximately -0.000515.

Although the silhouette score indicated moderate separation, the clusters did not identify a clearly superior region because several observations produced outputs effectively equal to zero.

Clustering therefore added little useful guidance.

**Function 2**

The strong silhouette score suggested that its observations formed relatively distinct input-space regions.

The differences in output performance between clusters provided additional evidence for focusing on the strongest-performing region.

This was particularly useful because its GP remained unreliable.

**Function 3**

The clusters provided some supporting evidence for the region containing the current best observations, although their relatively weak separation limited confidence in using K-Means independently.

**Function 4**

The cluster containing the strongest observations performed better than the alternatives, supporting continued exploitation of its recently successful region.

**Function 5**

The clustering results reinforced the importance of the upper-boundary region.

This agreed with the previous optimisation history and the predictions of both surrogate models.

**Function 6**

The clustering results provided additional support for the region identified through earlier GP and NN local refinement.

However, the weak silhouette score meant this evidence was supplementary.

**Function 7**

The cluster containing the current best observation showed stronger performance than competing regions, supporting continued local refinement.

**Function 8**

The strongest cluster had a mean output of approximately 9.6695 and contained the best observation of 9.984981.

However, the silhouette score of 0.1853 meant that this clustering structure should receive relatively little weight in candidate selection.

### 6.2 Overall Assessment

The strongest clustering evidence came from Functions 2 and 5.

Functions 3, 4, 6 and 7 received useful supporting information, but their relatively weak cluster separation meant K-Means should not replace the existing surrogate models.

Functions 1 and 8 gained relatively little additional guidance.

The cluster centroids were not treated as optimal points themselves. Instead, the clusters were used to identify areas in which further candidate generation could be investigated.

## 7. Generate Cluster-Guided Candidates

The strongest-performing clusters were used to guide additional candidate generation.

Potential new points within promising regions were evaluated using GP predictions, NN predictions and their distance from existing observations.

This allowed clustering to influence the search without directly determining the final submissions.

### 7.1 Cluster-Guided Results

| Function | GP Prediction | NN Prediction | Nearest Distance |
|---|---:|---:|---:|
| 2 | 0.365797 | 0.661354 | 0.027213 |
| 3 | 0.037513 | -0.012296 | 0.039857 |
| 4 | 0.722666 | 0.412112 | 0.021532 |
| 5 | 8321.188371 | 8326.644042 | 0.011843 |
| 6 | -0.228475 | -0.216792 | 0.050226 |
| 7 | 3.087322 | 3.020521 | 0.110278 |
| 8 | 9.933104 | 10.252357 | 0.196623 |

Several candidates appeared promising.

**Function 6** showed close agreement between GP and NN predictions, with both models predicting an improvement over the current best of -0.239769.

**Function 7** also produced encouraging results, with the GP predicting an improvement above the current best of 3.024610 and the NN producing a similar estimate.

Function 4's GP predicted approximately 0.7227 against the current best of 0.6230, although the NN was considerably less optimistic.

Function 3 showed similar disagreement between the models.

**Function 5** produced strong model agreement, but its candidate was extremely close to an existing observation and both models predicted performance below the current best.

Function 8 provided greater novelty and an optimistic NN prediction, but its weak clustering structure reduced confidence in the recommendation.

Overall, cluster-guided candidates were retained as alternatives rather than accepted automatically.

## 8. Function-Specific Candidate Refinement

Additional candidates were generated using function-specific strategies developed in previous rounds.

The purpose was to compare cluster-guided recommendations against alternatives incorporating surrogate reliability, realised prediction accuracy, local refinement and novelty.

### 8.1 Refinement Strategies

| Function | Strategy | Reasoning |
|---|---|---|
| 1 | Space-filling exploration | Neither surrogate reliable |
| 2 | NN-supported cluster refinement | GP remains unreliable; clustering relatively strong |
| 3 | Local GP/NN refinement | Recent NN validation improvement |
| 4 | GP-weighted refinement | GP more accurate on latest realised output |
| 5 | Upper-boundary refinement | Strong historical evidence near `[1,1,1,1]` |
| 6 | Local GP/NN refinement | Both models identify promising region |
| 7 | Local GP/NN refinement | Latest new best and model agreement |
| 8 | GP-weighted refinement | GP more accurate; clustering relatively weak |

The refined alternatives were compared against the cluster-guided candidates before selecting the final submission points.

## 9. Compare Final Candidate Alternatives

### 9.1 Function 1 — Space-Filling Exploration

Function 1 continued to have no reliable surrogate signal.

The selected candidate was:

`[0.249506, 0.999107]`

Its distance to the nearest existing observation was approximately 0.246275.

This represented a deliberately exploratory query rather than one selected for predicted performance.

### 9.2 Function 2 — Cluster-Guided Selection

Function 2's GP continued to reach the minimum length-scale bound, while its NN provided more useful guidance.

The cluster-guided candidate was:

`[0.729306, 0.186865]`

Its NN predicted output was approximately 0.661354.

The candidate remained within a relatively high-performing cluster and was approximately 0.027213 from its nearest observation.

The clustering evidence and NN support justified retaining this candidate, although it was not predicted to exceed the current best.

### 9.3 Function 3 — Local Refinement

Function 3's NN validation improved this week, allowing its predictions to receive more consideration.

The refined candidate was:

`[0.428706, 0.513778, 0.457382]`

The GP predicted approximately 0.037818, while the NN predicted -0.012456.

The substantial disagreement between models indicated uncertainty over the expected output, but the candidate remained in a locally promising region.

Its nearest-observation distance was approximately 0.038875.

### 9.4 Function 4 — GP-Weighted Refinement

Function 4's GP had demonstrated considerably stronger realised accuracy than the NN.

The refined candidate was:

`[0.400697, 0.362771, 0.375829, 0.441739]`

The GP predicted approximately 0.744666, exceeding the current best of 0.622950.

The NN was less optimistic, predicting approximately 0.352589.

The candidate's nearest-observation distance was 0.035225, providing greater novelty than the cluster-guided alternative.

The refined candidate was therefore preferred.

### 9.5 Function 5 — Targeted Boundary Refinement

Function 5 required an additional comparison because all evidence continued to favour the region close to `[1,1,1,1]`.

Three candidates were considered:

| Candidate | GP Prediction | NN Prediction | Nearest Distance |
|---|---:|---:|---:|
| Cluster-guided | 8321.19 | 8326.64 | 0.011843 |
| More novel refinement | 7899.38 | 7935.63 | 0.033245 |
| Targeted boundary refinement | 8141.39 | 8155.83 | 0.020448 |

The targeted candidate was:

`[0.981966, 0.999701, 0.990442, 0.998795]`

It provided an intermediate balance between predicted performance and novelty.

Both models agreed closely, although neither predicted an improvement over the current best of 8662.48.

The comparison reinforced the conclusion that the strongest-known region was located at or very close to the upper boundary.

### 9.6 Function 6 — GP/NN Refinement

The refined candidate was:

`[0.400874, 0.395758, 0.500319, 0.800539, 0.004733]`

The GP predicted -0.228270 and the NN predicted -0.211156.

Both predictions exceeded the current best of -0.239769.

The candidate also provided slightly greater novelty than the cluster-guided alternative, with a nearest-observation distance of 0.053264.

This was one of the more strongly supported selections.

### 9.7 Function 7 — GP/NN Refinement

The refined candidate was:

`[0.176017, 0.276563, 0.482241, 0.296795, 0.315358, 0.706009]`

The GP predicted approximately 3.128395, while the NN predicted 3.071368.

Both exceeded the current best of 3.024610.

Although the cluster-guided candidate offered greater novelty, the refined point received stronger predictions from both models.

Its nearest-observation distance of approximately 0.082579 was considered sufficient to provide useful new information.

### 9.8 Function 8 — GP-Weighted Refinement

The refined candidate was:

`[0.077042, 0.136953, 0.185881, 0.169530, 0.768770, 0.510795, 0.209002, 0.691691]`

The GP predicted approximately 9.998146 and the NN predicted 9.986669.

Both predictions were close to or slightly above the current best of 9.984981.

The close model agreement was more convincing than the highly optimistic NN prediction for the cluster-guided alternative.

Given the GP's stronger realised accuracy and Function 8's weak clustering structure, the refined candidate was preferred.

## 10. Final Week 11 Query Selections

The eight final candidates were checked for correct dimensionality, permitted input bounds and duplication against existing observations.

| Function | Final Proposed Point | Nearest Distance |
|---|---|---:|
| 1 | `[0.249506, 0.999107]` | 0.246275 |
| 2 | `[0.729306, 0.186865]` | 0.027213 |
| 3 | `[0.428706, 0.513778, 0.457382]` | 0.038875 |
| 4 | `[0.400697, 0.362771, 0.375829, 0.441739]` | 0.035225 |
| 5 | `[0.981966, 0.999701, 0.990442, 0.998795]` | 0.020448 |
| 6 | `[0.400874, 0.395758, 0.500319, 0.800539, 0.004733]` | 0.053264 |
| 7 | `[0.176017, 0.276563, 0.482241, 0.296795, 0.315358, 0.706009]` | 0.082578 |
| 8 | `[0.077042, 0.136953, 0.185881, 0.169530, 0.768770, 0.510795, 0.209002, 0.691691]` | 0.109510 |

All eight candidates passed the saved validation checks.

Every point had the correct dimensionality, remained within `[0,1]` and was distinct from the existing observations.

### Submission Format

    Function 1: 0.249506-0.999107
    Function 2: 0.729306-0.186865
    Function 3: 0.428706-0.513778-0.457382
    Function 4: 0.400697-0.362771-0.375829-0.441739
    Function 5: 0.981966-0.999701-0.990442-0.998795
    Function 6: 0.400874-0.395758-0.500319-0.800539-0.004733
    Function 7: 0.176017-0.276563-0.482241-0.296795-0.315358-0.706009
    Function 8: 0.077042-0.136953-0.185881-0.169530-0.768770-0.510795-0.209002-0.691691

## 11. Week 11 Reflection

Week 11 demonstrated how unsupervised learning could provide additional information about the structure of the search space without replacing the established Bayesian Optimisation approach.

Two of the latest queries produced new best observations, with Functions 4 and 7 continuing to improve through local refinement.

The realised prediction analysis reinforced the importance of treating GP and NN reliability separately for each function. The GP performed particularly well for Functions 4, 7 and 8, while the NN produced more accurate predictions for Functions 3, 5 and 6.

The introduction of K-Means clustering provided an additional perspective on the accumulated observations.

Functions 2 and 5 demonstrated the clearest cluster separation, suggesting that the observations in these functions occupied relatively distinct regions of their input spaces.

For Functions 3, 4, 6 and 7, clustering provided useful supplementary evidence for promising regions already identified through surrogate modelling and local refinement.

Functions 1 and 8 demonstrated why clustering needed to be interpreted cautiously. Function 1 had no clearly superior cluster despite moderate separation, while Function 8 showed weak separation despite having a cluster containing its best-performing observations.

The cluster-guided candidate analysis generated several useful alternatives, particularly for Functions 6 and 7.

However, final selection remained dependent on GP and NN predictions, realised accuracy, historical observations and novelty rather than clustering alone.

Function 5's targeted boundary comparison also reinforced the evidence that performance decreased when moving away from `[1,1,1,1]`, supporting continued local refinement close to the upper boundary.

Overall, Week 11 expanded the analysis through K-Means clustering while preserving the function-specific optimisation strategy developed over previous rounds.

The next round will incorporate the returned outputs and investigate whether dimensionality reduction techniques can provide further insight into the structure of the higher-dimensional functions.

---

**Associated code:** `Week_11.ipynb`


