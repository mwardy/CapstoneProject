
# Module 22 – Week 11: Bayesian Optimisation

## Methodology

Week 11 continued the Bayesian optimisation of eight unknown functions, incorporating the observations returned from Week 10. The approach remained based on Gaussian Process (GP) regression, neural-network (NN) models, model validation, neighbourhood evidence and function-specific candidate refinement.

An additional focus this week was the introduction of **K-Means clustering** to investigate whether the accumulated observations formed identifiable regions within each function's search space. Rather than only examining the observations nearest to individual candidates, clustering was used to assess whether high-performing points belonged to broader promising regions.

The workflow comprised eight stages:

1. Incorporate Week 10 observations and assess realised prediction accuracy.
2. Refit GP and NN surrogates and evaluate model reliability.
3. Analyse recent optimisation progress and investigate possible stagnation.
4. Apply K-Means clustering and evaluate cluster performance.
5. Generate candidates using the strongest-performing clusters.
6. Generate function-specific alternatives using established refinement strategies.
7. Compare candidate recommendations and perform additional boundary refinement where necessary.
8. Validate and format the final submissions.

K-Means was applied to the input observations, with **k = 2–5** evaluated using silhouette scores. Cluster performance was subsequently assessed using mean output, best output and cluster membership.

Cluster-guided candidates were compared with function-specific alternatives using GP and NN predictions, model reliability, realised prediction accuracy, novelty and previous optimisation performance.

Clustering was used as additional evidence rather than replacing the existing optimisation methods. Greater weight was given to its results where clusters were more clearly separated and demonstrated meaningful differences in performance.

---

## 1. Week 10 Results

The eight observations returned from the previous week's submissions were incorporated and compared with the previous best output for each function.

| **Function** | **Previous best** | **New output** | **Improved?** |
| ------------ | ----------------- | -------------- | ------------- |
| F1 | 0.000000 | 0.000000 | No |
| F2 | 0.746962 | 0.666153 | No |
| F3 | -0.000863 | -0.010582 | No |
| F4 | 0.583730 | 0.622950 | Yes |
| F5 | 8662.482500 | 8115.177655 | No |
| F6 | -0.239769 | -0.283228 | No |
| F7 | 2.929370 | 3.024610 | Yes |
| F8 | 9.984981 | 9.960611 | No |

**Two of the eight functions achieved new best outputs.**

F4 continued its recent improvement, increasing from 0.583730 to 0.622950. Although the gain of approximately 0.039220 was smaller than previous improvements, it provided further evidence that the established local region remained productive.

F7 also achieved a new best, improving from 2.929370 to 3.024610. This supported the previous week's decision to continue local refinement in a region identified by both surrogate models and neighbourhood evidence.

The remaining functions did not improve their existing bests.

F2 returned 0.666153, below its best of 0.746962 but still a relatively strong result. F3 again performed worse than its current best despite remaining close to the established high-performing region.

F5 returned 8115.177655, approximately 547 below its best of 8662.482500. This provided further evidence that moving away from `[1,1,1,1]` reduced performance.

F6 and F8 remained relatively close to their existing bests without improving them, while F1 again returned effectively zero following model-independent exploration.

These results supported continued refinement for F4 and F7, while suggesting that the structure of the promising regions identified for the other functions required further investigation.

### 1.1 Realised Model Prediction Accuracy

The GP and NN predictions recorded for the previous week's submitted candidates were compared with their actual returned outputs.

Absolute prediction errors were calculated where model predictions were available.

| **Function** | **GP absolute error** | **NN absolute error** | **More accurate model** |
| ------------ | --------------------- | --------------------- | ----------------------- |
| F1 | — | — | Not applicable |
| F2 | — | 0.0433 | NN only |
| F3 | 0.0221 | 0.0099 | NN |
| F4 | 0.0270 | 0.0889 | GP |
| F5 | 41.22 | 28.87 | NN |
| F6 | 0.0741 | 0.0526 | NN |
| F7 | 0.0440 | 0.0707 | GP |
| F8 | 0.0235 | 0.1246 | GP |

F1 could not be assessed because its candidate was selected through model-independent exploration. F2 had only an NN prediction available.

F4 provided particularly encouraging evidence for GP reliability. Its GP prediction of approximately 0.6499 was close to the realised new best of 0.622950, producing an absolute error of only 0.0270.

F7 also showed useful predictive performance, with the GP predicting approximately 2.9806 compared with the actual new best of 3.024610. The NN was also reasonably accurate, supporting the previously identified region.

F8's GP was substantially more accurate than its NN, with errors of 0.0235 and 0.1246 respectively. This reinforced the decision to place greater weight on GP predictions for this function.

For F3, F5 and F6, the NN produced the smaller prediction errors.

F5 was particularly notable because both models predicted the decline associated with moving away from the upper boundary reasonably accurately, despite the much larger scale of its outputs.

F2's NN prediction was also considerably more accurate than in the previous round, with an error of approximately 0.0433.

Overall, the results supported maintaining function-specific model weighting. The GP had stronger recent realised evidence for F4, F7 and F8, while the NN provided closer predictions for F3, F5 and F6.

---

## 2. Surrogate Diagnostics

### 2.1 Gaussian Process Models

The GP models were refitted using the expanded datasets. Fitted RBF length scales and optimisation warnings were examined to identify changes in model behaviour.

| **Function** | **Previous length scale** | **New length scale** | **At lower bound?** |
| ------------ | ------------------------- | -------------------- | ------------------- |
| F1 | 0.00001 | 0.00001 | Yes |
| F2 | 0.00001 | 0.00001 | Yes |
| F3 | 0.0968 | 0.0642 | No |
| F4 | 0.316 | 0.272 | No |
| F5 | 0.338 | 0.359 | No |
| F6 | 0.518 | 0.470 | No |
| F7 | 0.320 | 0.338 | No |
| F8 | 1.070 | 1.080 | No |

The most important finding was that **F1 and F2 continued to reach the GP length-scale lower bound of 1e-05**.

The additional observations therefore did not resolve the modelling difficulties previously identified for these functions. Their GP predictions continued to receive limited weight during candidate selection.

F3's length scale decreased from 0.0968 to 0.0642. This suggested that the model was representing increasingly local variation, consistent with the recent observations where small movements from the current best produced substantially worse outputs.

F4's length scale decreased slightly to 0.272 but remained well away from the lower bound. Combined with its accurate realised prediction and latest new best, this continued to support GP-led refinement.

F5 and F7 showed modest increases in their fitted length scales, while F6 decreased slightly from 0.518 to 0.470.

F8 remained particularly stable, with its length scale increasing marginally from 1.070 to 1.080. It continued to have the largest fitted length scale of the eight functions.

Overall, F3–F8 retained usable GP models without optimisation warnings, while F1 and F2 remained unreliable.

### 2.2 Neural-Network Validation

The NN models were retrained and evaluated using leave-one-out cross-validation (LOOCV), comparing their mean absolute error against a mean-prediction baseline.

| **Function** | **NN LOOCV MAE** | **Baseline MAE** | **Beats baseline?** |
| ------------ | ---------------- | ---------------- | ------------------- |
| F1 | 0.0278 | 0.0009 | No |
| F2 | 0.2369 | 0.2392 | Yes |
| F3 | 0.0460 | 0.0543 | Yes |
| F4 | 1.4783 | 7.7991 | Yes |
| F5 | 160.03 | 2752.03 | Yes |
| F6 | 0.2398 | 0.5474 | Yes |
| F7 | 0.4733 | 0.7777 | Yes |
| F8 | 0.3672 | 1.0152 | Yes |

**Seven of the eight NN models outperformed the mean-prediction baseline**, compared with six in the previous round.

F3 showed the most important improvement. Its NN LOOCV MAE decreased from approximately 0.0769 to 0.0460, allowing it to outperform the baseline. This was consistent with its more accurate realised prediction in the previous round.

F5 also improved substantially, with an NN MAE of approximately 160.03 compared with a baseline of 2752.03.

F4 and F8 continued to demonstrate useful validation performance, while F6 and F7 remained below their baselines despite weaker results than in Week 10.

F2 only narrowly outperformed its baseline, with an MAE of 0.2369 compared with 0.2392. Although its latest realised prediction was reasonably accurate, the narrow validation margin suggested that the NN should still be treated cautiously.

F1 continued to fail validation, supporting the decision to retain model-independent exploration.

Overall, the updated validation suggested that the NN could provide useful information for most functions, although the strength of the evidence varied considerably.

---

## 3. Optimisation Progress

The running-best outputs were examined to assess whether each function continued to improve or was beginning to plateau.

The analysis considered the location of the current best observation, the number of observations since the last improvement and the frequency of previous improvements.

| **Function** | **Current best** | **Recent progress** |
| ------------ | ---------------- | ------------------- |
| F1 | ~0.000000 | No improvement across 17 subsequent observations. |
| F2 | 0.746962 | Best found five observations ago; no recent improvement. |
| F3 | -0.000863 | Best found two observations ago; subsequent refinement unsuccessful. |
| F4 | 0.622950 | Latest observation improved the best; continued progress. |
| F5 | 8662.482500 | Best found three observations ago; subsequent boundary moves weaker. |
| F6 | -0.239769 | Relatively recent best; latest observation unsuccessful. |
| F7 | 3.024610 | Latest observation established a new best. |
| F8 | 9.984981 | Relatively recent best; latest observation remained close. |

F1 showed the clearest evidence of stagnation. Its best output had been established at the second observation, with 17 subsequent observations failing to improve it.

F2 also showed signs of plateauing, although its latest output of 0.666153 suggested that its strongest region remained relevant.

F3 had achieved relatively frequent improvements historically, but its two latest local searches failed to improve on -0.000863. This suggested that very tight refinement might now be producing diminishing returns.

F4 showed the strongest evidence of sustained progress, with 12 recorded improvements and another new best in the latest round.

F5 also demonstrated strong historical progress, although its recent unsuccessful boundary queries reinforced the importance of the exact upper-boundary region.

F6 and F8 retained relatively recent best observations, suggesting that their strongest regions remained promising despite the latest unsuccessful queries.

F7 continued to improve, with the latest observation establishing a new best of 3.024610.

Overall, the results distinguished F4 and F7 as actively improving functions, F3, F5, F6 and F8 as functions with relatively recent bests, and F1 and F2 as showing stronger evidence of stagnation.

This provided useful context for the clustering analysis.

---

## 4. K-Means Clustering

### 4.1 Cluster Selection

K-Means clustering was introduced to identify whether the accumulated observations formed distinguishable regions within each function's input space.

Clustering was applied to the **input variables only**, with outputs evaluated separately after the clusters had been identified.

This allowed the clustering process to identify spatial structure independently of observed performance.

The number of clusters was assessed using silhouette scores for k = 2, 3, 4 and 5. The value with the highest score was selected for each function.

| **Function** | **Selected k** | **Silhouette score** |
| ------------ | -------------- | -------------------- |
| F1 | 4 | 0.4066 |
| F2 | 3 | 0.6462 |
| F3 | 5 | 0.3073 |
| F4 | 5 | 0.3068 |
| F5 | 2 | 0.4331 |
| F6 | 4 | 0.2572 |
| F7 | 5 | 0.3002 |
| F8 | 4 | 0.1853 |

F2 demonstrated the clearest cluster structure, with three clusters and a silhouette score of 0.6462.

F5 also showed reasonably clear separation, selecting two clusters with a score of 0.4331.

F1 had moderate separation at 0.4066, although this did not necessarily imply that its clusters differed meaningfully in output performance.

F3, F4 and F7 produced weaker scores of approximately 0.30, suggesting that the identified groups were less clearly separated.

F6 had a silhouette score of 0.2572, while F8 produced the weakest result at only 0.1853.

These findings suggested that clustering could provide useful additional evidence for some functions, particularly F2 and F5, but should not receive equal weight across all eight.

### 4.2 Cluster Performance

The performance of each identified cluster was examined using its mean output, best output, number of observations and centroid.

Particular attention was given to the cluster containing the current best observation.

| **Function** | **Best cluster** | **Cluster mean output** | **Cluster best output** |
| ------------ | ---------------- | ----------------------- | ----------------------- |
| F1 | 0 | -0.000515 | ~0.000000 |
| F2 | 2 | 0.591846 | 0.746962 |
| F3 | 2 | -0.046103 | -0.000863 |
| F4 | 0 | -5.322216 | 0.622950 |
| F5 | 1 | 5678.018467 | 8662.482500 |
| F6 | 2 | -0.596464 | -0.239769 |
| F7 | 1 | 1.983345 | 3.024610 |
| F8 | 0 | 9.669458 | 9.984981 |

F2 provided the strongest overall clustering evidence. Its best observation belonged to a cluster with a mean output of 0.591846, while its silhouette score indicated relatively clear separation from other regions.

This supported continued refinement within the identified high-performing cluster.

F5 also demonstrated a clear distinction between clusters. Its strongest cluster had a mean output of 5678.018467 and contained the boundary optimum of 8662.482500.

The strong cluster was concentrated towards higher input values, providing additional support for the established boundary-focused strategy.

F4 and F7 showed useful differences in cluster performance despite relatively weak silhouette scores. Their best-performing clusters contained the latest new best observations, supporting continued local refinement in these regions.

F3 and F6 also had their best observations within their strongest-performing clusters, although weaker separation meant that clustering provided supporting rather than decisive evidence.

F1's clustering offered limited additional information. Although the observations could be divided into four groups, none identified a clearly superior region.

F8 required particular caution because its silhouette score of only 0.1853 suggested that the clusters were poorly separated.

**Overall, clustering provided the strongest additional evidence for F2 and F5**, with more limited supporting evidence for F3, F4, F6 and F7.

The clustering results were therefore incorporated into the existing function-specific approach rather than used independently.

---

## 5. Cluster-Guided Candidates

The strongest-performing clusters were used to identify promising regions for further candidate generation.

Candidates were generated within these regions and evaluated using the existing GP and NN models, while retaining a minimum distance from previously observed points.

Cluster centroids were not treated as optimal candidates themselves.

| **Function** | **GP prediction** | **NN prediction** | **Nearest distance** |
| ------------ | ----------------- | ----------------- | -------------------- |
| F2 | 0.365797 | 0.661354 | 0.027213 |
| F3 | 0.037513 | -0.012296 | 0.039857 |
| F4 | 0.722666 | 0.412112 | 0.021532 |
| F5 | 8321.19 | 8326.64 | 0.011843 |
| F6 | -0.228475 | -0.216792 | 0.050226 |
| F7 | 3.087322 | 3.020521 | 0.110278 |
| F8 | 9.933104 | 10.252357 | 0.196623 |

F2's cluster-guided candidate was:

`[0.729306, 0.186865]`

The NN predicted 0.661354, below the current best of 0.746962 but relatively strong compared with several recent observations.

The GP was considerably less optimistic, predicting 0.365797. However, its continuing length-scale problem meant that the NN and clustering evidence were considered more informative.

F3's candidate received a strong GP prediction of 0.037513, but the NN predicted -0.012296. This disagreement suggested that additional local comparison was necessary before selection.

F4's GP predicted 0.722666, above the current best of 0.622950, while the NN was more conservative at 0.412112.

F5 demonstrated particularly close model agreement. The GP and NN predicted 8321.19 and 8326.64 respectively, supporting the high-performing boundary region.

However, the candidate was only 0.011843 from its nearest observation, limiting the additional information likely to be gained from the query.

F6 and F7 produced some of the strongest cluster-guided candidates.

For F6, both models predicted improvement over the current best of -0.239769, while maintaining a nearest-observation distance of 0.050226.

F7's GP predicted 3.087322 compared with the current best of 3.024610. Its NN prediction of 3.020521 was also close to the existing best, and the candidate retained considerable novelty at 0.110278.

F8's NN predicted 10.252357, substantially above the current best of 9.984981. However, the GP predicted only 9.933104, while the function's weak clustering structure reduced confidence in the recommendation.

Overall, the cluster-guided search provided useful alternatives, particularly for F2, F6 and F7.

However, the differences between model predictions and cluster reliability reinforced the need to compare these candidates with function-specific alternatives before submission.

---

## 6. Function-Specific Candidate Refinement

Additional candidates were generated using strategies tailored to each function's previous results, surrogate reliability, clustering evidence and recent optimisation progress.

The approach retained model-independent exploration for F1, NN-supported cluster refinement for F2, boundary refinement for F5 and local GP–NN refinement for the remaining functions.

Where recent realised prediction accuracy provided stronger evidence for one surrogate, its influence was increased.

| **Function** | **Selected refinement approach** | **Reasoning** |
| ------------ | -------------------------------- | ------------- |
| F1 | Space-filling exploration | GP unreliable, NN failed validation and no recent improvement. |
| F2 | NN-supported cluster refinement | Strong clustering evidence, validated NN and unreliable GP. |
| F3 | Local GP–NN refinement | Improved NN validation and supporting regional evidence. |
| F4 | GP-weighted local refinement | Recent GP accuracy and continued improvement. |
| F5 | Novel boundary refinement | Strong boundary performance with a minimum novelty constraint. |
| F6 | Local GP–NN refinement | Both models supported improvement within the strongest cluster. |
| F7 | Local GP–NN refinement | Latest new best and good regional model agreement. |
| F8 | GP-weighted local refinement | Stronger recent GP accuracy and weak clustering structure. |

### 6.1 Refined Candidate Results

| **Function** | **GP prediction** | **NN prediction** | **Nearest distance** |
| ------------ | ----------------- | ----------------- | -------------------- |
| F1 | — | — | 0.246275 |
| F2 | — | 0.619287 | 0.036269 |
| F3 | 0.037818 | -0.012456 | 0.038875 |
| F4 | 0.744666 | 0.352589 | 0.035225 |
| F5 | 7899.378880 | 7935.631782 | 0.033245 |
| F6 | -0.228270 | -0.211156 | 0.053264 |
| F7 | 3.128395 | 3.071368 | 0.082579 |
| F8 | 9.998146 | 9.986669 | 0.109510 |

F1's model-independent search selected:

`[0.249506, 0.999107]`

The candidate was approximately 0.246275 from its nearest observation, providing substantial novelty without relying on unreliable surrogate predictions.

F2's NN-supported refinement identified:

`[0.665824, 0.186906]`

The NN predicted 0.619287, below the current best but within the strongest-performing cluster. This provided slightly greater novelty than the cluster-guided candidate.

F3's refined candidate was:

`[0.428706, 0.513778, 0.457382]`

The GP predicted 0.037818, substantially above the current best of -0.000863, while the NN remained more conservative at -0.012456.

Although the models disagreed, the updated validation and clustering evidence supported continued investigation of the region.

F4's GP-weighted refinement identified:

`[0.400697, 0.362771, 0.375829, 0.441739]`

The GP predicted 0.744666, above the current best of 0.622950, while the NN predicted only 0.352589.

Greater weight was placed on the GP because it had been more accurate on the latest realised output. The refined candidate also provided greater novelty than the cluster-generated alternative.

F5's initial boundary refinement identified:

`[0.967622, 0.993026, 0.997140, 0.999716]`

Both models predicted a relatively high output, but neither expected an improvement over the existing best at `[1,1,1,1]`.

The candidate was approximately 0.033245 from its nearest observation, providing greater novelty than the cluster-guided alternative.

F6's refined candidate was:

`[0.400874, 0.395758, 0.500319, 0.800539, 0.004733]`

The GP and NN predicted -0.228270 and -0.211156 respectively, both above the current best of -0.239769.

The models showed relatively close agreement, and the candidate remained within the identified high-performing region.

F7's refinement identified:

`[0.176017, 0.276563, 0.482241, 0.296795, 0.315358, 0.706009]`

Both models predicted improvement over the current best of 3.024610, with GP and NN predictions of 3.128395 and 3.071368 respectively.

The candidate retained meaningful novelty while remaining within the strongest-performing region.

F8's GP-weighted refinement selected:

`[0.077042, 0.136953, 0.185881, 0.169530, 0.768770, 0.510795, 0.209002, 0.691691]`

The GP predicted 9.998146 and the NN predicted 9.986669, both slightly above the current best of 9.984981.

The substantially stronger agreement between the models made this candidate more convincing than the earlier cluster-guided alternative.

---

## 7. Final Candidate Selection

The cluster-guided and function-specific candidates were compared using predicted performance, model agreement, realised prediction accuracy, clustering evidence, uncertainty and candidate novelty.

| **Function** | **Preferred approach** | **Selection rationale** |
| ------------ | ---------------------- | ----------------------- |
| F1 | Space-filling exploration | Neither surrogate nor clustering identified a reliable improvement region. |
| F2 | Cluster-guided | Stronger NN prediction within the best-performing cluster. |
| F3 | Refined local candidate | Slightly stronger GP prediction; alternatives were very similar. |
| F4 | GP-weighted refinement | Stronger GP prediction and greater novelty than the cluster candidate. |
| F5 | Additional boundary investigation | Trade-off between strong boundary predictions and insufficient novelty. |
| F6 | Refined local candidate | Both models slightly favoured the refined point. |
| F7 | Refined local candidate | Stronger GP and NN predictions while retaining useful novelty. |
| F8 | GP-weighted refinement | Greater model agreement and stronger GP support. |

For F2, the cluster-guided candidate `[0.729306, 0.186865]` was preferred over the refined alternative because its NN prediction of 0.661354 was higher than 0.619287.

Although the cluster candidate had slightly less novelty, its nearest-observation distance of 0.027213 remained sufficient to provide a distinct query.

F3's two candidates were extremely similar. The refined candidate offered a slightly stronger GP prediction, while the cluster candidate had marginally stronger NN and novelty evidence. The refined candidate was therefore retained without further investigation.

For F4, the refined candidate was preferred because its GP prediction increased from 0.722666 to 0.744666 while its nearest-observation distance increased from 0.021532 to 0.035225.

For F6, both models slightly favoured the refined candidate, which also provided marginally greater novelty.

F7's refined candidate received stronger predictions from both models. Although the cluster candidate had greater novelty, the refined point remained approximately 0.082579 from existing observations.

F8's refined candidate was preferred because the GP and NN predictions were closely aligned, unlike the more speculative cluster-guided recommendation.

F5 required one final investigation because neither existing candidate provided an ideal balance between predicted performance and novelty.

### 7.1 Function 5 Targeted Boundary Refinement

F5's current best remained 8662.482500 at `[1,1,1,1]`.

The cluster-guided candidate was predicted to perform relatively well but was only 0.011843 from an existing observation. The refined boundary candidate provided greater novelty at 0.033245 but had noticeably weaker model predictions.

An intermediate novelty range of **0.02–0.04** was therefore investigated.

Candidates were generated close to the upper boundary and assessed using equally weighted standardised GP and NN predictions.

| **Approach** | **GP prediction** | **NN prediction** | **Nearest distance** |
| ------------ | ----------------- | ----------------- | -------------------- |
| Cluster-guided | 8321.19 | 8326.64 | 0.011843 |
| Novel boundary | 7899.38 | 7935.63 | 0.033245 |
| Targeted boundary | 8141.386269 | 8155.828807 | 0.020448 |

The targeted search identified:

`[0.981966, 0.999701, 0.990442, 0.998795]`

The GP predicted 8141.386269 and the NN predicted 8155.828807, with GP uncertainty of 4.961240.

This provided a compromise between the higher predictions but low novelty of the cluster-guided candidate and the greater novelty but weaker predictions of the earlier boundary refinement.

Both models continued to predict performance below the existing best, reinforcing the evidence that the strongest observed region lay at or extremely close to the upper boundary.

**The targeted candidate was selected primarily to obtain a distinct observation while remaining close to the established optimum.**

### 7.2 Final Selections

| **Function** | **Final candidate** | **Selection rationale** |
| ------------ | ------------------- | ----------------------- |
| F1 | [0.249506, 0.999107] | Space-filling exploration where surrogates remain unreliable. |
| F2 | [0.729306, 0.186865] | Cluster-guided selection supported by NN and strong clustering evidence. |
| F3 | [0.428706, 0.513778, 0.457382] | Local refinement with a slightly stronger GP prediction. |
| F4 | [0.400697, 0.362771, 0.375829, 0.441739] | GP-weighted refinement following another new best. |
| F5 | [0.981966, 0.999701, 0.990442, 0.998795] | Targeted boundary refinement balancing novelty and predicted output. |
| F6 | [0.400874, 0.395758, 0.500319, 0.800539, 0.004733] | Refined candidate supported by both surrogate models. |
| F7 | [0.176017, 0.276563, 0.482241, 0.296795, 0.315358, 0.706009] | Strong GP–NN agreement within the highest-performing region. |
| F8 | [0.077042, 0.136953, 0.185881, 0.169530, 0.768770, 0.510795, 0.209002, 0.691691] | GP-weighted refinement with improved cross-model agreement. |

---

## 8. Validation and Submission

The final candidates were checked for:

- Correct dimensionality for each function.
- Input values within the permitted `[0,1]` bounds.
- No duplicate observations.
- Distance from existing observations.
- Consistent formatting to six decimal places.

All eight candidates passed the validation checks.

Nearest-observation distances ranged from approximately **0.020448 for F5** to **0.246275 for F1**, reflecting the different exploitation and exploration strategies.

F1 remained the most exploratory selection, consistent with the continued lack of reliable surrogate guidance.

F5 had the smallest novelty distance, reflecting the strong evidence for performance close to `[1,1,1,1]`.

F6, F7 and F8 retained meaningful novelty distances of approximately 0.053264, 0.082579 and 0.109510 respectively, while remaining focused on their identified high-performing regions.

### Final Week 11 Submission Points

    Function 1: 0.249506-0.999107
    Function 2: 0.729306-0.186865
    Function 3: 0.428706-0.513778-0.457382
    Function 4: 0.400697-0.362771-0.375829-0.441739
    Function 5: 0.981966-0.999701-0.990442-0.998795
    Function 6: 0.400874-0.395758-0.500319-0.800539-0.004733
    Function 7: 0.176017-0.276563-0.482241-0.296795-0.315358-0.706009
    Function 8: 0.077042-0.136953-0.185881-0.169530-0.768770-0.510795-0.209002-0.691691

These points match the final selections produced by the Week 11 notebook.

---

## Reflection

Week 11 demonstrated the value of incorporating clustering as an additional source of evidence within the existing Bayesian optimisation workflow.

The latest results showed that local refinement remained productive for F4 and F7, both of which achieved new best outputs. However, the remaining functions continued to show different optimisation behaviour, reinforcing the need for function-specific selection.

The introduction of K-Means provided an additional perspective on the regions identified through previous neighbourhood and surrogate analysis.

F2 and F5 showed the strongest clustering structure. For F2, the clustering analysis identified a relatively well-separated high-performing region that supported continued NN-guided refinement despite the GP remaining unreliable.

For F5, clustering reinforced the importance of the upper boundary, with the strongest cluster concentrated towards high input values and containing the existing best at `[1,1,1,1]`.

F4 and F7 also showed useful regional structure, although their silhouette scores were weaker. Their strongest clusters contained the latest new best observations, providing additional support for continued refinement.

F3 and F6 benefited from clustering as secondary evidence. In both cases, the identified high-performing regions were consistent with the areas already favoured by previous optimisation methods.

F8 demonstrated an important limitation of the approach. Its low silhouette score suggested that the observations did not form clearly separated clusters, so clustering was given relatively little weight. The final candidate instead relied more heavily on GP accuracy and model agreement.

F1 remained the most difficult function. Neither surrogate provided reliable predictions, and clustering did not identify a clearly superior region. Model-independent exploration therefore remained preferable.

The realised prediction analysis was also useful. F4, F7 and F8 continued to provide encouraging evidence for the GP, while the NN was relatively more accurate for F3, F5 and F6. This supported adjusting model influence by function rather than using identical weights throughout.

F5's final boundary investigation illustrated the importance of balancing predicted performance against candidate novelty. The targeted search identified an intermediate candidate that remained close to the strongest observed region without repeating an almost identical query.

One methodological limitation is that Week 11 used an NN with a single hidden layer of 20 neurons, compared with the two 50-neuron hidden layers used in Week 10. Consequently, changes in NN validation performance between these weeks cannot be attributed solely to additional observations.

More generally, clustering was limited by the relatively small observation datasets and by the assumption that meaningful regions could be represented through K-Means partitions. Strong silhouette scores demonstrated spatial separation but did not independently establish high predictive performance. Cluster results were therefore used alongside GP and NN evidence rather than treated as definitive.

Overall, Week 11 expanded the optimisation methodology by introducing cluster-guided candidate generation while retaining the established function-specific approach. The final selections combined clustering, surrogate validation, realised prediction accuracy, regional performance and novelty to balance exploration and exploitation across the eight functions.

