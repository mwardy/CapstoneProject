
# Week 8 — Reassessing Surrogate Reliability and Targeted Local Refinement

**Imperial College AI/ML Capstone Project | Module 19**

## Objective

The objective of Week 8 was to incorporate the latest observations, reassess the reliability of the Gaussian Process (GP) and neural-network (NN) models, and refine the approach used to select new query points.

Previous rounds demonstrated that model agreement and strong predictions did not necessarily translate into improvements. This week therefore placed greater emphasis on model validation, observed neighbourhood performance and the reliability of individual surrogate models.

The optimisation continued using Upper Confidence Bound (UCB), Expected Improvement (EI), neural-network predictions and function-specific local refinement. Additional investigation was conducted for Function 1, whose surrogate models continued to struggle.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 7 queries were incorporated into the existing datasets and compared with the previous best observations.

| Function | Previous Best | New Output | Result |
|---|---:|---:|---|
| 1 | 0.000000 | 0.000000 | No improvement |
| 2 | 0.746962 | 0.569904 | No improvement |
| 3 | -0.015498 | -0.010190 | New best |
| 4 | -0.200431 | 0.186912 | New best |
| 5 | 7121.151955 | 8662.482500 | New best |
| 6 | -0.379740 | -0.388687 | No improvement |
| 7 | 2.929370 | 2.369289 | No improvement |
| 8 | 9.911249 | 9.930811 | New best |

Four of the eight queries produced new best observations.

**Function 4** achieved a substantial improvement, increasing from approximately -0.2004 to 0.1869. This was an encouraging result following the strong agreement between acquisition functions and neural-network recommendations in the previous round.

**Function 5** also improved considerably, increasing from approximately 7121.15 to 8662.48. The result supported the decision to investigate the upper boundary of its search space and provided further evidence that its highest-performing region was concentrated near large values of all four input variables.

Functions 3 and 8 achieved smaller improvements, while Functions 1, 2, 6 and 7 did not improve.

Function 2 was particularly informative because its previous candidate had strong GP–NN agreement. The unsuccessful result demonstrated that agreement between models does not guarantee an improved observation.

Function 1 again produced an effectively zero output, reinforcing its continuing modelling difficulties.

Overall, these results supported a function-specific optimisation strategy that placed increasing weight on actual historical performance and neighbouring observations.

## 2. Refit and Diagnose the Surrogate Models

The Gaussian Process and neural-network models were refitted using the expanded datasets.

The existing configurations were retained to allow changes in model behaviour to be assessed without introducing unnecessary methodological differences.

### 2.1 Gaussian Process Diagnostics

The RBF Gaussian Processes were refitted and their optimised length scales examined.

| Function | Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 1e-05 |
| 3 | 0.199 |
| 4 | 0.416 |
| 5 | 0.273 |
| 6 | 0.581 |
| 7 | 0.302 |
| 8 | 1.040 |

Most functions continued to produce relatively stable length scales.

Functions 3–8 remained above the lower length-scale bound, suggesting that their models could identify relationships between the existing observations.

However, **Function 2 experienced a significant deterioration**, with its fitted length scale falling from approximately 0.0431 in Week 7 to the lower bound of `1e-05`.

This was particularly important because Function 2 had previously produced useful GP predictions and had achieved a new best observation in Week 6.

Function 1 also continued to reach the lower bound, indicating that its longstanding modelling problem remained unresolved.

### 2.2 Prediction and Uncertainty Diagnostics

The fitted GPs were evaluated across 100,000 candidate points per function.

| Function | Predicted Mean Range | Uncertainty Range |
|---|---:|---:|
| 1 | 0.000560 | 0.001570 |
| 2 | 0.651312 | 0.244076 |
| 3 | 0.388723 | 0.078255 |
| 4 | 35.821887 | 7.855238 |
| 5 | 7847.668688 | 2510.829993 |
| 6 | 2.303126 | 0.523703 |
| 7 | 3.026197 | 0.818402 |
| 8 | 4.878650 | 0.653663 |

Functions 3–8 showed meaningful variation in predictions and uncertainty, suggesting that their Gaussian Processes continued to provide useful guidance.

Function 2 was more complicated. Despite its fitted length scale reaching the lower bound, its predicted outputs still varied across the candidate space. However, its maximum predicted output of approximately 0.5696 was below the current observed best of 0.7470.

This suggested that the GP was not currently identifying a particularly promising region and that its recommendations should receive less weight.

Function 1 continued to produce extremely small predicted outputs, offering limited evidence for selecting one unexplored region over another.

The results suggested that additional evidence would be needed for Functions 1 and 2 before selecting their next queries.

## 3. Retrain and Validate Neural Networks

The neural-network ensembles introduced in Week 5 were retrained using the additional observations.

The architecture remained unchanged, with five independently initialised networks per function and two hidden layers containing 16 neurons each.

### 3.1 Neural-Network Training Performance

| Function | Mean Final Training Loss |
|---|---:|
| 1 | 0.000963 |
| 2 | 0.010989 |
| 3 | 0.000408 |
| 4 | 0.000284 |
| 5 | 0.000230 |
| 6 | 0.000370 |
| 7 | 0.000338 |
| 8 | 0.000381 |

The models continued to achieve relatively low training losses.

Functions 3–8 all recorded mean training losses close to or below 0.0004, indicating that their networks could represent the existing observations effectively.

Function 2 was the main exception, with its training loss increasing to approximately 0.0110.

This was particularly interesting because its GP also experienced fitting difficulties during the same round.

However, training losses alone were not considered reliable indicators of generalisation, so Leave-One-Out Cross-Validation was repeated.

### 3.2 Leave-One-Out Cross-Validation

| Function | NN LOOCV MAE | Mean Baseline MAE | Beats Baseline |
|---|---:|---:|---|
| 1 | 0.001704 | 0.001050 | No |
| 2 | 0.219676 | 0.237993 | Yes |
| 3 | 0.062478 | 0.051662 | No |
| 4 | 1.825570 | 7.284354 | Yes |
| 5 | 230.163312 | 2049.255132 | Yes |
| 6 | 0.241315 | 0.513894 | Yes |
| 7 | 0.399610 | 0.639930 | Yes |
| 8 | 0.354980 | 0.962395 | Yes |

The neural networks outperformed the mean-prediction baseline for **six of the eight functions**.

Functions 4, 5, 6 and 8 continued to demonstrate particularly strong relative improvements, supporting their use as supplementary surrogate models.

Function 2 also passed validation, although only narrowly. Its MAE of approximately 0.2197 was slightly below the baseline of 0.2380.

This was important because the neural network could potentially provide an alternative source of information following the deterioration of Function 2's GP.

Functions 1 and 3 again failed to outperform the baseline, suggesting that their NN predictions should receive limited weight when selecting the next candidates.

Overall, the results reinforced the importance of evaluating surrogate reliability individually rather than assuming that all networks were equally useful.

## 4. Compare Acquisition-Function Recommendations

The updated Gaussian Processes were used to evaluate candidate points under the established acquisition-function settings.

UCB was examined using different exploration parameters, while EI provided an alternative assessment of expected improvement over the current observed maximum.

The purpose was to determine whether the acquisition functions continued to identify similar promising regions or whether model recommendations had become unstable.

### 4.1 Function-Specific Findings

**Function 1**

The different UCB and EI settings continued to produce the same GP candidate. However, the GP remained at the lower length-scale bound and the neural network failed validation.

Agreement between acquisition functions was therefore not considered sufficient evidence for selecting the candidate.

**Function 2**

The GP had become unreliable, producing an extremely short fitted length scale and predictions that did not exceed its current best observation.

Its acquisition recommendations were consequently treated cautiously, with greater attention given to NN validation and previously successful neighbourhoods.

**Function 3**

The latest query produced a new best observation of -0.010190. The GP remained relatively stable, while the NN continued to fail LOOCV.

This supported retaining GP-based local refinement rather than relying on NN recommendations.

**Function 4**

The substantial improvement to 0.186912 justified continuing to explore the surrounding high-performing region.

The GP remained informative, and the NN also demonstrated strong cross-validation performance, providing evidence for combining their recommendations.

**Function 5**

The new best output of 8662.48 strengthened the case for continuing to investigate the upper boundary of the search space.

Although some acquisition settings favoured more exploratory candidates, the strongest historical evidence continued to support local boundary refinement.

**Function 6**

The GP remained informative, and its NN continued to perform well in cross-validation.

Both models were therefore considered useful for identifying potential refinements around the current best observations.

**Function 7**

The previous query failed to improve, but the highest-performing observations remained concentrated within a relatively promising region.

Model recommendations were assessed for both predictive performance and candidate novelty.

**Function 8**

The latest query produced a small new best output of 9.930811.

This supported continued local refinement, particularly where lower-exploration GP recommendations remained close to previously successful observations.

## 5. Compare GP and Neural-Network Recommendations

The candidates identified by the acquisition functions were compared with those produced by the neural-network ensembles.

The comparison considered candidate locations, predicted outputs, validation performance and the degree of agreement between models.

### 5.1 Interpretation of Model Agreement

The results demonstrated that agreement was most useful when accompanied by reliable validation and supporting observations.

For **Function 2**, the neural network provided a potentially more useful recommendation than the GP because it continued to outperform the LOOCV baseline despite the deterioration in GP fitting.

For **Function 3**, the GP remained preferable because the NN continued to demonstrate poor generalisation.

Functions 4 and 5 showed stronger evidence for the established high-performing regions, with both surrogate approaches supporting further local refinement.

Function 6 also presented useful agreement between model recommendations near its strongest existing observations.

For Function 7, leading recommendations often returned existing observations, demonstrating the need for additional local candidate generation.

Function 8 produced a choice between an exploratory recommendation and a more conservative local candidate. Given the latest successful query, the latter was considered better supported.

Function 1 continued to show substantial disagreement between the surrogate models, with a distance of approximately 0.6312 between their recommendations.

Because neither model was sufficiently reliable, additional investigation was required.

## 6. Assess Candidate Neighbourhoods

The nearest existing observations to the proposed candidates were examined to determine whether the model predictions were supported by actual historical results.

This was particularly important for functions where acquisition scores were attractive but the selected candidates were located far from previously successful observations.

### 6.1 Key Neighbourhood Findings

**Function 2:** The NN candidate was located near observations with outputs of approximately 0.611205, 0.366010 and 0.507418. In contrast, the exploratory GP candidate was associated with substantially weaker observations. This supported using the NN-identified region, although its predicted output remained below the current best.

**Function 3:** The latest improvement provided evidence for continued refinement near the established high-performing region. The GP remained more reliable than the NN.

**Function 4:** The latest result of 0.186912 provided strong evidence for exploiting the surrounding region, rather than moving towards distant candidates with weaker historical support.

**Function 5:** The upper-boundary region continued to produce exceptional outputs, and model recommendations remained concentrated around the largest input values.

**Function 6:** Both surrogate models identified candidate regions supported by relatively strong previous observations.

**Function 7:** The strongest model recommendations remained near the best-known observation, but a new point was required to avoid duplication.

**Function 8:** Both NN and lower-κ UCB recommendations were associated with strong observations. However, the GP candidate provided stronger local support and remained closer to the established successful region.

**Function 1:** Nearby observed outputs offered little meaningful differentiation between candidate regions, further supporting a model-independent exploration strategy.

The neighbourhood analysis therefore favoured local refinement for most functions, with Function 1 remaining the principal exception.

## 7. Additional Investigation of Function 1

Function 1 was investigated separately because neither surrogate approach provided sufficiently reliable guidance.

The RBF GP again reached the lower length-scale bound, while the NN continued to fail LOOCV.

Alternative Gaussian Process configurations were considered, including:

- Matérn 3/2
- Matérn 5/2
- RBF with a WhiteKernel noise component

These were assessed using Leave-One-Out Cross-Validation against the mean-prediction baseline.

The purpose was to determine whether the original RBF kernel was responsible for the poor modelling performance or whether the available observations themselves provided insufficient structure.

As the surrogate evidence remained inconclusive, maximum-distance space-filling exploration was investigated.

### 7.1 Space-Filling Candidates

Candidate points were evaluated using their distance to the nearest existing observation.

| Candidate | Coordinates | Nearest-Observation Distance |
|---|---|---:|
| 1 | `[0.998294, 0.000709]` | 0.307658 |
| 2 | `[0.684654, 0.001334]` | 0.305975 |
| 3 | `[0.006886, 0.695109]` | 0.301267 |
| 4 | `[0.786713, 0.000852]` | 0.269276 |
| 5 | `[0.898148, 0.005297]` | 0.265794 |

The three strongest candidates provided similar degrees of exploration, with nearest-observation distances between approximately 0.301 and 0.308.

The GP assigned effectively identical predictions and uncertainty to these points, providing little basis for distinguishing between them.

The NN showed greater variation but continued to lack reliable validation performance.

Candidate 1 was therefore preferred because it maximised distance from existing observations, with limited secondary support from the NN prediction.

This was a deliberately exploration-focused decision rather than an attempt to exploit unreliable model predictions.

## 8. Final Function-Specific Strategy

The final selections considered recent query performance, GP reliability, UCB and EI recommendations, NN validation, candidate neighbourhoods and the requirement to select previously unobserved points.

| Function | Selected Strategy | Main Reason |
|---|---|---|
| 1 | Space-filling exploration | Neither surrogate reliable |
| 2 | NN-supported region | GP unreliable; NN and neighbourhood evidence stronger |
| 3 | Local GP refinement | Latest improvement; NN fails validation |
| 4 | Local refinement | Major improvement and GP/NN support |
| 5 | Local boundary refinement | Strongest results concentrated near upper boundary |
| 6 | Local GP/NN refinement | Both models support promising region |
| 7 | Local novel refinement | Strong region but leading candidates duplicate observations |
| 8 | Lower-κ UCB refinement | New best and strong local evidence |

For Functions 4, 5 and 7, additional local searches were necessary because some leading model recommendations corresponded to existing observations.

Novel candidates were generated around the current best observations, constrained to the permitted `[0, 1]` input range and assessed using the surrogate models.

This allowed the optimisation to continue refining successful regions without repeating previously submitted queries.

### 8.1 Local Refinement Results

**Function 4**

Selected candidate:

`[0.386248, 0.381010, 0.343907, 0.423092]`

The selected point remained close to the newly successful region, allowing further exploitation following the substantial improvement recorded in the latest round.

**Function 5**

Selected candidate:

`[1.000000, 1.000000, 0.970000, 1.000000]`

The candidate remained close to the upper boundary, reflecting the strong historical results and model predictions in this region.

**Function 6**

Selected candidate:

`[0.426973, 0.384648, 0.468006, 0.752822, 0.084308]`

The candidate was selected from a locally promising region supported by both surrogate models.

**Function 7**

Selected candidate:

`[0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025]`

The local search identified a novel point approximately 0.0479 from the current best.

The GP predicted approximately 2.920647 and the NN predicted approximately 2.936110, providing useful agreement between the models.

**Function 8**

Selected candidate:

`[0.026226, 0.080912, 0.128250, 0.056836, 0.748250, 0.399531, 0.253527, 0.720750]`

The lower-κ UCB recommendation remained close to the latest successful region and was preferred over more exploratory alternatives.

## 9. Final Validation and Submission

Before submission, the final candidates were checked to ensure:

- Every function had the correct number of input dimensions.
- All coordinates remained within the permitted `[0, 1]` range.
- No candidate duplicated an existing observation.
- Each final selection reflected the intended optimisation strategy.

All eight final candidates passed the saved validation checks.

The nearest-observation distances ranged from approximately 0.0141 for Function 4 to 0.3077 for Function 1, reflecting the different degrees of exploration and local refinement.

The final points were formatted to six decimal places for submission.

## 10. Week 8 Reflection

Week 8 reinforced the importance of evaluating surrogate reliability and historical observations when making optimisation decisions.

Four of the previous queries produced new best outputs, with particularly substantial improvements for Functions 4 and 5. These results supported continued local refinement of established successful regions.

However, the deterioration of Function 2's Gaussian Process demonstrated that surrogate behaviour could change significantly as additional observations were incorporated. Although its neural network continued to outperform the LOOCV baseline, the relatively modest validation advantage required caution.

Function 1 remained the most challenging problem. Neither GP nor NN modelling provided reliable evidence of a promising region, so model-independent space-filling exploration was retained.

Functions 4, 5 and 7 also highlighted the need to distinguish between identifying a promising region and selecting a valid new query. Where model recommendations returned existing observations, additional local searches were used to identify novel candidates.

Overall, the optimisation strategy increasingly relied on a combination of surrogate predictions, cross-validation, historical query performance and observed neighbourhood evidence.

This provided a more robust basis for making function-specific decisions while retaining the original objective of identifying progressively higher-performing inputs.

---

**Associated code:** `Week_08.ipynb`



