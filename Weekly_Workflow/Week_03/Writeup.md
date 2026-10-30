
# Week 3 — Introducing SVMs and Adapting the Optimisation Strategy

**Imperial College AI/ML Capstone Project | Module 14**

## Objective

The objective of Week 3 was to incorporate the latest observations, reassess the Gaussian Process (GP) models and investigate whether Support Vector Machines (SVMs) could provide additional information about the unknown functions.

The previous round demonstrated that additional observations could materially improve GP behaviour, particularly for Function 6. However, the RBF models for some functions continued to experience difficulties with extremely small length scales, making their predictions and uncertainty less informative.

This week therefore introduced linear and nonlinear SVM classification as a supplementary method for identifying higher- and lower-performing regions of the search space.

The existing GP and Upper Confidence Bound (UCB) approach remained the baseline, with alternative strategies considered where the surrogate models were unable to provide reliable guidance.

## 1. Incorporate and Assess the New Observations

The outputs received from the Week 2 queries were added to the existing datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | 0.000000 | 0.000000 | No improvement |
| 2 | 0.611205 | 0.366010 | 0.611205 | No improvement |
| 3 | -0.020167 | -0.060870 | -0.020167 | No improvement |
| 4 | -0.300645 | -3.743172 | -0.300645 | No improvement |
| 5 | 2329.015854 | 2765.819787 | 2765.819787 | New best |
| 6 | -0.714265 | -0.458511 | -0.458511 | New best |
| 7 | 1.950612 | 0.975233 | 1.950612 | No improvement |
| 8 | 9.821986 | 9.621122 | 9.821986 | No improvement |

Two of the eight queries produced new best observations.

**Function 5** improved from approximately 2329.02 to 2765.82, providing further evidence that the optimisation was identifying a promising region of its input space. This represented a second consecutive improvement following the substantial increase recorded in Week 2.

**Function 6** also improved, increasing from approximately -0.7143 to -0.4585. This was particularly encouraging because Function 6 had initially experienced difficulties with its RBF Gaussian Process. After an exploration-based query in Week 1, the additional observation had improved its fitted model sufficiently to return to UCB in Week 2.

The remaining functions did not improve their previous best outputs. However, each new observation still contributed information that could be used to update the models and assess the effectiveness of the previous strategies.

## 2. Refit and Evaluate the Gaussian Process Models

The RBF Gaussian Process models were refitted using all available observations.

The fitted kernel length scales were examined to determine whether the additional data had improved the models or revealed continuing limitations.

| Function | Week 3 Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0427 |
| 3 | 0.202 |
| 4 | 0.495 |
| 5 | 0.413 |
| 6 | 1e-05 |
| 7 | 0.244 |
| 8 | 0.931 |

Most functions continued to produce length scales that were broadly consistent with previous rounds.

**Function 5** increased from approximately 0.225 to 0.413 following another successful query, suggesting that its updated model was capturing relationships across a wider region of the search space.

Functions 2, 3, 4, 7 and 8 also continued to produce usable fitted models.

However, Functions 1 and 6 both reached the lower RBF length-scale bound of `1e-05`.

Function 1 had experienced this issue since the beginning of the project, while Function 6 had returned to the lower bound after previously improving to approximately 0.481.

Although Function 6's latest query had produced a new best observation, the updated GP appeared to be struggling again. Further investigation was therefore necessary before using its predictions to select the next query.

### 2.1 Examine Predictions and Uncertainty

To investigate the behaviour of Functions 1 and 6, both fitted models were evaluated across 100,000 randomly generated candidate points.

The range of predicted means and standard deviations was examined to determine whether the models could distinguish between different regions of their input spaces.

| Metric | Function 1 | Function 6 |
|---|---:|---:|
| Minimum predicted mean | -0.0003005052 | -1.5012665979 |
| Maximum predicted mean | -0.0000001770 | -1.5012665979 |
| Predicted mean range | 0.0003003282 | 0.0000000000 |
| Minimum uncertainty | 0.0000342068 | 0.5421470063 |
| Maximum uncertainty | 0.0009966631 | 0.5421470063 |
| Uncertainty range | 0.0009624562 | 0.0000000000 |

Despite both models reaching the lower length-scale bound, their behaviour differed.

**Function 1** showed some variation in predicted output and uncertainty. However, the continued convergence warning and very small length scale suggested that its GP remained unreliable for selecting promising unexplored points.

**Function 6** produced identical predicted means and uncertainty across the candidate space at the reported precision. Consequently, its UCB scores could not meaningfully distinguish between the sampled candidates.

This meant that the standard GP and UCB approach was not suitable for Function 6 in its current form.

An alternative source of information was therefore investigated.

## 3. Introduce Support Vector Machines

Support Vector Machines were introduced as an additional method for investigating whether the existing observations contained identifiable high- and low-performing regions.

Unlike Gaussian Process regression, which predicts continuous outputs and associated uncertainty, SVM classification attempts to distinguish between predefined classes.

Because the unknown functions produce continuous outputs, a binary classification problem was constructed for each function.

The median observed output was used as the threshold:

- Observations above the median were classified as **high-performing**.
- Observations at or below the median were classified as **low-performing**.

The median was selected because it provided a simple, function-specific threshold and generally produced balanced classes.

This classification was intended as a diagnostic tool rather than a replacement for Bayesian Optimisation. High-performing classifications were relative to the observations collected so far and did not indicate the true maximum of a function.

### 3.1 Create High- and Low-Performing Classes

| Function | Median Threshold | Low-Performing Observations | High-Performing Observations |
|---|---:|---:|---:|
| 1 | 0.000000 | 6 | 6 |
| 2 | 0.229792 | 6 | 6 |
| 3 | -0.094190 | 9 | 8 |
| 4 | -15.756741 | 16 | 16 |
| 5 | 64.431793 | 11 | 11 |
| 6 | -1.446370 | 11 | 11 |
| 7 | 0.088196 | 16 | 16 |
| 8 | 7.940817 | 21 | 21 |

The median threshold produced balanced classes across all eight functions.

For most functions, the observations were divided equally between the two classes. Function 3 contained an odd number of observations and was divided into nine lower-performing and eight higher-performing observations.

This allowed linear and nonlinear SVM approaches to be compared without substantial class imbalance.

### 3.2 Compare Linear and RBF SVMs

Two classification models were evaluated:

1. **Linear SVM:** Tests whether the higher- and lower-performing observations can be separated using a linear decision boundary.
2. **RBF SVM:** Allows a nonlinear decision boundary, potentially capturing more complex relationships between the input variables.

The models were evaluated using three-fold cross-validation to compare their classification performance.

| Function | Linear SVM Mean Accuracy | RBF SVM Mean Accuracy | Higher Accuracy |
|---|---:|---:|---|
| 1 | 0.167 | 0.500 | RBF |
| 2 | 0.667 | 0.833 | RBF |
| 3 | 0.344 | 0.522 | RBF |
| 4 | 0.627 | 0.782 | RBF |
| 5 | 0.649 | 0.458 | Linear |
| 6 | 0.821 | 0.768 | Linear |
| 7 | 0.785 | 0.815 | RBF |
| 8 | 0.857 | 0.833 | Linear |

The comparison produced mixed results.

**Functions 2 and 4** showed noticeable improvements with the RBF SVM, suggesting that nonlinear decision boundaries may be useful for distinguishing relatively high- and low-performing observations.

**Function 6** achieved a relatively strong linear SVM accuracy of 0.821, compared with 0.768 for the RBF alternative. This was particularly interesting because its Gaussian Process was currently unable to distinguish between candidate points.

**Function 8** also performed strongly with a linear SVM, achieving an accuracy of 0.857, although the RBF alternative performed similarly.

Functions 1 and 3 remained difficult to classify, with neither model providing particularly convincing results.

Function 5 favoured the linear SVM, although its accuracy remained more moderate.

Overall, the results suggested that no single SVM kernel was consistently superior across all eight functions.

The analysis also demonstrated that useful information about higher- and lower-performing regions could sometimes be obtained from classification models even when the Gaussian Process regression models were struggling.

However, the limited number of observations meant that cross-validation results should be interpreted cautiously.

## 4. Reassess Exploration and Exploitation

Before selecting the third-round queries, the strategy for each function was reassessed using several sources of evidence:

- Whether the latest query improved the highest observed output.
- Whether the updated Gaussian Process produced meaningful predictions and uncertainty.
- Whether the SVM analysis identified useful separation between relatively high- and low-performing regions.
- Whether continued exploitation of previously promising regions or additional exploration appeared more appropriate.

The intention was not to replace the established Bayesian Optimisation methodology unnecessarily, but to adapt the selection approach where the available evidence justified doing so.

### Function-Specific Assessment

| Function | Assessment | Proposed Strategy |
|---|---|---|
| 1 | RBF length scale remains at the lower bound; SVM performance is unconvincing | Maximum-distance exploration |
| 2 | GP remains usable; RBF SVM suggests nonlinear structure | GP + UCB |
| 3 | GP remains usable; SVM results provide limited additional guidance | GP + UCB |
| 4 | Previously identified promising region remains worth investigating | GP + UCB |
| 5 | Successive improvements support continued exploitation | GP + UCB |
| 6 | GP predictions are uninformative; linear SVM performs relatively well | SVM-informed selection |
| 7 | GP remains usable; both SVM models perform reasonably well | GP + UCB |
| 8 | GP remains usable; SVM models show useful classification performance | GP + UCB |

Functions 2, 3, 4, 5, 7 and 8 therefore retained the existing GP and UCB approach.

Functions 1 and 6 were treated separately because their current Gaussian Processes could not provide sufficiently reliable guidance.

## 5. Select the Third-Round Queries

### 5.1 GP + UCB Selection

For Functions 2, 3, 4, 5, 7 and 8, the updated RBF Gaussian Process models were used to evaluate 100,000 candidate points.

The UCB acquisition function was retained with an exploration parameter of `κ = 2.5`.

UCB(x) = μ(x) + κ × σ(x)

The selected candidates were those with the highest UCB scores within their respective candidate sets.

| Function | Predicted Output | Uncertainty | UCB Score |
|---|---:|---:|---:|
| 2 | 0.449335 | 0.185666 | 0.913500 |
| 3 | -0.076907 | 0.077556 | 0.116983 |
| 4 | -0.064270 | 0.420976 | 0.988170 |
| 5 | 2631.682259 | 240.331386 | 3232.510725 |
| 7 | 1.662117 | 0.258204 | 2.307626 |
| 8 | 9.635913 | 0.434286 | 10.721628 |

The results highlighted different balances between predicted performance and uncertainty.

**Function 4** appeared particularly promising. Its selected candidate had a predicted output of approximately -0.064, compared with the current observed best of -0.301. Although uncertainty remained, this suggested that the candidate could potentially improve the known maximum.

**Function 5** continued to focus on the high-performing region identified in previous rounds. Its predicted output of approximately 2631.68 was below the current observed best of 2765.82, but uncertainty increased its UCB score to approximately 3232.51.

This suggested that the model was continuing to exploit a promising region while retaining some exploratory potential.

Functions 3 and 8 selected candidates with relatively substantial uncertainty, indicating a greater contribution from exploration.

### 5.2 Compare Selected Points with Current Best Observations

The proposed UCB selections were compared with each function's highest-performing observation using Euclidean distance.

| Function | Current Best Output | Distance from Current Best |
|---|---:|---:|
| 2 | 0.611205 | 0.0452 |
| 3 | -0.020167 | 0.8461 |
| 4 | -0.300645 | 0.1376 |
| 5 | 2765.819787 | 0.2030 |
| 7 | 1.950612 | 0.1612 |
| 8 | 9.821986 | 0.8392 |

The distances suggested that Functions 2, 4 and 7 remained relatively close to their current best-performing observations, indicating greater exploitation of promising regions.

Function 5 was also reasonably close to its current best and continued to target the region that had produced successive improvements.

Functions 3 and 8 selected substantially more distant candidates, suggesting that the uncertainty component of UCB was encouraging greater exploration.

These results supported retaining `κ = 2.5`, as the same acquisition parameter was already producing different exploration–exploitation behaviour according to each function's model.

### 5.3 Function 1 — Maximum-Distance Exploration

Function 1 continued to experience difficulties with its RBF Gaussian Process.

Its fitted length scale remained at the lower bound, and the SVM comparison had not provided sufficiently convincing evidence to guide query selection.

Maximum-distance exploration was therefore retained.

For each candidate point, the Euclidean distance to its nearest existing observation was calculated. The candidate with the greatest minimum distance was selected.

| Metric | Result |
|---|---|
| Selection method | Maximum-distance exploration |
| Selected point | `[0.000045, 0.003062]` |
| Distance to nearest observation | 0.321671 |

The selected point targeted a relatively unexplored region of Function 1's input space.

The objective was to gather additional information that could potentially improve future GP fitting rather than rely on acquisition scores from a model that continued to exhibit limitations.

### 5.4 Function 6 — Linear SVM-Informed Selection

Function 6 required a different approach because its updated RBF Gaussian Process produced effectively identical predictions and uncertainty across the candidate space.

However, the linear SVM had achieved a cross-validation accuracy of approximately 0.821, indicating that it could distinguish between relatively higher- and lower-performing observations.

A linear SVM was therefore fitted using all available Function 6 observations.

The fitted model was used to calculate decision scores for 100,000 candidate points.

The highest-scoring candidate was identified as a potential new query.

| Metric | Result |
|---|---|
| Selection method | Linear SVM |
| Selected point | `[0.013758, 0.147930, 0.445597, 0.994845, 0.013469]` |
| SVM decision score | 5.272052 |
| Distance from current best | 0.402781 |

The selected candidate achieved a strong positive SVM decision score of approximately 5.27.

Its distance of approximately 0.403 from the current best observation suggested that it provided some exploration while remaining within a region strongly classified as higher-performing.

Although the SVM did not predict the continuous output value or provide GP-style uncertainty, its classification results offered a useful alternative where UCB could not meaningfully distinguish between candidate points.

The selected candidate was therefore retained for Function 6.

## 6. Final Validation

Before submission, the eight proposed query points were combined and validated.

The checks confirmed that:

- Every coordinate was within the permitted `[0, 1]` range.
- No proposed query duplicated an existing observation.
- Function 1 retained its maximum-distance exploration selection.
- Functions 2, 3, 4, 5, 7 and 8 retained their highest-UCB selections.
- Function 6 retained its linear SVM-informed selection.

The distance from each query to its nearest existing observation was also calculated.

| Function | Selection Method | Distance to Nearest Observation | In Bounds | Duplicate |
|---|---|---:|---|---|
| 1 | Maximum-distance exploration | 0.321671 | Yes | No |
| 2 | GP + UCB | 0.045156 | Yes | No |
| 3 | GP + UCB | 0.306635 | Yes | No |
| 4 | GP + UCB | 0.137608 | Yes | No |
| 5 | GP + UCB | 0.203041 | Yes | No |
| 6 | Linear SVM | 0.402781 | Yes | No |
| 7 | GP + UCB | 0.161151 | Yes | No |
| 8 | GP + UCB | 0.800486 | Yes | No |

The validation showed that the selections were consistent with their intended strategies.

Functions 2, 4, 5 and 7 remained relatively close to previously sampled regions, while Functions 1, 3, 6 and particularly 8 provided greater exploration.

Function 1 was also confirmed as the intended maximum-distance exploration candidate.

## 7. Final Week 3 Query Selections

The final eight proposed query points were formatted to six decimal places for submission.

| Function | Selection Method | Final Proposed Query |
|---|---|---|
| 1 | Maximum-distance exploration | `[0.000045, 0.003062]` |
| 2 | GP + UCB | `[0.709627, 0.881953]` |
| 3 | GP + UCB | `[0.732035, 0.997300, 0.015653]` |
| 4 | GP + UCB | `[0.391976, 0.424757, 0.494823, 0.464424]` |
| 5 | GP + UCB | `[0.671629, 0.981118, 0.990458, 0.985189]` |
| 6 | Linear SVM | `[0.013758, 0.147930, 0.445597, 0.994845, 0.013469]` |
| 7 | GP + UCB | `[0.021549, 0.313334, 0.360216, 0.141732, 0.295328, 0.686548]` |
| 8 | GP + UCB | `[0.025529, 0.451238, 0.098209, 0.031391, 0.412421, 0.138573, 0.114934, 0.994361]` |

### Submission Format

```text
Function 1: 0.000045-0.003062
Function 2: 0.709627-0.881953
Function 3: 0.732035-0.997300-0.015653
Function 4: 0.391976-0.424757-0.494823-0.464424
Function 5: 0.671629-0.981118-0.990458-0.985189
Function 6: 0.013758-0.147930-0.445597-0.994845-0.013469
Function 7: 0.021549-0.313334-0.360216-0.141732-0.295328-0.686548
Function 8: 0.025529-0.451238-0.098209-0.031391-0.412421-0.138573-0.114934-0.994361
```

## 8. Week 3 Reflection

Week 3 demonstrated the importance of adapting the optimisation strategy according to the behaviour of each function.

The latest observations improved the known maxima for Functions 5 and 6, but also revealed that the Gaussian Process models could change substantially as new information became available.

Function 5 continued to benefit from exploitation of a previously identified promising region, producing another improvement in its highest observed output.

Function 6 presented a more complex situation. Despite its latest query improving the observed maximum, its updated RBF Gaussian Process returned to the lower length-scale bound and produced effectively constant predictions across the candidate space.

The introduction of SVM classification provided an alternative way to investigate the structure of the unknown functions. In particular, Function 6's relatively strong linear SVM performance suggested that classification could identify broadly promising regions even when the GP was unable to guide UCB selection.

However, the SVM approach was not treated as a replacement for Bayesian Optimisation. Its classification scores represented confidence in relative performance categories rather than predictions of continuous function outputs.

The existing GP and UCB methodology therefore remained the preferred approach for functions with informative surrogate models.

Function 1 continued to use maximum-distance exploration because neither its GP diagnostics nor its SVM results provided sufficiently convincing guidance.

Overall, Week 3 combined the existing Bayesian Optimisation approach with supplementary classification analysis, allowing the selection strategy to respond to the different characteristics and modelling difficulties of the eight functions.

In the next round, the returned outputs will be evaluated, the Gaussian Processes will be refitted, and alternative exploration parameters and acquisition functions may be considered to further refine the selection strategy.

---

**Associated code:** `Week_03.ipynb` contains the original Week 3 analysis, SVM comparisons, candidate selection procedures and saved outputs.



