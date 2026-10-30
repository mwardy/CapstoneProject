
# Week 9 — Scaling the Observations and Refining Promising Regions

**Imperial College AI/ML Capstone Project | Module 20**

## Objective

The objective of Week 9 was to investigate how the optimisation process was progressing as additional observations accumulated, with particular attention to whether increasing dataset sizes were producing diminishing returns or changes in model reliability.

Previous rounds had demonstrated that the eight functions responded differently to additional observations. Some continued to achieve substantial improvements, while others appeared to be approaching stable high-performing regions.

This week therefore introduced a more systematic examination of optimisation progress, including running-best outputs and the magnitude of successive improvements.

The Gaussian Process (GP) and neural-network (NN) models were also reassessed to determine whether their predictive reliability was changing as the datasets increased.

Final candidate selections continued to combine acquisition-function recommendations, model validation, historical performance and observed neighbourhood evidence, with greater emphasis on targeted local refinement where promising regions had already been established.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 8 queries were compared with the previous highest observed outputs before being incorporated into their respective datasets.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | 0.000000 | 0.000000 | No improvement |
| 2 | 0.746962 | 0.657120 | 0.746962 | No improvement |
| 3 | -0.010190 | -0.000863 | -0.000863 | New best |
| 4 | 0.186912 | 0.387262 | 0.387262 | New best |
| 5 | 8662.482500 | 8115.434723 | 8662.482500 | No improvement |
| 6 | -0.379740 | -0.274388 | -0.274388 | New best |
| 7 | 2.929370 | 2.922059 | 2.929370 | No improvement |
| 8 | 9.930811 | 9.957032 | 9.957032 | New best |

Four of the eight queries produced new best observations.

**Function 4** achieved the largest improvement relative to its previous best, increasing from approximately 0.1869 to 0.3873. This provided further evidence that continued refinement of the recently identified high-performing region remained productive.

**Function 3** improved from -0.010190 to -0.000863, moving substantially closer to zero. This continued its recent sequence of successful queries.

**Function 6** improved from -0.379740 to -0.274388, representing another meaningful gain following earlier difficulties with its Gaussian Process.

**Function 8** achieved a smaller improvement, increasing from 9.930811 to 9.957032.

The unsuccessful queries were also informative.

Function 7 returned 2.922059, only slightly below its existing best of 2.929370, indicating that the optimisation remained within a strong region despite not improving the maximum.

Function 5 returned 8115.43 after moving slightly away from its previous boundary optimum of 8662.48. Although unsuccessful, the result provided additional evidence about the sensitivity of the objective near the upper boundary.

Function 2 remained below its best, while Function 1's deliberate space-filling exploration again returned effectively zero.

Following these observations, the updated datasets contained between 18 observations for Functions 1 and 2 and 48 observations for Function 8.

## 2. Assess Optimisation Progress as the Datasets Scale

As the datasets increased, the complete optimisation histories were examined to determine whether additional observations were continuing to produce meaningful improvements.

For each function, the running-best output was calculated and the number of improvements recorded.

This provided an indication of how optimisation progress varied with dataset size and dimensionality.

### 2.1 Running-Best Performance

| Function | Observations | Current Best | Number of New Bests | General Behaviour |
|---|---:|---:|---:|---|
| 1 | 18 | ~0.000000 | 3 | Effectively flat |
| 2 | 18 | 0.746962 | 3 | Infrequent improvements |
| 3 | 23 | -0.000863 | 8 | Strong recent progress |
| 4 | 38 | 0.387262 | 11 | Continued substantial improvement |
| 5 | 28 | 8662.482500 | 11 | Strong improvement near upper boundary |
| 6 | 28 | -0.274388 | 5 | Intermittent but improving |
| 7 | 38 | 2.929370 | 6 | Recent stability around best region |
| 8 | 48 | 9.957032 | 9 | Continued but smaller gains |

The running-best trajectories demonstrated that increasing the number of observations had not produced a uniform pattern across the functions.

**Functions 3 and 4** provided the clearest evidence of continued optimisation progress.

Function 3 had improved across its four most recent observations, while Function 4 continued to produce substantial gains and had moved from exclusively negative best outputs to a positive maximum.

Function 5 also demonstrated significant improvement across the optimisation history, particularly in its later observations. However, its latest query failed to improve the maximum, raising the question of whether the upper-boundary region was beginning to show diminishing returns.

Function 7 appeared to be entering a more stable phase. Its recent observations remained close to the best but had not established further improvements.

**Function 8 showed the clearest indication of diminishing returns.** Its best output continued to increase, but the improvements were becoming relatively small as the observed maximum approached 10.

Function 6 had improved intermittently, while Function 2 had produced relatively few improvements overall.

Function 1 remained unusual because its recorded best outputs were effectively zero, meaning that technically higher observations did not necessarily represent meaningful practical progress.

Overall, the results suggested that optimisation performance was highly function-specific rather than following a single scaling relationship.

## 3. Examine the Magnitude of Recent Improvements

To investigate diminishing returns more directly, the size of each improvement in the running-best output was calculated.

Absolute changes were preferred to percentage changes because several functions had negative outputs or crossed zero, making percentage improvements difficult to interpret consistently.

### 3.1 Recent Improvement Patterns

| Function | Recent Improvement Behaviour |
|---|---|
| 1 | Changes remain effectively zero |
| 2 | Only two improvements after the initial best: 0.072209 and 0.135757 |
| 3 | Latest gains increased to 0.005309 and 0.009327 |
| 4 | Latest gains of 0.387342 and 0.200350 |
| 5 | Recent successful improvements ranged from approximately 261.97 to 1834.48 |
| 6 | Recent gains increased from 0.001637 to 0.077134 and 0.105352 |
| 7 | Strong later gains, followed by two observations without improvement |
| 8 | Generally declining improvement sizes, with latest gain of 0.026220 |

The results reinforced the conclusion that diminishing returns were not consistent across all functions.

**Functions 3, 4 and 6** continued to generate meaningful improvements. In particular, recent gains in Functions 3 and 6 had increased rather than diminished.

Function 5 also continued to demonstrate substantial improvement over its recent history, although the latest query had failed to establish a new maximum.

Function 8 provided the strongest evidence of diminishing gains. Its improvement sizes had generally declined from 0.621928 to 0.223504, 0.089263 and 0.019562, before increasing slightly to 0.026220 in the latest round.

Although this was not a perfectly consistent downward trend, the overall reduction in improvement magnitude was clear.

Function 7 may also have been approaching a more stable region, although two unsuccessful observations were insufficient to establish a definitive plateau.

There was no clear evidence of a sudden emergence of new model capabilities simply because the datasets had grown. Any such conclusion would require supporting changes in predictive reliability or optimisation behaviour.

The next step was therefore to examine whether the surrogate models had become more informative with the additional observations.

## 4. Refit and Evaluate the Surrogate Models

The Gaussian Process and neural-network models were updated using the expanded datasets.

The existing modelling configurations were retained to ensure that changes in performance could be interpreted in the context of additional observations rather than unnecessary changes to the methodology.

### 4.1 Gaussian Process Assessment

The RBF Gaussian Processes were refitted and assessed using their optimised length scales, predicted outputs and uncertainty.

**Functions 1 and 2 remained problematic**, with their fitted length scales reaching the lower bound of `1e-05`.

For Function 1, this continued a longstanding issue, with the GP unable to provide convincing evidence of promising unexplored regions.

Function 2 had developed a similar problem in the previous round. Although its model still produced some variation in predictions, its extremely short fitted length scale raised concerns about generalisation.

The remaining functions produced more informative GP behaviour and continued to support acquisition-based candidate selection.

The results demonstrated that adding observations did not automatically resolve surrogate modelling difficulties.

### 4.2 Neural-Network Cross-Validation

The neural-network ensembles were retrained using the updated observations.

As in previous rounds, Leave-One-Out Cross-Validation (LOOCV) was used to evaluate their generalisation performance.

The NN Mean Absolute Error (MAE) was compared with a simple mean-prediction baseline.

| Function | NN LOOCV MAE | Mean Baseline MAE | Beats Baseline |
|---|---:|---:|---|
| 1 | 0.001837 | 0.000995 | No |
| 2 | 0.236724 | 0.241635 | Yes |
| 3 | 0.056881 | 0.053409 | No |
| 4 | 1.385355 | 7.460801 | Yes |
| 5 | 255.307417 | 2324.264043 | Yes |
| 6 | 0.255144 | 0.530835 | Yes |
| 7 | 0.443789 | 0.712291 | Yes |
| 8 | 0.309128 | 0.981238 | Yes |

The neural networks outperformed the baseline for **six of the eight functions**.

Functions 4 and 5 continued to show particularly strong relative validation performance, while Functions 6, 7 and 8 also demonstrated useful predictive structure.

Function 2 narrowly outperformed its baseline, with an MAE of 0.236724 compared with 0.241635.

Although the advantage was small, this was important because Function 2's Gaussian Process remained unreliable. The neural network therefore continued to provide potentially useful alternative guidance.

Functions 1 and 3 failed to outperform their baselines.

Function 3 was particularly interesting because its optimisation performance had improved substantially despite its neural network continuing to fail validation.

This demonstrated that successful query selection did not necessarily indicate that every surrogate model was becoming more accurate.

Overall, the additional observations had not produced a sudden change in neural-network reliability. Model performance continued to vary substantially by function.

## 5. Compare GP Acquisition-Function Candidates

The updated GPs were used to evaluate 100,000 candidate points across the permitted `[0, 1]` search space.

The established acquisition comparison was retained:

- UCB with `κ = 0.5, 1.0, 1.5 and 2.5`.
- Expected Improvement with `ξ = 0%, 1%, 5% and 10%` of each function's observed output standard deviation.

The same settings as the previous round allowed changes in candidate behaviour to be assessed without introducing new acquisition parameters.

### 5.1 Key Acquisition-Function Findings

**Function 1**

All UCB and EI settings selected the same candidate:

`[0.094177, 0.975622]`

The GP predicted approximately -0.000529, with uncertainty of 0.001544.

However, this agreement was not considered reliable because the GP continued to reach the lower length-scale bound.

**Function 2**

Low-κ UCB returned the previously submitted Week 8 point:

`[0.697103, 0.994806]`

Higher κ settings and EI selected a more exploratory candidate:

`[0.403694, 0.391789]`

The exploratory candidate had greater uncertainty but weaker surrounding historical performance.

Given the GP's poor reliability, neither recommendation was accepted automatically.

**Function 3**

Increasing κ produced progressively more exploratory candidates.

The κ = 0.5 candidate remained relatively close to the observed data, while κ = 2.5 selected a point substantially further away.

All EI settings selected the same candidate as UCB with κ = 1.5:

`[0.779394, 0.644459, 0.001936]`

However, the latest successful observations provided a reason to favour more conservative local refinement.

**Function 4**

UCB with κ = 0.5 produced a strong predicted output of approximately 0.736288.

Increasing κ placed progressively greater emphasis on exploration, with κ = 2.5 selecting a much more distant point with a predicted output of approximately -9.678100.

This provided little reason to abandon the recently successful region.

**Function 5**

Lower-κ UCB continued to favour the established upper-boundary region.

Higher κ settings selected more exploratory candidates, but these were substantially further from the strongest observations.

Given the repeated success around the upper boundary, local refinement remained preferable.

**Function 6**

The acquisition comparison supported continued investigation around the current high-performing region.

More exploratory candidates were also considered, but their surrounding observations generally provided weaker evidence than those supporting local refinement.

**Function 7**

Low-κ UCB effectively returned the existing best observation.

This confirmed that the surrogate model continued to identify a promising region, but a novel candidate would be needed before submission.

**Function 8**

More exploitative acquisition settings remained close to the strongest observed region.

Higher-κ recommendations became substantially more exploratory, despite the latest observation producing another new best.

The recent pattern of small improvements suggested that continued targeted refinement was more appropriate than an unsupported move into a distant region.

Overall, the acquisition analysis favoured relatively local selections for most functions, with Functions 1 and 2 requiring separate consideration because of their GP reliability issues.

## 6. Compare Neural-Network and GP Recommendations

The neural-network ensembles were used to generate alternative candidate recommendations.

These were compared with the GP acquisition selections, considering predicted outputs, model disagreement, candidate locations and evidence from nearby observations.

### 6.1 Model Agreement and Reliability

**Function 1**

The GP and NN identified different regions, but neither surrogate had sufficiently strong validation evidence.

This supported retaining model-independent exploration.

**Function 2**

The NN continued to identify a region associated with stronger observations than the exploratory GP alternative.

Its LOOCV performance was only marginally better than the baseline, so the recommendation still required caution. Nevertheless, it provided stronger evidence than the unreliable GP.

**Function 3**

The NN and lower-κ GP candidates remained close to the latest high-performing region.

However, because the NN failed LOOCV, its prediction was treated as supporting evidence rather than a primary basis for selection.

**Function 4**

Both GP and NN recommendations remained associated with the recently successful region.

The latest substantial improvement and strong NN validation supported further local refinement.

**Function 5**

Both surrogate approaches continued to favour the upper-boundary region.

More exploratory GP candidates had weaker historical support, making local boundary refinement the preferred strategy.

**Function 6**

Both models identified promising regions near the latest best observation.

The recent improvement suggested that slightly broader local refinement could be justified, provided it remained supported by model predictions and historical observations.

**Function 7**

Both surrogate approaches continued to identify the established high-performing region.

However, the leading GP recommendation duplicated an existing observation, requiring a new candidate to be selected nearby.

**Function 8**

Both models continued to identify candidates near the highest-performing observations.

The neural network predicted a potential improvement and demonstrated relatively low ensemble disagreement, supporting further local refinement despite the smaller gains recorded recently.

## 7. Assess Candidate Neighbourhoods

The nearest existing observations to each model recommendation were examined to determine whether predicted performance was supported by actual historical results.

The analysis compared neural-network candidates with low- and high-κ GP candidates.

### 7.1 Neighbourhood Findings

| Function | Main Finding |
|---|---|
| 1 | Neither surrogate region has convincing support; outputs remain effectively zero |
| 2 | NN candidate located near stronger observations than exploratory GP candidate |
| 3 | NN and low-κ GP candidates close to recent best observations |
| 4 | Local candidates supported by recent outputs of 0.387262 and 0.186912 |
| 5 | Local upper-boundary region contains outputs above 8000; exploratory GP region much weaker |
| 6 | Local candidates supported by recent best of -0.274388 |
| 7 | Strong local region identified, but leading GP candidate duplicates existing best |
| 8 | NN and low-κ GP candidates supported by the strongest observed neighbourhood |

The neighbourhood evidence strongly favoured locally focused optimisation for Functions 2–8.

**Function 2** provided a clear example of why surrogate reliability and observed results needed to be considered together.

Its neural-network candidate was close to observations producing 0.471844, 0.538996 and the current best of 0.746962.

In contrast, the exploratory GP candidate was surrounded by considerably weaker outputs of 0.214965, -0.013858 and -0.065624.

The NN-supported region therefore appeared more promising.

**Function 4** also favoured local refinement. Its latest two best observations were 0.186912 and 0.387262, while more exploratory candidates were associated with substantially weaker previous results.

**Function 5** demonstrated an even more pronounced difference. The local region contained outputs above 8000, while the exploratory GP candidate was associated with nearby outputs of only approximately 64–233.

There was therefore little empirical evidence supporting a substantial move away from the upper boundary.

**Function 6** favoured continued refinement around its recently improved region, although the combination of GP and NN support suggested that somewhat broader local exploration might be justified.

**Function 7** remained concentrated around a clearly successful region, but a new point was required to avoid duplication.

**Function 8** also favoured local refinement. Although recent gains were becoming smaller, both surrogate models continued to identify candidates associated with its strongest observations.

Function 3 similarly favoured local refinement despite its NN failing LOOCV, because its GP and observed data independently supported the recently successful area.

Function 1 remained the main exception. Neither model could reliably distinguish between candidate regions, and observed neighbourhoods offered little useful evidence.

Overall, the neighbourhood comparison reinforced the conclusion that Functions 2–8 should primarily focus on exploitation or controlled local refinement, while Function 1 remained better suited to space-filling exploration.

## 8. Function-Specific Local Refinement

For functions with established high-performing regions, additional candidates were generated near the current best observations.

This was particularly important where leading model recommendations duplicated existing observations or where further local optimisation was supported by recent improvements.

Different local search radii were considered, allowing the trade-off between predicted performance and candidate novelty to be assessed.

### 8.1 Final Strategy by Function

| Function | Strategy | Main Reasoning |
|---|---|---|
| 1 | Space-filling exploration | GP and NN remain unreliable |
| 2 | NN-supported refinement | Stronger neighbourhood evidence than GP |
| 3 | Fine local refinement | Repeated recent improvements near current best |
| 4 | Fine local refinement | Strong recent gains and cross-model support |
| 5 | Boundary refinement with novelty constraint | Best outputs concentrated near upper boundary |
| 6 | Broader local refinement | Recent improvement and positive GP/NN predictions |
| 7 | Conservative local refinement | Stable high-performing region; avoid duplication |
| 8 | Broader local refinement | Continued improvement despite diminishing gains |

### 8.2 Important Local Search Decisions

**Function 1 — Space-Filling Exploration**

Because neither surrogate provided reliable guidance, the candidate with the greatest distance from existing observations was selected.

Final candidate:

`[0.682178, 0.001903]`

The distance to its nearest existing observation was approximately 0.306755.

This remained a deliberate exploration-focused selection.

**Function 2 — NN-Supported Refinement**

The neural network and observed neighbourhood evidence favoured continued investigation near stronger historical observations.

Final candidate:

`[0.704305, 0.038527]`

The candidate was approximately 0.077676 from its nearest observation, allowing some exploration without relying on the poorly fitted GP.

**Function 3 — Fine Local Refinement**

The recent sequence of improvements justified remaining close to the latest successful region.

Final candidate:

`[0.473523, 0.500295, 0.456235]`

Its nearest-observation distance of approximately 0.010539 reflected a highly local refinement strategy.

**Function 4 — Fine Local Refinement**

The recent substantial improvement supported further exploitation of the same region.

Final candidate:

`[0.388825, 0.366595, 0.350533, 0.420635]`

The nearest-observation distance was approximately 0.016260.

**Function 5 — Boundary Refinement**

The upper-boundary region continued to provide the strongest historical results.

A minimum novelty constraint was retained to prevent selection of an effectively duplicate point.

Final candidate:

`[1.000000, 0.969998, 1.000000, 1.000000]`

The point was approximately 0.030002 from its nearest observation.

**Function 6 — Broader Local Refinement**

Increasing the local search radius produced more optimistic predictions from both surrogate models.

At a radius of 0.10, the GP predicted approximately -0.195871 and the NN approximately -0.229762, both above the current best of -0.274388.

Final candidate:

`[0.440497, 0.372759, 0.525613, 0.791828, 0.000000]`

Its nearest-observation distance was approximately 0.110780.

This represented a broader local move supported by both models rather than an unrestricted exploratory query.

**Function 7 — Conservative Local Refinement**

The more exploratory alternatives offered higher NN predictions but also increased disagreement and weaker GP support.

A smaller local search radius was therefore preferred.

Final candidate:

`[0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374]`

The NN predicted approximately 2.985400, above the current best of 2.929370.

The nearest-observation distance was approximately 0.039502.

**Function 8 — Broader Local Refinement**

Although recent gains were diminishing, the latest query had established another new best.

A broader local candidate was supported by both models, with GP and NN predicted outputs of approximately 9.976275 and 10.009696 respectively.

Both exceeded the existing best of 9.957032.

Final candidate:

`[0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246]`

The nearest-observation distance was approximately 0.190714.

This provided greater novelty while remaining within a locally supported region.

## 9. Final Week 9 Query Selections

The selected candidates were rounded to six decimal places and checked against the updated datasets.

Each candidate was validated for dimensionality, permitted input bounds and duplication against existing observations.

### Final Proposed Queries

| Function | Final Proposed Point | Nearest Observation Distance |
|---|---|---:|
| 1 | `[0.682178, 0.001903]` | 0.306755 |
| 2 | `[0.704305, 0.038527]` | 0.077676 |
| 3 | `[0.473523, 0.500295, 0.456235]` | 0.010539 |
| 4 | `[0.388825, 0.366595, 0.350533, 0.420635]` | 0.016260 |
| 5 | `[1.000000, 0.969998, 1.000000, 1.000000]` | 0.030002 |
| 6 | `[0.440497, 0.372759, 0.525613, 0.791828, 0.000000]` | 0.110780 |
| 7 | `[0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374]` | 0.039502 |
| 8 | `[0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246]` | 0.190714 |

All eight candidates passed the saved validation checks.

Every point had the correct dimensionality, remained within the `[0, 1]` input space and was distinct from all existing observations.

The nearest-observation distances reflected the different optimisation strategies.

Functions 3, 4, 5 and 7 remained relatively close to successful observations, while Functions 6 and 8 made broader local moves.

Function 2 provided an intermediate case, while Function 1 remained the most exploratory selection.

### Submission Format

    Function 1: 0.682178-0.001903
    Function 2: 0.704305-0.038527
    Function 3: 0.473523-0.500295-0.456235
    Function 4: 0.388825-0.366595-0.350533-0.420635
    Function 5: 1.000000-0.969998-1.000000-1.000000
    Function 6: 0.440497-0.372759-0.525613-0.791828-0.000000
    Function 7: 0.199451-0.330731-0.460263-0.255214-0.346569-0.603374
    Function 8: 0.031565-0.175185-0.160043-0.130670-0.742356-0.487463-0.194164-0.622246

## 10. Week 9 Reflection

Week 9 demonstrated that increasing the number of observations did not produce a uniform pattern of optimisation progress across the eight functions.

Four of the latest queries improved the observed maxima, with particularly encouraging results for Functions 3, 4 and 6.

The scaling analysis showed that Functions 3 and 4 continued to improve strongly, while Function 6 had also achieved increasingly meaningful gains after its earlier modelling difficulties.

Function 5 retained a strong high-performing region around the upper boundary, although the latest query failed to improve its maximum.

Function 8 provided the clearest evidence of diminishing returns. Its recent improvements had become smaller, but the continued identification of new best observations suggested that local refinement remained worthwhile.

Function 7 also appeared to be approaching a more stable region, although its latest result remained very close to the best.

The surrogate-model analysis reinforced the importance of assessing reliability separately for each function. Functions 1 and 2 continued to experience Gaussian Process difficulties, while the neural networks outperformed their mean-prediction baselines for six functions.

However, the additional observations had not produced a sudden improvement in model reliability or clear evidence of emergent behaviour.

The candidate neighbourhood analysis provided strong support for local refinement across Functions 2–8. Model recommendations were considered alongside actual historical observations, helping to distinguish promising candidates from exploratory suggestions with little empirical support.

The final selections reflected this evidence. Functions 3, 4, 5 and 7 favoured relatively conservative refinement, while Functions 6 and 8 justified broader local moves supported by both surrogate models.

Function 1 continued to use model-independent space-filling exploration because neither GP nor NN modelling provided sufficiently reliable guidance.

Overall, Week 9 introduced a more systematic assessment of scaling and diminishing returns while reinforcing the value of function-specific optimisation decisions.

The following round will evaluate whether the selected refinements produce further improvements and whether changes in prediction accuracy provide additional evidence about surrogate reliability as the datasets grow.

---

**Associated code:** `Week_09.ipynb`


