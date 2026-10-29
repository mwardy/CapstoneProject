
# Module 19 – Week 8: Bayesian Optimisation

## Methodology

Week 8 continued the Bayesian optimisation of eight unknown functions, incorporating the observations returned from Week 7. The approach remained based on Gaussian Process (GP) regression, Upper Confidence Bound (UCB), Expected Improvement (EI), and neural-network (NN) ensembles.

The workflow comprised eight stages:

1. Incorporate Week 7 observations and assess improvements.
2. Refit GP and NN surrogates and evaluate model reliability.
3. Generate candidates using UCB and EI acquisition functions.
4. Generate NN candidates and compare surrogate recommendations.
5. Quantify GP–NN agreement.
6. Evaluate neighbourhood evidence and investigate Function 1 separately.
7. Select function-specific candidates, including additional local refinement.
8. Validate and format the final submissions.

As in previous weeks, **100,000 candidate points per function** were evaluated. UCB was assessed at κ = 0.5 and 1.5, while EI used ξ = 0% and 5% of the observed output standard deviation. Five-model NN ensembles provided alternative recommendations, with reliability assessed through leave-one-out cross-validation (LOOCV) against a mean-prediction baseline.

Candidate selection considered predicted performance, uncertainty, model reliability, agreement between surrogates, proximity to existing observations and recent optimisation results.

---

## 1. Week 7 Results

The eight Week 7 observations were incorporated and compared with the previous best output for each function.

| Function | Previous best | Week 7 output | Improved? |
|---|---:|---:|---|
| F1 | 0.000000 | 0.000000 | No |
| F2 | 0.746962 | 0.569904 | No |
| F3 | -0.015498 | -0.010190 | Yes |
| F4 | -0.200431 | 0.186912 | Yes |
| F5 | 7121.151955 | 8662.482500 | Yes |
| F6 | -0.379740 | -0.388687 | No |
| F7 | 2.929370 | 2.369289 | No |
| F8 | 9.911249 | 9.930811 | Yes |

**Four of the eight functions achieved new best outputs.**

F4 and F5 showed the largest improvements. F4 increased from -0.200431 to 0.186912, while F5 reached 8662.482500 following the previous week's upper-boundary query.

F3 and F8 also improved, although by smaller margins. F6 remained close to its existing best, while F2 and F7 produced weaker outputs.

F1 again returned effectively zero, continuing the difficulty encountered when modelling this function.

These results supported continued local refinement for several functions, particularly F4 and F5, while reinforcing the need for different strategies where previous recommendations had been less successful.

---

## 2. Surrogate Diagnostics

### 2.1 Gaussian Process Models

The GP models were refitted using the expanded datasets. Fitted RBF length scales and prediction behaviour were examined to identify potential modelling problems.

| Function | RBF length scale | Prediction range | Uncertainty range |
|---|---:|---:|---:|
| F1 | 0.00001 | 0.000560 | 0.001570 |
| F2 | 0.00001 | 0.651312 | 0.244076 |
| F3 | 0.199 | 0.388723 | 0.078255 |
| F4 | 0.416 | 35.821887 | 7.855238 |
| F5 | 0.273 | 7847.668688 | 2510.829993 |
| F6 | 0.581 | 2.303126 | 0.523703 |
| F7 | 0.302 | 3.026197 | 0.818402 |
| F8 | 1.040 | 4.878650 | 0.653663 |

The most important finding was that **both F1 and F2 reached the GP length-scale lower bound of 1e-05**.

F1 had exhibited this behaviour previously, but F2's length scale decreased from 0.0431 in Week 7 to the lower bound in Week 8. This raised concerns about the GP's ability to generalise between observations.

For F2, the maximum predicted output was approximately 0.5696, below the existing best of 0.746962. Its GP recommendations were therefore treated cautiously.

The remaining GP length scales were comparatively stable. F5's decreased from 0.363 to 0.273 following its substantial improvement at the upper boundary.

F4 and F5 exhibited particularly large uncertainty ranges, supporting the continued use of acquisition functions rather than selecting candidates solely from predicted means.

### 2.2 Neural-Network Validation

The five-model NN ensembles were retrained and evaluated using LOOCV.

| Function | Mean training loss | NN LOOCV MAE | Baseline MAE | Beats baseline? |
|---|---:|---:|---:|---|
| F1 | 0.000963 | 0.001704 | 0.001050 | No |
| F2 | 0.010989 | 0.219676 | 0.237993 | Yes |
| F3 | 0.000408 | 0.062478 | 0.051662 | No |
| F4 | 0.000284 | 1.825570 | 7.284354 | Yes |
| F5 | 0.000230 | 230.163312 | 2049.255132 | Yes |
| F6 | 0.000370 | 0.241315 | 0.513894 | Yes |
| F7 | 0.000338 | 0.399610 | 0.639930 | Yes |
| F8 | 0.000381 | 0.354980 | 0.962395 | Yes |

Six of the eight NN ensembles outperformed the mean-prediction baseline.

F4, F5, F6, F7 and F8 demonstrated useful validation performance, supporting the use of their NN recommendations as additional evidence during candidate selection.

F2 also passed validation, although narrowly. Its NN LOOCV MAE was 0.219676 compared with a baseline of 0.237993. This was particularly relevant because its GP length scale had reached the lower bound.

F1 and F3 again failed to outperform their baselines, so their NN predictions were given limited weight.

Although most networks achieved low training losses, these were not interpreted as proof of generalisation. LOOCV remained the primary measure of NN reliability.

---

## 3. GP Acquisition Candidates

The refitted GP models were used to generate candidate points with UCB and EI.

Two UCB settings (κ = 0.5 and 1.5) and two EI settings (ξ = 0% and 5%) were compared. Each candidate was assessed using predicted output, uncertainty, acquisition behaviour and distance from existing observations.

| Function | Main GP findings |
|---|---|
| F1 | All four acquisition settings selected [0.094177, 0.975622], although the GP remained unreliable. |
| F2 | UCB κ=0.5 returned an existing observation. UCB κ=1.5 and both EI settings selected [0.403694, 0.391789]. |
| F3 | UCB κ=0.5 favoured local refinement, while higher exploration settings moved further from existing observations. |
| F4 | Both UCB settings remained near the new best. EI selected a distant, uncertain region with poor predicted output. |
| F5 | Both UCB settings favoured the successful upper-boundary region, while EI preferred distant exploration. |
| F6 | UCB κ=0.5 predicted a small improvement in a relatively local region. |
| F7 | UCB κ=0.5 returned the existing best. Higher-exploration settings agreed on a novel alternative. |
| F8 | UCB κ=0.5 remained relatively local and predicted a small improvement; other settings explored further away. |

The strongest evidence for local refinement came from F4 and F5, where the previous week's improvements supported continued exploitation of successful regions.

F6 and F8 also had relatively promising local UCB recommendations.

For F1 and F2, however, acquisition results were less reliable because their GP length scales had reached the lower bound.

The results reinforced the importance of evaluating acquisition recommendations alongside observed performance and model diagnostics rather than automatically selecting the highest acquisition score.

---

## 4. Neural-Network Candidates

The NN ensembles independently selected candidates from the same 100,000-point search spaces.

Candidate selection used the highest ensemble mean prediction, with model disagreement and distance from existing observations recorded as supporting diagnostics.

| Function | NN findings |
|---|---|
| F1 | Selected [0.724967, 0.999094], predicting 0.001008. The NN failed LOOCV. |
| F2 | Selected [0.697103, 0.994806], predicting 0.696942. The NN passed LOOCV. |
| F3 | Selected [0.437451, 0.559044, 0.487388], near the current best, but failed LOOCV. |
| F4 | Returned the existing best observation, supporting the successful region but not providing a novel query. |
| F5 | Selected [0.995347, 0.980915, 0.954873, 0.970449], matching both GP UCB candidates. |
| F6 | Selected a nearby candidate predicting -0.345084, above the current best of -0.379740. |
| F7 | Returned the existing best observation, requiring a novel local alternative. |
| F8 | Selected a more exploratory candidate predicting 10.207430, above the existing best of 9.930811. |

The strongest cross-model support emerged for F4, F5 and F6.

F5 was particularly notable because the NN and both UCB settings independently selected the same novel candidate near the successful upper boundary.

F4 and F7 also showed strong preferences for their established best regions, although their NN recommendations duplicated existing observations.

F2's NN provided an important alternative to its unreliable GP. F8's NN predicted an improvement, but its more exploratory location required further comparison against the locally focused GP recommendation.

---

## 5. GP–NN Candidate Comparison

Euclidean distances were calculated between each NN recommendation and the four GP acquisition candidates.

| Function | UCB κ=0.5 | UCB κ=1.5 | EI ξ=0% | EI ξ=5% |
|---|---:|---:|---:|---:|
| F1 | 0.6312 | 0.6312 | 0.6312 | 0.6312 |
| F2 | 0.8368 | 0.6706 | 0.6706 | 0.6706 |
| F3 | 0.0751 | 0.5999 | 0.5790 | 0.5999 |
| F4 | 0.0419 | 0.0591 | 0.9254 | 0.9254 |
| F5 | 0.0000 | 0.0000 | 1.3683 | 1.3683 |
| F6 | 0.0920 | 0.2137 | 0.1762 | 0.2137 |
| F7 | 0.0000 | 0.2220 | 0.2220 | 0.2220 |
| F8 | 0.3396 | 1.0053 | 1.0736 | 1.0736 |

F4–F7 demonstrated the strongest agreement around established high-performing regions.

F5 showed exact agreement between the NN and both UCB recommendations on a novel candidate. F6 also showed close agreement between the validated NN and the lower-κ UCB recommendation.

For F4, the NN returned the existing best but remained close to the local GP candidates. F7 showed exact agreement on an existing observation, making further local search necessary.

F3 demonstrated local agreement, although its NN failed validation.

F1 and F2 exhibited substantial disagreement, while F8 showed moderate disagreement with the closest GP candidate.

**Cross-model agreement was treated as supporting evidence rather than proof of candidate quality.** Agreement was less informative where models failed validation or simply returned an existing observation.

---

## 6. Neighbourhood Evidence

The three nearest observed outputs to leading GP and NN candidates were examined to assess whether recommendations were supported by historical performance.

| Function | Neighbourhood findings |
|---|---|
| F1 | Both GP and NN candidate regions had nearby outputs approximately equal to zero. |
| F2 | NN region: 0.611205, 0.366010, 0.507418. Exploratory GP region: 0.214965, -0.013858, -0.065624. |
| F3 | Local UCB region: -0.010190, -0.016451, -0.020167. |
| F4 | Local UCB: 0.186912, -0.200431, -0.300645. Exploratory EI: -16.507159, -7.966775, -6.702089. |
| F5 | Local UCB/NN: 8662.482500, 6691.931674, 7121.151955. Exploratory EI: 64.443440, 78.434389, 9.972332. |
| F6 | Local UCB: -0.379740, -0.388687, -0.458511. |
| F7 | Leading region: 2.929370, 2.536939, 2.369289. |
| F8 | Local UCB: 9.930811, 9.821986, 9.840530. NN: 9.930811, 9.911249, 9.821986. |

The strongest evidence for continued local refinement was in F4 and F5. Both had recently achieved substantial improvements, and their local candidates were surrounded by much stronger observed outputs than the exploratory EI alternatives.

F2's neighbourhood results favoured the NN recommendation over the exploratory GP candidate. This was particularly relevant because the NN passed LOOCV while the GP was unreliable.

F3, F6 and F7 also showed useful local support.

For F8, both the NN and low-κ UCB regions had strong nearby observations. The GP recommendation was more local and was preferred given the recent successful refinement.

F1 remained unresolved because neither model's recommended region had meaningful supporting observations.

### 6.1 Additional Function 1 Investigation

F1 continued to present the greatest modelling difficulty.

Its GP length scale again reached the lower bound, its NN failed LOOCV, and the models disagreed substantially about the next query.

To investigate whether the GP kernel was responsible, alternative configurations were evaluated using LOOCV.

| GP kernel | LOOCV MAE | Baseline MAE | Beats baseline? |
|---|---:|---:|---|
| RBF | 0.001099 | 0.001050 | No |
| Matérn 3/2 | 0.001106 | 0.001050 | No |
| Matérn 5/2 | 0.001107 | 0.001050 | No |
| RBF + WhiteKernel | 0.001099 | 0.001050 | No |

None of the alternative kernels outperformed the mean-prediction baseline.

The separate RBF fit produced a length scale of approximately 0.00409. Although above the lower bound, this remained very small relative to the [0,1] search space.

Adding a WhiteKernel noise component also failed to improve validation performance.

These results suggested that F1's modelling difficulty was not resolved by changing the GP kernel. With neither GP nor NN demonstrating reliable predictive performance, a model-independent exploration strategy was investigated.

### 6.2 Function 1 Space-Filling Search

A space-filling search was performed across 100,000 points in the two-dimensional [0,1] search space.

Each candidate was ranked by its distance from the nearest existing observation. Five leading points were retained, requiring a minimum separation of 0.10 between candidates.

| Rank | Candidate | Nearest-observation distance |
|---|---|---:|
| 1 | [0.998294, 0.000709] | 0.307658 |
| 2 | [0.684654, 0.001334] | 0.305975 |
| 3 | [0.006886, 0.695109] | 0.301267 |
| 4 | [0.786713, 0.000852] | 0.269276 |
| 5 | [0.898148, 0.005297] | 0.265794 |

The strongest space-filling candidates were substantially more novel than the GP and NN recommendations, whose nearest-observation distances were approximately 0.0953 and 0.1442 respectively.

The three leading space-filling candidates were then compared using GP and NN predictions.

| Candidate | GP prediction | NN prediction |
|---|---:|---:|
| [0.998294, 0.000709] | -0.000560 | 0.000593 |
| [0.684654, 0.001334] | -0.000560 | 0.000432 |
| [0.006886, 0.695109] | -0.000560 | -0.000247 |

All three candidates had nearby observed outputs approximately equal to zero. The GP assigned essentially identical predictions, providing little differentiation.

The NN weakly favoured the first candidate, although its poor validation performance limited confidence in this prediction.

**[0.998294, 0.000709] was selected primarily because it maximised distance from existing observations.**

This represented deliberate exploration rather than a claim that the selected point was likely to outperform the current best.

---

## 7. Final Candidate Selection

The final selections combined recent optimisation performance, GP and NN reliability, acquisition results, model agreement, neighbourhood evidence and candidate novelty.

Additional local searches were performed for F4, F5 and F7 because their strongest model recommendations were concentrated around existing high-performing observations.

| Function | Selected approach | Reasoning |
|---|---|---|
| F1 | Space-filling exploration | Both surrogates failed validation; alternative kernels offered no improvement. |
| F2 | NN-supported selection | GP unreliable; NN passed LOOCV and had stronger neighbourhood evidence. |
| F3 | Local GP refinement | Recent improvement and strong nearby outputs; NN failed LOOCV. |
| F4 | Local refinement | Substantial new best and strong GP–NN support around the successful region. |
| F5 | Boundary refinement | Major improvement at [1,1,1,1], with strong support for continued boundary exploration. |
| F6 | Local GP/NN refinement | GP and validated NN favoured nearby high-performing regions. |
| F7 | Novel local refinement | Models favoured an existing best observation, requiring a new nearby point. |
| F8 | Local UCB refinement | Recent improvement and strong neighbourhood evidence favoured a conservative continuation. |

### 7.1 Additional Local Refinement

For F4, local search identified:

`[0.386248, 0.381010, 0.343907, 0.423092]`

This candidate was approximately 0.0141 from the current best, with both models supporting continued refinement of the region.

For F7, local search identified:

`[0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025]`

The candidate was approximately 0.0479 from the current best. GP and NN predictions were 2.920647 and 2.936110 respectively.

For F5, the initial local search produced:

`[1.000000, 1.000000, 0.999995, 1.000000]`

Although technically novel, this was only 0.000005 from the existing best observation `[1,1,1,1]`.

A **minimum nearest-observation distance of 0.03** was therefore imposed to avoid an almost identical query.

The resulting candidate was:

`[1.000000, 1.000000, 0.970000, 1.000000]`

The GP predicted 8391.49 and the NN predicted 8342.42, both below the existing best of 8662.48 but still supporting the high-performing boundary region.

This novelty constraint provided a more informative query while maintaining the locally focused strategy.

### 7.2 Final Selections

| Function | Final candidate | Selection rationale |
|---|---|---|
| F1 | [0.998294, 0.000709] | Maximises space-filling coverage where surrogates are unreliable. |
| F2 | [0.697103, 0.994806] | Validated NN and stronger neighbourhood evidence than GP alternatives. |
| F3 | [0.463879, 0.497888, 0.452731] | Local UCB κ=0.5 refinement after a new best. |
| F4 | [0.386248, 0.381010, 0.343907, 0.423092] | Novel local refinement supported by both models. |
| F5 | [1.000000, 1.000000, 0.970000, 1.000000] | Boundary refinement with a 0.03 novelty constraint. |
| F6 | [0.426973, 0.384648, 0.468006, 0.752822, 0.084308] | Local UCB κ=0.5 selection supported by NN and neighbourhood evidence. |
| F7 | [0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025] | Novel local search near the current best. |
| F8 | [0.026226, 0.080912, 0.128250, 0.056836, 0.748250, 0.399531, 0.253527, 0.720750] | Local UCB κ=0.5 continuation after a new best. |

---

## 8. Validation and Submission

The final candidates were checked for:

- Correct dimensionality for each function.
- Input values within the permitted [0,1] bounds.
- No duplicate observations.
- Distance from existing observations.
- Consistent formatting to six decimal places.

All eight candidates passed the validation checks.

Nearest-observation distances ranged from approximately 0.0141 for F4 to 0.3077 for F1, reflecting the different exploitation and exploration strategies.

### Final Week 8 Submission Points

```text
Function 1: 0.998294-0.000709
Function 2: 0.697103-0.994806
Function 3: 0.463879-0.497888-0.452731
Function 4: 0.386248-0.381010-0.343907-0.423092
Function 5: 1.000000-1.000000-0.970000-1.000000
Function 6: 0.426973-0.384648-0.468006-0.752822-0.084308
Function 7: 0.157801-0.305871-0.468629-0.230835-0.339422-0.619025
Function 8: 0.026226-0.080912-0.128250-0.056836-0.748250-0.399531-0.253527-0.720750
```

These points match the final selections produced by the Week 8 notebook.

---

## Reflection

Week 8 demonstrated the importance of adapting the exploration–exploitation balance to the reliability of each surrogate and the performance of previous observations.

The strongest evidence supported continued local refinement for F4 and F5, where substantial recent improvements, strong nearby observations and agreement between the GP and NN models pointed towards established high-performing regions. The additional novelty constraint for F5 was particularly useful, preventing the submission of a point almost identical to the existing best.

F1 required a different approach. Both surrogates failed validation, and alternative GP kernels did not resolve the problem. Space-filling exploration therefore provided a more defensible way to gather information than relying on unreliable predicted optima.

F2 also illustrated the value of combining different forms of evidence. Its GP became unreliable, but the NN continued to outperform the baseline and identified a region with stronger observed support.

More generally, the results showed that cross-model agreement can strengthen confidence in a region but does not guarantee improvement, particularly when both models favour an existing observation.

One methodological limitation remains: the NN architecture used two hidden layers of 20 neurons in Weeks 7–8, compared with 16 neurons in earlier weeks. Consequently, changes in NN performance across the full project cannot be attributed solely to additional observations. This distinction should be retained when interpreting results across weeks.

Overall, Week 8 strengthened the function-specific selection process by combining model validation, acquisition behaviour, neighbourhood evidence and explicit novelty checks, while introducing a model-independent exploration strategy for the function where surrogate predictions remained least reliable.


