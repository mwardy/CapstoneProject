
# Module 20 – Week 9: Bayesian Optimisation

## Methodology

Week 9 continued the Bayesian optimisation of eight unknown functions using the observations returned from Week 8. The established workflow combined Gaussian Process (GP) regression, Upper Confidence Bound (UCB), Expected Improvement (EI), neural-network (NN) ensembles, cross-validation and observed neighbourhood evidence. This week also examined whether optimisation gains were diminishing as the datasets grew.

The workflow comprised eight stages:

1. Incorporate Week 8 results and compare them with previous best outputs.
2. Examine running-best trajectories and the sizes of recent improvements.
3. Refit GP and NN models and reassess predictive reliability.
4. Generate GP acquisition candidates and compare exploration settings.
5. Generate NN candidates and evaluate ensemble disagreement.
6. Compare GP–NN recommendations and nearby observed outputs.
7. Conduct function-specific local or exploratory searches and select candidates.
8. Validate dimensionality, bounds, novelty and submission formatting.

For GP acquisition comparisons, **100,000 candidate points per function** were evaluated using UCB κ = 0.5, 1.0, 1.5 and 2.5, and EI ξ = 0%, 1%, 5% and 10% of the observed output standard deviation. Local refinement examined **50,000 candidates** around current best observations at radii of 0.03, 0.05 and 0.10. Final selections considered predictions, uncertainty, validation performance, neighbourhood evidence and novelty rather than acquisition scores alone.

**Implementation consistency.** The uploaded CSVs already contained the Week 8 observations, so the revised notebook checks for existing observations before appending them. This prevents duplicate rows without changing the intended dataset. Week 9 uses NN hidden layers of **(50, 50)** rather than **(20, 20)** in Weeks 7–8. Consequently, differences in NN performance between weeks cannot be attributed solely to additional observations. The original Week 9 architecture and calculations were preserved.

---

## 1. Week 8 Results

The eight returned Week 8 outputs were compared with the previous best for each function.

| Function | Previous best | Week 8 output | Improved? |
|---|---:|---:|---|
| F1 | 0.000000 | 0.000000 | No |
| F2 | 0.746962 | 0.657120 | No |
| F3 | -0.010190 | -0.000863 | Yes |
| F4 | 0.186912 | 0.387262 | Yes |
| F5 | 8662.482500 | 8115.434723 | No |
| F6 | -0.379740 | -0.274388 | Yes |
| F7 | 2.929370 | 2.922059 | No |
| F8 | 9.930811 | 9.957032 | Yes |

**Four of the eight functions established new best outputs.**

F4 produced the clearest improvement, increasing from 0.186912 to 0.387262. This was particularly encouraging because the previous week had also produced a substantial improvement in the same region.

F3 improved from -0.010190 to -0.000863, moving considerably closer to zero. F6 also achieved a meaningful improvement of approximately 0.105352, while F8 increased from 9.930811 to 9.957032.

The remaining functions provided useful information despite not improving. F7 returned 2.922059, only slightly below its best of 2.929370, reinforcing the strength of its current region. F5 returned 8115.434723 after moving approximately 0.03 away from the boundary optimum, remaining high but below the existing best of 8662.482500.

F2 remained below its previous best, while F1's space-filling exploration again produced an output effectively equal to zero.

These results supported continued local refinement for F3, F4 and F6, while suggesting that F7 and F8 might be entering periods of smaller incremental gains.

The updated datasets contained between **18 observations for F1 and F2** and **48 observations for F8**.

---

## 2. Optimisation Progress and Diminishing Returns

### 2.1 Running-Best Performance

To assess how optimisation performance changed as observations accumulated, the running best was calculated for each function.

| Function | Observations | Current best | New bests recorded |
|---|---:|---:|---:|
| F1 | 18 | ~0.000000 | 3 |
| F2 | 18 | 0.746962 | 3 |
| F3 | 23 | -0.000863 | 8 |
| F4 | 38 | 0.387262 | 11 |
| F5 | 28 | 8662.482500 | 11 |
| F6 | 28 | -0.274388 | 5 |
| F7 | 38 | 2.929370 | 6 |
| F8 | 48 | 9.957032 | 9 |

The running-best trajectories showed substantial differences between functions.

F3 and F4 demonstrated the strongest evidence of continued optimisation progress. F3 established new bests across its four most recent observations, while F4 continued improving substantially and had moved from negative outputs to a positive best of 0.387262.

F5 also showed strong later-stage progress, particularly across observations 23–27. However, the most recent query failed to improve on its boundary optimum.

F6 had improved intermittently but achieved a meaningful new best in Week 8. F7 showed strong later-stage improvement followed by two observations without a new best, although its latest output remained very close to the optimum.

F8 continued improving but with generally smaller gains as its observed best approached 10. This provided the clearest initial evidence of diminishing returns.

F2 had achieved only three recorded bests across 18 observations, indicating intermittent progress rather than a clear trend towards smaller improvements.

F1 remained effectively flat. Although three observations were technically recorded as new bests, their values were extremely close to zero and did not represent meaningful progress at the scale of the objective.

### 2.2 Size of Recent Improvements

The absolute size of improvements in the running best was also examined. Absolute rather than percentage changes were used because several functions contained negative outputs or crossed zero.

| Function | Recent improvement pattern |
|---|---|
| F1 | Recorded gains remain effectively zero. |
| F2 | Only two improvements after the initial best, of 0.072209 and 0.135757. |
| F3 | Recent gains varied between 0.000953 and 0.014668, with the latest two increasing to 0.005309 and 0.009327. |
| F4 | Latest two improvements of 0.387342 and 0.200350 remained substantial. |
| F5 | Five recent improvements ranged from 261.970 to 1834.482, with the latest new best adding 1541.331. |
| F6 | Recent improvements increased from 0.001637 to 0.077134 and 0.105352. |
| F7 | Later gains of 0.188939, 0.397388 and 0.392431 were followed by two unsuccessful queries. |
| F8 | Recent gains generally declined from 0.621928 to 0.223504, 0.089263 and 0.019562, followed by 0.026220. |

The improvement-size analysis reinforced the conclusion that diminishing returns were function-specific.

F3, F4 and F6 continued producing meaningful improvements, with recent gains in F3 and F6 actually increasing. F5 also showed substantial progress, despite its most recent unsuccessful query.

F8 showed the clearest reduction in improvement size. Its latest improvement of 0.026220 was slightly larger than the preceding gain of 0.019562, so the decline was not perfectly consistent, but recent gains were generally much smaller than earlier improvements.

F7 may also have been entering a more stable phase, although two unsuccessful observations were insufficient to establish that the region had plateaued.

Overall, there was no clear evidence of a general scaling relationship or sudden emergent behaviour across the eight functions. Additional observations were producing different returns depending on the objective and the regions being explored.

---

## 3. Surrogate Diagnostics

### 3.1 Gaussian Process Models

The GP models were refitted using the expanded datasets, and their fitted RBF length scales and log marginal likelihoods were examined.

| Function | RBF length scale | Log marginal likelihood |
|---|---:|---:|
| F1 | 0.00001 | -25.540894 |
| F2 | 0.00001 | -25.540894 |
| F3 | 0.201 | -25.852938 |
| F4 | 0.313 | -23.818726 |
| F5 | 0.303 | -23.970400 |
| F6 | 0.527 | -22.245983 |
| F7 | 0.323 | -34.151061 |
| F8 | 1.050 | -21.138922 |

**F1 and F2 both remained at the GP length-scale lower bound of 1e-05.**

This was particularly important because the additional observations had not resolved the models' difficulty in identifying a useful smooth relationship between inputs and outputs.

Their GP predictions therefore continued to receive limited weight during candidate selection.

The remaining functions retained substantially larger length scales, ranging from approximately 0.201 for F3 to 1.050 for F8.

F4's GP remained usable despite the substantial recent changes in its observed optimum. F5, F6 and F7 also retained length scales that supported further GP-based candidate analysis.

F8 had the largest fitted length scale, consistent with the smoother behaviour represented by its surrogate model.

There was no obvious sudden change in GP behaviour as the datasets increased. F1 and F2 remained problematic, while the remaining functions continued to provide useful model-based evidence.

### 3.2 Neural-Network Validation

The NN models were retrained and evaluated using leave-one-out cross-validation (LOOCV), comparing mean absolute error against a mean-prediction baseline.

| Function | NN LOOCV MAE | Baseline MAE | Beats baseline? |
|---|---:|---:|---|
| F1 | 0.001837 | 0.000995 | No |
| F2 | 0.236724 | 0.241635 | Yes |
| F3 | 0.056881 | 0.053409 | No |
| F4 | 1.385355 | 7.460801 | Yes |
| F5 | 255.307417 | 2324.264043 | Yes |
| F6 | 0.255144 | 0.530835 | Yes |
| F7 | 0.443789 | 0.712291 | Yes |
| F8 | 0.309128 | 0.981238 | Yes |

Six of the eight NN models outperformed the baseline.

F4 and F5 demonstrated particularly strong validation performance relative to their baselines, while F6, F7 and F8 also showed useful predictive performance.

F2 continued to outperform its baseline, although only marginally. Given its unreliable GP, this made the NN a potentially more useful source of evidence, but its recommendations still required careful assessment.

F1 and F3 failed validation. F1 therefore remained unreliable across both surrogate approaches, while F3's continued optimisation success was not accompanied by sufficiently strong NN generalisation.

The results did not provide convincing evidence of a sudden improvement in model reliability as the datasets grew.

One limitation is that Week 9 used a `(50, 50)` NN architecture rather than the `(20, 20)` architecture used in Weeks 7–8. Consequently, changes in NN validation performance cannot be attributed solely to additional observations. The LOOCV diagnostic also evaluated an individual NN fitting procedure rather than the complete five-model prediction ensemble.

---

## 4. GP Acquisition Candidates

The updated GP models generated candidate recommendations using UCB and EI.

UCB was evaluated at κ = 0.5, 1.0, 1.5 and 2.5, while EI used ξ = 0%, 1%, 5% and 10% of the observed output standard deviation.

| Function | Main GP findings |
|---|---|
| F1 | All acquisition settings selected [0.094177, 0.975622], despite the unreliable GP. |
| F2 | Low-κ UCB returned the existing Week 8 point; higher-κ UCB and EI favoured [0.403694, 0.391789]. |
| F3 | Increasing κ produced progressively more exploratory candidates; EI selected the κ=1.5 region. |
| F4 | UCB κ=0.5 predicted 0.736288, while higher κ values increasingly favoured uncertainty. |
| F5 | Low-κ UCB favoured the upper boundary; high-κ UCB and EI selected a distant exploratory region. |
| F6 | Low-κ UCB predicted -0.248253; EI ξ=0% predicted -0.220587. |
| F7 | Low-κ UCB returned the existing best; EI favoured a more exploratory candidate. |
| F8 | Low-κ UCB predicted 9.967635, slightly above the current best, while higher κ values favoured exploration. |

F3, F4, F6 and F8 showed the expected progression towards more exploratory candidates as κ increased.

F4 and F6 offered particularly promising local opportunities. F4's low-κ candidate predicted 0.736288 compared with the current best of 0.387262, while F6 had multiple relatively local candidates predicted above its current best of -0.274388.

F8 also retained potential for improvement, although the predicted gain from its local UCB candidate was relatively small.

F5 showed a pronounced distinction between exploitation and exploration. Low-κ UCB remained near the successful upper-boundary region, while high-κ UCB and EI selected `[0.019672, 0.003040, 0.043324, 0.976161]`, approximately 0.686332 from its nearest observation.

That exploratory candidate had a predicted output of only 1607.436, substantially below the existing best of 8662.482500, despite its high uncertainty.

For F1 and F2, the GP length-scale problems continued to limit confidence in acquisition recommendations.

The acquisition analysis therefore identified possible regions for exploitation and exploration without determining the final submissions independently.

---

## 5. Neural-Network Candidates

The NN ensembles generated independent recommendations from the candidate search spaces.

Predicted outputs, ensemble disagreement and nearest-observation distances were recorded for each recommendation.

| Function | NN candidate | Predicted output | NN disagreement | Nearest distance |
|---|---|---:|---:|---:|
| F1 | [0.517917, 0.000290] | 0.001946 | 0.001331 | 0.182315 |
| F2 | [0.702517, 0.009487] | 0.831771 | 0.431009 | 0.106007 |
| F3 | [0.465542, 0.513952, 0.462107] | -0.001013 | 0.000717 | 0.018674 |
| F4 | [0.386457, 0.364301, 0.315578, 0.423391] | 0.173431 | 0.101814 | 0.032892 |
| F5 | [0.996673, 0.749297, 0.983441, 0.989318] | 8132.577652 | 837.663492 | 0.247618 |
| F6 | [0.476200, 0.401879, 0.450627, 0.837966, 0.006748] | -0.248980 | 0.042060 | 0.127622 |
| F7 | [0.150161, 0.338209, 0.511025, 0.190914, 0.382394, 0.542431] | 2.893229 | 0.055542 | 0.110492 |
| F8 | [0.012982, 0.180599, 0.189901, 0.256228, 0.760257, 0.471612, 0.155487, 0.557332] | 10.056825 | 0.027228 | 0.234486 |

F2's NN recommendation predicted 0.831771, above the current best of 0.746962. However, ensemble disagreement was relatively high at 0.431009, so the apparent improvement required caution.

F4's NN selected a candidate close to the established successful region, although its predicted output remained below the current best. This provided regional support rather than a direct prediction of improvement.

F5's NN also favoured the upper part of the search space, but its disagreement was substantial and its predicted output remained below the existing optimum.

F6 provided stronger evidence of continued improvement. Its validated NN predicted -0.248980, above the current best of -0.274388, with relatively low disagreement.

F7 remained focused on its established high-performing region, with low disagreement but a predicted output slightly below the existing best.

F8's NN predicted 10.056825, above the current best of 9.957032, with low disagreement of 0.027228. This suggested that further improvement might still be possible despite the smaller recent gains.

F1 and F3 recommendations were given limited weight because their NN models failed LOOCV validation.

---

## 6. GP–NN Comparison and Neighbourhood Evidence

### 6.1 GP–NN Agreement

Euclidean distances were calculated between each NN recommendation and the GP acquisition candidates.

| Function | Closest GP method | GP–NN distance |
|---|---|---:|
| F1 | UCB κ=0.5 | 1.063404 |
| F2 | UCB κ=1.5 | 0.485232 |
| F3 | UCB κ=0.5 | 0.058180 |
| F4 | UCB κ=0.5 | 0.129899 |
| F5 | UCB κ=1.5 | 0.221396 |
| F6 | EI ξ=0% | 0.193602 |
| F7 | UCB κ=0.5 | 0.125569 |
| F8 | UCB κ=0.5 | 0.243999 |

F3 showed the closest numerical agreement, although its NN failed validation, limiting the significance of that agreement.

F4 and F7 demonstrated relatively strong regional agreement between the validated NN and low-κ GP candidates.

F6 also showed moderate agreement, consistent with both models identifying potential for further improvement.

F5's NN remained reasonably close to the locally focused UCB candidates but was much further from the exploratory EI and high-κ recommendations.

F8 showed moderate agreement with low-κ UCB, while the more exploratory GP candidates diverged substantially from the NN recommendation.

F1 and F2 showed considerable disagreement. F1 remained particularly problematic because neither surrogate provided reliable predictive guidance. For F2, the validated NN offered more useful evidence than the GP, although its ensemble disagreement remained high.

Cross-model agreement was therefore treated as supporting evidence rather than proof that a candidate would improve the objective.

### 6.2 Neighbourhood Evidence

The observed outputs nearest to the main GP and NN recommendations were examined to assess whether the candidate regions had historical support.

| Function | Neighbourhood findings |
|---|---|
| F1 | Both GP and NN regions had nearby outputs effectively equal to zero. |
| F2 | NN region included 0.471844, 0.538996 and 0.746962; exploratory GP region had substantially weaker observations. |
| F3 | Local candidates were near outputs of -0.000863, -0.010190 and -0.016451. |
| F4 | Local region included 0.387262 and 0.186912; exploratory GP region had much poorer outputs. |
| F5 | Upper-boundary region included outputs above 8000; exploratory GP region had nearby outputs around 64–233. |
| F6 | Local candidates were supported by -0.274388, -0.388687 and -0.379740. |
| F7 | NN region included 2.922059, 2.929370 and 2.536939. |
| F8 | NN and low-κ GP regions were close to observations of 9.930811, 9.911249 and 9.957032. |

The neighbourhood evidence strongly favoured local refinement for F3–F8.

F4 and F6 had recently achieved new bests, and their locally focused recommendations remained associated with the strongest observed regions.

F5 showed the most pronounced difference between local and exploratory alternatives. Its established boundary region contained outputs above 8000, while the exploratory GP region had much weaker nearby observations.

F7 also remained strongly concentrated around its established best, although the GP's lowest-κ recommendation effectively duplicated an existing observation.

F8 continued to show strong local support despite smaller recent improvements. Both surrogate approaches identified regions near the strongest observed outputs.

F2's NN candidate was supported by a stronger neighbourhood than the exploratory GP alternative, reinforcing the decision to rely more heavily on NN evidence.

F1 remained unresolved because neither surrogate's recommended region had meaningful neighbourhood support.

Overall, the evidence favoured controlled local refinement for F2–F8 and model-independent exploration for F1.

---

## 7. Function-Specific Candidate Selection

Candidate selection combined recent optimisation performance, surrogate reliability, acquisition behaviour, model agreement, neighbourhood evidence and novelty.

Additional searches were performed where the strongest recommendations were close to existing observations or where different refinement distances needed to be compared.

### 7.1 Multi-Radius Local Refinement

For F3–F8, 50,000 local candidates were generated around the current best at radii of 0.03, 0.05 and 0.10.

GP and NN predictions were standardised before being combined, allowing the local searches to compare model recommendations without differences in output scale affecting the selection score.

| Function | Multi-radius findings |
|---|---|
| F3 | All radii converged very close to the current best; GP predictions were approximately -0.00017 to -0.00024. |
| F4 | All radii converged on a similar local region; GP predictions ranged from 0.531330 to 0.563957. |
| F5 | All radii selected points effectively identical to [1,1,1,1], with nearest distances around 0.0001. |
| F6 | Larger radii improved both GP and NN predictions, with the strongest results at 0.10. |
| F7 | All radii remained promising; the larger-radius NN prediction improved but disagreement increased. |
| F8 | Both models predicted improvement at all radii, with greater novelty at 0.10. |

F3 and F4 showed strong convergence across search radii. This supported relatively conservative local refinement around their recent successful observations.

F6 behaved differently. Increasing the radius improved GP predictions from -0.235263 to -0.195871 and NN predictions from -0.253795 to -0.229762. This supported a broader local move.

F7's larger-radius candidate produced a stronger NN prediction but also increased disagreement and weakened GP support. A more conservative candidate was therefore preferred.

F8 showed continued improvement potential across all three radii. The 0.10 candidate was particularly interesting because both models predicted outputs above the current best while providing greater novelty.

F5 required separate treatment because the unconstrained local searches repeatedly converged almost exactly on its existing boundary optimum.

### 7.2 Function 5: Novelty-Constrained Refinement

F5's initial local searches produced candidates approximately 0.0001 from the existing best `[1,1,1,1]`.

Although technically distinct, these candidates offered negligible additional information.

Minimum novelty constraints of 0.03, 0.05 and 0.10 were therefore compared.

| Minimum novelty | Candidate | GP prediction | NN prediction | GP uncertainty | NN disagreement |
|---|---|---:|---:|---:|---:|
| 0.03 | [1, 0.969998, 1, 1] | 8287.623 | 8688.604 | 247.690 | 118.483 |
| 0.05 | [1, 0.949994, 1, 1] | 8001.755 | 8710.650 | 410.334 | 197.197 |
| 0.10 | [1, 0.899981, 1, 1] | 7187.507 | 8743.780 | 802.300 | 369.571 |

The results showed increasing disagreement between the models as candidates moved further from the boundary optimum.

At 0.03 novelty, the GP still predicted a relatively strong output of 8287.623, while the NN predicted a small improvement over the current best of 8662.482500.

At larger novelty distances, the NN became increasingly optimistic, but GP predictions deteriorated and uncertainty increased substantially.

The 0.03 candidate therefore provided the strongest balance between meaningful novelty and model support.

**Selected F5 candidate:** `[1.000000, 0.969998, 1.000000, 1.000000]`

### 7.3 Function 1: Model-Independent Exploration

F1 continued to require a different strategy.

Its GP length scale remained at the lower bound, the NN failed LOOCV, the two models strongly disagreed and nearby observations offered no meaningful differentiation.

A model-independent space-filling search was therefore used.

The selected candidate was:

`[0.682178, 0.001903]`

Its nearest existing observation was approximately **0.306754** away, with the next two nearest observations at distances of 0.308314 and 0.316118.

This represented a substantially underexplored region of the two-dimensional search space.

The candidate was selected primarily to increase coverage and information gain, rather than because either surrogate predicted a strong output.

### 7.4 Function 2: NN-Supported Refinement

F2 was investigated separately because its GP remained unreliable while its NN continued to outperform the LOOCV baseline.

NN-only searches were performed around the current best at radii of 0.03, 0.05 and 0.10.

| Radius | Candidate | NN prediction | NN disagreement | Nearest distance |
|---|---|---:|---:|---:|
| 0.03 | [0.711375, 0.108192] | 0.718309 | 0.149730 | 0.018533 |
| 0.05 | [0.710141, 0.088175] | 0.743700 | 0.211086 | 0.030706 |
| 0.10 | [0.704305, 0.038527] | 0.810157 | 0.363055 | 0.077676 |

Increasing the radius improved the NN prediction but also increased disagreement.

The 0.03 candidate had the lowest disagreement but was predicted below the current best. The 0.05 candidate was predicted approximately equal to the current best.

At radius 0.10, the NN predicted 0.810157, above the current best of 0.746962, while providing greater novelty.

Although disagreement increased to 0.363055, the direction of the search was consistent with the unconstrained NN recommendation towards lower values of the second input.

**Selected F2 candidate:** `[0.704305, 0.038527]`

The selection represented a cautious compromise between potential improvement, novelty and increasing NN disagreement.

### 7.5 Final Candidate Selection

| Function | Selected approach | Final candidate |
|---|---|---|
| F1 | Space-filling exploration | [0.682178, 0.001903] |
| F2 | NN refinement, radius 0.10 | [0.704305, 0.038527] |
| F3 | Local refinement, radius 0.03 | [0.473523, 0.500295, 0.456235] |
| F4 | Local refinement, radius 0.05 | [0.388825, 0.366595, 0.350533, 0.420635] |
| F5 | Boundary refinement, novelty 0.03 | [1.000000, 0.969998, 1.000000, 1.000000] |
| F6 | Local refinement, radius 0.10 | [0.440497, 0.372759, 0.525613, 0.791828, 0.000000] |
| F7 | Local refinement, radius 0.03 | [0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374] |
| F8 | Local refinement, radius 0.10 | [0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246] |

F3's radius 0.03 candidate was selected because it had the strongest GP prediction of approximately -0.000166, while its NN remained insufficiently validated.

F4's radius 0.05 candidate had the strongest GP prediction of approximately 0.563957, above the current best of 0.387262, while remaining close to the established high-performing region.

F6's radius 0.10 candidate was supported by both models, with GP and NN predictions of approximately -0.195871 and -0.229762 respectively, both above the current best of -0.274388.

F7 favoured a conservative radius 0.03 selection because the larger-radius NN prediction came with greater disagreement and weaker GP support.

F8's radius 0.10 candidate was selected because both models predicted improvement, with GP and NN predictions of approximately 9.976275 and 10.009696 compared with the current best of 9.957032.

The selections therefore reflected different balances between exploration, exploitation and novelty rather than applying a single acquisition rule across all eight functions.

---

## 8. Validation and Submission

The final candidates were rounded to six decimal places and checked for:

- Correct dimensionality.
- Input values within the permitted [0,1] bounds.
- No duplication of existing observations.
- Distance from the nearest existing observation.
- Correct submission formatting.

All eight candidates passed validation.

Nearest-observation distances ranged from approximately 0.010539 for F3 to 0.306755 for F1.

F3 and F4 remained close to their latest best observations, consistent with fine local refinement. F5 retained its deliberately imposed novelty distance of approximately 0.03.

F6 and F8 made broader local moves, with nearest-observation distances of approximately 0.110780 and 0.190714 respectively.

F2's nearest-observation distance was approximately 0.077676, while F1 remained the most exploratory selection.

### Final Week 9 Submission Points

```text
Function 1: 0.682178-0.001903
Function 2: 0.704305-0.038527
Function 3: 0.473523-0.500295-0.456235
Function 4: 0.388825-0.366595-0.350533-0.420635
Function 5: 1.000000-0.969998-1.000000-1.000000
Function 6: 0.440497-0.372759-0.525613-0.791828-0.000000
Function 7: 0.199451-0.330731-0.460263-0.255214-0.346569-0.603374
Function 8: 0.031565-0.175185-0.160043-0.130670-0.742356-0.487463-0.194164-0.622246
```

These points match the final selections produced by the Week 9 notebook.

---

## Reflection

Week 9 demonstrated that increasing the number of observations does not produce uniform optimisation progress. F3, F4 and F6 continued to improve meaningfully, while F8 showed generally smaller gains as its best output approached 10. F7 remained close to its established best without improving in the two most recent queries, and F5 retained a strong boundary region despite an unsuccessful latest query.

The additional scaling and improvement-size analysis helped distinguish **infrequent improvement** from **diminishing improvement size**. F2's gains were intermittent, whereas F8 showed clearer evidence of shrinking increments. Nevertheless, the latest F8 result was still a new best, and both surrogates supported further controlled refinement. Smaller gains alone were therefore insufficient reason to abandon a promising region.

Surrogate reliability remained function-specific. F1 and F2 still had GP length scales at the lower bound, while NN validation failed for F1 and F3. This justified space-filling exploration for F1, NN-supported refinement for F2, and greater reliance on GP and neighbourhood evidence for F3. The multi-radius searches further showed why a single refinement radius would not be suitable for all functions: F3, F4 and F7 favoured conservative steps, while F6 and F8 justified broader local searches.

F5 again illustrated the importance of explicit novelty checks. Without a minimum distance, its local search repeatedly selected points effectively identical to the current best. The 0.03 constraint preserved the successful boundary strategy while providing a more informative new observation.

Two implementation considerations are important for reproducibility. First, the input CSVs already included the latest observations, so the revised notebook avoids appending them twice. Second, Week 9's NN architecture changed to `(50, 50)` from `(20, 20)` in Weeks 7–8. Consequently, any cross-week changes in NN validation results reflect both dataset growth and a change in model capacity, not dataset growth alone. The LOOCV diagnostic also assesses an individual NN fitting procedure rather than the complete prediction ensemble.

Overall, Week 9 retained the established Bayesian optimisation methodology while using the original scaling analysis and local-search comparisons to make more targeted decisions. The resulting selections balanced model reliability, recent observed improvement, neighbourhood evidence and meaningful novelty without treating any one surrogate or acquisition score as decisive.

