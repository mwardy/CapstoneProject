
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

**Implementation consistency.** The uploaded CSVs already contained the Week 8 observations, so the revised notebook checks for existing observations before appending them. This prevents duplicate rows without changing the intended dataset. Week 9 uses NN hidden layers of **(50, 50)** rather than **(20, 20)** in Weeks 7–8. Consequently, differences in NN performance between weeks cannot be attributed solely to additional observations. The original Week 9 architecture and calculations were preserved. In contrast to the two UCB settings and two EI settings assessed in Week 8, Week 9 compares four of each; this is an expansion of the acquisition comparison, not an identical settings grid.

---

## 1. Week 8 Results

The returned outputs were compared against the previous best for each function before the updated datasets were used for Week 9 modelling.

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

**Four of eight functions achieved new best outputs.** F4 showed the largest positive change relative to its previous best, improving by 0.200350. F6 improved by 0.105352 and F3 by 0.009327, while F8 achieved a smaller gain of 0.026220.

F5 remained high after moving deliberately away from the boundary optimum, although its new output of 8115.434723 did not exceed 8662.482500. F7 returned 2.922059, very close to its best of 2.929370, suggesting that the local region remained promising. F2 did not improve, and F1's space-filling query again returned effectively zero.

These mixed results supported function-specific selection: continued refinement for F3, F4 and F6; careful assessment of diminishing returns for F7 and F8; and different strategies for the less reliable F1 and F2 models.

---

## 2. Optimisation Progress and Diminishing Returns

Running-best trajectories were examined across the complete observation histories. Absolute improvement sizes were used because some functions have negative outputs or cross zero, making percentage changes difficult to interpret.

| Function | Observations | Current best | Running-best interpretation |
|---|---:|---:|---|
| F1 | 18 | ~0 | Effectively flat despite additional observations. |
| F2 | 18 | 0.746962 | Three recorded new bests; progress intermittent. |
| F3 | 23 | -0.000863 | Eight new bests; strong recent progress. |
| F4 | 38 | 0.387262 | Eleven new bests; repeated substantial recent gains. |
| F5 | 28 | 8662.482500 | Eleven new bests; strong gains before the latest unsuccessful query. |
| F6 | 28 | -0.274388 | Five new bests; latest observation improved meaningfully. |
| F7 | 38 | 2.929370 | Six new bests; recent stability near the best region. |
| F8 | 48 | 9.957032 | Nine new bests; gains becoming relatively small. |

F3 and F4 provided the clearest evidence of continuing progress. F3 improved across its four most recent observations, while F4 continued to establish substantially higher best outputs. F6 also showed renewed progress: after a small gain of 0.001637, subsequent improvements increased to 0.077134 and 0.105352.

F5's recent gains remained variable and substantial, including a gain of 1541.331 before the latest unsuccessful query. Its failure to improve in Week 8 was not, by itself, sufficient evidence that the boundary region had been exhausted.

F7 had not improved in its two latest observations, although the most recent result was very close to its best. F8 showed the clearest signs of diminishing returns: recent gains of 0.621928, 0.223504, 0.089263 and 0.019562 were followed by a modest increase to 0.026220. The sequence was not strictly decreasing, but gains had generally become smaller as the observed best approached 10.

F2's progress was infrequent rather than consistently diminishing. F1 remained effectively flat; any technically recorded gains were negligible at the scale of its objective.

Overall, dataset growth did not produce a uniform scaling pattern or clear evidence of an emergent change in model capability. The apparent diminishing returns were primarily function- and region-specific.

---

## 3. Surrogate Diagnostics

### 3.1 Gaussian Process Models

GP models were refitted using the expanded datasets. Their RBF length scales and log marginal likelihoods were reviewed to identify persistent instability or changes in model behaviour.

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

**F1 and F2 remained at the GP length-scale lower bound of 1e-05.** Adding another observation did not resolve their difficulty in learning useful smooth relationships between observations, so their acquisition recommendations continued to require caution.

F3–F8 retained substantially larger fitted length scales. F4's GP remained usable despite rapid recent improvements, supporting further investigation of its successful region. F8 had the largest fitted length scale, consistent with a smoother modelled response than F1 or F2. The GP diagnostics did not indicate a sudden change in reliability as the datasets expanded.

### 3.2 Neural-Network Validation

The NNs were retrained and assessed with leave-one-out cross-validation (LOOCV), comparing prediction error against a mean-prediction baseline.

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

**Six of eight NN models outperformed their baselines.** F4 and F5 performed particularly strongly relative to their baselines; F6–F8 also provided useful evidence for NN-supported candidate selection.

F2 narrowly outperformed the baseline, making its NN more useful than its unreliable GP, although its predictions still warranted caution. F1 and F3 failed validation. F3's failure was notable because its optimisation results continued to improve, demonstrating that successful queries do not necessarily imply improved NN generalisation.

The validation procedure should also be distinguished from the final five-model ensemble: the LOOCV diagnostic evaluates the specified individual NN fitting procedure rather than cross-validating the full ensemble. Combined with the architecture change to (50, 50), this limits strict cross-week comparisons of NN reliability.

---

## 4. GP Acquisition Candidates

The updated GP models were used to compare UCB and EI recommendations across exploration settings. Predicted outputs, uncertainty, acquisition scores and distances from existing observations were considered together.

| Function | Main GP findings |
|---|---|
| F1 | All UCB and EI settings selected [0.094177, 0.975622], but the GP remained unreliable. |
| F2 | Low-κ UCB returned the existing Week 8 observation; higher-κ UCB and EI selected [0.403694, 0.391789]. |
| F3 | Higher κ increased exploration; κ=0.5 stayed local, while all EI settings selected the κ=1.5 candidate. |
| F4 | κ=0.5 predicted 0.736288 near the strong region; high κ moved towards much poorer predicted outputs and greater uncertainty. |
| F5 | Low-κ UCB favoured the upper-boundary region; high-κ UCB and EI selected a distant candidate with predicted output 1607.436. |
| F6 | Low-κ UCB predicted -0.248253, above the best of -0.274388; EI at ξ=0% predicted -0.220587. |
| F7 | Low-κ UCB effectively returned the existing best; higher-κ UCB and EI favoured a novel region. |
| F8 | κ=0.5 predicted 9.967635, slightly above the best of 9.957032; higher κ increasingly prioritised novelty. |

F4 and F6 offered particularly promising exploitation opportunities: their lower-exploration candidates were predicted above the latest observed best, and both functions had recently improved.

F5 again showed a pronounced contrast between the successful boundary region and distant, uncertain GP exploration. Its high-exploration candidate was [0.019672, 0.003040, 0.043324, 0.976161], approximately 0.686332 from its nearest observation, but had much weaker predicted performance than the established region.

F8 highlighted the exploration–exploitation trade-off. Its low-κ candidate predicted a small improvement, while higher-κ settings moved considerably further from the data. Given the smaller recent gains, it was important to test whether broader local refinement offered a better compromise than either tiny local steps or distant exploration.

For F1 and F2, agreement between acquisition settings was not interpreted as model reliability because both GP length scales remained at the lower bound.

---

## 5. Neural-Network Candidates

Five-model NN ensembles generated independent recommendations. Mean predictions, disagreement between models, distance from existing observations and LOOCV reliability were considered.

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

F2's NN predicted an improvement over its best of 0.746962, but its high ensemble disagreement of 0.431009 reduced confidence in the exact prediction. Its recommendation was therefore treated as a promising direction for further investigation rather than accepted automatically.

F4 and F7 favoured established high-performing regions, although their NN predictions remained below their current bests. F5 similarly retained a broad preference for the upper region, but its high disagreement suggested that moving away from the boundary optimum required caution.

F6's validated NN predicted -0.248980, above the best of -0.274388, with relatively low disagreement. F8's validated NN predicted 10.056825, above the best of 9.957032, with low disagreement of 0.027228. These provided useful reasons to investigate refinement even where recent gains had become smaller.

F1 and F3 predictions were given limited weight because their NNs failed LOOCV.

---

## 6. GP–NN Comparison and Neighbourhood Evidence

### 6.1 GP–NN Agreement

Euclidean distances were calculated between each NN recommendation and the GP candidates. The closest GP acquisition method was recorded for each function.

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

F4 and F7 showed relatively strong regional agreement between validated NNs and lower-exploration GP recommendations. F6 also showed moderate agreement around a promising region. F5's NN was closer to local UCB candidates than to the distant exploratory GP region.

F3 had the smallest distance, but its NN failed LOOCV, so the agreement was not treated as strong independent validation. F1 and F2 showed substantial disagreement. F8's distance of 0.243999 suggested some regional similarity but was not particularly close point-level agreement in eight dimensions.

### 6.2 Neighbourhood Evidence

The nearest observed outputs around leading NN, low-κ UCB and high-κ UCB candidates were compared to assess whether the recommended regions had empirical support.

| Function | Main neighbourhood evidence |
|---|---|
| F1 | Neighbours around both model candidates remained effectively zero; neither region was supported. |
| F2 | NN neighbours included 0.471844, 0.538996 and 0.746962; exploratory GP neighbours were much weaker. |
| F3 | Local neighbours included -0.000863, -0.010190 and -0.016451; distant exploration had weaker support. |
| F4 | Local region included 0.387262 and 0.186912; exploratory GP neighbours included -13.527649 and -10.069633. |
| F5 | Upper region included 8115.434723; exploratory GP neighbours were approximately 64–233. |
| F6 | Local NN neighbours included -0.274388, -0.388687 and -0.379740. |
| F7 | NN neighbours included 2.922059, 2.929370 and 2.536939. |
| F8 | NN neighbours included 9.930811, 9.911249 and 9.957032; low-κ GP also occupied a strong region. |

The neighbourhood results broadly favoured controlled local refinement for F3–F8. F4 and F6 had recently improved, and their recommended local regions were supported by strong observations. F5's contrast was particularly striking: observed outputs above 8000 near the boundary versus values near 64–233 around the exploratory GP candidate.

F2's NN region was better supported than the exploratory GP region, consistent with its NN passing LOOCV while its GP remained unreliable. F8's strong neighbourhood support suggested that smaller recent gains did not necessarily mean that its best region was exhausted.

F1 remained unresolved. Neither surrogate was validated adequately, the recommendations differed substantially and nearby observations offered no meaningful distinction. Model-independent exploration remained the most defensible approach.

---

## 7. Function-Specific Candidate Selection

Selections used the combined evidence rather than one common acquisition rule. F1 and F2 were treated separately because their GP models remained unreliable. For F3–F8, local searches evaluated **50,000 candidates at each of three radii: 0.03, 0.05 and 0.10** around the current best. GP and NN predictions were standardised before combination to avoid differences in objective scale affecting the joint ranking. Existing observations were excluded.

### 7.1 Multi-Radius Local Refinement

| Function | Findings across local radii |
|---|---|
| F3 | All radii converged very close to the best; GP predictions approximately -0.00017 to -0.00024. |
| F4 | All radii converged on the same region; GP predictions 0.531330–0.563957, above the best of 0.387262. |
| F5 | All radii returned points only ~0.0001 from [1,1,1,1], offering negligible novelty. |
| F6 | Larger radii improved both models' predictions; at radius 0.10, GP predicted -0.195871 and NN -0.229762. |
| F7 | All radii stayed near the best; larger radii increased NN optimism but also uncertainty and disagreement. |
| F8 | Both models predicted potential improvement; radius 0.10 provided greater novelty while remaining locally supported. |

F3 and F4 favoured fine refinement because their searches converged tightly on the strongest observed regions. For F3, the GP was given greater weight than the NN because the NN failed validation. F4's strongest GP prediction was 0.563957, substantially above its best of 0.387262.

F6 differed: increasing the radius improved both GP and NN predictions, suggesting that useful refinement could extend beyond the immediate neighbourhood. F7 required more caution because higher NN predictions at larger radii came with increased disagreement. F8 supported a somewhat broader local search despite diminishing recent gains, because both models remained optimistic.

### 7.2 Function 5: Novelty-Constrained Boundary Refinement

F5's initial local searches repeatedly selected points almost identical to the existing best at `[1,1,1,1]`. To avoid an uninformative near-duplicate query, minimum nearest-observation distances of 0.03, 0.05 and 0.10 were compared.

| Minimum novelty | Candidate | GP prediction | NN prediction | GP uncertainty | NN disagreement |
|---|---|---:|---:|---:|---:|
| 0.03 | [1, 0.969998, 1, 1] | 8287.623017 | 8688.604318 | 247.689521 | 118.482518 |
| 0.05 | [1, 0.949994, 1, 1] | 8001.755491 | 8710.650195 | 410.333828 | 197.196894 |
| 0.10 | [1, 0.899981, 1, 1] | 7187.506528 | 8743.779604 | 802.299569 | 369.571101 |

The NN became more optimistic as the candidate moved further away, but GP predictions deteriorated and both GP uncertainty and NN disagreement increased sharply. This was not interpreted as stronger evidence for improvement.

The **0.03 novelty candidate** offered the best balance between a meaningful new query and support from the established boundary region. It was selected for F5.

### 7.3 Function 1: Model-Independent Exploration

F1 was handled separately because the GP length scale remained at its lower bound, the NN failed LOOCV, and the models disagreed strongly. A maximum-distance space-filling search was used instead of relying on unreliable predictions.

The selected candidate was **[0.682178, 0.001903]**, approximately **0.306754** from its nearest existing observation. Its next two nearest-observation distances were 0.308314 and 0.316118.

This choice prioritised exploration of an under-sampled region rather than an expectation of immediate objective improvement. It continued the model-independent strategy introduced in Week 8.

### 7.4 Function 2: NN-Supported Refinement

F2 was also treated separately because its GP remained unreliable. Its NN passed LOOCV, albeit narrowly, and its preferred region had stronger observed support. NN-only searches were therefore performed at radii 0.03, 0.05 and 0.10 around the current best.

| Radius | Candidate | NN prediction | NN disagreement | Nearest distance |
|---|---|---:|---:|---:|
| 0.03 | [0.711375, 0.108192] | 0.718309 | 0.149730 | 0.018533 |
| 0.05 | [0.710141, 0.088175] | 0.743700 | 0.211086 | 0.030706 |
| 0.10 | [0.704305, 0.038527] | 0.810157 | 0.363055 | 0.077676 |

Predicted performance improved with radius, but NN disagreement also increased. The 0.03 candidate was less novel and predicted below the best of 0.746962; the 0.05 candidate approximately matched it. The 0.10 candidate predicted an improvement and moved meaningfully towards lower values of the second input, consistent with the unconstrained NN recommendation.

The **radius 0.10 candidate** was selected as a cautious test of this direction, recognising that its higher disagreement limited confidence in the predicted output.

### 7.5 Final Selections

| Function | Selection strategy | Final candidate |
|---|---|---|
| F1 | Space-filling exploration | [0.682178, 0.001903] |
| F2 | NN-only refinement, radius 0.10 | [0.704305, 0.038527] |
| F3 | Local refinement, radius 0.03 | [0.473523, 0.500295, 0.456235] |
| F4 | Local refinement, radius 0.05 | [0.388825, 0.366595, 0.350533, 0.420635] |
| F5 | Boundary refinement, novelty 0.03 | [1.000000, 0.969998, 1.000000, 1.000000] |
| F6 | Local refinement, radius 0.10 | [0.440497, 0.372759, 0.525613, 0.791828, 0.000000] |
| F7 | Local refinement, radius 0.03 | [0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374] |
| F8 | Local refinement, radius 0.10 | [0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246] |

F3 was selected from the radius 0.03 search using the combined standardised GP and NN prediction score. The GP prediction at this point was -0.000166; because the NN had not passed LOOCV, its contribution to the score should be interpreted cautiously. F4 used radius 0.05, selected by the same combined-score procedure; the chosen point also had a strong GP prediction (0.563957) within a tightly convergent local region.

F6 used radius 0.10 because both GP (-0.195871) and NN (-0.229762) predicted improvement over its best (-0.274388), with meaningful novelty. F7 used radius 0.03 because it balanced the strongest GP support with lower NN disagreement, while its NN prediction of 2.985400 remained above the best of 2.929370.

F8 used radius 0.10 because both GP (9.976275) and NN (10.009696) predicted improvement over its best (9.957032), while the candidate remained within a locally supported region. This offered broader refinement without adopting the poorly supported distant GP exploration.

---

## 8. Validation and Submission

The selected points were rounded to six decimal places and checked for correct dimensionality, coordinates within `[0,1]`, duplicate observations and nearest-observation distance. All eight passed the original Week 9 validation checks.

Nearest-observation distances ranged from **0.010539 for F3** to **0.306755 for F1**. F5 retained approximately 0.03 novelty, F2 was 0.077676 from its nearest observation, F6 was 0.110780 and F8 was 0.190714. These distances reflect the intended differences between fine local refinement and broader exploration.

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

These eight points match the original Week 9 notebook’s saved results and the final submitted selections. In a separate execution using scikit-learn 1.8.0 and the supplied cumulative observations, seven of the eight candidates reproduced exactly at six decimal places. The F2 NN-only refinement returned a different point; the original submitted coordinate has been retained rather than silently replaced.

---

### Reproducibility note

The revised notebook checks that it has loaded the eight correct cumulative datasets rather than accepting any older `function_i.csv` file with the same name. It displays intermediate results as tables and compares each final selection with the candidate recalculated by its designated search. The original saved Week 9 run reported F2 NN LOOCV MAE **0.236724** versus baseline **0.241635**, and selected `[0.704305, 0.038527]` at radius 0.10. An independent test run with **scikit-learn 1.8.0** produced F2 LOOCV MAE **0.257302** and the alternative local point `[0.701892, 0.038659]`, despite matching the other seven final coordinates. The reason for this cross-environment discrepancy is not yet established and should not be interpreted as a new optimisation experiment or as grounds to change the submitted F2 point. The discussion above reports the results saved by the original Week 9 execution, not a blended set of outcomes from different runs.

---

## Reflection

Week 9 demonstrated that increasing the number of observations does not produce uniform optimisation progress. F3, F4 and F6 continued to improve meaningfully, while F8 showed generally smaller gains as its best output approached 10. F7 remained close to its established best without improving in the two most recent queries, and F5 retained a strong boundary region despite an unsuccessful latest query.

The additional scaling and improvement-size analysis helped distinguish **infrequent improvement** from **diminishing improvement size**. F2's gains were intermittent, whereas F8 showed clearer evidence of shrinking increments. Nevertheless, the latest F8 result was still a new best, and both surrogates supported further controlled refinement. Smaller gains alone were therefore insufficient reason to abandon a promising region.

Surrogate reliability remained function-specific. F1 and F2 still had GP length scales at the lower bound, while NN validation failed for F1 and F3. This justified space-filling exploration for F1, NN-supported refinement for F2, and greater reliance on GP and neighbourhood evidence for F3. The multi-radius searches further showed why a single refinement radius would not be suitable for all functions: F3, F4 and F7 favoured conservative steps, while F6 and F8 justified broader local searches.

F5 again illustrated the importance of explicit novelty checks. Without a minimum distance, its local search repeatedly selected points effectively identical to the current best. The 0.03 constraint preserved the successful boundary strategy while providing a more informative new observation.

Two implementation considerations are important for reproducibility. First, the input CSVs already included the latest observations, so the revised notebook avoids appending them twice. Second, Week 9's NN architecture changed to `(50, 50)` from `(20, 20)` in Weeks 7–8. Consequently, any cross-week changes in NN validation results reflect both dataset growth and a change in model capacity, not dataset growth alone. The LOOCV diagnostic also assesses an individual NN fitting procedure rather than the complete prediction ensemble.

Overall, Week 9 retained the established Bayesian optimisation methodology while using the original scaling analysis and local-search comparisons to make more targeted decisions. The resulting selections balanced model reliability, recent observed improvement, neighbourhood evidence and meaningful novelty without treating any one surrogate or acquisition score as decisive.

