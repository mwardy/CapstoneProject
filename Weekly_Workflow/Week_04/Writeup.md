
# Module 15 — Week 4: UCB exploration tuning and Expected Improvement

## Weekly objective and step-by-step plan

The objective of Week 4 was to incorporate the latest query results, reassess the Gaussian process (GP) models and investigate whether adjusting the exploration parameter in upper confidence bound (UCB) could improve query selection.

In previous weeks, I had used a fixed UCB exploration parameter of κ = 2.5. This week, I compared three settings (κ = 1.5, 2.5 and 3.5) to understand how changing the balance between exploration and exploitation affected the selected candidates.

I also introduced Expected Improvement (EI) as an alternative acquisition function, allowing me to compare two different approaches to selecting promising query points.

Particular attention remained on Functions 1 and 6, where the Gaussian processes had experienced persistent modelling difficulties.

The process followed six steps:

1. Incorporate the new observations and compare them with the previous best outputs.
2. Refit the Gaussian processes and examine their fitted length scales, predictions and uncertainty.
3. Compare UCB exploration settings of κ = 1.5, 2.5 and 3.5.
4. Reassess the exploration–exploitation balance for each function.
5. Compare UCB with Expected Improvement and investigate alternatives for Functions 1 and 6.
6. Select, validate and format the eight final queries.

## Methodology

I incorporated the latest observations into the cumulative datasets and refitted the Gaussian processes using RBF kernels, normalised outputs, ten optimiser restarts and a random state of 42.

During the initial Week 4 fitting, I noticed that the optimiser restart parameter had been omitted. This caused Functions 1–7 to return the minimum permitted RBF length scale. I corrected this by restoring `n_restarts_optimizer=10`, consistent with the methodology used in previous weeks, before interpreting the results.

For functions with usable GP models, I compared three UCB exploration settings:

**UCB(x) = μ(x) + κσ(x)**

Here, μ(x) is the predicted function output, σ(x) is the predictive uncertainty and κ determines the relative importance of exploration.

A lower κ places greater emphasis on the predicted mean and therefore exploitation, while a higher κ places greater emphasis on uncertainty and exploration.

I compared κ = 1.5, 2.5 and 3.5 using the same 100,000 candidate points for each function.

I then introduced Expected Improvement, which estimates the expected improvement over the current best observed output rather than combining the predicted mean and uncertainty directly.

For Functions 1 and 6, where the Gaussian processes remained unreliable, I investigated maximum-distance exploration as a model-independent alternative.

## Results and interpretation by step

### Step 1 — Incorporate new observations

I compared the outputs returned from Week 3's proposed queries with the previous best observations.

| Function | Previous best | New output | Updated best | Result |
|---|---:|---:|---:|---|
| 1 | ≈0 | ≈0 | ≈0 | No improvement |
| 2 | 0.611205 | 0.548608 | 0.611205 | No improvement |
| 3 | -0.020167 | -0.115141 | -0.020167 | No improvement |
| 4 | -0.300645 | -1.850268 | -0.300645 | No improvement |
| 5 | 2765.819787 | 4595.479318 | 4595.479318 | New best |
| 6 | -0.458511 | -1.182842 | -0.458511 | No improvement |
| 7 | 1.950612 | 2.139551 | 2.139551 | New best |
| 8 | 9.821986 | 9.645301 | 9.821986 | No improvement |

Two of the eight queries produced new best outputs.

Function 5 improved from approximately 2765.82 to 4595.48, an increase of 66.2%. This was the third successive improvement for the function, providing further evidence that the search had identified a particularly promising region of its input space.

Function 7 also improved, from approximately 1.9506 to 2.1396, suggesting that GP + UCB was continuing to identify useful candidates.

The remaining functions did not improve, although Function 2's latest output of 0.5486 was relatively close to its current best of 0.6112.

Function 6's SVM-informed query produced an output of -1.1828, below the previous best of -0.4585. This suggested that identifying a broadly higher-performing region through classification did not necessarily provide sufficiently precise guidance for maximising the continuous function.

These results informed the strategy adjustments considered later in the week.

### Step 2 — Refit and evaluate the Gaussian processes

I refitted the RBF Gaussian processes using the expanded datasets and examined their fitted length scales.

| Function | Fitted RBF length scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0516 |
| 3 | 0.198 |
| 4 | 0.428 |
| 5 | 0.367 |
| 6 | 1e-05 |
| 7 | 0.296 |
| 8 | 0.949 |

Functions 2, 3, 4, 5, 7 and 8 produced length scales above the lower bound and were retained for GP-based acquisition comparisons.

However, Functions 1 and 6 again reached the minimum permitted length scale of 1e-05.

As in Week 3, I examined their predicted means and uncertainty across 100,000 candidate points to determine whether the models could meaningfully distinguish between different regions.

| Diagnostic | Function 1 | Function 6 |
|---|---:|---:|
| Predicted mean range | 0.0002773740 | 0.0000000000 |
| Uncertainty range | 0.0009507851 | 0.0000000000 |

Function 1 continued to show some variation in its predictions and uncertainty. However, its repeated convergence issues and extremely small predicted outputs meant that I remained cautious about relying on its GP for query selection.

Function 6 produced identical predicted means and uncertainty across all candidate points. Consequently, changing κ would not meaningfully alter the UCB rankings, making the GP unsuitable for selecting its next query.

I therefore excluded Functions 1 and 6 from the UCB comparison and considered alternative approaches separately.

### Step 3 — Compare UCB exploration settings

For the six functions with usable Gaussian processes, I compared κ values of 1.5, 2.5 and 3.5.

Each setting was evaluated using the same 100,000 candidate points for each function, allowing the effect of changing κ to be examined without introducing differences from candidate sampling.

The following table compares the predicted outputs of the selected candidates.

| Function | κ = 1.5 | κ = 2.5 | κ = 3.5 |
|---|---:|---:|---:|
| 2 | 0.5560 | 0.3754 | 0.3424 |
| 3 | -0.0339 | -0.0751 | -0.0830 |
| 4 | -0.3091 | -0.6945 | -18.3618 |
| 5 | 4561.79 | 4090.67 | 4090.67 |
| 7 | 1.9591 | 1.9591 | 1.6413 |
| 8 | 10.0591 | 9.5513 | 9.2533 |

Increasing κ generally resulted in candidates with lower predicted outputs but greater uncertainty.

Function 4 demonstrated this particularly clearly. At κ = 1.5, the selected candidate had a predicted output of -0.3091, close to its current best of -0.3006. At κ = 3.5, the predicted output fell to -18.3618, reflecting a much stronger emphasis on exploration.

Function 5 showed a similar trade-off. At κ = 1.5, the selected candidate had a predicted output of 4561.79, close to its newly achieved maximum of 4595.48. Higher κ values selected a more uncertain candidate further from the successful region, with a lower predicted output of 4090.67.

Function 8 was also notable. The κ = 1.5 candidate had a predicted output of 10.0591, above the current observed maximum of 9.8220. Higher κ values moved towards more uncertain regions with lower predicted outputs.

Function 7 was comparatively stable, with κ = 1.5 and 2.5 selecting exactly the same candidate.

To investigate the effect of κ further, I calculated the distances between candidates selected under different settings.

| Function | Distance: κ 1.5 → 2.5 | Distance: κ 2.5 → 3.5 |
|---|---:|---:|
| 2 | 0.8507 | 0.1135 |
| 3 | 0.6195 | 0.0707 |
| 4 | 0.0478 | 0.9480 |
| 5 | 0.1438 | 0.0000 |
| 7 | 0.0000 | 0.1986 |
| 8 | 0.7627 | 0.9939 |

Functions 2 and 3 showed substantial changes between κ = 1.5 and 2.5, while the higher two settings selected relatively similar regions.

Function 4 remained relatively stable between κ = 1.5 and 2.5 but moved dramatically at κ = 3.5. This reinforced the concern that the highest exploration setting could move the search away from promising regions.

Function 8 was particularly sensitive to κ, with all three settings selecting substantially different parts of the search space.

In contrast, Functions 5 and 7 were more stable across some settings.

Overall, the comparison demonstrated that the exploration parameter could materially affect candidate selection, but its influence varied considerably between functions.

### Step 4 — Reassess exploration and exploitation

I used the latest observations, GP diagnostics and κ comparison to select provisional strategies for each function.

Rather than choosing κ solely according to the highest acquisition score, I considered predicted performance, uncertainty, recent improvements and the location of the selected candidates.

The provisional decisions were:

- Function 1: Continue maximum-distance exploration because the GP remained unreliable.
- Function 2: Prefer κ = 1.5, as the latest query was relatively close to the best output and the lower κ selected a promising nearby region.
- Function 3: Prefer κ = 1.5 to reduce the emphasis on exploration following an unsuccessful query.
- Function 4: Prefer κ = 1.5 to avoid the particularly distant, low-prediction candidate selected at κ = 3.5.
- Function 5: Prefer κ = 1.5 to exploit the region responsible for three successive improvements.
- Function 6: Investigate an alternative strategy because the GP remained uninformative and the previous SVM-informed query had not improved performance.
- Function 7: Provisionally retain κ = 1.5 following the latest improvement, with the selected point also supported by κ = 2.5.
- Function 8: Prefer κ = 1.5 because it selected a candidate with a predicted output above the current best.

Although κ = 1.5 was provisionally preferred for all six functions with usable Gaussian processes, the reasons differed.

Function 5 provided the strongest evidence for exploitation, having improved from approximately 1088.86 to 4595.48 over successive rounds.

For Functions 2, 3 and 4, the lower κ setting offered a more cautious alternative to further strong exploration following unsuccessful queries.

Function 8 also favoured exploitation, although the substantial differences between candidate locations suggested that the choice of acquisition strategy remained important.

### Step 5 — Compare UCB with Expected Improvement

I introduced Expected Improvement as an alternative acquisition function for Functions 2, 3, 4, 5, 7 and 8.

Unlike UCB, which combines predicted performance and uncertainty using a fixed exploration parameter, EI estimates the expected improvement over the current best observed output.

I compared the EI-selected candidates with those selected using UCB at κ = 1.5.

| Function | UCB predicted output | EI predicted output | Distance between candidates |
|---|---:|---:|---:|
| 2 | 0.5560 | 0.5901 | 0.0085 |
| 3 | -0.0339 | -0.0312 | 0.0228 |
| 4 | -0.3091 | -0.3091 | 0.0000 |
| 5 | 4561.79 | 4561.79 | 0.0000 |
| 7 | 1.9591 | 2.0538 | 0.1893 |
| 8 | 10.0591 | 10.0803 | 0.2621 |

EI and UCB selected very similar candidates for Functions 2 and 3, with distances of just 0.0085 and 0.0228 respectively.

For Functions 4 and 5, both methods selected exactly the same points.

This agreement provided additional support for the selected regions, particularly Function 5, where previous queries had produced repeated improvements.

Functions 7 and 8 showed greater differences between the two acquisition methods.

For Function 7, EI selected a candidate with a predicted output of 2.0538, compared with 1.9591 for UCB. The EI candidate also had lower uncertainty and remained relatively close to the current best observation, supporting a more exploitative strategy.

For Function 8, EI selected a candidate with a predicted output of 10.0803, slightly above UCB's 10.0591. Both predictions exceeded the current observed maximum of 9.8220, suggesting that the GP identified this broader region as promising.

The agreement between EI and UCB for Functions 2–5 increased confidence in the selected regions, while their differences for Functions 7 and 8 demonstrated the value of comparing acquisition functions rather than relying on a single method.

#### Alternative strategies for Functions 1 and 6

Functions 1 and 6 required separate treatment because their Gaussian processes remained unreliable.

For Function 1, I retained maximum-distance exploration, continuing the strategy used in previous rounds.

For Function 6, the previous SVM-informed query had failed to improve the observed maximum, while the refitted GP continued to produce identical predictions and uncertainty across candidate points.

I therefore also used maximum-distance exploration for Function 6.

For both functions, I generated 100,000 candidate points and selected the candidate with the greatest distance from its nearest existing observation.

| Function | Nearest observation distance | Distance from current best |
|---|---:|---:|
| 1 | 0.3153 | 0.4137 |
| 6 | 0.8782 | 0.8810 |

Function 1's selected candidate was approximately 0.315 from its nearest existing observation, providing further exploration of an under-sampled region.

Function 6's candidate was considerably more exploratory, with a nearest-observation distance of 0.878 and a distance of 0.881 from the current best.

Although this represented a higher-risk selection, it offered an opportunity to collect information from a substantially different region after the GP and previous SVM-informed approach had provided limited guidance.

### Step 6 — Validate the final proposed queries

I selected the final queries using the combined evidence from the latest observations, GP diagnostics, UCB comparison and Expected Improvement.

EI was preferred for Functions 2, 3, 7 and 8 because it selected candidates with stronger predicted outputs while maintaining a reasonable balance between exploration and exploitation.

For Functions 4 and 5, EI and UCB at κ = 1.5 selected identical candidates.

Functions 1 and 6 retained maximum-distance exploration.

I then checked that all selected points were within the permitted [0, 1] input range and did not duplicate existing observations.

| Function | Selection method | Nearest observation distance |
|---|---|---:|
| 1 | Max-distance exploration | 0.315336 |
| 2 | Expected Improvement | 0.030161 |
| 3 | Expected Improvement | 0.178778 |
| 4 | EI / UCB agreement | 0.091845 |
| 5 | EI / UCB agreement | 0.193062 |
| 6 | Max-distance exploration | 0.878156 |
| 7 | Expected Improvement | 0.149214 |
| 8 | Expected Improvement | 0.518070 |

All eight proposed queries passed the validation checks.

The final queries, formatted to six decimal places, were:

| Function | Selection method | Proposed query coordinates |
|---|---|---|
| 1 | Max-distance exploration | `[0.417430, 0.463246]` |
| 2 | Expected Improvement | `[0.730707, 0.915532]` |
| 3 | Expected Improvement | `[0.433866, 0.716516, 0.000863]` |
| 4 | EI / UCB agreement | `[0.443740, 0.493644, 0.409790, 0.452219]` |
| 5 | EI / UCB agreement | `[0.851064, 0.985689, 0.919598, 0.979330]` |
| 6 | Max-distance exploration | `[0.033216, 0.058439, 0.124012, 0.011504, 0.042834]` |
| 7 | Expected Improvement | `[0.154752, 0.313081, 0.341855, 0.169387, 0.318877, 0.633022]` |
| 8 | Expected Improvement | `[0.148540, 0.297539, 0.201860, 0.328776, 0.636564, 0.500531, 0.197949, 0.479426]` |

## Weekly summary and next steps

Week 4 produced new best observations for Functions 5 and 7, with Function 5 achieving a particularly substantial improvement.

The main methodological development was investigating how changing κ affected UCB candidate selection. The results demonstrated that increasing the emphasis on exploration could substantially change the selected region, particularly for Functions 2, 3, 4 and 8.

Introducing Expected Improvement provided a second way to evaluate promising candidates. EI and UCB selected very similar points for Functions 2–5, while their differences for Functions 7 and 8 highlighted the importance of considering alternative acquisition functions.

Functions 1 and 6 remained challenging because their Gaussian processes continued to reach the lower RBF length-scale bound. Maximum-distance exploration was retained for Function 1 and reintroduced for Function 6 after the previous SVM-informed query failed to improve performance.

The next steps were to:

- Submit the eight proposed query points and assess their returned outputs.
- Incorporate the new observations and refit the GP models.
- Evaluate whether the greater emphasis on exploitation improves performance.
- Reassess the suitability of EI and UCB for individual functions.
- Continue investigating alternative modelling approaches for Functions 1 and 6 if their GP difficulties persist.

Overall, Week 4 demonstrated that acquisition-function selection and exploration weighting can have a substantial influence on Bayesian optimisation. The final queries reflected a greater emphasis on exploiting promising regions where the GP models were informative, while retaining deliberate exploration where reliable model guidance was unavailable.

