
# Module 18 — Week 7: Acquisition Hyperparameter Tuning and Candidate Stability

## Weekly objective and step-by-step plan

Week 7 builds on the function-specific optimisation strategy developed during the previous rounds. Rather than introducing another modelling approach, the main focus this week was on tuning the Gaussian Process acquisition-function hyperparameters and assessing how robust the resulting candidate recommendations were.

Gaussian Process Regression (GPR) remained the primary surrogate model, with Upper Confidence Bound (UCB) and Expected Improvement (EI) used to identify promising query points. However, instead of relying on a single exploration setting, I tested several values of the acquisition parameters to examine how sensitive the recommendations were to the exploration–exploitation balance.

The neural-network (NN) ensemble introduced in Week 5 was retained as an independent comparison. Its predictive reliability was reassessed using Leave-One-Out Cross-Validation (LOOCV), and its recommendations were considered alongside the GP results, historical optimisation performance and nearby observed outputs.

The objective was not simply to select the candidate with the highest predicted output or acquisition score, but to understand how dependent each recommendation was on the modelling assumptions and hyperparameter settings.

The analysis followed nine main steps:

1. Incorporate the Week 6 outputs and compare them with the previous best observations.
2. Refit and diagnose the Gaussian Process models using the expanded datasets.
3. Test multiple UCB κ and EI ξ settings to examine acquisition sensitivity.
4. Assess candidate stability across the different exploration settings.
5. Retrain and validate the neural-network ensembles using LOOCV.
6. Generate NN candidates and compare them with the tuned GP recommendations.
7. Examine the nearest existing observations around the leading candidates.
8. Select one candidate per function using the combined evidence.
9. Validate the final coordinates and prepare the eight submission points.

## Methodology

The optimisation framework remained broadly consistent with the previous weeks, with the main methodological development being the systematic comparison of acquisition-function hyperparameters.

Each function was modelled independently using a Gaussian Process with an RBF kernel. The models were refitted using all available observations at the start of Week 7.

For candidate evaluation, 100,000 points were generated within the permitted [0, 1] search space for each function. The same candidate sets were retained throughout the GP acquisition comparisons so that changes in recommendations could be attributed to the acquisition parameters rather than different randomly generated points.

UCB was evaluated using κ values of 0.5, 1.0, 1.5 and 2.5. Lower values place greater emphasis on the predicted mean, while higher values increase the importance of uncertainty.

EI was evaluated using ξ values corresponding to 0%, 1%, 5% and 10% of each function's observed output standard deviation. This scaling allowed relative exploration settings to be compared across functions with substantially different output magnitudes.

The neural-network ensemble was also retrained using the expanded datasets. Five independently initialised networks were used for each function, and their predictions were combined to produce an alternative candidate recommendation. LOOCV performance was compared against a mean-prediction baseline before deciding how much weight to give the NN results.

Finally, the leading GP and NN candidates were assessed using their distances from existing observations and the observed outputs in their surrounding neighbourhoods.

Candidate selection remained function-specific, combining model predictions, acquisition sensitivity, validation performance, previous optimisation results and local evidence.

## Results and interpretation by step

### Step 1 — Incorporate and assess the Week 6 results

The eight outputs returned from the Week 6 queries were incorporated into the accumulated datasets and compared with the previous best observations.

| Function | Previous best | New output | Updated best | Improved? |
|---|---:|---:|---:|---|
| F1 | 0.000000 | ≈0 | 0.000000 | No |
| F2 | 0.746962 | 0.471844 | 0.746962 | No |
| F3 | -0.016451 | -0.015498 | -0.015498 | Yes |
| F4 | -0.300645 | -0.200431 | -0.200431 | Yes |
| F5 | 6691.931674 | 7121.151955 | 7121.151955 | Yes |
| F6 | -0.456874 | -0.379740 | -0.379740 | Yes |
| F7 | 2.536939 | 2.929370 | 2.929370 | Yes |
| F8 | 9.911249 | 9.840530 | 9.911249 | No |

Five of the eight Week 6 queries produced new best observations, providing encouraging evidence for the increasingly function-specific selection strategy.

Functions 3, 4 and 6 improved following EI-selected queries. Function 3 continued its gradual improvement from -0.016451 to -0.015498, while Function 4 improved from -0.300645 to -0.200431. Function 6 also improved from -0.456874 to -0.379740.

Function 5 achieved another substantial improvement, increasing from 6691.93 to 7121.15. This provided further support for the neural-network approach, which had selected the previous query. Function 7 also improved considerably, increasing from 2.536939 to 2.929370 following a candidate supported by agreement between UCB and EI.

The unsuccessful queries were equally useful for assessing method performance. Function 2 returned 0.471844, below its previous best of 0.746962. Function 8's NN-selected candidate returned 9.840530, slightly below the best of 9.911249. This was the second consecutive round in which the NN had failed to improve Function 8, suggesting that its recommendations should be treated more cautiously despite reasonable predictive validation.

Function 1 again returned a value effectively equal to zero, providing little additional information about its optimum.

Overall, these results supported continuing with a mixed approach rather than selecting one surrogate or acquisition function for all eight problems. In particular, the recent improvements strengthened the case for EI on Functions 3, 4 and 6, the NN on Function 5, and GP acquisition agreement on Function 7.

### Step 2 — Refit and diagnose the Gaussian Processes

The GP models were refitted using all observations available at the beginning of Week 7. The fitted RBF length scales were examined to assess whether the additional data had changed the behaviour of the models.

| Function | Fitted RBF length scale |
|---|---:|
| F1 | 0.00001 |
| F2 | 0.0431 |
| F3 | 0.193 |
| F4 | 0.436 |
| F5 | 0.363 |
| F6 | 0.558 |
| F7 | 0.300 |
| F8 | 1.020 |

Most functions continued to produce relatively stable fitted length scales.

Functions 3, 4 and 6 showed relatively small increases compared with the previous week. Function 3 increased from approximately 0.185 to 0.193, Function 4 from 0.423 to 0.436, and Function 6 from 0.535 to 0.558.

Function 7's length scale increased from approximately 0.248 to 0.300 following its new best observation. Function 2 decreased from 0.0521 to 0.0431, suggesting that the GP continued to favour a relatively local response structure.

Function 1 remained the main exception. Its RBF length scale again reached the lower bound of 0.00001, reproducing the convergence issue seen in previous rounds. This suggests that the GP was still struggling to identify a meaningful smooth relationship between nearby observations.

Consequently, GP recommendations for Function 1 continued to be treated cautiously.

#### Prediction and uncertainty diagnostics

The refitted GP models were evaluated across 100,000 candidate points per function to examine their predicted outputs and uncertainty.

| Function | Key GP diagnostic |
|---|---|
| F1 | Predicted range: 0.000595; maximum uncertainty: 0.001626 |
| F2 | Predicted maximum: 0.7905; maximum uncertainty: 0.2550 |
| F3 | Predicted maximum: -0.0118; maximum uncertainty: 0.0782 |
| F4 | Predicted maximum: -0.2003; maximum uncertainty: 7.4108 |
| F5 | Predicted maximum: 7121.16; maximum uncertainty: 2074.52 |
| F6 | Predicted maximum: -0.3656; maximum uncertainty: 0.5196 |
| F7 | Predicted maximum: 2.9294; maximum uncertainty: 0.7721 |
| F8 | Predicted maximum: 9.9641; maximum uncertainty: 0.6576 |

Functions 2–8 showed meaningful variation in their predicted outputs and uncertainty, providing a useful basis for acquisition-function tuning.

For several functions, the highest GP predictions remained close to their strongest observed outputs, suggesting that the models continued to identify recently successful regions.

Function 1 remained difficult to interpret. Its predictions were concentrated around zero, and its fitted kernel continued to exhibit convergence problems.

The diagnostic results supported retaining the GP models for the acquisition analysis, while accounting for the different levels of reliability across functions.

### Step 3 — Tune the acquisition-function hyperparameters

The main methodological development this week was testing how different exploration settings affected the candidates selected by UCB and EI.

#### UCB hyperparameter tuning

UCB was evaluated using four κ values:

- κ = 0.5
- κ = 1.0
- κ = 1.5
- κ = 2.5

Lower κ values place greater emphasis on the predicted mean, encouraging exploitation. Higher values increase the contribution of predictive uncertainty, encouraging exploration.

The same 100,000 candidate points were used for every setting.

| Function | Response to increasing κ |
|---|---|
| F1 | Identical candidate at all four settings |
| F2 | Gradual movement away from the current best |
| F3 | Increasingly exploratory recommendations |
| F4 | Same candidate at κ = 0.5–1.5; major change at 2.5 |
| F5 | Existing best at κ = 0.5–1.5; exploratory point at 2.5 |
| F6 | Strong progressive movement towards exploration |
| F7 | Existing best at low κ; more exploratory thereafter |
| F8 | Gradual movement away from the current best |

The results showed that κ had a meaningful influence on candidate selection for most functions.

Functions 2, 3, 6 and 8 became progressively more exploratory as κ increased. For Function 6, the distance from the current best increased from approximately 0.163 to 0.677.

Functions 4, 5 and 7 demonstrated more pronounced threshold effects. Function 4 selected the same candidate at κ = 0.5, 1.0 and 1.5, before moving substantially further into an uncertain region at κ = 2.5.

Function 5 returned the existing best observation at the lower settings, indicating that exploitation remained strongly focused on the successful region. Increasing κ to 2.5 instead produced a more exploratory recommendation.

Function 1 selected the same candidate at all four settings. Although this indicated stability within the GP acquisition analysis, the underlying model's convergence problems meant that this agreement could not be interpreted as strong evidence of predictive reliability.

Overall, the UCB results demonstrated that applying a single κ value to all eight functions could produce unnecessarily exploratory recommendations for some problems.

#### Expected Improvement hyperparameter tuning

EI was evaluated using ξ values corresponding to:

- 0% of the observed output standard deviation.
- 1% of the observed output standard deviation.
- 5% of the observed output standard deviation.
- 10% of the observed output standard deviation.

Scaling ξ to each function's observed output variation allowed the same relative settings to be compared across functions with substantially different output magnitudes.

| Function | Response to increasing ξ |
|---|---|
| F1 | Identical candidate at all settings |
| F2 | Very stable; only small movements |
| F3 | Gradually more exploratory |
| F4 | Major change between 0% and 1% |
| F5 | Identical candidate at every setting |
| F6 | Stable at 0–1%; more exploratory at 5–10% |
| F7 | Identical candidate at every setting |
| F8 | Progressively more exploratory |

EI sensitivity varied considerably between functions.

Functions 1, 5 and 7 selected the same candidate across all ξ settings, while Function 2 remained relatively stable.

Functions 3 and 8 became progressively more exploratory as ξ increased.

Functions 4 and 6 demonstrated clearer threshold effects. Function 4 was particularly sensitive, moving from a candidate approximately 0.0775 from the current best at ξ = 0% to one approximately 0.9250 away at ξ = 1%.

This was an important result because it demonstrated that relatively small changes in an acquisition parameter could substantially alter the recommended search region, even when the underlying GP remained unchanged.

### Step 4 — Assess candidate stability

The UCB and EI tuning results were compared to assess how dependent the recommendations were on the selected exploration settings.

Candidate stability was considered alongside model reliability, rather than being treated as a guarantee that a recommendation was correct.

Functions 2 and 5 showed relatively consistent recommendations across several settings. Function 2 repeatedly favoured a region close to its current best, while Function 5 continued to identify its strongest existing observation under several exploitative settings.

Functions 4 and 6 were more sensitive. Increasing exploration could move the selected candidates substantially away from the strongest observed regions.

Function 7 showed a useful distinction between exploitation and exploration. Its most exploitative UCB setting returned the existing best observation, while moderate settings identified a nearby novel candidate.

Function 8 demonstrated progressively greater movement away from the strongest observed region as the exploration settings increased.

These results suggested that relatively conservative acquisition settings might be preferable where recent queries had already produced improvements and there was strong local evidence supporting further refinement.

However, stability alone was not considered sufficient evidence. Function 1 demonstrated why this distinction matters: all tested acquisition settings agreed, but the GP itself remained unreliable.

The analysis therefore reinforced the importance of considering acquisition sensitivity together with predictive validation, historical performance and observed neighbourhoods.

### Step 5 — Retrain and validate the neural-network ensembles

The NN ensembles were retrained using the expanded Week 7 datasets.

As in previous weeks, five independently initialised networks were used for each function, and their predictions were combined to provide an alternative surrogate to the GP.

The mean final training losses were:

| Function | Mean training loss |
|---|---:|
| F1 | 0.000924 |
| F2 | 0.006460 |
| F3 | 0.000429 |
| F4 | 0.000198 |
| F5 | 0.000276 |
| F6 | 0.000301 |
| F7 | 0.000427 |
| F8 | 0.000392 |

The networks achieved low training losses across all eight functions.

Functions 3–8 generally showed particularly low losses, while Function 2 exhibited greater variation between individual models.

However, low training loss alone does not demonstrate that a model can generalise successfully to unseen observations. LOOCV was therefore used as the more important measure of predictive reliability.

#### Leave-One-Out Cross-Validation

LOOCV was used to evaluate the NN's ability to predict observations that were excluded from training.

Each observation was held out in turn, and the model was trained on the remaining observations before predicting the excluded output. The resulting Mean Absolute Error (MAE) was compared against a simple mean-prediction baseline.

| Function | NN LOOCV MAE | Mean baseline MAE | NN better? |
|---|---:|---:|---|
| F1 | 0.001776 | 0.001111 | No |
| F2 | 0.218453 | 0.237181 | Yes |
| F3 | 0.063001 | 0.050129 | No |
| F4 | 1.906639 | 7.081622 | Yes |
| F5 | 228.443505 | 1689.428734 | Yes |
| F6 | 0.235898 | 0.497547 | Yes |
| F7 | 0.376668 | 0.586378 | Yes |
| F8 | 0.358344 | 0.941538 | Yes |

The NN outperformed the mean baseline for six of the eight functions, compared with five in Week 6.

Function 2 was particularly encouraging. Its NN MAE decreased from approximately 0.2674 to 0.2185 and moved below the mean baseline for the first time.

Functions 4, 5, 6 and 8 continued to demonstrate useful predictive performance, with substantial improvements over their respective baselines.

Functions 1 and 3 remained less reliable, as their NN models continued to perform worse than the mean baseline.

These results supported retaining the NN as an independent source of evidence, while giving less weight to its recommendations for Functions 1 and 3.

#### Methodological observation — NN architecture consistency

One inconsistency identified in the Week 7 implementation was that the NN models used two hidden layers of 20 neurons, compared with the 16-neuron layers used in previous weeks.

Although the overall modelling approach remained similar, this introduced an additional variable into comparisons of NN performance between rounds.

Consequently, changes in LOOCV performance cannot be attributed entirely to the additional observations, because the architecture change may also have influenced the models' predictions and generalisation.

This was not an intentional part of the week's hyperparameter investigation, which focused on the GP acquisition functions.

The Week 7 results were retained as generated, but the architecture difference should be accounted for when interpreting changes in NN performance. Standardising the configuration in subsequent rounds would improve the consistency of comparisons.

### Step 6 — Generate and compare GP and NN candidates

The trained NN ensembles were used to generate an alternative candidate for each function.

Candidate selection was based primarily on the mean prediction across the five networks, while disagreement between ensemble members was retained as an indication of prediction consistency.

| Function | NN predicted output | Model disagreement |
|---|---:|---:|
| F1 | 0.000741 | 0.000834 |
| F2 | 0.706124 | 0.082928 |
| F3 | -0.014537 | 0.003243 |
| F4 | -0.001964 | 0.398781 |
| F5 | 7103.413178 | 20.900199 |
| F6 | -0.347172 | 0.039748 |
| F7 | 2.977398 | 0.203612 |
| F8 | 10.264397 | 0.164985 |

The NN candidates showed varying levels of agreement with the GP recommendations.

Function 2 was particularly encouraging. Following its improved LOOCV performance, the NN selected a point only approximately 0.0210 from the current best observation and very close to the low-κ UCB and low-ξ EI candidates.

Function 4 showed even stronger agreement, with the NN selecting exactly the same candidate as UCB at κ = 0.5–1.5 and EI at ξ = 0%.

Function 5 returned the existing best observation as its preferred candidate. Although this could not provide a novel query, it reinforced the model's preference for the high-performing region identified in previous rounds.

Functions 6 and 7 produced NN predictions of -0.347172 and 2.977398 respectively, both above their current best observed outputs.

Function 8 again produced an optimistic NN prediction of 10.264397. However, its previous two NN-selected queries had failed to improve the best observation, so this recommendation was treated cautiously despite its strong LOOCV performance.

Functions 1 and 3 received less weight because their networks continued to fail the validation comparison.

#### Quantitative GP–NN agreement

The Euclidean distance between each NN candidate and the candidates generated by the UCB and EI tuning settings was calculated.

Smaller distances indicate that the two surrogate approaches are identifying similar regions, while larger distances suggest greater disagreement.

| Function | Closest NN–GP distance | Interpretation |
|---|---:|---|
| F1 | 0.3384 | Clear disagreement |
| F2 | 0.0022 | Very strong agreement |
| F3 | 0.2358 | Limited agreement |
| F4 | 0.0000 | Exact agreement |
| F5 | 0.0000 | Exact agreement, but existing observation |
| F6 | 0.1857 | Moderate agreement |
| F7 | 0.1785 | Moderate agreement |
| F8 | 0.2994 | Some agreement with exploitative settings |

Functions 2 and 4 provided the strongest evidence of cross-model agreement.

For Function 2, the NN recommendation was only 0.0022 from the candidate selected by UCB at κ = 0.5 and EI at ξ = 0%.

For Function 4, agreement was exact across several acquisition settings.

Function 5 also showed exact agreement with the lower-κ UCB recommendations, although the selected point was already an existing observation.

Agreement was weaker for Functions 6 and 7, but their NN recommendations remained relatively close to the less exploratory GP candidates.

Functions 1 and 3 showed limited agreement, reinforcing the decision to give their NN recommendations less weight.

Overall, cross-model agreement provided useful supporting evidence where the NN also passed LOOCV and the recommended region was supported by strong existing observations. Agreement was not treated as an independent guarantee of improvement.

### Step 7 — Examine candidate neighbourhoods

The three nearest existing observations surrounding the leading GP and NN candidates were examined.

For UCB, κ values of 0.5, 1.5 and 2.5 were selected to represent relatively exploitative, intermediate and exploratory settings.

For EI, ξ values of 0%, 5% and 10% of the observed output standard deviation were considered.

The NN candidates were also included as an independent comparison.

The purpose was to determine whether the recommended points were close to previously successful observations or represented more exploratory choices.

**Function 1**

Both GP and NN candidates were located near observations with outputs effectively equal to zero.

Neither approach identified a clearly superior region, and the GP's continuing convergence problems limited confidence in its recommendations.

**Function 2**

The GP and NN candidates were located very close to the current best observation of 0.746962.

The strong local evidence, combined with close agreement between the models, supported further exploitation of this region.

**Function 3**

The NN candidate was relatively close to the strongest observations.

However, the NN continued to fail LOOCV, reducing confidence in its recommendation despite the favourable neighbourhood.

**Function 4**

The locally focused GP and NN candidate was surrounded by relatively strong observations.

By comparison, the more exploratory GP alternative was located near substantially poorer outputs.

This supported selecting the locally focused candidate rather than increasing exploration.

**Function 5**

Both surrogate models continued to identify the high-performing region around the current best observation of 7121.15.

However, several recommendations returned the existing best point, which could not be submitted again.

A new nearby candidate was therefore required.

**Function 6**

The NN and low-κ UCB candidates remained relatively close to the strongest observed outputs.

More exploratory GP settings moved substantially further away from this region.

Given the recent successful EI query, further local refinement appeared preferable to a large exploratory move.

**Function 7**

The NN and moderately exploratory GP candidates remained close to the new best observation of 2.929370 and other strong observations.

This supported continuing to investigate the region responsible for the previous improvement.

**Function 8**

The NN and low-exploration GP candidates remained closer to the strongest observed outputs than the more exploratory alternatives.

Given the recent unsuccessful NN queries, the neighbourhood evidence supported considering a more conservative GP candidate.

#### Neighbourhood summary

The neighbourhood analysis generally favoured relatively local recommendations this week.

Functions 2 and 4 showed particularly strong evidence, combining favourable observed neighbourhoods with close GP–NN agreement.

Functions 6 and 7 also appeared suitable for further refinement around recently successful regions.

Function 5 required a new local candidate because the strongest recommendations duplicated the current best observation.

Functions 1 and 3 remained less certain due to their modelling limitations.

These findings supported prioritising candidates with strong observed neighbourhoods where recent optimisation had been successful, rather than automatically selecting the most exploratory acquisition recommendation.

### Step 8 — Make function-specific final selections

The final candidates were selected individually using the combined evidence from historical optimisation performance, GP reliability, acquisition hyperparameter sensitivity, NN validation, cross-model agreement and neighbouring observations.

Candidates were not selected purely on their predicted outputs or acquisition scores.

| Function | Selected approach | Main reasoning |
|---|---|---|
| F1 | GP UCB/EI agreement | All GP settings agree; NN fails LOOCV |
| F2 | UCB κ = 0.5 / EI ξ = 0% | Very strong GP–NN agreement and local support |
| F3 | UCB κ = 0.5 | NN fails LOOCV; conservative GP selection |
| F4 | GP/NN agreement | Exact cross-model agreement and strong neighbourhood |
| F5 | Dedicated local refinement | Standard recommendations duplicate existing best |
| F6 | UCB κ = 0.5 | Lower exploration favoured by neighbourhood evidence |
| F7 | UCB κ = 1.5 / EI agreement | Novel candidate near recently successful region |
| F8 | UCB κ = 0.5 / EI ξ = 0% | Conservative GP candidate following unsuccessful NN queries |

#### Function 1 — GP UCB/EI agreement

Selected point: `[0.858598, 0.697368]`

Function 1 remained the most difficult to model reliably.

The GP length scale was still fixed at its lower bound of 0.00001, and the NN again failed LOOCV validation.

However, all tested UCB and EI settings selected the same GP candidate, while the NN recommendation differed substantially and lacked validation support.

With no clearly superior observed region, the consistent GP recommendation was selected, although confidence remained low.

#### Function 2 — UCB κ = 0.5 / EI ξ = 0%

Selected point: `[0.706079, 0.158030]`

Function 2 showed particularly strong agreement between the GP and NN.

The NN passed LOOCV and selected a candidate only 0.0022 from the GP recommendation.

The selected point was approximately 0.0211 from the current best observation of 0.746962, and the GP predicted an output of 0.784588.

Given the strong model agreement and favourable neighbourhood, further local refinement was preferred to increasing exploration.

#### Function 3 — UCB κ = 0.5

Selected point: `[0.450320, 0.561567, 0.490526]`

Function 3 achieved another small improvement in the previous round, increasing its best from -0.016451 to -0.015498.

The NN proposed a more locally focused point but continued to fail LOOCV, so its recommendation received limited weight.

The low-κ UCB candidate provided the least exploratory GP recommendation, with a predicted output of -0.018745.

This was selected as a relatively conservative continuation of the search, retaining some exploration without relying on the less reliable NN model.

#### Function 4 — GP/NN agreement

Selected point: `[0.384332, 0.391111, 0.336887, 0.416488]`

Function 4 produced a new best of -0.200431 in the previous round.

This week's NN recommendation exactly matched UCB at κ = 0.5–1.5 and EI at ξ = 0%.

The candidate also remained within a relatively strong observed neighbourhood, while more exploratory alternatives moved towards regions associated with poor outputs.

Although the NN prediction of -0.001964 was optimistic and accompanied by considerable model disagreement, the agreement on candidate location provided useful supporting evidence.

This was one of the stronger selections of the week.

#### Function 5 — Local refinement towards the upper boundary

Selected point: `[1.000000, 1.000000, 1.000000, 1.000000]`

Function 5 required a different approach because the NN and several low-κ UCB settings returned the existing best observation.

As that point could not be resubmitted, a dedicated local search was performed around the current best.

Both surrogate models supported movement towards the upper boundary of the search space.

The selected point received a GP prediction of approximately 7920.78 and an NN prediction of approximately 8059.00.

It was only 0.0642 from the current best observation of 7121.15, allowing further exploitation of the successful region while remaining a novel query.

The predicted improvement should nevertheless be treated cautiously because the selected point lies directly on the boundary and has not yet been evaluated by the objective function.

#### Function 6 — UCB κ = 0.5

Selected point: `[0.387946, 0.435172, 0.345126, 0.743513, 0.076825]`

Function 6 improved from -0.456874 to -0.379740 following the previous EI-selected query.

However, the hyperparameter analysis showed that increasing exploration moved the GP recommendations rapidly away from the strongest observed region.

The low-κ UCB candidate remained relatively close to the new best and had a predicted output of -0.387785.

The validated NN also supported continued refinement rather than substantial exploration.

The lower-κ candidate was therefore selected as a conservative continuation of the successful region.

#### Function 7 — UCB κ = 1.5 / EI agreement

Selected point: `[0.054205, 0.216830, 0.421532, 0.346756, 0.385001, 0.543902]`

Function 7 achieved a substantial improvement in the previous round, increasing its best observation to 2.929370.

UCB at κ = 0.5 returned the existing best observation, which could not be resubmitted.

UCB at κ = 1.0–1.5 and the tested EI settings identified the same nearby novel candidate.

The GP predicted an output of 2.620240, while the NN also passed LOOCV and selected a point approximately 0.1785 away.

The candidate was selected to continue investigating the region responsible for the previous improvement while avoiding a duplicate query.

#### Function 8 — UCB κ = 0.5 / EI ξ = 0%

Selected point: `[0.030327, 0.231267, 0.050266, 0.208295, 0.818520, 0.414744, 0.307096, 0.583908]`

The NN continued to outperform the LOOCV baseline and predicted an improvement to 10.264397.

However, the previous two NN-selected queries had failed to improve the current best observation of 9.911249.

The low-exploration GP candidate predicted 9.945039 and remained closer to the strongest observed neighbourhood than the more exploratory alternatives.

UCB at κ = 0.5 and EI at ξ = 0% also agreed on this candidate.

A more conservative GP recommendation was therefore preferred for this round.

### Step 9 — Validate the final proposed queries

The eight selected candidates were subjected to a final validation check before submission.

Each point was checked for:

- Correct dimensionality for its function.
- Coordinates within the permitted [0, 1] search space.
- Duplicate observations.
- Distance from the nearest existing observation.
- Correct six-decimal formatting.

All eight candidates passed the validation checks.

The nearest-observation distances ranged from approximately 0.0211 for Function 2 to 0.3452 for Function 8, reflecting the different levels of local refinement and exploration selected for each problem.

Function 5's boundary candidate `[1, 1, 1, 1]` was confirmed to be within bounds and distinct from the existing observations.

#### Final Week 7 submission points

```text
Function 1: 0.858598-0.697368
Function 2: 0.706079-0.158030
Function 3: 0.450320-0.561567-0.490526
Function 4: 0.384332-0.391111-0.336887-0.416488
Function 5: 1.000000-1.000000-1.000000-1.000000
Function 6: 0.387946-0.435172-0.345126-0.743513-0.076825
Function 7: 0.054205-0.216830-0.421532-0.346756-0.385001-0.543902
Function 8: 0.030327-0.231267-0.050266-0.208295-0.818520-0.414744-0.307096-0.583908
```

## Weekly summary and next steps

Week 7 represented a further shift towards a more evidence-driven and function-specific optimisation strategy. Five of the eight queries submitted in the previous round produced new best observations, including particularly strong improvements for Functions 5 and 7. This provided encouraging evidence that combining model validation, acquisition-function comparison and local neighbourhood analysis was helping to identify promising regions.

The main development this week was testing the sensitivity of the GP acquisition functions to different exploration settings. Comparing UCB κ values of 0.5, 1.0, 1.5 and 2.5, alongside EI ξ values scaled to the observed output variation, showed that some functions were relatively stable while others changed considerably as greater weight was placed on exploration.

Functions 2 and 5 showed relatively consistent recommendations, while Functions 4 and 6 were more sensitive to changes in the acquisition parameters. This reinforced the importance of selecting acquisition settings individually rather than applying a single configuration across all eight functions.

The NN surrogate also continued to provide useful alternative recommendations, outperforming the mean baseline in LOOCV for six of the eight functions. Function 2 was particularly encouraging, showing improved predictive performance and producing a candidate very close to the GP recommendation. Function 4 also showed exact agreement between the GP and NN candidates under several acquisition settings.

One methodological limitation was the change from 16 to 20 neurons per hidden layer in the Week 7 NN implementation. Although the overall approach remained similar, this introduced an additional variable into comparisons with earlier weeks. The architecture should therefore be standardised in subsequent rounds to improve consistency when assessing changes in validation performance.

The final selections reflected the different levels of evidence available for each function. Functions 2 and 4 focused on regions supported by strong agreement between the GP and NN, while Functions 3, 6 and 8 used more conservative GP settings to remain relatively close to successful observations. Function 7 continued to investigate the region that produced its substantial improvement in the previous round.

Function 1 remained difficult to model reliably, with its GP length scale again reaching the lower bound and the NN failing LOOCV validation. Its selection therefore remained lower confidence despite agreement between the GP acquisition settings.

Function 5 required a slightly different approach because several models returned the existing best observation. A dedicated local refinement search was performed, resulting in the selection of `[1, 1, 1, 1]`. Both surrogate models predicted an improvement towards this boundary, although this remains an untested prediction that will need to be assessed using the next returned output.

Overall, Week 7 demonstrated the value of combining historical optimisation performance, predictive validation, acquisition-function sensitivity and observed neighbourhoods when making function-specific decisions. It also highlighted the importance of maintaining consistent modelling configurations when comparing results across iterations.

The next round will assess whether the selected candidates improve the observed maxima and whether the more locally focused strategy continues to produce successful results.

