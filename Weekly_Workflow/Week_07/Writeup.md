
# Week 7 — Acquisition Hyperparameter Tuning and Candidate Stability

**Module 18 | Week 7**

## Objective

Week 7 builds on the function-specific optimisation strategy developed during previous rounds. Rather than introducing another modelling approach, the focus this week was on tuning the acquisition-function hyperparameters and assessing how robust the resulting candidate recommendations were.

Gaussian Process Regression (GPR) remained the primary surrogate, using Expected Improvement (EI) and Upper Confidence Bound (UCB) to identify promising query points. However, instead of relying on a single exploration setting, I tested multiple values of the acquisition parameters to examine how sensitive the recommendations were to the exploration–exploitation balance.

The neural-network (NN) surrogate introduced in Week 5 was retained as an independent comparison, with its reliability assessed using Leave-One-Out Cross-Validation (LOOCV).

The final objective was to select one candidate per function using a combination of GP predictions, acquisition-function stability, NN validation, historical optimisation performance and the observed neighbourhood surrounding each candidate.

## Methodology

The analysis followed eight main stages:

1. Incorporate the Week 6 outputs and assess improvements against previous best observations.
2. Refit and diagnose the Gaussian Process models.
3. Test multiple UCB κ and EI ξ settings and assess candidate stability.
4. Retrain and validate the neural-network ensembles.
5. Generate NN candidates and compare them with the tuned GP recommendations.
6. Examine the nearest existing observations around leading candidates.
7. Select one candidate per function using the combined evidence.
8. Validate the final coordinates and prepare the submission points.

## Step 1 — Incorporate and Assess the Week 6 Results

The eight outputs returned from the Week 6 queries were incorporated into the accumulated datasets and compared with the previous best observations.

| Function | Previous Best | New Output | Updated Best | Improved? |
|:--|--:|--:|--:|:--|
| 1 | 0.000000 | ≈0 | 0.000000 | No |
| 2 | 0.746962 | 0.471844 | 0.746962 | No |
| 3 | -0.016451 | -0.015498 | -0.015498 | Yes |
| 4 | -0.300645 | -0.200431 | -0.200431 | Yes |
| 5 | 6691.931674 | 7121.151955 | 7121.151955 | Yes |
| 6 | -0.456874 | -0.379740 | -0.379740 | Yes |
| 7 | 2.536939 | 2.929370 | 2.929370 | Yes |
| 8 | 9.911249 | 9.840530 | 9.911249 | No |

**Five of the eight Week 6 queries produced new best observations.**

Function 5 achieved another substantial improvement, increasing from 6691.93 to 7121.15. This provided further support for the neural-network approach, which had been responsible for selecting the previous query.

Function 7 also improved considerably, increasing from 2.536939 to 2.929370 following a candidate supported by agreement between UCB and EI.

Functions 3, 4 and 6 improved using EI-selected candidates. Function 4 increased from -0.300645 to -0.200431, while Function 6 improved from -0.456874 to -0.379740.

The unsuccessful queries were also informative. Function 2 returned 0.471844, below its previous best of 0.746962. Function 8's NN-selected query returned 9.840530, slightly below the best of 9.911249. This represented the second consecutive round in which the NN had failed to improve Function 8.

Function 1 again returned a value effectively equal to zero, providing little additional information about its optimum.

Overall, the results supported retaining a mixed, function-specific approach rather than selecting a single surrogate or acquisition function for all eight problems.

## Step 2 — Refit and Diagnose the Gaussian Processes

The Gaussian Process models were refitted using all observations available at the beginning of Week 7.

The fitted RBF length scales were examined to assess whether the additional observations had changed the behaviour of the models.

| Function | RBF Length Scale |
|:--|--:|
| 1 | 0.00001 |
| 2 | 0.0431 |
| 3 | 0.193 |
| 4 | 0.436 |
| 5 | 0.363 |
| 6 | 0.558 |
| 7 | 0.300 |
| 8 | 1.020 |

Most functions continued to produce reasonable fitted length scales.

Functions 3, 4 and 6 showed relatively small increases from the previous week, suggesting that the additional observations had not substantially disrupted their fitted relationships.

Function 7's length scale increased from approximately 0.248 to 0.300 following its new best result.

Function 6 continued to show a stable GP fit, with its length scale increasing from 0.535 to 0.558. This was encouraging given its earlier convergence problems.

**Function 1 remained the main exception.** Its fitted length scale again reached the lower bound of 0.00001, suggesting that the RBF GP was still struggling to identify a meaningful smooth relationship across the available observations.

Consequently, GP recommendations for Function 1 continued to be treated cautiously.

### Prediction and Uncertainty Diagnostics

I generated 100,000 candidate points for each function and evaluated the GP predictions and uncertainty across the search space.

| Function | Key Diagnostic Result |
|:--|:--|
| 1 | Predicted range: 0.000595; maximum uncertainty: 0.001626 |
| 2 | Predicted maximum: 0.7905; maximum uncertainty: 0.2550 |
| 3 | Predicted maximum: -0.0118; maximum uncertainty: 0.0782 |
| 4 | Predicted maximum: -0.2003; maximum uncertainty: 7.4108 |
| 5 | Predicted maximum: 7121.16; maximum uncertainty: 2074.52 |
| 6 | Predicted maximum: -0.3656; maximum uncertainty: 0.5196 |
| 7 | Predicted maximum: 2.9294; maximum uncertainty: 0.7721 |
| 8 | Predicted maximum: 9.9641; maximum uncertainty: 0.6576 |

Functions 2–8 showed meaningful variation in predicted outputs and uncertainty, providing a useful basis for acquisition-function tuning.

For several functions, the GP's highest predicted values remained close to the strongest observed outputs, suggesting that the models continued to identify the recently successful regions.

Function 1 remained difficult to interpret, with predictions concentrated around zero and its fitted kernel continuing to exhibit convergence issues.

The same candidate sets were retained for the acquisition-function comparisons so that differences between recommendations could be attributed to changes in hyperparameters rather than different randomly generated points.

## Step 3 — Tune the Acquisition-Function Hyperparameters

The main methodological development this week was testing multiple exploration settings for UCB and EI.

### UCB Hyperparameter Tuning

UCB was evaluated using four κ values:

- κ = 0.5
- κ = 1.0
- κ = 1.5
- κ = 2.5

Lower κ values place greater emphasis on the predicted mean, encouraging exploitation. Higher κ values increase the influence of uncertainty, encouraging exploration.

The resulting candidate movements were compared across all eight functions.

| Function | Response to Increasing κ |
|:--|:--|
| 1 | No change; identical candidate at all settings |
| 2 | Gradual movement away from the current best |
| 3 | Increasingly exploratory recommendations |
| 4 | Same candidate at κ = 0.5–1.5; major change at 2.5 |
| 5 | Existing best at κ = 0.5–1.5; exploratory point at 2.5 |
| 6 | Strong progressive movement towards exploration |
| 7 | Existing best at low κ; more exploratory thereafter |
| 8 | Gradual movement away from the current best |

The results demonstrated that κ had a meaningful influence on candidate selection for most functions.

Functions 2, 3, 6 and 8 became progressively more exploratory as κ increased.

Functions 4, 5 and 7 showed more pronounced threshold effects, where relatively small changes in κ caused a substantial change in the selected candidate.

Function 1 returned the same candidate at every setting. Although this suggested stability in the acquisition recommendations, the underlying GP remained unreliable, so this agreement was not treated as strong evidence on its own.

The results supported using function-specific exploration settings rather than automatically applying κ = 1.5 across all eight functions.

### Expected Improvement Hyperparameter Tuning

EI was evaluated using ξ values equivalent to:

- 0% of the observed output standard deviation.
- 1% of the observed output standard deviation.
- 5% of the observed output standard deviation.
- 10% of the observed output standard deviation.

Scaling ξ to the observed output variation allowed the settings to be compared across functions with very different output magnitudes.

| Function | Response to Increasing ξ |
|:--|:--|
| 1 | No change |
| 2 | Very stable; only small candidate movements |
| 3 | Gradually more exploratory |
| 4 | Major change between 0% and 1% |
| 5 | No change |
| 6 | Stable at 0–1%; more exploratory at 5–10% |
| 7 | No change |
| 8 | Progressively more exploratory |

EI sensitivity varied substantially between functions.

Functions 1, 5 and 7 returned identical candidates across all ξ settings, while Function 2 remained relatively stable.

Functions 3 and 8 became more exploratory as ξ increased.

Functions 4 and 6 demonstrated stronger threshold effects. Function 4 was particularly sensitive, moving from a candidate approximately 0.0775 from the current best at ξ = 0% to one approximately 0.9250 away at ξ = 1%.

This was an important observation because it showed that the acquisition-function settings could materially affect the recommended search region, even when the underlying GP remained unchanged.

### Candidate Stability

Comparing the UCB and EI tuning results provided a measure of how dependent the recommendations were on the selected exploration parameters.

For Functions 2 and 5, several settings continued to favour relatively consistent regions, suggesting that their recommendations were comparatively robust.

For Functions 4 and 6, increasing exploration could move the selected candidate substantially away from the strongest observed regions.

This suggested that relatively conservative settings might be preferable where recent queries had already produced improvements and there was strong local evidence supporting further refinement.

Overall, the hyperparameter analysis reinforced the importance of considering candidate stability alongside acquisition scores and predicted outputs.

## Step 4 — Retrain and Validate the Neural Networks

The neural-network ensembles were retrained using the expanded datasets.

As in previous weeks, five independently initialised networks were used for each function, and their predictions were combined to provide an alternative surrogate to the GP.

The mean final training losses were:

| Function | Mean Training Loss |
|:--|--:|
| 1 | 0.000924 |
| 2 | 0.006460 |
| 3 | 0.000429 |
| 4 | 0.000198 |
| 5 | 0.000276 |
| 6 | 0.000301 |
| 7 | 0.000427 |
| 8 | 0.000392 |

The ensembles achieved low training losses across all eight functions.

Functions 3–8 generally showed particularly low losses, while Function 2 exhibited greater variation between individual models.

However, training performance alone was not sufficient to establish whether the networks could predict unseen observations.

I therefore repeated LOOCV, comparing the NN's Mean Absolute Error (MAE) with a simple mean-prediction baseline.

### Leave-One-Out Cross-Validation

| Function | NN LOOCV MAE | Mean Baseline MAE | NN Better? |
|:--|--:|--:|:--|
| 1 | 0.001776 | 0.001111 | No |
| 2 | 0.218453 | 0.237181 | Yes |
| 3 | 0.063001 | 0.050129 | No |
| 4 | 1.906639 | 7.081622 | Yes |
| 5 | 228.443505 | 1689.428734 | Yes |
| 6 | 0.235898 | 0.497547 | Yes |
| 7 | 0.376668 | 0.586378 | Yes |
| 8 | 0.358344 | 0.941538 | Yes |

The NN outperformed the mean baseline for **six of the eight functions**, compared with five in Week 6.

Function 2 was particularly encouraging. Its NN MAE decreased from approximately 0.2674 in the previous week to 0.2185, allowing it to outperform the baseline again.

Functions 4, 5, 6 and 8 continued to demonstrate useful predictive performance, with substantial improvements over their respective baselines.

Functions 1 and 3 remained less reliable, as their NN models continued to perform worse than the mean baseline.

These results supported retaining the NN as an independent source of evidence, while giving less weight to its recommendations for Functions 1 and 3.

### Methodological Observation — Neural-Network Architecture

One inconsistency identified in the Week 7 implementation was that the neural-network models used two hidden layers of **20 neurons**, rather than the 16-neuron layers used in previous weeks.

Although the overall modelling approach remained similar, this introduced an additional variable into the comparison of neural-network performance between rounds.

Consequently, improvements in LOOCV performance cannot be attributed entirely to the additional observations, because the change in architecture may also have influenced the models' predictions and generalisation.

This was not an intentional part of the week's hyperparameter investigation, which focused primarily on the GP acquisition functions.

The architecture should therefore be standardised in subsequent rounds to improve the consistency of comparisons.

## Step 5 — Generate and Compare Neural-Network Candidates

The trained NN ensembles were used to generate alternative candidate points for each function.

The mean prediction across the five networks was used to identify promising points, while disagreement between the networks was retained as an indication of prediction consistency.

| Function | NN Predicted Output | Model Disagreement |
|:--|--:|--:|
| 1 | 0.000741 | 0.000834 |
| 2 | 0.706124 | 0.082928 |
| 3 | -0.014537 | 0.003243 |
| 4 | -0.001964 | 0.398781 |
| 5 | 7103.413178 | 20.900199 |
| 6 | -0.347172 | 0.039748 |
| 7 | 2.977398 | 0.203612 |
| 8 | 10.264397 | 0.164985 |

The NN candidates provided useful comparisons with the tuned GP recommendations.

**Function 2** was particularly encouraging. Following its improved LOOCV performance, the NN selected a point only 0.0210 from the current best observation and very close to the low-κ UCB and low-ξ EI candidates.

**Function 4** showed even stronger agreement, with the NN selecting exactly the same candidate as UCB at κ = 0.5–1.5 and EI at ξ = 0%.

**Function 5** returned the existing best observation as its preferred candidate. Although this could not be submitted again, it reinforced the NN's preference for the high-performing region identified in previous rounds.

**Functions 6 and 7** produced promising NN predictions of -0.347172 and 2.977398 respectively, both above their current best observations.

**Function 8** again produced an optimistic NN prediction, this time 10.264397. However, the previous two NN-selected queries had failed to improve the best output, so this recommendation was treated cautiously despite its strong LOOCV performance.

Functions 1 and 3 received less weight because their networks continued to fail the validation comparison.

### Quantitative GP–NN Agreement

I calculated the Euclidean distance between each NN candidate and the candidates generated by the different UCB and EI settings.

| Function | Closest NN–GP Distance | Interpretation |
|:--|--:|:--|
| 1 | 0.3384 | Clear disagreement |
| 2 | 0.0022 | Very strong agreement |
| 3 | 0.2358 | Limited agreement |
| 4 | 0.0000 | Exact agreement |
| 5 | 0.0000 | Exact agreement, but existing observation |
| 6 | 0.1857 | Moderate agreement |
| 7 | 0.1785 | Moderate agreement |
| 8 | 0.2994 | Some agreement with exploitative settings |

Functions 2 and 4 showed the strongest evidence of agreement between the two surrogate approaches.

For Function 2, the NN recommendation was only 0.0022 from the GP candidate selected by UCB at κ = 0.5 and EI at ξ = 0%.

For Function 4, the agreement was exact across several settings.

Function 5 also showed exact agreement with the low-κ UCB recommendation, although the point was already present in the dataset.

Agreement was weaker for Functions 6 and 7, but their NN recommendations remained relatively close to the less exploratory GP candidates.

Functions 1 and 3 showed limited agreement, reinforcing the decision to give their NN recommendations less weight.

The results demonstrated that cross-model agreement could provide useful supporting evidence, particularly when the NN also passed LOOCV and the candidate was supported by strong neighbouring observations.

## Step 6 — Neighbourhood Analysis

The three nearest existing observations surrounding the leading GP and NN candidates were examined.

For UCB, κ values of 0.5, 1.5 and 2.5 were used to represent different exploration levels.

For EI, ξ values of 0%, 5% and 10% of the observed output standard deviation were considered.

The NN candidates were also included.

The objective was to determine whether the recommended points were close to previously successful observations or represented more exploratory choices.

### Function 1

Both GP and NN candidates were located near observations with outputs effectively equal to zero.

Neither approach identified a clearly superior region, and the GP's continuing convergence problems limited confidence in its recommendations.

### Function 2

The GP and NN candidates were located very close to the current best observation of 0.746962.

The strong local evidence, combined with the close agreement between the models, supported further exploitation of this region.

### Function 3

The NN candidate was relatively close to the strongest observations.

However, the NN continued to fail LOOCV, reducing confidence in its recommendation despite the favourable neighbourhood.

### Function 4

The locally focused GP and NN candidate was surrounded by relatively strong observations.

By comparison, the more exploratory GP alternative was located near substantially poorer outputs.

This supported selecting the locally focused candidate rather than increasing exploration.

### Function 5

Both surrogate models continued to identify the high-performing region around the current best observation of 7121.15.

However, several recommendations returned the existing best point, which could not be submitted again.

A new nearby candidate was therefore required.

### Function 6

The NN and low-κ UCB candidates remained relatively close to the strongest observed outputs.

More exploratory GP settings moved substantially further away from this region.

Given the recent successful EI query, further local refinement appeared preferable to a large exploratory move.

### Function 7

The NN and moderately exploratory GP candidates remained close to the new best observation of 2.929370 and other strong observations.

This supported continuing to investigate the region responsible for the previous improvement.

### Function 8

The NN and low-exploration GP candidates remained closer to the strongest observed outputs than the more exploratory alternatives.

Given the recent unsuccessful NN queries, the neighbourhood evidence supported considering a more conservative GP candidate.

### Neighbourhood Summary

The neighbourhood analysis generally favoured relatively local recommendations.

Functions 2 and 4 showed particularly strong evidence, combining favourable observed neighbourhoods with close GP–NN agreement.

Functions 6 and 7 also appeared suitable for further refinement around recently successful regions.

Function 5 required a new local candidate because the strongest recommendations duplicated the current best.

Functions 1 and 3 remained less certain due to their modelling limitations.

## Step 7 — Final Candidate Selection

The final candidates were selected individually using the combined evidence from historical optimisation performance, GP reliability, acquisition hyperparameter sensitivity, NN validation, cross-model agreement and neighbouring observations.

### Function 1 — GP UCB/EI Agreement

**Selected point:** `[0.858598, 0.697368]`

Function 1 remained the most difficult to model reliably.

The GP length scale was still fixed at its lower bound, and the NN again failed LOOCV.

However, all tested UCB and EI settings selected the same GP candidate, while the NN recommendation differed substantially and lacked validation support.

With no clearly superior observed region, the consistent GP recommendation was selected, although confidence remained low.

### Function 2 — UCB κ = 0.5 / EI ξ = 0%

**Selected point:** `[0.706079, 0.158030]`

Function 2 showed particularly strong agreement between the GP and NN.

The NN passed LOOCV and selected a candidate only 0.0022 from the GP recommendation.

The selected point was approximately 0.0211 from the current best observation of 0.746962, and the GP predicted an output of 0.784588.

Given the strong model agreement and favourable neighbourhood, further local refinement was preferred to increasing exploration.

### Function 3 — UCB κ = 0.5

**Selected point:** `[0.450320, 0.561567, 0.490526]`

Function 3 achieved another small improvement in the previous round, increasing the best from -0.016451 to -0.015498.

The NN proposed a more locally focused point but continued to fail LOOCV, so its recommendation received limited weight.

The low-κ UCB candidate provided the least exploratory GP recommendation, with a predicted output of -0.018745.

This was selected as a relatively conservative continuation of the search.

### Function 4 — GP/NN Agreement

**Selected point:** `[0.384332, 0.391111, 0.336887, 0.416488]`

Function 4 produced a new best of -0.200431 in the previous round.

This week's NN recommendation exactly matched UCB at κ = 0.5–1.5 and EI at ξ = 0%.

The candidate also remained within a relatively strong observed neighbourhood, while more exploratory alternatives moved towards regions associated with poor outputs.

Although the NN prediction of -0.001964 was optimistic and accompanied by considerable model disagreement, the agreement on candidate location provided useful supporting evidence.

This was one of the stronger selections of the week.

### Function 5 — Local Refinement Towards the Upper Boundary

**Selected point:** `[1.000000, 1.000000, 1.000000, 1.000000]`

Function 5 required a different approach because the NN and several low-κ UCB settings returned the existing best observation.

As that point could not be resubmitted, a dedicated local search was performed around the current best.

Both surrogate models supported movement towards the upper boundary of the search space.

The selected point received a GP prediction of approximately 7920.78 and an NN prediction of approximately 8059.00.

It was only 0.0642 from the current best observation of 7121.15, allowing further exploitation of the successful region while remaining a novel query.

The prediction should nevertheless be treated cautiously because the selected point lies directly on the boundary.

### Function 6 — UCB κ = 0.5

**Selected point:** `[0.387946, 0.435172, 0.345126, 0.743513, 0.076825]`

Function 6 improved from -0.456874 to -0.379740 following the previous EI-selected query.

However, the hyperparameter analysis showed that increasing exploration moved the GP recommendations rapidly away from the strongest observed region.

The low-κ UCB candidate remained relatively close to the new best and had a predicted output of -0.387785.

The validated NN also supported continued refinement rather than substantial exploration.

The lower-κ candidate was therefore selected as a conservative continuation of the successful region.

### Function 7 — UCB κ = 1.5 / EI Agreement

**Selected point:** `[0.054205, 0.216830, 0.421532, 0.346756, 0.385001, 0.543902]`

Function 7 achieved a substantial improvement in the previous round, increasing its best observation to 2.929370.

UCB at κ = 0.5 returned the existing best observation, which could not be resubmitted.

UCB at κ = 1.0–1.5 and the tested EI settings identified the same nearby novel candidate.

The GP predicted an output of 2.620240, while the NN also passed LOOCV and selected a point approximately 0.1785 away.

The candidate was selected to continue exploring around the region responsible for the previous improvement.

### Function 8 — UCB κ = 0.5 / EI ξ = 0%

**Selected point:** `[0.030327, 0.231267, 0.050266, 0.208295, 0.818520, 0.414744, 0.307096, 0.583908]`

The NN continued to outperform the LOOCV baseline and predicted an improvement to 10.264397.

However, the previous two NN-selected queries had failed to improve the current best observation of 9.911249.

The low-exploration GP candidate predicted 9.945039 and remained closer to the strongest observed neighbourhood than the more exploratory alternatives.

UCB at κ = 0.5 and EI at ξ = 0% also agreed on this candidate.

A more conservative GP recommendation was therefore preferred for this round.

## Step 8 — Final Validation and Submission

The selected candidates were checked for:

- Correct dimensionality for each function.
- Coordinates within the permitted `[0, 1]` search space.
- Duplicate observations.
- Distance from the nearest existing observation.
- Correct six-decimal formatting.

All eight candidates passed the validation checks.

The nearest-observation distances ranged from approximately 0.0211 for Function 2 to 0.3452 for Function 8.

Function 5's boundary candidate `[1, 1, 1, 1]` was also confirmed to be within bounds and distinct from the existing observations.

### Final Week 7 Submission Points

    Function 1: 0.858598-0.697368
    Function 2: 0.706079-0.158030
    Function 3: 0.450320-0.561567-0.490526
    Function 4: 0.384332-0.391111-0.336887-0.416488
    Function 5: 1.000000-1.000000-1.000000-1.000000
    Function 6: 0.387946-0.435172-0.345126-0.743513-0.076825
    Function 7: 0.054205-0.216830-0.421532-0.346756-0.385001-0.543902
    Function 8: 0.030327-0.231267-0.050266-0.208295-0.818520-0.414744-0.307096-0.583908

## Overall Thoughts on Week 7

Week 7 represented a further shift towards a more evidence-driven and function-specific optimisation strategy. Five of the eight queries submitted in the previous round produced new best observations, including particularly strong improvements for Functions 5 and 7. This provided encouraging evidence that combining model validation, acquisition-function comparison and local neighbourhood analysis was helping to identify promising regions.

The main development this week was testing the sensitivity of the Gaussian Process acquisition functions to different exploration settings. Comparing UCB κ values of 0.5, 1.0, 1.5 and 2.5, alongside EI ξ values scaled to the observed output variation, showed that some functions were relatively stable while others changed considerably as greater weight was placed on exploration. Functions 2 and 5 showed relatively consistent recommendations, while Functions 4 and 6 were more sensitive to changes in the acquisition parameters. This reinforced the importance of selecting acquisition settings individually rather than applying a single configuration across all eight functions.

The neural-network surrogate also continued to provide useful alternative recommendations, outperforming the mean baseline in LOOCV for six of the eight functions. Function 2 was particularly encouraging, returning to better-than-baseline predictive performance and producing a candidate very close to the GP recommendation. Function 4 also showed exact agreement between the GP and NN candidates under several acquisition settings, providing additional support for its selected region.

One methodological inconsistency identified during the analysis was that the Week 7 neural-network implementation used two hidden layers of 20 neurons, compared with the 16-neuron layers used in previous weeks. Although the overall modelling approach remained similar, this introduces a limitation when comparing predictive performance between rounds. Improvements in LOOCV results cannot be attributed entirely to the additional observations, as the change in architecture may also have influenced the model's predictions and generalisation. This was not an intentional part of the week's hyperparameter tuning, which focused on the GP acquisition functions, and the NN architecture should therefore be standardised in subsequent rounds to ensure more meaningful comparisons.

The final selections reflected the different levels of evidence available for each function. Functions 2 and 4 focused on regions supported by strong agreement between the GP and NN, while Functions 3, 6 and 8 used more conservative GP settings to remain relatively close to successful observations. Function 7 continued to explore around the region that produced its substantial improvement in the previous round. Function 1 remained difficult to model reliably, with its GP length scale again reaching the lower bound and the NN failing LOOCV validation.

Function 5 required a slightly different approach because several models returned the existing best observation, which could not be submitted again. A local refinement search was therefore performed, resulting in the selection of `[1, 1, 1, 1]`. Both surrogate models predicted an improvement towards this boundary, although this remains an extrapolation that will need to be assessed using the next returned output.

Overall, Week 7 demonstrated the value of combining historical optimisation performance, predictive validation, acquisition-function sensitivity and observed neighbourhoods when making function-specific decisions. It also highlighted the importance of maintaining consistent modelling configurations when comparing results across iterations.

The next round will provide an opportunity to assess whether the selected candidates improve the observed maxima and whether the more locally focused strategy continues to produce successful results.
