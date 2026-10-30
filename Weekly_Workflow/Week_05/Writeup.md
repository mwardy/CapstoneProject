
# Week 5 — Introducing Neural Networks as an Alternative Surrogate Model

**Imperial College AI/ML Capstone Project | Module 16**

## Objective

The objective of Week 5 was to incorporate the latest observations, reassess the Gaussian Process (GP) models and investigate whether neural networks could provide an additional method for identifying promising regions of the unknown functions.

Previous rounds had established GP regression and acquisition functions as the main Bayesian Optimisation approach, while alternative strategies had been introduced where the Gaussian Processes provided limited guidance. Week 4 also demonstrated the potential benefits of Expected Improvement (EI) and a more exploitative Upper Confidence Bound (UCB) setting.

This week introduced a small neural-network surrogate for each function. Rather than replacing the GP models, the neural networks were intended to provide independent predictions that could support or challenge the existing acquisition-based recommendations.

The final query selections would continue to consider model reliability, historical performance and the balance between exploration and exploitation.

## 1. Incorporate and Assess the New Observations

The outputs returned from the Week 4 queries were added to the existing datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | -0.005913 | 0.000000 | No improvement |
| 2 | 0.611205 | 0.507418 | 0.611205 | No improvement |
| 3 | -0.020167 | -0.107206 | -0.020167 | No improvement |
| 4 | -0.300645 | -1.438643 | -0.300645 | No improvement |
| 5 | 4595.479318 | 4857.449601 | 4857.449601 | New best |
| 6 | -0.458511 | -2.043880 | -0.458511 | No improvement |
| 7 | 2.139551 | 2.536939 | 2.536939 | New best |
| 8 | 9.821986 | 9.911249 | 9.911249 | New best |

Three of the eight queries produced new best observations.

**Function 5** continued its strong upward trend, improving from approximately 4595.48 to 4857.45. This represented its fourth consecutive improvement and provided further evidence that the optimisation had identified a promising high-performing region.

**Function 7** also improved from approximately 2.14 to 2.54, while Function 8 increased from approximately 9.82 to 9.91.

These results provided encouraging evidence for the more exploitative EI-based approach introduced in Week 4.

Functions 2, 3 and 4 did not improve despite also using EI, demonstrating that acquisition-based predictions do not guarantee better outcomes.

Function 1 again failed to improve under maximum-distance exploration. Function 6's highly exploratory query also performed considerably worse than its existing best, suggesting that repeating aggressive exploration might not be appropriate.

## 2. Refit and Diagnose the Gaussian Process Models

The RBF Gaussian Processes were refitted using all available observations, retaining the existing kernel configuration and optimisation settings.

The fitted length scales were examined to assess whether the additional data had improved the models.

| Function | Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0494 |
| 3 | 0.173 |
| 4 | 0.414 |
| 5 | 0.467 |
| 6 | 1e-05 |
| 7 | 0.330 |
| 8 | 0.979 |

Functions 2–5, 7 and 8 produced fitted length scales above the minimum bound, suggesting that their Gaussian Processes continued to identify relationships within the available observations.

**Functions 1 and 6** again converged to the lower length-scale bound of `1e-05`, indicating that the additional observations had not resolved their previous modelling difficulties.

### 2.1 Prediction and Uncertainty Diagnostics

To assess whether the models could meaningfully distinguish between candidate regions, the predicted means and uncertainties were evaluated across 100,000 randomly generated points.

| Function | Predicted Mean Range | Uncertainty Range |
|---|---:|---:|
| 1 | 0.000680 | 0.001703 |
| 2 | 0.699132 | 0.237428 |
| 3 | 0.389118 | 0.077909 |
| 4 | 34.460542 | 7.316418 |
| 5 | 6264.230066 | 1228.812886 |
| 6 | 1.149715 | 0.509894 |
| 7 | 2.487448 | 0.642344 |
| 8 | 4.801047 | 0.669038 |

The diagnostic results showed meaningful variation in predictions and uncertainty across most functions.

Interestingly, **Function 6 now showed variation in both predicted mean and uncertainty**, despite its fitted length scale remaining at the lower bound. This differed from the effectively constant predictions observed in previous rounds.

Function 1 also showed some variation, although its persistent convergence warnings continued to raise concerns about generalisation.

These findings suggested that neither function should be judged solely on its fitted length scale. However, their previous instability meant that GP-based recommendations would still require additional scrutiny.

## 3. Compare UCB and Expected Improvement

Rather than repeating the full exploration-parameter comparison from Week 4, UCB was retained with the more exploitative setting of `κ = 1.5` and compared directly with Expected Improvement.

Both acquisition functions were evaluated using the same 100,000 candidate points for each function, allowing their recommendations to be compared consistently.

Function 1 was initially excluded from this comparison because of its continuing GP instability, while Functions 2–8 were evaluated.

### 3.1 Comparison of Candidate Recommendations

| Function | UCB Predicted Output | EI Predicted Output | Distance Between Candidates |
|---|---:|---:|---:|
| 2 | 0.431873 | 0.429437 | 0.0253 |
| 3 | -0.034992 | -0.013239 | 0.1611 |
| 4 | -0.365609 | -0.212533 | 0.0193 |
| 5 | 4895.382638 | 5007.171751 | 0.0444 |
| 6 | -1.510608 | -1.510608 | 0.0000 |
| 7 | 2.427148 | 2.427148 | 0.0000 |
| 8 | 9.632442 | 9.915401 | 0.7320 |

The comparison produced varying levels of agreement.

**Functions 2, 4 and 5** showed strong agreement between UCB and EI, with their recommended candidates separated by less than 0.05. Function 7 produced identical selections.

For Function 3, EI selected a candidate with a predicted output of approximately -0.0132, above the current observed best of -0.0202. This provided a reason to favour EI over the UCB alternative.

Function 4 also produced an EI candidate predicted to outperform the existing best, while Function 5's EI prediction of approximately 5007.17 was above its newly observed maximum of 4857.45.

**Function 6 required particular caution.** Although UCB and EI selected the same candidate, its predicted output of -1.5106 was substantially below the current best of -0.4585. The selected point was also very distant from the best-known region, suggesting that agreement between acquisition functions did not necessarily indicate a reliable recommendation.

Function 8 showed the largest disagreement, with the two candidates separated by 0.7320. EI remained closer to previously successful regions, while UCB favoured a more exploratory location.

Overall, EI was provisionally preferred for Functions 3, 4, 5 and 8, while the identical UCB/EI recommendation for Function 7 was retained as a strong candidate. Function 2 showed close agreement in a different region of its search space, while Function 6 remained unresolved.

These were provisional decisions pending comparison with the neural-network models.

## 4. Introduce Neural-Network Surrogate Models

A neural network was introduced as an alternative regression model for each function.

Because the datasets remained relatively small, the architecture was deliberately kept compact to avoid unnecessary complexity.

### 4.1 Model Architecture and Training

The models were implemented in PyTorch with the following configuration:

- Two hidden layers containing 16 neurons each.
- ReLU activation functions.
- Standardised inputs and outputs.
- Five independently initialised neural networks per function.
- Ensemble predictions calculated by averaging the five networks.

The networks were trained to predict continuous function outputs from their input coordinates.

Using five independently initialised networks reduced dependence on a single random initialisation and allowed disagreement between predictions to be examined.

### Training Results

| Function | Mean Final Training Loss |
|---|---:|
| 1 | 0.001044 |
| 2 | 0.002267 |
| 3 | 0.000001 |
| 4 | 0.000000 |
| 5 | 0.000004 |
| 6 | 0.000013 |
| 7 | 0.000000 |
| 8 | 0.000000 |

The networks achieved very low training losses across all eight functions.

Functions 3–8 fitted the available observations almost perfectly, while Functions 1 and 2 showed slightly greater variation between independently initialised models.

Although this demonstrated that the architecture could represent the observed data, extremely low training losses could also indicate overfitting.

Training performance alone was therefore insufficient to determine whether the neural networks would provide useful predictions for unexplored candidate points.

### 4.2 Leave-One-Out Cross-Validation

Leave-One-Out Cross-Validation (LOOCV) was used to assess whether the neural networks generalised beyond their training observations.

Each observation was withheld in turn, and the network was trained using the remaining observations before predicting the held-out output.

Performance was measured using Mean Absolute Error (MAE) and compared with a baseline that predicted the mean of the remaining training outputs.

| Function | NN LOOCV MAE | Mean Baseline MAE | NN Beats Baseline |
|---|---:|---:|---|
| 1 | 0.002595 | 0.001255 | No |
| 2 | 0.188740 | 0.225146 | Yes |
| 3 | 0.066222 | 0.046560 | No |
| 4 | 1.998674 | 6.651296 | Yes |
| 5 | 192.581456 | 1041.351340 | Yes |
| 6 | 0.327649 | 0.453136 | Yes |
| 7 | 0.315686 | 0.474058 | Yes |
| 8 | 0.328345 | 0.900332 | Yes |

The neural networks outperformed the mean baseline for **six of the eight functions**.

Functions 4, 5 and 8 showed particularly strong relative improvements, while Functions 2, 6 and 7 also demonstrated useful predictive ability.

However, Functions 1 and 3 failed to outperform the baseline, despite achieving extremely low training losses.

This suggested that their networks were not generalising reliably and that their recommendations should be treated cautiously.

The results supported using neural networks as an additional source of evidence for Functions 2, 4, 5, 6, 7 and 8, rather than automatically relying on their predictions across all functions.

### 4.3 Generate Neural-Network Candidates

Each ensemble evaluated 100,000 randomly generated candidate points.

The candidate with the highest average predicted output was selected, while disagreement between the five networks was recorded as an additional diagnostic.

| Function | NN Predicted Output | Model Disagreement | Distance from Current Best |
|---|---:|---:|---:|
| 1 | 0.004556 | 0.001415 | 0.7732 |
| 2 | 1.116515 | 0.831745 | 0.9722 |
| 3 | 0.000985 | 0.027674 | 0.5262 |
| 4 | -0.570388 | 0.191094 | 0.0578 |
| 5 | 5001.947266 | 90.387283 | 0.0883 |
| 6 | -0.403257 | 0.039523 | 0.0659 |
| 7 | 2.835739 | 0.333867 | 0.3545 |
| 8 | 10.356458 | 0.134738 | 0.3843 |

Several findings were particularly interesting.

**Function 5** produced a strong neural-network recommendation close to its current best observation. The predicted output of approximately 5001.95 exceeded the existing maximum of 4857.45, providing independent support for continued exploitation of the same region identified by the GP models.

**Function 6** selected a candidate only 0.0659 from its current best, predicting an improved output of approximately -0.4033. The relatively low disagreement between networks, combined with successful cross-validation, provided a potentially useful alternative to its unreliable GP recommendation.

Function 8 also produced a promising result, with a predicted output of approximately 10.36 compared with its current best of 9.91.

However, not all neural-network predictions appeared equally reliable.

Function 2 predicted a substantial improvement but selected a point far from its current best, with very high disagreement between the five networks.

Functions 1 and 3 also predicted potential improvements, but their poor cross-validation results meant that their recommendations could not be considered sufficiently reliable.

These findings reinforced the need to evaluate neural-network predictions alongside existing observations and GP recommendations.

## 5. Compare Gaussian Process and Neural-Network Recommendations

The GP and neural-network candidates were compared to determine whether the different modelling approaches identified similar promising regions.

Function 1 was also included in this comparison, with UCB and EI both selecting the same candidate.

### 5.1 Comparison of Candidate Locations

| Function | UCB–EI Distance | EI–NN Distance | Main Observation |
|---|---:|---:|---|
| 1 | 0.0000 | 0.3605 | GP agreement; NN less reliable |
| 2 | 0.0253 | ~0.31 | GP agreement; large NN disagreement |
| 3 | 0.1611 | Substantial | EI better supported than NN |
| 4 | 0.0193 | 0.1198 | All models identify similar region |
| 5 | 0.0444 | 0.0718 | Strong agreement across models |
| 6 | 0.0000 | 1.3150 | Major disagreement; NN better supported |
| 7 | 0.0000 | 0.2738 | GP agreement near successful region |
| 8 | 0.7320 | Over 0.74 | Substantial disagreement across models |

The strongest overall agreement occurred in **Functions 4 and 5**, where the GP and neural-network approaches identified similar regions.

Function 5 was particularly encouraging because the independent models continued to concentrate around an area that had produced repeated improvements.

Function 7 also showed strong agreement between UCB and EI, while its neural-network recommendation was more exploratory.

Function 6 produced the clearest disagreement. Its GP candidate was substantially separated from the neural-network recommendation, despite the GP continuing to experience fitting problems.

Function 8 also showed considerable disagreement between the three approaches, although its neural network had performed well in cross-validation.

The comparison demonstrated that no single surrogate approach should automatically determine the next query for every function.

### 5.2 Assess Neighbouring Observations

Before making the final selections, the nearest existing observation to each candidate was examined.

This provided an additional check on whether model recommendations were supported by previously observed performance.

| Function | UCB Nearest Distance | EI Nearest Distance | NN Nearest Distance |
|---|---:|---:|---:|
| 1 | 0.1400 | 0.1400 | 0.2982 |
| 2 | 0.0487 | 0.0494 | 0.3549 |
| 3 | 0.1832 | 0.1325 | 0.3477 |
| 4 | 0.1303 | 0.1142 | 0.0578 |
| 5 | 0.1468 | 0.1096 | 0.0883 |
| 6 | 0.7451 | 0.7451 | 0.0659 |
| 7 | 0.1401 | 0.1401 | 0.3545 |
| 8 | 0.7299 | 0.4896 | 0.3843 |

The neighbourhood analysis provided several useful distinctions.

For Functions 2 and 3, the GP recommendations were closer to strong historical observations than the neural-network alternatives. This supported retaining EI.

For Functions 4 and 5, all three approaches remained close to established high-performing regions. EI was preferred because it provided strong predicted performance while retaining support from the surrounding observations.

Function 7's common UCB/EI candidate was only 0.1401 from its new best observation, supporting further exploitation.

**Function 6** showed the clearest advantage for the neural network. Its selected candidate was only 0.0659 from the current best, while the nearest observation to the GP candidate was 0.7451 away and had produced a substantially weaker output.

**Function 8** also favoured the neural network, whose recommendation was closer to the latest best observation than either GP candidate.

Function 1 remained uncertain because both approaches had reliability limitations. However, the common UCB/EI candidate was closer to previously sampled observations than the neural-network alternative.

## 6. Final Function-Specific Strategy

The final selections combined surrogate model performance, recent query results, acquisition-function agreement and neighbouring observations.

| Function | Selected Strategy | Reasoning |
|---|---|---|
| 1 | EI / UCB agreement | GP candidate closer to existing observations; NN unreliable |
| 2 | Expected Improvement | Strong GP agreement and better local support than NN |
| 3 | Expected Improvement | Stronger predicted output; NN failed cross-validation |
| 4 | Expected Improvement | Promising predicted improvement and local GP/NN agreement |
| 5 | Expected Improvement | Strong agreement in repeatedly successful region |
| 6 | Neural Network | Better cross-validation support and proximity to current best |
| 7 | EI / UCB agreement | Identical GP selections near recently successful region |
| 8 | Neural Network | Strong cross-validation and promising prediction near latest best |

The neural network therefore influenced the final selections for Functions 6 and 8, where its recommendations were better supported than the GP alternatives.

For the remaining functions, GP-based acquisition methods were retained.

This maintained the Bayesian Optimisation approach while allowing an alternative surrogate to contribute where the evidence justified it.

## 7. Final Week 5 Query Selections

All eight final proposed query points were checked to confirm that every coordinate remained within the permitted `[0, 1]` range and that none duplicated an existing observation.

The distance to the nearest existing observation was also recorded to verify consistency with the intended selection strategy.

| Function | Selection Method | Final Proposed Point | Nearest Distance |
|---|---|---|---:|
| 1 | EI / UCB agreement | `[0.943533, 0.359421]` | 0.140043 |
| 2 | Expected Improvement | `[0.713092, 0.138155]` | 0.049374 |
| 3 | Expected Improvement | `[0.384686, 0.478261, 0.416295]` | 0.132463 |
| 4 | Expected Improvement | `[0.462451, 0.342242, 0.380323, 0.445636]` | 0.114172 |
| 5 | Expected Improvement | `[0.938942, 0.989452, 0.984552, 0.971823]` | 0.109600 |
| 6 | Neural Network | `[0.248909, 0.369411, 0.484746, 0.674732, 0.011701]` | 0.065893 |
| 7 | EI / UCB agreement | `[0.132842, 0.294043, 0.285122, 0.050957, 0.281982, 0.619768]` | 0.140085 |
| 8 | Neural Network | `[0.034135, 0.275291, 0.032179, 0.508682, 0.760754, 0.617950, 0.034127, 0.349420]` | 0.384306 |

All eight points passed the final validation checks.

The selections reflected the function-specific strategies developed during the analysis, with EI and UCB retained where the Gaussian Processes provided strong evidence and neural networks used where the alternative surrogate was better supported.

### Submission Format

```text
Function 1: 0.943533-0.359421
Function 2: 0.713092-0.138155
Function 3: 0.384686-0.478261-0.416295
Function 4: 0.462451-0.342242-0.380323-0.445636
Function 5: 0.938942-0.989452-0.984552-0.971823
Function 6: 0.248909-0.369411-0.484746-0.674732-0.011701
Function 7: 0.132842-0.294043-0.285122-0.050957-0.281982-0.619768
Function 8: 0.034135-0.275291-0.032179-0.508682-0.760754-0.617950-0.034127-0.349420
```

## 8. Week 5 Reflection

Week 5 demonstrated the value of comparing alternative surrogate models rather than relying exclusively on Gaussian Process regression.

The latest observations produced new best outputs for Functions 5, 7 and 8, supporting the continued use of targeted exploitation where promising regions had already been identified.

The Gaussian Processes remained useful for most functions, with Expected Improvement generally selecting candidates with strong predicted performance and supporting evidence from nearby observations.

The introduction of neural networks provided an additional perspective. Although the networks achieved extremely low training losses, Leave-One-Out Cross-Validation demonstrated that this did not necessarily translate into reliable generalisation.

Functions 1 and 3 failed to outperform the mean baseline, highlighting the risk of overfitting when relatively flexible models are trained on small datasets.

However, six functions outperformed the baseline, with particularly encouraging results for Functions 4, 5 and 8.

The neural network was especially useful for Function 6, where the Gaussian Process continued to experience fitting difficulties. Its recommendation was much closer to the current best observation and received stronger support from cross-validation and ensemble agreement.

Function 8 also benefited from the additional model comparison, with the neural-network candidate predicting a potential improvement over the current best.

Overall, the results supported retaining Gaussian Processes and acquisition functions as the main optimisation framework, while using neural networks as complementary models where they demonstrated useful predictive performance.

The following round will evaluate the returned outputs and assess whether the neural-network-informed selections provide improvements over the existing GP-based strategies.

---

**Associated code:** `Week_05.ipynb`



