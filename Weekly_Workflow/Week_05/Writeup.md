
# Module 16 — Week 5: Gaussian Process and Neural Network Surrogate Comparison

## Weekly objective and step-by-step plan

The objective of Week 5 was to incorporate the latest query results, reassess the Gaussian Process (GP) models and introduce a neural network (NN) as an alternative surrogate for the eight unknown functions.

In previous weeks, I had established Gaussian Process Regression with Upper Confidence Bound (UCB) and Expected Improvement (EI) as the main optimisation approaches. However, Functions 1 and 6 continued to experience modelling difficulties, while the performance of the other functions suggested that a more exploitative approach could be beneficial.

This week, I introduced a small neural-network ensemble to investigate whether a different modelling approach could identify useful patterns in the observations and provide additional guidance for selecting the next queries.

The intention was not to replace the Gaussian Processes automatically, but to compare the different approaches and make function-specific decisions based on model reliability, predicted performance and previous results.

The process followed seven steps:

1. Incorporate the new observations and compare them with the previous best outputs.
2. Refit the Gaussian Processes and examine their fitted length scales, predictions and uncertainty.
3. Compare UCB with κ = 1.5 against Expected Improvement.
4. Train a small neural-network ensemble and evaluate its generalisation performance.
5. Compare GP and neural-network recommendations.
6. Make function-specific selections using model diagnostics and nearby observations.
7. Validate and format the eight final queries.

## Methodology

I incorporated the latest observations into the cumulative datasets and refitted the Gaussian Processes using RBF kernels, normalised outputs, ten optimiser restarts and a random state of 42.

For GP-based candidate selection, I retained UCB with κ = 1.5 following the exploration comparison conducted in Week 4 and compared it directly with Expected Improvement. Both acquisition functions were evaluated using the same 100,000 candidate points for each function.

I also introduced a neural-network surrogate using PyTorch. Because the available datasets remained relatively small, I used a compact architecture with two hidden layers of 16 neurons each and ReLU activation functions.

Inputs and outputs were standardised before training, and five independently initialised networks were trained for each function. Their predictions were averaged to reduce dependence on individual random initialisations.

To assess whether the neural networks could generalise beyond their training observations, I used Leave-One-Out Cross-Validation (LOOCV). Each observation was excluded in turn, and the model's prediction was compared with the held-out output. Performance was measured using mean absolute error (MAE) and compared with a baseline that predicted the mean of the remaining training observations.

Finally, I compared candidates from UCB, EI and the neural-network ensembles, considering their predicted outputs, uncertainty, model agreement and proximity to existing observations before selecting the final queries.

## Results and interpretation by step

### Step 1 — Incorporate new observations

I compared the outputs returned from Week 4's proposed queries with the previous best observations.

| Function | Previous best | New output | Updated best | Result |
|---|---:|---:|---:|---|
| 1 | ≈0 | -0.005913 | ≈0 | No improvement |
| 2 | 0.611205 | 0.507418 | 0.611205 | No improvement |
| 3 | -0.020167 | -0.107206 | -0.020167 | No improvement |
| 4 | -0.300645 | -1.438643 | -0.300645 | No improvement |
| 5 | 4595.479318 | 4857.449601 | 4857.449601 | New best |
| 6 | -0.458511 | -2.043880 | -0.458511 | No improvement |
| 7 | 2.139551 | 2.536939 | 2.536939 | New best |
| 8 | 9.821986 | 9.911249 | 9.911249 | New best |

Three of the eight queries produced new best outputs.

Function 5 improved from approximately 4595.48 to 4857.45, an increase of 5.7%. This continued the strong performance observed over successive rounds, providing further evidence that the search had identified a promising region.

Function 7 also improved, from approximately 2.1396 to 2.5369, an increase of 18.6%. Function 8 achieved a smaller improvement, from approximately 9.8220 to 9.9112.

These results provided encouraging evidence for the more exploitative EI-based approach used in Week 4, particularly for Functions 5, 7 and 8.

However, Functions 2, 3 and 4 did not improve despite also using EI. This demonstrated that promising surrogate predictions did not necessarily translate into better observed outputs.

Function 1 again failed to improve under maximum-distance exploration, while Function 6's highly exploratory query produced an output of -2.0439, substantially below its previous best of -0.4585.

This suggested that continuing aggressive maximum-distance exploration for Function 6 might not be appropriate.

### Step 2 — Refit and evaluate the Gaussian Processes

I refitted the RBF Gaussian Processes using the expanded datasets and examined their fitted length scales.

| Function | Fitted RBF length scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0494 |
| 3 | 0.173 |
| 4 | 0.414 |
| 5 | 0.467 |
| 6 | 1e-05 |
| 7 | 0.330 |
| 8 | 0.979 |

Functions 2, 3, 4, 5, 7 and 8 produced length scales above the lower bound, suggesting that their Gaussian Processes continued to identify structure within the observed data.

Functions 1 and 6 again reached the minimum permitted length scale of 1e-05, consistent with the convergence problems observed in previous rounds.

As in Week 4, I examined the predicted means and uncertainty across 100,000 candidate points to determine whether the models could meaningfully distinguish between different regions.

| Function | Predicted mean range | Uncertainty range |
|---|---:|---:|
| 1 | 0.000680 | 0.001703 |
| 2 | 0.699132 | 0.237428 |
| 3 | 0.389118 | 0.077909 |
| 4 | 34.460542 | 7.316418 |
| 5 | 6264.230066 | 1228.812886 |
| 6 | 1.149715 | 0.509894 |
| 7 | 2.487448 | 0.642344 |
| 8 | 4.801047 | 0.669037 |

The results showed meaningful variation in both predicted outputs and uncertainty across all eight functions.

Function 1 continued to produce relatively small predicted values and remained difficult to model reliably.

However, Function 6 showed an important change. In Week 4, its GP had produced effectively identical predictions across the candidate space. This week, its predicted mean varied by approximately 1.15, while uncertainty varied by 0.51.

Although the fitted length scale remained problematic, the latest exploratory observation appeared to have provided additional information to the surrogate model.

This suggested that Function 6's GP could be reconsidered as a potential source of guidance, although its predictions still required caution.

### Step 3 — Compare GP acquisition strategies

Rather than repeating the full κ comparison from Week 4, I retained UCB with κ = 1.5 and compared it directly with Expected Improvement.

Both methods were evaluated using the same 100,000 candidate points for each function.

Functions 2–8 were included initially, while Function 1 was treated separately because of its continuing GP instability.

| Function | UCB predicted output | EI predicted output | Distance between candidates |
|---|---:|---:|---:|
| 2 | 0.431873 | 0.429437 | 0.0253 |
| 3 | -0.034992 | -0.013239 | 0.1611 |
| 4 | -0.365609 | -0.212533 | 0.0193 |
| 5 | 4895.382638 | 5007.171751 | 0.0444 |
| 6 | -1.510608 | -1.510608 | 0.0000 |
| 7 | 2.427148 | 2.427148 | 0.0000 |
| 8 | 9.632442 | 9.915401 | 0.7320 |

Functions 2, 4 and 5 showed strong agreement between UCB and EI, suggesting that the Gaussian Processes were consistently identifying similar promising regions.

For Function 2, both methods selected candidates in a different region from the current best, around input coordinates [0.71, 0.12–0.14]. Their close agreement suggested that this was a feature of the GP predictions rather than the choice of acquisition function.

Function 3 showed greater disagreement, although EI selected a candidate with a predicted output of -0.0132, above the current observed maximum of -0.0202.

Function 4 showed very strong agreement, with the two candidates separated by just 0.0193. EI selected a point with a predicted output of -0.2125, also above the current observed best of -0.3006.

Function 5 was particularly encouraging. EI predicted an output of 5007.17, above the current observed maximum of 4857.45. Both acquisition functions selected candidates in the same successful region, supporting continued exploitation.

Function 7 produced identical UCB and EI candidates, located approximately 0.1401 from the latest best observation.

Function 8 showed the largest disagreement, with candidates separated by 0.7320. EI remained closer to the region that had produced the latest improvement, while UCB selected a more exploratory point.

Function 6 also produced identical UCB and EI candidates. However, the predicted output of -1.5106 was substantially worse than its current best of -0.4585, and the candidate was located far from existing high-performing observations.

Given the continuing GP convergence problems, this agreement was not considered sufficient evidence to select the candidate.

Overall, the GP comparison favoured EI for Functions 3, 4, 5 and 8, while the identical UCB/EI recommendation for Function 7 provided a strong provisional candidate. Function 2's GP recommendations identified an alternative region worth investigating, and Function 6 remained unresolved.

### Step 4 — Introduce and evaluate a neural-network surrogate

I introduced a small neural-network ensemble using PyTorch to provide an alternative modelling approach.

Each network used two hidden layers with 16 neurons each, ReLU activation functions and standardised inputs and outputs. Five independently initialised networks were trained for each function, and their predictions were averaged.

The architecture was deliberately kept compact because the datasets remained relatively small.

#### Neural-network training performance

The networks achieved the following mean final training losses.

| Function | Mean final training loss |
|---|---:|
| 1 | 0.001044 |
| 2 | 0.002267 |
| 3 | 0.000001 |
| 4 | ≈0 |
| 5 | 0.000004 |
| 6 | 0.000013 |
| 7 | ≈0 |
| 8 | ≈0 |

The neural networks achieved extremely low training losses across all eight functions.

Functions 3–8 were fitted almost perfectly, while Functions 1 and 2 showed slightly greater variation between the independently initialised networks.

Although this indicated that the networks had sufficient flexibility to fit the existing observations, the small datasets meant that overfitting was a significant concern.

I therefore used Leave-One-Out Cross-Validation to assess their ability to generalise to observations excluded from training.

#### Leave-One-Out Cross-Validation

| Function | NN MAE | Mean baseline MAE | NN better? |
|---|---:|---:|---|
| 1 | 0.002595 | 0.001255 | No |
| 2 | 0.188740 | 0.225146 | Yes |
| 3 | 0.066222 | 0.046560 | No |
| 4 | 1.998674 | 6.651296 | Yes |
| 5 | 192.581456 | 1041.351340 | Yes |
| 6 | 0.327649 | 0.453136 | Yes |
| 7 | 0.315686 | 0.474058 | Yes |
| 8 | 0.328345 | 0.900332 | Yes |

The neural networks outperformed the mean baseline for six of the eight functions.

Functions 4, 5 and 8 showed particularly strong improvements relative to the baseline, while Functions 2, 6 and 7 also demonstrated useful predictive performance.

Functions 1 and 3 did not outperform the baseline, suggesting that their low training losses were not matched by reliable generalisation.

This reinforced the importance of evaluating models on unseen observations rather than relying solely on training performance.

#### Neural-network candidate selection

I then used the five-network ensembles to evaluate 100,000 candidate points for each function.

The candidate with the highest mean predicted output was selected, and the standard deviation between the five network predictions was recorded as a measure of model disagreement.

| Function | NN predicted output | Model disagreement | Distance from current best |
|---|---:|---:|---:|
| 1 | 0.004556 | 0.001415 | 0.7732 |
| 2 | 1.116515 | 0.831745 | 0.9722 |
| 3 | 0.000985 | 0.027674 | 0.5262 |
| 4 | -0.570388 | 0.191094 | 0.0578 |
| 5 | 5001.947266 | 90.387283 | 0.0883 |
| 6 | -0.403257 | 0.039523 | 0.0659 |
| 7 | 2.835739 | 0.333867 | 0.3545 |
| 8 | 10.356458 | 0.134738 | 0.3843 |

Function 5 produced one of the strongest recommendations. The neural network predicted an output of approximately 5002, above the current best, and selected a point close to the same high-performing region identified by the Gaussian Processes.

Function 6 also produced an interesting candidate, predicting -0.4033, which would improve on its current best of -0.4585. The candidate was only 0.0659 from the current best observation, and disagreement between networks was relatively low.

For Function 8, the neural network predicted 10.3565, above the current best of 9.9112. Its cross-validation performance and relatively modest model disagreement made this a credible alternative to the EI recommendation.

Function 7's neural network also predicted an improvement, reaching 2.8357, but selected a more exploratory candidate with greater model disagreement than some of the other functions.

In contrast, Function 2's network predicted a potentially large improvement but showed very high disagreement between its five networks. This reduced confidence in the selected candidate despite the network outperforming the baseline during cross-validation.

Functions 1 and 3 were treated cautiously because their neural networks had failed to outperform the mean baseline.

Overall, the neural-network results suggested that the alternative surrogate could provide useful supporting evidence, particularly for Functions 5, 6 and 8, but should not automatically replace the GP recommendations.

### Step 5 — Compare GP and neural-network recommendations

I compared the candidates selected by UCB, EI and the neural-network ensembles to determine whether the different approaches identified similar promising regions.

For Function 1, I also generated UCB and EI candidates as additional evidence, despite the continuing GP convergence problems.

Both GP acquisition methods selected the same point for Function 1, although the predicted output remained below its current best.

The distances between model recommendations were then calculated.

| Function | NN → UCB distance | NN → EI distance | UCB → EI distance |
|---|---:|---:|---:|
| 1 | 0.3605 | 0.3605 | 0.0000 |
| 2 | 0.3066 | 0.3169 | 0.0253 |
| 3 | 0.7054 | 0.6475 | 0.1611 |
| 4 | 0.1381 | 0.1198 | 0.0193 |
| 5 | 0.0966 | 0.0718 | 0.0444 |
| 6 | 1.3150 | 1.3150 | 0.0000 |
| 7 | 0.2738 | 0.2738 | 0.0000 |
| 8 | 0.7429 | 0.7940 | 0.7320 |

Function 5 showed the strongest agreement between all three approaches, with the neural-network candidate only 0.0718 from EI and 0.0966 from UCB.

Combined with its repeated improvements, this provided further support for continuing to exploit the identified region.

Function 4 also showed reasonable agreement between the models, with all three candidates concentrated around the existing high-performing region.

Function 2 showed strong agreement between UCB and EI, while the neural-network recommendation was approximately 0.31 from both and had high model disagreement.

Function 3's neural-network candidate was substantially separated from both GP recommendations. Given its weaker cross-validation performance, the GP remained the preferred modelling approach.

Function 6 showed the greatest disagreement between the neural network and GP, with their candidates separated by 1.3150. Given the GP convergence issues and stronger neural-network cross-validation results, the NN candidate appeared more credible.

Function 7 showed identical GP recommendations, while the neural network selected a point further from the recently successful region. Although the NN predicted a potentially stronger output, the GP candidate had greater support from the latest observations.

Function 8 also showed substantial disagreement between all three approaches. However, the neural network had performed well in cross-validation and predicted an improvement, making it a credible alternative to EI.

Overall, the comparison demonstrated that different surrogate models could provide complementary information, but agreement alone was not sufficient evidence of reliability.

### Step 6 — Make function-specific final selections

Before selecting the final queries, I examined the existing observations nearest to each candidate.

This helped determine whether model predictions were supported by previous empirical results or represented potentially unreliable extrapolation.

| Function | Preferred candidate | Nearest observation distance | Nearest observed output |
|---|---|---:|---:|
| 1 | GP (EI/UCB) | 0.1400 | ≈0 |
| 2 | EI | 0.0494 | 0.538996 |
| 3 | EI | 0.1325 | -0.020167 |
| 4 | EI | 0.1142 | -0.300645 |
| 5 | EI | 0.1096 | 4857.449601 |
| 6 | Neural Network | 0.0659 | -0.458511 |
| 7 | GP (EI/UCB) | 0.1401 | 2.536939 |
| 8 | Neural Network | 0.3843 | 9.911249 |

The neighbourhood analysis provided further support for the GP candidates for Functions 2, 3, 4, 5 and 7.

Function 5's EI candidate remained close to the latest best observation, supporting continued exploitation of the successful region.

Function 6's neural-network candidate was only 0.0659 from its current best observation, compared with a distance of 0.7451 between the GP candidate and its nearest observation.

This, combined with the NN's cross-validation performance, supported selecting the neural-network candidate rather than the much more exploratory GP recommendation.

For Function 8, the neural-network candidate was closer to the latest best observation than either GP recommendation and predicted a stronger output, making it the preferred selection.

Function 1 remained uncertain because neither surrogate model provided particularly convincing evidence. However, the common EI/UCB candidate was preferred over the neural-network alternative, which had failed to outperform the baseline during cross-validation.

The final selections therefore favoured EI for Functions 2–5, GP agreement for Functions 1 and 7, and the neural-network surrogate for Functions 6 and 8.

### Step 7 — Validate the final proposed queries

I validated all eight selected points to ensure that every coordinate was within the permitted [0, 1] input range and that no point duplicated an existing observation.

| Function | Selection method | Nearest observation distance |
|---|---|---:|
| 1 | EI / UCB agreement | 0.140043 |
| 2 | Expected Improvement | 0.049374 |
| 3 | Expected Improvement | 0.132463 |
| 4 | Expected Improvement | 0.114172 |
| 5 | Expected Improvement | 0.109600 |
| 6 | Neural Network | 0.065893 |
| 7 | EI / UCB agreement | 0.140085 |
| 8 | Neural Network | 0.384306 |

All eight proposed queries passed the validation checks.

The final queries, formatted to six decimal places, were:

| Function | Selection method | Proposed query coordinates |
|---|---|---|
| 1 | EI / UCB agreement | `[0.943533, 0.359421]` |
| 2 | Expected Improvement | `[0.713092, 0.138155]` |
| 3 | Expected Improvement | `[0.384686, 0.478261, 0.416295]` |
| 4 | Expected Improvement | `[0.462451, 0.342242, 0.380323, 0.445636]` |
| 5 | Expected Improvement | `[0.938942, 0.989452, 0.984552, 0.971823]` |
| 6 | Neural Network | `[0.248909, 0.369411, 0.484746, 0.674732, 0.011701]` |
| 7 | EI / UCB agreement | `[0.132842, 0.294043, 0.285122, 0.050957, 0.281982, 0.619768]` |
| 8 | Neural Network | `[0.034135, 0.275291, 0.032179, 0.508682, 0.760754, 0.617950, 0.034127, 0.349420]` |

## Weekly summary and next steps

Week 5 produced new best observations for Functions 5, 7 and 8, providing further evidence that the more exploitative approach introduced in Week 4 was effective for several functions.

The main methodological development was introducing a neural-network ensemble as an alternative surrogate to the Gaussian Process.

Although the networks achieved extremely low training losses, Leave-One-Out Cross-Validation showed that this did not always translate into reliable predictions on unseen observations. Functions 1 and 3 failed to outperform the mean baseline, reinforcing the importance of evaluating generalisation rather than relying solely on training performance.

The comparison between GP and neural-network recommendations was particularly useful for Functions 5, 6 and 8.

Function 5 showed strong agreement between all three modelling approaches, reinforcing the decision to continue exploiting its increasingly successful region.

For Function 6, the neural network provided a more credible candidate than the GP, which continued to experience convergence problems. The selected NN candidate remained close to the current best observation and offered a predicted improvement.

Function 8 also demonstrated the value of an alternative surrogate, with the neural network identifying a candidate supported by cross-validation performance, relatively low model disagreement and proximity to the latest best observation.

The next steps were to:

- Submit the eight proposed query points and assess their returned outputs.
- Incorporate the new observations and refit the GP models.
- Evaluate whether the neural-network selections for Functions 6 and 8 improve performance.
- Continue comparing GP and neural-network recommendations where their predictions differ.
- Reassess the reliability of Functions 1 and 6 and investigate further modelling adjustments if necessary.

Overall, Week 5 demonstrated that neural networks could provide useful additional information alongside Gaussian Processes. The final selections continued to favour GP-based acquisition methods where they were well supported, while incorporating neural-network recommendations where the alternative surrogate provided stronger evidence.


