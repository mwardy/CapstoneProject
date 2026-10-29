
# Module 17 — Week 6: Historical Model Performance and Function-Specific Optimisation

## Weekly objective and step-by-step plan

The objective of Week 6 was to incorporate the latest query results, reassess the Gaussian Process (GP) and neural-network (NN) models, and introduce a historical-performance check to help inform future optimisation decisions.

In Week 5, I introduced neural-network ensembles as an alternative surrogate to Gaussian Process Regression. This provided additional evidence when comparing promising candidate regions, particularly for functions where the GP had experienced modelling difficulties.

This week, I retained the existing modelling approaches but introduced a review of previous query outcomes. Rather than relying exclusively on predicted outputs, I wanted to consider whether the methods selected in the previous round had actually improved the best-known observations.

The aim was to make more informed function-specific decisions by combining model predictions, uncertainty, predictive reliability, historical performance and the outputs of nearby observations.

The process followed nine steps:

1. Incorporate the new observations and compare them with the previous best outputs.
2. Refit the Gaussian Processes and examine their fitted length scales, predictions and uncertainty.
3. Generate and compare UCB and Expected Improvement candidates.
4. Retrain the neural-network ensembles and evaluate their predictive performance.
5. Compare GP and neural-network recommendations.
6. Introduce a historical-performance check based on previous query outcomes.
7. Examine the neighbourhoods surrounding candidate points.
8. Make function-specific selections using the combined evidence.
9. Validate and format the eight final queries.

## Methodology

I incorporated the latest observations into the cumulative datasets and refitted the Gaussian Processes using RBF kernels, normalised outputs, ten optimiser restarts and a random state of 42.

For GP-based candidate selection, I retained Upper Confidence Bound (UCB) with κ = 1.5 and compared it directly with Expected Improvement (EI). Both acquisition functions were evaluated using the same candidate set of 100,000 points for each function.

I also retained the neural-network architecture introduced in Week 5. Each function was modelled using an ensemble of five independently initialised PyTorch networks, each with two hidden layers containing 16 neurons and ReLU activation functions.

Inputs and outputs were standardised, and the networks were trained using the same procedure as the previous round. Their predictions were averaged to generate alternative candidate recommendations, while disagreement between the networks provided an indication of predictive uncertainty.

I repeated Leave-One-Out Cross-Validation (LOOCV) to evaluate the neural networks' ability to generalise beyond their training observations. Mean absolute error (MAE) was compared with a simple baseline that predicted the mean of the remaining training observations.

The main methodological development was introducing a historical-performance check. For each function, I reviewed the method used to select the previous query and whether that query had improved the best-known output.

This was used as supporting evidence rather than a formal weighting system, since a single successful or unsuccessful query does not necessarily establish the reliability of a model.

Finally, I compared candidate neighbourhoods using the actual observed outputs surrounding each recommendation. The final selections were based on model reliability, acquisition-function agreement, historical performance, predicted outputs and proximity to successful observations.

## Results and interpretation by step

### Step 1 — Incorporate new observations

I compared the outputs returned from Week 5's proposed queries with the previous best observations.

| Function | Previous best | New output | Updated best | Result |
|---|---:|---:|---:|---|
| 1 | ≈0 | ≈0 | ≈0 | No improvement |
| 2 | 0.611205 | 0.746962 | 0.746962 | New best |
| 3 | -0.020167 | -0.016451 | -0.016451 | New best |
| 4 | -0.300645 | -0.638343 | -0.300645 | No improvement |
| 5 | 4857.449601 | 6691.931674 | 6691.931674 | New best |
| 6 | -0.458511 | -0.456874 | -0.456874 | New best |
| 7 | 2.536939 | 1.644736 | 2.536939 | No improvement |
| 8 | 9.911249 | 9.742277 | 9.911249 | No improvement |

Four of the eight Week 5 queries produced new best outputs.

Function 5 achieved the largest improvement, increasing from approximately 4857.45 to 6691.93, a gain of 37.8%. This continued the strong upward trend observed across recent rounds and provided further evidence that the search had identified a productive region.

Function 2 improved from 0.6112 to 0.7470, while Function 3 achieved a smaller improvement from -0.02017 to -0.01645. Both of these queries had been selected using Expected Improvement.

Function 6 also produced a new best, improving slightly from -0.45851 to -0.45687. This was particularly interesting because the previous query had been selected using the neural-network surrogate.

Functions 4, 7 and 8 did not improve, while Function 1 again returned an output effectively equal to zero.

These mixed results provided useful evidence for evaluating the historical performance of the different selection methods later in the analysis.

### Step 2 — Refit and evaluate the Gaussian Processes

I refitted the RBF Gaussian Processes using the expanded datasets and examined their fitted length scales.

| Function | Fitted RBF length scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0521 |
| 3 | 0.185 |
| 4 | 0.423 |
| 5 | 0.389 |
| 6 | 0.535 |
| 7 | 0.248 |
| 8 | 1.000 |

Most functions continued to produce fitted length scales above the lower bound.

Function 1 remained problematic, with its GP again converging to the minimum permitted length scale of 1e-05. This suggested that the model was still struggling to identify a smooth relationship across the available observations.

However, Function 6 showed a substantial improvement.

Its fitted length scale increased from 1e-05 in Week 5 to 0.535 in Week 6. This suggested that the additional observation had allowed the GP to identify considerably more structure in the function.

As a result, I reconsidered the use of GP-based acquisition methods for Function 6 rather than automatically favouring the neural network.

#### Prediction and uncertainty diagnostics

I evaluated the Gaussian Processes across 100,000 candidate points and recorded the variation in predicted means and predictive uncertainty.

| Function | Predicted mean range | Uncertainty range |
|---|---:|---:|
| 1 | 0.000635 | 0.001654 |
| 2 | 0.831417 | 0.259242 |
| 3 | 0.389101 | 0.078040 |
| 4 | 34.434331 | 7.386272 |
| 5 | 7326.020374 | 1726.249284 |
| 6 | 2.348109 | 0.508036 |
| 7 | 2.575815 | 0.664717 |
| 8 | 4.822485 | 0.659779 |

The diagnostics showed meaningful variation in predicted outputs and uncertainty for Functions 2–8.

Function 6 was particularly encouraging, with its predicted mean now ranging from approximately -2.662 to -0.313.

This represented a substantial change from the earlier rounds when the GP had struggled to distinguish between different candidate regions.

Function 1 continued to show relatively limited predicted variation and an extremely small fitted length scale, so confidence in its GP recommendations remained low.

Overall, the results suggested that GP-based acquisition methods remained appropriate for Functions 2–8, including Function 6 following its improved fit.

### Step 3 — Compare UCB and Expected Improvement

I generated candidate points using the updated GP models and compared UCB with κ = 1.5 against Expected Improvement.

Both methods were evaluated using the same 100,000 candidate points for each function.

| Function | UCB predicted output | EI predicted output | Distance between candidates |
|---|---:|---:|---:|
| 1 | -0.000635 | -0.000635 | 0.0000 |
| 2 | 0.644023 | 0.679240 | 0.0077 |
| 3 | -0.048421 | -0.010868 | 0.3695 |
| 4 | -0.455899 | -0.278717 | 0.0766 |
| 5 | 6213.849027 | 6213.849027 | 0.0000 |
| 6 | -0.338919 | -0.313445 | 0.1444 |
| 7 | 2.512543 | 2.512543 | 0.0000 |
| 8 | 9.793025 | 9.793025 | 0.0000 |

UCB and EI selected identical candidates for Functions 1, 5, 7 and 8.

Function 2 also showed extremely close agreement, with the two recommendations separated by just 0.0077.

For Function 2, EI selected a candidate close to the latest best observation of 0.746962, supporting continued exploitation of the successful region.

Function 3 showed greater disagreement between the two methods. EI predicted an output of -0.010868, above the current observed maximum of -0.016451, while UCB selected a more exploratory point with a weaker predicted output.

Function 4 showed relatively close agreement, although EI selected a candidate with a stronger predicted output of -0.278717, compared with -0.455899 for UCB.

Function 5 continued to focus on its established high-performing region, with both acquisition functions selecting exactly the same candidate.

Function 6 produced particularly encouraging results. Following the improvement in its GP fit, EI predicted an output of -0.313445, substantially better than the current best of -0.456874.

This provided a reason to reconsider GP-based optimisation for Function 6 following its earlier modelling difficulties.

Function 7 showed complete agreement between UCB and EI, although the previous GP-selected query had failed to improve the function.

Function 8 also showed complete agreement, in contrast to Week 5 when the two methods had selected substantially different regions.

Although Function 1 showed agreement between UCB and EI, this was treated cautiously because both methods relied on the same problematic GP model.

Overall, the comparison suggested that GP-based acquisition methods continued to identify promising candidates, but their reliability needed to be considered alongside the historical results and alternative neural-network recommendations.

### Step 4 — Retrain and evaluate the neural-network ensembles

I retained the same neural-network architecture introduced in Week 5 to allow consistent comparisons as additional observations became available.

Each function used an ensemble of five independently initialised networks, each containing two hidden layers with 16 neurons and ReLU activation functions.

#### Neural-network training performance

The networks achieved the following mean final training losses.

| Function | Mean final training loss |
|---|---:|
| 1 | 0.014672 |
| 2 | 0.023384 |
| 3 | 0.012198 |
| 4 | 0.001875 |
| 5 | 0.000085 |
| 6 | 0.003526 |
| 7 | 0.000907 |
| 8 | 0.001338 |

The neural networks continued to fit the available observations, although training performance varied between functions and individual ensemble members.

Functions 4–8 generally achieved low training losses, while Functions 1–3 showed greater variation.

Function 5 produced particularly consistent low losses despite the substantial increase in its best observed output.

However, low training loss alone does not establish whether a model generalises to unseen observations.

I therefore repeated Leave-One-Out Cross-Validation.

#### Leave-One-Out Cross-Validation

| Function | NN MAE | Mean baseline MAE | NN better? |
|---|---:|---:|---|
| 1 | 0.001675 | 0.001179 | No |
| 2 | 0.267443 | 0.241496 | No |
| 3 | 0.067227 | 0.048537 | No |
| 4 | 1.827993 | 6.878180 | Yes |
| 5 | 273.005550 | 1375.410321 | Yes |
| 6 | 0.309082 | 0.476405 | Yes |
| 7 | 0.323396 | 0.507559 | Yes |
| 8 | 0.294854 | 0.920553 | Yes |

The neural networks outperformed the mean baseline for five of the eight functions.

Functions 4–8 continued to demonstrate useful predictive performance, with Function 5 showing a particularly large improvement relative to the baseline.

Functions 1 and 3 again failed to outperform the baseline.

Function 2 also performed slightly worse than the baseline this week, despite having outperformed it in Week 5.

These results suggested that neural-network recommendations for Functions 1–3 should receive less weight in the final selection process.

#### Neural-network candidate selection

I used the five-network ensembles to evaluate 100,000 candidate points for each function.

The candidate with the highest mean predicted output was selected, and the standard deviation between the five network predictions was recorded as a measure of model disagreement.

| Function | NN predicted output | Model disagreement |
|---|---:|---:|
| 1 | 0.003677 | 0.002351 |
| 2 | 2.724940 | 1.237037 |
| 3 | -0.001139 | 0.032445 |
| 4 | -0.374836 | 0.459252 |
| 5 | 6691.842285 | 47.407719 |
| 6 | -0.406389 | 0.128387 |
| 7 | 2.412130 | 0.419323 |
| 8 | 10.052735 | 0.132650 |

Function 2 produced an unusually high predicted output of 2.724940, but this was accompanied by substantial disagreement between the networks and weak cross-validation performance.

Consequently, this recommendation was not considered sufficiently reliable.

Function 5's neural network predicted approximately 6691.84, close to its current best observation of 6691.93. This suggested that the network had identified the same high-performing region.

Function 6's neural network predicted -0.406389, above its current best, while Function 8's network predicted 10.052735, also exceeding its existing best.

These recommendations were retained for comparison with the GP candidates.

Overall, the neural-network results continued to provide useful alternative evidence, particularly for Functions 5, 6 and 8, but their reliability varied between functions.

### Step 5 — Compare GP and neural-network recommendations

I compared the candidates selected by UCB, EI and the neural-network ensembles by calculating the Euclidean distances between their proposed points.

| Function | NN → UCB distance | NN → EI distance | UCB → EI distance |
|---|---:|---:|---:|
| 1 | 0.4923 | 0.4923 | 0.0000 |
| 2 | 0.2851 | 0.2908 | 0.0077 |
| 3 | 0.5382 | 0.2600 | 0.3695 |
| 4 | 0.1355 | 0.1504 | 0.0766 |
| 5 | 0.1887 | 0.1887 | 0.0000 |
| 6 | 0.2208 | 0.2055 | 0.1444 |
| 7 | 0.3327 | 0.3327 | 0.0000 |
| 8 | 0.7737 | 0.7737 | 0.0000 |

Functions 4, 5 and 6 showed relatively close agreement between the modelling approaches.

For Function 5, both the GP and neural network continued to identify the same broad high-performing region, supporting further exploitation.

Function 6 also showed increasing agreement between the models following the improvement in its GP fit.

Function 8 produced the largest disagreement between the GP and neural network, with their recommendations separated by 0.7737.

Although the neural network continued to perform well in cross-validation, its previous query had failed to improve the function, so the recommendation required further investigation.

Function 2 also showed a clear difference between the models. UCB and EI were almost identical, while the NN candidate was considerably further away and had weak validation performance.

For Function 3, the NN candidate was closer to EI than UCB, but the NN again failed to outperform the baseline.

Function 4 showed relatively strong agreement between all three approaches, suggesting that multiple models were identifying a similar promising region.

Function 7 showed identical GP recommendations, while the NN candidate was further away from the GP-selected region.

Overall, the comparison reinforced the importance of considering multiple sources of evidence rather than treating model agreement alone as sufficient justification for selecting a point.

### Step 6 — Introduce a historical-performance check

The main methodological development in Week 6 was incorporating the historical success of previous query selections into the final decision process.

For each function, I reviewed the method used in Week 5 and whether its selected query improved the best-known output.

| Function | Previous selection method | Change in output | Outcome |
|---|---|---:|---|
| 1 | EI / UCB agreement | ≈0 | No improvement |
| 2 | Expected Improvement | +0.135757 | New best |
| 3 | Expected Improvement | +0.003716 | New best |
| 4 | Expected Improvement | -0.337698 | No improvement |
| 5 | Expected Improvement | +1834.482072 | New best |
| 6 | Neural Network | +0.001637 | New best |
| 7 | EI / UCB agreement | -0.892203 | No improvement |
| 8 | Neural Network | -0.168972 | No improvement |

The historical comparison showed that Expected Improvement had been successful for Functions 2, 3 and 5, while the neural network had produced a small improvement for Function 6.

Function 5's improvement was particularly substantial, strengthening the case for continuing to exploit the productive region identified by the GP.

The previous neural-network selection for Function 8 had failed to improve the best output, reducing confidence in repeating the approach without additional supporting evidence.

Similarly, the previous GP-agreement selection for Function 7 had not improved the function, so agreement between UCB and EI was not considered sufficient justification on its own.

Function 4's previous EI selection had also failed to improve, meaning that the current candidate required additional support from model predictions and nearby observations.

This historical-performance check was used as supporting evidence rather than a formal weighting system, since the success or failure of a single query does not necessarily establish whether a model is reliable.

### Step 7 — Examine candidate neighbourhoods

I examined the nearest existing observations surrounding each UCB, EI and neural-network candidate.

This allowed the model recommendations to be compared with actual observed outputs rather than relying entirely on surrogate predictions.

| Function | UCB nearest distance | EI nearest distance | NN nearest distance |
|---|---:|---:|---:|
| 1 | 0.1807 | 0.1807 | 0.3077 |
| 2 | 0.0385 | 0.0309 | 0.3164 |
| 3 | 0.2255 | 0.1407 | 0.2033 |
| 4 | 0.1042 | 0.1049 | 0.0619 |
| 5 | 0.1651 | 0.1651 | 0.0286 |
| 6 | 0.2812 | 0.2131 | 0.2223 |
| 7 | 0.1727 | 0.1727 | 0.2813 |
| 8 | 0.6614 | 0.6614 | 0.3972 |

For Function 2, EI was only 0.0309 from the new best observation of 0.746962, providing strong evidence for continued exploitation.

Function 3's EI candidate was also relatively close to strong existing observations and was better supported than the neural-network alternative.

For Function 4, the NN candidate was closest to the best observation but was also near several weaker recent outputs. EI remained within the high-performing region while moving slightly away from those unsuccessful points.

Function 5's neural-network candidate was particularly interesting, lying only 0.0286 from the new best observation of 6691.93.

This compared with a distance of 0.1651 for the GP candidate and suggested that a small local refinement around the most successful observation might be worthwhile.

For Function 6, all three approaches identified regions around the strongest existing observations, supporting the view that the GP and neural network were beginning to converge.

Function 7's GP candidate was closer to its strongest observations than the NN alternative.

For Function 8, the NN candidate was substantially closer to previous strong observations than the GP candidate, although both remained relatively exploratory.

Overall, the neighbourhood analysis provided additional evidence for distinguishing between competing recommendations, particularly where model predictions differed.

### Step 8 — Make function-specific final selections

I selected the final queries using the combined evidence from GP diagnostics, UCB/EI agreement, neural-network validation, historical performance and candidate neighbourhoods.

| Function | Selected method | Confidence | Main reason |
|---|---|---|---|
| 1 | EI / UCB agreement | Low | GP candidate closer to observations; both models uncertain |
| 2 | Expected Improvement | High | Recent EI success and strong GP agreement |
| 3 | Expected Improvement | High | EI predicts improvement and NN validation is weak |
| 4 | Expected Improvement | Medium | Promising local region despite mixed evidence |
| 5 | Neural Network | High | Very close to new best and strong NN validation |
| 6 | Expected Improvement | Medium–High | Improved GP fit and promising EI prediction |
| 7 | EI / UCB agreement | Medium–High | GP candidate better supported by nearby observations |
| 8 | Neural Network | Medium | Strong NN validation and closer to successful observations |

**Function 1:** Neither model provided particularly reliable evidence. The GP remained at its minimum length scale and the NN failed cross-validation. However, the common EI/UCB candidate was closer to existing observations and was selected as the more conservative option.

**Function 2:** EI was selected following its successful previous query and the extremely close agreement between the current UCB and EI candidates. The NN failed validation and selected a much more distant point.

**Function 3:** EI was preferred because it predicted an improvement over the current best and was supported by stronger local observations than the NN candidate.

**Function 4:** EI was retained despite mixed evidence. The candidate remained close to the best observed region while avoiding some of the weaker recent observations.

**Function 5:** The neural network was selected in preference to EI. Both approaches identified the same productive region, but the NN candidate was substantially closer to the new best observation and continued to perform strongly in cross-validation.

**Function 6:** EI was selected following the substantial improvement in the GP's fitted length scale. The candidate predicted a meaningful improvement over the current best and provided an opportunity to test whether the recovered GP could now guide optimisation successfully.

**Function 7:** The identical UCB/EI candidate was preferred because it remained closer to the strongest historical observations than the NN alternative, despite the previous GP-selected query failing to improve.

**Function 8:** The neural network was selected because its candidate was closer to successful historical observations and predicted an improvement over the current best. However, confidence remained moderate because the previous NN selection had not improved the function.

Overall, the final selections comprised four EI candidates, two EI/UCB agreement candidates and two neural-network candidates.

### Step 9 — Validate the final proposed queries

I validated all eight selected points to ensure that every coordinate was within the permitted [0, 1] input range and that no point duplicated an existing observation.

| Function | Selection method | Nearest observation distance |
|---|---|---:|
| 1 | EI / UCB agreement | 0.180671 |
| 2 | Expected Improvement | 0.030884 |
| 3 | Expected Improvement | 0.140675 |
| 4 | Expected Improvement | 0.104897 |
| 5 | Neural Network | 0.028637 |
| 6 | Expected Improvement | 0.213058 |
| 7 | EI / UCB agreement | 0.172684 |
| 8 | Neural Network | 0.397197 |

All eight proposed queries passed the validation checks.

The final queries, formatted to six decimal places, were:

| Function | Selection method | Proposed query coordinates |
|---|---|---|
| 1 | EI / UCB agreement | `[0.773956, 0.438878]` |
| 2 | Expected Improvement | `[0.729513, 0.111999]` |
| 3 | Expected Improvement | `[0.308260, 0.310001, 0.479247]` |
| 4 | Expected Improvement | `[0.329887, 0.354827, 0.355151, 0.453774]` |
| 5 | Neural Network | `[0.939869, 0.992261, 0.978815, 0.999723]` |
| 6 | Expected Improvement | `[0.397519, 0.393747, 0.471443, 0.711380, 0.165385]` |
| 7 | EI / UCB agreement | `[0.174885, 0.324058, 0.479146, 0.269144, 0.331135, 0.614490]` |
| 8 | Neural Network | `[0.042609, 0.051269, 0.314729, 0.127230, 0.965221, 0.369000, 0.150066, 0.389171]` |

## Weekly summary and next steps

Week 6 produced new best observations for Functions 2, 3, 5 and 6, with Function 5 achieving a particularly substantial improvement of approximately 37.8%.

The main methodological development was introducing a historical-performance check to evaluate whether previous model selections had actually improved the best-known outputs.

This provided additional evidence for the final decisions without introducing a formal weighting system or increasing model complexity.

Function 6 showed an important improvement in its Gaussian Process fit, with the fitted length scale increasing from 1e-05 to 0.535. This suggested that the growing dataset was allowing the model to identify more meaningful structure and justified reconsidering GP-based acquisition methods.

The neural-network ensembles continued to provide useful alternative recommendations, although cross-validation showed that they were not consistently reliable for all functions.

Functions 4–8 outperformed the mean baseline, while Functions 1–3 did not. This reinforced the importance of treating neural-network predictions as supporting evidence rather than automatically selecting their highest predicted outputs.

The historical-performance check also highlighted differences between functions. The repeated success of EI for Function 5 supported continued exploitation, while the previous failure of the NN candidate for Function 8 encouraged a more cautious interpretation of its new recommendation.

The final decisions reflected these differences. Function 5 used the neural network for a local refinement around its latest best observation, Function 6 returned to EI following the improvement in its GP, and Function 8 retained the NN because its candidate had stronger neighbourhood and validation evidence.

The next steps were to:

- Submit the eight proposed query points and assess their returned outputs.
- Incorporate the new observations and refit the GP and neural-network models.
- Evaluate whether the GP-based selection for Function 6 improves performance.
- Continue monitoring the historical success of different selection methods.
- Reassess candidate neighbourhoods and model agreement as the datasets grow.

Overall, Week 6 demonstrated that no single surrogate or acquisition method was consistently preferable across all eight functions.

Combining model predictions with predictive validation, historical performance and actual neighbouring observations provided a more balanced basis for selecting new queries.


