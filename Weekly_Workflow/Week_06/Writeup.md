
# Week 06 — Historical Model Performance and Function-Specific Optimisation

**Module 17 | Week 6**

## Objective

The objective of Week 6 was to refine the existing optimisation approach by incorporating the historical performance of previous query selections into the decision-making process.

In Week 5, I introduced neural-network ensembles as an alternative surrogate to Gaussian Process Regression (GPR). This provided additional evidence for identifying promising regions, particularly where the GP models had struggled.

This week, I retained the same GP and neural-network approaches but introduced a historical performance check to assess whether previous selections had actually improved the best-known outputs.

Rather than introducing additional model complexity, the aim was to make more informed function-specific decisions by combining model predictions, predictive reliability, historical performance and the outputs of nearby observations.

## Methodology

### Step 1 — Incorporate and Evaluate the Latest Observations

I added the outputs returned from the Week 5 queries to the existing datasets and compared them with the previous best observations.

| Function | Previous Best | New Output | Updated Best | Improved? |
|:--|--:|--:|--:|:--|
| 1 | ≈0 | ≈0 | ≈0 | No |
| **2** | **0.611205** | **0.746962** | **0.746962** | **Yes** |
| **3** | **-0.020167** | **-0.016451** | **-0.016451** | **Yes** |
| 4 | -0.300645 | -0.638343 | -0.300645 | No |
| **5** | **4857.449601** | **6691.931674** | **6691.931674** | **Yes** |
| **6** | **-0.458511** | **-0.456874** | **-0.456874** | **Yes** |
| 7 | 2.536939 | 1.644736 | 2.536939 | No |
| 8 | 9.911249 | 9.742277 | 9.911249 | No |

Four of the eight Week 5 queries produced new best outputs.

Function 5 achieved the largest improvement, increasing from **4857.45 to 6691.93**, a gain of approximately **37.8%**. This continued the strong upward trend observed across recent rounds.

Function 2 improved from 0.6112 to 0.7470, while Function 3 achieved a smaller improvement from -0.02017 to -0.01645.

Function 6 also produced a new best, improving slightly from -0.45851 to -0.45687. This was particularly interesting because the previous query had been selected using the neural-network surrogate.

Functions 4, 7 and 8 did not improve, while Function 1 again returned an output effectively equal to zero.

These mixed results provided useful evidence for assessing the historical success of different selection methods later in the analysis.

### Step 2 — Refit and Diagnose the Gaussian Processes

I refitted the RBF Gaussian Process models using all available observations and retained the same modelling settings as previous rounds.

The fitted length scales were examined to assess whether the models continued to identify meaningful relationships within the data.

| Function | Fitted RBF Length Scale |
|:--|--:|
| **1** | **1e-5** |
| 2 | 0.0521 |
| 3 | 0.185 |
| 4 | 0.423 |
| 5 | 0.389 |
| **6** | **0.535** |
| 7 | 0.248 |
| 8 | 1.000 |

Most functions continued to produce reasonable fitted length scales.

Function 1 remained problematic, with its GP again converging to the minimum permitted length scale of `1e-5`. This suggested that the model was still struggling to identify a smooth relationship across the available observations.

However, **Function 6 showed a substantial improvement**.

Its fitted length scale increased from `1e-5` in Week 5 to **0.535** in Week 6. This suggested that the additional observation had allowed the GP to identify more structure in the function.

As a result, I reconsidered the use of GP-based acquisition methods for Function 6 rather than automatically favouring the neural network.

#### Prediction and Uncertainty Diagnostics

I evaluated the Gaussian Processes across a large random candidate set and recorded the variation in predicted means and predictive uncertainty.

| Function | Predicted Mean Range | Uncertainty Range |
|:--|--:|--:|
| 1 | 0.000635 | 0.001654 |
| 2 | 0.831417 | 0.259242 |
| 3 | 0.389101 | 0.078040 |
| 4 | 34.434331 | 7.386272 |
| 5 | 7326.020374 | 1726.249284 |
| **6** | **2.348109** | **0.508036** |
| 7 | 2.575815 | 0.664717 |
| 8 | 4.822485 | 0.659779 |

The diagnostics showed meaningful variation in predicted outputs and uncertainty for Functions 2–8.

Function 6 was particularly encouraging, with its predicted mean now ranging from approximately **-2.662 to -0.313**.

This supported the view that its GP had become more informative following the additional observations.

Function 1 continued to show limited predicted variation and an extremely small fitted length scale, so confidence in its GP recommendations remained low.

### Step 3 — Generate UCB and Expected Improvement Candidates

I generated candidate points using the updated GP models and compared two acquisition strategies:

- **Upper Confidence Bound (UCB)** with κ = 1.5.
- **Expected Improvement (EI)**.

Both approaches were evaluated on the same candidate set, allowing their recommendations to be compared directly.

| Function | UCB Predicted Output | EI Predicted Output | UCB–EI Distance |
|:--|--:|--:|--:|
| 1 | -0.000635 | -0.000635 | 0.0000 |
| 2 | 0.644023 | 0.679240 | 0.0077 |
| 3 | -0.048421 | -0.010868 | 0.3695 |
| 4 | -0.455899 | -0.278717 | 0.0766 |
| 5 | 6213.849027 | 6213.849027 | 0.0000 |
| 6 | -0.338919 | -0.313445 | 0.1444 |
| 7 | 2.512543 | 2.512543 | 0.0000 |
| 8 | 9.793025 | 9.793025 | 0.0000 |

UCB and EI selected identical candidates for Functions **1, 5, 7 and 8**.

Function 2 also showed very strong agreement, with the two recommendations separated by only **0.0077**.

Functions 3, 4 and 6 showed greater differences between acquisition methods.

For Function 3, EI predicted an output of **-0.010868**, above the current best of -0.016451, while UCB selected a more exploratory point with a weaker predicted output.

Function 5 continued to focus on its established high-performing region, with both acquisition methods selecting exactly the same candidate.

For Function 6, the improved GP produced promising predictions. EI predicted **-0.313445**, substantially better than the current best of -0.456874.

This provided a reason to reconsider GP-based optimisation for Function 6 following its earlier modelling difficulties.

Although Function 1 showed complete agreement between UCB and EI, this was treated cautiously because both acquisition functions depended on the same problematic GP model.

### Step 4 — Retrain and Evaluate the Neural-Network Ensembles

I retained the neural-network architecture introduced in Week 5 to allow more consistent comparisons as additional observations became available.

Each function used an ensemble of five independently initialised networks, each containing:

- Two hidden layers with 16 neurons each.
- ReLU activation functions.
- Standardised inputs and outputs.

#### Training Performance

| Function | Mean Final Training Loss |
|:--|--:|
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

However, low training loss alone does not establish whether a model generalises to unseen observations.

I therefore repeated Leave-One-Out Cross-Validation.

#### Leave-One-Out Cross-Validation

| Function | NN MAE | Mean Baseline MAE | NN Better? |
|:--|--:|--:|:--|
| 1 | 0.001675 | 0.001179 | No |
| 2 | 0.267443 | 0.241496 | No |
| 3 | 0.067227 | 0.048537 | No |
| 4 | 1.827993 | 6.878180 | Yes |
| 5 | 273.005550 | 1375.410321 | Yes |
| 6 | 0.309082 | 0.476405 | Yes |
| 7 | 0.323396 | 0.507559 | Yes |
| 8 | 0.294854 | 0.920553 | Yes |

The neural networks outperformed the mean baseline for **five of the eight functions**.

Functions 4–8 continued to demonstrate useful predictive performance, with Function 5 showing a particularly large improvement over the baseline.

Functions 1 and 3 again failed to outperform the baseline.

Function 2 also performed slightly worse than the baseline this week, despite having outperformed it in Week 5.

These results suggested that neural-network recommendations for Functions 1–3 should receive less weight in the final selection process.

#### Neural-Network Candidate Generation

I then used the trained ensembles to generate alternative candidate points, recording the mean predicted output and disagreement between the five networks.

| Function | NN Predicted Output | Model Disagreement |
|:--|--:|--:|
| 1 | 0.003677 | 0.002351 |
| 2 | 2.724940 | 1.237037 |
| 3 | -0.001139 | 0.032445 |
| 4 | -0.374836 | 0.459252 |
| 5 | 6691.842285 | 47.407719 |
| 6 | -0.406389 | 0.128387 |
| 7 | 2.412130 | 0.419323 |
| 8 | 10.052735 | 0.132650 |

Function 2 produced an unusually high predicted output of **2.724940**, but this was accompanied by substantial disagreement between the networks and weak cross-validation performance.

Consequently, this recommendation was not considered sufficiently reliable.

Function 5's neural network predicted approximately **6691.84**, close to its current best observation of 6691.93, suggesting that it had identified the same high-performing region.

Function 6's neural network predicted -0.406389, above its current best, while Function 8's network predicted **10.052735**, also exceeding its existing best.

These recommendations were retained for comparison with the GP candidates.

### Step 5 — Compare GP and Neural-Network Recommendations

I calculated the Euclidean distances between the neural-network, UCB and EI candidate points.

This provided an indication of whether the different models were identifying similar promising regions.

| Function | NN → UCB | NN → EI | UCB → EI |
|:--|--:|--:|--:|
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

Function 8 produced the largest disagreement between the GP and neural network, with their recommendations separated by **0.7737**.

Although the NN continued to perform well in cross-validation, its previous query had failed to improve the function, so the recommendation required further investigation.

The results reinforced the importance of considering multiple sources of evidence rather than treating model agreement as sufficient justification for selecting a point.

### Step 6 — Introduce a Historical Model-Performance Check

The main methodological development in Week 6 was incorporating the historical success of previous query selections into the final decision process.

For each function, I reviewed the method used in the previous round and whether its selected query improved the best-known output.

| Function | Previous Selection Method | Improvement | Outcome |
|:--|:--|--:|:--|
| 1 | EI / UCB agreement | ≈0 | No improvement |
| 2 | Expected Improvement | +0.135757 | New best |
| 3 | Expected Improvement | +0.003716 | New best |
| 4 | Expected Improvement | -0.337698 | No improvement |
| 5 | Expected Improvement | +1834.482072 | New best |
| 6 | Neural Network | +0.001637 | New best |
| 7 | EI / UCB agreement | -0.892203 | No improvement |
| 8 | Neural Network | -0.168972 | No improvement |

The historical comparison showed that Expected Improvement had been successful for Functions **2, 3 and 5**, while the neural network had produced a small improvement for Function 6.

Function 5's improvement was particularly substantial, strengthening the case for continuing to exploit the productive region identified by the GP.

The previous neural-network selection for Function 8 had failed to improve the best output, reducing confidence in repeating the approach without additional supporting evidence.

Similarly, the previous GP-agreement selection for Function 7 had not improved the function, so agreement between UCB and EI was not considered sufficient justification on its own.

This historical performance check was used as **supporting evidence rather than a formal weighting system**, since the success or failure of a single query does not necessarily establish whether a model is reliable.

### Step 7 — Examine Candidate Neighbourhoods

I examined the nearest existing observations surrounding each UCB, EI and neural-network candidate.

This allowed the model recommendations to be compared with actual observed outputs rather than relying entirely on surrogate predictions.

| Function | UCB Nearest Distance | EI Nearest Distance | NN Nearest Distance |
|:--|--:|--:|--:|
| 1 | 0.1807 | 0.1807 | 0.3077 |
| 2 | 0.0385 | 0.0309 | 0.3164 |
| 3 | 0.2255 | 0.1407 | 0.2033 |
| 4 | 0.1042 | 0.1049 | 0.0619 |
| 5 | 0.1651 | 0.1651 | 0.0286 |
| 6 | 0.2812 | 0.2131 | 0.2223 |
| 7 | 0.1727 | 0.1727 | 0.2813 |
| 8 | 0.6614 | 0.6614 | 0.3972 |

For Function 2, EI was only **0.0309** from the new best observation of 0.746962, providing strong evidence for continued exploitation.

Function 3's EI candidate was also relatively close to strong existing observations and was better supported than the neural-network alternative.

For Function 4, the NN candidate was closest to the best observation but was also near several weaker recent outputs. EI remained within the high-performing region while moving slightly away from those unsuccessful points.

Function 5's neural-network candidate was particularly interesting, lying only **0.0286** from the new best observation of 6691.93.

This compared with a distance of 0.1651 for the GP candidate and suggested that a small local refinement around the most successful observation might be worthwhile.

For Function 6, all three approaches identified regions around the strongest existing observations, supporting the view that the GP and neural network were beginning to converge.

Function 7's GP candidate was closer to its strongest observations than the NN alternative.

For Function 8, the NN candidate was substantially closer to previous strong observations than the GP candidate, although both remained relatively exploratory.

### Step 8 — Make Function-Specific Final Selections

The final queries were selected by considering model reliability, acquisition-function agreement, predicted performance, historical success and neighbouring observed outputs.

| Function | Selected Method | Confidence | Main Reason |
|:--|:--|:--|:--|
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

Overall, the final selections comprised **four EI candidates, two EI/UCB agreement candidates and two neural-network candidates**.

### Step 9 — Validate and Finalise the Queries

I checked that all selected coordinates were within the permitted `[0, 1]` search space and that none duplicated an existing observation.

| Function | Selection Method | Nearest Observation Distance | Valid |
|:--|:--|--:|:--|
| 1 | EI / UCB agreement | 0.180671 | Yes |
| 2 | Expected Improvement | 0.030884 | Yes |
| 3 | Expected Improvement | 0.140675 | Yes |
| 4 | Expected Improvement | 0.104897 | Yes |
| 5 | Neural Network | 0.028637 | Yes |
| 6 | Expected Improvement | 0.213058 | Yes |
| 7 | EI / UCB agreement | 0.172684 | Yes |
| 8 | Neural Network | 0.397197 | Yes |

All eight selected candidates passed the validation checks.

#### Final Query Points

    Function 1: 0.773956-0.438878
    Function 2: 0.729513-0.111999
    Function 3: 0.308260-0.310001-0.479247
    Function 4: 0.329887-0.354827-0.355151-0.453774
    Function 5: 0.939869-0.992261-0.978815-0.999723
    Function 6: 0.397519-0.393747-0.471443-0.711380-0.165385
    Function 7: 0.174885-0.324058-0.479146-0.269144-0.331135-0.614490
    Function 8: 0.042609-0.051269-0.314729-0.127230-0.965221-0.369000-0.150066-0.389171

## Results and Reflection

Week 6 demonstrated the value of incorporating historical optimisation performance into the selection process rather than relying exclusively on surrogate predictions.

Four of the eight previous queries produced new best outputs, with Function 5 achieving a particularly substantial improvement.

The Gaussian Process for Function 6 also showed an important recovery, with its fitted length scale increasing from `1e-5` to 0.535. This suggested that the growing dataset was allowing the model to identify a more meaningful relationship and justified reconsidering GP-based acquisition methods.

The neural-network ensembles continued to provide useful alternative recommendations, although cross-validation showed that they were not consistently reliable for all functions.

Functions 4–8 outperformed the mean baseline, while Functions 1–3 did not. This reinforced the importance of treating neural-network predictions as supporting evidence rather than automatically selecting their highest predicted outputs.

The historical-performance check added a further layer of information. In particular, the repeated success of EI for Function 5 supported continued exploitation of its high-performing region, while the previous failure of the NN candidate for Function 8 encouraged a more cautious interpretation of its new recommendation.

The final decisions reflected these differences. Function 5 used the neural network for a local refinement around its latest best observation, Function 6 returned to EI following the improvement in its GP, and Function 8 retained the NN because its candidate had stronger neighbourhood and validation evidence.

Overall, the main lesson from Week 6 was that **no single surrogate or acquisition method was consistently preferable across all eight functions**.

Combining model predictions with predictive validation, historical performance and actual neighbouring observations provided a more balanced basis for selecting new queries.

The next iteration will assess whether these decisions improve the observed maxima and whether the increasing datasets provide stronger evidence for choosing between GP and neural-network recommendations.

