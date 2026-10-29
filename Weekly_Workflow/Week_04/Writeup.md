
# Week 04 — UCB Exploration Tuning and Expected Improvement

**Module 15 | Week 4**

## Objective

The aim of Week 4 was to incorporate the latest observations, reassess the Gaussian Process (GP) models and investigate whether changing the exploration parameter in Upper Confidence Bound (UCB) could improve query selection.

I compared three UCB exploration settings (κ = 1.5, 2.5 and 3.5) and introduced **Expected Improvement (EI)** as an alternative acquisition function. The objective was to select appropriate strategies for each of the eight functions rather than continuing with a fixed exploration setting.

## Methodology

### Step 1 — Update the Observations and Assess Improvements

I added the outputs returned from the Week 3 queries to the existing datasets and compared them with the previous best results.

| Function | Previous Best | New Output | Updated Best | Improved? |
|:--|--:|--:|--:|:--|
| 1 | ≈0 | ≈0 | ≈0 | No |
| 2 | 0.611205 | 0.548608 | 0.611205 | No |
| 3 | -0.020167 | -0.115141 | -0.020167 | No |
| 4 | -0.300645 | -1.850268 | -0.300645 | No |
| **5** | **2765.819787** | **4595.479318** | **4595.479318** | **Yes** |
| 6 | -0.458511 | -1.182842 | -0.458511 | No |
| **7** | **1.950612** | **2.139551** | **2.139551** | **Yes** |
| 8 | 9.821986 | 9.645301 | 9.821986 | No |

Functions **5 and 7** achieved new best outputs.

Function 5 improved from **2765.82 to 4595.48**, an increase of approximately **66.2%**, reinforcing the value of searching its previously identified high-performing region.

Function 7 also improved from **1.9506 to 2.1396**, suggesting that GP-UCB was continuing to identify promising candidates.

The previous SVM-informed selection for Function 6 did not improve its best output, suggesting that classification alone might not provide sufficiently precise guidance for maximising the continuous function.

### Step 2 — Refit and Diagnose the Gaussian Processes

I refitted the RBF Gaussian Process models using all available observations and examined their fitted length scales.

During the initial fitting, I noticed that the optimiser restart parameter had been omitted. This caused Functions 1–7 to return the minimum permitted length scale. I corrected this by restoring `n_restarts_optimizer=10`, consistent with the earlier methodology, before interpreting the models.

| Function | Fitted RBF Length Scale |
|:--|--:|
| **1** | **1e-5** |
| 2 | 0.0516 |
| 3 | 0.198 |
| 4 | 0.428 |
| 5 | 0.367 |
| **6** | **1e-5** |
| 7 | 0.296 |
| 8 | 0.949 |

Functions 2, 3, 4, 5, 7 and 8 produced length scales above the lower bound and were retained for GP-based acquisition comparisons.

Functions **1 and 6** again reached the minimum length scale of `1e-5`. I therefore examined their predictions across 100,000 candidate points.

| Diagnostic | Function 1 | Function 6 |
|:--|--:|--:|
| Predicted mean range | 0.0002774 | 0.0000000 |
| Uncertainty range | 0.0009508 | 0.0000000 |
| GP suitable for UCB? | Questionable | No |

Function 1 showed some variation in predictions and uncertainty, but its persistent convergence problems remained concerning.

Function 6 produced effectively identical predictions and uncertainty across the candidate space. Consequently, changing κ would not meaningfully change the candidate ranking, so I excluded it from the UCB comparison.

### Step 3 — Compare Different UCB Exploration Settings

In previous rounds, I used a fixed UCB exploration parameter of **κ = 2.5**. This week, I compared κ values of **1.5, 2.5 and 3.5** for Functions 2, 3, 4, 5, 7 and 8.

The acquisition function was:

**UCB(x) = μ(x) + κσ(x)**

A lower κ gives greater weight to the predicted mean, encouraging exploitation, while a higher κ gives greater weight to uncertainty and exploration.

I evaluated the three settings against the same 100,000 candidate points for each function.

The table below compares the predicted outputs of the selected candidates.

| Function | κ = 1.5 | κ = 2.5 | κ = 3.5 |
|:--|--:|--:|--:|
| 2 | 0.5560 | 0.3754 | 0.3424 |
| 3 | -0.0339 | -0.0751 | -0.0830 |
| 4 | -0.3091 | -0.6945 | -18.3618 |
| 5 | 4561.79 | 4090.67 | 4090.67 |
| 7 | 1.9591 | 1.9591 | 1.6413 |
| 8 | 10.0591 | 9.5513 | 9.2533 |

The comparison showed that increasing κ generally selected candidates with lower predicted outputs but greater uncertainty.

**Function 4** demonstrated this particularly clearly. At κ = 1.5, the selected candidate had a predicted output of **-0.3091**, close to its current best of -0.3006. At κ = 3.5, the predicted output fell to **-18.3618**, reflecting a much stronger emphasis on exploration.

For **Function 5**, κ = 1.5 selected a point with a predicted output of **4561.79**, close to the current best of 4595.48. Higher κ values selected a more uncertain candidate further from the successful region.

Function 8 also showed a substantial difference, with κ = 1.5 predicting **10.0591**, above the current observed maximum of 9.8220.

I also compared distances between candidates selected under different κ settings.

| Function | Distance: 1.5 → 2.5 | Distance: 2.5 → 3.5 |
|:--|--:|--:|
| 2 | 0.8507 | 0.1135 |
| 3 | 0.6195 | 0.0707 |
| 4 | 0.0478 | 0.9480 |
| 5 | 0.1438 | 0.0000 |
| 7 | 0.0000 | 0.1986 |
| 8 | 0.7627 | 0.9939 |

Functions 2, 3 and 8 showed substantial changes in candidate location as κ increased, indicating that the exploration setting materially influenced the next query.

Function 7 was more stable, with κ = 1.5 and 2.5 selecting exactly the same candidate. Function 5 also selected the same point at κ = 2.5 and 3.5.

Overall, the comparison demonstrated that a single fixed κ may not be appropriate for every function. Based on the recent results and candidate diagnostics, I provisionally preferred **κ = 1.5** for all six functions with usable GPs, although the reasons varied between functions.

### Step 4 — Select Function-Specific Strategies

I combined the latest observations, GP diagnostics and κ comparison to reassess the exploration–exploitation balance.

| Function | Provisional Strategy | Reasoning |
|:--|:--|:--|
| 1 | Maximum-distance exploration | Persistent GP convergence issues |
| 2 | UCB, κ = 1.5 | Stronger predicted output near current best |
| 3 | UCB, κ = 1.5 | Reduced exploration after unsuccessful query |
| 4 | UCB, κ = 1.5 | Avoid excessive exploration |
| 5 | UCB, κ = 1.5 | Exploit repeatedly successful region |
| 6 | Alternative strategy required | GP uninformative; SVM query unsuccessful |
| 7 | UCB, κ = 1.5 | Recent improvement supports local search |
| 8 | UCB, κ = 1.5 | Strong predicted output above current best |

Function 5 provided the strongest evidence for exploitation, having improved from approximately **1088.86 to 4595.48** over successive rounds.

For Functions 2, 3, 4 and 8, the lower κ setting offered stronger predicted outputs following unsuccessful exploratory queries.

Functions 1 and 6 required separate treatment because their Gaussian Process models remained unreliable.

### Step 5 — Compare UCB with Expected Improvement

I introduced **Expected Improvement (EI)** as an alternative acquisition function.

Unlike UCB, which directly combines predicted performance and uncertainty, EI estimates the expected improvement over the current best observed output.

I compared EI-selected candidates with those selected using UCB at κ = 1.5.

| Function | UCB Predicted Output | EI Predicted Output | Distance Between Candidates |
|:--|--:|--:|--:|
| 2 | 0.5560 | 0.5901 | 0.0085 |
| 3 | -0.0339 | -0.0312 | 0.0228 |
| 4 | -0.3091 | -0.3091 | 0.0000 |
| 5 | 4561.79 | 4561.79 | 0.0000 |
| 7 | 1.9591 | 2.0538 | 0.1893 |
| 8 | 10.0591 | 10.0803 | 0.2621 |

EI and UCB selected very similar candidates for **Functions 2 and 3**, with distances of just 0.0085 and 0.0228 respectively.

For **Functions 4 and 5**, both methods selected exactly the same points. This provided additional support for the selected regions, particularly Function 5, where previous queries had produced repeated improvements.

Functions 7 and 8 showed greater disagreement between the methods.

For Function 7, EI selected a candidate with a stronger predicted output and slightly less uncertainty, supporting greater exploitation of the recently successful region.

For Function 8, EI predicted an output of **10.0803**, compared with **10.0591** for UCB. Both predictions exceeded the current best observed output of 9.8220, making this a particularly promising function for the next round.

**Alternative strategies for Functions 1 and 6**

Because the GP models for Functions 1 and 6 remained unreliable, I retained model-independent exploration rather than using UCB or EI.

For both functions, I generated 100,000 candidate points and selected the point with the greatest distance from its nearest existing observation.

| Function | Nearest Observation Distance | Distance from Current Best |
|:--|--:|--:|
| 1 | 0.3153 | 0.4137 |
| 6 | 0.8782 | 0.8810 |

Function 6's selected point was particularly exploratory. However, after the previous SVM-informed query failed to improve performance and the GP continued to produce flat predictions, collecting information from a substantially different region was considered worthwhile.

### Step 6 — Final Selection and Validation

I selected the final eight queries using the combined evidence from the latest observations, GP diagnostics, UCB comparison and Expected Improvement.

EI was preferred for Functions 2, 3, 7 and 8 because it selected candidates with stronger predicted outputs while maintaining an appropriate level of uncertainty.

For Functions 4 and 5, EI and UCB selected identical candidates. Functions 1 and 6 retained maximum-distance exploration.

| Function | Final Selection Method | Nearest Observation Distance | Valid |
|:--|:--|--:|:--|
| 1 | Maximum-distance | 0.315336 | Yes |
| 2 | Expected Improvement | 0.030161 | Yes |
| 3 | Expected Improvement | 0.178778 | Yes |
| 4 | EI / UCB agreement | 0.091845 | Yes |
| 5 | EI / UCB agreement | 0.193062 | Yes |
| 6 | Maximum-distance | 0.878156 | Yes |
| 7 | Expected Improvement | 0.149214 | Yes |
| 8 | Expected Improvement | 0.518070 | Yes |

All eight points were within the permitted `[0, 1]` input range and did not duplicate any previous observations.

#### Final Query Points

    Function 1: 0.417430-0.463246
    Function 2: 0.730707-0.915532
    Function 3: 0.433866-0.716516-0.000863
    Function 4: 0.443740-0.493644-0.409790-0.452219
    Function 5: 0.851064-0.985689-0.919598-0.979330
    Function 6: 0.033216-0.058439-0.124012-0.011504-0.042834
    Function 7: 0.154752-0.313081-0.341855-0.169387-0.318877-0.633022
    Function 8: 0.148540-0.297539-0.201860-0.328776-0.636564-0.500531-0.197949-0.479426

## Results and Reflection

Week 4 demonstrated that changing the exploration parameter can substantially affect the candidates selected by GP-UCB. In particular, Functions 2, 3, 4 and 8 showed that higher κ values could move the search into significantly different regions, often at the expense of predicted performance.

The introduction of Expected Improvement provided an additional way to evaluate promising candidates. Its close agreement with UCB for Functions 2–5 increased confidence in the selected regions, while differences for Functions 7 and 8 highlighted the importance of comparing acquisition strategies rather than relying on a single method.

Function 5 continued to provide strong evidence for exploiting a high-performing region, following another substantial improvement in its observed maximum. Functions 1 and 6 remained challenging because their Gaussian Process models were unable to provide sufficiently reliable guidance, making model-independent exploration necessary.

Overall, I selected a more exploitative strategy for the functions with usable Gaussian Processes, while retaining deliberate exploration for Functions 1 and 6. The next round will help assess whether these changes improve performance and whether alternative surrogate models could provide more useful guidance for the functions that continue to present modelling difficulties.

