
# Week 03 — Gaussian Process Optimisation and SVM Analysis

**Module 14 | Week 3**

## Objective

The aim of Week 3 was to incorporate the latest observations, evaluate the performance of the previous queries and select the next input for each of the eight unknown functions. I continued using **Gaussian Process Regression (GPR)** and **Upper Confidence Bound (UCB)** as the primary optimisation approach, while introducing **Support Vector Machines (SVMs)** to investigate whether classification could provide additional guidance.

Particular attention was given to Functions 1 and 6, where previous Gaussian Process models had experienced convergence issues.

## Methodology

### Step 1 — Update the Observations and Assess Improvements

I added the eight outputs returned from the second-round queries to the existing datasets and compared them with the previous best observations.

| Function | Previous Best | New Output | Updated Best | Improved? |
|:--|--:|--:|--:|:--|
| 1 | ≈0 | ≈0 | ≈0 | No |
| 2 | 0.611205 | 0.366010 | 0.611205 | No |
| 3 | -0.020167 | -0.060870 | -0.020167 | No |
| 4 | -0.300645 | -3.743172 | -0.300645 | No |
| **5** | **2329.015854** | **2765.819787** | **2765.819787** | **Yes** |
| **6** | **-0.714265** | **-0.458511** | **-0.458511** | **Yes** |
| 7 | 1.950612 | 0.975233 | 1.950612 | No |
| 8 | 9.821986 | 9.621122 | 9.821986 | No |

Functions **5 and 6** produced new best outputs. Function 5 improved by approximately **18.8%**, continuing the positive trend from the previous round and suggesting that the search was identifying a promising region.

Function 6 also improved, moving from -0.7143 to -0.4585. This was encouraging because its Gaussian Process had previously experienced modelling difficulties. The remaining six functions did not improve, although their new observations still contributed information for the next round of model fitting.

### Step 2 — Refit the Gaussian Processes and Investigate Convergence

I refitted the RBF Gaussian Processes using the expanded datasets and examined their optimised length scales.

| Function | Fitted RBF Length Scale |
|:--|--:|
| **1** | **1e-5** |
| 2 | 0.0427 |
| 3 | 0.202 |
| 4 | 0.495 |
| 5 | 0.413 |
| **6** | **1e-5** |
| 7 | 0.244 |
| 8 | 0.931 |

Most functions continued to produce usable fitted models. Function 5's length scale increased from approximately **0.225 to 0.413**, suggesting a change in the model's estimated relationship between nearby inputs.

However, Functions **1 and 6** both reached the minimum RBF length scale of `1e-5`. I therefore investigated their predictions across 100,000 randomly generated candidate points to determine whether the models could meaningfully distinguish between different inputs.

| Diagnostic | Function 1 | Function 6 |
|:--|--:|--:|
| Predicted mean range | 0.0003003 | 0.0000000 |
| Uncertainty range | 0.0009625 | 0.0000000 |
| GP suitable for UCB? | Questionable | No |

Function 1 showed some variation in its predictions, but the recurring convergence warning indicated that the GP remained unreliable.

Function 6 was more problematic. Its predicted means and uncertainties were effectively identical across all candidate points. As a result, UCB would provide no meaningful basis for ranking unexplored inputs.

This demonstrated why examining predictive behaviour was more useful than relying solely on convergence warnings or previous query performance.

### Step 3 — Introduce and Compare SVM Models

I introduced SVMs to investigate whether the observed inputs could distinguish between relatively high- and low-performing regions.

Because the functions produce continuous outputs, I converted the observations into binary classes using the **median output for each function**. Observations above the median were classified as high-performing, while those at or below it were classified as low-performing.

I then fitted both linear and RBF-kernel SVMs and compared their performance using **three-fold cross-validation**.

| Function | Linear SVM Accuracy | RBF SVM Accuracy | Better Model |
|:--|--:|--:|:--|
| 1 | 0.167 | 0.500 | RBF |
| 2 | 0.667 | 0.833 | RBF |
| 3 | 0.344 | 0.522 | RBF |
| 4 | 0.627 | 0.782 | RBF |
| 5 | 0.649 | 0.458 | Linear |
| **6** | **0.821** | **0.768** | **Linear** |
| 7 | 0.785 | 0.815 | RBF |
| 8 | 0.857 | 0.833 | Linear |

The results varied across functions. RBF SVMs performed better for Functions 1, 2, 3, 4 and 7, while linear SVMs performed better for Functions 5, 6 and 8.

Functions **2 and 4** showed particularly useful improvements with the nonlinear RBF kernel, suggesting that their higher-performing observations may be separated by more complex boundaries.

For **Function 6**, the linear SVM achieved an accuracy of **0.821**, making it a potentially useful alternative source of guidance given the failure of its current Gaussian Process.

However, SVM classification does not directly predict continuous function outputs or provide the predictive uncertainty required by UCB. I therefore treated the results as supplementary evidence rather than replacing Bayesian Optimisation for every function.

### Step 4 — Reassess Exploration and Exploitation

I combined the latest observations, GP diagnostics and SVM results to determine the most appropriate query strategy for each function.

| Function | Selected Strategy | Reasoning |
|:--|:--|:--|
| 1 | Maximum-distance exploration | GP remains unreliable; SVM results inconclusive |
| 2 | GP + UCB | GP usable; nonlinear SVM suggests useful structure |
| 3 | GP + UCB | Stable GP; limited evidence for changing strategy |
| 4 | GP + UCB | Promising previously identified region |
| 5 | GP + UCB | Successive improvements support further exploitation |
| 6 | Linear SVM | GP predictions uninformative; strong SVM accuracy |
| 7 | GP + UCB | GP remains usable |
| 8 | GP + UCB | Stable GP; retain exploration–exploitation balance |

For the six functions with usable Gaussian Processes, I retained **UCB with κ = 2.5**, consistent with the previous round.

Functions 1 and 6 required separate strategies because their GP diagnostics indicated that the normal acquisition process was unsuitable or unreliable.

### Step 5 — Generate and Evaluate Candidate Queries

For Functions 2, 3, 4, 5, 7 and 8, I generated 100,000 candidate points and calculated:

**UCB(x) = μ(x) + 2.5σ(x)**

The candidates with the highest UCB scores were selected for further evaluation.

| Function | Predicted Output | Uncertainty | UCB Score |
|:--|--:|--:|--:|
| 2 | 0.449335 | 0.185666 | 0.913500 |
| 3 | -0.076907 | 0.077556 | 0.116983 |
| 4 | -0.064270 | 0.420976 | 0.988170 |
| 5 | 2631.682259 | 240.331386 | 3232.510725 |
| 7 | 1.662117 | 0.258204 | 2.307626 |
| 8 | 9.635913 | 0.434286 | 10.721628 |

The results showed different exploration–exploitation behaviour across functions.

**Functions 2, 4 and 7** selected points relatively close to their current best observations, indicating a greater emphasis on exploitation. Function 4 was particularly interesting because its predicted output of **-0.0643** was better than the current observed maximum of **-0.3006**, although this prediction was subject to considerable uncertainty.

**Function 5** continued to search around its previously successful region. Its selected candidate retained high values for the second, third and fourth input dimensions, consistent with the pattern that had produced successive improvements.

In contrast, **Functions 3 and 8** selected more distant candidates, suggesting that uncertainty was encouraging greater exploration.

This supported retaining κ = 2.5, as the acquisition function was already producing different search behaviours depending on the individual GP predictions.

### Step 6 — Apply Alternative Strategies to Functions 1 and 6

**Function 1 — Maximum-Distance Exploration**

Function 1's GP continued to experience convergence problems, while neither SVM provided convincing evidence of a reliable high-performing region.

I therefore selected the candidate with the greatest distance from its nearest existing observation.

- Selected point: `[0.000045, 0.003062]`
- Distance to nearest observation: **0.321671**

This deliberately explored a relatively unsampled part of the search space rather than relying on potentially misleading GP predictions.

**Function 6 — Linear SVM Selection**

Function 6's GP produced constant predictions and uncertainty across the candidate space, making UCB unsuitable.

I instead used the fitted linear SVM to score 100,000 candidate points and selected the candidate with the strongest decision score for the higher-performing class.

- Selected point: `[0.013758, 0.147930, 0.445597, 0.994845, 0.013469]`
- SVM decision score: **5.2721**
- Distance from current best: **0.4028**

The selected candidate was classified strongly within the higher-performing region while remaining some distance from the best existing observation.

Although the SVM score is not a prediction of the function's output, this provided an alternative approach to selecting a query when the Gaussian Process could not distinguish between candidates.

### Step 7 — Validate and Finalise the Queries

I checked that all eight selected points were within the permitted `[0, 1]` input range, did not duplicate previous observations and matched the intended selection strategies.

| Function | Method | Nearest Observation Distance | Valid |
|:--|:--|--:|:--|
| 1 | Maximum-distance | 0.321671 | Yes |
| 2 | GP + UCB | 0.045156 | Yes |
| 3 | GP + UCB | 0.306635 | Yes |
| 4 | GP + UCB | 0.137608 | Yes |
| 5 | GP + UCB | 0.203041 | Yes |
| 6 | Linear SVM | 0.402781 | Yes |
| 7 | GP + UCB | 0.161151 | Yes |
| 8 | GP + UCB | 0.800486 | Yes |

All eight queries passed validation. The final points were formatted to six decimal places for submission.

#### Final Query Points

    Function 1: 0.000045-0.003062
    Function 2: 0.709627-0.881953
    Function 3: 0.732035-0.997300-0.015653
    Function 4: 0.391976-0.424757-0.494823-0.464424
    Function 5: 0.671629-0.981118-0.990458-0.985189
    Function 6: 0.013758-0.147930-0.445597-0.994845-0.013469
    Function 7: 0.021549-0.313334-0.360216-0.141732-0.295328-0.686548
    Function 8: 0.025529-0.451238-0.098209-0.031391-0.412421-0.138573-0.114934-0.994361

## Results and Reflection

Week 3 produced new best observed outputs for **Functions 5 and 6**, while the expanded datasets provided further evidence about the behaviour of the remaining functions.

The main development was introducing SVMs as an additional diagnostic and using their results to inform function-specific optimisation decisions. Although GP-UCB remained appropriate for most functions, the persistent modelling problems for Functions 1 and 6 demonstrated that alternative selection methods could be valuable.

Function 5 continued to provide evidence that exploiting a promising region could produce successive improvements. Function 6 illustrated a different challenge: despite achieving a better observed output, its refitted Gaussian Process became uninformative, requiring an alternative query strategy.

In the next round, I will assess the outputs from these selections and investigate whether adjusting the UCB exploration parameter or introducing another acquisition function could improve the balance between exploration and exploitation.

