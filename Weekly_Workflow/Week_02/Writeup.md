
# Week 2 — Updating the Gaussian Processes and Evaluating Kernel Performance

**Imperial College AI/ML Capstone Project | Module 13**

## Objective

The objective of Week 2 was to incorporate the outputs received from the first round of Bayesian Optimisation, assess the effectiveness of the initial query selections, and update the Gaussian Process (GP) models using the expanded datasets.

The initial results demonstrated that the GP and Upper Confidence Bound (UCB) approach could identify promising regions for several functions. However, Functions 1 and 6 had previously experienced difficulties with extremely small RBF length scales, resulting in largely uninformative predictions.

This week therefore focused on evaluating how the additional observations affected the models, investigating an alternative kernel for Function 1, and continuing to balance exploration and exploitation when selecting the next set of queries.

## 1. Incorporate and Assess the New Observations

The eight outputs returned from the Week 1 queries were added to their respective datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | 0.000000 | 0.000000 | No improvement |
| 2 | 0.611205 | -0.082041 | 0.611205 | No improvement |
| 3 | -0.034835 | -0.020167 | -0.020167 | New best |
| 4 | -4.025542 | -0.300645 | -0.300645 | New best |
| 5 | 1088.859618 | 2329.015854 | 2329.015854 | New best |
| 6 | -0.714265 | -2.661556 | -0.714265 | No improvement |
| 7 | 1.364968 | 1.950612 | 1.950612 | New best |
| 8 | 9.598482 | 9.821986 | 9.821986 | New best |

Five of the eight queries produced new best observations, with Functions 4 and 5 showing particularly substantial improvements.

Function 4 improved from approximately -4.026 to -0.301, while Function 5 increased from approximately 1088.86 to 2329.02. Functions 3, 7 and 8 also improved on their previous maxima, suggesting that the UCB approach had successfully identified promising regions of their search spaces.

Functions 1, 2 and 6 did not improve. However, Functions 1 and 6 had deliberately used exploration-based selections in Week 1 because their Gaussian Processes provided limited guidance. Although these queries did not improve the observed maxima, they provided additional information from relatively unexplored areas.

## 2. Refit the RBF Gaussian Process Models

Following the incorporation of the new observations, the RBF Gaussian Process models were refitted.

The fitted length scales were examined to determine whether the additional data had improved the models' ability to represent the unknown functions.

| Function | Week 2 Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0453 |
| 3 | 0.202 |
| 4 | 0.522 |
| 5 | 0.225 |
| 6 | 0.481 |
| 7 | 0.354 |
| 8 | 0.913 |

Most functions produced fitted length scales that were broadly consistent with the previous round.

The most significant change occurred in **Function 6**, where the fitted length scale increased from the lower bound of `1e-05` to approximately `0.481`.

Although Function 6's exploratory query did not improve its highest observed output, the additional observation appears to have helped the GP identify relationships across a wider region of the input space.

This suggested that Function 6 could return to the standard GP and UCB approach for the next query.

**Function 1 remained problematic**, with the fitted length scale continuing to reach the lower bound of `1e-05`. This indicated that the RBF model was still struggling to represent the function meaningfully.

An alternative kernel was therefore investigated before selecting its next query.

## 3. Compare RBF and Matérn Kernels for Function 1

To investigate whether the difficulties with Function 1 were caused by the smoothness assumptions of the RBF kernel, an alternative Matérn kernel was fitted and compared against the existing model.

The Matérn kernel was configured with `nu = 2.5`.

### Fitted Kernel Comparison

| Kernel | Fitted Length Scale |
|---|---:|
| RBF | 1e-05 |
| Matérn (ν = 2.5) | 1e-05 |

Both kernels converged to the same lower length-scale bound.

This suggested that the modelling difficulty was not simply due to the smoothness assumptions of the RBF kernel.

### Comparison of Predictions and Uncertainty

The two fitted models were evaluated across 100,000 candidate points to determine whether either could better distinguish between unexplored regions.

| Kernel | Predicted Output Range | Uncertainty Range |
|---|---|---|
| RBF | -0.0003278239 to -0.0000001931 | 0.0000355799 to 0.0010366701 |
| Matérn | -0.0003278239 to -0.0000003215 | 0.0000459032 to 0.0010366701 |

Both models produced very similar prediction and uncertainty ranges.

Although some variation was present, neither kernel demonstrated a clear advantage in identifying promising unexplored regions.

As a result, there was insufficient evidence to justify replacing the existing RBF approach with Matérn for Function 1.

**The decision was therefore made to continue using maximum-distance exploration for Function 1**, allowing further information to be collected before relying on GP-based query selection.

## 4. Generate Candidates and Apply UCB

For Functions 2–8, the updated RBF Gaussian Process models were used to evaluate 100,000 candidate points.

As in Week 1, the candidates were generated within the permitted input range of 0 to 1.

The Upper Confidence Bound acquisition function was retained:

UCB(x) = μ(x) + κ × σ(x)

Where:

- `μ(x)` represents the GP predicted mean.
- `σ(x)` represents the associated predictive uncertainty.
- `κ = 2.5` controls the balance between exploitation and exploration.

The same exploration parameter was retained to maintain consistency with the previous round and evaluate how the updated models responded to the additional observations.

### UCB Results

| Function | Predicted Output | Uncertainty | UCB Score |
|---|---:|---:|---:|
| 2 | 0.441071 | 0.190032 | 0.916151 |
| 3 | -0.055801 | 0.074303 | 0.129955 |
| 4 | -1.666506 | 0.886853 | 0.550627 |
| 5 | 2112.903796 | 303.291367 | 2871.132212 |
| 6 | -0.664546 | 0.355483 | 0.224161 |
| 7 | 1.772966 | 0.240968 | 2.375386 |
| 8 | 9.712587 | 0.462513 | 10.868871 |

The updated models produced different balances between predicted output and uncertainty.

Functions 5 and 7 selected candidates with relatively strong predicted outputs, suggesting a greater emphasis on exploiting regions that had already produced successful observations.

Other functions, including Function 4, received a larger contribution from uncertainty, indicating that exploration remained an important component of their selections.

Function 6 also produced a meaningful UCB selection following the improvement in its fitted Gaussian Process.

Before finalising the queries, the selected points were compared with the current highest-performing observations to better understand their exploration–exploitation behaviour.

## 5. Compare Proposed Queries with Current Best Observations

The Euclidean distance between each proposed query and the current highest-performing observation was calculated.

This provided an additional indication of whether the optimisation was focusing on previously successful areas or exploring more distant regions.

Distance was not treated as an absolute measure of exploration or exploitation but was considered alongside predicted output and uncertainty.

| Function | Proposed Query Point | Distance from Current Best |
|---|---|---:|
| 2 | `[0.655286, 0.929896]` | 0.0475 |
| 3 | `[0.397530, 0.090477, 0.447319]` | 0.3171 |
| 4 | `[0.480756, 0.474000, 0.229059, 0.429178]` | 0.1609 |
| 5 | `[0.514842, 0.856105, 0.961506, 0.998492]` | 0.1658 |
| 6 | `[0.246549, 0.344645, 0.479525, 0.733976, 0.025332]` | 0.5785 |
| 7 | `[0.080334, 0.317465, 0.485517, 0.055569, 0.406883, 0.931976]` | 0.2490 |
| 8 | `[0.078127, 0.668711, 0.054188, 0.209488, 0.863552, 0.717799, 0.074439, 0.866300]` | 0.9190 |

The distance comparison revealed several differences in the models' behaviour.

**Function 2** selected a point extremely close to its current best observation, indicating a strong emphasis on exploitation.

Functions 4, 5 and 7 also remained relatively close to previously promising regions.

In contrast, Functions 6 and 8 selected more distant points, suggesting a greater exploratory component.

This demonstrated that the same UCB parameter of `κ = 2.5` could produce different balances between exploration and exploitation depending on each model's predictions and uncertainty.

There was therefore no clear reason to introduce function-specific UCB parameters at this stage.

## 6. Maximum-Distance Exploration for Function 1

As neither the RBF nor Matérn kernel provided sufficiently informative predictions for Function 1, the maximum-distance exploration strategy from Week 1 was retained.

For each candidate point, the Euclidean distance to its nearest existing observation was calculated.

The candidate with the largest minimum distance was selected, targeting a comparatively unexplored region of the function's input space.

### Exploration Result

| Metric | Value |
|---|---|
| Selection method | Maximum-distance exploration |
| Selected point | `[0.999973, 0.999843]` |
| Distance to nearest observation | 0.345641 |
| Within permitted bounds | Yes |
| Duplicate existing observation | No |

The selected point was located near the upper-right boundary of Function 1's input space.

Its nearest existing observation was approximately 0.346 units away, confirming that it targeted a relatively poorly sampled region.

The intention was to gather additional information that could potentially improve the Gaussian Process in subsequent rounds, rather than attempt to exploit unreliable predictions.

## 7. Final Week 2 Query Selections

Before finalising the second-round queries, the selected points were checked to ensure that:

- Every input coordinate remained within the permitted `[0, 1]` range.
- No proposed query duplicated an existing observation.
- Function 1's point corresponded to the maximum-distance exploration selection.
- Functions 2–8 used their highest-UCB candidate selections.

### Final Proposed Queries

| Function | Selection Method | Final Query Point |
|---|---|---|
| 1 | Maximum-distance exploration | `[0.999973, 0.999843]` |
| 2 | GP + UCB | `[0.655286, 0.929896]` |
| 3 | GP + UCB | `[0.397530, 0.090477, 0.447319]` |
| 4 | GP + UCB | `[0.480756, 0.474000, 0.229059, 0.429178]` |
| 5 | GP + UCB | `[0.514842, 0.856105, 0.961506, 0.998492]` |
| 6 | GP + UCB | `[0.246549, 0.344645, 0.479525, 0.733976, 0.025332]` |
| 7 | GP + UCB | `[0.080334, 0.317465, 0.485517, 0.055569, 0.406883, 0.931976]` |
| 8 | GP + UCB | `[0.078127, 0.668711, 0.054188, 0.209488, 0.863552, 0.717799, 0.074439, 0.866300]` |

The final validation confirmed that all eight queries were within the permitted input range and did not duplicate previously observed points.

The selection process also demonstrated that the updated GP models continued to identify different balances between exploitation and exploration across the functions.

## 8. Week 2 Reflection

The second round of Bayesian Optimisation provided useful evidence about how additional observations can affect Gaussian Process performance.

Five of the eight Week 1 queries had improved the highest observed outputs, supporting the initial use of GP and UCB for functions where the fitted models were informative.

One of the most significant developments was the improvement in Function 6's RBF length scale. Although its previous exploration-based query had not improved its maximum output, the additional observation appeared to improve the fitted model sufficiently for it to return to UCB-based selection.

Function 1 continued to present difficulties. Both the RBF and Matérn kernels reached the lower length-scale bound and produced similar predictions, so maximum-distance exploration remained the preferred approach.

The decision to retain `κ = 2.5` was also supported by the different behaviours observed across functions. Some selections remained close to existing high-performing observations, while others moved into less explored areas.

Overall, Week 2 reinforced the importance of assessing the reliability of the surrogate models rather than relying automatically on their acquisition scores.

In the following round, the newly returned outputs will be incorporated into the datasets, the Gaussian Processes will be refitted, and additional techniques may be investigated to better understand the structure of the unknown functions.

---

**Associated code:** `Week_02.ipynb` contains the original Week 2 analysis and saved outputs.
