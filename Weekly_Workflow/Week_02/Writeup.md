
# Module 13 — Week 2: Gaussian process optimisation

## Weekly objective and step-by-step plan

The objective of Week 2 was to incorporate the results from the first round of queries, assess the performance of the initial optimisation strategy and select a second query point for each of the eight functions.

A particular focus was whether the additional observations had improved the Gaussian process (GP) models for Functions 1 and 6, which had required maximum-distance exploration in Week 1 because their predictions were largely uninformative.

The process followed seven steps:

1. Incorporate the new observations and compare them with the previous best outputs.
2. Refit the RBF Gaussian processes and assess changes in the fitted length scales.
3. Compare RBF and Matérn kernels for Function 1.
4. Generate candidate points and apply upper confidence bound (UCB) where the models were informative.
5. Assess the proposed queries in terms of exploration and exploitation.
6. Select an exploration-based query for Function 1.
7. Validate the final proposed queries.

## Methodology

The eight new outputs from Week 1 were added to their respective datasets, giving each function an additional observation.

I then refitted the Gaussian process models using RBF kernels, normalised outputs, ten optimiser restarts and a random state of 42, consistent with the initial approach.

For Functions 2–8, I generated 100,000 candidate points per function within the permitted input space of [0, 1] and applied GP + UCB:

**UCB(x) = μ(x) + κσ(x)**

Where μ(x) represents the predicted output, σ(x) represents predictive uncertainty and κ controls the relative weighting of exploration.

I retained κ = 2.5 to maintain consistency with Week 1 and assess how the additional observations affected query selection without changing the acquisition parameter.

For Function 1, I investigated whether a Matérn kernel with ν = 2.5 could improve on the RBF model. As neither model provided sufficiently informative predictions, I continued using maximum-distance exploration.

## Results and interpretation by step

### Step 1 — Incorporate new observations

I compared the eight returned outputs with the previous highest observations to assess the effectiveness of the first-round queries.

| Function | Previous best | New output | Improvement | Result |
|---|---:|---:|---:|---|
| 1 | ~0.000000 | 0.000000 | ~0.000000 | No improvement |
| 2 | 0.611205 | -0.082041 | -0.693246 | No improvement |
| 3 | -0.034835 | -0.020167 | +0.014668 | New best |
| 4 | -4.025542 | -0.300645 | +3.724898 | New best |
| 5 | 1088.859618 | 2329.015854 | +1240.156236 | New best |
| 6 | -0.714265 | -2.661556 | -1.947291 | No improvement |
| 7 | 1.364968 | 1.950612 | +0.585644 | New best |
| 8 | 9.598482 | 9.821986 | +0.223504 | New best |

Five of the eight queries produced new best observations, with Functions 4 and 5 showing particularly large improvements. Function 4 improved from approximately -4.03 to -0.30, while Function 5 more than doubled its previous maximum.

Functions 3, 7 and 8 also improved, suggesting that GP + UCB had identified promising areas for several functions.

Functions 1, 2 and 6 did not improve. However, Functions 1 and 6 had deliberately used maximum-distance exploration in Week 1. Although their new outputs were not higher, these observations provided information from previously underrepresented regions of the search space.

### Step 2 — Refit the RBF Gaussian processes

After incorporating the new observations, I refitted the RBF models to assess whether their fitted behaviour had changed.

| Function | Updated RBF length scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.0453 |
| 3 | 0.202 |
| 4 | 0.522 |
| 5 | 0.225 |
| 6 | 0.481 |
| 7 | 0.354 |
| 8 | 0.913 |

The most significant change was in Function 6. Its fitted length scale increased from the lower bound of 1e-05 in Week 1 to approximately 0.481.

This suggested that the additional observation had helped the GP identify relationships between inputs and outputs across a wider region of the search space.

Although the exploratory query had not improved Function 6's maximum output, it appeared to have improved the model's ability to distinguish between candidate locations.

I therefore decided that Function 6 could return to GP + UCB for the second round.

Function 1, however, continued to reach the lower length-scale bound of 1e-05 and produced another convergence warning.

This indicated that the additional exploratory observation had not resolved the original modelling issue, so I decided to investigate an alternative kernel before selecting its next query.

### Step 3 — Compare RBF and Matérn kernels for Function 1

I tested a Matérn kernel with ν = 2.5 to determine whether a different smoothness assumption could improve the GP model for Function 1.

The RBF kernel assumes a very smooth underlying function, whereas the Matérn kernel allows greater flexibility in the smoothness of the relationships being modelled.

Both kernels converged to the same lower length-scale bound.

| Kernel | Fitted length scale |
|---|---:|
| RBF | 1e-05 |
| Matérn (ν = 2.5) | 1e-05 |

I then compared their predicted outputs and uncertainty across 100,000 candidate points.

| Kernel | Predicted output range | Uncertainty range |
|---|---|---|
| RBF | -0.0003278239 to -0.0000001931 | 0.0000355799 to 0.0010366701 |
| Matérn | -0.0003278239 to -0.0000003215 | 0.0000459032 to 0.0010366701 |

The two models produced similar prediction and uncertainty ranges. Although there was some variation, neither kernel demonstrated a clear improvement in its ability to distinguish between unexplored regions.

This suggested that the modelling difficulty was not simply caused by the RBF kernel's smoothness assumptions. The available observations may not yet contain enough information for either model to represent the function effectively.

I therefore decided to continue using an exploration-focused strategy for Function 1 rather than relying on GP + UCB.

### Step 4 — Generate candidate points and apply UCB

For Functions 2–8, I generated 100,000 candidate points per function and calculated the predicted output, uncertainty and UCB score using the updated RBF models.

I retained κ = 2.5, allowing the acquisition function to balance exploitation of promising regions with exploration of uncertain areas.

The highest-scoring UCB candidates produced the following results:

| Function | Predicted output | Uncertainty | UCB score |
|---|---:|---:|---:|
| 2 | 0.441071 | 0.190032 | 0.916151 |
| 3 | -0.055801 | 0.074303 | 0.129955 |
| 4 | -1.666506 | 0.886853 | 0.550627 |
| 5 | 2112.903796 | 303.291367 | 2871.132212 |
| 6 | -0.664546 | 0.355483 | 0.224161 |
| 7 | 1.772966 | 0.240968 | 2.375386 |
| 8 | 9.712587 | 0.462513 | 10.868871 |

Functions 5 and 7 selected candidates with relatively strong predicted outputs following their successful first-round queries, suggesting that the models were continuing to identify promising regions.

For Function 4, uncertainty contributed substantially to the selected candidate's UCB score, indicating a stronger exploratory element.

Function 6 also produced a meaningful UCB selection following the improvement in its fitted RBF length scale. This was a change from Week 1, when its nearly constant predictions had made GP + UCB ineffective.

### Step 5 — Assess exploration and exploitation

I compared each UCB-selected query with the coordinates of its highest observed output, using Euclidean distance.

This provided an indication of whether the selected points were concentrated around previously successful regions or explored more distant areas.

| Function | Current best output | Distance from current best |
|---|---:|---:|
| 2 | 0.611205 | 0.0475 |
| 3 | -0.020167 | 0.3171 |
| 4 | -0.300645 | 0.1609 |
| 5 | 2329.015854 | 0.1658 |
| 6 | -0.714265 | 0.5785 |
| 7 | 1.950612 | 0.2490 |
| 8 | 9.821986 | 0.9190 |

Function 2 selected a point particularly close to its current best observation, with a distance of approximately 0.048. Functions 4, 5 and 7 also remained relatively close to their best-known points, suggesting a greater exploitation component.

Functions 6 and 8 selected more distant points, indicating greater exploration relative to their current best observations.

However, distance alone does not determine whether a query is exploratory or exploitative. GP + UCB selects points based on predicted performance and uncertainty, so a more distant query may still have a strong predicted output.

The results showed that the same κ value could produce different exploration–exploitation behaviour depending on the fitted model.

I therefore retained κ = 2.5 rather than introduce function-specific values without a clear reason.

### Step 6 — Select an exploration point for Function 1

As neither RBF nor Matérn provided sufficiently informative predictions for Function 1, I continued with maximum-distance exploration.

For each of the 100,000 candidate points, I calculated its Euclidean distance to the nearest existing observation and selected the candidate with the greatest minimum distance.

| Measure | Result |
|---|---|
| Selected point | `[0.99997255, 0.99984253]` |
| Distance to nearest observation | 0.345641 |

The selected point was near the upper-right boundary of the two-dimensional search space and was approximately 0.346 from its nearest existing observation.

This approach deliberately prioritised exploration rather than exploitation because the GP models could not provide reliable guidance about which locations were likely to produce higher outputs.

Although this did not guarantee an immediate improvement, the new observation could help improve the model in subsequent rounds.

### Step 7 — Validate the final proposed queries

Before finalising the second-round queries, I checked that all selected coordinates were within the permitted input range of [0, 1] and that none duplicated an existing observation.

Function 1's query was also confirmed as the maximum-distance candidate identified from its candidate set, while Functions 2–8 were selected using GP + UCB.

The final proposed queries were:

| Function | Selection method | Proposed query coordinates |
|---|---|---|
| 1 | Max-distance exploration | `[0.999973, 0.999843]` |
| 2 | GP + UCB | `[0.655286, 0.929896]` |
| 3 | GP + UCB | `[0.397530, 0.090477, 0.447319]` |
| 4 | GP + UCB | `[0.480756, 0.474000, 0.229059, 0.429178]` |
| 5 | GP + UCB | `[0.514842, 0.856105, 0.961506, 0.998492]` |
| 6 | GP + UCB | `[0.246549, 0.344645, 0.479525, 0.733976, 0.025332]` |
| 7 | GP + UCB | `[0.080334, 0.317465, 0.485517, 0.055569, 0.406883, 0.931976]` |
| 8 | GP + UCB | `[0.078127, 0.668711, 0.054188, 0.209488, 0.863552, 0.717799, 0.074439, 0.866300]` |

All eight proposed queries passed the validation checks.

## Weekly summary and next steps

Week 2 incorporated eight new observations, five of which improved the previous best outputs. Functions 4 and 5 showed particularly strong improvements, while Functions 3, 7 and 8 also achieved new maxima.

The main methodological development was the change in Function 6's GP model. Its fitted RBF length scale increased substantially, allowing it to return to GP + UCB after using maximum-distance exploration in Week 1.

Function 1 continued to present difficulties. Testing a Matérn kernel did not provide a clear improvement over RBF, so maximum-distance exploration remained the preferred approach.

The results also showed that retaining κ = 2.5 could produce different balances of exploration and exploitation across the functions without requiring individual parameter adjustments.

The next steps were to:

- Submit the eight proposed query points and obtain their outputs.
- Incorporate the returned observations into the datasets.
- Refit the GP models and assess any further changes in their behaviour.
- Continue investigating alternative modelling approaches if Function 1 remained difficult to model.

Overall, Week 2 demonstrated the value of adapting the optimisation strategy to each function while maintaining a consistent GP + UCB approach wherever the models provided useful predictions.
