
# Module 13 — Week 2: Gaussian Process Optimisation

## Methodology

### Step 1 — Incorporate New Observations and Assess Week 1 Performance

I began by adding the eight new observations returned from the Week 1 queries to their respective datasets. I then compared each new output with the previous highest observed value to assess the effectiveness of the initial optimisation strategy.

The results were:

| Function | Previous Best | New Output | Improvement | Result |
|:--|--:|--:|--:|:--|
| **1** | ~0.000000 | 0.000000 | ~0.000000 | No improvement |
| **2** | 0.611205 | -0.082041 | -0.693246 | No improvement |
| **3** | -0.034835 | -0.020167 | +0.014668 | New best |
| **4** | -4.025542 | -0.300645 | +3.724898 | New best |
| **5** | 1088.859618 | 2329.015854 | +1240.156236 | New best |
| **6** | -0.714265 | -2.661556 | -1.947291 | No improvement |
| **7** | 1.364968 | 1.950612 | +0.585644 | New best |
| **8** | 9.598482 | 9.821986 | +0.223504 | New best |

**Five of the eight queries produced new best observations**, with Functions 4 and 5 showing particularly substantial improvements.

Function 4 improved from approximately -4.03 to -0.30, while Function 5 more than doubled its previous maximum, increasing from approximately 1,089 to 2,329.

Functions 3, 7 and 8 also improved on their previous maxima, suggesting that UCB had successfully identified promising areas for several functions.

Functions 1, 2 and 6 did not improve. However, Functions 1 and 6 had deliberately used distance-based exploration in Week 1 because their Gaussian Process models were unable to provide meaningful guidance through UCB.

Although these queries did not produce higher outputs, they provided additional observations from previously unexplored regions. These could still be valuable for improving the models in subsequent iterations.

### Step 2 — Refit the RBF Gaussian Processes

After incorporating the new observations, I refitted the **Gaussian Process (GP)** models using the same **Radial Basis Function (RBF)** kernel as Week 1.

The aim was to determine whether the additional information had changed how the models represented each function, particularly for Functions 1 and 6, which previously had extremely small fitted length scales.

The updated fitted length scales were:

| Function | Updated RBF Length Scale |
|:--|--:|
| **1** | 1e-05 |
| **2** | 0.0453 |
| **3** | 0.202 |
| **4** | 0.522 |
| **5** | 0.225 |
| **6** | 0.481 |
| **7** | 0.354 |
| **8** | 0.913 |

**Key Observations**

Most functions produced reasonable fitted length scales, with the models continuing to identify relationships between observations across different regions of the search space.

The most significant change was observed in **Function 6**.

In Week 1, its RBF length scale had reached the lower bound of 1e-05, causing the model to produce almost identical predictions across unobserved candidate points.

After incorporating the exploratory observation, the fitted length scale increased substantially to **0.481**.

This suggested that the additional observation had helped the model identify a more useful relationship between inputs and outputs.

Although the Week 1 query had not improved Function 6's maximum output, it appeared to have improved the GP's ability to model the function.

As a result, I decided that Function 6 could return to the standard UCB acquisition strategy.

**Function 1**, however, continued to reach the lower bound of 1e-05 and produced another convergence warning.

This suggested that the additional exploratory observation had not resolved the original modelling issue.

I therefore decided to investigate whether an alternative kernel could provide a more informative model before selecting Function 1's next query.

### Step 3 — Compare RBF and Matérn Kernels for Function 1

As Function 1 continued to produce an extremely small RBF length scale, I tested a **Matérn kernel** as an alternative.

The RBF kernel assumes a particularly smooth underlying function, whereas the Matérn kernel allows greater flexibility in the smoothness of the relationships being modelled.

I used a Matérn kernel with **ν = 2.5** and compared its fitted behaviour with the existing RBF model.

The optimised kernels were:

| Kernel | Fitted Length Scale |
|:--|--:|
| **RBF** | 1e-05 |
| **Matérn (ν = 2.5)** | 1e-05 |

Both kernels reached the same lower bound, suggesting that changing the smoothness assumptions alone had not resolved the modelling issue.

However, I wanted to assess whether the two models produced different predictions despite their similarly small length scales.

I therefore generated **100,000 candidate points** for Function 1 and compared the predicted mean outputs and uncertainty from each model.

| Kernel | Predicted Output Range | Uncertainty Range |
|:--|:--|:--|
| **RBF** | -0.0003278239 to -0.0000001931 | 0.0000355799 to 0.0010366701 |
| **Matérn** | -0.0003278239 to -0.0000003215 | 0.0000459032 to 0.0010366701 |

**Assessment**

The two kernels produced very similar prediction and uncertainty ranges across the candidate points.

Although some variation was present, neither kernel provided a clear improvement in the ability to distinguish between unexplored regions.

This suggested that the difficulty was not simply caused by the RBF kernel's smoothness assumptions. Instead, the existing observations may not yet contain enough information for either model to represent the function effectively.

As neither kernel demonstrated a clear advantage, I decided to retain an **exploration-based approach for Function 1**, rather than rely on UCB to select a point using predictions that remained difficult to interpret.

I would continue to gather observations from less explored areas and reassess the model as more information became available.

### Step 4 — Generate Candidate Points and Apply UCB

For Functions **2–8**, I generated another **100,000 random candidate points** within the permitted input range of 0 to 1.

Using the updated RBF Gaussian Processes, I calculated the predicted mean output and uncertainty for each candidate.

I then applied the **Upper Confidence Bound (UCB)** acquisition function:

**UCB(x) = μ(x) + κσ(x)**

Where:

- **μ(x):** Predicted mean output.
- **σ(x):** Predictive standard deviation.
- **κ:** Exploration parameter.

I retained **κ = 2.5**, consistent with Week 1.

This allowed me to assess the impact of the new observations and updated models without introducing additional changes to the acquisition strategy.

For each function, I selected the candidate with the highest UCB score.

| Function | Predicted Output | Uncertainty | Highest UCB Score |
|:--|--:|--:|--:|
| **2** | 0.441071 | 0.190032 | 0.916151 |
| **3** | -0.055801 | 0.074303 | 0.129955 |
| **4** | -1.666506 | 0.886853 | 0.550627 |
| **5** | 2112.903796 | 303.291367 | 2871.132212 |
| **6** | -0.664546 | 0.355483 | 0.224161 |
| **7** | 1.772966 | 0.240968 | 2.375386 |
| **8** | 9.712587 | 0.462513 | 10.868871 |

**Assessment**

The updated models produced different balances of predicted performance and uncertainty across the functions.

Functions **5 and 7** selected candidates with particularly strong predicted outputs following their successful first-round queries. This suggested that the models were continuing to identify promising regions around areas that had already performed well.

For **Function 4**, uncertainty made a relatively large contribution to the selected candidate's UCB score, indicating a greater exploratory element.

Importantly, **Function 6** now produced a meaningful UCB selection following the improvement in its fitted RBF length scale.

This represented a change from Week 1, when the model's almost constant predictions had made UCB ineffective.

Before finalising these queries, I wanted to examine how far the selected points were from the current best observations to better understand the balance between exploration and exploitation.

### Step 5 — Assess the Proposed Queries Against Current Best Observations

I compared each UCB-selected point with the input coordinates associated with the highest observed output for its function.

Using **Euclidean distance**, I calculated how far each proposed query was from the current best-known point.

This provided an additional indication of whether the model was exploiting an established high-performing region or exploring a different part of the search space.

Distance was not treated as a definitive measure of exploration or exploitation, but considered alongside the predicted outputs and uncertainty estimates.

| Function | Current Best Output | Distance From Current Best |
|:--|--:|--:|
| **2** | 0.611205 | 0.0475 |
| **3** | -0.020167 | 0.3171 |
| **4** | -0.300645 | 0.1609 |
| **5** | 2329.015854 | 0.1658 |
| **6** | -0.714265 | 0.5785 |
| **7** | 1.950612 | 0.2490 |
| **8** | 9.821986 | 0.9190 |

**Interpretation**

**Function 2** selected a point extremely close to its current best observation, with a distance of approximately **0.048**.

This suggested that the updated UCB model was focusing on an established promising region, despite the previous query failing to improve the maximum output.

**Functions 4, 5 and 7** also selected points relatively close to their current best observations, indicating a continued exploitation component following their successful first-round results.

In contrast, **Functions 6 and 8** selected more distant points.

Function 6's proposed query was approximately **0.579** from its current best observation, while Function 8's was approximately **0.919** away.

These larger distances suggested a greater exploratory element, although the selections were still based on UCB rather than deliberate maximum-distance exploration.

The comparison demonstrated that the same κ value could produce different behaviour depending on each model's predictions and uncertainty.

I therefore decided to retain **κ = 2.5** for this round rather than introduce function-specific exploration parameters without a clear reason to do so.

### Step 6 — Select an Exploration Point for Function 1

As neither the RBF nor Matérn model provided sufficiently informative predictions for Function 1, I continued using the **distance-based exploration strategy** introduced in Week 1.

For each of the 100,000 candidate points, I calculated its Euclidean distance to every existing observation.

I then identified the distance to the nearest observation and selected the candidate with the greatest minimum distance.

This approach prioritised a relatively unexplored region rather than relying on the GP's predicted output.

**Selected Function 1 Query**

| Measure | Result |
|:--|:--|
| **Selected Point** | `[0.99997255, 0.99984253]` |
| **Distance to Nearest Observation** | 0.345641 |

The selected point was located near the upper-right boundary of Function 1's two-dimensional search space.

Its distance of approximately **0.346** from the nearest existing observation indicated that it represented a relatively poorly sampled region among the candidates tested.

As the Gaussian Process models continued to provide limited guidance, I retained this point as an information-gathering query.

The intention was to use the resulting observation to improve the dataset and reassess the model in subsequent rounds.

### Step 7 — Final Validation and Query Selection

Before finalising the second-round queries, I checked that all selected points were valid.

The checks confirmed that:

- Every input coordinate was within the permitted range of **0 to 1**.
- None of the proposed queries duplicated an existing observation.
- Function 1's selected point was the maximum-distance candidate identified from its candidate set.
- Functions 2–8 used the highest-UCB candidates selected from their updated RBF models.

The final proposed query points were:

| Function | Selection Method | Proposed Query Point |
|:--|:--|:--|
| **1** | Distance-based exploration | `[0.999973, 0.999843]` |
| **2** | UCB | `[0.655286, 0.929896]` |
| **3** | UCB | `[0.397530, 0.090477, 0.447319]` |
| **4** | UCB | `[0.480756, 0.474000, 0.229059, 0.429178]` |
| **5** | UCB | `[0.514842, 0.856105, 0.961506, 0.998492]` |
| **6** | UCB | `[0.246549, 0.344645, 0.479525, 0.733976, 0.025332]` |
| **7** | UCB | `[0.080334, 0.317465, 0.485517, 0.055569, 0.406883, 0.931976]` |
| **8** | UCB | `[0.078127, 0.668711, 0.054188, 0.209488, 0.863552, 0.717799, 0.074439, 0.866300]` |

The final validation confirmed that all eight proposed queries were within bounds and did not duplicate any existing observations.

The updated models also demonstrated different exploration–exploitation behaviour across the functions, supporting the decision to retain the existing UCB exploration parameter.

The most important methodological development was the improvement in **Function 6's Gaussian Process**, allowing it to return to UCB after using distance-based exploration in Week 1.

Function 1 remained the exception, requiring further exploration because neither RBF nor Matérn had yet produced sufficiently informative predictions.

## Week 2 Process Summary

The Week 2 optimisation process consisted of the following stages:

1. **Evaluate Previous Queries:** Incorporated the eight returned observations and identified five new best outputs, including substantial improvements for Functions 4 and 5.
2. **Refit Gaussian Processes:** Updated the RBF models and identified a significant improvement in Function 6's fitted length scale.
3. **Investigate Alternative Kernels:** Compared RBF and Matérn for Function 1, finding that neither resolved the modelling difficulties.
4. **Apply UCB:** Used 100,000 candidate points and UCB with κ = 2.5 to select queries for Functions 2–8.
5. **Assess Exploration and Exploitation:** Compared proposed queries with current best observations to evaluate their relative positions in the search space.
6. **Continue Targeted Exploration:** Selected a maximum-distance query for Function 1 to gather additional information.
7. **Validate Final Queries:** Confirmed that all proposed points were within bounds and did not duplicate existing observations.

