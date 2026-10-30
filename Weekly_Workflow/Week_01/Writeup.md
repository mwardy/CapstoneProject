
# Week 1 — Initial Bayesian Optimisation

## Objective

Week 1 establishes the initial modelling approach using the provided observations, Gaussian Process (GP) regression, and an Upper Confidence Bound (UCB) acquisition function. The analysis also investigates cases where the fitted GP provides little useful guidance and adopts an exploration-based alternative where appropriate.

## 1. Initial Data Exploration

The input and output arrays were loaded for each function. The number of observations, input dimensions, and highest and lowest observed outputs were examined to understand the starting data.

| Function | Initial Observations | Input Dimensions | Highest Observed Output | Lowest Observed Output |
|---|---:|---:|---:|---:|
| 1 | 10 | 2 | 7.710875e-16 | -0.003606 |
| 2 | 10 | 2 | 0.611205 | -0.065624 |
| 3 | 15 | 3 | -0.034835 | -0.398926 |
| 4 | 30 | 4 | -4.025542 | -32.625660 |
| 5 | 20 | 4 | 1088.859618 | 0.112940 |
| 6 | 20 | 5 | -0.714265 | -2.571170 |
| 7 | 30 | 6 | 1.364968 | 0.002701 |
| 8 | 40 | 8 | 9.598482 | 5.592193 |

The starting datasets differ both in size and in the scales of their outputs. Each function is therefore modelled independently rather than combining observations across functions.

## 2. Fit Gaussian Process Models

An RBF-kernel Gaussian Process was fitted independently for each function, using the input coordinates to predict the associated output. The notebook uses `GaussianProcessRegressor` with output normalisation, ten optimiser restarts, and a fixed model random state of `42`.

The initial fitted RBF length scales were:

| Function | Initial Fitted RBF Length Scale |
|---|---:|
| 1 | 1e-05 |
| 2 | 0.125 |
| 3 | 0.188 |
| 4 | 0.500 |
| 5 | 1e-05 |
| 6 | 1e-05 |
| 7 | 0.281 |
| 8 | 0.892 |

Functions 1, 5, and 6 reached the default lower length-scale bound, prompting further investigation. The lower bound was relaxed from `1e-05` to `1e-10` in diagnostic fits:

- **Function 1:** The fitted length scale reached `1e-10`.
- **Function 5:** The fitted length scale settled at approximately `0.269`.
- **Function 6:** The fitted length scale reached `1e-10`.

This suggested that Function 5 could be represented more effectively after widening the search bounds, whereas Functions 1 and 6 continued to favour extremely short length scales. The notebook consequently replaces the Function 5 GP with the wider-bound fit and retains the original fitted GP models for Functions 1 and 6.

## 3. Generate Candidate Points and Predict Outputs

Using NumPy's random generator with seed `42`, the notebook generates **100,000 candidate points per function**, with every input coordinate sampled between 0 and 1. Each fitted GP predicts the mean output and associated standard deviation for the candidate points.

| Function | Predicted Mean Range | Predicted Uncertainty Range |
|---|---|---|
| 1 | -0.000361 to -0.000361 | 0.001082 to 0.001082 |
| 2 | -0.077317 to 0.643977 | 0.000068 to 0.225365 |
| 3 | -0.397833 to -0.018176 | 0.002777 to 0.084213 |
| 4 | -35.682037 to -0.922445 | 0.067992 to 5.263611 |
| 5 | -71.908033 to 1113.402974 | 12.509737 to 245.522395 |
| 6 | -1.495390 to -1.495390 | 0.449000 to 0.449000 |
| 7 | -0.020560 to 1.208211 | 0.061869 to 0.302129 |
| 8 | 5.286512 to 10.217038 | 0.064678 to 0.681040 |

**Functions 1 and 6** showed effectively constant predicted means and uncertainties at the displayed precision across the candidate set. This is consistent with their extremely short fitted length scales: most candidate points are treated as having little correlation with the existing observations. As a result, their UCB scores provide little meaningful discrimination between unexplored candidates.

The other six models showed variation in their predictions and uncertainty, allowing UCB to guide query selection.

## 4. Evaluate the Upper Confidence Bound Acquisition Function

For each candidate, the notebook computes:

UCB(x) = μ(x) + κ × σ(x)

Where:

- `μ(x)` is the GP predicted mean.
- `σ(x)` is the predicted standard deviation.
- `κ = 2.5` controls the contribution from uncertainty.

Higher UCB scores can therefore reflect either strong predicted performance, greater uncertainty, or both.

The highest-UCB candidate was identified for each function. However, the UCB results for Functions 1 and 6 were **diagnostic only** and were not used as their final selections because their models provided nearly flat acquisition scores. For Functions 2, 3, 4, 5, 7, and 8, the highest-UCB candidates were retained.

| Function | Highest-UCB Predicted Output | Uncertainty | UCB Score |
|---|---:|---:|---:|
| 2 | 0.537302 | 0.174852 | 0.974432 |
| 3 | -0.041871 | 0.071974 | 0.138065 |
| 4 | -1.109357 | 0.992949 | 1.373015 |
| 5 | 1003.710099 | 139.196890 | 1351.702325 |
| 7 | 1.136197 | 0.193256 | 1.619336 |
| 8 | 10.015346 | 0.507951 | 11.285223 |

## 5. Use Maximum-Distance Exploration for Functions 1 and 6

Because UCB could not meaningfully distinguish between candidates for Functions 1 and 6, the next query for each was selected using **maximum-distance exploration**.

For each of the 100,000 candidate points, the Euclidean distance to the nearest existing observation was computed. The candidate with the largest such distance was selected to sample a relatively unexplored part of the input space.

| Function | Selected Exploration Point | Distance to Nearest Observation |
|---|---|---:|
| 1 | `[0.001243, 0.996919]` | 0.394922 |
| 6 | `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]` | 0.886056 |

This prioritised information gathering rather than immediate exploitation for the two functions. Once their returned outputs become available in the next round, the GP models can be refitted and reassessed.

## 6. Final Week 1 Query Selections

The final proposed points combine the maximum-distance selections for Functions 1 and 6 with the highest-UCB selections for the other functions.

The distance shown below is the Euclidean distance **from the current highest-performing observed point**, not the distance to the nearest existing observation used in the exploration step.

| Function | Method | Final Proposed Point (Six Decimal Places) | Distance from Current Best |
|---|---|---|---:|
| 1 | Maximum-distance exploration | `[0.001243, 0.996919]` | 0.7760 |
| 2 | GP + UCB | `[0.821778, 0.980301]` | 0.1307 |
| 3 | GP + UCB | `[0.409499, 0.399000, 0.519487]` | 0.2903 |
| 4 | GP + UCB | `[0.399157, 0.433203, 0.357699, 0.461233]` | 0.2857 |
| 5 | GP + UCB | `[0.352268, 0.854957, 0.929210, 0.999541]` | 0.1833 |
| 6 | Maximum-distance exploration | `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]` | 1.1281 |
| 7 | GP + UCB | `[0.120627, 0.403921, 0.379973, 0.146533, 0.335853, 0.763305]` | 0.2062 |
| 8 | GP + UCB | `[0.243397, 0.036115, 0.078358, 0.053721, 0.783854, 0.278393, 0.034947, 0.429032]` | 0.9368 |

The distances indicate different exploration–exploitation behaviour. Functions 2, 3, 4, 5, and 7 generally selected points relatively close to their current best observations, while Function 8's UCB selection was further away.

The more distant selections for Functions 1 and 6 were deliberate consequences of the alternative exploration strategy.

## 7. Week 1 Reflection

The initial round established a baseline Bayesian Optimisation workflow for all eight functions. RBF Gaussian Processes could be fitted to the existing observations, and UCB with `κ = 2.5` provided a way to balance predicted performance with uncertainty where the models could distinguish between candidate points.

The main modelling difficulty occurred in Functions 1 and 6, whose extremely short RBF length scales led to nearly constant predictions across the sampled candidate space. Rather than relying on an uninformative acquisition function, the queries for those functions were selected to explore areas distant from existing observations.

The Function 5 length-scale warning was also investigated, and a more informative fit was obtained by widening its kernel bounds.

The following round will incorporate the returned outputs, assess whether the previous choices improved the observed maxima, and refit the models using the expanded datasets. Alternative kernels may be considered if the problems in Functions 1 and 6 persist.

---

**Associated code:** `Week_01.ipynb` contains the original Week 1 analysis and saved outputs.


