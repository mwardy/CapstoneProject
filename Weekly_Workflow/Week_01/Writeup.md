
# Module 12 — Week 1: Gaussian Process Optimisation

## Weekly Objective and Step-by-Step Plan

The objective of Week 1 was to establish an initial optimisation approach for eight unknown black-box functions and select one new query point for each function.

The process followed six steps:

1. Examine the initial observations and identify the highest and lowest outputs.
2. Fit RBF Gaussian Process (GP) models and investigate convergence warnings.
3. Generate 100,000 candidate points per function and assess predicted outputs and uncertainty.
4. Use Upper Confidence Bound (UCB) to identify promising candidates.
5. Apply distance-based exploration where GP predictions were uninformative.
6. Assess and record the final proposed query points.

## Methodology

Each function was modelled independently using Gaussian Process Regression with an RBF kernel, normalised outputs, ten optimiser restarts and a random state of 42.

For each function, 100,000 candidate points were generated uniformly within the permitted input space of [0, 1] using a fixed random seed.

Candidate points were ranked using the Upper Confidence Bound acquisition function:

**UCB(x) = μ(x) + κσ(x)**

Where:
- **μ(x)** is the GP's predicted mean output.
- **σ(x)** is the predictive standard deviation.
- **κ** controls the balance between exploration and exploitation.

I selected **κ = 2.5**, allowing UCB to consider both predicted performance and uncertainty.

Where GP predictions were effectively constant across the candidate space, I used maximum-distance exploration instead.

## Results and Interpretation by Step

### Step 1 — Examine Initial Observations

I examined the initial datasets and identified the highest and lowest observed outputs for each function.

| Function | Highest Observed Output | Lowest Observed Output |
|---|---:|---:|
| 1 | 7.710875e-16 | -0.003606 |
| 2 | 0.611205 | -0.065624 |
| 3 | -0.034835 | -0.398926 |
| 4 | -4.025542 | -32.625660 |
| 5 | 1088.859618 | 0.112940 |
| 6 | -0.714265 | -2.571170 |
| 7 | 1.364968 | 0.002701 |
| 8 | 9.598482 | 5.592193 |

The observations showed considerable variation in output scale. Function 1 had a maximum extremely close to zero, whereas Function 5 exceeded 1,000.

This reinforced the importance of modelling each function independently.

### Step 2 — Fit RBF Gaussian Processes

The initial GP models produced RBF length scales at the lower bound of **1e-5** for Functions 1, 5 and 6.

I investigated whether widening the permitted length-scale range would allow the optimiser to identify more suitable values.

| Function | Length Scale After Relaxing Bounds | Interpretation |
|---|---:|---|
| 1 | 1e-10 | Continued towards the minimum permitted value |
| 5 | 0.269 | Settled at a more moderate length scale |
| 6 | 1e-10 | Continued towards the minimum permitted value |

Function 5 converged to a length scale of 0.269 after widening the bounds. I therefore **refitted Function 5 using the wider bounds and replaced its original GP model**.

Functions 1 and 6 continued to favour extremely small length scales. This suggested that their models were treating nearby observations as becoming unrelated over very short distances.

I retained their original GP models but investigated whether the resulting predictions were sufficiently informative for candidate selection.

### Step 3 — Generate Candidates and Assess Predictions

I generated 100,000 random candidate points for each function and used the fitted GPs to calculate predicted means and uncertainty.

Functions 1 and 6 produced effectively constant predictions across the candidate search space.

| Function | Approximate Predicted Mean | Approximate Uncertainty |
|---|---:|---:|
| 1 | -0.000361 | 0.001082 |
| 6 | -1.495390 | 0.449000 |

Their extremely small RBF length scales meant that the models treated almost all unobserved candidates as unrelated to the existing observations.

Consequently, neither predicted performance nor uncertainty provided a meaningful basis for distinguishing between candidate points.

The remaining six functions showed variation in their predictions and uncertainty, allowing the planned acquisition strategy to proceed.

### Step 4 — Apply Upper Confidence Bound

I calculated UCB scores for all 100,000 candidate points per function using **κ = 2.5**.

For Functions **2, 3, 4, 5, 7 and 8**, the highest-scoring candidates were retained as the proposed queries.

For Functions **1 and 6**, the nearly constant predictions resulted in effectively flat UCB scores. Although the calculation still returned a highest-scoring candidate, this selection was not meaningfully supported by the GP.

I therefore decided not to use the UCB-selected points for these two functions.

### Step 5 — Apply Maximum-Distance Exploration

For Functions 1 and 6, I adopted **Max-distance exploration**.

I calculated each candidate's Euclidean distance to its nearest existing observation and selected the candidate with the greatest nearest-observation distance.

| Function | Selected Exploration Point | Distance to Nearest Observation |
|---|---|---:|
| 1 | `[0.001243, 0.996919]` | 0.394922 |
| 6 | `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]` | 0.886056 |

These were the most distant candidates identified among the 100,000 sampled points for their respective functions.

This approach prioritised gathering information from less-explored regions rather than relying on uninformative GP predictions.

### Step 6 — Assess Final Proposed Query Points

Finally, I compared each proposed query with the coordinates of the highest observed output for its function.

| Function | Selection Method | Distance from Current Best |
|---|---|---:|
| 1 | Max-distance exploration | 0.7760 |
| 2 | GP + UCB | 0.1307 |
| 3 | GP + UCB | 0.2903 |
| 4 | GP + UCB | 0.2857 |
| 5 | GP + UCB | 0.1833 |
| 6 | Max-distance exploration | 1.1281 |
| 7 | GP + UCB | 0.2062 |
| 8 | GP + UCB | 0.9368 |

Functions 2, 3, 4, 5 and 7 selected points relatively close to their best existing observations, suggesting that GP + UCB retained an exploitation component.

Function 8 selected a more distant point, indicating greater exploration relative to its current best observation.

Functions 1 and 6 also selected distant points, although this was a deliberate consequence of adopting Max-distance exploration.

The exact coordinates for all eight proposed queries are recorded in the notebook's **Final Submission Summary**.

## Weekly Summary and Next Steps

Week 1 established a reproducible GP-based optimisation approach using RBF kernels, 100,000 candidate points per function and UCB with κ = 2.5.

The main finding was that **an acquisition function depends on its underlying surrogate model providing informative predictions**.

GP + UCB was used for six functions, while Functions 1 and 6 required Max-distance exploration because their GP predictions and uncertainty were effectively constant.

The next steps were to:

- Submit the eight proposed query points and obtain their outputs.
- Incorporate the new observations into the existing datasets.
- Refit the GP models and reassess their behaviour.
- Consider alternative kernels, such as Matérn, if Functions 1 and 6 continued to produce uninformative predictions.

These results provided the starting point for adapting the optimisation strategy as further observations became available.

