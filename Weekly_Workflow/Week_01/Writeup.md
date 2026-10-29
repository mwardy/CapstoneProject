# Module 12 — Week 1: Gaussian Process Optimisation

## Weekly Objective and Step-by-Step Plan

The objective of Week 1 was to establish a baseline optimisation approach for eight unknown black-box functions and identify one new query point for each function.

The process followed six steps:

1. Examine the initial observations and identify the highest and lowest outputs.
2. Fit Gaussian Process (GP) models using RBF kernels and investigate convergence warnings.
3. Generate 100,000 candidate points per function and examine predicted outputs and uncertainty.
4. Apply Upper Confidence Bound (UCB) to identify promising candidates.
5. Use an alternative exploration strategy where GP predictions were uninformative.
6. Assess the final query points against the best existing observations.

## Methodology

Each function was modelled independently using Gaussian Process Regression with an RBF kernel. This provided estimates of both the expected output and uncertainty at previously unobserved locations.

For each function, 100,000 random candidate points were generated within the permitted input space of [0, 1] for every dimension.

Candidate points were ranked using the Upper Confidence Bound acquisition function:

**UCB(x) = μ(x) + κσ(x)**

Where μ(x) represents the predicted output, σ(x) represents predictive uncertainty, and κ controls the balance between exploration and exploitation.

I selected **κ = 2.5**, allowing the acquisition function to consider both high predicted outputs and uncertain regions.

Where the fitted GP could not meaningfully distinguish between candidates, I used distance-based exploration instead.

## Results and Interpretation by Step

### Step 1 — Examine Initial Observations

The initial observations established the highest and lowest outputs for each function.

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

The observations revealed considerable differences in output scale. Function 1 had a maximum extremely close to zero, whereas Function 5 exceeded 1,000.

This reinforced the decision to model each function separately rather than assume similar behaviour across the eight functions.

### Step 2 — Fit RBF Gaussian Processes

The initial GP fitting produced convergence warnings for **Functions 1, 5 and 6**, where the optimised RBF length scales reached their lower bounds.

I investigated whether relaxing the lower bound would allow the models to find more suitable values.

| Function | Length Scale After Relaxing Bounds | Interpretation |
|---|---:|---|
| 1 | 1e-10 | Continued towards the minimum permitted value |
| 5 | 0.269 | Settled at a more moderate length scale |
| 6 | 1e-10 | Continued towards the minimum permitted value |

Function 5 improved after relaxing the bounds, suggesting that its original warning was associated with the optimiser reaching the permitted limit.

Functions 1 and 6 continued to favour extremely small length scales. This suggested that their GPs were modelling very rapid changes over short distances.

I therefore retained the original models for Functions 1 and 6 and investigated whether their predictions were sufficiently informative for candidate selection.

### Step 3 — Generate Candidates and Assess Predictions

I generated **100,000 random candidate points per function** and used the fitted GPs to calculate predicted means and standard deviations.

The most important finding was that Functions 1 and 6 produced effectively constant predictions across the candidate search space.

| Function | Approximate Predicted Mean | Approximate Uncertainty |
|---|---:|---:|
| 1 | -0.000361 | 0.001082 |
| 6 | -1.495390 | 0.449000 |

Their extremely small RBF length scales meant that the models treated almost all unobserved candidate locations as unrelated to the existing observations.

Consequently, neither predicted performance nor uncertainty provided a meaningful basis for distinguishing between candidate points.

The remaining six functions showed variation in their predictions and uncertainty, allowing the planned acquisition strategy to proceed.

### Step 4 — Apply Upper Confidence Bound

I calculated UCB scores using **κ = 2.5** and selected the highest-scoring candidate for each function.

For Functions **2, 3, 4, 5, 7 and 8**, UCB provided a useful ranking by combining predicted performance with uncertainty.

For Functions **1 and 6**, however, the near-constant predictions resulted in almost flat UCB scores. Although the calculation still identified numerical maxima, these selections were not meaningfully supported by the models.

I therefore decided not to use their UCB-selected points.

### Step 5 — Apply Distance-Based Exploration

For Functions 1 and 6, I adopted a distance-based exploration strategy.

Rather than relying on unreliable GP predictions, I selected the candidate with the greatest distance from its nearest existing observation.

| Function | Selected Exploration Point | Distance to Nearest Observation |
|---|---|---:|
| 1 | `[0.001243, 0.996919]` | 0.394922 |
| 6 | `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]` | 0.886056 |

These were the most distant candidates identified among the 100,000 sampled points for their respective functions.

The intention was to obtain information from less-explored regions rather than immediately maximise predicted performance.

Once the new observations became available, I planned to incorporate them into the datasets and reassess the GP models.

### Step 6 — Assess Final Query Points

Finally, I compared the proposed queries with the coordinates of the highest observed output for each function.

| Function | Selection Method | Distance from Current Best |
|---|---|---:|
| 1 | Distance-based exploration | 0.7760 |
| 2 | UCB | 0.1307 |
| 3 | UCB | 0.2903 |
| 4 | UCB | 0.2857 |
| 5 | UCB | 0.1833 |
| 6 | Distance-based exploration | 1.1281 |
| 7 | UCB | 0.2062 |
| 8 | UCB | 0.9368 |

Functions 2, 3, 4, 5 and 7 selected points relatively close to their best existing observations, indicating that UCB retained an exploitation component.

Function 8 selected a more distant point, suggesting greater exploration.

Functions 1 and 6 also selected distant points, although this was a deliberate consequence of switching to distance-based exploration.

The exact proposed coordinates for all eight functions are recorded in the accompanying Python notebook.

## Weekly Summary and Next Steps

Week 1 established a Gaussian Process and UCB-based optimisation approach for the eight black-box functions.

The main finding was that **the effectiveness of an acquisition function depends on the quality of its underlying surrogate model**.

Although UCB was suitable for six functions, the nearly constant GP predictions for Functions 1 and 6 required a different strategy.

The next steps were to:

- Submit the eight selected query points and obtain their outputs.
- Incorporate the new observations into the existing datasets.
- Refit the GP models and assess whether the additional data improved their predictions.
- Consider alternative kernels, such as Matérn, if Functions 1 and 6 continued to produce uninformative predictions.

This provided a starting point for adapting the optimisation strategy as additional observations became available.
