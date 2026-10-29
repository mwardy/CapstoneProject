
# Module 12 — Week 1: Gaussian process optimisation

## Weekly objective and step-by-step plan

The objective of Week 1 was to establish an initial optimisation approach for eight unknown black-box functions and select one new query point for each function.

A key consideration was how to balance exploration and exploitation. Exploitation involves selecting points in regions expected to produce strong outputs, while exploration involves investigating less-understood regions that might contain better solutions.

The process followed six steps:

1. Examine the initial observations and identify the highest and lowest outputs.
2. Fit Gaussian process (GP) models using RBF kernels and investigate convergence warnings.
3. Generate 100,000 candidate points per function and assess predicted outputs and uncertainty.
4. Apply upper confidence bound (UCB) to balance exploration and exploitation.
5. Use maximum-distance exploration where GP predictions were uninformative.
6. Assess the final proposed queries and their exploration–exploitation characteristics.

## Methodology

Each function was modelled independently using Gaussian process regression with an RBF kernel, normalised outputs, ten optimiser restarts and a random state of 42.

For each function, 100,000 candidate points were generated uniformly within the permitted input space of [0, 1] for every dimension, using a fixed random seed.

The initial acquisition strategy combined Gaussian process predictions with upper confidence bound (GP + UCB):

**UCB(x) = μ(x) + κσ(x)**

Where:

- **μ(x):** Predicted mean output from the GP.
- **σ(x):** Predictive standard deviation, representing uncertainty.
- **κ:** Parameter controlling the balance between exploration and exploitation.

I selected κ = 2.5 to give uncertainty a meaningful weighting while retaining the influence of predicted performance.

The aim was to exploit regions where the GP predicted strong outputs while continuing to explore uncertain areas that might contain better solutions.

However, where the GP produced effectively constant predictions and uncertainty, I used maximum-distance exploration instead. This involved selecting the candidate furthest from its nearest existing observation, prioritising coverage of the search space over predicted performance.

## Results and interpretation by step

### Step 1 — Examine initial observations

I examined the initial datasets to establish the highest and lowest observed outputs for each function.

| Function | Highest observed output | Lowest observed output |
|---|---:|---:|
| 1 | 7.710875e-16 | -0.003606 |
| 2 | 0.611205 | -0.065624 |
| 3 | -0.034835 | -0.398926 |
| 4 | -4.025542 | -32.625660 |
| 5 | 1088.859618 | 0.112940 |
| 6 | -0.714265 | -2.571170 |
| 7 | 1.364968 | 0.002701 |
| 8 | 9.598482 | 5.592193 |

The initial observations showed considerable differences in output scale. Function 1 had a maximum extremely close to zero, whereas Function 5 exceeded 1,000.

This reinforced the decision to model each function independently rather than assume similar behaviour across the eight functions.

The best observed outputs also provided a reference point for evaluating whether subsequent query selections concentrated around previously promising regions or explored more distant areas.

### Step 2 — Fit RBF Gaussian processes

The initial GP models produced convergence warnings for Functions 1, 5 and 6, where the optimised RBF length scales reached the lower bound of 1e-5.

The RBF length scale controls how quickly the model expects outputs to change as input coordinates change. Extremely small length scales can cause predictions at most unobserved locations to become weakly influenced by the existing data.

I investigated whether widening the permitted length-scale range would allow the optimiser to identify more suitable values.

| Function | Length scale after relaxing bounds | Interpretation |
|---|---:|---|
| 1 | 1e-10 | Continued towards the minimum permitted value |
| 5 | 0.269 | Settled at a more moderate length scale |
| 6 | 1e-10 | Continued towards the minimum permitted value |

Function 5 converged to a length scale of approximately 0.269 after widening the bounds. I therefore refitted Function 5 using the wider bounds and replaced its original GP model.

Functions 1 and 6 continued to favour extremely small length scales, suggesting that the fitted models were treating observations as becoming unrelated over very short distances.

I retained their original GP models but investigated whether their predictions and uncertainty estimates were sufficiently informative for candidate selection.

This was particularly important for the exploration–exploitation trade-off. If the GP could not distinguish between promising and uncertain regions, the UCB acquisition function would have limited value.

### Step 3 — Generate candidates and assess predictions

I generated 100,000 random candidate points per function and used the fitted GPs to calculate predicted means and uncertainty.

The most important finding was that Functions 1 and 6 produced effectively constant predictions across the candidate search space.

| Function | Approximate predicted mean | Approximate uncertainty |
|---|---:|---:|
| 1 | -0.000361 | 0.001082 |
| 6 | -1.495390 | 0.449000 |

Their extremely small RBF length scales meant that the models treated almost all unobserved candidates as unrelated to the existing observations.

Consequently, neither predicted performance nor uncertainty provided a meaningful basis for distinguishing between candidates.

This created two problems:

- **Exploitation:** The models could not reliably identify which unobserved locations were expected to produce higher outputs.
- **Exploration:** Predictive uncertainty was effectively constant, so the models could not identify which candidate locations were comparatively more uncertain.

The remaining six functions showed meaningful variation in their predictions and uncertainty, allowing the planned GP + UCB strategy to proceed.

### Step 4 — Apply upper confidence bound

I calculated UCB scores for all 100,000 candidate points per function using κ = 2.5.

UCB balances two competing objectives:

- **Exploitation:** Prioritising locations where the GP predicts relatively high outputs based on the available observations.
- **Exploration:** Prioritising locations where uncertainty is greater, allowing potentially promising but less-understood regions to be investigated.

By combining these components, GP + UCB can identify candidates offering either strong predicted performance, substantial uncertainty or a combination of both.

For Functions 2, 3, 4, 5, 7 and 8, the GP produced meaningful variation in predicted outputs and uncertainty. The highest-scoring UCB candidates were therefore retained as the proposed queries.

For Functions 1 and 6, however, the effectively constant predictions and uncertainty resulted in nearly identical UCB scores across the candidate space.

Although the calculation still returned a highest-scoring candidate, there was little evidence that this location offered a better exploration–exploitation balance than other candidates.

Using these points would have meant relying on a numerical maximum that was not meaningfully supported by the GP models.

I therefore decided not to use the UCB-selected points for Functions 1 and 6.

### Step 5 — Apply maximum-distance exploration

For Functions 1 and 6, I adopted maximum-distance exploration rather than relying on GP + UCB.

The reasoning was that the fitted models could not reliably identify regions with higher predicted performance or distinguish areas of greater uncertainty.

Consequently, attempting to exploit their predictions was unlikely to provide a meaningful advantage.

Instead, I deliberately prioritised exploration over immediate exploitation.

I calculated each candidate's Euclidean distance to its nearest existing observation and selected the candidate with the greatest nearest-observation distance.

| Function | Selected exploration point | Distance to nearest observation |
|---|---|---:|
| 1 | `[0.001243, 0.996919]` | 0.394922 |
| 6 | `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]` | 0.886056 |

These were the most distant candidates identified among the 100,000 sampled points for their respective functions.

By selecting points far from existing observations, the intention was to obtain information about previously underrepresented regions of the search space.

This approach did not guarantee an immediate improvement in the objective value. Instead, it prioritised gathering new information that could potentially improve subsequent GP models and future optimisation decisions.

Once the returned outputs became available, I planned to incorporate them into the existing datasets and reassess the models.

This decision also showed why it was useful to adapt the acquisition strategy to the behaviour of each fitted model rather than apply the same approach to every function.

### Step 6 — Assess final proposed query points

Finally, I compared each proposed query with the coordinates of the highest observed output for its function.

This helped assess whether the selections were concentrated around previously promising regions or located further away.

| Function | Selection method | Distance from current best |
|---|---|---:|
| 1 | Max-distance exploration | 0.7760 |
| 2 | GP + UCB | 0.1307 |
| 3 | GP + UCB | 0.2903 |
| 4 | GP + UCB | 0.2857 |
| 5 | GP + UCB | 0.1833 |
| 6 | Max-distance exploration | 1.1281 |
| 7 | GP + UCB | 0.2062 |
| 8 | GP + UCB | 0.9368 |

Functions 2, 3, 4, 5 and 7 selected points relatively close to their best existing observations, suggesting that exploitation remained an important component of their GP + UCB selections.

Function 8 selected a more distant point, indicating greater exploration relative to its current best observation.

Functions 1 and 6 also selected distant points, but their selection was different. Rather than balancing predicted performance against uncertainty, these functions used an explicitly exploration-focused strategy because their GP models could not meaningfully distinguish candidates.

Their distances from the current best observations were approximately 0.776 and 1.128, respectively.

It is important to distinguish distance from the current best observation from distance to the nearest existing observation. The latter was the actual selection criterion for maximum-distance exploration.

Although proximity to the current best observation can suggest an exploitation component, it does not establish precisely how much of a UCB score came from predicted performance rather than uncertainty.

Overall, the final selections reflected two approaches: balancing exploration and exploitation through GP + UCB where the models were informative, and prioritising exploration where they were not.

The exact coordinates for all eight proposed queries are recorded in the notebook's final submission summary.

## Weekly summary and next steps

Week 1 established a GP-based optimisation approach using RBF kernels, 100,000 candidate points per function and UCB with κ = 2.5.

The main methodological decision concerned the exploration–exploitation trade-off.

For six functions, GP + UCB provided a way to balance both objectives. However, the uninformative GP predictions for Functions 1 and 6 meant that exploring previously underrepresented regions was more appropriate than attempting to exploit unreliable predictions.

This showed that the usefulness of an acquisition function depends on whether the underlying GP model can meaningfully distinguish between candidate locations.

The next steps were to:

- Submit the eight proposed query points and obtain their outputs.
- Incorporate the new observations into the existing datasets.
- Refit the GP models and reassess their predictions and uncertainty.
- Consider alternative kernels, such as Matérn, if Functions 1 and 6 continued to produce uninformative predictions.

These results provided a starting point for adapting the optimisation strategy as additional observations became available.

