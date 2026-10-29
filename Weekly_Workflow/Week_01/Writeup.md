
# Module 12 — Week 1: Gaussian Process Optimisation

## Methodology

### Step 1 — Examine the Initial Observations

I began by examining the initial datasets for all eight functions, identifying the highest and lowest observed outputs and their corresponding input coordinates. This established a baseline for each function and helped me understand the differences in output scale before fitting the models.

| Function | Highest Observed Output | Lowest Observed Output |
|:--|--:|--:|
| **1** | 7.710875e-16 | -0.003606 |
| **2** | 0.611205 | -0.065624 |
| **3** | -0.034835 | -0.398926 |
| **4** | -4.025542 | -32.625660 |
| **5** | 1088.859618 | 0.112940 |
| **6** | -0.714265 | -2.571170 |
| **7** | 1.364968 | 0.002701 |
| **8** | 9.598482 | 5.592193 |

The initial observations showed considerable variation between functions, particularly in the scale of their outputs. For example, Function 1 had a maximum observed output extremely close to zero, whereas Function 5 had a maximum above 1,000. This reinforced the need to model each function separately rather than assume they would behave similarly.

The corresponding input coordinates are available in the Python notebook.

### Step 2 — Fit RBF Gaussian Processes

I fitted a **Gaussian Process (GP)** model to each function using a **Radial Basis Function (RBF)** kernel. The aim was to use the existing observations to estimate how each function might behave at previously untested input points.

During the fitting process, Functions **1, 5 and 6** produced convergence warnings because their optimised RBF length scales reached the lower bound.

The length scale determines how quickly the GP expects the function's output to change as the inputs change. A very small length scale suggests rapid variation over short distances, which can make it difficult for the model to generalise to unexplored regions.

Rather than ignore these warnings, I investigated whether the lower bound was preventing the optimiser from finding a more suitable value.

**The main question was:** If I relaxed the lower bound, would the model continue pushing the length scale towards zero, or would it settle at a more reasonable value?

After widening the permitted length-scale range, the results were:

| Function | Result After Relaxing Bounds | Interpretation |
|:--|:--|:--|
| **1** | Length scale reached 1e-10 | Continued to favour an extremely small length scale |
| **5** | Length scale settled at 0.269 | Original warning appeared to be caused by the default lower bound |
| **6** | Length scale reached 1e-10 | Continued to favour an extremely small length scale |

**Function 5** behaved more sensibly after the bounds were relaxed. Its length scale settled at **0.269**, suggesting that the original warning had been caused by the optimiser reaching the default limit rather than an underlying modelling problem.

**Functions 1 and 6**, however, continued to push their length scales towards the new lower bound of **1e-10**. This suggested that the data itself was causing the GP to model extremely rapid changes in output.

I therefore decided to retain the original GP models for Functions 1 and 6 rather than continue reducing the lower bound. Before using these models to select new queries, I needed to investigate whether their predictions and uncertainty estimates were meaningful.

### Step 3 — Generate Candidate Points and Predict Mean and Uncertainty

For each function, I randomly generated **100,000 candidate points** across the permitted input space, with each input dimension ranging from 0 to 1.

I then used the fitted Gaussian Processes to estimate two values for every candidate:

- **Predicted Mean:** The output the GP expects at that input point.
- **Predictive Uncertainty:** How uncertain the GP is about that prediction, represented by its predictive standard deviation.

These estimates provided a way to assess both the expected performance of each candidate and the potential value of exploring less certain areas.

The predicted ranges across the 100,000 candidates were:

| Function | Predicted Output Range | Uncertainty Range |
|:--|:--|:--|
| **1** | -0.000361 to -0.000361 | 0.001082 to 0.001082 |
| **2** | -0.077317 to 0.643977 | 0.000068 to 0.225365 |
| **3** | -0.397833 to -0.018176 | 0.002777 to 0.084213 |
| **4** | -35.682037 to -0.922445 | 0.067992 to 5.263611 |
| **5** | -71.908033 to 1113.402974 | 12.509737 to 245.522395 |
| **6** | -1.495390 to -1.495390 | 0.449000 to 0.449000 |
| **7** | -0.020560 to 1.208211 | 0.061869 to 0.302129 |
| **8** | 5.286512 to 10.217038 | 0.064678 to 0.681040 |

**Key Observation**

Functions **1 and 6** produced effectively identical predicted means and uncertainty values across all 100,000 candidate points.

This appeared to be a consequence of their extremely small RBF length scales. The models were treating almost all unobserved points as unrelated to the existing observations, resulting in almost constant predictions across the search space.

This was an important limitation because the next stage of optimisation relied on using predicted outputs and uncertainty to distinguish between candidates.

If these values were effectively identical everywhere, an acquisition function would have little useful information on which to base its selections.

The remaining six functions showed meaningful variation in their predictions and uncertainty estimates, so I could proceed with the planned acquisition strategy.

### Step 4 — Calculate Upper Confidence Bound (UCB)

To select the next query point, I used the **Upper Confidence Bound (UCB)** acquisition function.

UCB combines the predicted mean output with the uncertainty of that prediction:

**UCB(x) = μ(x) + κσ(x)**

Where:

- **μ(x):** Predicted mean output from the Gaussian Process.
- **σ(x):** Predictive standard deviation, representing uncertainty.
- **κ:** Parameter controlling the balance between exploration and exploitation.

I selected **κ = 2.5**, giving uncertainty a meaningful weighting while still considering the predicted performance of each candidate.

This approach balances two competing objectives:

- **Exploitation:** Selecting points where the GP predicts a high output, based on what has already been observed.
- **Exploration:** Selecting points where the GP is less certain, allowing potentially promising regions to be investigated.

I calculated UCB for each of the 100,000 candidate points and identified the point with the highest score for each function.

| Function | Predicted Output | Uncertainty | Highest UCB Score |
|:--|--:|--:|--:|
| **1** | -0.000361 | 0.001082 | 0.002344 |
| **2** | 0.537302 | 0.174852 | 0.974432 |
| **3** | -0.041871 | 0.071974 | 0.138065 |
| **4** | -1.109357 | 0.992949 | 1.373015 |
| **5** | 1003.710099 | 139.196890 | 1351.702325 |
| **6** | -1.495390 | 0.449000 | -0.372891 |
| **7** | 1.136197 | 0.193256 | 1.619336 |
| **8** | 10.015345 | 0.507951 | 11.285223 |

**Assessment**

For Functions **2, 3, 4, 5, 7 and 8**, UCB provided a useful way to rank candidates by combining predicted performance and uncertainty.

These functions showed enough variation across the search space for UCB to distinguish between different points, so I retained the highest-scoring candidates as the proposed queries.

However, Functions **1 and 6** remained problematic.

Because their predicted means and uncertainty estimates were effectively constant, their UCB scores were also almost identical across the candidate space.

Although the calculation still returned a highest-scoring point, this selection was not meaningful. The GP was not providing enough information to identify one candidate as more promising than another.

I therefore decided not to use the UCB-selected points for Functions 1 and 6 and instead investigated a more exploration-focused approach.

### Step 5 — Use Distance-Based Exploration for Functions 1 and 6

Given the limitations identified in the GP models for Functions **1 and 6**, I decided to use **distance-based exploration** rather than rely on their UCB scores.

The reasoning was that if the model could not distinguish between candidates based on predicted performance or uncertainty, the next best approach was to gather information from a relatively unexplored region.

I therefore selected the candidate point with the greatest distance from its nearest existing observation.

This meant prioritising areas of the search space that were least represented in the initial dataset, rather than selecting points based on unreliable GP predictions.

The selected exploration points were:

| Function | Selected Exploration Point | Distance to Nearest Observation |
|:--|:--|--:|
| **1** | `[0.00124268, 0.99691888]` | 0.394922 |
| **6** | `[0.91049992, 0.03922596, 0.99959186, 0.08178911, 0.93948142]` | 0.886056 |

**Function 1**

The selected point was `[0.001243, 0.996919]`, approximately **0.395** from its nearest existing observation.

This represented the most unexplored candidate identified from the 100,000 points tested.

**Function 6**

The selected point was `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]`, approximately **0.886** from its nearest existing observation.

Again, this was the candidate furthest from the existing observations among those tested.

**Reasoning Behind the Decision**

The objective was not necessarily to maximise the immediate predicted output, but to **increase the information available to the model**.

Selecting points far from existing observations reduced the likelihood of repeatedly sampling regions that had already been explored.

Once the new outputs were available, I planned to incorporate them into the existing datasets and reassess the GP models.

I also considered investigating alternative kernels, such as **Matérn**, in future iterations to determine whether they could provide a better representation of Functions 1 and 6.

This was an important methodological decision because it demonstrated that the acquisition strategy needed to respond to the behaviour of the fitted models rather than be applied identically to every function.

### Step 6 — Assess the Final Query Points

Finally, I compared the proposed query points against the current highest-performing observations.

The purpose was to understand whether each query represented an attempt to exploit a promising region or explore a more distant part of the search space.

I calculated the distance between each proposed query and the input coordinates associated with the current highest observed output.

| Function | Current Best Output | Distance From Current Best |
|:--|--:|--:|
| **1** | 7.710875e-16 | 0.7760 |
| **2** | 0.611205 | 0.1307 |
| **3** | -0.034835 | 0.2903 |
| **4** | -4.025542 | 0.2857 |
| **5** | 1088.859618 | 0.1833 |
| **6** | -0.714265 | 1.1281 |
| **7** | 1.364968 | 0.2062 |
| **8** | 9.598482 | 0.9368 |

**Interpretation**

Functions **2, 3, 4, 5 and 7** selected points relatively close to their current best observations.

This suggested that UCB was retaining an exploitation component, identifying candidates in regions where previous observations had already demonstrated relatively strong performance.

**Function 8** selected a substantially more distant point, approximately **0.937** from its current best observation.

This indicated greater exploration within its higher-dimensional search space.

**Functions 1 and 6** also selected distant points, although this was intentional rather than a direct consequence of UCB.

Their distances from the current best observations were approximately **0.776** and **1.128**, respectively, reflecting the decision to prioritise unexplored regions after identifying limitations in their Gaussian Process models.

Overall, this comparison helped confirm that the proposed queries reflected the intended balance between exploitation and exploration, with a more deliberate exploratory strategy used where the GP predictions were unreliable.

## Week 1 Process Summary

The Week 1 optimisation process consisted of the following stages:

1. **Initial Data Exploration:** Examined the existing observations and identified the highest and lowest outputs for each function.
2. **Gaussian Process Modelling:** Fitted RBF Gaussian Processes and investigated convergence warnings, particularly for Functions 1, 5 and 6.
3. **Candidate Generation:** Generated 100,000 random candidate points per function and calculated predicted means and uncertainty.
4. **UCB Acquisition:** Applied UCB with **κ = 2.5** to select candidates for Functions 2, 3, 4, 5, 7 and 8.
5. **Alternative Exploration:** Used distance-based exploration for Functions 1 and 6 because their GP predictions and UCB scores were effectively flat.
6. **Query Assessment:** Compared the selected points with the current best observations to assess the balance between exploration and exploitation.


