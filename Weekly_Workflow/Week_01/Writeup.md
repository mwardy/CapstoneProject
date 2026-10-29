
# Module 12 — Week 1

## Methodology

### Step 1 — Search for the Highest and Lowest Observed Outputs and Note the Corresponding Inputs

I began by identifying the highest and lowest observed outputs for each of the eight functions, along with their corresponding input coordinates.

**Function 1**

**Highest Observed Output:**  
- x1: 7.310236e-01  
- x2: 7.329999e-01  
- Output: 7.710875e-16

**Lowest Observed Output:**  
- x1: 0.650114  
- x2: 0.681526  
- Output: -0.003606

**Function 2**

**Highest Observed Output:**  
- x1: 0.702637  
- x2: 0.926564  
- Output: 0.611205

**Lowest Observed Output:**  
- x1: 0.142699  
- x2: 0.349005  
- Output: -0.065624

**Function 3**

**Highest Observed Output:**  
- x1: 0.492581  
- x2: 0.611593  
- x3: 0.340176  
- Output: -0.034835

**Lowest Observed Output:**  
- x1: 0.151837  
- x2: 0.439991  
- x3: 0.990882  
- Output: -0.398926

**Function 4**

**Highest Observed Output:**  
- x1: 0.577766  
- x2: 0.428772  
- x3: 0.425826  
- x4: 0.249007  
- Output: -4.025542

**Lowest Observed Output:**  
- x1: 0.948389  
- x2: 0.894513  
- x3: 0.851638  
- x4: 0.552196  
- Output: -32.625660

**Function 5**

**Highest Observed Output:**  
- x1: 0.224189  
- x2: 0.846480  
- x3: 0.879484  
- x4: 0.878516  
- Output: 1088.859618

**Lowest Observed Output:**  
- x1: 0.438350  
- x2: 0.804340  
- x3: 0.210245  
- x4: 0.151295  
- Output: 0.112940

**Function 6**

**Highest Observed Output:**  
- x1: 0.728186  
- x2: 0.154693  
- x3: 0.732552  
- x4: 0.693997  
- x5: 0.056401  
- Output: -0.714265

**Lowest Observed Output:**  
- x1: 0.125720  
- x2: 0.862725  
- x3: 0.028544  
- x4: 0.246605  
- x5: 0.751206  
- Output: -2.571170

**Function 7**

**Highest Observed Output:**  
- x1: 0.057896  
- x2: 0.491672  
- x3: 0.247422  
- x4: 0.218118  
- x5: 0.420428  
- x6: 0.730970  
- Output: 1.364968

**Lowest Observed Output:**  
- x1: 0.879871  
- x2: 0.397962  
- x3: 0.003635  
- x4: 0.956991  
- x5: 0.264514  
- x6: 0.114869  
- Output: 0.002701

**Function 8**

**Highest Observed Output:**  
- x1: 0.056447  
- x2: 0.065956  
- x3: 0.022929  
- x4: 0.038786  
- x5: 0.403935  
- x6: 0.801055  
- x7: 0.488307  
- x8: 0.893085  
- Output: 9.598482

**Lowest Observed Output:**  
- x1: 0.984933  
- x2: 0.699506  
- x3: 0.998885  
- x4: 0.180148  
- x5: 0.580143  
- x6: 0.231087  
- x7: 0.490827  
- x8: 0.313683  
- Output: 5.592193

### Step 2 — Fit RBF Gaussian Processes

Functions **1, 5 and 6** produced convergence warnings because the optimised RBF length scale reached its lower bound, suggesting that the Gaussian Process was modelling very rapid changes in output. The model predictions and uncertainty therefore needed to be checked before using them to select new query points.

The key question was:

**If the lower bound is relaxed, does the model push the length scale even closer to zero, or does it settle at a sensible value?**

**Findings:**

- **Function 5:** The model behaved more sensibly once the optimiser was allowed to search a wider range. The Gaussian Process settled on a length scale of **0.269**, suggesting that the original warning was caused by the optimiser reaching the default lower limit.
- **Functions 1 and 6:** Even after allowing much smaller values, both models immediately pushed the length scale down to **1e-10** and produced the same warning again. This suggested that the data itself was leading the GP to model extremely rapid changes in the functions.
- **Decision:** I retained the original GP models for Functions 1 and 6 rather than continuing to reduce the minimum length scale.

The convergence warnings were therefore investigated by relaxing the RBF length-scale bounds. Function 5 converged to a more stable length scale, while Functions 1 and 6 continued to converge to the minimum value. Their original Gaussian Process models were retained for the initial optimisation.

### Step 3 — Generate Candidate Points and Predict Mean and Uncertainty

For each function, **100,000 candidate points** were randomly generated across the permitted input space between 0 and 1.

The fitted Gaussian Process was then used to estimate the **predicted mean output** and **uncertainty** for each candidate. This provided an indication of both how well each unobserved point was expected to perform and how confident the model was in that prediction.

**Function 1**  
- Dimensions: 2  
- Candidate Points: 100,000  
- Predicted Output Range: -0.000361 to -0.000361  
- Uncertainty Range: 0.001082 to 0.001082

**Function 2**  
- Dimensions: 2  
- Candidate Points: 100,000  
- Predicted Output Range: -0.077317 to 0.643977  
- Uncertainty Range: 0.000068 to 0.225365

**Function 3**  
- Dimensions: 3  
- Candidate Points: 100,000  
- Predicted Output Range: -0.397833 to -0.018176  
- Uncertainty Range: 0.002777 to 0.084213

**Function 4**  
- Dimensions: 4  
- Candidate Points: 100,000  
- Predicted Output Range: -35.682037 to -0.922445  
- Uncertainty Range: 0.067992 to 5.263611

**Function 5**  
- Dimensions: 4  
- Candidate Points: 100,000  
- Predicted Output Range: -71.908033 to 1113.402974  
- Uncertainty Range: 12.509737 to 245.522395

**Function 6**  
- Dimensions: 5  
- Candidate Points: 100,000  
- Predicted Output Range: -1.495390 to -1.495390  
- Uncertainty Range: 0.449000 to 0.449000

**Function 7**  
- Dimensions: 6  
- Candidate Points: 100,000  
- Predicted Output Range: -0.020560 to 1.208211  
- Uncertainty Range: 0.061869 to 0.302129

**Function 8**  
- Dimensions: 8  
- Candidate Points: 100,000  
- Predicted Output Range: 5.286512 to 10.217038  
- Uncertainty Range: 0.064678 to 0.681040

**Overview**

Functions **1 and 6** showed an issue caused by their extremely small RBF length scales.

For these functions, the Gaussian Process produced almost identical predictions and uncertainty across all 100,000 candidate points, suggesting that it was treating almost all unobserved points as unrelated to the existing observations.

As a result, **UCB would be unable to meaningfully distinguish between candidate points** for these two functions.

Functions **2, 3, 4, 5, 7 and 8** showed variation in both predicted outputs and uncertainty and could therefore proceed normally with UCB.

### Step 4 — Calculate Upper Confidence Bound (UCB)

The predicted outputs and uncertainty were combined using the **Upper Confidence Bound (UCB)** acquisition function, with **κ = 2.5** to provide a balance between exploitation and exploration.

UCB was calculated for each of the 100,000 candidate points, and the point with the highest score was selected as the proposed query.

This allowed both points with high predicted outputs and uncertain areas with potential for improvement to be considered.

**Function 1**  
- Selected Point: `[0.77395605, 0.43887844]`  
- Predicted Output: -0.00036060626443627054  
- Uncertainty: 0.0010818187933090686  
- UCB: 0.002343940718836401

**Function 2**  
- Selected Point: `[0.8217782, 0.98030138]`  
- Predicted Output: 0.5373024810236847  
- Uncertainty: 0.17485188463025975  
- UCB: 0.9744321925993341

**Function 3**  
- Selected Point: `[0.40949935, 0.39899996, 0.51948667]`  
- Predicted Output: -0.04187059316125945  
- Uncertainty: 0.07197438103562513  
- UCB: 0.1380653594278034

**Function 4**  
- Selected Point: `[0.39915655, 0.43320344, 0.35769894, 0.461233]`  
- Predicted Output: -1.109357344181845  
- Uncertainty: 0.9929487520649243  
- UCB: 1.3730145359804657

**Function 5**  
- Selected Point: `[0.35226763, 0.85495745, 0.92921014, 0.99954104]`  
- Predicted Output: 1003.7100993151818  
- Uncertainty: 139.19689039784117  
- UCB: 1351.7023253097846

**Function 6**  
- Selected Point: `[0.42871117, 0.38510403, 0.39053256, 0.72350556, 0.95165422]`  
- Predicted Output: -1.4953899396939465  
- Uncertainty: 0.44899959039781934  
- UCB: -0.3728909636993982

**Function 7**  
- Selected Point: `[0.12062732, 0.40392121, 0.37997272, 0.14653261, 0.33585285, 0.76330458]`  
- Predicted Output: 1.136196501448108  
- Uncertainty: 0.1932559409633074  
- UCB: 1.6193363538563763

**Function 8**  
- Selected Point: `[0.24339702, 0.03611478, 0.07835798, 0.05372058, 0.78385373, 0.27839289, 0.03494745, 0.42903195]`  
- Predicted Output: 10.015345100551379  
- Uncertainty: 0.5079509989217438  
- UCB: 11.285222597855737

**Overall Assessment**

UCB was calculated for each of the 100,000 candidate points using **κ = 2.5**, combining the GP's predicted output with its uncertainty.

The candidate with the highest UCB was selected for each function, allowing points with strong predicted performance, high uncertainty, or a combination of both to be prioritised.

This provided a balance between exploiting promising areas and exploring less certain regions of the search space.

Functions **1 and 6** remained problematic because their near-constant predictions and uncertainty resulted in an almost flat UCB surface, meaning their selected points could not yet be considered reliable.

### Step 5 — Handling Functions 1 and 6

For Functions **1 and 6**, the RBF Gaussian Process produced extremely small length scales, resulting in almost identical predictions and uncertainty across the candidate search space.

As UCB was therefore unable to meaningfully distinguish between candidate points, I decided to use an **exploration-based approach** instead.

A point would be selected from a relatively unexplored area of the search space to provide the model with new information.

Once the output from this query was received, it would be added to the existing observations and the RBF model reassessed. In future iterations, an alternative kernel, such as **Matérn**, could also be introduced and compared with RBF to determine whether it provided a more suitable representation of these functions.

**Reasoning Behind the Exploration Strategy**

As the GP models for Functions 1 and 6 were unable to identify more promising areas through their predictions or uncertainty, the aim was to maximise the information gained from the next query rather than immediately target a higher output.

Selecting a point far from the existing observations reduced duplication of areas already sampled and provided information about a new part of the search space.

This represented a deliberately exploration-focused decision, which could then be reassessed once the new output was received.

**Function 1**  
- Selected Exploration Point: `[0.00124268, 0.99691888]`  
- Distance to Nearest Existing Observation: 0.39492227534104346

**Function 6**  
- Selected Exploration Point: `[0.91049992, 0.03922596, 0.99959186, 0.08178911, 0.93948142]`  
- Distance to Nearest Existing Observation: 0.8860560302850456

**Summary**

Using this exploration-based approach, Function 1 selected the point `[0.001243, 0.996919]`, approximately **0.395** from its nearest existing observation.

Function 6 selected `[0.910500, 0.039226, 0.999592, 0.081789, 0.939481]`, approximately **0.886** from its nearest existing observation.

These represented the most unexplored candidates identified from the 100,000 points tested.

As the GP and UCB could not meaningfully differentiate between candidates for these functions, the aim was to maximise the information gained from the next query rather than immediately exploit an area predicted to have a high output.

### Step 6 — Assessing Selected Query Points

The proposed query points were compared with the current highest-performing observations to assess how the optimisation strategy was navigating each search space.

For Functions **2, 3, 4, 5 and 7**, the selected points remained relatively close to the current best observations, suggesting that UCB retained an exploitation component by searching around promising regions.

Function **8** selected a substantially more distant point, indicating greater exploration within its higher-dimensional search space.

Functions **1 and 6** also selected distant points, although this was intentional because an exploration-based strategy had been adopted following the limitations identified with their RBF models.

**Function 1**  
- Current Best Point: `[0.731024, 0.733]`  
- Current Best Output: 7.710875114502849e-16  
- Proposed New Point: `[0.001243, 0.996919]`  
- Distance From Current Best: 0.776

**Function 2**  
- Current Best Point: `[0.702637, 0.926564]`  
- Current Best Output: 0.6112052157614438  
- Proposed New Point: `[0.821778, 0.980301]`  
- Distance From Current Best: 0.1307

**Function 3**  
- Current Best Point: `[0.492581, 0.611593, 0.340176]`  
- Current Best Output: -0.034835313350078584  
- Proposed New Point: `[0.409499, 0.399, 0.519487]`  
- Distance From Current Best: 0.2903

**Function 4**  
- Current Best Point: `[0.577766, 0.428772, 0.425826, 0.249007]`  
- Current Best Output: -4.025542281908162  
- Proposed New Point: `[0.399157, 0.433203, 0.357699, 0.461233]`  
- Distance From Current Best: 0.2857

**Function 5**  
- Current Best Point: `[0.224189, 0.84648, 0.879484, 0.878516]`  
- Current Best Output: 1088.8596181962705  
- Proposed New Point: `[0.352268, 0.854957, 0.92921, 0.999541]`  
- Distance From Current Best: 0.1833

**Function 6**  
- Current Best Point: `[0.728186, 0.154693, 0.732552, 0.693997, 0.056401]`  
- Current Best Output: -0.7142649478202404  
- Proposed New Point: `[0.9105, 0.039226, 0.999592, 0.081789, 0.939481]`  
- Distance From Current Best: 1.1281

**Function 7**  
- Current Best Point: `[0.057896, 0.491672, 0.247422, 0.218118, 0.420428, 0.73097]`  
- Current Best Output: 1.3649683044991994  
- Proposed New Point: `[0.120627, 0.403921, 0.379973, 0.146533, 0.335853, 0.763305]`  
- Distance From Current Best: 0.2062

**Function 8**  
- Current Best Point: `[0.056447, 0.065956, 0.022929, 0.038786, 0.403935, 0.801055, 0.488307, 0.893085]`  
- Current Best Output: 9.598482002566342  
- Proposed New Point: `[0.243397, 0.036115, 0.078358, 0.053721, 0.783854, 0.278393, 0.034947, 0.429032]`  
- Distance From Current Best: 0.9368

## Week 1 Process Summary

- Examined the initial observations.
- Fitted the RBF Gaussian Processes.
- Generated 100,000 candidate points and predicted mean outputs and uncertainty.
- Used **UCB (κ = 2.5)** to select points for Functions 2, 3, 4, 5, 7 and 8.
- Used deliberate exploration for Functions 1 and 6 because their GP/UCB surfaces were effectively flat.
- Sense-checked the proposed points against the current best observations.

