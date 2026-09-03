# Capstone Project
**Capstone project for Imperial AI/ML Course**

**Overall Objective**

The capstone project involves using Bayesian Optimisation to find the maximum values of eight unknown functions. It begins with 10 known data points for each function, which are used to build an initial model and select the first new query point. The functions have different numbers of input dimensions, meaning that some have relatively simple search spaces while others have many more possible combinations of inputs to explore. As the number of dimensions increases, the search space becomes more complex, making it more difficult to identify promising areas from the limited observations available. This means that the balance between exploring uncertain areas and exploiting areas that already appear promising may differ between functions. Each week, one additional data point is received for each function based on the queries submitted, allowing the models to be updated with more information. As the project progresses, the aim is to use this growing dataset to improve the models and make increasingly informed decisions about where to query next. Overall, the project demonstrates how an iterative optimisation approach can gradually improve its understanding of unknown problems of varying complexity and move towards better solutions.

**Hypothetic Methodology**

- Examine the initial data — Check the inputs and outputs for each function, including the number of dimensions, highest and lowest observed outputs, and output range.
- Fit a Gaussian Process — Train an RBF Gaussian Process for each function using the existing input-output observations.
- Generate candidate points — Create a large number of possible new input combinations between 0 and 1 for each function.
- Predict output and uncertainty — Use each GP to calculate the predicted mean (μ) and uncertainty (σ) for the candidate points.
- Calculate UCB — Score each candidate using Upper Confidence Bound, combining predicted performance with uncertainty to balance exploitation and exploration.
- Select and submit the next point — Choose the candidate with the highest UCB for each function, check whether the choice appears sensible, and format each query to six decimal places for submission.
- Reflect and update — Explain the reasoning behind the selections and, once the new outputs are received, add them to the existing data and repeat the optimisation process.

**Module 12 (Week 1)**

**Methodology**

**Step 1. Search for highest and lowest observed output and note the corresponding input**

Function 1:
Highest observed output:  
x1        7.310236e-01  
x2        7.329999e-01  
output    7.710875e-16  
  
Lowest observed output:  
x1        0.650114  
x2        0.681526  
output   -0.003606  

Function 2:  
Highest observed output:  
x1        0.702637  
x2        0.926564  
output    0.611205  

Lowest observed output:  
x1        0.142699  
x2        0.349005  
output   -0.065624  

Function 3:  
Highest observed output:  
x1        0.492581  
x2        0.611593  
x3        0.340176  
output   -0.034835  

Lowest observed output:  
x1        0.151837  
x2        0.439991  
x3        0.990882  
output   -0.398926  

Function 4:  
Highest observed output:  
x1        0.577766  
x2        0.428772  
x3        0.425826  
x4        0.249007  
output   -4.025542  

Lowest observed output:  
x1         0.948389  
x2         0.894513  
x3         0.851638  
x4         0.552196  
output   -32.625660  

Function 5:  
Highest observed output:  
x1           0.224189  
x2           0.846480  
x3           0.879484  
x4           0.878516  
output    1088.859618  

Lowest observed output:  
x1        0.438350  
x2        0.804340  
x3        0.210245  
x4        0.151295  
output    0.112940  

Function 6:  
Highest observed output:  
x1        0.728186  
x2        0.154693  
x3        0.732552  
x4        0.693997  
x5        0.056401  
output   -0.714265  

Lowest observed output:  
x1        0.125720  
x2        0.862725  
x3        0.028544  
x4        0.246605  
x5        0.751206  
output   -2.571170  

Function 7:  
Highest observed output:  
x1        0.057896  
x2        0.491672  
x3        0.247422  
x4        0.218118  
x5        0.420428  
x6        0.730970  
output    1.364968  

Lowest observed output:  
x1        0.879871  
x2        0.397962  
x3        0.003635  
x4        0.956991  
x5        0.264514  
x6        0.114869  
output    0.002701  

Function 8:  
Highest observed output:  
x1        0.056447  
x2        0.065956  
x3        0.022929  
x4        0.038786  
x5        0.403935  
x6        0.801055  
x7        0.488307  
x8        0.893085  
output    9.598482  

Lowest observed output:  
x1        0.984933  
x2        0.699506  
x3        0.998885  
x4        0.180148  
x5        0.580143  
x6        0.231087  
x7        0.490827  
x8        0.313683  
output    5.592193  

**Step 2. Fit RBF Gaussian Processes**

Functions 1, 5 and 6 produced convergence warnings because the optimised RBF length scale reached its lower bound, suggesting the GP was modelling very rapid changes in output; the model predictions and uncertainty will therefore be checked before using them to select new query points.
- If we relax the lower bound, does the model simply push the length scale even closer to zero, or does it settle at a sensible value?

After checking:
- Function 5 behaves much more sensibly once I let the optimiser search a wider range. The GP settled on 0.269, which is a perfectly reasonable length scale, suggesting the original warning was artificial, caused by the optimiser hitting the default limit.
- Functions 1 & 6: Even after allowing much smaller values, both models immediately pushed the length scale down to 1e-10 and produced the same warning again. Telling me the data itself is making the GP believe the function changes extremely rapidly.
- Because of this, I'll keep the original GP models for Functions 1 and 6 rather than chasing an even smaller length scale.

The convergence warnings were investigated by relaxing the RBF length-scale bounds. Function 5 converged to a stable length scale, while Functions 1 and 6 continued to converge to the minimum value, so their original Gaussian Process models were retained for the initial optimisation.

**Step 3. Generate Candidate Points and Predict Mean + Uncertainty**
For each function, 100,000 candidate points were randomly generated across the permitted input space between 0 and 1. The fitted Gaussian Process was then used to estimate the predicted mean output and uncertainty for each candidate. This provides an indication of both how well each unobserved point is expected to perform and how confident the model is in that prediction.

Function 1:  
Dimensions: 2  
Candidate points: 100000  
Predicted output range: -0.000361 to -0.000361  
Uncertainty range: 0.001082 to 0.001082  

Function 2:  
Dimensions: 2  
Candidate points: 100000  
Predicted output range: -0.077317 to 0.643977  
Uncertainty range: 0.000068 to 0.225365  

Function 3:  
Dimensions: 3  
Candidate points: 100000  
Predicted output range: -0.397833 to -0.018176  
Uncertainty range: 0.002777 to 0.084213  

Function 4:  
Dimensions: 4  
Candidate points: 100000  
Predicted output range: -35.682037 to -0.922445  
Uncertainty range: 0.067992 to 5.263611  

Function 5:  
Dimensions: 4  
Candidate points: 100000  
Predicted output range: -71.908033 to 1113.402974  
Uncertainty range: 12.509737 to 245.522395  

Function 6:  
Dimensions: 5  
Candidate points: 100000  
Predicted output range: -1.495390 to -1.495390  
Uncertainty range: 0.449000 to 0.449000  

Function 7:  
Dimensions: 6  
Candidate points: 100000  
Predicted output range: -0.020560 to 1.208211  
Uncertainty range: 0.061869 to 0.302129  

Function 8:  
Dimensions: 8  
Candidate points: 100000  
Predicted output range: 5.286512 to 10.217038  
Uncertainty range: 0.064678 to 0.681040  

**Overview**  
Functions 1 and 6 showed an issue caused by their extremely small RBF length scales. For these functions, the Gaussian Process produced almost identical predictions and uncertainty across all 100,000 candidate points, suggesting that it was treating almost all unobserved points as unrelated to the existing observations. As a result, UCB would be unable to meaningfully distinguish between candidate points for these two functions. Functions 2, 3, 4, 5, 7 and 8 showed variation in both predicted outputs and uncertainty and could therefore proceed normally with UCB.

**Step 4. Calculate UCB**
The predicted outputs and uncertainty were combined using the Upper Confidence Bound (UCB) acquisition function, with κ set to 2.5 to provide a balance between exploitation and exploration. UCB was calculated for each of the 100,000 candidate points, and the point with the highest score was selected as the proposed query. This allows both points with high predicted outputs and uncertain areas with potential for improvement to be considered.

Function 1:  
Selected point: [0.77395605 0.43887844]  
Predicted output: -0.00036060626443627054  
Uncertainty: 0.0010818187933090686  
UCB: 0.002343940718836401  

Function 2:  
Selected point: [0.8217782  0.98030138]  
Predicted output: 0.5373024810236847  
Uncertainty: 0.17485188463025975  
UCB: 0.9744321925993341  

Function 3:  
Selected point: [0.40949935 0.39899996 0.51948667]  
Predicted output: -0.04187059316125945  
Uncertainty: 0.07197438103562513  
UCB: 0.1380653594278034  

Function 4:  
Selected point: [0.39915655 0.43320344 0.35769894 0.461233]  
Predicted output: -1.109357344181845  
Uncertainty: 0.9929487520649243  
UCB: 1.3730145359804657  

Function 5:  
Selected point: [0.35226763 0.85495745 0.92921014 0.99954104]  
Predicted output: 1003.7100993151818  
Uncertainty: 139.19689039784117  
UCB: 1351.7023253097846  

Function 6:  
Selected point: [0.42871117 0.38510403 0.39053256 0.72350556 0.95165422]  
Predicted output: -1.4953899396939465  
Uncertainty: 0.44899959039781934  
UCB: -0.3728909636993982  

Function 7:  
Selected point: [0.12062732 0.40392121 0.37997272 0.14653261 0.33585285 0.76330458]  
Predicted output: 1.136196501448108  
Uncertainty: 0.1932559409633074  
UCB: 1.6193363538563763  

Function 8:   
Selected point: [0.24339702 0.03611478 0.07835798 0.05372058 0.78385373 0.27839289 0.03494745 0.42903195]  
Predicted output: 10.015345100551379  
Uncertainty: 0.5079509989217438  
UCB: 11.285222597855737  

**Overall**
UCB was calculated for each of the 100,000 candidate points using a κ value of 2.5, combining the GP’s predicted output with its uncertainty. The candidate with the highest UCB was selected for each function, allowing points with either strong predicted performance, high uncertainty, or a combination of both to be prioritised. This provided a balance between exploiting promising areas and exploring less certain regions of the search space. Functions 1 and 6 remained problematic because their near-constant predictions and uncertainty resulted in an almost flat UCB surface, meaning their selected points could not yet be considered reliable.

**Step 5. Handling Functions 1 and 6**

For Functions 1 and 6, the RBF Gaussian Process produced extremely small length scales, resulting in almost identical predictions and uncertainty across the candidate search space. As UCB was therefore unable to meaningfully distinguish between candidate points, an exploration-based approach will be used instead. A point will be selected from a relatively unexplored area of the search space to provide the model with new information.

Once the output from this query is received, it will be added to the existing observations and the RBF model reassessed. In future iterations/weeks of the project, an alternative kernel (e.g. Matérn) could also be introduced and compared with RBF to determine whether it provides a more suitable representation of these functions. 

As the GP models for Functions 1 and 6 were unable to identify more promising areas through their predictions or uncertainty, the aim was to maximise the information gained from the next query rather than immediately target a higher output. Selecting a point far from the existing observations reduces duplication of the areas already sampled and provides information about a new part of the search space. This represents a deliberately exploration-focused decision, which can then be reassessed once the new output is received.

Function 1:  
Selected exploration point: [0.00124268 0.99691888]  
Distance to nearest existing observation: 0.39492227534104346  

Function 6:  
Selected exploration point: [0.91049992 0.03922596 0.99959186 0.08178911 0.93948142]  
Distance to nearest existing observation: 0.8860560302850456  

Using this exploration-based approach, Function 1 selected the point [0.001243, 0.996919], which was approximately 0.395 from its nearest existing observation. Function 6 selected [0.910500, 0.039226, 0.999592, 0.081789, 0.939481], approximately 0.886 from its nearest existing observation. These represented the most unexplored candidates identified from the 100,000 points tested. As the GP and UCB could not meaningfully differentiate between candidates for these functions, the aim was to maximise the information gained from the next query rather than immediately exploit an area predicted to have a high output.

**Step 6. Assessing Selected Query Points**

The proposed query points were compared with the current highest-performing observations to assess how the optimisation strategy was navigating each search space. For Functions 2, 3, 4, 5 and 7, the selected points remained relatively close to the current best observations, suggesting that UCB retained an exploitation component by searching around promising regions. Function 8 selected a substantially more distant point, indicating greater exploration within its higher-dimensional search space. Functions 1 and 6 also selected distant points, although this was intentional because an exploration-based strategy had been adopted following the limitations identified with their RBF models.

Function 1:  
Current best point: [0.731024 0.733   ]  
Current best output: 7.710875114502849e-16  
Proposed new point: [0.001243 0.996919]  
Distance from current best: 0.776  

Function 2:  
Current best point: [0.702637 0.926564]  
Current best output: 0.6112052157614438  
Proposed new point: [0.821778 0.980301]  
Distance from current best: 0.1307  

Function 3:  
Current best point: [0.492581 0.611593 0.340176]  
Current best output: -0.034835313350078584  
Proposed new point: [0.409499 0.399    0.519487]  
Distance from current best: 0.2903  

Function 4:  
Current best point: [0.577766 0.428772 0.425826 0.249007]  
Current best output: -4.025542281908162  
Proposed new point: [0.399157 0.433203 0.357699 0.461233]  
Distance from current best: 0.2857  

Function 5:  
Current best point: [0.224189 0.84648  0.879484 0.878516]  
Current best output: 1088.8596181962705  
Proposed new point: [0.352268 0.854957 0.92921  0.999541]  
Distance from current best: 0.1833  

Function 6:  
Current best point: [0.728186 0.154693 0.732552 0.693997 0.056401]  
Current best output: -0.7142649478202404  
Proposed new point: [0.9105   0.039226 0.999592 0.081789 0.939481]  
Distance from current best: 1.1281  

Function 7:  
Current best point: [0.057896 0.491672 0.247422 0.218118 0.420428 0.73097 ]  
Current best output: 1.3649683044991994  
Proposed new point: [0.120627 0.403921 0.379973 0.146533 0.335853 0.763305]  
Distance from current best: 0.2062  

Function 8:  
Current best point: [0.056447 0.065956 0.022929 0.038786 0.403935 0.801055 0.488307 0.893085]  
Current best output: 9.598482002566342  
Proposed new point: [0.243397 0.036115 0.078358 0.053721 0.783854 0.278393 0.034947 0.429032]  
Distance from current best: 0.9368  

**Week 1 Process:**
- Examined the initial observations.
- Fitted the RBF Gaussian Processes.
- Generated 100,000 candidate points and predicted mean + uncertainty.
- Used UCB (κ = 2.5) to select points for Functions 2, 3, 4, 5, 7 and 8.
- Used deliberate exploration for Functions 1 and 6 because their GP/UCB surfaces were effectively flat.
- Sense-checked the proposed points against the current best observations.

**Module 13 (Week 2)**

**Methodology**
