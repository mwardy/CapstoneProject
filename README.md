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

**Initial Data Summary**
- The initial datasets varied in size, from 10 observations for the 2D functions to 40 observations for the 8D function.

Function 1: 10 observations, 2 dimensions
Function 2: 10 observations, 2 dimensions
Function 3: 15 observations, 3 dimensions
Function 4: 30 observations, 4 dimensions
Function 5: 20 observations, 4 dimensions
Function 6: 20 observations, 5 dimensions
Function 7: 30 observations, 6 dimensions
Function 8: 40 observations, 8 dimensions

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

**New Output Values**

Function 1:	0
Function 2:	-0.08204067029832242
Function 3:	-0.02016690747473939
Function 4:	-0.3006446531358864
Function 5:	2329.015853969669
Function 6:	-2.661555742668831
Function 7:	1.9506122834052728
Function 8:	9.821986472796599

**Methodology**

1. Add the eight new observations to each function's dataset and compare them with the previous best outputs.
2. Evaluate Week 1's strategy — identify where UCB worked well (especially 3, 4, 5, 7 and 8) and where it didn't (1, 2 and 6).
3. Refit the Gaussian Processes with the additional observations and check whether the RBF behaviour has improved, particularly the length-scale/prediction issues for Functions 1 and 6.
4. Test an alternative kernel where justified — particularly Functions 1 and 6 — and compare its behaviour with our RBF baseline rather than automatically replacing RBF.
5. Adapt exploration vs exploitation by function. Strong new results such as Functions 4 and 5 give us a reason to investigate promising regions more closely, whereas poorly understood functions may still warrant exploration.
6. Generate and assess the second-round queries, using UCB where the GP is behaving sensibly and an appropriate alternative where it isn't.

**Step 1. Compare to previous best outputs**
The new observations were compared with the previous maximum outputs to assess the performance of the first-round strategy. Five of the eight queries produced new best observations, with Functions 4 and 5 showing particularly large improvements. Functions 3, 7 and 8 also improved on their previous maxima, suggesting that the UCB approach successfully identified promising areas for several functions. Functions 1, 2 and 6 did not improve. However, Functions 1 and 6 had deliberately used an exploration-focused strategy due to limitations identified in their RBF models, meaning these results still provide useful information about previously unexplored areas of their search spaces.

Function 1:  
Previous best: 0.000000  
New output:    0.000000  
Difference:    -0.000000  
Updated best:  0.000000  
Result: Did not improve  

Function 2:  
Previous best: 0.611205
New output:    -0.082041
Difference:    -0.693246
Updated best:  0.611205
Result: Did not improve

Function 3:  
Previous best: -0.034835
New output:    -0.020167
Difference:    0.014668
Updated best:  -0.020167
Result: NEW BEST

Function 4:  
Previous best: -4.025542
New output:    -0.300645
Difference:    3.724898
Updated best:  -0.300645
Result: NEW BEST

Function 5:  
Previous best: 1088.859618
New output:    2329.015854
Difference:    1240.156236
Updated best:  2329.015854
Result: NEW BEST

Function 6:  
Previous best: -0.714265
New output:    -2.661556
Difference:    -1.947291
Updated best:  -0.714265
Result: Did not improve

Function 7:  
Previous best: 1.364968
New output:    1.950612
Difference:    0.585644
Updated best:  1.950612
Result: NEW BEST

Function 8:  
Previous best: 9.598482
New output:    9.821986
Difference:    0.223504
Updated best:  9.821986
Result: NEW BEST

**Step 2. Refit the RBF Models**
- What has changed now that the GP has one additional observation?

After incorporating the new observations, the RBF Gaussian Processes were refitted to assess whether the additional information changed their representation of each function. Most functions produced broadly similar length scales to the previous round. However, Function 6 changed substantially from a length scale at the lower bound of 1e-05 to 0.481. Although its exploratory query did not improve the maximum output, the new observation appears to have provided useful information that allowed the GP to model relationships across a wider region of the search space. Function 1 remained at the lower bound of 1e-05 and continued to produce a convergence warning, suggesting that the RBF kernel still struggles to represent this function.

Function 1:  
Fitted kernel: RBF(length_scale=1e-05)   

Function 2:  
Fitted kernel: RBF(length_scale=0.0453)  

Function 3:  
Fitted kernel: RBF(length_scale=0.202)  

Function 4:  
Fitted kernel: RBF(length_scale=0.522)  

Function 5:  
Fitted kernel: RBF(length_scale=0.225)  

Function 6:  
Fitted kernel: RBF(length_scale=0.481)  

Function 7:  
Fitted kernel: RBF(length_scale=0.354)  

Function 8:  
Fitted kernel: RBF(length_scale=0.913)  

Based on these results, the RBF model will continue to be used for Functions 2–8, as the fitted length scales now appear suitable for distinguishing between candidate points. Function 6 can therefore return to the standard UCB approach following the improvement in its model. Function 1 will be treated separately, as its length scale remains at the lower bound despite the additional exploratory observation. An alternative kernel, such as Matérn, will therefore be tested against RBF to determine whether it provides a more informative model before selecting the next query point.

**Step 3. Compare RBF and Matérn for Function 1**

RBF kernel:  
RBF(length_scale=1e-05)  

Matérn kernel:  
Matern(length_scale=1e-05, nu=2.5)  

The Matérn kernel did not improve the fit for Function 1, as its optimised length scale also reached the lower bound of 1e-05. This suggests that the modelling difficulty is not simply caused by the smoothness assumption of the RBF kernel, but by the limited structure available in the current observations. As neither kernel provided a clearly more informative fit, Function 1 will continue to be treated cautiously and may require further exploration before model-driven query selection becomes reliable.

**Step 3b. Compare RBF vs Matérn predictions**

As both the RBF and Matérn kernels produced the same extremely small length scale for Function 1, their predictions will now be compared across 100,000 candidate points. The aim is to determine whether either model can meaningfully distinguish between unexplored areas by examining the variation in predicted outputs and uncertainty. If both models continue to produce almost identical predictions across the search space, this would suggest that neither currently provides reliable guidance for UCB, and a more exploration-focused query will be used instead.

RBF  
Predicted output range: -0.0003278239 to -0.0000001931  
Uncertainty range:      0.0000355799 to 0.0010366701  

MATERN  
Predicted output range: -0.0003278239 to -0.0000003215  
Uncertainty range:      0.0000459032 to 0.0010366701  

The predictions from the RBF and Matérn models were compared across 100,000 candidate points. Both produced very similar predicted output and uncertainty ranges, indicating that changing the kernel did not materially improve the model's ability to distinguish between candidate regions. As neither model currently provides a clear advantage for Function 1, a further exploration-based query will be used to gather information from a poorly sampled area of the search space rather than relying on UCB to exploit the current predictions.

**Step 4. Generate candidate points and use UCB**

For Functions 2–8, 100,000 new candidate points will be generated and evaluated using the updated RBF Gaussian Process models. UCB will again be used to combine predicted output and uncertainty, allowing the queries to balance exploitation of promising regions with exploration of uncertain areas. The exploration parameter κ will initially remain at 2.5, providing consistency with the first round while allowing the effect of the additional observations and updated models to be assessed before making further changes. Function 6 can now return to this UCB approach following the improvement in its fitted RBF model, while Function 1 will remain separate and use an exploration-based strategy because neither RBF nor Matérn provided sufficiently informative predictions.

Function 2:  
Selected point: [0.65528565 0.92989635]  
Predicted output: 0.44107070745260113  
Uncertainty: 0.19003228777907213  
UCB: 0.9161514269002815  

Function 3:  
Selected point: [0.39752993 0.09047714 0.44731895]  
Predicted output: -0.05580113542024088  
Uncertainty: 0.07430259032309242  
UCB: 0.12995534038749015  

Function 4:  
Selected point: [0.48075601 0.47400033 0.22905919 0.42917843]  
Predicted output: -1.666506404136829  
Uncertainty: 0.8868534178318199  
UCB: 0.550627140442721   

Function 5:  
Selected point: [0.51484244 0.85610509 0.96150603 0.99849219]  
Predicted output: 2112.903795601483  
Uncertainty: 303.2913666553336  
UCB: 2871.132212239817   

Function 6:  
Selected point: [0.24654859 0.34464495 0.47952502 0.733976   0.02533225]  
Predicted output: -0.6645463641633735  
Uncertainty: 0.3554831146702006  
UCB: 0.224161422512128  

Function 7:  
Selected point: [0.08033361 0.31746516 0.48551741 0.0555686  0.40688258 0.93197557]  
Predicted output: 1.7729655691895485  
Uncertainty: 0.24096833783176155  
UCB: 2.375386413768952  

Function 8:  
Selected point: [0.07812669 0.66871085 0.05418798 0.20948756 0.86355186 0.7177987 0.0744394  0.86629963]  
Predicted output: 9.712587226001919  
Uncertainty: 0.4625134992197021  
UCB: 10.868870974051173  

Using the updated models, UCB produced different balances of predicted performance and uncertainty across the functions. Functions 5 and 7 selected points with particularly strong predicted outputs following their successful first-round queries, suggesting greater exploitation of the promising regions identified previously. Other functions, such as Function 4, received a larger contribution from uncertainty, indicating a greater exploratory element. Function 6 also now produced a meaningful UCB selection following the improvement in its fitted RBF model. Before finalising the queries, the proposed points will be compared with the current best observations to better understand the balance between exploration and exploitation and determine whether any adjustment to the strategy is justified.

**Step 4. Compare proposed queries with current best points**

The proposed UCB queries were compared with the current best-known observations using Euclidean distance. This provides an additional indication of whether each query is exploiting an established high-performing region or exploring a different area of the search space. Distance is not treated as a fixed measure of exploration or exploitation, but is considered alongside predicted output and model uncertainty when assessing each proposed query.

Function 2:  
Current best point: [0.702637 0.926564]  
Current best output: 0.611205  

Proposed new point: [0.655286 0.929896]  
Distance from current best: 0.0475  

Function 3:  
Current best point: [0.409499 0.399    0.519487]  
Current best output: -0.020167  

Proposed new point: [0.39753  0.090477 0.447319]  
Distance from current best: 0.3171  

Function 4:  
Current best point: [0.399157 0.433203 0.357699 0.461233]  
Current best output: -0.300645  

Proposed new point: [0.480756 0.474    0.229059 0.429178]  
Distance from current best: 0.1609   

Function 5:  
Current best point: [0.352268 0.854957 0.92921  0.999541]  
Current best output: 2329.015854  

Proposed new point: [0.514842 0.856105 0.961506 0.998492]  
Distance from current best: 0.1658    

Function 6:  
Current best point: [0.728186 0.154693 0.732552 0.693997 0.056401]  
Current best output: -0.714265  

Proposed new point: [0.246549 0.344645 0.479525 0.733976 0.025332]  
Distance from current best: 0.5785  

Function 7:  
Current best point: [0.120627 0.403921 0.379973 0.146533 0.335853 0.763305]  
Current best output: 1.950612  

Proposed new point: [0.080334 0.317465 0.485517 0.055569 0.406883 0.931976]  
Distance from current best: 0.249  

Function 8:  
Current best point: [0.243397 0.036115 0.078358 0.053721 0.783854 0.278393 0.034947 0.429032]  
Current best output: 9.821986  

Proposed new point: [0.078127 0.668711 0.054188 0.209488 0.863552 0.717799 0.074439 0.8663  ]  
Distance from current best: 0.919  

The distance comparison showed that the updated UCB strategy is naturally producing different exploration–exploitation behaviour across the functions. Function 2 selected a point extremely close to its current best observation, while Functions 4, 5 and 7 also remained relatively close to promising regions. In contrast, Functions 6 and 8 selected substantially more distant points, indicating greater exploration of their search spaces. As the same κ value of 2.5 is already producing different behaviour according to each model's predictions and uncertainty, it will be retained for this round rather than introducing function-specific values unnecessarily.

**Step 5. Select an exploration point for Function 1**

As neither the RBF nor Matérn model provided sufficiently informative predictions for Function 1, an exploration-based approach was used instead of UCB. The distance between each candidate point and its nearest existing observation was calculated, and the candidate with the greatest minimum distance was selected. This targets a poorly sampled region of the search space, allowing the next observation to provide new information that may improve the Gaussian Process in future rounds.

Function 1 - Exploration  
Selected point: [0.99997255 0.99984253]  
Distance to nearest existing observation: 0.3456413874867929  

The maximum-distance approach selected a point near the upper-right boundary of Function 1's search space, with a distance of 0.346 from its nearest existing observation. This confirms that the query targets a relatively poorly sampled region rather than an area already represented in the data. As the GP models currently provide limited guidance for Function 1, this point was retained as an information-gathering query that may help improve the model in subsequent rounds.

**Step 6 - Final Check**

Before finalising the second-round queries, each selected point was checked to ensure that all input values remained within the permitted range and that no query duplicated an existing observation. The Function 1 query was also verified as the maximum-distance exploratory candidate, while the remaining functions had already been selected as the highest-UCB candidates from their respective candidate sets. This final check was used to confirm that the proposed queries were valid before submission.

Function 1:  
Selected point: [0.999973 0.999843]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.345641  
Maximum-distance selection correct: True   

Function 2:   
Selected point: [0.655286 0.929896]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.047468  
Selection method: UCB  

Function 3:  
Selected point: [0.39753  0.090477 0.447319]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.291689  
Selection method: UCB  

Function 4:  
Selected point: [0.480756 0.474    0.229059 0.429178]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.160930  
Selection method: UCB  

Function 5:  
Selected point: [0.514842 0.856105 0.961506 0.998492]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.165759  
Selection method: UCB  

Function 6:  
Selected point: [0.246549 0.344645 0.479525 0.733976 0.025332]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.562812  
Selection method: UCB  

Function 7:  
Selected point: [0.080334 0.317465 0.485517 0.055569 0.406883 0.931976]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.249013  
Selection method: UCB  
 
Function 8:  
Selected point: [0.078127 0.668711 0.054188 0.209488 0.863552 0.717799 0.074439 0.8663  ]  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.774658  
Selection method: UCB  

The final validation confirmed that all eight proposed queries were within the permitted input range and did not duplicate any existing observations. Function 1 was confirmed as the maximum-distance exploratory candidate, while Functions 2–8 were selected using UCB from their updated RBF models. The distance checks also showed different behaviour between the functions, with Functions 2, 4, 5 and 7 remaining relatively close to existing observations, while Functions 6 and 8 moved further into less explored regions. This supports the decision to retain κ = 2.5, as the same UCB setting was already producing a mixture of exploration and exploitation depending on the information available for each function.


**Module 14 (Week 3)**

**New Output Values**

Function 1:	1.868495640628433e-192
Function 2:	0.36601049236039945
Function 3:	-0.06086980027965498
Function 4:	-3.743172382958963
Function 5:	2765.8197868846855
Function 6:	-0.45851061638506707
Function 7:	0.975232656186371
Function 8:	9.62112231085

**Methodology**

1. Add and assess the new observations
The outputs returned from the second-round queries will be added to the existing datasets and compared with the previous best values. This will show which queries improved the known maximum and provide evidence of how successful the previous exploration and exploitation decisions were.

2. Refit and evaluate the Gaussian Process models
The Gaussian Processes will be refitted using the expanded datasets. Changes in fitted length scales, predictions and uncertainty will be examined to determine whether the additional observations have improved the models or revealed any continuing limitations. Particular attention will again be given to Function 1.

3. Introduce SVMs and investigate high- and low-performing regions
As SVMs are introduced in this stage of the course, they will be tested as an additional way of understanding the functions. The existing observations can be divided into relatively high- and low-performing outputs, allowing an SVM to test whether these regions can be distinguished from their input characteristics.

4. Compare linear and nonlinear SVM approaches
A linear SVM can be compared with a kernel SVM to investigate whether the higher-performing observations appear to be separated by a simple boundary or whether nonlinear relationships are present. The aim is to determine whether SVMs provide useful additional information for query selection rather than automatically replacing the existing optimisation approach.

5. Reassess exploration versus exploitation
The performance of the previous queries, updated GP predictions, model uncertainty and any useful evidence from the SVM analysis will be considered together. Functions with clearly promising regions may justify further exploitation, while poorly understood functions may continue to benefit from exploration.

6. Select the third-round queries
The existing GP + UCB approach will remain the baseline where it continues to behave effectively. SVM results or alternative strategies will only influence the query selection where there is evidence that they provide an improvement over the existing approach.

7. Validate and submit the final queries
The eight proposed points will be checked to ensure that they fall within the permitted [0,1] range, do not duplicate previous observations and reflect the intended strategy. They will then be formatted to six decimal places for submission.

**Step 1. Add and assess the new observations**

The outputs returned from the second-round queries were added to the existing observations for each function. These were compared with the previous best outputs to assess how successful the previous query strategy had been. This also provides the expanded dataset that will be used to refit the models for the third round.

Function 1:   
Previous best: 0.000000  
New output:    0.000000  
Difference:    -0.000000  
Updated best:  0.000000  
Result: Did not improve  

Function 2:   
Previous best: 0.611205  
New output:    0.366010  
Difference:    -0.245195  
Updated best:  0.611205  
Result: Did not improve  

Function 3:  
Previous best: -0.020167  
New output:    -0.060870  
Difference:    -0.040703  
Updated best:  -0.020167  
Result: Did not improve  

Function 4:  
Previous best: -0.300645  
New output:    -3.743172  
Difference:    -3.442528  
Updated best:  -0.300645  
Result: Did not improve  

Function 5:  
Previous best: 2329.015854  
New output:    2765.819787  
Difference:    436.803933  
Updated best:  2765.819787  
Result: NEW BEST  

Function 6:  
Previous best: -0.714265  
New output:    -0.458511  
Difference:    0.255754  
Updated best:  -0.458511  
Result: NEW BEST  

Function 7:  
Previous best: 1.950612  
New output:    0.975233  
Difference:    -0.975380  
Updated best:  1.950612  
Result: Did not improve  
 
Function 8:  
Previous best: 9.821986  
New output:    9.621122  
Difference:    -0.200864  
Updated best:  9.821986  
Result: Did not improve  

The latest queries produced new best outputs for Functions 5 and 6. Function 5 improved from 2329.02 to 2765.82, providing further evidence that the high-performing region identified in previous rounds remains promising. Function 6 improved from -0.7143 to -0.4585, suggesting that returning to the GP/UCB approach after its earlier modelling issues was successful. The remaining functions did not improve their previous best values, although these observations still provide additional information about their response surfaces and can be used when refitting the models.

Function 6:  
Week 1: RBF model problematic → deliberate exploration  
Week 2: extra observation improved the GP → returned to UCB  
Week 3 result: UCB query produced a new best  

**Step 2. Refit and evaluate the Gaussian Process models**

The Gaussian Process models were refitted using the expanded datasets containing all observations collected so far. The fitted RBF length scales were then examined to assess how the models had changed following the latest queries and to identify any continuing modelling issues. Particular attention was given to Function 1, which previously produced an extremely small length scale and uninformative predictions.

Function 1:   
Fitted kernel: RBF(length_scale=1e-05)  

Function 2:   
Fitted kernel: RBF(length_scale=0.0427)  

Function 3:  
Fitted kernel: RBF(length_scale=0.202)  

Function 4:  
Fitted kernel: RBF(length_scale=0.495)  

Function 5:  
Fitted kernel: RBF(length_scale=0.413)  

Function 6:  
Fitted kernel: RBF(length_scale=1e-05)  

Function 7:  
Fitted kernel: RBF(length_scale=0.244)   
  
Function 8:   
Fitted kernel: RBF(length_scale=0.931)  

After refitting the RBF Gaussian Processes, most functions continued to produce stable length scales. Function 5 increased from approximately 0.225 to 0.413 following another strong observation, while Functions 2, 3, 4, 7 and 8 remained broadly similar to the previous round. Function 1 continued to hit the lower length-scale bound of 1e-05, indicating that its modelling issue remains unresolved. Function 6 also returned to the lower bound after previously improving to 0.481, suggesting that the latest observation materially changed the fitted relationship and that the RBF model may again be struggling to represent this function reliably.

**Step 2b. Check predictions and uncertainty**

As Functions 1 and 6 both reached the lower RBF length-scale bound, their predictions were examined across 100,000 randomly generated candidate points. The range of predicted outputs and uncertainty was assessed to determine whether the Gaussian Processes could meaningfully distinguish between different regions of the search space. If predictions and uncertainty remain almost constant across candidates, UCB would provide little useful guidance for selecting the next query.

Function 1:  
Predicted mean minimum: -0.0003005052  
Predicted mean maximum: -0.0000001770  
Predicted mean range:   0.0003003282  

Uncertainty minimum:    0.0000342068  
Uncertainty maximum:    0.0009966631  
Uncertainty range:      0.0009624562  

Function 6:
Predicted mean minimum: -1.5012665979  
Predicted mean maximum: -1.5012665979  
Predicted mean range:   0.0000000000  

Uncertainty minimum:    0.5421470063  
Uncertainty maximum:    0.5421470063  
Uncertainty range:      0.0000000000  

The diagnostic showed different behaviour for the two functions despite both reaching the lower RBF length-scale bound. Function 1 now showed some variation in both predicted output and uncertainty across the candidate space, suggesting that the model can distinguish between at least some regions, although the boundary warning remains a concern. In contrast, Function 6 produced identical predicted means and uncertainty across all 100,000 candidates. This means that UCB cannot meaningfully rank unexplored points for Function 6, so the current RBF model should not be used directly for its next query.

**Step 3. Introduce SVMs**

SVMs were introduced as an additional method for investigating the structure of the functions and identifying whether the input variables can distinguish between relatively high- and low-performing regions. As each function produces a continuous output rather than a predefined class, the observations will first be converted into a simple binary classification problem. For each function, the median observed output will be used as the threshold: observations above the median will be classified as high-performing, while observations at or below the median will be classified as low-performing. The median was chosen because it provides a simple function-specific threshold and should produce relatively balanced classes, which is particularly useful given the limited number of observations available.

A linear SVM and an RBF-kernel SVM will then be fitted to these classes. The linear model will test whether the higher- and lower-performing observations can be separated using a relatively simple linear boundary, while the RBF model allows a nonlinear boundary to be learned. Their performance will be compared using three-fold cross-validation so that the limited observations can be used more efficiently than with a single train/test split. If the RBF SVM performs noticeably better, this may provide evidence that the distinction between higher- and lower-performing regions is nonlinear.

This analysis will be treated as an additional diagnostic rather than a replacement for Bayesian Optimisation. The median threshold only describes relative performance within the observations collected so far and does not represent the true maximum or a known boundary in the underlying function. Similarly, SVM classification does not provide the predictive uncertainty used by UCB. The results will therefore be considered alongside the Gaussian Process models, previous query performance and exploration–exploitation behaviour when deciding whether they provide useful information for the next query.

**Step 3a. Create high/low classes**

Function 1:   
Median threshold: 0.000000  
Low-performing observations:  6  
High-performing observations: 6  

Function 2:   
Median threshold: 0.229792  
Low-performing observations:  6  
High-performing observations: 6  

Function 3:  
Median threshold: -0.094190  
Low-performing observations:  9  
High-performing observations: 8  

Function 4:  
Median threshold: -15.756741  
Low-performing observations:  16  
High-performing observations: 16  

Function 5:  
Median threshold: 64.431793  
Low-performing observations:  11  
High-performing observations: 11  

Function 6:  
Median threshold: -1.446370  
Low-performing observations:  11  
High-performing observations: 11  

Function 7:  
Median threshold: 0.088196  
Low-performing observations:  16  
High-performing observations: 16  
  
Function 8:   
Median threshold: 7.940817  
Low-performing observations:  21  
High-performing observations: 21  

Using the median as the classification threshold produced balanced high- and low-performing groups across all eight functions. Functions with an even number of observations were split equally between the two classes, while Function 3, with 17 observations, was divided into nine lower-performing and eight higher-performing observations. This confirms that the median provides a suitable threshold for the exploratory SVM analysis without creating substantial class imbalance. The resulting classes can now be used to compare whether linear or nonlinear SVM decision boundaries are better able to distinguish relatively high-performing regions.

**3b. Compare linear and RBF SVMs**

Function 1:   
Linear SVM mean accuracy: 0.167
RBF SVM mean accuracy:    0.500

Function 2:   
Linear SVM mean accuracy: 0.667
RBF SVM mean accuracy:    0.833

Function 3:  
Linear SVM mean accuracy: 0.344
RBF SVM mean accuracy:    0.522

Function 4:  
Linear SVM mean accuracy: 0.627
RBF SVM mean accuracy:    0.782

Function 5:  
Linear SVM mean accuracy: 0.649
RBF SVM mean accuracy:    0.458

Function 6:  
Linear SVM mean accuracy: 0.821
RBF SVM mean accuracy:    0.768

Function 7:  
Linear SVM mean accuracy: 0.785
RBF SVM mean accuracy:    0.815
  
Function 8:   
Linear SVM mean accuracy: 0.857
RBF SVM mean accuracy:    0.833

The SVM comparison produced mixed results across the eight functions. The RBF SVM achieved higher cross-validation accuracy for Functions 1, 2, 3, 4 and 7, suggesting that allowing a nonlinear decision boundary may be useful for separating relatively high- and low-performing observations in these functions. The improvement was particularly clear for Functions 2 and 4. However, Function 1 remained difficult to classify, with the RBF model achieving only 0.500 accuracy despite substantially outperforming the linear model.

For Functions 5, 6 and 8, the linear SVM performed better than the RBF alternative. Functions 6 and 8 achieved particularly strong linear accuracies of 0.821 and 0.857 respectively, suggesting that their high- and low-performing observations may contain some relatively simple separable structure. Function 5 also favoured the linear model, although its accuracy of 0.649 was more moderate. Overall, the results show that no single SVM kernel is consistently best across all eight functions and support continuing to consider each function individually.

Overall:
F2: RBF 0.833 vs Linear 0.667 → useful evidence favouring nonlinear separation.  
F4: RBF 0.782 vs Linear 0.627 → similar evidence.  
F6: Linear 0.821 vs RBF 0.768 → interesting, especially because its GP is currently failing.  
F8: Linear 0.857 vs RBF 0.833 → both are strong; not much evidence that nonlinear complexity is necessary.  
F1: neither is convincing.  
F3: neither particularly convincing.  
F5: linear somewhat better, but neither outstanding.  
F7: both reasonably strong and very similar.  

**Step 4. Reassess exploration vs exploitation**

Before selecting the third-round queries, the strategy for each function was reassessed using the results collected so far. This considered whether the previous query improved the known maximum, whether the updated Gaussian Process was producing reliable predictions and uncertainty, and whether the SVM analysis identified useful structure between higher- and lower-performing observations. Functions with reliable models and established promising regions could therefore favour exploitation, while functions with weak or uninformative models could benefit from greater exploration. The aim was not to apply a fixed strategy to every function, but to use the accumulated evidence to determine an appropriate exploration–exploitation balance for each one.

Function-by-function assessment:  
Function 1: Continued exploration is appropriate because the latest query did not improve the best output, the RBF model remains at its lower length-scale bound, and the SVM results provided little useful separation between high- and low-performing observations.  
Function 2: A balanced UCB approach will be retained. The GP remains usable, while the relatively strong RBF SVM accuracy suggests that some nonlinear structure may exist in the higher-performing region.  
Function 3: A balanced UCB approach will continue because the GP remains stable, while neither SVM provided sufficiently strong evidence to justify changing the existing strategy.  
Function 4: UCB will be retained with some emphasis on the previously identified promising region. Although the latest query did not improve the maximum, the previous round produced a substantial improvement and the RBF SVM also showed reasonably strong classification performance.  
Function 5: Greater exploitation of the promising region is appropriate because successive queries have increased the best output from 1088.86 to 2329.02 and then 2765.82. The updated GP also remains usable, providing strong evidence for continuing to search around this region.  
Function 6: The latest query produced a new best output, but the refitted RBF GP now produces identical predictions and uncertainty across candidate points, making UCB unsuitable in its current form. The relatively strong linear SVM accuracy of 0.821 provides an alternative source of information that can be investigated alongside exploration.  
Function 7: A balanced UCB approach will be retained. The latest query did not improve the current maximum, but the GP remains usable and both SVM models showed reasonably good classification performance without a clear advantage for either.  
Function 8: A balanced UCB approach will also be retained. The latest output was relatively close to the current best, the GP remains stable, and both SVM models performed strongly, suggesting useful structure without providing a clear reason to replace the existing optimisation approach.  
  
**Step 5. Select the third-round queries**

The third-round query points were selected using the function-specific strategies identified in Step 4. For Functions 2, 3, 4, 5, 7 and 8, the existing Gaussian Process and UCB approach was retained with κ = 2.5, allowing predicted performance and uncertainty to jointly determine the preferred candidate. Function 1 was treated separately using maximum-distance exploration because its GP remained unreliable and the SVM analysis provided little useful guidance. Function 6 was also treated separately because its refitted GP produced identical predictions and uncertainty across the candidate space, making UCB unable to meaningfully rank candidates. For this function, the SVM results will instead be investigated as an additional source of guidance. All proposed points will be inspected before final selection rather than automatically accepted.

**Step 5a. GP + UCB for Functions 2, 3, 4, 5, 7 and 8**

Function 2:   
Selected point: [0.70962749 0.88195252]  
Predicted output: 0.449335  
Uncertainty:      0.185666  
UCB:              0.913500  

Function 3:  
Selected point: [0.73203545 0.99729984 0.01565321]  
Predicted output: -0.076907  
Uncertainty:      0.077556  
UCB:              0.116983  

Function 4:  
Selected point: [0.3919759  0.42475707 0.49482291 0.46442447]  
Predicted output: -0.064270  
Uncertainty:      0.420976  
UCB:              0.988170  

Function 5:  
Selected point: [0.67162912 0.98111794 0.99045818 0.98518882]  
Predicted output: 2631.682259  
Uncertainty:      240.331386  
UCB:              3232.510725  

Function 7:  
Selected point: [0.02154929 0.31333367 0.36021627 0.14173151 0.29532786 0.68654814]  
Predicted output: 1.662117  
Uncertainty:      0.258204  
UCB:              2.307626  
  
Function 8:   
Selected point: [0.02552911 0.45123759 0.09820892 0.03139148 0.4124209  0.13857332 0.11493429 0.99436062]  
Predicted output: 9.635913  
Uncertainty:      0.434286  
UCB:              10.721628  

Things that stand out:  
- F2 looks quite exploitative. Its proposed point is close in x-space to the existing best [0.702637, 0.926564].  
- F3 looks much more exploratory. UCB is accepting a fairly mediocre predicted mean in exchange for uncertainty.  
- F4 looks particularly interesting: predicted output is −0.064, substantially better than our observed best of −0.301, although there is considerable uncertainty. This looks like a promising UCB candidate.  
- F5 is doing what we hoped. It remains around the high-performing region, with x2–x4 all very high. Its predicted output of 2631.68 is below the observed best of 2765.82, but uncertainty raises its UCB to 3232.51. So we're exploiting the region while still moving into an untested part of it.  
- F7 appears to remain around the general area where we've previously found strong outputs.  
- F8 has relatively high uncertainty, so UCB seems to be giving exploration more weight.

**Step 5b — Compare with the current best points**

Function 2:     
Current best point: [0.702637 0.926564]  
Current best output: 0.611205  

Proposed new point: [0.709627 0.881953]  
Distance from current best: 0.0452  

Function 3:  
Current best point: [0.409499 0.399    0.519487]  
Current best output: -0.020167  

Proposed new point: [0.732035 0.9973   0.015653]  
Distance from current best: 0.8461  

Function 4:  
Current best point: [0.399157 0.433203 0.357699 0.461233]  
Current best output: -0.300645  

Proposed new point: [0.391976 0.424757 0.494823 0.464424]  
Distance from current best: 0.1376  

Function 5:   
Current best point: [0.514842 0.856105 0.961506 0.998492]  
Current best output: 2765.819787  

Proposed new point: [0.671629 0.981118 0.990458 0.985189]  
Distance from current best: 0.2030  

Function 7:  
Current best point: [0.120627 0.403921 0.379973 0.146533 0.335853 0.763305]  
Current best output: 1.950612  

Proposed new point: [0.021549 0.313334 0.360216 0.141732 0.295328 0.686548]  
Distance from current best: 0.1612  
  
Function 8:   
Current best point: [0.243397 0.036115 0.078358 0.053721 0.783854 0.278393 0.034947 0.429032]  
Current best output: 9.821986  

Proposed new point: [0.025529 0.451238 0.098209 0.031391 0.412421 0.138573 0.114934 0.994361]  
Distance from current best: 0.8392  

Useful observations:
- F4 looks very promising. It's only 0.138 from the current best, and the GP predicts −0.064, compared with the observed best of −0.301. There is uncertainty around that prediction, but that's exactly what UCB is designed to account for.
- F5 also passes our check. A distance of 0.203 in four dimensions isn't extremely close, but it's clearly still exploiting the broader high-performing region. In particular, the proposed values for x2–x4 remain very high, consistent with the area that gave us 2329 → 2766.
- F3 and F8 are the exploratory ones. Distances of 0.846 and 0.839 are substantial, but that's not inherently bad — their UCB scores are telling us the uncertainty makes those regions worth testing.

The distances from the current best observations showed that UCB was naturally producing different exploration–exploitation behaviour across the functions. Functions 2, 4 and 7 remained relatively close to their current best points, indicating greater exploitation of known promising regions. Function 5 also remained reasonably close to its latest best observation, which is consistent with the decision to continue exploiting the region that has produced successive improvements. In contrast, Functions 3 and 8 selected substantially more distant points, indicating greater exploration driven by model uncertainty. This variation supports retaining κ = 2.5, as the same acquisition setting is already adapting the balance between exploration and exploitation according to each function's GP predictions and uncertainty.

**Step 5c. Function 1**

Planned strategy was maximum-distance exploration because:  
- RBF is still hitting its lower length-scale bound.  
- The GP remains questionable.  
- Neither SVM provided convincing guidance.  
- Last week's exploratory point didn't find an improvement.

Function 1:  
Selected point: [4.50627487e-05 3.06208459e-03]  
Distance to nearest existing observation: 0.32167082653265755  

For Function 1, maximum-distance exploration selected [0.000045, 0.003062], with a distance of approximately 0.322 from its nearest existing observation. This deliberately targets a relatively unexplored region of the input space rather than relying on the RBF/UCB model, which continued to show modelling issues and was not supported by useful SVM classification performance.

**Step 5d. Function 6: use the SVM**

For Function 6, the current RBF Gaussian Process could not meaningfully rank candidate points because it produced identical predictions and uncertainty across the search space. However, the linear SVM achieved a relatively strong cross-validation accuracy of 0.821, suggesting that it could distinguish between higher- and lower-performing regions reasonably well. A linear SVM was therefore fitted to all available Function 6 observations and used to score 100,000 candidate points. Candidates with higher decision scores are classified more strongly as belonging to the higher-performing region. The highest-scoring candidate will initially be examined alongside its distance from the current best point before deciding whether it provides a sensible next query.

Function 6:  
Selected point: [0.01375812 0.14792993 0.44559686 0.99484501 0.01346948]  
SVM decision score: 5.272052074088469  

Current best point: [0.246549 0.344645 0.479525 0.733976 0.025332]  
Distance from current best: 0.4027814448701828  

The SVM-selected candidate achieved a strong positive decision score of 5.27 and was approximately 0.403 from the current best observation. This suggests that the point provides a degree of exploration while remaining strongly within the region classified by the SVM as higher-performing. It was therefore retained as the proposed query for Function 6.

**Step 6: Final Validation**

Before submission, the eight proposed query points were combined and validated. Each point was checked to ensure that all coordinates remained within the permitted [0,1] range and that no query duplicated an existing observation. The distance from each proposed point to its nearest existing observation was also calculated as a final sense-check on how exploratory or exploitative the selection was. Function 1 was additionally confirmed as the intended maximum-distance exploration point, while the remaining functions were retained according to their selected UCB or SVM-based strategies.

Function 1:   
Selected point: [4.500e-05 3.062e-03]  
Selection method: Maximum-distance exploration  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.321671   
Maximum-distance selection correct: True  

Function 2:     
Selected point: [0.709627 0.881953]   
Selection method: RBF GP + UCB  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.045156  

Function 3:  
Selected point: [0.732035 0.9973   0.015653]  
Selection method: RBF GP + UCB  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.306635  

Function 4:    
Selected point: [0.391976 0.424757 0.494823 0.464424]  
Selection method: RBF GP + UCB  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.137608  

Function 5:    
Selected point: [0.671629 0.981118 0.990458 0.985189]  
Selection method: RBF GP + UCB  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.203041  

Function 6:  
Selected point: [0.013758 0.14793  0.445597 0.994845 0.013469]  
Selection method: Linear SVM  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.402781  

Function 7:  
Selected point: [0.021549 0.313334 0.360216 0.141732 0.295328 0.686548]  
Selection method: RBF GP + UCB  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.161151  
  
Function 8:   
Selected point: [0.025529 0.451238 0.098209 0.031391 0.412421 0.138573 0.114934 0.994361]  
Selection method: RBF GP + UCB  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.800486  

The final validation confirmed that all eight proposed queries were within the permitted input range and did not duplicate any existing observations. Function 1 was confirmed as the intended maximum-distance exploration point, while Functions 2, 3, 4, 5, 7 and 8 retained their UCB-selected candidates and Function 6 retained the SVM-informed candidate. The nearest-observation distances also reflected the intended variation in strategy: Functions 2, 4, 5 and 7 remained relatively close to previously sampled regions, while Functions 1, 3, 6 and particularly 8 provided greater exploration. Overall, the final selections combine exploitation of promising regions with exploration where the available models indicate greater uncertainty or where the GP was unable to provide reliable guidance.

**Final Points For Submission**

Function 1: 0.000045-0.003062  
Function 2: 0.709627-0.881953  
Function 3: 0.732035-0.997300-0.015653  
Function 4: 0.391976-0.424757-0.494823-0.464424  
Function 5: 0.671629-0.981118-0.990458-0.985189  
Function 6: 0.013758-0.147930-0.445597-0.994845-0.013469  
Function 7: 0.021549-0.313334-0.360216-0.141732-0.295328-0.686548  
Function 8: 0.025529-0.451238-0.098209-0.031391-0.412421-0.138573-0.114934-0.994361  

Strategy chosen:  
F1: maximum-distance exploration  
F2: GP + UCB  
F3: GP + UCB  
F4: GP + UCB  
F5: GP + UCB, continuing to exploit the promising region  
F6: linear SVM-informed selection  
F7: GP + UCB  
F8: GP + UCB  









So far:

Week 1: establish GP + UCB baseline.
Week 2: diagnose individual functions and adapt where the GP struggles.
Week 3: introduce SVMs and use them where they genuinely add information.
Week 4: investigate whether different κ values improve the exploration–exploitation balance for individual functions.
