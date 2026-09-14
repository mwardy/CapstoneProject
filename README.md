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
- Function 1: Continued exploration is appropriate because the latest query did not improve the best output, the RBF model remains at its lower length-scale bound, and the SVM results provided little useful separation between high- and low-performing observations.  
- Function 2: A balanced UCB approach will be retained. The GP remains usable, while the relatively strong RBF SVM accuracy suggests that some nonlinear structure may exist in the higher-performing region.  
- Function 3: A balanced UCB approach will continue because the GP remains stable, while neither SVM provided sufficiently strong evidence to justify changing the existing strategy.  
- Function 4: UCB will be retained with some emphasis on the previously identified promising region. Although the latest query did not improve the maximum, the previous round produced a substantial improvement and the RBF SVM also showed reasonably strong classification performance.  
- Function 5: Greater exploitation of the promising region is appropriate because successive queries have increased the best output from 1088.86 to 2329.02 and then 2765.82. The updated GP also remains usable, providing strong evidence for continuing to search around this region.  
- Function 6: The latest query produced a new best output, but the refitted RBF GP now produces identical predictions and uncertainty across candidate points, making UCB unsuitable in its current form. The relatively strong linear SVM accuracy of 0.821 provides an alternative source of information that can be investigated alongside exploration.  
- Function 7: A balanced UCB approach will be retained. The latest query did not improve the current maximum, but the GP remains usable and both SVM models showed reasonably good classification performance without a clear advantage for either.  
- Function 8: A balanced UCB approach will also be retained. The latest output was relatively close to the current best, the GP remains stable, and both SVM models performed strongly, suggesting useful structure without providing a clear reason to replace the existing optimisation approach.  
  
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





**Module 15 - Week 4**

**Methodology**

Step 1 – Update the data and assess results  
- Add the latest input-output pairs to each dataset and compare them with the previous best results. This will show whether the strategies used in the previous round were successful and help determine whether they should be retained or adjusted.  

Step 2 – Refit and diagnose the Gaussian Processes  
- Refit the RBF Gaussian Process models using all available observations and examine their fitted length scales, predictions and uncertainty. Particular attention will be given to Functions 1 and 6, where the GP previously struggled to distinguish meaningfully between candidate points.  

Step 3 – Compare different UCB κ values  
- Rather than continuing with a fixed κ = 2.5, compare κ = 1.5, 2.5 and 3.5 for functions with usable GP models. These represent increasing emphasis on exploration. The resulting candidates will be compared using predicted performance, uncertainty and their location relative to existing observations rather than simply selecting the largest UCB value.  

Step 4 – Select function-specific strategies  
- Use the accumulated results to decide the appropriate exploration–exploitation balance for each function. Functions showing repeated improvement, particularly Function 5, may favour greater exploitation, while poorly understood functions may benefit from further exploration. Previous SVM results will also be considered where useful.  

Step 5 – Compare with Expected Improvement  
- Where appropriate, test Expected Improvement (EI) alongside UCB. EI provides an alternative way of selecting points by considering the expected improvement over the current best result, allowing us to assess whether the acquisition method itself materially changes the proposed query.  

Step 6 – Final selection and validation  
- Select the final eight queries using the combined evidence from the latest results, GP diagnostics, κ comparison and any useful alternative methods. Check that every point is within [0,1], does not duplicate an existing observation and provides a sensible level of exploration or exploitation before submission.  

**Step 1 - Update the data and assess results**

Function 1:   
Previous best: 0.000000
New output:    0.000000
Difference:    -0.000000
Updated best:  0.000000
Result: Did not improve

Function 2:     
Previous best: 0.611205
New output:    0.548608
Difference:    -0.062597
Updated best:  0.611205
Result: Did not improve

Function 3:  
Previous best: -0.020167
New output:    -0.115141
Difference:    -0.094975
Updated best:  -0.020167
Result: Did not improve

Function 4:    
Previous best: -0.300645
New output:    -1.850268
Difference:    -1.549624
Updated best:  -0.300645
Result: Did not improve

Function 5:    
Previous best: 2765.819787
New output:    4595.479318
Difference:    1829.659532
Updated best:  4595.479318
Result: NEW BEST 

Function 6:  
Previous best: -0.458511
New output:    -1.182842
Difference:    -0.724332
Updated best:  -0.458511
Result: Did not improve

Function 7:  
Previous best: 1.950612
New output:    2.139551
Difference:    0.188939
Updated best:  2.139551
Result: NEW BEST
  
Function 8:   
Previous best: 9.821986
New output:    9.645301
Difference:    -0.176685
Updated best:  9.821986
Result: Did not improve

The latest queries produced new best outputs for Functions 5 and 7. Function 5 showed the largest improvement, increasing from 2765.82 to 4595.48 and providing further evidence that the search has identified a particularly strong region of its input space. Function 7 also improved from 1.95 to 2.14. The remaining functions did not improve, although Function 2 remained relatively close to its existing best. The SVM-informed query for Function 6 did not improve on the previous best, suggesting that SVM classification may be useful for identifying broadly higher-performing regions but does not necessarily identify the highest-output points within those regions. These results will be used to further adapt the strategy for the next round.

**Step 2 – Refit and diagnose the Gaussian Processes**

The Gaussian Process models were refitted using all available observations, retaining the RBF kernel used in previous rounds. Ten optimiser restarts were used to improve the likelihood of finding a suitable kernel length scale rather than relying on a single optimisation attempt. The fitted length scales were then examined for each function, as very small values can indicate that the GP is modelling observations as changing extremely rapidly across the input space and may therefore provide limited guidance for unseen candidates. These diagnostics will determine which functions have sufficiently informative GP models to proceed to the UCB κ comparison.

Correction made to initial fitting:  
During the initial Week 4 fitting, the optimiser restart parameter used in previous rounds was accidentally omitted. This resulted in Functions 1–7 all returning the minimum permitted RBF length scale of 1e-05, which was inconsistent with previous model behaviour and suggested that the optimisation had become trapped at the lower bound. The models were therefore refitted using n_restarts_optimizer=10, restoring consistency with the previous methodology before interpreting the results.

Function 1:   
Fitted kernel: RBF(length_scale=1e-05)

Function 2:     
Fitted kernel: RBF(length_scale=0.0516)

Function 3:  
Fitted kernel: RBF(length_scale=0.198)

Function 4:    
Fitted kernel: RBF(length_scale=0.428)

Function 5:    
Fitted kernel: RBF(length_scale=0.367)

Function 6:  
Fitted kernel: RBF(length_scale=1e-05)

Function 7:  
Fitted kernel: RBF(length_scale=0.296)
  
Function 8:   
Fitted kernel: RBF(length_scale=0.949)

Following the corrected fitting process, Functions 2, 3, 4, 5, 7 and 8 produced RBF length scales above the lower bound, ranging from 0.0516 for Function 2 to 0.949 for Function 8. These models therefore appear suitable for further GP-based analysis. Functions 1 and 6 again reached the minimum length scale of 1e-05, consistent with issues observed in previous rounds. This suggests that their current RBF models may struggle to generalise smoothly between observations, so their predictions should be examined separately before using UCB. Overall, the results support proceeding with the planned κ comparison for Functions 2, 3, 4, 5, 7 and 8.

**Step 2b - Diagnostics for F1 and F6**

Before excluding Functions 1 and 6 from the UCB comparison, their GP predictions were examined across a large sample of candidate points. The range of predicted means and uncertainties was calculated to determine whether the models could meaningfully distinguish between different regions of the input space. If predictions remain effectively constant, changing κ would provide little useful information because UCB would be unable to rank candidates meaningfully.

Function 1:   
Predicted mean minimum: -0.0002773894  
Predicted mean maximum: -0.0000000154  
Predicted mean range:   0.0002773740  

Uncertainty minimum:    0.0000101201  
Uncertainty maximum:    0.0009609052  
Uncertainty range:      0.0009507851  

Function 6:     
Predicted mean minimum: -1.4874220585  
Predicted mean maximum: -1.4874220585  
Predicted mean range:   0.0000000000  

Uncertainty minimum:    0.5341917888  
Uncertainty maximum:    0.5341917888  
Uncertainty range:      0.0000000000  

The additional diagnostics confirmed that Functions 1 and 6 should not currently rely on standard GP/UCB selection. Function 6 produced completely identical predicted means and uncertainty across all 100,000 candidate points, meaning that its GP cannot distinguish between different regions of the search space and changing κ would have no effect on candidate ranking. Function 1 showed some variation in both predictions and uncertainty, but its RBF length scale remained at the minimum bound and the predicted outputs were extremely small. Given this persistent instability across multiple rounds, its GP predictions will continue to be treated cautiously. Functions 2, 3, 4, 5, 7 and 8 will therefore proceed to the κ comparison, while alternative strategies will be considered separately for Functions 1 and 6.

Position going into Step 3:  
- F1: GP still problematic → deal with separately.  
- F2: usable GP → κ comparison.  
- F3: usable GP → κ comparison.  
- F4: usable GP → κ comparison.  
- F5: usable GP + repeated major improvements → κ comparison, with a strong reason to consider lower κ/exploitation.  
- F6: GP completely flat → κ comparison would be pointless; deal with separately.  
- F7: usable GP + latest query produced a new best → κ comparison.  
- F8: usable and very stable GP → κ comparison.

**Step 3 - Compare different UCB κ values**

In previous rounds, UCB used a fixed κ value of 2.5 for all functions. This week, κ values of 1.5, 2.5 and 3.5 were compared for Functions 2, 3, 4, 5, 7 and 8, where the GP models remained informative. Lower κ places greater emphasis on the predicted mean and therefore exploitation, while higher κ gives greater weight to uncertainty and exploration. For a fair comparison, the same 100,000 randomly generated candidate points were evaluated under all three κ values for each function. The selected candidates were compared using their predicted output, uncertainty and distance from the current best and nearest existing observations.

**Step 3a – Generate and compare the candidates**

Function 2:     
Current best output: 0.611205  

KAPPA = 1.5  
Selected point: [0.73856  0.912272]  
Predicted output:               0.556031  
Uncertainty:                    0.121843  
UCB:                            0.738795  
Distance from current best:     0.0387  
Distance from nearest existing: 0.0387  

KAPPA = 2.5  
Selected point: [0.688986 0.063026]  
Predicted output:               0.375397   
Uncertainty:                    0.210671  
UCB:                            0.902075  
Distance from current best:     0.8636  
Distance from nearest existing: 0.0652  

KAPPA = 3.5  
Selected point: [0.725891 0.170323]  
Predicted output:               0.342377  
Uncertainty:                    0.221941  
UCB:                            1.119170  
Distance from current best:     0.7566  
Distance from nearest existing: 0.0759  


Function 3:    
Current best output: -0.020167  

KAPPA = 1.5  
Selected point: [0.41795  0.732195 0.005457]  
Predicted output:               -0.033932  
Uncertainty:                    0.053728  
UCB:                            0.046659  
Distance from current best:     0.6126  
Distance from nearest existing: 0.1921  

KAPPA = 2.5  
Selected point: [0.511158 0.731701 0.617932]  
Predicted output:               -0.075090  
Uncertainty:                    0.074427  
UCB:                            0.110977  
Distance from current best:     0.3615  
Distance from nearest existing: 0.3032  

KAPPA = 3.5  
Selected point: [0.565168 0.760792 0.653023]  
Predicted output:               -0.082998  
Uncertainty:                    0.077189  
UCB:                            0.187163  
Distance from current best:     0.4159  
Distance from nearest existing: 0.3541  



Function 4:     
Current best output: -0.300645  

KAPPA = 1.5  
Selected point: [0.44374  0.493644 0.40979  0.452219]  
Predicted output:               -0.309086  
Uncertainty:                    0.456642  
UCB:                            0.375877  
Distance from current best:     0.0918  
Distance from nearest existing: 0.0918  

KAPPA = 2.5  
Selected point: [0.476265 0.525427 0.417514 0.439754]  
Predicted output:               -0.694545  
Uncertainty:                    0.674163  
UCB:                            0.990863  
Distance from current best:     0.1360  
Distance from nearest existing: 0.1360  

KAPPA = 3.5  
Selected point: [0.047215 0.978118 0.999658 0.02652 ]  
Predicted output:               -18.361781  
Uncertainty:                    6.681320  
UCB:                            5.022841  
Distance from current best:     1.0109  
Distance from nearest existing: 0.5669  


Function 5:      
Current best output: 4595.479318  

KAPPA = 1.5  
Selected point: [0.851064 0.985689 0.919598 0.97933 ]  
Predicted output:               4561.793859  
Uncertainty:                    452.311632  
UCB:                            5240.261307  
Distance from current best:     0.1931  
Distance from nearest existing: 0.1931  

KAPPA = 2.5  
Selected point: [0.994129 0.987119 0.927372 0.967532]  
Predicted output:               4090.674685  
Uncertainty:                    726.333454  
UCB:                            5906.508319  
Distance from current best:     0.3291  
Distance from nearest existing: 0.3291  

KAPPA = 3.5  
Selected point: [0.994129 0.987119 0.927372 0.967532]  
Predicted output:               4090.674685  
Uncertainty:                    726.333454  
UCB:                            6632.841773  
Distance from current best:     0.3291  
Distance from nearest existing: 0.3291  


Function 7:    
Current best output: 2.139551  

KAPPA = 1.5  
Selected point: [0.091435 0.306185 0.316439 0.020166 0.225886 0.61874 ]  
Predicted output:               1.959149  
Uncertainty:                    0.276601  
UCB:                            2.374051  
Distance from current best:     0.1762  
Distance from nearest existing: 0.1762  

KAPPA = 2.5  
Selected point: [0.091435 0.306185 0.316439 0.020166 0.225886 0.61874 ]  
Predicted output:               1.959149  
Uncertainty:                    0.276601  
UCB:                            2.650652  
Distance from current best:     0.1762  
Distance from nearest existing: 0.1762  

KAPPA = 3.5  
Selected point: [0.178516 0.448011 0.378124 0.00201  0.138808 0.62281 ]  
Predicted output:               1.641296  
Uncertainty:                    0.383505  
UCB:                            2.983562  
Distance from current best:     0.3020  
Distance from nearest existing: 0.2911  


Function 8:    
Current best output: 9.821986  

KAPPA = 1.5  
Selected point: [0.137618 0.223807 0.054101 0.335736 0.52861  0.334617 0.211854 0.436061]  
Predicted output:               10.059119  
Uncertainty:                    0.187332  
UCB:                            10.340117  
Distance from current best:     0.4756  
Distance from nearest existing: 0.4756  

KAPPA = 2.5  
Selected point: [0.410732 0.64041  0.007318 0.44934  0.728174 0.071621 0.051404 0.007358]  
Predicted output:               9.551346  
Uncertainty:                    0.451204  
UCB:                            10.679357  
Distance from current best:     0.8824  
Distance from nearest existing: 0.8337  

KAPPA = 3.5  
Selected point: [0.356959 0.085948 0.02191  0.938974 0.856847 0.711437 0.157641 0.027439]  
Predicted output:               9.253277  
Uncertainty:                    0.569759  
UCB:                            11.247434  
Distance from current best:     1.0823  
Distance from nearest existing: 0.8796  


Overview:  
- Function 2: κ = 1.5 strongly exploits the current best region, only 0.039 away and with a predicted output of 0.556. κ = 2.5 and 3.5 move towards substantially more exploratory regions.  
- Function 3: All three settings introduce some exploration. κ = 1.5 has the strongest predicted output (−0.034), while κ = 3.5 accepts a poorer prediction in exchange for greater uncertainty.  
- Function 4: κ = 1.5 remains close to the current best and predicts −0.309, while κ = 3.5 moves dramatically away and has a much poorer predicted mean of −18.36. This illustrates the risk of excessive exploration.  
- Function 5: κ = 1.5 stays closest to the newly identified high-performing region and predicts 4561.79, close to the current best of 4595.48. Higher κ moves further away and sacrifices predicted performance for uncertainty.  
- Function 7: κ = 1.5 and 2.5 select exactly the same point, suggesting the preferred candidate is relatively robust to moderate changes in κ. κ = 3.5 produces a more exploratory alternative.  
- Function 8: There is a particularly clear progression. κ = 1.5 selects a candidate predicted at 10.059, above the current best of 9.822, whereas κ = 2.5 and 3.5 move progressively further away and accept lower predicted means for greater uncertainty.  

The κ comparison showed that changing the exploration weighting materially affected candidate selection for most functions. Lower κ generally favoured candidates with stronger predicted outputs and closer proximity to known high-performing regions, while higher κ increasingly prioritised uncertain regions. This was particularly clear for Functions 4 and 8, where κ = 3.5 selected substantially more distant points with lower predicted outputs but much greater uncertainty. Function 5 also showed a clear trade-off: κ = 1.5 remained closer to the region that produced the latest best output of 4595.48, while higher κ values moved further away in exchange for greater uncertainty. Function 7 was comparatively stable, with κ = 1.5 and 2.5 selecting exactly the same candidate. Overall, the results suggest that using a single fixed κ across all functions may not provide the most appropriate exploration–exploitation balance.


**Step 3b – Further compare the effect of changing κ**

To further assess the effect of κ, the distances between the candidates selected using κ = 1.5, 2.5 and 3.5 were calculated for each function. This shows whether changing κ produces genuinely different areas of the search space or only small variations around the same region. Large distances would suggest that the exploration weighting has a meaningful influence on the next query, while small distances would indicate that candidate selection is relatively robust to the choice of κ.

Function 2:     
Distance κ 1.5 → 2.5: 0.8507  
Distance κ 1.5 → 3.5: 0.7421  
Distance κ 2.5 → 3.5: 0.1135  

Function 3:  
Distance κ 1.5 → 2.5: 0.6195  
Distance κ 1.5 → 3.5: 0.6647  
Distance κ 2.5 → 3.5: 0.0707  

Function 4:    
Distance κ 1.5 → 2.5: 0.0478  
Distance κ 1.5 → 3.5: 0.9597  
Distance κ 2.5 → 3.5: 0.9480  

Function 5:    
Distance κ 1.5 → 2.5: 0.1438  
Distance κ 1.5 → 3.5: 0.1438  
Distance κ 2.5 → 3.5: 0.0000  

Function 7:  
Distance κ 1.5 → 2.5: 0.0000  
Distance κ 1.5 → 3.5: 0.1986  
Distance κ 2.5 → 3.5: 0.1986  
  
Function 8:   
Distance κ 1.5 → 2.5: 0.7627  
Distance κ 1.5 → 3.5: 0.9229  
Distance κ 2.5 → 3.5: 0.9939  

Overview:
- F2: 1.5 is very different from 2.5/3.5, while 2.5 and 3.5 are similar. This looks almost like a choice between an exploitation region and an exploration region.  
- F3: similar pattern — 2.5 and 3.5 are only 0.071 apart, while 1.5 selects somewhere substantially different.  
- F4: 1.5 and 2.5 are extremely similar (0.048), but 3.5 suddenly jumps almost a full unit away. That supports our suspicion that 3.5 may be over-exploring F4.  
- F5: relatively stable overall. Interestingly, 2.5 and 3.5 select exactly the same point, while 1.5 stays somewhat closer to the current best.  
- F7: very stable. 1.5 and 2.5 are identical, and even the 3.5 candidate isn't enormously far away.  
- F8: by far the least stable. All three κ values select substantially different regions. That makes the κ decision particularly consequential for this function.  

The candidate-distance comparison confirmed that the effect of changing κ varies considerably between functions. Functions 2, 3 and 8 showed large movements between the lower and higher κ settings, indicating that the exploration weighting has a substantial influence on where the next query is placed. Function 4 was relatively stable between κ = 1.5 and 2.5 but changed dramatically at κ = 3.5. In contrast, Function 7 selected exactly the same point at κ = 1.5 and 2.5, while Function 5 selected the same point at κ = 2.5 and 3.5. These results suggest that κ tuning is particularly important for some functions, while others are relatively robust across part of the tested range.

**Step 4 – Select function-specific strategies**

The results from the previous rounds, GP diagnostics and κ comparison were combined to select a provisional strategy for each function. Rather than choosing κ solely from the highest UCB value, the decision considered recent query performance, predicted output, uncertainty and the amount of exploration produced by each setting. Functions with evidence of a strong-performing region were given greater emphasis on exploitation, while greater exploration was considered where recent queries had been unsuccessful. Functions 1 and 6 were treated separately because their current GP models were not considered sufficiently reliable for standard UCB selection.

Function-by-function decisions:

- Function 1: The RBF length scale remained at its lower bound of 1e-05, continuing the modelling issues seen in previous rounds. Maximum-distance exploration was therefore retained rather than relying on potentially unreliable GP predictions.  
  
- Function 2: The latest query produced 0.549, relatively close to the current best of 0.611. κ = 1.5 selected a point close to this high-performing region with the strongest predicted output of the three settings, so greater exploitation was preferred.  
  
- Function 3: The latest exploratory query produced −0.115, considerably below the current best of −0.020. κ = 1.5 offered the strongest predicted output while still maintaining some distance from existing observations, providing a more balanced alternative to further strong exploration.  
  
- Function 4: The latest query did not improve the current best. κ = 3.5 produced a very distant candidate with a particularly poor predicted output, while κ = 1.5 remained close to the best-known region and predicted performance close to the existing maximum. κ = 1.5 was therefore preferred.  
  
- Function 5: Successive queries have increased the best output from 1088.86 to 2329.02, 2765.82 and now 4595.48. κ = 1.5 remained closest to this increasingly successful region and predicted an output of 4561.79, providing strong justification for continued exploitation.  
  
- Function 6: The GP continued to produce identical predictions and uncertainty across the candidate space, making UCB unsuitable regardless of κ. The previous SVM-guided query also failed to improve the best output, so neither approach was automatically retained and an alternative strategy will be investigated.  
  
- Function 7: The latest query produced a new best of 2.140. κ = 1.5 and 2.5 selected exactly the same candidate, suggesting that the proposed point was relatively robust to moderate changes in the exploration weighting. κ = 1.5 was provisionally retained.  
  
- Function 8: κ had a particularly large influence on candidate selection. The κ = 1.5 candidate had a predicted output of 10.059, above the current best of 9.822, while higher κ values moved substantially further away with lower predicted outputs. Greater exploitation was therefore preferred following the unsuccessful exploratory query from the previous round.


The function-specific assessment resulted in κ = 1.5 being provisionally preferred for Functions 2, 3, 4, 5, 7 and 8. Although the same value was selected across these functions, the reasons differed: some showed evidence supporting exploitation of known high-performing regions, while others had recently received unsuccessful exploratory queries. Function 5 provided the strongest case for exploitation following three successive improvements, while Function 8's κ = 1.5 candidate was notable for having a predicted output above the current observed maximum. Function 1 retained maximum-distance exploration because of its persistent GP issues. No provisional query was selected for Function 6 because its GP remained completely uninformative and the previous SVM-guided query had not improved the best output.

**Step 5 – Compare with Expected Improvement**

Expected Improvement was used as an alternative acquisition function for Functions 2, 3, 4, 5, 7 and 8. Unlike UCB, which directly combines predicted performance and uncertainty using κ, EI estimates the expected improvement of a candidate over the current best observed output. The same 100,000 candidate points used in the κ comparison were evaluated where possible, allowing the EI-selected candidate to be compared with the provisional κ = 1.5 UCB selection. Predicted output, uncertainty and distances from the current best and existing observations were examined to determine whether EI supported a similar exploration–exploitation strategy. Functions 1 and 6 were excluded because their current GP models were not considered sufficiently reliable for acquisition-based selection.

Function 2:     
EI selected point: [0.730707 0.915532]  
Predicted output:               0.590106  
Uncertainty:                    0.095042  
Expected Improvement:           0.024378  
Distance from current best:     0.0302  
Distance from nearest existing: 0.0302  
Distance from κ=1.5 UCB point:  0.0085  

Function 3:  
EI selected point: [0.433866 0.716516 0.000863]  
Predicted output:               -0.031179  
Uncertainty:                    0.051758  
Expected Improvement:           0.011821  
Distance from current best:     0.6086  
Distance from nearest existing: 0.1788  
Distance from κ=1.5 UCB point:  0.0228  

Function 4:    
EI selected point: [0.44374  0.493644 0.40979  0.452219]  
Predicted output:               -0.309086  
Uncertainty:                    0.456642  
Expected Improvement:           0.173102  
Distance from current best:     0.0918  
Distance from nearest existing: 0.0918  
Distance from κ=1.5 UCB point:  0.0000  

Function 5:    
EI selected point: [0.851064 0.985689 0.919598 0.97933 ]  
Predicted output:               4561.793859  
Uncertainty:                    452.311632  
Expected Improvement:           164.098981  
Distance from current best:     0.1931  
Distance from nearest existing: 0.1931  
Distance from κ=1.5 UCB point:  0.0000  
 
Function 7:  
EI selected point: [0.154752 0.313081 0.341855 0.169387 0.318877 0.633022]  
Predicted output:               2.053815  
Uncertainty:                    0.212603  
Expected Improvement:           0.045405  
Distance from current best:     0.1492  
Distance from nearest existing: 0.1492  
Distance from κ=1.5 UCB point:  0.1893  
  
Function 8:   
EI selected point: [0.14854  0.297539 0.20186  0.328776 0.636564 0.500531 0.197949 0.479426]  
Predicted output:               10.080326  
Uncertainty:                    0.149570  
Expected Improvement:           0.251350  
Distance from current best:     0.5181  
Distance from nearest existing: 0.5181  
Distance from κ=1.5 UCB point:  0.2621   

Function-by-function interpretation:  
- Function 2: EI selected a point only 0.0085 from the κ = 1.5 UCB candidate. Both methods therefore strongly favour the same region close to the current best, providing good evidence for greater exploitation.  
- Function 3: EI and UCB were also very similar, with their candidates only 0.0228 apart. Despite being some distance from the current best, both methods independently identify approximately the same region as promising.  
- Function 4: EI selected exactly the same candidate as κ = 1.5 UCB. This strengthens the case for the more exploitative point and provides further evidence against the much more exploratory κ = 3.5 alternative.  
- Function 5: EI and κ = 1.5 UCB again selected exactly the same point. Combined with three successive improvements in the same broader region, this provides particularly strong support for retaining this candidate.  
- Function 7: EI and UCB differed more substantially, with a distance of 0.1893 between their candidates. EI remained slightly closer to the current best and predicted 2.054, while the UCB candidate had a slightly lower predicted output of 1.959 but greater uncertainty. Both therefore represent relatively local searches, but with different exploration–exploitation balances.  
- Function 8: EI selected a different point from UCB, but importantly predicted an output of 10.080, slightly higher than UCB's 10.059, with lower uncertainty. Both exceed the current observed best of 9.822, suggesting that the GP identifies this broader region as particularly promising. EI may therefore be the stronger candidate to consider for this function.  


The Expected Improvement comparison broadly supported the more exploitative strategy identified through the κ analysis. For Functions 2 and 3, EI selected points very close to the κ = 1.5 UCB candidates, while for Functions 4 and 5 the two acquisition methods selected exactly the same points. This provides additional confidence that these regions offer a favourable balance between predicted performance and uncertainty. Function 7 showed greater disagreement between the methods, although both candidates remained relatively close to the current best region. Function 8 also produced different candidates, but both UCB and EI predicted outputs above the current observed maximum. Overall, the agreement between the two acquisition functions supports the decision to reduce the emphasis on exploration for several functions at this stage of the optimisation.

**Step 5b – Alternative strategies for Functions 1 and 6**

Functions 1 and 6 required separate treatment because their Gaussian Process models were not sufficiently reliable for UCB or EI selection. Function 1 retained maximum-distance exploration, continuing the strategy used in previous rounds. For Function 6, the previous SVM-guided query did not improve the best result and the current GP remained completely flat. Maximum-distance exploration was therefore also tested as a simple model-independent alternative. For both functions, 100,000 candidate points were generated and the candidate with the greatest distance from its nearest existing observation was selected, targeting relatively unexplored areas without relying on unreliable model predictions.

Function 1:  
Maximum-distance exploration point: [0.41743  0.463246]  
Distance from nearest existing: 0.3153  
Distance from current best:      0.4137  
  
Function 6:   
Maximum-distance exploration point: [0.033216 0.058439 0.124012 0.011504 0.042834]  
Distance from nearest existing: 0.8782  
Distance from current best:      0.8810  

Maximum-distance exploration identified new candidates for both functions without relying on their problematic GP models. For Function 1, the selected point was 0.315 from its nearest existing observation, providing further exploration of an under-sampled region. Function 6 produced a much more exploratory candidate, with a nearest-observation distance of 0.878 and a distance of 0.881 from the current best. Given the higher dimensionality of Function 6 and the failure of the previous SVM-guided query to improve performance, this provides an opportunity to collect information from a substantially different region of the search space.

Step 5 summary:  
Overall, Step 5 provided additional support for the provisional UCB choices while also providing model-independent alternatives for the two problematic functions. EI closely agreed with κ = 1.5 UCB for Functions 2–5, including identical selections for Functions 4 and 5, while Functions 7 and 8 showed greater differences between acquisition methods. Maximum-distance exploration provided viable alternatives for Functions 1 and 6, where GP-based acquisition remained unsuitable. These results provide a complete set of candidate strategies to take forward to final selection.

**Step 6 – Final selection and validation**

The final query for each function was selected by combining the evidence from recent performance, GP diagnostics, the κ comparison and Expected Improvement. Where UCB and EI proposed very similar points, preference was given to the candidate with the stronger predicted output while maintaining a sensible level of uncertainty. Functions with repeated improvements were treated more exploitatively, while Functions 1 and 6 retained model-independent exploration because their Gaussian Process models remained unreliable. The final eight points were then checked for valid bounds, duplicate observations and distance from the existing dataset before submission.

By function:
- Function 1 – Maximum-distance exploration: Retained because the RBF GP remains unreliable across successive rounds. The new point explores another poorly sampled part of the space.
- Function 2 – EI: EI and κ = 1.5 UCB selected almost identical regions, but EI predicted a slightly stronger output (0.590 vs 0.556) and remained very close to the current best. EI therefore provides the more exploitative choice.
- Function 3 – EI: EI and κ = 1.5 UCB were again very similar. EI produced the slightly stronger predicted output (−0.031 vs −0.034), so it was preferred.
- Function 4 – EI/UCB agreement: Both methods selected exactly the same candidate, providing strong support for the point.
- Function 5 – EI/UCB agreement: Both methods again selected exactly the same point. Combined with three consecutive improvements, this is the strongest-supported query in the current round.
- Function 6 – Maximum-distance exploration: The GP remained completely flat and the previous SVM-guided query failed to improve the maximum. A model-independent exploratory point was therefore preferred to repeating an unsuccessful model-based strategy.
- Function 7 – EI: The latest round produced a new best. EI remained closer to that successful region and predicted a stronger output than the UCB alternative, so greater exploitation was preferred.
- Function 8 – EI: Both EI and κ = 1.5 UCB predicted improvement over the current best, but EI predicted the slightly higher value (10.080) with lower uncertainty. EI was therefore selected.

Function 1:   
Selected point: [0.41743  0.463246]  
Selection method: Maximum-distance exploration  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.315336  
Distance from current best:      0.413652  

Function 2:     
Selected point: [0.730707 0.915532]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.030161  
Distance from current best:      0.030161  

Function 3:  
Selected point: [0.433866 0.716516 0.000863]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.178778  
Distance from current best:      0.608589  

Function 4:    
Selected point: [0.44374  0.493644 0.40979  0.452219]  
Selection method: EI / UCB agreement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.091845  
Distance from current best:      0.091845  

Function 5:    
Selected point: [0.851064 0.985689 0.919598 0.97933 ]  
Selection method: EI / UCB agreement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.193062  
Distance from current best:      0.193062  

Function 6:  
Selected point: [0.033216 0.058439 0.124012 0.011504 0.042834]  
Selection method: Maximum-distance exploration  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.878156  
Distance from current best:      0.880958  

Function 7:  
Selected point: [0.154752 0.313081 0.341855 0.169387 0.318877 0.633022]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.149214  
Distance from current best:      0.149214  
  
Function 8:   
Selected point: [0.14854  0.297539 0.20186  0.328776 0.636564 0.500531 0.197949 0.479426]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.518070  
Distance from current best:      0.518070  

The final validation confirmed that all eight proposed queries were within the permitted input range and did not duplicate any existing observations. Functions 2, 4, 5 and 7 remained relatively close to their current best points, which is consistent with the greater exploitation emphasis introduced this week. Function 3 and especially Function 8 retained a stronger exploratory element despite being selected through Expected Improvement. Functions 1 and 6 were the most exploratory by design, reflecting the continuing limitations of their Gaussian Process models. Overall, the final selections provide a mixture of exploitation in functions with established promising regions and model-independent exploration where the surrogate models remain unreliable.  
Function 6 represents the highest-risk selection this round because it lies substantially further from existing observations; however, this was a deliberate choice to gather information from a genuinely unexplored region after both the GP and previous SVM-guided approach provided limited useful guidance.


**Module 15 - Week 4**

**Methodology**

The strategy continues the gradual development used throughout the project. After establishing GP + UCB as a baseline, diagnosing problematic functions, introducing SVM classification and then comparing UCB settings with Expected Improvement, this iteration introduces a neural network as an alternative surrogate. Rather than increasing complexity for its own sake, the aim is to test whether a different model can identify useful patterns in the growing datasets and whether its recommendations support or challenge those produced by the Gaussian Process.

Step 1 – Update and evaluate the data
The outputs returned from the fourth query will be paired with the points submitted last week and added to each function's dataset. The new results will be compared with the previous best values to identify which queries improved performance and whether the strategies used last week were successful.

Step 2 – Refit and diagnose the Gaussian Processes
An RBF Gaussian Process will be refitted to each updated dataset. The fitted length scales and variation in predicted mean and uncertainty will be examined to determine whether each GP is providing meaningful information. Particular attention will again be given to Functions 1 and 6, where the GP has previously struggled.

Step 3 – Generate GP-based candidates
For functions with reliable GP models, new candidate points will be generated using the acquisition strategies that performed well in previous rounds. Expected Improvement and a relatively exploitative UCB setting will be compared, using the growing history of results to decide whether each function should favour exploration or exploitation.

Step 4 – Introduce a neural-network surrogate
A simple neural network will be trained for each function as a secondary surrogate model. The network will learn the relationship between the existing input coordinates and observed outputs. The architecture will initially be kept relatively small because the datasets remain limited, reducing unnecessary complexity and the risk of overfitting.

Step 5 – Compare GP and neural-network recommendations
Candidate points suggested by the neural networks will be compared with those identified by the GP-based acquisition functions. Agreement between the models may provide additional confidence in a region, while large differences may indicate uncertainty or that the neural network has not yet generalised reliably. The neural network will therefore support rather than automatically replace the GP.

Step 6 – Function-specific final selection
The final query for each function will be chosen using its individual history, recent performance and model diagnostics. Functions showing repeated improvement around known high-performing regions can be treated more exploitatively, while wider exploration can still be considered where the available models provide weak guidance.

Step 7 – Validate and save
All eight selected points will be checked to ensure that every coordinate lies between 0 and 1, no point duplicates an existing observation, and the distance from previous observations is sensible for the intended strategy. The updated datasets will then be saved ready for the next iteration.

**Step 1. Update and evaluate the new observations**

The eight outputs returned from the previous round were first paired with their corresponding submitted input points and added to the datasets saved at the end of Week 4. Each new result was then compared with the previous maximum for its function. This provides an initial assessment of how successful last week's function-specific strategies were before the surrogate models are refitted.

Function 1:   
Previous best: 0.000000  
New output:    -0.005913  
Difference:    -0.005913  
Updated best:  0.000000  
Result: Did not improve  

Function 2:     
Previous best: 0.611205  
New output:    0.507418  
Difference:    -0.103787  
Updated best:  0.611205  
Result: Did not improve  

Function 3:  
Previous best: -0.020167  
New output:    -0.107206  
Difference:    -0.087039  
Updated best:  -0.020167  
Result: Did not improve  

Function 4:    
Previous best: -0.300645  
New output:    -1.438643  
Difference:    -1.137998  
Updated best:  -0.300645  
Result: Did not improve  
 
Function 5:    
Previous best: 4595.479318  
New output:    4857.449601  
Difference:    261.970283  
Updated best:  4857.449601  
Result: NEW BEST  

Function 6:  
Previous best: -0.458511  
New output:    -2.043880  
Difference:    -1.585369  
Updated best:  -0.458511  
Result: Did not improve  

Function 7:  
Previous best: 2.139551  
New output:    2.536939  
Difference:    0.397388  
Updated best:  2.536939  
Result: NEW BEST  
  
Function 8:   
Previous best: 9.821986  
New output:    9.911249  
Difference:    0.089263  
Updated best:  9.911249  
Result: NEW BEST  

Three of the eight queries produced new best outputs. Function 5 improved from 4595.48 to 4857.45, continuing the strong upward trend seen over recent iterations. Function 7 also improved from 2.14 to 2.54, while Function 8 produced a smaller improvement from 9.82 to 9.91. These results provide encouraging evidence for the more exploitative EI-based strategy used for these functions last week. Functions 2, 3 and 4 did not improve despite also being selected using EI, showing that agreement with the surrogate model does not guarantee improvement. Function 1 again failed to improve under maximum-distance exploration, while the highly exploratory Function 6 query performed substantially worse than its existing best. This suggests that repeating aggressive maximum-distance exploration for Function 6 may not be appropriate in the next round.

**Step 2. Refit and diagnose the GPs**

The Gaussian Process models were refitted using all observations available after adding the latest returned query results. The same RBF kernel and optimisation settings used in previous rounds were retained so that changes in the fitted models could be attributed to the additional data rather than a change in methodology. The fitted length scales were then inspected for each function, with particular attention to Functions 1 and 6, which have previously shown unstable or uninformative GP behaviour.

Function 1:   
Fitted kernel: RBF(length_scale=1e-05)  

Function 2:     
Fitted kernel: RBF(length_scale=0.0494)  

Function 3:  
Fitted kernel: RBF(length_scale=0.173)  

Function 4:    
Fitted kernel: RBF(length_scale=0.414)  

Function 5:    
Fitted kernel: RBF(length_scale=0.467)  

Function 6:  
Fitted kernel: RBF(length_scale=1e-05)  

Function 7:  
Fitted kernel: RBF(length_scale=0.33)  
  
Function 8:   
Fitted kernel: RBF(length_scale=0.979)  

Refitting the Gaussian Processes with the latest observations produced sensible RBF length scales for six of the eight functions. Functions 2–5, 7 and 8 fitted length scales between 0.049 and 0.979, suggesting that the models continue to identify structure in their observed data. Functions 1 and 6 again converged to the lower length-scale bound of 1e-05 and generated convergence warnings. This is consistent with their behaviour in the previous iteration and indicates that the additional observations have not resolved the underlying GP fitting problem. These two functions therefore require further diagnostics before GP-based acquisition functions are used.

**Step 2B. Prediction and uncertainty diagnostics**

Because a fitted length scale alone does not determine whether a Gaussian Process provides useful guidance, the models were next evaluated across a large random candidate set. For each function, the range of predicted means and predictive uncertainty was measured. A model that produces meaningful variation across candidate points can potentially distinguish promising regions, whereas near-identical predictions across the search space would make acquisition-based candidate selection unreliable.

Function 1:   
Predicted mean minimum: -0.0006799102  
Predicted mean maximum: -0.0000000377  
Predicted mean range:   0.0006798725  

Uncertainty minimum:    0.0000181309  
Uncertainty maximum:    0.0017215356  
Uncertainty range:      0.0017034047   

Function 2:     
Predicted mean minimum: -0.0822506899  
Predicted mean maximum: 0.6168811209  
Predicted mean range:   0.6991318109  

Uncertainty minimum:    0.0000031216  
Uncertainty maximum:    0.2374310817  
Uncertainty range:      0.2374279601  

Function 3:  
Predicted mean minimum: -0.3973810066  
Predicted mean maximum: -0.0082630242  
Predicted mean range:   0.3891179823  

Uncertainty minimum:    0.0000008049  
Uncertainty maximum:    0.0779100026  
Uncertainty range:      0.0779091977   

Function 4:    
Predicted mean minimum: -34.4842397129  
Predicted mean maximum: -0.0236981328  
Predicted mean range:   34.4605415801  

Uncertainty minimum:    0.0000826978  
Uncertainty maximum:    7.3165011327  
Uncertainty range:      7.3164184349  

Function 5:    
Predicted mean minimum: -1205.5152412033  
Predicted mean maximum: 5058.7148250304  
Predicted mean range:   6264.2300662337  

Uncertainty minimum:    0.0139380635  
Uncertainty maximum:    1228.8268239542  
Uncertainty range:      1228.8128858907  

Function 6:  
Predicted mean minimum: -2.6603226817  
Predicted mean maximum: -1.5106078009  
Predicted mean range:   1.1497148808  

Uncertainty minimum:    0.0247412070  
Uncertainty maximum:    0.5346355207  
Uncertainty range:      0.5098943138  

Function 7:  
Predicted mean minimum: -0.0687460326  
Predicted mean maximum: 2.4187016929  
Predicted mean range:   2.4874477256  

Uncertainty minimum:    0.0000065315  
Uncertainty maximum:    0.6423503787  
Uncertainty range:      0.6423438471   
  
Function 8:   
Predicted mean minimum: 5.1604354842  
Predicted mean maximum: 9.9614819973  
Predicted mean range:   4.8010465131  

Uncertainty minimum:    0.0000106207  
Uncertainty maximum:    0.6690477426  
Uncertainty range:      0.6690371219  

The prediction diagnostics showed meaningful variation in both predicted outputs and uncertainty for all eight functions. Functions 2–5, 7 and 8 continued to provide clear differentiation between candidate points, supporting their use for GP-based optimisation. Function 1 also produced variation despite its length scale remaining at the lower bound, although its unusual GP fit means its predictions should still be treated cautiously. Most importantly, Function 6 no longer produced identical predictions across the search space. Its predicted mean varied by approximately 1.15 and its uncertainty by 0.51, compared with effectively zero variation in previous rounds. This suggests that the latest exploratory observation, although poor in terms of objective value, has provided useful information to the surrogate model. The Function 6 GP should therefore be reconsidered as a possible source of guidance this week rather than automatically excluded.

**Step 3. Compare GP acquisition strategies**

The GP-based candidate selection was simplified this week using the results from the previous iteration. Rather than repeating the full κ comparison, a lower UCB value of κ = 1.5 was retained as the more exploitative setting and compared directly with Expected Improvement. Both acquisition functions were evaluated on the same 100,000 candidate points for each function so that any differences were due to the acquisition strategy rather than random sampling. Functions 2–8 were included, while Function 1 was held back for comparison with alternative methods because of its continuing GP instability.

**Step 3A. Generate EI and UCB candidates**

Function 2:  
UCB selected point: [0.713179 0.112872]  
Predicted output: 0.431873  
EI selected point:  [0.713092 0.138155]  
Predicted output: 0.429437  
Distance between UCB and EI candidates: 0.0253  

Function 3:  
UCB selected point: [0.414712 0.608726 0.505958]  
Predicted output: -0.034992  
EI selected point:  [0.384686 0.478261 0.416295]  
Predicted output: -0.013239  
Distance between UCB and EI candidates: 0.1611  

Function 4:  
UCB selected point: [0.467774 0.324669 0.374707 0.447129]  
Predicted output: -0.365609  
EI selected point:  [0.462451 0.342242 0.380323 0.445636]  
Predicted output: -0.212533  
Distance between UCB and EI candidates: 0.0193  

Function 5:  
UCB selected point: [0.975917 0.983441 0.990513 0.948800]  
Predicted output: 4895.382638  
EI selected point:  [0.938942 0.989452 0.984552 0.971823]  
Predicted output: 5007.171751  
Distance between UCB and EI candidates: 0.0444  

Function 6:  
UCB selected point: [0.982408 0.040400 0.122787 0.464265 0.964442]  
UCB predicted output: -1.510608  
EI selected point:  [0.982408 0.040400 0.122787 0.464265 0.964442]  
EI predicted output: -1.510608  
Distance between UCB and EI candidates: 0.0000  

Function 7:  
UCB selected point: [0.132842 0.294043 0.285122 0.050957 0.281982 0.619768]  
UCB predicted output: 2.427148  
EI selected point:  [0.132842 0.294043 0.285122 0.050957 0.281982 0.619768]  
EI predicted output: 2.427148  
Distance between UCB and EI candidates: 0.0000  

Function 8:  
UCB selected point: [0.303661 0.221402 0.038340 0.552794 0.966939 0.042888 0.140457 0.049640]  
Predicted output: 9.632442  
EI selected point:  [0.289602 0.316966 0.042193 0.143128 0.616869 0.054183 0.288523 0.512239]  
Predicted output: 9.915401  
Distance between UCB and EI candidates: 0.7320  

Comparing Expected Improvement with UCB at κ = 1.5 produced varying levels of agreement across the functions. Functions 2, 4, 5 and 7 produced similar or identical recommendations, suggesting relatively consistent GP guidance. Function 3 showed moderate disagreement, although both methods remained reasonably close to the current best region. Function 8 showed the largest difference, with UCB selecting a substantially more exploratory point while EI remained closer to the successful region identified last week. Function 6 produced identical UCB and EI candidates, but the selected point was very distant from the current best and existing observations. Given the continuing length-scale warning for Function 6, this agreement should not automatically be interpreted as strong evidence for the candidate.

Function-by-function interpretation:
- F2: Very strong agreement. The candidates are only 0.0253 apart. Interestingly, both methods have moved away from the current best into a different region around [0.71, 0.12–0.14]. That looks like a genuine feature of the GP rather than the acquisition function.  
- F3: Moderate disagreement (0.1611). EI is more exploitative, sitting only 0.1325 from the current best and predicting −0.0132, which is actually better than the current observed best of −0.0202. EI looks particularly interesting here.  
- F4: Very strong agreement (0.0193). EI is slightly closer to the current best and predicts −0.213, again better than the observed best of −0.301. Good candidate region.  
- F5: Strong agreement (0.0444) around the region that has repeatedly produced improvements. EI predicts 5007.17, above the new observed best of 4857.45. Given F5's recent history, I'd continue favouring exploitation here.  
- F6: This needs caution. UCB and EI select exactly the same point, which initially looks convincing. However, look at the prediction: −1.510608. That's essentially the GP's background prediction from our diagnostics, and the point is a huge 1.31 away from the current best. This looks much more like an artefact of the unusual GP fit than a genuinely promising optimum. I would not submit this point based on the GP agreement alone.  
- F7: Excellent agreement — literally the same point. It's only 0.1401 from the new best region. Given that last week's EI query produced another improvement, this is currently one of our strongest candidates.  
- F8: Very different recommendations (0.7320 apart). UCB is taking a much bigger exploratory step, whereas EI stays closer to the region that just gave us a new best. Since last week's EI selection succeeded, EI currently looks more compelling, although we'll wait for the NN comparison.  

**Step 3B. Provisional GP strategy by function**

The UCB and Expected Improvement results were then interpreted function by function rather than selecting one acquisition method uniformly. The provisional choice considered agreement between the two methods, predicted output, recent query performance and proximity to known high-performing regions. These selections were not treated as final decisions, because the neural-network surrogate introduced in the next step will provide an independent comparison.

Provisional interpretation:   
- F2: UCB and EI are very close and both point to a different low-\(x_2\) region. Treat this as a credible alternative region worth further comparison.  
- F3: EI currently looks stronger because it predicts -0.013239, above the current best of -0.020167, while remaining closer to the best-known region.  
- F4: Strong agreement between UCB and EI; EI has the stronger predicted mean. Provisional preference: EI.  
- F5: Strong agreement in the same successful region, with EI predicting 5007.17, above the current best 4857.45. Provisional preference: EI.  
- F6: Although UCB and EI agree exactly, the candidate is extremely far from the current best and the GP still has a problematic 1e-05 length scale. No GP candidate should be trusted yet.  
- F7: UCB and EI select exactly the same point close to the new best region. Strong provisional candidate.  
- F8: UCB and EI disagree substantially. Given that last week's EI query produced a new best, EI is provisionally preferred.  

The GP acquisition comparison produced strong provisional candidates for Functions 3, 4, 5, 7 and 8, with Expected Improvement generally favoured where it offered stronger predicted performance or remained closer to recently successful regions. Function 2 showed close agreement between UCB and EI on a distinctly different region of the search space, making it a useful candidate for further comparison. Function 6 remained unresolved because its acquisition methods agreed on a highly exploratory point despite the GP continuing to show an unstable length-scale fit. These provisional decisions will now be compared with an independent neural-network surrogate before final query selection.

**Step 4a: train a small PyTorch network for each function and check whether training looks stable.**

A simple neural-network surrogate was introduced as a secondary modelling approach rather than replacing the Gaussian Process. Because the available datasets are still relatively small, the architecture was deliberately kept compact, using two hidden layers with 16 neurons in each layer and ReLU activation functions. Inputs and outputs were standardised before training so that functions with very different numerical scales could be handled consistently. Five independently initialised networks were trained for each function and their predictions averaged, reducing dependence on a single random initialisation. The resulting ensemble predictions were then used as an additional source of evidence when identifying promising regions of the search space.

Function 1:  
Mean final training loss: 0.001044  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.005221  

Function 2:  
Mean final training loss: 0.002267  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.010432  

Function 3:  
Mean final training loss: 0.000001  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.000003  

Function 4:  
Mean final training loss: 0.000000    
Minimum final training loss: 0.000000  
Maximum final training loss: 0.000001  

Function 5:  
Mean final training loss: 0.000004  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.000016  

Function 6:  
Mean final training loss: 0.000013  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.000034  

Function 7:  
Mean final training loss: 0.000000  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.000002  

Function 8:  
Mean final training loss: 0.000000  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.000000  

The neural networks achieved very low training losses across all eight functions. Functions 3–8 were fitted almost perfectly, while Functions 1 and 2 showed slightly greater variation between the five independently initialised networks. Although this demonstrates that the compact architecture has sufficient flexibility to represent the observed data, the extremely low losses may also indicate overfitting given the relatively small datasets. Training performance alone is therefore insufficient to determine whether the neural networks provide reliable surrogate models, and their ability to generalise to unseen observations will be assessed before using them for candidate selection.

**Step 4B. Test generalisation**

Because the neural networks achieved extremely low training losses, Leave-One-Out Cross-Validation was used to assess whether they could generalise beyond the observations used for training. Each observation was removed in turn, the remaining observations were used to train the network, and the held-out output was predicted. Performance was measured using mean absolute error and compared with a simple baseline that predicted the mean of the remaining training outputs. This provides a more cautious assessment of whether the neural network contains useful predictive information before it is used to propose new queries.

Function 1:  
NN LOOCV MAE:       0.002595  
Mean baseline MAE:  0.001255  
NN beats baseline:  False  

Function 2:  
NN LOOCV MAE:       0.188740  
Mean baseline MAE:  0.225146  
NN beats baseline:  True  

Function 3:  
NN LOOCV MAE:       0.066222  
Mean baseline MAE:  0.046560  
NN beats baseline:  False  

Function 4:  
NN LOOCV MAE:       1.998674  
Mean baseline MAE:  6.651296  
NN beats baseline:  True  

Function 5:  
NN LOOCV MAE:       192.581456  
Mean baseline MAE:  1041.351340  
NN beats baseline:  True  

Function 6:  
NN LOOCV MAE:       0.327649  
Mean baseline MAE:  0.453136  
NN beats baseline:  True  

Function 7:  
NN LOOCV MAE:       0.315686  
Mean baseline MAE:  0.474058  
NN beats baseline:  True  

Function 8:  
NN LOOCV MAE:       0.328345  
Mean baseline MAE:  0.900332  
NN beats baseline:  True  

Leave-One-Out Cross-Validation showed that the neural networks generalised better than the mean baseline for six of the eight functions. The strongest relative improvements were observed for Functions 4, 5 and 8, while Functions 2, 6 and 7 also outperformed the baseline. Functions 1 and 3 did not outperform the baseline, suggesting that their extremely low training losses were not matched by reliable predictions on unseen observations. Neural-network recommendations will therefore be treated as useful secondary evidence for Functions 2, 4, 5, 6, 7 and 8, but will not be relied upon for Functions 1 and 3.

**Step 4C. Generate NN candidates**

The five neural networks trained for each function were next used to evaluate 100,000 randomly generated candidate points. Predictions from the independently initialised networks were converted back to the original output scale and averaged, with the candidate having the highest mean predicted output selected. The standard deviation between network predictions was also recorded as a simple measure of model disagreement. Neural-network candidates were generated for all functions for comparison, although recommendations for Functions 1 and 3 were treated cautiously because their networks did not outperform the baseline during cross-validation.

Function 1:  
NN selected point: [9.78948e-01 6.56000e-04]  
NN predicted output:             0.004556  
NN model disagreement:          0.001415  
Distance from current best:     0.7732  
Distance from nearest existing: 0.2982   

Function 2:  
NN selected point: [9.98493e-01 5.18000e-04]  
NN predicted output:             1.116515  
NN model disagreement:          0.831745  
Distance from current best:     0.9722  
Distance from nearest existing: 0.3549  

Function 3:  
NN selected point: [0.297078 0.002134 0.846217]  
NN predicted output:             0.000985  
NN model disagreement:          0.027674  
Distance from current best:     0.5262  
Distance from nearest existing: 0.3477  

Function 4:  
NN selected point: [0.394245 0.436007 0.403516 0.426489]  
NN predicted output:             -0.570388  
NN model disagreement:          0.191094  
Distance from current best:     0.0578  
Distance from nearest existing: 0.0578  

Function 5:  
NN selected point: [0.937033 0.990945 0.91798  0.998735]  
NN predicted output:             5001.947266  
NN model disagreement:          90.387283  
Distance from current best:     0.0883  
Distance from nearest existing: 0.0883  

Function 6:  
NN selected point: [0.248909 0.369411 0.484746 0.674732 0.011701]  
NN predicted output:             -0.403257  
NN model disagreement:          0.039523  
Distance from current best:     0.0659  
Distance from nearest existing: 0.0659  

Function 7:  
NN selected point: [0.028519 0.317694 0.105297 0.051944 0.24629  0.446805]  
NN predicted output:             2.835739  
NN model disagreement:          0.333867  
Distance from current best:     0.3545  
Distance from nearest existing: 0.3545  

Function 8:  
NN selected point: [0.034135 0.275291 0.032179 0.508682 0.760754 0.61795  0.034127 0.34942 ]  
NN predicted output:             10.356458  
NN model disagreement:          0.134738  
Distance from current best:     0.3843  
Distance from nearest existing: 0.3843  

Function-by-function interpretation:  

Function 1: The neural network selected a relatively exploratory point and predicted an output of 0.00456, which would exceed the current best. However, the network failed to outperform the mean baseline during cross-validation, so this prediction is not considered sufficiently reliable for query selection.  

Function 2: The neural network predicted a potentially large improvement to 1.1165, but the selected point was far from the current best and disagreement between the five networks was very high at 0.8317. Despite the network outperforming the baseline during cross-validation, this individual recommendation therefore appears unreliable and the GP currently provides stronger guidance.  

Function 3: The neural network predicted a small positive output of 0.00099, potentially exceeding the current best of −0.0202. However, the network failed to outperform the baseline during cross-validation, so its recommendation will not be prioritised over the GP-based candidates.  

Function 4: The neural-network candidate was only 0.0578 from the current best point and remained within the same general high-performing region identified by the GP. Although its predicted output of −0.5704 does not exceed the existing best, the agreement in location provides additional support for continued exploitation of this region.  

Function 5: The neural network produced one of the strongest recommendations, selecting a point only 0.0883 from the current best and predicting an output of 5001.95. This is above the existing best of 4857.45 and is in a similar region to the promising EI and UCB candidates, providing independent support for continued exploitation.  

Function 6: The neural network selected a point only 0.0659 from the current best and predicted an improved output of −0.4033, with relatively low disagreement of 0.0395 between networks. As the neural network also outperformed the baseline during cross-validation, this provides a potentially useful alternative to the much more exploratory and less convincing GP recommendation.  

Function 7: The neural network predicted an improvement to 2.8357 but selected a moderately more exploratory point, 0.3545 from the current best, with some disagreement between the individual networks. The recommendation is therefore promising, although the closer agreement between EI and UCB around the recently successful region currently provides stronger evidence.  

Function 8: The neural network selected a moderately exploratory point and predicted an improvement from the current best of 9.9112 to 10.3565. The network performed well relative to the baseline during cross-validation and disagreement between the five networks was relatively modest, making this a credible alternative to the EI recommendation.  

Overview:  
The neural-network candidates varied considerably in both their location and apparent reliability. Functions 4, 5 and 6 produced recommendations close to existing high-performing regions, while Functions 7 and 8 suggested moderately more exploratory points with predicted outputs above their current best values. Function 5 was particularly encouraging, with the neural network independently identifying a similar high-performing region to the GP models and predicting an output of approximately 5002. Function 6 also produced a candidate close to the current best with relatively low disagreement between the five networks, providing a useful alternative to the highly exploratory GP recommendation. In contrast, Functions 1 and 2 produced highly exploratory neural-network candidates, and Function 2 showed particularly high disagreement between networks. Combined with the earlier cross-validation results, these findings demonstrate the importance of using the neural network as supporting evidence rather than automatically selecting its highest predicted point.  

**Step 5. Direct model comparison**

The GP and neural-network recommendations were compared to assess whether the different surrogate approaches identified similar promising regions. Function 1 had initially been excluded from GP-based candidate selection because its RBF length scale remained at the lower bound. However, after its neural network also failed to outperform the baseline during cross-validation, the GP was reconsidered as an additional source of evidence. UCB and Expected Improvement candidates were therefore generated for Function 1 using the same settings as the other functions. For all eight functions, Euclidean distance was then calculated between the NN, UCB and EI recommendations. Model agreement was considered alongside GP diagnostics, NN cross-validation performance and previous query results rather than being treated as sufficient evidence on its own.

Function 1:
UCB (κ = 1.5)
[0.943533 0.359421]
Predicted output: -0.000680

EXPECTED IMPROVEMENT
[0.943533 0.359421]
Predicted output: -0.000680

Distance between UCB and EI candidates: 0.0000

UCB and EI selected exactly the same candidate for Function 1, although the predicted output of −0.000680 remained below the current best. The agreement provides an additional candidate for comparison, but confidence remains limited because the GP continues to exhibit an unstable length-scale fit.

**Step 5B. Compare all three approaches**

Function 1:  
NN → UCB distance: 0.3605  
NN → EI distance:  0.3605  
UCB → EI distance: 0.0000  

Function 2:  
NN → UCB distance: 0.3066  
NN → EI distance:  0.3169  
UCB → EI distance: 0.0253  

Function 3:  
NN → UCB distance: 0.7054  
NN → EI distance:  0.6475  
UCB → EI distance: 0.1611  

Function 4:  
NN → UCB distance: 0.1381  
NN → EI distance:  0.1198  
UCB → EI distance: 0.0193  

Function 5:  
NN → UCB distance: 0.0966  
NN → EI distance:  0.0718  
UCB → EI distance: 0.0444  

Function 6:  
NN → UCB distance: 1.3150  
NN → EI distance:  1.3150  
UCB → EI distance: 0.0000  

Function 7:  
NN → UCB distance: 0.2738  
NN → EI distance:  0.2738  
UCB → EI distance: 0.0000  

Function 8:  
NN → UCB distance: 0.7429  
NN → EI distance:  0.7940  
UCB → EI distance: 0.7320  

Function-by-function interpretation:  

Function 1: UCB and EI selected exactly the same candidate, while the NN recommendation was 0.3605 away. However, both models have reliability concerns: the GP continues to hit its lower length-scale bound and the NN failed to outperform the baseline during cross-validation. The GP agreement therefore provides some additional evidence, but F1 remains a relatively uncertain selection.  

Function 2: UCB and EI were extremely close (0.0253), whereas the NN was approximately 0.31 from both. The NN candidate also showed very high disagreement between its five networks. The GP recommendation therefore provides the stronger evidence for this function.  

Function 3: The NN candidate was substantially separated from both GP recommendations and the NN failed to outperform the baseline during cross-validation. The GP therefore remains the preferred modelling approach, with EI providing the stronger provisional candidate.  

Function 4: All three approaches identified broadly similar regions. UCB and EI were only 0.0193 apart, while the NN was 0.1198 from EI. Combined with the NN's strong cross-validation performance, this provides useful independent support for continued exploitation around the existing high-performing region.  

Function 5: This produced the strongest multi-model agreement. The NN was only 0.0718 from EI and 0.0966 from UCB, with the two GP candidates themselves only 0.0444 apart. Together with the repeated improvements in observed output, this provides strong evidence that the models are converging on a genuinely promising region.  

Function 6: UCB and EI selected the same candidate, but the NN recommendation was 1.3150 away. Given the continuing GP length-scale problem, the poor result from last week's distant exploration and the NN's successful cross-validation performance, the NN recommendation currently appears more credible than the GP agreement.  

Function 7: UCB and EI again selected exactly the same point, while the NN candidate was 0.2738 away. Both models have some supporting evidence, but the GP recommendation remains closer to the recently successful region and therefore currently provides the stronger candidate.  

Function 8: All three approaches differed substantially, with UCB and EI 0.7320 apart and the NN more than 0.74 from both. Unlike some of the other disagreements, the NN performed well in cross-validation and predicted an improvement, while last week's EI selection also successfully produced a new best. There is therefore credible evidence for both EI and the NN, requiring a more careful final choice.  

Summary:  
The comparison showed that agreement between the GP and neural-network approaches varied substantially between functions. Functions 4 and 5 showed the strongest overall agreement, with all three recommendations concentrated within relatively similar regions. Functions 1 and 7 showed complete agreement between UCB and EI but greater separation from the neural network, while Function 2 also showed close agreement between the two GP acquisition methods. Functions 3, 6 and 8 displayed considerably greater disagreement between modelling approaches. Function 6 produced the clearest conflict, with the neural-network candidate 1.3150 from the identical UCB/EI recommendation, while Function 8 showed substantial separation between all three candidates. These results support retaining a function-specific selection strategy rather than applying a single model uniformly across all eight functions.

**Step 6. Final function-specific selection**

Before selecting the final queries, the leading model recommendations were compared with the existing observations surrounding each candidate. For each UCB, EI and neural-network recommendation, the nearest observed points and their outputs were examined. This provides a final check that model predictions are consistent with the available empirical evidence and helps distinguish genuine high-performing regions from potentially unreliable extrapolation. The final selection for each function will therefore combine model recommendations, model reliability, recent query performance and the behaviour of nearby observations.

Function 1:  
UCB/EI nearest observation: Distance 0.1400 | Output 0.000000  
NN nearest observation:     Distance 0.2982 | Output 0.000000  

Function 2:  
UCB nearest observation:    Distance 0.0487 | Output 0.538996  
EI nearest observation:     Distance 0.0494 | Output 0.538996  
NN nearest observation:     Distance 0.3549 | Output 0.538996  

Function 3:  
UCB nearest observation:    Distance 0.1832 | Output -0.034835  
EI nearest observation:     Distance 0.1325 | Output -0.020167  
NN nearest observation:     Distance 0.3477 | Output -0.118048  

Function 4:  
UCB nearest observation:    Distance 0.1303 | Output -0.300645  
EI nearest observation:     Distance 0.1142 | Output -0.300645  
NN nearest observation:     Distance 0.0578 | Output -0.300645  

Function 5:  
UCB nearest observation:    Distance 0.1468 | Output 4857.449601  
EI nearest observation:     Distance 0.1096 | Output 4857.449601  
NN nearest observation:     Distance 0.0883 | Output 4857.449601  

Function 6:  
UCB/EI nearest observation: Distance 0.7451 | Output -1.144785  
NN nearest observation:     Distance 0.0659 | Output -0.458511  

Function 7:  
UCB/EI nearest observation: Distance 0.1401 | Output 2.536939  
NN nearest observation:     Distance 0.3545 | Output 2.536939  

Function 8:  
UCB nearest observation:    Distance 0.7299 | Output 9.821986  
EI nearest observation:     Distance 0.4896 | Output 9.821986  
NN nearest observation:     Distance 0.3843 | Output 9.911249  

Function-by-function interpretation:  

Function 1: The GP candidate was closer to an existing observation than the NN candidate, but both surrogate approaches have reliability concerns. The common UCB/EI candidate was therefore provisionally preferred, although confidence remains low.  

Function 2: Both GP candidates were approximately 0.05 from an observation producing 0.5390, whereas the NN candidate was considerably further away. This reinforces the GP region, with EI provisionally preferred.  

Function 3: EI was only 0.1325 from the current best observation of −0.0202. The NN candidate was further from strong observations and had previously failed cross-validation, strengthening the case for EI.  

Function 4: All three approaches remained close to the current best region. Although the NN candidate was closest, EI moved slightly further from the two recent weaker observations while remaining close to the best, so EI was provisionally preferred.  

Function 5: All three candidates remained close to the sequence of increasingly strong historical observations. Combined with strong agreement between the models, this provides substantial evidence for continued exploitation, with EI provisionally preferred.  

Function 6: The NN candidate was only 0.0659 from the current best observation, compared with 0.7451 for the GP candidate's nearest observation. Combined with the GP fitting problems and successful NN cross-validation, the NN candidate was preferred.  

Function 7: The common UCB/EI candidate was only 0.1401 from the new best and remained close to several other strong observations. This provides stronger empirical support than the more distant NN recommendation.  

Function 8: The NN candidate was closest to the new best observation, at a distance of 0.3843, compared with 0.4896 for EI and 0.7299 for UCB. Together with successful NN cross-validation, this shifted the provisional preference towards the NN candidate.  

Summary:
The neighbourhood analysis provided additional evidence for distinguishing between competing model recommendations. Functions 2, 3, 5, 6 and 7 had candidates positioned close to relatively strong historical observations, while Function 4 showed that all three approaches remained around the same established high-performing region. The analysis was particularly useful for Function 6, where the neural-network candidate was substantially closer to the current best observation than the GP candidate. For Function 8, the neural-network candidate was also closer to the new best observation than either GP recommendation. Function 1 remained uncertain, although the GP candidate was located closer to previously evaluated points than the neural-network alternative.

The final provisional selections were made using a function-specific combination of surrogate reliability, acquisition behaviour and neighbouring observations. EI was preferred for Functions 2–5, while the identical EI/UCB recommendation was retained for Function 7. The neural network materially influenced the strategy for Functions 6 and 8, where its candidates had stronger empirical support than the GP alternatives. Function 1 remained the least certain, but the common EI/UCB candidate was preferred over the less reliable neural-network recommendation. Overall, the neural network complemented rather than replaced the existing GP framework, allowing the selection strategy to adapt where independent evidence supported doing so.

**Step 7. Final Validation**

The eight provisional query points were subjected to a final validation check before submission. Each point was checked to confirm that all coordinates remained within the permitted [0,1] search space and that it did not duplicate an existing observation. The selected method and distance to the nearest existing observation were also recorded as a final check that each recommendation remained consistent with the function-specific strategy developed during the analysis.

All eight final query points passed the validation checks. Every coordinate remained within the permitted [0,1] search space and none duplicated an existing observation. The final selections reflect the function-specific strategy developed during the analysis, combining Expected Improvement and UCB where the GP provided stronger evidence and neural-network recommendations for Functions 6 and 8 where the alternative surrogate was better supported. The points were therefore considered suitable for submission.

Function 1:  
Selected point: [0.943533 0.359421]  
Selection method: EI / UCB agreement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.140043   

Function 2:  
Selected point: [0.713092 0.138155]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.049374  

Function 3:  
Selected point: [0.384686 0.478261 0.416295]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.132463  

Function 4:  
Selected point: [0.462451 0.342242 0.380323 0.445636]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.114172  

Function 5:  
Selected point: [0.938942 0.989452 0.984552 0.971823]  
Selection method: Expected Improvement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.109600  

Function 6:  
Selected point: [0.248909 0.369411 0.484746 0.674732 0.011701]  
Selection method: Neural Network  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.065893  

Function 7:  
Selected point: [0.132842 0.294043 0.285122 0.050957 0.281982 0.619768]  
Selection method: EI / UCB agreement  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.140085  

Function 8:  
Selected point: [0.034135 0.275291 0.032179 0.508682 0.760754 0.61795  0.034127 0.34942 ]  
Selection method: Neural Network  
In bounds: True  
Duplicate existing point: False  
Distance to nearest observation: 0.384306  



So far:

Week 1: Establish a Gaussian Process + UCB baseline for all eight functions.
Week 2: Diagnose function-specific GP behaviour and adapt the strategy where the surrogate model struggles, including testing alternative kernels and greater exploration.
Week 3: Introduce SVM classification to identify high- and low-performing regions and use it selectively where it adds useful information.
Week 4: Investigate the exploration–exploitation trade-off by comparing different UCB κ values and Expected Improvement, leading to more function-specific query selection.
Week 5: Introduce a simple neural-network surrogate as a secondary model and compare its recommendations with the established GP-based approach.
