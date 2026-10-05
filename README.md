# Capstone Project
**Capstone project for Imperial AI/ML Course**

**Overall Objective**

The capstone project involves using Bayesian Optimisation to find the maximum values of eight unknown functions. It begins with an initial set of known observations for each function, which are used to build an initial model and select the first new query point. The functions have different numbers of input dimensions, meaning that some have relatively simple search spaces while others have many more possible combinations of inputs to explore. As the number of dimensions increases, the search space becomes more complex, making it more difficult to identify promising areas from the limited observations available. This means that the balance between exploring uncertain areas and exploiting areas that already appear promising may differ between functions. Each week, one additional data point is received for each function based on the queries submitted, allowing the models to be updated with more information. As the project progresses, the aim is to use this growing dataset to improve the models and make increasingly informed decisions about where to query next. Overall, the project demonstrates how an iterative optimisation approach can gradually build a better representation of unknown functions of varying complexity and move towards better solutions.

**Proposed Methodology**

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





**Module 16 - Week 5**

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

**Module 17 (Week 6)**

**Methodology**

Week 6 builds on the GP and neural-network comparison introduced in Week 5, but shifts the focus towards learning from the performance of previous optimisation decisions. The latest query results will first be incorporated into the growing datasets and used to assess which selection methods have successfully improved each function. Gaussian Process models will then be refitted and used to generate UCB and Expected Improvement candidates, while the same small neural-network ensemble introduced in Week 5 will provide an alternative set of recommendations. Keeping the NN architecture unchanged allows its behaviour to be compared more consistently as additional observations become available.  
The main development this week is the introduction of a historical performance check. Rather than treating GP and neural-network recommendations equally in every round, previous query outcomes will be used as additional evidence about which approaches appear reliable for individual functions. This will not operate as a strict weighting rule, since a single unsuccessful query does not necessarily indicate a poor model, but will contribute to the final decision alongside model predictions, uncertainty, predictive performance and the outputs of nearby observations.  
Candidate neighbourhoods will again be examined to compare model recommendations with the actual observed behaviour of each function. The final query for each function will therefore be selected using multiple sources of evidence while maintaining the function-specific approach developed throughout the capstone. This represents an incremental refinement of the existing methodology rather than the introduction of additional model complexity.  

Step 1 – Incorporate and evaluate the latest results  
- Add the eight Week 5 query points and their returned outputs to the existing datasets. Compare each new result with the previous best value to determine which functions improved and update the best-known observation for each function.  
- This provides the starting point for Week 6 and, importantly, allows the success of last week's different selection methods to be assessed.  
  
Step 2 – Refit and diagnose the Gaussian Process models  
- Refit the Gaussian Process surrogate for each function using all accumulated observations. Continue using the RBF kernel for consistency with previous rounds.  
- Record the fitted length scale and examine the range of predicted means and uncertainties across candidate points. Particular attention will again be given to Functions 1 and 6, where previous rounds showed very small fitted length scales and reduced confidence in the GP.  

Step 3 – Generate GP candidates using UCB and Expected Improvement  
- Generate candidate points for each function using both:  
  - Upper Confidence Bound (UCB), with κ = 1.5  
  - Expected Improvement (EI)  
- Compare the proposed locations, predicted outputs, uncertainty and distances from existing observations. Agreement between UCB and EI will provide stronger evidence for a region, while disagreement may indicate a more meaningful exploration–exploitation decision.  

Step 4 – Generate neural-network candidates  
- Retain the same neural-network architecture introduced in Week 5 so that its performance can be compared consistently as more observations become available.  
- For each function, train an ensemble of five neural networks, each containing two hidden layers of 16 neurons with ReLU activation. Use the ensemble to generate an alternative candidate based on predicted output and disagreement between the individual networks.  
- The architecture will deliberately remain unchanged this week rather than introducing additional complexity.  

Step 5 – Check neural-network predictive performance  
- Assess whether the neural networks appear to generalise beyond the observations used for training and compare their predictive performance with a simple baseline.  
- This prevents very low training loss from being interpreted automatically as evidence that the neural network is a reliable surrogate. Where predictive performance is weak, the NN recommendation will receive less weight in the final decision.  

Step 6 – Assess historical model performance (new for Week 6)  
- Introduce a simple historical performance check to examine whether the methods selected in previous rounds actually resulted in improved outputs.  
- For each function, consider:  
  - Which method generated the previous query;  
  - Whether that query improved the best-known result;   
  - The size of any improvement;  
  - Whether similar methods have performed consistently in previous rounds.  
- This information will inform rather than automatically determine the next query. For example, the strong recent performance of EI for Function 5 would increase confidence in continuing to exploit the GP-identified region, while the failure of the NN candidate to improve Function 8 last week would reduce the weight given to another NN recommendation unless other evidence supports it.  

Step 7 – Compare candidate neighbourhoods  
- For each UCB, EI and NN candidate, identify the nearest existing observations and their outputs.  
- This provides a direct check against the actual black-box results rather than relying entirely on model predictions. Candidates located near several strong observations will receive greater support, while candidates surrounded by weak observations may be rejected even if their predicted output is attractive.  

Step 8 – Make a function-specific final decision  
- Evaluate the evidence separately for each of the eight functions rather than applying one acquisition strategy universally.  
- Final selection will consider the GP prediction and uncertainty, UCB/EI agreement, NN predictive performance and disagreement, historical performance of the selection methods, nearby observed outputs, and whether additional exploration or exploitation appears appropriate.  
- The intention is not to create a formal automated weighting system, but to make each decision using a consistent set of evidence.  

Step 9 – Validate and format final queries  
- Check that every selected coordinate lies within the permitted [0,1] search space and does not duplicate an existing observation. Record the final selection method and distance to the nearest observation before formatting each point to six decimal places for submission.  

**Step 1. Incorporate and evaluate the Week 5 results**

The Week 5 query points and their returned outputs were first added to the existing datasets. Each new output was compared with the previous best for its function to determine whether the selected query had produced an improvement. This establishes the updated dataset for Week 6 and provides the first evidence for assessing how successfully the different selection methods performed in the previous round.

Function 1:  
Previous best: 0.000000  
New output:    -0.000000  
Difference:    -0.000000  
Updated best:  0.000000  
Result: Did not improve  

Function 2:  
Previous best: 0.611205  
New output:    0.746962  
Difference:    0.135757  
Updated best:  0.746962  
Result: NEW BEST  

Function 3:  
Previous best: -0.020167  
New output:    -0.016451  
Difference:    0.003716  
Updated best:  -0.016451  
Result: NEW BEST  

Function 4:  
Previous best: -0.300645  
New output:    -0.638343  
Difference:    -0.337698  
Updated best:  -0.300645  
Result: Did not improve  

Function 5:  
Previous best: 4857.449601  
New output:    6691.931674  
Difference:    1834.482072  
Updated best:  6691.931674  
Result: NEW BEST  

Function 6:  
Previous best: -0.458511  
New output:    -0.456874  
Difference:    0.001637  
Updated best:  -0.456874  
Result: NEW BEST  

Function 7:  
Previous best: 2.536939  
New output:    1.644736  
Difference:    -0.892203  
Updated best:  2.536939  
Result: Did not improve   

Function 8:  
Previous best: 9.911249  
New output:    9.742277  
Difference:    -0.168972  
Updated best:  9.911249  
Result: Did not improve  

Four of the eight Week 5 queries produced new best results. Functions 2 and 3 improved using Expected Improvement, while Function 5 produced a particularly large improvement from 4857.45 to 6691.93. Function 6 also produced a small new best using the neural-network candidate, providing some support for the alternative surrogate in this function. Functions 4, 7 and 8 did not improve, while Function 1 again returned a value effectively equal to zero. These mixed results provide useful evidence for comparing the historical performance of the different selection methods later in the Week 6 analysis.

**Step 2. Refit and diagnose the Gaussian Process models**

The Gaussian Process models were refitted using all available observations, including the latest returned outputs. The same RBF kernel and model settings were retained to allow direct comparison with previous weeks. The fitted length scales were examined for each function, with particular attention given to Functions 1 and 6, where previous rounds indicated that the GP was struggling to identify a smooth underlying relationship. This provides an updated assessment of GP reliability before generating new candidate points.

Function 1:  
Fitted kernel: RBF(length_scale=1e-05)  

Function 2:  
Fitted kernel: RBF(length_scale=0.0521)  

Function 3:  
Fitted kernel: RBF(length_scale=0.185)  

Function 4:  
Fitted kernel: RBF(length_scale=0.423)  

Function 5:  
Fitted kernel: RBF(length_scale=0.389)  

Function 6:  
Fitted kernel: RBF(length_scale=0.535)  

Function 7:  
Fitted kernel: RBF(length_scale=0.248)  

Function 8:  
Fitted kernel: RBF(length_scale=1)  

The refitted GP models remained stable for most functions, with Functions 2–5, 7 and 8 producing reasonable fitted length scales. Function 1 continued to reach the lower length-scale bound of 1e-05, indicating that the GP is still struggling to identify a smooth relationship for this function. In contrast, Function 6 changed substantially, with its fitted length scale increasing from 1e-05 last week to 0.535. This suggests that the additional observation has allowed the GP to identify considerably more structure in Function 6, so GP-based recommendations should be reconsidered alongside the neural-network approach this week rather than automatically favouring the NN.

**Step 2B. Prediction and uncertainty diagnostic**

To complement the fitted length-scale diagnostic, each GP was evaluated across a large set of candidate points. The ranges of predicted means and predictive uncertainties were recorded to assess whether the models meaningfully distinguish between different regions of the search space. This provides an additional check of GP behaviour before UCB and Expected Improvement are used to select candidate points.

Function 1:  
Predicted mean minimum: -0.0006345829  
Predicted mean maximum: -0.0000000358  
Predicted mean range:   0.0006345470  

Uncertainty minimum:    0.0000176862  
Uncertainty maximum:    0.0016717864  
Uncertainty range:      0.0016541003  

Function 2:  
Predicted mean minimum: -0.0824904227  
Predicted mean maximum: 0.7489262743  
Predicted mean range:   0.8314166970  

Uncertainty minimum:    0.0002595097  
Uncertainty maximum:    0.2595015299  
Uncertainty range:      0.2592420202   

Function 3:  
Predicted mean minimum: -0.3976338817  
Predicted mean maximum: -0.0085327376  
Predicted mean range:   0.3891011442  

Uncertainty minimum:    0.0000781193  
Uncertainty maximum:    0.0781185128  
Uncertainty range:      0.0780403935  

Function 4:  
Predicted mean minimum: -34.6476933117  
Predicted mean maximum: -0.2133622981  
Predicted mean range:   34.4343310136  

Uncertainty minimum:    0.0085059193  
Uncertainty maximum:    7.3947778303  
Uncertainty range:      7.3862719110  

Function 5:  
Predicted mean minimum: -733.8178440534  
Predicted mean maximum: 6592.2025297858  
Predicted mean range:   7326.0203738392  

Uncertainty minimum:    1.7964221446  
Uncertainty maximum:    1728.0457063821  
Uncertainty range:      1726.2492842375  

Function 6:  
Predicted mean minimum: -2.6615544551  
Predicted mean maximum: -0.3134452427  
Predicted mean range:   2.3481092124  

Uncertainty minimum:    0.0005630624  
Uncertainty maximum:    0.5085988317  
Uncertainty range:      0.5080357693    

Function 7:  
Predicted mean minimum: -0.0632733559  
Predicted mean maximum: 2.5125425181  
Predicted mean range:   2.5758158740  

Uncertainty minimum:    0.0006653826  
Uncertainty maximum:    0.6653825615  
Uncertainty range:      0.6647171789  

Function 8:  
Predicted mean minimum: 5.1234602949  
Predicted mean maximum: 9.9459454693  
Predicted mean range:   4.8224851744  

Uncertainty minimum:    0.0010804513  
Uncertainty maximum:    0.6608592094  
Uncertainty range:      0.6597787581  

The prediction diagnostics showed meaningful variation in both predicted outputs and uncertainty for Functions 2–8. Function 6 showed a particularly notable improvement in GP behaviour, with the predicted mean now ranging from −2.662 to −0.313 following the increase in its fitted length scale. This supports reconsidering GP-based candidates for Function 6 alongside the neural network this week. Function 1 continued to show very limited predicted variation and retained its extremely small fitted length scale, so confidence in its GP recommendations remains low. The remaining functions produced sufficiently varied predictions and uncertainty estimates to proceed with GP-based acquisition methods.

**Step 3. Generate UCB and Expected Improvement candidates**

UCB and Expected Improvement were next applied to the updated GP models to generate two candidate points for each function. UCB was retained with κ = 1.5 for consistency with the previous round, while EI was used to identify points offering the greatest expected improvement over the current best observation. The two recommendations were compared using their predicted outputs, uncertainty, proximity to existing observations and distance from one another. Agreement between the acquisition functions provides additional confidence in a region, while substantial disagreement highlights cases requiring further investigation.

Condensed output:  
Function 1:  
UCB: [0.773956 0.438878] | Predicted output: -0.000635  
EI:  [0.773956 0.438878] | Predicted output: -0.000635  
UCB → EI distance: 0.0000  
 
Function 2:  
UCB: [0.732816 0.105087] | Predicted output: 0.644023  
EI:  [0.729513 0.111999] | Predicted output: 0.679240  
UCB → EI distance: 0.0077  

Function 3:    
UCB: [0.446987 0.643104 0.558807] | Predicted output: -0.048421  
EI:  [0.308260 0.310001 0.479247] | Predicted output: -0.010868  
UCB → EI distance: 0.3695  

Function 4:  
UCB: [0.375361 0.358970 0.346315 0.392967] | Predicted output: -0.455899  
EI:  [0.329887 0.354827 0.355151 0.453774] | Predicted output: -0.278717  
UCB → EI distance: 0.0766  

Function 5:  
UCB/EI: [0.979194 0.890392 0.988241 0.846073]  
Predicted output: 6213.849027  
UCB → EI distance: 0.0000  

Function 6:  
UCB: [0.450329 0.401298 0.605152 0.701731 0.160904] | Predicted output: -0.338919  
EI:  [0.397519 0.393747 0.471443 0.711380 0.165385] | Predicted output: -0.313445  
UCB → EI distance: 0.1444  

Function 7:  
UCB/EI: [0.174885 0.324058 0.479146 0.269144 0.331135 0.614490]  
Predicted output: 2.512543  
UCB → EI distance: 0.0000  

Function 8:  
UCB/EI: [0.439936 0.123702 0.034861 0.474625 0.754089 0.093924 0.236209 0.059251]  
Predicted output: 9.793025  
UCB → EI distance: 0.0000  

The updated GP models produced several areas of agreement between the acquisition functions. UCB and EI selected identical candidates for Functions 1, 5, 7 and 8, while their recommendations for Function 2 were also extremely close. Functions 3, 4 and 6 showed greater disagreement, suggesting that the exploration–exploitation trade-off remains more important for these functions. Function 5 continued to focus on the established high-performing region, while the improved GP for Function 6 predicted outputs above the current best for both acquisition candidates. Function 1 showed acquisition-function agreement, but this should be treated cautiously because of its persistent GP length-scale problem.

Function-by-function:  
- F2: Very strong agreement. EI is only 0.0309 from our new best of 0.746962, and the two candidates are only 0.0077 apart. This looks like a strong exploitation region.  
- F3: This is a genuine decision. UCB goes considerably further out with higher uncertainty, whereas EI stays closer to known observations and predicts −0.010868, which would beat our current best of −0.016451.  
- F5: Particularly interesting. UCB and EI independently select exactly the same point. Given the repeated historical improvements in this region, that's strong evidence for the candidate.  
- F6: Very encouraging after the GP recovery. EI predicts −0.313445 against a current best of −0.456874. We certainly shouldn't default to the NN this week.  
- F7: UCB/EI agreement again, although last week's GP-selected candidate failed. Our historical-performance check will matter here.  
- F8: UCB/EI now agree exactly, whereas last week they were 0.7320 apart. That's a substantial change. However, the candidate is 0.7221 from the current best, so it's still fairly exploratory.  
- F1: The agreement looks impressive superficially, but because the GP is still at RBF(length_scale=1e-05), agreement between two acquisition functions based on the same unreliable surrogate isn't independent confirmation.  

**Step 4. Generate neural-network candidates**

**Step 4a. Retrain the neural-network ensembles**

The neural-network ensembles were retrained using the expanded Week 6 datasets. Five networks were fitted independently for each function using the same architecture as the previous round: two hidden layers of 16 neurons with ReLU activation. The architecture and training procedure were deliberately kept unchanged so that differences in performance could be attributed primarily to the additional observations rather than changes to the model itself. Final training losses were recorded as an initial check of model fit, before predictive performance was assessed separately in Step 4B.

Function 1:
Mean final training loss: 0.014672  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.072590  

Function 2:
Mean final training loss: 0.023384  
Minimum final training loss: 0.000267  
Maximum final training loss: 0.087228  

Function 3:
Mean final training loss: 0.012198  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.042261  

Function 4:
Mean final training loss: 0.001875  
Minimum final training loss: 0.000256  
Maximum final training loss: 0.005423  

Function 5:
Mean final training loss: 0.000085  
Minimum final training loss: 0.000021  
Maximum final training loss: 0.000164  

Function 6:
Mean final training loss: 0.003526  
Minimum final training loss: 0.000011  
Maximum final training loss: 0.009342  

Function 7:
Mean final training loss: 0.000907  
Minimum final training loss: 0.000000  
Maximum final training loss: 0.002452  

Function 8:
Mean final training loss: 0.001338  
Minimum final training loss: 0.000133  
Maximum final training loss: 0.003092  

The neural-network ensembles were able to fit the updated datasets, although training performance varied between functions and individual networks. Functions 4–8 generally achieved low mean training losses, while Functions 1–3 showed greater variation between ensemble members. Function 5 produced particularly consistent low losses despite the substantial increase in its best observed output. However, training loss alone does not indicate how well the models generalise, so predictive performance was assessed separately before using the neural networks for candidate selection.

**Step 4B. Check neural-network predictive performance**

The predictive performance of the neural networks was assessed before their recommendations were considered. Leave-one-out cross-validation was used because the datasets remain small, allowing each observation to be tested while retaining as much data as possible for training. The resulting mean absolute error was compared with a simple mean-prediction baseline. This provides a straightforward indication of whether the neural network is learning useful relationships beyond simply fitting the available training observations.

Function 1:
NN LOOCV MAE:       0.001675  
Mean baseline MAE:  0.001179  
NN beats baseline:  False  

Function 2:
NN LOOCV MAE:       0.267443  
Mean baseline MAE:  0.241496  
NN beats baseline:  False   

Function 3:
NN LOOCV MAE:       0.067227  
Mean baseline MAE:  0.048537  
NN beats baseline:  False  

Function 4:
NN LOOCV MAE:       1.827993  
Mean baseline MAE:  6.878180  
NN beats baseline:  True  

Function 5:
NN LOOCV MAE:       273.005550  
Mean baseline MAE:  1375.410321  
NN beats baseline:  True  

Function 6:
NN LOOCV MAE:       0.309082  
Mean baseline MAE:  0.476405  
NN beats baseline:  True  

Function 7:
NN LOOCV MAE:       0.323396  
Mean baseline MAE:  0.507559  
NN beats baseline:  True  

Function 8:
NN LOOCV MAE:       0.294854  
Mean baseline MAE:  0.920553  
NN beats baseline:  True  

The predictive-performance check showed that the neural network outperformed the simple mean baseline for Functions 4–8, but not for Functions 1–3. Functions 1 and 3 therefore continued the weaker NN performance observed in the previous round, while Function 2 changed from outperforming the baseline in Week 5 to performing slightly worse this week. In contrast, Functions 4–8 continued to show evidence that the neural networks were learning useful predictive relationships. These results will be used as supporting evidence rather than an automatic selection rule when comparing the NN and GP candidates.

**Step 4C. Generate neural-network candidates**

The validated neural-network ensembles were next used to generate an alternative candidate for each function. Predictions from all five networks were combined to calculate the mean predicted output and disagreement between models across the candidate space. Candidate selection considered both predicted performance and ensemble disagreement, providing an alternative recommendation to compare with the GP-based UCB and EI candidates. The validation results from Step 4B will subsequently determine how much confidence is placed in each NN recommendation.

Function 1:  
NN candidate: [0.998294 0.000709]  
Predicted output: 0.003677  
Model disagreement: 0.002351  

Function 2:  
NN candidate: [0.998307 0.001142]  
Predicted output: 2.724940  
Model disagreement: 1.237037  

Function 3:  
NN candidate: [0.364502 0.118478 0.645908]  
Predicted output: -0.001139  
Model disagreement: 0.032445  

Function 4:  
NN candidate: [0.453836 0.433425 0.386591 0.463858]  
Predicted output: -0.374836  
Model disagreement: 0.459252  

Function 5:  
NN candidate: [0.939869 0.992261 0.978815 0.999723]  
Predicted output: 6691.842285  
Model disagreement: 47.407719  
 
Function 6:  
NN candidate: [0.457009 0.308462 0.494921 0.626848 0.011430]  
Predicted output: -0.406389  
Model disagreement: 0.128387  

Function 7:  
NN candidate: [0.229165 0.134659 0.270334 0.289636 0.333549 0.781285]  
Predicted output: 2.412130  
Model disagreement: 0.419323  

Function 8:  
NN candidate: [0.042609 0.051269 0.314729 0.127230 0.965221 0.369000 0.150066 0.389171]  
Predicted output: 10.052735  
Model disagreement: 0.132650  

The neural-network candidates differed considerably across functions. Functions 4–6 produced candidates relatively close to the current best observations, suggesting local refinement of known high-performing regions. Function 8 produced a more exploratory candidate with a predicted output of 10.053, above the current best of 9.911. In contrast, the very high prediction of 2.725 for Function 2 was accompanied by substantial model disagreement and weak validation performance, reducing confidence in this recommendation. Functions 1 and 3 also produced potentially attractive predictions, but their neural networks failed to outperform the baseline, so these candidates should be treated cautiously.

**Step 5. Compare GP and NN candidates**

The GP and neural-network recommendations were next compared directly by calculating the distance between the NN candidate and the UCB and EI candidates for each function. The distance between the two GP acquisition candidates was also retained. Close agreement between independently generated recommendations may provide additional support for a region, while large differences indicate that the models are interpreting the available observations differently and require further investigation before a final query is selected.

Function 1:  
NN → UCB distance: 0.4923
NN → EI distance:  0.4923
UCB → EI distance: 0.0000

Function 2:  
NN → UCB distance: 0.2851
NN → EI distance:  0.2908
UCB → EI distance: 0.0077

Function 3:  
NN → UCB distance: 0.5382
NN → EI distance:  0.2600
UCB → EI distance: 0.3695

Function 4:  
NN → UCB distance: 0.1355
NN → EI distance:  0.1504
UCB → EI distance: 0.0766

Function 5:  
NN → UCB distance: 0.1887
NN → EI distance:  0.1887
UCB → EI distance: 0.0000
 
Function 6:  
NN → UCB distance: 0.2208
NN → EI distance:  0.2055
UCB → EI distance: 0.1444 

Function 7:  
NN → UCB distance: 0.3327
NN → EI distance:  0.3327
UCB → EI distance: 0.0000

Function 8:  
NN → UCB distance: 0.7737
NN → EI distance:  0.7737
UCB → EI distance: 0.0000

The comparison showed varying levels of agreement between the GP and neural-network recommendations. Functions 4, 5 and 6 showed relatively similar candidate regions across the models, while Functions 1, 2, 3, 7 and particularly 8 showed greater disagreement. UCB and EI agreed exactly for Functions 1, 5, 7 and 8 and were also very close for Function 2. However, agreement between GP acquisition functions does not necessarily confirm that the underlying GP is reliable, particularly for Function 1. The larger differences between models highlight where validation, historical performance and neighbourhood evidence will be particularly important in the final selection.  

Function-by-function comparison:  
- Function 1: UCB and EI selected the same candidate, while the NN recommendation was 0.4923 away. However, the GP still has the extremely small fitted length scale and the NN failed the predictive-performance check, so neither recommendation can yet be considered particularly reliable.  
- Function 2: UCB and EI were almost identical, with a distance of only 0.0077, while the NN candidate was considerably further away. Combined with the NN failing the predictive-performance check this week, the evidence currently favours the GP region.  
- Function 3: The NN candidate was closer to EI (0.2600) than UCB (0.5382), but the NN again failed to outperform the baseline. EI therefore currently appears more credible than either the NN or the more exploratory UCB recommendation.  
- Function 4: All three candidates were relatively close, with the NN only 0.1355 from UCB and 0.1504 from EI. The NN also passed the predictive-performance check, suggesting that multiple models may be identifying a similar promising region.  
- Function 5: UCB and EI selected exactly the same candidate, while the NN proposed a different point within the broader high-performing region. Given the continued strong NN validation and repeated improvements in this area, both approaches remain credible.  
- Function 6: The NN candidate was reasonably close to both GP recommendations, particularly EI at 0.2055. This is encouraging because the NN continues to validate well while the GP fit has improved substantially this week, suggesting increasing agreement between the two modelling approaches.  
- Function 7: UCB and EI selected the same candidate, while the NN recommendation was 0.3327 away. Both models remain plausible because the NN passed validation, although the failure of last week's GP-selected point to improve means GP agreement alone should not determine the decision.  
- Function 8: This produced the strongest disagreement, with the NN candidate 0.7737 from the identical UCB/EI recommendation. The NN continues to perform well in validation, but its previous selected candidate failed to improve the function, so further evidence is needed before choosing between the two regions.  

**Step 6. Historical model-performance check**

To incorporate information from previous optimisation decisions, the recent performance of the selected method for each function was reviewed. For each previous query, the selection method and resulting output were compared with the best value available before that query. This provides evidence about whether particular approaches have successfully identified improving regions for individual functions. Historical performance is treated as supporting evidence rather than a strict weighting rule, since the success or failure of a single query does not necessarily establish the quality of a model.

**Step 6A. Create the historical-performance table**

Function-by-function interpretation:  
- Function 1: The previous UCB/EI selection did not improve the best result, adding little support for repeating the GP recommendation this week. Combined with the persistent GP length-scale problem, other evidence will be particularly important.  
- Function 2: EI produced a meaningful new best, improving from 0.6112 to 0.7470. This strengthens the case for the GP/EI region this week, especially because the current UCB and EI recommendations are almost identical.  
- Function 3: EI produced another new best, although the improvement was relatively small. Combined with the NN's weak predictive performance, this increases confidence in considering EI again.  
- Function 4: The previous EI selection failed to improve the best result. Historical performance therefore provides no additional reason to favour EI, making the current candidate neighbourhood evidence more important.  
- Function 5: EI produced a very large improvement from 4857.45 to 6691.93. This provides strong evidence that the GP has identified a productive region and supports continued exploitation around it.  
- Function 6: The NN candidate produced a new best, although only by a small margin. This supports continued consideration of the NN, but the substantial improvement in the GP fit this week means both models should now be compared.  
- Function 7: The previous UCB/EI agreement candidate failed to improve. The identical GP recommendation this week therefore requires additional support rather than being selected purely because both acquisition functions agree.  
- Function 8: The previous NN selection failed to improve despite its attractive prediction. This reduces confidence in relying on another NN prediction alone, particularly given the large disagreement between the GP and NN candidates this week.  

The historical performance check showed that four of the eight Week 5 selections produced new best results. Expected Improvement was successful for Functions 2, 3 and 5, with Function 5 showing by far the largest improvement at +1834.48. The neural-network selection also produced a new best for Function 6, although the improvement was small at +0.00164. In contrast, the GP-agreement selections for Functions 1 and 7 did not improve, while the neural-network selection for Function 8 was also unsuccessful. These results provide useful function-specific evidence, but are treated as supporting information rather than proof that one method is consistently superior.

**Step 7. Compare candidate neighbourhoods**

The local neighbourhood around each UCB, EI and neural-network candidate was examined using the actual observed data. For each candidate, the three nearest existing observations and their outputs were identified. This provides a direct check of whether a model recommendation lies close to previously successful or unsuccessful regions. The neighbourhood evidence will be considered alongside model predictions, validation performance and historical method performance when making the final function-specific selections.

Function 1:  
UCB/EI nearest: Distance = 0.1807 | Output = 0.000000  
NN nearest:     Distance = 0.3077 | Output = 0.000000  

Function 2:  
UCB nearest:    Distance = 0.0385 | Output = 0.746962  
EI nearest:     Distance = 0.0309 | Output = 0.746962  
NN nearest:     Distance = 0.3164 | Output = 0.746962  

Function 3:  
UCB nearest:    Distance = 0.2255 | Output = -0.034835  
EI nearest:     Distance = 0.1407 | Output = -0.020167  
NN nearest:     Distance = 0.2033 | Output = -0.060870  

Function 4:  
UCB nearest:    Distance = 0.1042 | Output = -0.300645  
EI nearest:     Distance = 0.1049 | Output = -0.300645  
NN nearest:     Distance = 0.0619 | Output = -0.300645  

Function 5:  
UCB/EI nearest: Distance = 0.1651 | Output = 6691.931674  
NN nearest:     Distance = 0.0286 | Output = 6691.931674  

Function 6:  
UCB nearest:    Distance = 0.2812 | Output = -0.456874  
EI nearest:     Distance = 0.2131 | Output = -0.458511  
NN nearest:     Distance = 0.2223 | Output = -0.456874  

Function 7:  
UCB/EI nearest: Distance = 0.1727 | Output = 2.536939  
NN nearest:     Distance = 0.2813 | Output = 2.536939  

Function 8:  
UCB/EI nearest: Distance = 0.6614 | Output = 9.821986  
NN nearest:     Distance = 0.3972 | Output = 9.821986  

Function-by-function interpretation:  
- Function 1: The GP candidate is closer to existing observations than the NN candidate, but all nearby outputs are effectively zero. Neither neighbourhood provides convincing evidence of a better region. Combined with the unreliable GP fit and weak NN validation, F1 remains highly uncertain.  
- Function 2: Both GP candidates are extremely close to the new best observation of 0.746962, with EI only 0.0309 away. They are also close to the previous strong result of 0.538996. The NN candidate is substantially further from these observations and failed the predictive-performance check. This strongly favours the GP region, with EI slightly preferred.  
- Function 3: Both GP candidates have reasonable neighbourhoods, but EI is closer to the two strongest observations (−0.016451 and −0.020167) than UCB. The NN candidate's nearest observation is considerably weaker at −0.060870, and the NN failed validation. This provides further support for EI.  
- Function 4: All three candidates lie close to the current best of −0.300645. The NN is closest at 0.0619, but it is also very close to the weaker recent observations of −1.438643 and −0.638343. The GP candidates move slightly further away from these weaker observations while remaining close to the best, so the neighbourhood evidence does not clearly favour the NN despite its proximity.  
- Function 5: Both approaches are clearly targeting the successful region. However, the NN candidate is only 0.0286 from the new best of 6691.93, compared with 0.1651 for the identical UCB/EI candidate. It is also closer to the previous strong results of 4857.45 and 4595.48. Combined with strong NN validation, this makes the NN candidate more competitive than the model-distance comparison alone suggested.  
- Function 6: This is now a strong area of model convergence. UCB, EI and NN are all positioned around the two best observations of −0.456874 and −0.458511. EI is particularly interesting because it predicts −0.313445, substantially above the current best, while the GP fit has improved considerably this week. The NN remains credible following its successful query last round, but the evidence now supports reconsidering the GP.  
- Function 7: The identical UCB/EI candidate is closer than the NN candidate to all three of the strongest observations (2.536939, 2.139551 and 1.950612). Although last week's GP-selected query failed to improve, the local evidence this week favours the GP candidate over the NN recommendation.  
- Function 8: Both candidate regions are relatively distant from existing observations, reflecting the higher-dimensional search space. The NN candidate is nevertheless substantially closer to the three strongest recent observations than the GP candidate. However, last week's NN-selected query failed to improve, so this local evidence needs to be balanced against its weaker recent historical performance.  

The neighbourhood analysis provided further evidence for distinguishing between the competing candidates. Functions 2, 3, 5, 6 and 7 all had candidates positioned close to some of their strongest existing observations, while Functions 1 and 8 remained more uncertain. Function 5 showed particularly strong local evidence, with both model approaches lying near the sequence of recent improvements. For Function 6, all three candidates were located around the two best observations, providing further evidence that the GP and NN are now identifying a similar promising region. The results also reinforced the importance of considering observed neighbourhoods alongside model predictions, particularly where the surrogate models disagree.

**Step 8. Final candidate selection**

Final Week 6 candidates were selected on a function-by-function basis using the combined evidence from the GP and neural-network models. Decisions considered UCB and EI recommendations, neural-network predictive performance, agreement between models, the success of the previous round's selected method and the observed neighbourhood surrounding each candidate. No single model was applied automatically across all functions; instead, the method with the strongest overall evidence for each function was selected.

Function-by-function selection:

Function 1 — UCB/EI
Selected: [0.773956, 0.438878]
This is the least certain decision. The GP remains unreliable because its length scale is stuck at 1e-05, while the NN has again failed validation. However, UCB and EI agree exactly and their candidate is closer to existing observations (0.1807) than the NN (0.3077). With neither model providing strong predictive evidence, the GP candidate is the more conservative choice.
- Confidence: Low

Function 2 — Expected Improvement
Selected: [0.729513, 0.111999]
This is one of our clearest decisions. EI produced a new best last week, increasing the output from 0.611205 to 0.746962. This week's UCB and EI candidates are only 0.0077 apart, while EI lies just 0.0309 from that new best observation. The NN failed validation and proposes a much more distant candidate despite predicting 2.724940.
- Confidence: High

Function 3 — Expected Improvement
Selected: [0.308260, 0.310001, 0.479247]
EI produced a new best last round and now predicts −0.010868, which would improve on the current best of −0.016451. Its candidate also has stronger local evidence than the NN, which again failed validation. UCB is considerably more exploratory and predicts a weaker output of −0.048421.
- Confidence: High

Function 4 — Expected Improvement
Selected: [0.329887, 0.354827, 0.355151, 0.453774]
This is less decisive. The NN passed validation and is closest to the current best, but it is also extremely close to several weaker observations. EI remains near the best observed region while moving slightly away from those unsuccessful points and predicts −0.278717, marginally better than the current best of −0.300645.
- Confidence: Medium

Function 5 — Neural Network
Selected: [0.939869, 0.992261, 0.978815, 0.999723]
The neural network was selected in preference to EI this week. Both approaches identify the same highly productive region, but the NN is only 0.0286 from the new best of 6691.93, compared with 0.1651 for the GP candidate. It also remains close to the previous strong observations, and the NN continues to perform very well against the validation baseline. This makes a small local refinement around the extremely successful Week 5 point attractive.
- Confidence: High

Function 6 — Expected Improvement
Selected: [0.397519, 0.393747, 0.471443, 0.711380, 0.165385]
This is an interesting change from last week. The NN produced a small new best and still validates successfully, but the GP has now recovered and both models are targeting broadly similar regions. EI predicts −0.313445, substantially better than the current best of −0.456874, while remaining close to the two strongest observations. Selecting EI also provides useful evidence about whether the improved GP fit translates into better optimisation performance.
- Confidence: Medium–High

Function 7 — UCB/EI agreement
Selected: [0.174885, 0.324058, 0.479146, 0.269144, 0.331135, 0.614490]
Although last week's GP-selected point failed to improve, this week's UCB and EI candidates agree exactly and their candidate is closer than the NN to all three of the strongest existing observations. The NN remains credible based on validation, but its predicted output of 2.412130 is below the current best of 2.536939. The GP candidate therefore has stronger combined evidence.
- Confidence: Medium–High

Function 8 — Neural Network
Selected: [0.042609, 0.051269, 0.314729, 0.127230, 0.965221, 0.369000, 0.150066, 0.389171]
This is probably our most interesting judgement call. The GP acquisition functions agree exactly, but their candidate is far from observed data: its nearest observation is 0.6614 away. The NN candidate is substantially closer at 0.3972, continues to outperform the validation baseline and predicts 10.052735, above the current best of 9.911249. Last week's NN candidate failed to improve, so confidence should remain moderate rather than high, but on balance the combined evidence favours the NN candidate.
- Confidence: Medium

Overall, the final selections comprised four Expected Improvement candidates, two UCB/EI agreement candidates and two neural-network candidates. This mixture reflects the function-specific approach adopted throughout the optimisation rather than reliance on a single selection method.

**Step 9. Final candidate validation**

Before submission, the eight selected candidates were subjected to a final validation check. Each candidate was tested to confirm that all input values remained within the permitted [0,1] bounds and that the point did not duplicate an existing observation. The Euclidean distance to the nearest existing observation was also calculated to provide a final indication of how exploratory or exploitative each query was. This final check ensures that all submitted points are valid while retaining a record of their relationship to the existing data.

Function 1:  
Selected point: [0.773956 0.438878]
Selection method: EI / UCB agreement
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.180671

Function 2:  
Selected point: [0.729513 0.111999]
Selection method: Expected Improvement
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.030884

Function 3:  
Selected point: [0.30826  0.310001 0.479247]
Selection method: Expected Improvement
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.140675 

Function 4:  
Selected point: [0.329887 0.354827 0.355151 0.453774]
Selection method: Expected Improvement
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.104897

Function 5:  
Selected point: [0.939869 0.992261 0.978815 0.999723]
Selection method: Neural Network
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.028637

Function 6:  
Selected point: [0.397519 0.393747 0.471443 0.71138  0.165385]
Selection method: Expected Improvement
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.213058

Function 7:  
Selected point: [0.174885 0.324058 0.479146 0.269144 0.331135 0.61449 ]
Selection method: EI / UCB agreement
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.172684

Function 8:  
Selected point: [0.042609 0.051269 0.314729 0.12723  0.965221 0.369    0.150066 0.389171]
Selection method: Neural Network
In bounds: True
Duplicate existing point: False
Distance to nearest observation: 0.397197

All eight selected candidates passed the final validation checks. Every point remained within the permitted [0,1] input bounds and none duplicated an existing observation. Distances to the nearest existing observations ranged from 0.028637 for Function 5, reflecting deliberate local refinement of its strongly performing region, to 0.397197 for Function 8, where the selected neural-network candidate remains more exploratory. The final validation therefore confirmed that all eight candidates were suitable for submission.

**Module 18 (Week 7)**

**Methodology**

Week 7 builds on the function-specific optimisation strategy developed during the previous rounds. Rather than introducing another new modelling approach, the focus this week is on hyperparameter tuning and testing the robustness of the existing optimisation methods.
The Gaussian Process remains the primary surrogate model, with Expected Improvement (EI) and Upper Confidence Bound (UCB) used as the main acquisition functions. However, instead of relying on a single fixed acquisition setting, multiple values of the exploration parameters will be tested to examine how sensitive the recommended candidates are to the exploration–exploitation balance.
The neural-network surrogate introduced in Week 5 will continue to provide an independent comparison where its validation performance supports its use. Candidate selection will therefore remain function-specific and will combine GP predictions, acquisition-function behaviour, neural-network evidence, historical method performance and the observed neighbourhood surrounding each candidate.
Particular attention will be given to whether different hyperparameter settings converge on similar candidate regions. Agreement across settings will be treated as stronger evidence that a recommendation is robust, while substantial disagreement will indicate greater uncertainty and the need for a more cautious or exploratory selection.
The final objective is therefore not simply to identify the candidate with the highest acquisition score, but to understand how dependent each recommendation is on the model and hyperparameter assumptions used to generate it.

Steps:
Step 1 – Add and assess Week 6 results
Add the eight new observations to the datasets and compare them with the previous bests. Record which methods succeeded or failed to improve each function.

Step 2 – Refit and diagnose the GP models
Refit each Gaussian Process using the expanded dataset. Review fitted kernels, length scales, prediction ranges and uncertainty to identify any remaining modelling issues.

Step 3 – Tune acquisition hyperparameters
Test several UCB κ values (0.5, 1.0, 1.5, 2.5) and EI ξ values. Record the candidate, predicted output, uncertainty and acquisition score produced by each setting.

Step 4 – Assess candidate stability
Compare the candidates produced by different κ and ξ values. Determine whether recommendations remain in similar regions or change substantially as the exploration–exploitation balance is adjusted.

Step 5 – Refit and validate the neural network
Retrain the NN surrogate using the expanded datasets and repeat the LOOCV comparison against the mean baseline. Only treat the NN as strong evidence where validation supports it.

Step 6 – Compare GP and NN candidates
Generate NN candidates and compare them with the strongest tuned GP candidates, considering predicted outputs, uncertainty/model disagreement and distances between recommendations.

Step 7 – Check neighbourhood evidence
Examine the nearest existing observations around the leading candidates to determine whether they are supported by previously strong results or represent more exploratory choices.

Step 8 – Select final candidates
Choose one point per function using the combined evidence from previous method performance, GP reliability, hyperparameter stability, EI/UCB behaviour, NN validation and local observations.

Step 9 – Final validation and submission
Check bounds, dimensions, duplicates and six-decimal formatting before generating the eight submission strings.

**Step 1. Add and assess the Week 6 results**

The Week 6 query points and their returned outputs were added to the accumulated dataset for each function. Before fitting any new models, the new results were compared with the previous best observed values. This provides a direct measure of how successful the previous selection strategy was and allows recent method performance to inform the Week 7 candidate-selection process.

Function 1:  
Previous best: 0.000000  
New output:    0.000000  
Difference:    -0.000000  
Updated best:  0.000000  
Result:        Did not improve  

Function 2:  
Previous best: 0.746962  
New output:    0.471844  
Difference:    -0.275118  
Updated best:  0.746962  
Result:        Did not improve  

Function 3:  
Previous best: -0.016451  
New output:    -0.015498  
Difference:    0.000953  
Updated best:  -0.015498  
Result:        NEW BEST  

Function 4:  
Previous best: -0.300645  
New output:    -0.200431  
Difference:    0.100214  
Updated best:  -0.200431  
Result:        NEW BEST  

Function 5:  
Previous best: 6691.931674  
New output:    7121.151955  
Difference:    429.220281  
Updated best:  7121.151955  
Result:        NEW BEST  

Function 6:  
Previous best: -0.456874  
New output:    -0.379740  
Difference:    0.077134  
Updated best:  -0.379740  
Result:        NEW BEST  

Function 7:  
Previous best: 2.536939  
New output:    2.929370  
Difference:    0.392431  
Updated best:  2.929370  
Result:        NEW BEST  

Function 8:  
Previous best: 9.911249  
New output:    9.840530  
Difference:    -0.070719  
Updated best:  9.911249  
Result:        Did not improve  

Step 1 results: Five of the eight Week 6 queries produced new best observations, providing encouraging evidence that the increasingly function-specific selection strategy is working. Expected Improvement was particularly successful, producing new bests for Functions 3, 4 and 6. Function 4 improved from −0.300645 to −0.200431, while Function 6 improved from −0.456874 to −0.379740. Function 3 also continued its gradual improvement, moving from −0.016451 to −0.015498. Function 5 provided further support for the neural-network surrogate, with the NN-selected candidate increasing the best output from 6691.93 to 7121.15. The UCB/EI agreement candidate for Function 7 was also successful, producing a substantial improvement from 2.536939 to 2.929370.
The unsuccessful queries are also informative. Expected Improvement did not improve Function 2, with the new output of 0.471844 remaining below the current best of 0.746962. Similarly, the neural-network candidate for Function 8 returned 9.840530, which remained close to but below the best of 9.911249. This is the second consecutive round in which the NN has failed to improve Function 8, suggesting that its recommendations for this function should be treated more cautiously despite reasonable validation performance. Function 1 again returned a value effectively equal to zero, providing little additional evidence about its optimum and reinforcing the difficulty both surrogate approaches have had in modelling this function.
Overall, the results support continuing with a mixed, function-specific approach rather than selecting one model for all eight functions. In particular, the recent results strengthen the case for EI on Functions 3, 4 and 6 and the NN on Function 5, while Functions 1, 2 and 8 warrant greater scrutiny this week. Function 7's improvement also suggests that agreement between UCB and EI can provide useful evidence when both acquisition functions independently identify the same region.

**Step 2. Refit and diagnose the Gaussian Process models**

The Gaussian Process surrogate for each function was refitted using all observations available at the start of Week 7. The fitted kernel and length scale were examined first to identify whether the additional data had changed the behaviour of the surrogate, particularly for functions that had previously shown convergence issues. The range of predicted means and uncertainty across a large set of candidate points was then assessed to determine how strongly the GP differentiates between regions of the search space. These diagnostics were completed before tuning the acquisition-function hyperparameters so that changes caused by the GP itself could be distinguished from changes caused by κ or ξ.

Function 1:  
Fitted kernel: RBF(length_scale=1e-05)  

Function 2:  
Fitted kernel: RBF(length_scale=0.0431)  

Function 3:  
Fitted kernel: RBF(length_scale=0.193)  

Function 4:  
Fitted kernel: RBF(length_scale=0.436)  

Function 5:  
Fitted kernel: RBF(length_scale=0.363)  

Function 6:  
Fitted kernel: RBF(length_scale=0.558)  

Function 7:  
Fitted kernel: RBF(length_scale=0.3)  

Function 8:  
Fitted kernel: RBF(length_scale=1.02)  

The refitted GP models were generally stable following the addition of the Week 6 observations. Functions 2–8 produced finite RBF length scales ranging from 0.0431 to 1.02, while several of the functions that improved last week showed only relatively small changes in their fitted parameters. Function 3 increased from 0.185 to 0.193, Function 4 from 0.423 to 0.436 and Function 6 from 0.535 to 0.558. Function 7 showed a slightly larger increase from 0.248 to 0.300 following its new best result of 2.929370.
Function 1 remains the clear exception. Its fitted length scale is again fixed at the lower bound of 1e-05, producing the same convergence warning seen previously. This suggests that the RBF GP is still struggling to identify a meaningful smooth relationship between nearby observations for this function. GP recommendations for Function 1 should therefore continue to be treated cautiously rather than interpreted in the same way as those for the other functions.
Function 2's length scale decreased from 0.0521 to 0.0431, suggesting a relatively local response structure, while Functions 5 and 8 remained broadly stable at 0.363 and 1.02 respectively. Overall, the additional observations have not caused major instability in the GP models, so it is reasonable to retain them for the Week 7 acquisition-function analysis while accounting for the continuing reliability issue with Function 1.

**Step 2B. Prediction and uncertainty diagnostics**

To further assess the behaviour of each refitted GP, 100,000 candidate points were generated across the permitted search space. The GP was used to calculate the predicted mean and uncertainty for each candidate, allowing the range of predictions and uncertainty to be examined across each function. The same candidate sets will also be retained for the subsequent hyperparameter analysis, ensuring that differences between acquisition recommendations result from changes to the acquisition settings rather than different randomly generated candidate points.

Key diagnostic results:  
F1: predicted range = 0.000595, max uncertainty = 0.001626  
F2: predicted max = 0.7905, max uncertainty = 0.2550  
F3: predicted max = −0.0118, max uncertainty = 0.0782  
F4: predicted max = −0.2003, max uncertainty = 7.4108  
F5: predicted max = 7121.16, max uncertainty = 2074.52  
F6: predicted max = −0.3656, max uncertainty = 0.5196  
F7: predicted max = 2.9294, max uncertainty = 0.7721  
F8: predicted max = 9.9641, max uncertainty = 0.6576  

The diagnostics show meaningful variation in predicted outputs and uncertainty for Functions 2–8, providing a useful basis for acquisition-function tuning. Functions 3–7 have predicted maxima close to their strongest observed values, suggesting the GPs are identifying the recently successful regions. Function 1 remains the main exception, with predictions effectively around zero and uncertainty dominating the predicted mean variation, reinforcing the continued limitations of its RBF GP. Overall, the results support moving to Step 3 to test how different UCB κ and EI ξ values affect candidate selection.

**Step 3. Tune the acquisition-function hyperparameters**

UCB was evaluated using κ values of 0.5, 1.0, 1.5 and 2.5. Lower κ values place greater emphasis on exploitation, while higher values give greater weight to uncertainty and therefore exploration. The same 100,000 candidate points were used for every setting so that changes in the selected candidate could be attributed directly to κ rather than random candidate generation.

F1	No change	Same candidate at all four κ values  
F2	Gradual	Distance from best increased 0.021 → 0.048  
F3	Gradual	Distance increased 0.289 → 0.474  
F4	Threshold	Same candidate at 0.5–1.5; major exploration at 2.5  
F5	Threshold	Known best at 0.5–1.5; exploratory candidate at 2.5  
F6	Strong gradual	Distance increased 0.163 → 0.677  
F7	Threshold	Known best at 0.5; increasingly exploratory thereafter  
F8	Gradual	Distance increased 0.345 → 0.761  

The results show that κ had a meaningful effect on candidate selection for most functions. Functions 2, 3, 6 and 8 became progressively more exploratory as κ increased, while Functions 4, 5 and 7 showed more sudden changes once greater weight was placed on uncertainty. Function 1 remained unchanged across all settings, although its continuing GP reliability issues mean this stability should be treated cautiously. Overall, the results suggest that a single fixed κ is not necessarily appropriate across all eight functions and support considering the exploration level on a function-specific basis.

**Step 3B. EI hyperparameter tuning**

Expected Improvement was evaluated using ξ values equivalent to 0%, 1%, 5% and 10% of the observed output standard deviation for each function. Scaling ξ to each function's output variation allows the same relative settings to be compared despite the substantial differences in output magnitude between functions. As ξ increases, a greater improvement over the current best is required, changing the balance between exploiting known strong regions and considering less certain candidates.

F1	No change	Same candidate at all ξ values  
F2	Very stable	Only small movement as ξ increased  
F3	Gradual	Distance from best increased 0.347 → 0.389  
F4	Strong threshold	Local candidate at 0%; highly exploratory from 1% onwards  
F5	No change	Same candidate at every ξ  
F6	Threshold	Same at 0–1%; more exploratory at 5–10%  
F7	No change	Same candidate at every ξ  
F8	Strong gradual	Distance from best increased 0.345 → 0.761  

The results show that EI sensitivity to ξ varies considerably between functions. Functions 1, 5 and 7 selected the same candidate across all settings, while Function 2 was also relatively stable. Functions 3 and 8 became progressively more exploratory as ξ increased, while Functions 4 and 6 showed clearer threshold effects. Function 4 was particularly sensitive, moving from a candidate only 0.0775 from the current best at ξ = 0% to one 0.9250 away at ξ = 1%. Overall, this again supports selecting acquisition settings on a function-specific basis rather than applying a single fixed value across all eight functions.

**Step 4. Refit and validate the Neural Network**

The neural-network surrogate was retrained using the expanded Week 7 datasets and the same configuration used previously. Its predictive reliability was reassessed using Leave-One-Out Cross-Validation (LOOCV), comparing its Mean Absolute Error (MAE) against a simple mean-prediction baseline. Keeping the NN configuration unchanged allows changes in validation performance to be attributed primarily to the additional observations rather than changes to the model itself. The NN will only be treated as strong evidence for functions where it outperforms the baseline.

F1: Mean training loss = 0.000924  
F2: Mean training loss =	0.006460  
F3: Mean training loss =	0.000429  
F4: Mean training loss =	0.000198  
F5: Mean training loss =	0.000276  
F6: Mean training loss =	0.000301  
F7: Mean training loss =	0.000427  
F8: Mean training loss =	0.000392  

The retrained NN ensembles achieved low training losses across all eight functions. Functions 3–8 showed particularly small and consistent losses across the five random initialisations, while Function 2 showed the greatest variation between models. Overall, the networks fit the observed data closely, but low training error alone does not demonstrate that they generalise successfully to unseen observations. LOOCV is therefore used as the more important test of predictive reliability before deciding how much weight to give the NN recommendations this week.

**Step 4B. LOOCV validation**

LOOCV was used to test whether the neural network could generalise beyond the observations used for training. Each observation was removed in turn, the model was retrained on the remaining data, and its prediction for the held-out observation was recorded. The resulting MAE was compared with a simple mean-prediction baseline. This provides a more meaningful measure of NN reliability than training loss alone.

Function 1:  
NN LOOCV MAE:       0.001776  
Mean baseline MAE:  0.001111  
NN beats baseline:  False  

Function 2:  
NN LOOCV MAE:       0.218453  
Mean baseline MAE:  0.237181  
NN beats baseline:  True  

Function 3:  
NN LOOCV MAE:       0.063001  
Mean baseline MAE:  0.050129  
NN beats baseline:  False  

Function 4:  
NN LOOCV MAE:       1.906639  
Mean baseline MAE:  7.081622  
NN beats baseline:  True  

Function 5:  
NN LOOCV MAE:       228.443505  
Mean baseline MAE:  1689.428734  
NN beats baseline:  True  

Function 6:  
NN LOOCV MAE:       0.235898  
Mean baseline MAE:  0.497547  
NN beats baseline:  True  

Function 7:  
NN LOOCV MAE:       0.376668  
Mean baseline MAE:  0.586378  
NN beats baseline:  True  

Function 8:  
NN LOOCV MAE:       0.358344  
Mean baseline MAE:  0.941538  
NN beats baseline:  True  

The LOOCV results provide stronger evidence for the NN than the training losses alone. The NN outperformed the mean baseline for six of the eight functions, with Functions 4, 5, 6 and 8 showing particularly clear improvements over the baseline. Function 2 also passed the validation test for the first time, with its MAE falling from 0.2674 last week to 0.2185, suggesting that the additional observation has improved its predictive performance. Functions 1 and 3 continue to fail the baseline test, so NN recommendations for these functions will be given limited weight. Overall, the NN remains useful as a secondary surrogate, but its reliability continues to vary by function.

**Step 5. Generate and compare NN candidates**

The validated NN ensembles were used to generate an alternative candidate for each function. Candidate selection was based primarily on the mean prediction across the five models, while disagreement between ensemble members was retained as an indication of how consistently the networks supported the prediction. Distances from the current best and nearest existing observation were also calculated. These candidates provide an independent comparison with the GP recommendations, with greater weight given to the NN for functions where it successfully passed LOOCV validation.

Function 1:  
NN selected point: [0.703066 0.997962]  
NN predicted output:             0.000741  
NN model disagreement:          0.000834  
Distance from current best:     0.2664  
Distance from nearest existing: 0.1383  

Function 2:  
NN selected point: [0.708221 0.158532]  
NN predicted output:             0.706124  
NN model disagreement:          0.082928  
Distance from current best:     0.0210  
Distance from nearest existing: 0.0210  

Function 3:  
NN selected point: [0.320487 0.370858 0.441843]  
NN predicted output:             -0.014537  
NN model disagreement:          0.003243  
Distance from current best:     0.0725  
Distance from nearest existing: 0.0725  

Function 4:  
NN selected point: [0.384332 0.391111 0.336887 0.416488]  
NN predicted output:             -0.001964  
NN model disagreement:          0.398781  
Distance from current best:     0.0775  
Distance from nearest existing: 0.0665  

Function 5:  
NN selected point: [0.939869 0.992261 0.978815 0.999723]  
NN predicted output:             7103.413178  
NN model disagreement:          20.900199  
Distance from current best:     0.0000  
Distance from nearest existing: 0.0000  

Function 6:  
NN selected point: [0.434907 0.336577 0.469677 0.823758 0.051909]  
NN predicted output:             -0.347172  
NN model disagreement:          0.039748  
Distance from current best:     0.1737  
Distance from nearest existing: 0.1737  

Function 7:  
NN selected point: [0.071621 0.326216 0.438868 0.235047 0.379751 0.461629]  
NN predicted output:             2.977398  
NN model disagreement:          0.203612  
Distance from current best:     0.1979  
Distance from nearest existing: 0.1979  

Function 8:  
NN selected point: [0.01176  0.170084 0.028845 0.085061 0.639117 0.555954 0.195113 0.511384]  
NN predicted output:             10.264397  
NN model disagreement:          0.164985  
Distance from current best:     0.3583  
Distance from nearest existing: 0.3583  

The NN candidates showed varying levels of agreement with the GP analysis. Function 2 is particularly encouraging: after passing LOOCV for the first time, the NN selected a candidate only 0.0210 from the current best, closely matching the low-κ UCB region. Function 4 also showed strong agreement, with the NN selecting exactly the same candidate identified by UCB at κ = 0.5–1.5 and EI at ξ = 0%. Functions 6 and 7 produced plausible NN candidates close to their current high-performing regions, with predicted outputs of −0.3472 and 2.9774 respectively, both better than their current best values.

Function 5's NN returned directly to the existing best observation and therefore cannot be submitted, although this reinforces the model's preference for the successful region identified last week. Function 8 again produced an optimistic prediction of 10.2644, above the current best of 9.9112, but its recent optimisation performance means this should still be treated cautiously. Functions 1 and 3 failed LOOCV, so their NN candidates will receive limited weight despite their predicted outputs.

**Step 5B. Quantitative GP–NN candidate comparison**

The distance between each NN candidate and every candidate generated during the UCB κ and EI ξ tuning was calculated. Smaller distances indicate that the two surrogate approaches are independently identifying similar regions of the search space, while larger distances indicate greater disagreement. This provides a quantitative measure of model agreement to supplement the comparison of predicted outputs and validation performance.

F1: Closest NN–GP distance =	0.3384 =	Clear disagreement  
F2: Closest NN–GP distance =	0.0022 =	Very strong agreement  
F3: Closest NN–GP distance =	0.2358 =	Limited agreement  
F4: Closest NN–GP distance =	0.0000 =	Exact agreement at several settings  
F5: Closest NN–GP distance =	0.0000 =	Agreement with low-κ UCB, but duplicate  
F6: Closest NN–GP distance =	0.1857 =	Moderate agreement at low κ  
F7: Closest NN–GP distance =	0.1785 =	Moderate agreement  
F8: Closest NN–GP distance =	0.2994 =	Some agreement with exploitative settings  

The comparison provides particularly strong evidence of model agreement for Functions 2 and 4. For Function 2, the NN candidate was only 0.0022 from the κ = 0.5 UCB and ξ = 0% EI candidates, suggesting that the GP and NN are independently identifying essentially the same local region. Function 4 showed even stronger agreement, with the NN exactly matching the UCB candidate at κ = 0.5–1.5 and the EI candidate at ξ = 0%. Function 5 also exactly matched the lower-κ UCB recommendations, although this point is already an existing observation and therefore cannot be resubmitted.

Agreement was weaker for Functions 6 and 7, although the NN remained closest to the less exploratory GP recommendations. Function 8 showed some agreement with the most exploitative settings but diverged substantially as greater exploration was introduced. Functions 1 and 3 showed relatively weak agreement, which further reduces the weight given to their NN recommendations given that both also failed LOOCV validation.

**Step 6. Neighbourhood analysis**

The three nearest existing observations to each representative GP and NN candidate were examined to assess whether model recommendations were supported by actual observed performance in the surrounding region. For UCB, κ values of 0.5, 1.5 and 2.5 were selected to represent relatively exploitative, intermediate and exploratory settings. For EI, ξ values of 0%, 5% and 10% of the observed output standard deviation were used for the same reason. The NN candidate was also included as an independent comparison. Using these representative settings allowed the neighbourhoods of locally focused and more exploratory recommendations to be compared without repeating every candidate generated during hyperparameter tuning.

F1:	Both GP and NN candidates are near observations approximately equal to zero, providing little evidence of a clearly superior region.
F2:	Strong local support: GP and NN candidates are very close to the current best of 0.746962 and other relatively strong observations.
F3:	NN is closest to the strongest observations, but its failed LOOCV limits confidence in this evidence.
F4:	Local GP/NN candidate is surrounded by relatively strong observations; highly exploratory GP candidate is instead near very poor outputs.
F5:	Clear high-performing region around the current best, although the NN and low-κ UCB candidate duplicate the existing best.
F6:	NN and low-κ UCB remain relatively close to the three strongest observations; exploratory GP candidates move much further away.
F7:	NN and moderate GP candidates remain close to the new best and several other strong observations.
F8:	NN and low-exploration GP candidates remain closer to the strongest observed outputs than the more exploratory alternatives.

The neighbourhood analysis generally favours the more locally focused recommendations this week. Functions 2 and 4 have particularly strong support, combining good observed neighbourhoods with agreement between the GP and NN. Functions 6 and 7 also show evidence that further refinement around recently successful regions may be preferable to substantially increasing exploration. For Function 8, both the NN and low-exploration GP candidate remain reasonably close to the strongest observed outputs, whereas the more exploratory alternatives move considerably further from observed evidence. Function 5 clearly contains a productive local region, but a new nearby candidate will be required because several methods return the existing best point. Function 1 remains difficult to distinguish, while the favourable local evidence for the Function 3 NN candidate should be treated cautiously because that model failed LOOCV validation.

**Step 7. Final candidate selection**

Final candidates were selected on a function-by-function basis using the combined evidence from the week's analysis. Decisions considered recent optimisation performance, GP predictions and uncertainty, UCB and EI hyperparameter sensitivity, NN LOOCV performance, agreement between the GP and NN, and the observed neighbourhood surrounding each candidate. Greater emphasis was placed on locally supported recommendations where recent queries had successfully improved the objective, while more exploratory candidates were considered where local evidence was weak. Candidates were not selected purely on the highest predicted output or acquisition score.

Provisional selections:  
F1:	GP UCB/EI	NN fails LOOCV - GP settings all agree  
F2:	UCB κ=0.5 / EI ξ=0% region -	Very strong GP–NN agreement and strong neighbourhood  
F3:	UCB κ=0.5	- NN fails LOOCV; lower exploration after another new best  
F4:	UCB κ=0.5 / EI ξ=0% / NN agreement -	Exact cross-model agreement and strong local evidence  
F5:	Generate local refinement -	Models identify existing best, which cannot be resubmitted  
F6:	UCB κ=0.5 -	Recent EI success, but tuning/neighbourhood now favour lower exploration  
F7:	UCB κ=1.5 / EI ξ=0% region -	Move away from duplicate best while retaining strong local support  
F8:	UCB κ=0.5 / EI ξ=0% -	More conservative GP candidate has stronger neighbourhood support  

Function 1 — GP UCB/EI agreement
Selected: [0.858598, 0.697368]
Function 1 remains the least reliable modelling problem. The GP length scale is still fixed at its lower bound of 1e-05 and the NN again failed LOOCV validation. However, all tested UCB and EI hyperparameter settings selected exactly the same GP candidate, whereas the NN disagreed substantially and lacks validation support. With no clearly superior observed region, the unanimous GP recommendation provides the most consistent available evidence.

Function 2 — UCB κ=0.5 / EI ξ=0%
Selected: [0.706079, 0.158030]
Function 2 has unusually strong agreement this week. The NN passed LOOCV for the first time and selected a point only 0.0022 from the κ=0.5 UCB and ξ=0% EI candidate. The selected point is also only 0.0211 from the current best of 0.746962 and the GP predicts 0.784588. Given the strong cross-model agreement and local evidence, further refinement of this region is preferred to increasing exploration.

Function 3 — UCB κ=0.5
Selected: [0.450320, 0.561567, 0.490526]
Function 3 achieved another small improvement last week, increasing the best from −0.016451 to −0.015498. The NN proposes a more locally focused point with a slightly better predicted output, but it continues to fail LOOCV and therefore receives limited weight. Among the GP recommendations, κ=0.5 provides the least exploratory option and predicts −0.018745, while higher κ values progressively move further from the successful region. This provides a controlled balance between refinement and modest exploration.

Function 4 — GP/NN agreement
Selected: [0.384332, 0.391111, 0.336887, 0.416488]
This is one of the strongest selections this week. Function 4 produced a new best of −0.200431 last round, and the NN now exactly agrees with UCB at κ=0.5–1.5 and EI at ξ=0%. The candidate also lies within a relatively strong observed neighbourhood, whereas the more exploratory GP alternative moves into a region associated with substantially poorer outputs. The NN prediction of −0.001964 is optimistic and has relatively high model disagreement, but the exact agreement on candidate location provides stronger evidence than the prediction alone.

Function 5 — GP/NN local refinement
Selected: [1.000000, 1.000000, 1.000000, 1.000000]
The standard NN and lower-κ UCB recommendations returned the existing best observation and therefore could not provide a new query. A dedicated local search was consequently performed around the current best. Both surrogate models strongly supported movement towards the upper boundary, with the selected point receiving a GP prediction of 7920.78 and an NN prediction of 8059.00. The point is only 0.0642 from the current best observation of 7121.15, allowing further exploitation of the highly successful region while still providing a novel query. As the candidate lies on the boundary, the predicted improvement should still be treated cautiously.

Function 6 — UCB κ=0.5
Selected: [0.387946, 0.435172, 0.345126, 0.743513, 0.076825]
Expected Improvement successfully produced a new best last week, improving the output from −0.456874 to −0.379740. However, this week's hyperparameter analysis shows that increasing exploration rapidly moves the GP away from the strongest observed region. The κ=0.5 candidate remains relatively close to the new best and predicts −0.387785, while the validated NN also favours continued refinement rather than substantial exploration. The lower-κ candidate therefore provides a conservative continuation of the successful region.

Function 7 — UCB κ=1.5 / EI agreement
Selected: [0.054205, 0.216830, 0.421532, 0.346756, 0.385001, 0.543902]
The previous GP-selected candidate produced a substantial new best of 2.929370. UCB at κ=0.5 simply returns this existing point and therefore cannot be resubmitted. UCB at κ=1.0–1.5 and the tested EI settings identify the same nearby novel candidate, with a predicted output of 2.620240 and moderate uncertainty. The NN also passes LOOCV and selects a point only 0.1785 away from this recommendation, providing additional support for the same general region.

Function 8 — UCB κ=0.5 / EI ξ=0%
Selected: [0.030327, 0.231267, 0.050266, 0.208295, 0.818520, 0.414744, 0.307096, 0.583908]
The NN continues to pass LOOCV and predicts an improvement to 10.264397, but recent NN-selected queries have not improved the current best. The lower-exploration GP candidate predicts 9.945039 and remains substantially closer to the strongest observed neighbourhood than the more exploratory alternatives. UCB κ=0.5 and EI ξ=0% also agree on this candidate, providing a more conservative test of the promising region.

**Step 8. Final validation and submission**

The eight selected candidates were subjected to a final validation check before submission. Each point was checked for the correct dimensionality, whether all values fell within the permitted [0,1] bounds, whether it duplicated an existing observation, and its distance from the nearest existing point. The final candidates were then formatted to six decimal places for submission.

All eight final candidates passed the validation checks. Each point had the correct dimensionality, all coordinates were within the permitted [0,1] bounds, and none duplicated an existing observation. Distances to the nearest existing observations ranged from 0.0211 for Function 2 to 0.3452 for Function 8, reflecting the different levels of local refinement and exploration selected on a function-by-function basis. Function 5's boundary candidate [1,1,1,1] was also confirmed to be valid and novel. The points were then formatted to six decimal places for submission.


**Overall thoughts on Week 7**

This week represented a further shift towards a more evidence-driven and function-specific optimisation strategy. Five of the eight submitted points from the previous round produced new best values, including particularly strong improvements for Functions 5 and 7. This suggests that the increasing use of model validation, acquisition-function comparison and local evidence is helping to identify productive regions of the search space.

A major focus this week was testing how sensitive the GP recommendations were to different exploration settings. Comparing multiple UCB κ and EI ξ values showed that some functions were relatively stable, while others changed substantially as greater weight was placed on exploration. This reinforced the decision not to apply a single acquisition setting automatically across all functions. The NN also became more useful, passing LOOCV for six functions, including Function 2 for the first time. However, its recommendations were still treated cautiously where validation or recent optimisation performance was weaker.

The final submissions therefore use different strategies depending on the evidence available. Functions 2 and 4 focus on regions where the GP and NN showed particularly strong agreement, while Functions 3, 6 and 8 use relatively conservative GP settings to refine promising areas without moving too far into uncertain regions. Function 7 continues to explore around the region that produced a substantial new best last week. Function 1 remains the most difficult to model reliably, so its selection is lower confidence despite agreement between the GP acquisition settings. Function 5 is the most distinctive submission: repeated improvements have moved the best observation close to the upper boundary, and both models supported further movement towards this boundary, resulting in [1,1,1,1] being selected as a local refinement.

Overall, the submitted points reflect a greater emphasis on combining actual optimisation performance, model validation, acquisition sensitivity, cross-model agreement and observed neighbourhood evidence, rather than relying on any individual model or acquisition function. The next set of outputs should therefore be particularly useful for assessing whether this more selective approach continues to improve performance and whether the strong local trends identified for Functions 2, 4, 5 and 7 persist.

**Module 19 (Week 8)** 

**Methodology**

Step 1 – Add and assess the Week 7 results  
Add the eight new observations and compare each output with the previous best. Record which selections improved and use this to assess the success of last week's strategies.  

Step 2 – Refit and diagnose the models   
Refit the GP models and retrain the neural-network surrogates using the expanded datasets. Review GP length scales and uncertainty, and repeat NN validation to assess model reliability.  

Step 3 – Generate and compare GP candidates
Generate tuned UCB and EI candidates using the settings explored previously. Compare predicted performance, uncertainty, acquisition behaviour and candidate novelty.

Step 4 – Generate and compare NN candidates
Generate updated NN recommendations and interpret them according to LOOCV validation. Compare their predicted performance, disagreement and novelty with the GP recommendations.

Step 5 – Compare GP and NN candidates
Quantify the distance between GP and NN recommendations to identify areas of cross-model agreement and disagreement, while accounting for the reliability of each surrogate.

Step 6 – Neighbourhood and function-specific analysis
Examine nearby observed outputs around the strongest candidates and combine this with recent optimisation performance, model reliability and GP–NN agreement to determine the appropriate exploration–exploitation strategy for each function. 

Step 7 – Select final candidates  
Choose one point per function using the combined evidence, recording the method selected and the reasoning behind each choice.  

Step 8 – Final validation and submission  
Check dimensions, [0,1] bounds, duplicates and distances from existing observations. Format the eight final queries to six decimal places for submission.  

**Step 1 – Add and assess the Week 7 results**

The eight Week 7 query points and their returned outputs were added to the analysis and compared with the previous best result for each function. The difference between the new and previous best was calculated to identify which selections successfully improved the objective. These results provide the first evidence for assessing which of last week's function-specific strategies were successful before updating the datasets for Week 8.

Function 1:  
Previous best: 0.000000  
New output: 0.000000  
Result: Did not improve  

Function 2:  
Previous best: 0.746962  
New output: 0.569904  
Result: Did not improve  

Function 3:  
Previous best: −0.015498  
New output: −0.010190  
Result: NEW BEST  

Function 4:  
Previous best: −0.200431  
New output: 0.186912  
Result: NEW BEST  

Function 5:  
Previous best: 7121.151955  
New output: 8662.482500  
Result: NEW BEST  

Function 6:  
Previous best: −0.379740  
New output: −0.388687  
Result: Did not improve  

Function 7:  
Previous best: 2.929370  
New output: 2.369289  
Result: Did not improve  

Function 8:  
Previous best: 9.911249  
New output: 9.930811  
Result: NEW BEST  

The Week 7 queries produced four new bests across the eight functions. Functions 3 and 8 achieved smaller improvements, while Functions 4 and 5 showed much stronger gains. Function 4 moved from −0.200431 to 0.186912, while Function 5 increased substantially from 7121.15 to 8662.48.  
Of the unsuccessful queries, Function 6 remained very close to its existing best, whereas Functions 2 and 7 showed clearer declines. Function 2 is particularly informative because its candidate had strong GP–NN agreement last week, showing that agreement between models does not necessarily guarantee improvement. Function 1 again returned an effectively zero result and remains difficult to optimise reliably.  
Overall, the mixed results reinforce the value of assessing historical method performance separately for each function, rather than assuming that the same evidence or acquisition strategy will be equally successful across all eight functions.  

**Step 2. Refit and diagnose the models**  

The GP and neural-network surrogate models were refitted using the expanded Week 8 datasets and assessed separately using the same configurations as previous weeks. Keeping the model configurations unchanged allows changes in performance to be attributed mainly to the additional observations.  
For the Gaussian Processes, the fitted kernels and length scales were examined first to identify any continuing reliability issues. Prediction and uncertainty ranges were then assessed across 100,000 candidate points for each function to determine how effectively each GP differentiates between unexplored locations. Particular attention was given to Functions 1 and 2, where the fitted length scales reached the lower bound.  
The neural-network surrogates were then retrained separately using the established configuration. Their reliability was again assessed using LOOCV against the mean-prediction baseline, with NN recommendations given greater weight only where validation indicates that the model can generalise beyond its training observations.  

**Step 2a. GP refitting and diagnosis**

Function 1: RBF length scale = 0.00001  
Function 2: RBF length scale = 0.00001  
Function 3: RBF length scale = 0.199  
Function 4: RBF length scale = 0.416  
Function 5: RBF length scale = 0.273  
Function 6: RBF length scale = 0.581  
Function 7: RBF length scale = 0.302  
Function 8: RBF length scale = 1.04  

The refitted GP models remained relatively stable for Functions 3–8, with fitted length scales ranging from 0.199 to 1.04. Functions 3, 4, 6, 7 and 8 changed only modestly following the addition of the latest observations, while Function 5's length scale decreased from 0.363 to 0.273 following the strong boundary result.  
The most significant change occurred for Function 2, where the fitted length scale fell from 0.0431 to the lower bound of 1e-05, producing the same convergence warning already associated with Function 1. This suggests that the GP is now treating observations for Function 2 as varying over extremely short distances and may have difficulty generalising between them. Function 1 remains at the lower bound for another week.  
Overall, the GP currently appears more reliable for Functions 3–8 than Functions 1 and 2, although this will be assessed further using prediction and uncertainty behaviour before candidate selection.  

Function 1: Predicted range = 0.000560 | Uncertainty range = 0.001570  
Function 2: Predicted range = 0.651312 | Uncertainty range = 0.244076  
Function 3: Predicted range = 0.388723 | Uncertainty range = 0.078255  
Function 4: Predicted range = 35.821887 | Uncertainty range = 7.855238  
Function 5: Predicted range = 7847.668688 | Uncertainty range = 2510.829993  
Function 6: Predicted range = 2.303126 | Uncertainty range = 0.523703  
Function 7: Predicted range = 3.026197 | Uncertainty range = 0.818402  
Function 8: Predicted range = 4.878650 | Uncertainty range = 0.653663  

The prediction and uncertainty diagnostics show meaningful variation across most functions, but Functions 1 and 2 require particular caution because both fitted length scales reached the lower bound.  
Function 1 continues to produce extremely small predicted values and remains difficult for the GP to model reliably. Function 2 is more interesting: despite its length scale collapsing to 1e-05, the GP still produces a relatively wide prediction range of −0.0817 to 0.5696. However, its maximum prediction is now below the existing best of 0.7470, and the lower-bound length scale suggests that these predictions should not be relied upon heavily for selecting the next query.  
Functions 3–8 continue to show substantial variation in both predicted output and uncertainty, suggesting that their GPs remain useful for distinguishing between candidate regions. Functions 4 and 5 have particularly large uncertainty ranges, meaning acquisition-based selection will remain important rather than relying solely on the highest predicted mean.  
Overall, the results suggest that the GP remains useful for Functions 3–8, while Functions 1 and 2 should receive additional scrutiny and may require greater support from alternative evidence before final candidate selection.  

**Step 2B. NN refitting and validation**

Function 1: Mean training loss = 0.000963  
Function 2: Mean training loss = 0.010989  
Function 3: Mean training loss = 0.000408  
Function 4: Mean training loss = 0.000284  
Function 5: Mean training loss = 0.000230  
Function 6: Mean training loss = 0.000370  
Function 7: Mean training loss = 0.000338  
Function 8: Mean training loss = 0.000381  

The retrained NN ensembles again achieved low training losses across most functions. Functions 3–8 all recorded mean losses below 0.0004, indicating that the networks continue to fit the observed data closely. Function 1 was slightly higher at 0.000963 but remained similar to the previous round.  
Function 2 was the clear exception, with its mean training loss increasing from 0.006460 last week to 0.010989. This is particularly notable because its GP length scale also collapsed to the lower bound this week, suggesting that Function 2 may have become more difficult for both surrogate approaches to represent.  
Overall, the training results remain encouraging for most functions, but low training error alone does not demonstrate reliable prediction on unseen observations. LOOCV against the mean baseline therefore remains the more important test before deciding how much weight to give the NN for each function.  

**NN LOOCV validation**

Function 1:  
NN LOOCV MAE = 0.001704  
Mean baseline MAE = 0.001050  
NN beats baseline = False  

Function 2:  
NN LOOCV MAE = 0.219676  
Mean baseline MAE = 0.237993  
NN beats baseline = True  

Function 3:  
NN LOOCV MAE = 0.062478  
Mean baseline MAE = 0.051662  
NN beats baseline = False  

Function 4:  
NN LOOCV MAE = 1.825570  
Mean baseline MAE = 7.284354  
NN beats baseline = True  

Function 5:  
NN LOOCV MAE = 230.163312  
Mean baseline MAE = 2049.255132  
NN beats baseline = True  

Function 6:  
NN LOOCV MAE = 0.241315  
Mean baseline MAE = 0.513894  
NN beats baseline = True  

Function 7:  
NN LOOCV MAE = 0.399610  
Mean baseline MAE = 0.639930  
NN beats baseline = True  

Function 8:  
NN LOOCV MAE = 0.354980  
Mean baseline MAE = 0.962395  
NN beats baseline = True  

The LOOCV results show that the NN continues to outperform the mean-prediction baseline for six of the eight functions. Functions 4, 5, 6 and 8 again show particularly clear improvements over the baseline, providing continued support for using their NN models as secondary evidence during candidate selection. Function 7 also remains validated.  
Function 2 continues to pass LOOCV, although only narrowly, with an MAE of 0.219676 compared with 0.237993 for the baseline. This is important given the deterioration in its GP behaviour this week: while the NN also showed a higher training loss, its validation performance suggests that it still retains some predictive value and may provide useful alternative evidence where the GP is unreliable.  
Functions 1 and 3 again fail to outperform the baseline. Their NN recommendations should therefore continue to receive limited weight, particularly for Function 1 where both the GP and NN now provide weak modelling evidence.  
Overall, the additional observations have not substantially changed the NN reliability pattern. The NN remains useful for Functions 2 and 4–8, while Functions 1 and 3 continue to require greater reliance on other evidence.  

**Step 3. Generate and compare GP candidates**

The refitted GP models were used to generate candidate points using both Upper Confidence Bound (UCB) and Expected Improvement (EI). Based on the hyperparameter analysis conducted in the previous round, representative settings were selected rather than repeating the full tuning exercise. UCB was evaluated at κ = 0.5 and 1.5 to represent relatively exploitative and moderately exploratory strategies, while EI was evaluated using ξ = 0% and 5% of each function's observed output standard deviation.  
For each candidate, the predicted output, uncertainty, acquisition score and distance from the nearest existing observation were recorded. The candidates will be compared with recent optimisation performance and model reliability rather than selecting automatically from the highest acquisition score. Greater caution will be applied to Functions 1 and 2 because their GP length scales reached the lower bound during Step 2.  

F1: All four settings selected the same candidate [0.094177, 0.975622], showing complete stability across UCB and EI.  

F2: UCB κ=0.5 returned the previously submitted point, while UCB κ=1.5 and both EI settings selected the same more exploratory candidate [0.403694, 0.391789].  

F3: UCB κ=0.5 remained relatively local, only 0.0753 from an existing observation, while increasing exploration moved the recommendations progressively further away. EI ξ=0% provided an intermediate candidate.  

F4: Both UCB candidates remained close to the new successful region and predicted outputs above the current best. In contrast, both EI settings selected a highly exploratory point 0.5460 from the nearest observation, with a very poor predicted mean but high uncertainty.  

F5: Both UCB settings remained close to the successful upper-boundary region, whereas EI moved to a distant and highly uncertain region. None of the GP candidates predicted an improvement over the new best of 8662.48.  

F6: UCB κ=0.5 produced the most locally focused candidate and predicted a small improvement over the current best. Increasing κ or ξ progressively increased the distance and uncertainty.  

F7: UCB κ=0.5 effectively returned the existing best and is therefore unsuitable as a new query. UCB κ=1.5 and both EI settings agreed exactly on the same novel, more exploratory candidate.  

F8: UCB κ=0.5 remained substantially more local than the other recommendations and predicted a small improvement over the new best. Higher κ and both EI settings moved much further into unexplored regions.  

The GP candidates show that the appropriate exploration level continues to vary substantially between functions. Functions 3, 4, 5, 6 and 8 provide a clear distinction between locally focused and more exploratory recommendations, allowing recent optimisation performance to help determine which strategy should receive greater weight.  
The latest results provide particularly strong evidence for continued local refinement in Functions 4 and 5, which both achieved substantial new bests in the previous round. Their more exploratory EI candidates are driven largely by uncertainty and have substantially poorer predicted outputs. Function 6 also provides support for local refinement, while Function 8's recent improvement gives some support to the lower-κ UCB candidate.  
Functions 1 and 2 require greater caution because of their unreliable GP length scales. Function 1 nevertheless shows complete acquisition stability, with all settings selecting the same point. For Function 2, the exploitative UCB candidate returns the previous query while the remaining settings favour a substantially different region, meaning the GP evidence alone is insufficient to identify a preferred candidate. Function 7 similarly requires a novel point because its most exploitative recommendation effectively returns the existing best.  
Overall, the results show that recent optimisation performance and GP reliability should continue to be considered alongside acquisition scores when determining the appropriate exploration level for each function.  

**Step 4. Generate and compare NN candidates**

The refitted five-model NN ensembles were used to generate an alternative candidate for each function using the same 100,000 candidate points assessed by the GP. Candidate selection was based on the highest mean prediction across the ensemble, while disagreement between the five models was recorded as an indication of prediction consistency. Distance from the nearest existing observation was also calculated.  
The NN candidates were then compared with the GP recommendations from Step 3 to identify areas of cross-model agreement or disagreement. Greater weight was given to NN recommendations for Functions 2 and 4–8, which passed LOOCV validation, while candidates for Functions 1 and 3 were retained mainly for comparison because their NNs failed to outperform the baseline.  

F1: NN selected [0.724967, 0.999094], with predicted output 0.001008 and disagreement 0.000982. The candidate differs substantially from the GP recommendation, while the NN failed LOOCV.  

F2: NN selected [0.697103, 0.994806], predicting 0.696942 with disagreement 0.049182. The candidate is only 0.0685 from an existing observation but lies far from the current best point. It also differs substantially from the GP candidates.  

F3: NN selected [0.437451, 0.559044, 0.487388], predicting −0.009572. This is only 0.0135 from the new best point and suggests further local refinement, although the NN again failed LOOCV.  

F4: NN returned the existing new best [0.384332, 0.391111, 0.336887, 0.416488], with extremely low disagreement. This reinforces the NN's preference for the successful region but cannot provide a novel query.  

F5: NN selected [0.995347, 0.980915, 0.954873, 0.970449], exactly matching both UCB candidates from Step 3. The point is only 0.0574 from the successful [1,1,1,1] observation, providing strong cross-model support for continued refinement of the upper-boundary region.  

F6: NN selected [0.434907, 0.336577, 0.469677, 0.823758, 0.051909], predicting −0.345084, which would improve on the current best of −0.379740. The candidate remains relatively local and is broadly consistent with the lower-exploration GP region.  

F7: NN returned the existing best point, with very low model disagreement. Like UCB κ = 0.5, this confirms the model's preference for the established high-performing region but requires a nearby novel alternative.  

F8: NN selected a more exploratory point 0.4480 from the current best and predicts 10.207430, above the current best of 9.930811. The NN passes LOOCV, although its recommendation differs from the GP candidates and should therefore be assessed alongside recent method performance and neighbourhood evidence.  

Interpretation:

The NN results provide particularly strong support for the existing successful regions of Functions 4, 5 and 7. Functions 4 and 7 return their existing best observations, while Function 5 shows exact GP–NN agreement on a nearby novel point. Given the substantial improvement produced by Function 5's boundary query last week, this agreement provides strong evidence that the upper-boundary region remains important.  
Function 6 also shows encouraging evidence, with the validated NN predicting an improvement and remaining broadly consistent with the locally focused GP recommendations. Function 8's validated NN again predicts an improvement, but its candidate differs from the GP recommendations, meaning further comparison is needed before deciding whether to favour the NN or GP region.  
Functions 1 and 3 continue to have unvalidated NN models and their recommendations should therefore receive limited weight. Function 2 is more difficult: its NN remains validated but selects a substantially different region from the GP, while the GP itself is currently unreliable. This makes Function 2 a particularly important case for additional neighbourhood and historical evidence.  
Overall, the NN analysis strengthens the case for local refinement for several functions but does not yet resolve F1, F2 or F8. The next step should quantitatively compare the GP and NN candidate locations, as we did last week, before moving into neighbourhood analysis and final selection.  

**Step 5. GP–NN candidate comparison**

The distance between each NN candidate and the four representative GP candidates was calculated to quantify the level of agreement between the two surrogate approaches. Smaller distances indicate that the GP and NN are independently identifying similar regions, while larger distances indicate greater disagreement. These distances were considered alongside LOOCV validation and GP reliability, so close agreement was treated as stronger evidence only where the underlying models were considered sufficiently reliable.

Function 1:  
NN -> UCB k=0.5: 0.6312
NN -> UCB k=1.5: 0.6312
NN -> EI xi=0%: 0.6312
NN -> EI xi=5%: 0.6312
- Closest NN–GP distance = 0.6312 — strong disagreement

Function 2:  
NN -> UCB k=0.5: 0.8368
NN -> UCB k=1.5: 0.6706
NN -> EI xi=0%: 0.6706
NN -> EI xi=5%: 0.6706
- Closest NN–GP distance = 0.6706 — strong disagreement

Function 3:  
NN -> UCB k=0.5: 0.0751
NN -> UCB k=1.5: 0.5999
NN -> EI xi=0%: 0.5790
NN -> EI xi=5%: 0.5999
- Closest NN–GP distance = 0.0751 — strong local agreement with UCB κ = 0.5

Function 4:  
NN -> UCB k=0.5: 0.0419
NN -> UCB k=1.5: 0.0591
NN -> EI xi=0%: 0.9254
NN -> EI xi=5%: 0.9254
- Closest NN–GP distance = 0.0419 — strong agreement with local UCB candidates

Function 5:  
NN -> UCB k=0.5: 0.0000
NN -> UCB k=1.5: 0.0000
NN -> EI xi=0%: 1.3683
NN -> EI xi=5%: 1.3683
- Closest NN–GP distance = 0.0000 — exact agreement with both UCB candidates

Function 6:  
NN -> UCB k=0.5: 0.0920
NN -> UCB k=1.5: 0.2137
NN -> EI xi=0%: 0.1762
NN -> EI xi=5%: 0.2137
- Closest NN–GP distance = 0.0920 — strong agreement with UCB κ = 0.5

Function 7:  
NN -> UCB k=0.5: 0.0000
NN -> UCB k=1.5: 0.2220
NN -> EI xi=0%: 0.2220
NN -> EI xi=5%: 0.2220
- Closest NN–GP distance = 0.0000 — exact agreement with UCB κ = 0.5, although this is the existing best

Function 8:  
NN -> UCB k=0.5: 0.3396
NN -> UCB k=1.5: 1.0053
NN -> EI xi=0%: 1.0736
NN -> EI xi=5%: 1.0736
- Closest NN–GP distance = 0.3396 — moderate disagreement, with NN closest to UCB κ = 0.5

The comparison provides particularly strong evidence of GP–NN agreement for Functions 4, 5, 6 and 7. Function 5 shows the clearest useful agreement, with the NN and both UCB settings selecting exactly the same novel point close to the successful upper-boundary region. Function 6 also shows strong agreement between the validated NN and the locally focused UCB candidate. For Function 4, the NN returns the existing best but lies only 0.0419 from the κ = 0.5 UCB candidate, supporting continued refinement of the same successful region. Function 7 also shows exact agreement between the NN and κ = 0.5 UCB, although both effectively return the existing best and therefore do not provide a novel query.  

Function 3 shows close agreement between the NN and κ = 0.5 UCB, but this evidence should be treated cautiously because the NN failed LOOCV. Function 8 shows weaker agreement, with the NN remaining closest to the locally focused UCB candidate but still 0.3396 away. This suggests that the two validated surrogate approaches favour different parts of the broader promising region.  

Functions 1 and 2 show the strongest disagreement between the GP and NN. For Function 1, neither surrogate currently provides strong validation evidence, so the disagreement does little to resolve candidate selection. Function 2 is more informative because the NN continues to pass LOOCV while the GP length scale has collapsed to its lower bound. The large separation between their recommendations therefore provides further reason not to rely on GP–NN agreement alone for this function.  

Overall, the comparison strengthens the evidence for local refinement in Functions 4–7, particularly Function 5 where a novel candidate has exact cross-model support. Functions 1, 2 and 8 remain less clearly resolved and will benefit from examining the observed performance surrounding the leading candidate regions.  

**Step 6. Neighbourhood analysis**

The three nearest existing observations to each representative GP and NN candidate were examined to assess whether the recommendations were supported by actual observed performance in the surrounding region. The UCB κ = 0.5 and κ = 1.5 candidates, EI ξ = 0% and ξ = 5% candidates, and the NN candidate were included.  
Neighbourhood evidence was considered alongside model reliability and recent optimisation performance. Particular attention was given to Functions 1, 2 and 8, where the preceding analysis showed substantial GP–NN disagreement, and to Functions 4–7, where the models showed stronger agreement around established high-performing regions.  

Function 1: Neither GP nor NN region has meaningful local support; nearby observations are approximately zero.  

Function 2:  
NN nearest outputs: 0.611205, 0.366010, 0.507418  
Exploratory GP nearest outputs: 0.214965, −0.013858, −0.065624  
The NN region has stronger observed support.  

Function 3:  
Local UCB nearest outputs: −0.010190, −0.016451, −0.020167  
The more exploratory GP regions are associated with weaker observations.  

Function 4:   
Local UCB nearest outputs: 0.186912, −0.200431, −0.300645  
Exploratory EI nearest outputs: −16.507159, −7.966775, −6.702089  
Local refinement is substantially better supported.  

Function 5:  
Local UCB/NN nearest outputs: 8662.482500, 6691.931674, 7121.151955  
Exploratory EI nearest outputs: 64.443440, 78.434389, 9.972332  
There is very strong evidence for continued local refinement.  

Function 6:  
Local UCB nearest outputs: −0.379740, −0.388687, −0.458511  
The more exploratory candidates move further from the strongest observations.  

Function 7:  
Leading neighbourhood outputs: 2.929370, 2.536939, 2.369289  
Both local and moderately exploratory candidates remain within the established high-performing region.  

Function 8:  
Low-κ UCB nearest outputs: 9.930811, 9.821986, 9.840530  
NN nearest outputs: 9.930811, 9.911249, 9.821986  
Both have strong neighbourhood evidence, although the UCB candidate is closer to existing observations.  

The neighbourhood analysis provides strong support for a locally focused strategy across most functions this week. Functions 4 and 5 provide the clearest evidence: both achieved substantial improvements in the previous round, and their local candidates remain surrounded by the strongest observed outputs, whereas the exploratory EI regions are associated with much poorer historical performance.  
The NN candidate for Function 2 lies in a region containing several relatively strong observations, including 0.611205, 0.366010 and 0.507418. In contrast, the exploratory GP candidate is surrounded by substantially weaker observed outputs. Given that the GP length scale has collapsed to its lower bound while the NN continues to outperform its LOOCV baseline, the NN region currently has stronger supporting evidence. However, the NN prediction of 0.696942 remains below the current best of 0.746962, so the selection should be interpreted as refinement of the best-supported region rather than evidence of a likely improvement over the existing optimum.
Functions 3 and 6 similarly favour continued refinement around their established high-performing regions. Function 7 also has a clearly strong neighbourhood, although a novel candidate is required because the NN and low-κ UCB return the existing best. For Function 8, both the NN and low-κ UCB remain associated with strong observations, but the lower-κ GP candidate has stronger local support and remains closer to the observed data.  
Function 1 remains unresolved, as neither candidate region has meaningful neighbourhood evidence capable of distinguishing between them.  
Overall, the neighbourhood results reinforce the evidence from the preceding steps and give us a much stronger basis for function-specific final candidate selection rather than choosing candidates solely from model predictions or acquisition scores.  

**Step 6b. Additional Function 1 model diagnosis**

Function 1 was examined separately before final candidate selection because neither of the existing surrogate models currently provides strong evidence for the next query. The RBF Gaussian Process again fitted a length scale at the lower bound of 1e-05, suggesting that it is struggling to generalise between observations, while the neural network again failed to outperform the mean baseline under LOOCV. The GP and NN also selected substantially different candidates, with a distance of 0.6312 between their recommendations. Although all tested UCB and EI settings produced the same GP candidate, this stability does not necessarily indicate a reliable recommendation because all acquisition functions depend on the same problematic GP model.  

To investigate whether this behaviour is specific to the RBF kernel rather than Function 1 itself, several alternative GP configurations were therefore compared using Leave-One-Out Cross-Validation. The existing RBF was compared with Matérn 3/2, Matérn 5/2 and an RBF with a WhiteKernel noise component. LOOCV MAE was used to assess each model against the same mean-prediction baseline. The comparison was used to determine whether an alternative kernel could provide stronger evidence for Function 1 candidate selection, or whether continued poor validation would support placing greater emphasis on exploration rather than surrogate predictions.  

RBF: LOOCV MAE = 0.001099 | Baseline MAE = 0.001050 | Beats baseline = False  
Matérn 3/2: LOOCV MAE = 0.001106 | Baseline MAE = 0.001050 | Beats baseline = False  
Matérn 5/2: LOOCV MAE = 0.001107 | Baseline MAE = 0.001050 | Beats baseline = False  
RBF + noise: LOOCV MAE = 0.001099 | Baseline MAE = 0.001050 | Beats baseline = False  

Interpretation:  

None of the alternative GP configurations outperformed the mean-prediction baseline under LOOCV. The standard RBF produced the lowest validation error of the tested kernels, with an MAE of 0.001099 compared with the baseline of 0.001050, while both Matérn alternatives performed slightly worse. Adding a WhiteKernel noise component produced almost identical validation performance to the standard RBF.  
The fitted kernels also continued to indicate very short-range behaviour. Although the RBF fitted to the complete dataset produced a length scale of 0.00409 rather than the lower-bound value observed during the earlier GP diagnosis, this remains very small relative to the [0,1] search space. The RBF + noise model also fitted an extremely small noise level, providing little evidence that observational noise explains the modelling difficulty.  
Overall, the results suggest that Function 1's poor GP behaviour is not simply caused by the choice of RBF kernel. Changing the assumed smoothness or explicitly modelling noise did not improve generalisation. Combined with the NN's failure to outperform its baseline, neither surrogate currently provides strong validation evidence for Function 1.  

Implications for candidate selection:  

Because neither surrogate has demonstrated reliable predictive performance, the next Function 1 query will not be selected solely from the predicted optimum of either model. Before final candidate selection, a model-independent space-filling analysis will therefore be performed to identify points that are maximally distant from the existing observations.  
These exploratory candidates will be compared with the GP and NN recommendations. This provides an alternative way of selecting an informative query when the available surrogate models cannot reliably identify the optimum, while still allowing any agreement with the model-based recommendations to be considered.  

**Function 1 space-filling exploration**

As neither the GP nor NN demonstrated reliable predictive performance for Function 1, a model-independent space-filling analysis was used as an additional source of evidence. A set of 100,000 candidate points was generated across the [0,1]² search space and the distance from each candidate to its nearest existing observation was calculated. Candidates with the greatest minimum distance were identified as the most underexplored locations.  
The five leading candidates were retained for comparison, with a minimum separation of 0.10 imposed between selected candidates to avoid returning several nearly identical points from the same unexplored region. This approach does not attempt to predict where the objective will be highest; instead, it identifies locations that would provide the greatest additional coverage of the search space when the available surrogate models are unreliable.  

Space-filling exploration results  
Candidate 1: [0.998294, 0.000709] | Nearest-observation distance = 0.307658  
Candidate 2: [0.684654, 0.001334] | Nearest-observation distance = 0.305975   
Candidate 3: [0.006886, 0.695109] | Nearest-observation distance = 0.301267  
Candidate 4: [0.786713, 0.000852] | Nearest-observation distance = 0.269276  
Candidate 5: [0.898148, 0.005297] | Nearest-observation distance = 0.265794  

Interpretation:  
The space-filling analysis identified several substantially underexplored regions of the Function 1 search space. The three leading candidates are all approximately 0.30 from their nearest existing observation, compared with distances of 0.0953 for the GP recommendation and 0.1442 for the NN recommendation. The space-filling alternatives would therefore provide considerably more novel information than either surrogate-based candidate.  
The strongest unexplored regions are concentrated around the boundaries of the search space, particularly where the second input is close to zero. Candidate 1 [0.998294, 0.000709] has the greatest minimum distance from the existing observations, although Candidates 2 and 3 provide almost the same degree of novelty in different regions. Because this analysis is deliberately model-independent, the candidates should not be interpreted as having higher predicted outputs; their value is that they sample regions about which the existing dataset contains relatively little information.  
Given that neither the GP nor NN has demonstrated reliable predictive performance for Function 1, this provides a stronger justification for deliberate exploration than relying solely on either model's predicted optimum. The final Function 1 candidate will therefore be selected by considering these space-filling alternatives alongside the GP and NN recommendations and the historical performance of previously explored regions.  

**Final Function 1 exploratory candidate comparison**  

The three strongest space-filling candidates were compared before selecting the final Function 1 query. As their nearest-observation distances were very similar, space coverage alone provided little basis for choosing between them. Each candidate was therefore assessed using its three nearest observed outputs, together with predictions from the existing GP and NN models.  
Because neither surrogate passed validation for Function 1, their predictions were treated only as secondary evidence rather than reliable estimates of performance. The primary objective remained selecting an informative point in an underexplored region, with historical observations and model predictions used only to distinguish between candidates with similar space-filling value.  

Candidate 1: [0.998294, 0.000709]  
Nearest distance = 0.307658 | GP prediction = −0.000560 | NN prediction = 0.000593  

Candidate 2: [0.684654, 0.001334]  
Nearest distance = 0.305975 | GP prediction = −0.000560 | NN prediction = 0.000432  

Candidate 3: [0.006886, 0.695109]  
Nearest distance = 0.301267 | GP prediction = −0.000560 | NN prediction = −0.000247  

Interpretation:  
The three candidates provide almost identical levels of space-filling exploration, with nearest-observation distances between 0.3013 and 0.3077. Their three nearest observed outputs are also all approximately zero, meaning the existing observations provide little basis for distinguishing between the regions.    
The GP provides no additional discrimination, assigning exactly the same predicted output and uncertainty to all three candidates. This is consistent with its extremely short fitted length scale and reinforces the limited usefulness of its predictions for Function 1. The NN provides greater differentiation, favouring Candidate 1, followed by Candidate 2, but these predictions should be interpreted cautiously because the NN failed LOOCV validation. Candidate 1 also has greater NN disagreement than Candidates 2 and 3.    
Overall, none of the available evidence identifies a clearly superior region. However, Candidate 1 [0.998294, 0.000709] provides the greatest space-filling value while also receiving the highest NN prediction of the three alternatives. Although the NN prediction is not sufficiently reliable to justify the selection independently, it provides secondary support for the candidate that already maximises model-independent exploration. Candidate 1 will therefore be carried forward as the leading Function 1 candidate for final selection.  


**Step 7. Final candidate selection**

Final candidates will be selected on a function-by-function basis using the combined evidence from the Week 8 analysis. Decisions will consider recent optimisation performance, GP reliability, UCB and EI behaviour, NN LOOCV validation, GP–NN agreement, candidate novelty and observed neighbourhood performance.  
Greater emphasis will be placed on local refinement where recent queries have improved the objective and where the surrounding observations support a promising region. More exploratory candidates will be considered where local evidence is weak or where the models indicate that additional exploration may be beneficial. Candidates will not be selected solely on predicted output or acquisition score, and existing observations will not be resubmitted.  
For Functions 4, 5 and 7, the accumulated evidence favoured continued exploitation of established high-performing regions, but some of the strongest model recommendations corresponded to existing observations. Additional local searches were therefore performed around the current best point for each function. Candidate points were generated within a small neighbourhood of the current best, constrained to the [0,1] search space, and existing observations were excluded. The remaining candidates were evaluated using the GP and validated NN models, allowing novel points with strong cross-model support to be identified.  

Function 1:	Space-filling exploration	(GP alternatives and NN fail validation; Candidate 1 maximises exploration with limited secondary NN support)  
Function 2:	NN-supported region	(GP unreliable; NN validated and has stronger neighbourhood evidence)  
Function 3:	Local GP refinement	(New best; NN fails LOOCV; κ=0.5 locally supported)  
Function 4:	Local refinement	(Major new best; GP/NN support the same local region)  
Function 5:	Local boundary refinement	(Huge new best at [1,1,1,1]; GP and NN support continued refinement near the upper boundary)  
Function 6:	Local GP/NN region	(Both models and neighbourhood evidence support continued refinement)  
Function 7:	Local novel refinement	(Strong local region, but leading recommendations duplicate existing observations)  
Function 8:	κ=0.5/local region	(New best; strong neighbourhood evidence and more conservative than NN)    

Local refinement results:  
The local searches produced suitable novel candidates for Functions 4 and 7. Function 4 identified [0.386248, 0.381010, 0.343907, 0.423092], only 0.0141 from the current best, with both the GP and NN continuing to predict strong performance in this region. Function 7 similarly identified [0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025], with GP and NN predictions of 2.920647 and 2.936110 respectively, providing support for continued local refinement.  
The initial Function 5 search produced [1.000000, 1.000000, 0.999995, 1.000000], only 0.000005 from the existing best [1,1,1,1]. Although technically novel, this point was considered too close to the existing observation to provide meaningful additional information. A minimum nearest-observation distance of 0.03 was therefore introduced and the local search repeated.  
The constrained Function 5 search selected [1.000000, 1.000000, 0.970000, 1.000000]. The GP predicts 8391.49 and the NN predicts 8342.42, compared with the current best of 8662.48. Both models therefore continue to predict strong performance close to the upper boundary, while the novelty constraint provides a more informative query than an almost identical resubmission of the current best.  

Final candidate decisions:  
Based on the combined evidence above, one final candidate was selected for each function. Where the models provided reliable and consistent evidence, candidates were chosen from the strongest supported regions. Where model reliability was weaker, greater weight was given to observed performance, candidate novelty and exploration.  

Function 1 – Space-filling exploration  
Selected candidate: [0.998294, 0.000709]  

Neither the GP nor NN demonstrated reliable predictive performance, and the alternative GP kernels also failed to outperform the baseline. The selected candidate provides the greatest space-filling value, with a nearest-observation distance of 0.3077, while the NN provides limited secondary support for this choice.  

Function 2 – NN-supported region  
Selected candidate: [0.697103, 0.994806]  

The GP is given reduced weight because its fitted length scale reached the lower bound. The NN continues to outperform the mean baseline under LOOCV and identifies this candidate as its preferred region. The surrounding observed outputs also provide stronger support for this region than the more exploratory GP alternatives.  

Function 3 – Local GP refinement  
Selected candidate: [0.463879, 0.497888, 0.452731]  

The latest query produced a new best of −0.010190, supporting continued local refinement. The NN failed to outperform its baseline, so greater weight is given to the GP. The UCB κ = 0.5 candidate remains relatively close to the successful region and has strong nearby observed outputs.  

Function 4 – Local refinement  
Selected candidate: [0.386248, 0.381010, 0.343907, 0.423092]  

The latest observation produced a substantial new best of 0.186912. The selected local candidate remains close to this successful point while avoiding an exact resubmission. Both models continue to predict strong performance in the region, supporting further local refinement.  
  
Function 5 – Local boundary refinement  
Selected candidate: [1.000000, 1.000000, 0.970000, 1.000000]  

The previous [1,1,1,1] query produced a major new best of 8662.482500, providing strong evidence for continued boundary refinement. The novelty-constrained candidate remains close to this successful region while maintaining a distance of 0.03 from the existing best. Both GP and NN continue to predict high outputs at the selected point.  

Function 6 – Local GP/NN refinement  
Selected candidate: [0.426973, 0.384648, 0.468006, 0.752822, 0.084308]  

The selected UCB κ = 0.5 candidate remains within the strongest observed region and has a GP prediction slightly above the current best. The validated NN also favours a nearby region, providing additional support for continued local refinement rather than substantially greater exploration.  

Function 7 – Local novel refinement  
Selected candidate: [0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025]  

Both models provided strong evidence for remaining close to the current best, although their leading recommendations included an existing observation. The local search therefore identified a novel candidate 0.0479 from the current best. The GP and NN predict 2.920647 and 2.936110 respectively, providing good cross-model support.  

Function 8 – Local UCB refinement  
Selected candidate: [0.026226, 0.080912, 0.128250, 0.056836, 0.748250, 0.399531, 0.253527, 0.720750]  
  
The latest query produced a new best of 9.930811, supporting continued refinement. The UCB κ = 0.5 candidate provides a more local continuation of the successful region than the more exploratory alternatives, and its surrounding observations include several of the strongest outputs seen for Function 8.  

**Step 8. Final validation and submission**

Before submission, the eight selected candidates were checked to ensure that each had the correct dimensionality, all input values remained within the permitted [0,1] bounds, and no candidate duplicated an existing observation. The distance from each selected candidate to its nearest existing observation was also calculated as a final check on candidate novelty. The validated candidates were then formatted to six decimal places for submission.  

All eight final candidates passed the validation checks. Each candidate had the correct dimensionality, remained within the permitted [0,1] search space and was distinct from all existing observations.  
Nearest-observation distances ranged from 0.0141 for Function 4 to 0.3077 for Function 1, reflecting the different strategies used across the functions. Functions 4, 5 and 7 deliberately remain relatively close to existing high-performing observations as part of the local-refinement strategy, while Function 1 is substantially further from the existing dataset because model-independent exploration was prioritised.  

**Module 20 (Week 9)**

**Methodology**

Step 1 – Add and assess the Week 8 results
Add the eight new observations and compare each output with the previous best. Assess the success of last week's function-specific strategies and identify evidence of continued improvement, plateauing or diminishing returns.

Step 2 – Assess performance as the datasets scale
Review how optimisation performance and model reliability have changed as observations have accumulated. Compare recent predictions with subsequently observed outputs and consider whether additional data is producing steady improvements, diminishing returns or unexpected changes in model behaviour.

Step 3 – Refit and diagnose the models
Refit the GP models and retrain the NN surrogates using the expanded datasets. Review GP length scales and uncertainty and repeat NN LOOCV validation, paying particular attention to whether previously weak models are becoming more reliable as additional observations are added.

Step 4 – Generate and compare GP candidates
Generate updated UCB and EI candidates using the established settings. Compare predicted performance, uncertainty, acquisition behaviour and candidate novelty.

Step 5 – Generate and compare NN candidates
Generate updated NN recommendations and interpret them according to LOOCV performance. Compare their predictions, disagreement and novelty with the GP recommendations.

Step 6 – Compare models and local evidence
Assess GP–NN agreement and examine observed outputs around the strongest candidates. Consider whether each function shows continued improvement, local saturation/diminishing returns or evidence that a previously weak region/model has become more promising.

Step 7 – Function-specific candidate selection
Select one candidate per function using the combined evidence. Continue local refinement where improvements remain strong, use more cautious refinement where returns appear to be diminishing, and retain exploration where model reliability or regional evidence remains weak.

Step 8 – Final validation and submission
Check dimensionality, [0,1] bounds, duplicates and distance from existing observations before formatting the eight final queries to six decimal places.

**Step 1 – Add and assess the Week 8 results**

The eight returned outputs were first compared with the previous best observation for each function before being added to the datasets. The size and direction of each change were recorded to assess the success of the previous week's function-specific strategies. Particular attention was given to whether each function showed continued improvement, similar performance to the existing best, or evidence that improvements may be beginning to diminish.

Function 1:  
Previous best: 0.0000000000  
New output:    0.0000000000  
Difference:    -0.0000000000  
New best:      False  

Function 2:  
Previous best: 0.7469622512  
New output:    0.6571201028  
Difference:    -0.0898421484  
New best:      False  

Function 3:  
Previous best: -0.0101899399  
New output:    -0.0008627356  
Difference:    0.0093272043  
New best:      True  

Function 4:  
Previous best: 0.1869117254  
New output:    0.3872619363  
Difference:    0.2003502110  
New best:      True  

Function 5:  
Previous best: 8662.4825000000  
New output:    8115.4347231422  
Difference:    -547.0477768578  
New best:      False  

Function 6:  
Previous best: -0.3797395005  
New output:    -0.2743879717  
Difference:    0.1053515288  
New best:      True  

Function 7:  
Previous best: 2.9293698997  
New output:    2.9220592491  
Difference:    -0.0073106507  
New best:      False  

Function 8:  
Previous best: 9.9308110991  
New output:    9.9570315368  
Difference:    0.0262204376  
New best:      True  

The Week 8 strategy produced a strong overall result, with four of the eight queries improving the existing best. Function 4 produced the clearest improvement, increasing from 0.186912 to 0.387262 and providing further support for continued refinement of this region. Functions 3 and 6 also improved meaningfully, while Function 8 produced a smaller new best, potentially indicating that gains are becoming more incremental as optimisation progresses.  
The unsuccessful queries were also informative. Function 7 returned 2.922059 compared with the best of 2.929370, supporting the view that the current local region remains strong despite not producing an improvement. Function 5 remained high at 8115.434723 after deliberately moving away from the boundary optimum of 8662.482500, providing additional evidence about sensitivity around this region. Function 2 remained below its existing best, while Function 1's deliberate space-filling exploration again returned effectively zero.  
Overall, the results suggest that the functions are progressing differently as additional observations are accumulated. Some continue to produce substantial improvements, particularly Function 4, while others appear to be showing smaller gains or increasingly stable high-performing regions. These differences provide the basis for examining scaling, diminishing returns and possible changes in model behaviour in the next stage.  

Following assessment of the returned outputs, the eight new observations were added to their respective datasets. Each dataset increased by one observation, with the number of input dimensions remaining unchanged. The updated datasets therefore contain between 18 observations for the two-dimensional functions and 48 observations for the eight-dimensional Function 8. These expanded datasets will be used for the subsequent Week 9 analysis.

**Step 2. Assess performance as the datasets scale**

As the number of observations has increased, the optimisation history was examined to assess how additional data has affected performance across the eight functions. For each function, the running best output was calculated across the complete observation history. Recent changes in the best observed value were then compared with earlier improvements to identify whether performance is continuing to improve steadily, beginning to show diminishing returns, or changing more abruptly as additional observations are added.  
This analysis does not assume that increasing the dataset will produce the same behaviour for every function. Differences in dimensionality, function structure and the locations queried may cause additional observations to provide very different levels of information. Potential emergent behaviour will therefore be treated cautiously and investigated only where the optimisation or model behaviour changes noticeably rather than assuming that such behaviour must occur.  

Function 1: 18 observations | Current best = 0.000000 | Effectively flat, with no meaningful improvement despite additional observations.  
Function 2: 18 observations | Current best = 0.746962 | Only three new bests, indicating relatively intermittent improvement.  
Function 3: 23 observations | Current best = −0.000863 | Eight new bests, with particularly strong improvement across the most recent observations.  
Function 4: 38 observations | Current best = 0.387262 | Eleven new bests, including repeated substantial improvements in recent observations.  
Function 5: 28 observations | Current best = 8662.482500 | Eleven new bests, with strong recent improvement before the latest query failed to improve.  
Function 6: 28 observations | Current best = −0.274388 | Five new bests, with intermittent improvement but a meaningful new best from the latest query.  
Function 7: 38 observations | Current best = 2.929370 | Six new bests, with strong later-stage improvement followed by recent stability around the best region.  
Function 8: 48 observations | Current best = 9.957032 | Nine new bests, with continued but increasingly small recent improvements.  

The running-best trajectories show that increasing the number of observations has not produced a uniform improvement pattern across the eight functions. Functions 3 and 4 currently provide the clearest evidence of continued progress. Function 3 has produced new bests across each of its four most recent observations, while Function 4 has also improved repeatedly in the latest stages and has now moved from negative outputs to a best of 0.387262. There is therefore little evidence of diminishing returns in these functions at present.  
Function 5 also shows substantial improvement as observations have accumulated, particularly across observations 23–27. However, the latest query failed to improve on the best of 8662.482500, meaning it will be important to determine whether this represents the beginning of diminishing returns around the upper-boundary region or simply the result of moving away from the current best. Function 7 similarly improved strongly during the later stages before its two most recent observations failed to establish a new best, although the latest output remained very close to the optimum.  
Function 8 shows a different pattern. It continues to produce new bests, but recent improvements have been relatively small as the objective approaches 10. This is the clearest current indication of diminishing returns, although further observations are required before concluding that the function has reached a plateau. Function 6 has improved less consistently, but its latest observation produced a meaningful new best, while Function 2 has produced only three best improvements across 18 observations.  
Function 1 remains unusual. Although three observations are technically recorded as new bests, their outputs are effectively zero at the scale of the objective and do not represent meaningful optimisation progress.  
Overall, the results show that scaling the datasets is producing different returns across functions rather than a single scaling relationship. There is currently evidence of diminishing returns in some established high-performing regions, particularly Function 8, while Functions 3 and 4 continue to improve strongly. No clear emergent behaviour can yet be identified from optimisation performance alone; any such conclusion should also consider whether the GP or NN models show a noticeable change in reliability as the datasets grow.  

**Step 2b. Assess the size of recent improvements**

To examine whether optimisation gains are beginning to diminish as the datasets increase, the size of each improvement in the running best was calculated. Particular attention was given to the most recent improvements, allowing the magnitude of later gains to be compared with previous progress. Absolute rather than percentage changes were used because several functions contain negative values or cross zero, which would make percentage improvements difficult to interpret consistently.

Function 1: Recent recorded gains remain effectively zero, with no meaningful improvement in the objective.  
Function 2: Only two improvements have occurred after the initial best, with gains of 0.072209 and 0.135757. There is no clear pattern of diminishing improvement, but progress has been infrequent.  
Function 3: Recent gains have varied between 0.000953 and 0.014668. Importantly, the two latest improvements increased to 0.005309 and 0.009327, providing little evidence that progress is currently diminishing.  
Function 4: Recent gains remain substantial but variable. The latest two improvements were 0.387342 and 0.200350, showing continued strong progress despite being smaller than some earlier jumps.  
Function 5: Improvement sizes remain highly variable, ranging from 261.970 to 1834.482 across the five most recent new bests. The most recent improvement of 1541.331 was substantial.  
Function 6: Following a very small gain of 0.001637, the two subsequent improvements increased to 0.077134 and 0.105352, suggesting that progress has recently strengthened rather than diminished.  
Function 7: Later-stage gains of 0.188939, 0.397388 and 0.392431 remain meaningful, although the two observations since the current best have not produced further improvement.  
Function 8: The clearest reduction in improvement size is visible here. Gains decreased from 0.621928 to 0.223504, 0.089263 and 0.019562, before a small increase to 0.026220 in the latest observation.  

The improvement-size analysis strengthens the conclusion that diminishing returns are function-specific rather than a general consequence of increasing the dataset size. Functions 3, 4 and 6 continue to generate meaningful improvements, with recent gains in Functions 3 and 6 actually increasing. Function 5 has also produced several large improvements, although its latest query did not establish a new best. These functions therefore provide little evidence of a consistent decline in the value of additional observations at present.  
Function 8 provides the strongest evidence of diminishing returns. Although it continues to improve, the magnitude of recent gains has generally fallen as the best observed output has increased towards 10. The latest gain of 0.026220 was slightly larger than the preceding 0.019562, so the pattern is not perfectly monotonic, but the overall reduction from earlier gains remains clear.  
Function 7 may also be entering a more stable phase, as the current best has not been improved across the two subsequent observations. However, the latest output remained very close to the best, so there is not yet enough evidence to conclude that the region has fully plateaued. Function 2 shows infrequent rather than diminishing improvement, while Function 1 remains effectively flat.  
Overall, increasing the number of observations has produced different scaling behaviour across the functions. Some continue to benefit strongly from additional queries, while others show signs that further local refinement may provide progressively smaller gains. There is still no clear evidence of emergent behaviour from objective performance alone. This will be reconsidered after refitting the GP and NN models, where a noticeable change in model reliability as the datasets grow would provide stronger evidence of a change in behaviour.  

**Step 3. Refit and diagnose the models**

The GP and neural-network surrogate models were refitted using the expanded datasets. GP fitted length scales and predictive uncertainty were reviewed to identify whether the additional observations had changed model behaviour, particularly for functions that had previously shown very short length scales or weak generalisation.  
NN performance was reassessed using the same LOOCV procedure as the previous week and compared with the mean-prediction baseline. Maintaining the same validation approach allows changes in model reliability to be assessed consistently as the datasets increase. Particular attention was given to whether previously unreliable models improved with additional data, which may provide evidence of a change in model behaviour as the optimisation progresses.  

**Refit the GP models**

Function 1: 18 observations | RBF length scale = 1e-05 | Log marginal likelihood = −25.540894  
Function 2: 18 observations | RBF length scale = 1e-05 | Log marginal likelihood = −25.540894  
Function 3: 23 observations | RBF length scale = 0.201 | Log marginal likelihood = −25.852938  
Function 4: 38 observations | RBF length scale = 0.313 | Log marginal likelihood = −23.818726  
Function 5: 28 observations | RBF length scale = 0.303 | Log marginal likelihood = −23.970400  
Function 6: 28 observations | RBF length scale = 0.527 | Log marginal likelihood = −22.245983  
Function 7: 38 observations | RBF length scale = 0.323 | Log marginal likelihood = −34.151061  
Function 8: 48 observations | RBF length scale = 1.050 | Log marginal likelihood = −21.138922  

The updated GP diagnostics continue to show a clear difference between Functions 1–2 and the remaining functions. Despite receiving an additional observation, Functions 1 and 2 both retain length scales at the lower bound of 1e-05. This indicates that simply scaling these datasets from 17 to 18 observations has not resolved the GP's difficulty in identifying a smooth relationship between nearby observations. GP predictions for these functions should therefore continue to be treated cautiously.  
Functions 3–8 fitted substantially larger length scales, ranging from 0.201 for Function 3 to 1.050 for Function 8. Function 4 is particularly interesting following its recent optimisation performance: the GP retains a reasonable length scale while the objective has continued to improve substantially, supporting continued investigation of the newly identified high-performing region. Functions 5, 6 and 7 similarly retain plausible length scales for further GP-based candidate analysis.  
Function 8 now has the largest fitted length scale at 1.050, indicating substantially smoother modelled behaviour across its input space than Functions 1 and 2. This is consistent with the relatively stable high-performing region identified as recent improvements have become more incremental. 
Overall, the additional observations have not produced an obvious emergent change in GP behaviour. The previously problematic GPs for Functions 1 and 2 remain problematic, while the remaining models continue to provide usable surrogate representations. Any evidence that model reliability is changing as the datasets scale will therefore need to come from the subsequent validation and prediction analysis rather than the fitted GP length scales alone.  

**NN validation**

The neural-network surrogate models were retrained using the expanded datasets and evaluated using Leave-One-Out Cross-Validation (LOOCV). For each function, NN LOOCV MAE was compared with the same mean-prediction baseline used previously. The same model architecture and validation procedure were retained to allow changes in predictive reliability to be assessed consistently as additional observations were accumulated.

Function 1: NN LOOCV MAE = 0.001837 | Baseline MAE = 0.000995 | Beats baseline = False  
Function 2: NN LOOCV MAE = 0.236724 | Baseline MAE = 0.241635 | Beats baseline = True  
Function 3: NN LOOCV MAE = 0.056881 | Baseline MAE = 0.053409 | Beats baseline = False  
Function 4: NN LOOCV MAE = 1.385355 | Baseline MAE = 7.460801 | Beats baseline = True  
Function 5: NN LOOCV MAE = 255.307417 | Baseline MAE = 2324.264043 | Beats baseline = True  
Function 6: NN LOOCV MAE = 0.255144 | Baseline MAE = 0.530835 | Beats baseline = True  
Function 7: NN LOOCV MAE = 0.443789 | Baseline MAE = 0.712291 | Beats baseline = True  
Function 8: NN LOOCV MAE = 0.309128 | Baseline MAE = 0.981238 | Beats baseline = True  

The updated NN validation shows that the surrogate outperforms the mean baseline for six of the eight functions. Functions 4 and 5 show particularly strong validation performance relative to their baselines, while Functions 6, 7 and 8 also continue to provide evidence that the NN is capturing useful structure in the observed data.  
Function 2 continues to beat its baseline, although only by a relatively small margin. This is important because its GP remains unreliable with a length scale at the lower bound, meaning the validated NN may continue to provide the stronger model-based evidence for this function. However, its predictions should still be interpreted cautiously given the relatively small validation advantage.  
Functions 1 and 3 fail to outperform their baselines. Function 1 therefore remains problematic across both surrogate approaches, with neither the GP nor NN providing strong predictive evidence despite the additional observation. Function 3 is more interesting because optimisation performance has improved substantially even though its NN still fails validation. This suggests that recent optimisation success should not automatically be interpreted as improved NN reliability.  
Overall, the additional observations have not produced a sudden change in NN behaviour across the functions. Instead, the results reinforce the importance of assessing model reliability separately for each function as the datasets scale. There is therefore still no strong evidence of emergent model capability at this stage, although changes in validation performance will continue to be monitored as further observations are added.  

**Step 4. Generate and compare GP candidates**

The updated GP models were used to generate candidate points using the same acquisition-function comparison as the previous iteration. For each function, 100,000 candidate points were generated across the [0,1] search space. UCB was evaluated using κ values of 0.5, 1.0, 1.5 and 2.5, representing progressively greater emphasis on uncertainty and exploration.  
Expected Improvement was also evaluated using ξ values equivalent to 0%, 1%, 5% and 10% of the observed output standard deviation for each function. Predicted output, GP uncertainty, acquisition score and distance from the nearest existing observation were recorded for each selected candidate. Using the same settings as the previous iteration allows changes in candidate behaviour to be attributed to the expanded datasets rather than changes in the acquisition methodology.  
Because Functions 1 and 2 continue to have GP length scales at the lower bound, their recommendations will be treated cautiously regardless of acquisition score.  

Function 1: All UCB κ values and all EI ξ settings selected the same candidate [0.094177, 0.975622], with predicted output −0.000529, uncertainty 0.001544 and nearest-observation distance 0.095343. This apparent agreement should be treated cautiously because the GP length scale remains at its lower bound.  
Function 2: Low-κ UCB (0.5 and 1.0) returned the existing Week 8 point [0.697103, 0.994806], while κ = 1.5 and 2.5 selected the more exploratory [0.403694, 0.391789]. All EI settings also selected the exploratory candidate, which has substantially greater uncertainty (0.259715) and a nearest-observation distance of 0.113423.  
Function 3: Increasing κ produced progressively more exploratory candidates. κ = 0.5 remained relatively close to the observed data with a predicted output of −0.000991, while κ = 2.5 moved to a candidate 0.369217 from its nearest observation. All EI settings converged on the κ = 1.5 candidate [0.779394, 0.644459, 0.001936].  
Function 4: κ = 0.5 produced the strongest predicted output of 0.736288 and remained relatively close to existing observations. Increasing κ progressively favoured uncertainty, with κ = 2.5 selecting a much more distant candidate with a predicted output of −9.678100. EI at 0–5% selected the κ = 1.0 candidate, while EI at 10% shifted to the more exploratory κ = 1.5 candidate.  
Function 5: Low-κ UCB remained around the established high-performing upper-boundary region. κ = 0.5 selected a point effectively coinciding with an existing observation, while κ = 1.0 and 1.5 remained nearby. In contrast, κ = 2.5 and every EI setting selected the highly exploratory [0.019672, 0.003040, 0.043324, 0.976161], 0.686332 from the nearest observation, despite its much lower predicted output of 1607.436.  
Function 6: κ = 0.5 and 1.0 selected the same candidate with predicted output −0.248253, which is above the current observed best of −0.274388. κ = 1.5 and 2.5 progressively moved towards more uncertain regions. EI at 0% selected a slightly different candidate with the highest predicted output of −0.220587, while EI at 1–10% converged on the low-κ UCB candidate.  
Function 7: κ = 0.5 and 1.0 effectively returned the existing best observation. κ = 1.5 and 2.5 moved into increasingly novel regions. All EI settings selected the κ = 1.5 candidate [0.172191, 0.217932, 0.444885, 0.348858, 0.257791, 0.772985], with predicted output 2.418676 and nearest-observation distance 0.222032.  
Function 8: Candidate novelty increased substantially with κ. The κ = 0.5 candidate remained relatively local and was predicted at 9.967635, slightly above the current best of 9.957032. κ = 1.0–2.5 increasingly prioritised uncertainty. EI at 0–1% selected the κ = 1.0 candidate, while EI at 5–10% selected the highly exploratory κ = 2.5 candidate.  

The acquisition comparison shows a clearer exploration–exploitation structure than simply choosing a single GP recommendation. Functions 3, 4, 6 and 8 show a progressive movement away from observed regions as κ increases, demonstrating the expected effect of placing greater weight on uncertainty.  
Functions 4 and 6 currently provide particularly interesting exploitation opportunities. For Function 4, the low-κ candidate has a predicted output of 0.736288 compared with the current best of 0.387262, while Function 6 has multiple relatively local candidates predicted above its current best. Given that both functions also established new bests in Week 8, these results provide further support for investigating continued refinement around their recently successful regions.  
Function 8 presents a particularly useful example of the exploration–exploitation trade-off. The low-κ candidate predicts a small improvement over the existing optimum, while progressively higher κ values move substantially further from the observed data. Combined with the diminishing improvement sizes identified earlier, this suggests that the later candidate-selection stage should consider whether continued local refinement remains worthwhile or whether some additional exploration is justified. We should not decide that yet.  
Function 5 again shows a sharp divide between local exploitation and model-driven exploration. The exploratory candidate selected by κ = 2.5 and EI has very high uncertainty but a predicted output far below the current best. Given the strong historical performance of the upper-boundary region, this candidate will require particularly strong supporting evidence from the NN and neighbourhood analysis before it would justify moving away from that region.  
Functions 1 and 2 remain difficult to interpret because their GP length scales are still at the lower bound. Agreement between acquisition settings for Function 1 therefore should not be interpreted as genuine model confidence. Function 2 similarly shows acquisition behaviour dominated by the contrast between an existing observation and a high-uncertainty exploratory region, reinforcing the need to use its validated NN as an independent comparison.  
Overall, there is no need to select candidates yet. The acquisition analysis has identified where the GP favours exploitation, exploration or a compromise between the two. The next step should be to generate the updated NN candidates, after which we can examine cross-model agreement and determine whether the additional data has strengthened or weakened the evidence for particular regions.  

**Step 5. Generate and compare NN candidates**

The neural-network surrogate models were retrained on the complete updated datasets and used to generate candidate recommendations from 100,000 points within the [0,1] search space. Input and output scaling were applied using the complete dataset for each function before fitting the final surrogate.  
To account for the stochastic nature of neural-network training, multiple models were fitted using different random seeds. The mean prediction across the ensemble was used as the predicted output, while disagreement between the individual models was recorded as an additional indication of prediction stability. Candidate novelty was measured using distance from the nearest existing observation.  
NN recommendations were interpreted according to the LOOCV validation results. Greater weight will therefore be given to NN evidence for Functions 2 and 4–8, where the NN outperformed the baseline, while recommendations for Functions 1 and 3 will be treated cautiously.  

Function 1: Candidate = [0.517917, 0.000290] | Predicted output = 0.001946 | NN disagreement = 0.001331 | Nearest distance = 0.182315 | LOOCV validated = False  
Function 2: Candidate = [0.702517, 0.009487] | Predicted output = 0.831771 | NN disagreement = 0.431009 | Nearest distance = 0.106007 | LOOCV validated = True  
Function 3: Candidate = [0.465542, 0.513952, 0.462107] | Predicted output = −0.001013 | NN disagreement = 0.000717 | Nearest distance = 0.018674 | LOOCV validated = False  
Function 4: Candidate = [0.386457, 0.364301, 0.315578, 0.423391] | Predicted output = 0.173431 | NN disagreement = 0.101814 | Nearest distance = 0.032892 | LOOCV validated = True  
Function 5: Candidate = [0.996673, 0.749297, 0.983441, 0.989318] | Predicted output = 8132.577652 | NN disagreement = 837.663492 | Nearest distance = 0.247618 | LOOCV validated = True  
Function 6: Candidate = [0.476200, 0.401879, 0.450627, 0.837966, 0.006748] | Predicted output = −0.248980 | NN disagreement = 0.042060 | Nearest distance = 0.127622 | LOOCV validated = True  
Function 7: Candidate = [0.150161, 0.338209, 0.511025, 0.190914, 0.382394, 0.542431] | Predicted output = 2.893229 | NN disagreement = 0.055542 | Nearest distance = 0.110492 | LOOCV validated = True  
Function 8: Candidate = [0.012982, 0.180599, 0.189901, 0.256228, 0.760257, 0.471612, 0.155487, 0.557332] | Predicted output = 10.056825 | NN disagreement = 0.027228 | Nearest distance = 0.234486 | LOOCV validated = True  

The updated NN recommendations provide useful additional evidence for six functions where the model continues to outperform its LOOCV baseline. Functions 1 and 3 remain exceptions, so their apparently strong NN predictions should not be used independently for candidate selection.  
Function 2 is interesting because the validated NN predicts 0.831771, above the current best of 0.746962. However, disagreement between the ensemble models is relatively high at 0.431009. Combined with the unreliable GP, this means the NN recommendation is worth investigating but should not yet be treated as strong evidence of an improvement.  
Function 4 recommends a point very close to the existing high-performing region. Although its NN prediction of 0.173431 is below the current best of 0.387262, the location provides additional support for continued local refinement. This is particularly relevant because the GP's low-κ recommendations also favour the same general area.  
Function 5 remains focused towards the upper end of the search space, broadly consistent with the strong region identified previously. However, NN disagreement is high and the predicted output remains below the current best of 8662.482500. The recommendation therefore provides regional evidence rather than a clear reason to move away from the existing boundary optimum.  
Function 6 provides stronger evidence. The validated NN predicts −0.248980, above the current best of −0.274388, with relatively low disagreement. The GP also identified local candidates with similar predicted performance, suggesting that this may again be a useful region for refinement.  
Function 7 remains concentrated around the established high-performing region. The prediction of 2.893229 is slightly below the current best of 2.929370, but disagreement is low and the candidate remains relatively close to existing observations. This supports the previous conclusion that the region remains promising even though recent queries have not improved the optimum.  
Function 8 is particularly notable. The validated NN predicts 10.056825, above the current best of 9.957032, with low disagreement of 0.027228. This suggests that despite the diminishing improvement sizes identified in Step 2, the NN still identifies potential for further improvement rather than indicating a complete plateau. However, its candidate is reasonably novel, so agreement with the GP and neighbourhood evidence should be checked before selecting it.  
Overall, the NN results reinforce the need for function-specific rather than model-specific selection. There are particularly useful signals for Functions 4, 6, 7 and 8, while Functions 1 and 3 should continue to rely more heavily on other evidence. Function 2 remains unusual because the NN is validated but has substantial ensemble disagreement, while Function 5 retains a strong regional signal but comparatively high uncertainty.  

**Step 6. GP–NN agreement**

**Compare GP and NN candidates**

The distance between each NN recommendation and the candidates generated by the different GP acquisition settings was calculated to assess cross-model agreement. Rather than comparing the NN with a single GP recommendation, all UCB and EI candidates were considered because the acquisition settings represent different balances between exploration and exploitation.  
The closest GP candidate to the NN recommendation was identified for each function, together with the corresponding acquisition method and distance. Small distances indicate that the two surrogate approaches are independently identifying similar regions, while larger distances indicate greater disagreement. These results will be interpreted alongside model reliability, since agreement between unreliable models does not necessarily provide strong evidence for a candidate.  

Function 1: Closest GP method = UCB κ=0.5 | GP–NN distance = 1.063404 | Very strong disagreement between the GP and NN regions.  
Function 2: Closest GP method = UCB κ=1.5 | GP–NN distance = 0.485232 | Considerable disagreement remains, despite the NN passing LOOCV.  
Function 3: Closest GP method = UCB κ=0.5 | GP–NN distance = 0.058180 | Strongest GP–NN agreement across the functions, although the NN fails LOOCV.  
Function 4: Closest GP method = UCB κ=0.5 | GP–NN distance = 0.129899 | Relatively strong regional agreement around the more exploitative GP candidate.  
Function 5: Closest GP method = UCB κ=1.5 | GP–NN distance = 0.221396 | Moderate agreement between the NN and locally focused UCB candidates, but very strong disagreement with the exploratory EI/high-κ region.  
Function 6: Closest GP method = EI ξ=0% | GP–NN distance = 0.193602 | Moderate agreement, with several lower-exploration GP candidates also relatively close to the NN region.  
Function 7: Closest GP method = UCB κ=0.5 | GP–NN distance = 0.125569 | Relatively strong agreement around the established high-performing region.  
Function 8: Closest GP method = UCB κ=0.5 | GP–NN distance = 0.243999 | Moderate regional agreement, with substantially greater disagreement as the GP acquisition becomes more exploratory.  

The GP–NN comparison provides stronger support for local refinement in several functions. Functions 4 and 7 show relatively close agreement between the validated NN and low-κ GP recommendations, indicating that the two surrogate approaches independently favour broadly similar local regions. Function 6 also shows moderate agreement, consistent with both models predicting potential for continued improvement following its latest new best.  
Function 5 shows an important split. The NN is reasonably close to the lower-κ UCB candidates but very distant from the exploratory candidate selected by high-κ UCB and EI. Combined with the strong recent performance around the upper boundary, this currently gives greater cross-model support to the established region than to the highly exploratory alternative.  
Function 8 also shows a useful pattern. The validated NN is closest to the κ=0.5 GP recommendation, while disagreement increases substantially as κ increases. This suggests that both models retain some support for further refinement despite the diminishing gains identified earlier. However, a distance of 0.243999 in eight dimensions should not be interpreted as particularly close point-level agreement; the neighbourhood analysis will be more informative about whether they occupy similarly strong regions.  
Function 3 produces the smallest numerical GP–NN distance at 0.058180. However, because the NN still fails LOOCV, this agreement should not be given the same weight as agreement between validated models. Conversely, Functions 1 and 2 show substantial disagreement. Function 1 remains particularly unresolved because neither surrogate provides sufficiently reliable evidence. For Function 2, the NN passes validation while the GP remains at its lower-bound length scale, so the disagreement currently gives more reason to investigate the NN region than to treat both models equally.  
Overall, the results reinforce the function-specific strategy established in previous weeks. Cross-model agreement is most useful where it is supported by validation and recent observed performance rather than being treated as evidence on its own.  

**Neighbourhood analysis**

The observed neighbourhoods surrounding the main GP and NN recommendations were examined to provide an additional model-independent comparison of candidate quality. For each function, the nearest observed outputs to the NN candidate, low-κ UCB candidate and high-κ UCB candidate were identified. This allows local historical performance to be considered alongside predicted output, uncertainty and cross-model agreement.  
Particular attention was given to whether locally focused candidates were surrounded by consistently strong observations and whether more exploratory recommendations had any supporting evidence from the existing dataset.  
Function 1: Neither candidate region has meaningful neighbourhood evidence. Outputs surrounding both the NN and GP candidates remain effectively zero, so the neighbourhood analysis does not distinguish between the competing regions.  
Function 2: The NN candidate has considerably stronger local support, with its three nearest observations producing 0.471844, 0.538996 and the current best of 0.746962. The exploratory κ=2.5 GP candidate is surrounded by much weaker outputs of 0.214965, −0.013858 and −0.065624. The κ=0.5 candidate corresponds to the previously submitted observation of 0.657120.  
Function 3: Both the NN and κ=0.5 GP candidates are located near the strongest observed region, with neighbouring outputs including −0.000863, −0.010190 and −0.016451. The exploratory κ=2.5 region has substantially weaker neighbouring observations.  
Function 4: The NN and low-κ GP candidates are both supported by the recently successful region containing outputs of 0.387262 and 0.186912. In contrast, the κ=2.5 candidate is distant from the existing data and its nearest observed outputs are substantially poorer, including −13.527649 and −10.069633.  
Function 5: The NN and low-κ GP candidates remain associated with the established high-performing region, with neighbouring outputs reaching 8115.434723. The exploratory κ=2.5 candidate is considerably further from the observed data and its nearest outputs are only 109.571876, 233.223610 and 64.443440.  
Function 6: The NN candidate is locally supported by the current best of −0.274388 and other relatively strong observations of −0.388687 and −0.379740. The low-κ GP candidate also remains associated with the stronger observed region, while neighbourhood support weakens for the more exploratory κ=2.5 candidate.  
Function 7: The NN candidate is surrounded by outputs of 2.922059, the current best of 2.929370 and 2.536939. The κ=0.5 GP recommendation effectively returns the existing best itself, while the exploratory candidate is considerably further from these observations.  
Function 8: Both the NN and κ=0.5 GP candidates have strong neighbourhood support. The NN candidate is surrounded by outputs of 9.930811, 9.911249 and the current best of 9.957032, while the low-κ GP candidate is similarly close to the strongest observations. The κ=2.5 candidate is much more distant and has weaker local support.  

The neighbourhood analysis provides strong support for a locally focused strategy across most functions. This is particularly clear for Functions 3–8, where the NN and/or low-κ GP candidates remain associated with the strongest observed regions, while the more exploratory κ=2.5 candidates generally have substantially weaker historical support.  
Function 2 again favours the NN region. Its neighbouring observations include the current best of 0.746962, whereas the exploratory GP region is associated with substantially weaker outputs. Given that the NN continues to outperform its LOOCV baseline while the GP length scale remains at its lower bound, the NN region currently has stronger supporting evidence. However, the relatively high NN disagreement means that this evidence should still be treated cautiously.  
Functions 4 and 6 provide particularly strong evidence for continued local refinement. Both established new bests in the latest round, their surrogate models remain usable, and their locally focused candidates continue to be supported by strong observed neighbourhoods. The poor historical performance around the more exploratory GP regions gives little current reason to move substantially away from these areas.  
Function 5 also strongly favours its established high-performing region. The difference is especially pronounced: the local region contains outputs above 8000, while the nearest observations around the exploratory GP candidate are only around 64–233. Combined with the strong recent scaling performance of Function 5, there is currently little empirical support for abandoning the upper-boundary region.  
Function 7 remains concentrated around a clearly strong region, but the low-κ GP simply returns the existing best. A new candidate will therefore need to refine this neighbourhood while remaining sufficiently distinct from previous observations, similar to the novelty constraint used previously.  
Function 8 is particularly informative in the context of diminishing returns. Although recent improvements have become smaller, both surrogate approaches continue to identify candidates associated with the strongest observed neighbourhood. The NN additionally predicts a value above the current best with low ensemble disagreement. This suggests that diminishing returns do not necessarily imply that the region has been exhausted; further local refinement remains justified, although expected gains may be relatively small.  
Function 3 also strongly favours local refinement based on neighbourhood evidence, despite the NN failing LOOCV. Its latest observation produced a substantial new best and the low-κ GP independently remains near the same high-performing region, so the GP and observed data provide sufficient evidence without relying heavily on the NN.  
Function 1 remains unresolved. Both candidate regions are surrounded by values effectively equal to zero, while neither surrogate is reliable enough to distinguish between them. It therefore remains the strongest candidate for a model-independent exploration strategy, rather than repeatedly following unreliable surrogate predictions.  
Overall, Step 6 provides a fairly clear direction: Functions 2–8 mostly favour exploitation or controlled local refinement, while Function 1 remains the main case for broader exploration. Importantly, that's based on the combination of recent optimisation performance, validation, cross-model agreement and actual observed neighbourhoods rather than simply selecting whichever model predicts the highest value.  

**Step 7. Function-specific candidate selection**

Final candidates will be selected on a function-by-function basis using the combined evidence from the Week 9 analysis. Decisions will consider recent optimisation performance, evidence of diminishing returns, GP reliability, UCB and EI behaviour, NN LOOCV validation, GP–NN agreement, candidate novelty and observed neighbourhood performance.  
Greater emphasis will be placed on local refinement where recent observations have continued to improve the objective and where the surrounding observations and surrogate models support an established high-performing region. Exploration will receive greater consideration where model reliability remains weak or where there is insufficient evidence to distinguish between candidate regions.  
For functions where the strongest recommendations correspond to existing observations or where additional local refinement is justified, new candidates will be generated within a constrained neighbourhood of the current best. Existing observations will be excluded to ensure that all final submissions remain novel.  

Current direction:

Function 1: Space-filling exploration — GP remains at its lower-bound length scale, NN fails LOOCV, GP and NN strongly disagree, and neighbourhood evidence cannot distinguish between their recommendations.  

Function 2: NN-supported refinement — GP remains unreliable, while the validated NN candidate lies in a substantially stronger observed neighbourhood. NN disagreement remains relatively high, so the region should be refined cautiously rather than simply accepting the highest NN prediction.  

Function 3: Local GP refinement — the latest query produced another new best, the low-κ GP remains close to the strongest observed neighbourhood, and the NN fails LOOCV despite being regionally similar.  

Function 4: Local refinement — another substantial new best was achieved, and both the validated NN and low-κ GP continue to support the same high-performing region.  

Function 5: Local boundary refinement — the latest query did not beat the boundary optimum, but the established upper region remains overwhelmingly stronger than the exploratory alternative. This is also where we should be conscious of possible diminishing returns rather than moving arbitrarily far away.  

Function 6: Local GP/NN refinement — the latest query established a new best and both validated NN and lower-exploration GP recommendations remain associated with the strongest neighbourhood.  

Function 7: Local novel refinement — the established region remains strongly supported, but the low-κ GP recommendation effectively duplicates the current best, requiring a novel point nearby.  

Function 8: Controlled local refinement — recent gains show diminishing returns, but the latest query still established a new best and both the validated NN and low-κ GP remain supported by the strongest observed neighbourhood. The evidence therefore favours another careful local query rather than the highly exploratory GP alternatives.  

**Generate local candidates**

Functions 3–8 were taken forward for local refinement because the preceding analysis provided evidence that their strongest observed regions remained promising. For each function, 50,000 candidates were generated around the current best at radii of 0.03, 0.05 and 0.10. The candidates were evaluated using both GP and NN predictions, with the predictions standardised before being combined so that differences in output scale did not affect the comparison. Existing observations were excluded and candidate novelty, GP uncertainty and NN disagreement were also recorded. Comparing multiple radii allowed the degree of local refinement to be assessed rather than assuming the same search radius was appropriate for every function.  
Functions 1 and 2 were excluded from this combined GP–NN local search because their GP models remain unreliable, with length scales at the lower bound. Function 1 also fails NN validation and shows substantial disagreement between the GP and NN, so model-independent exploration remains more appropriate. For Function 2, the NN passes LOOCV validation and has stronger neighbourhood support, so it will instead be investigated using NN-supported refinement without giving equal weight to the unreliable GP.  

Function 3: All three radii converge very close to the current best. The selected candidates are only 0.0086–0.0109 from it, with GP predictions around −0.00017 to −0.00024 and NN predictions around −0.00068 to −0.00073. The results are highly stable across radius settings.  
Function 4: All three searches also converge on essentially the same local region. GP predictions range from 0.531330 to 0.563957, all above the current best of 0.387262, while the NN predicts 0.311005–0.326520. Candidate distances from the current best remain small at approximately 0.016–0.019.  
Function 5: All three searches collapse extremely close to [1,1,1,1], with nearest-observation distances of only 0.000103–0.000125. Both models continue to predict outputs close to the current best, but these candidates provide almost no meaningful novelty.  
Function 6: Increasing the radius produces progressively more ambitious candidates. GP predictions improve from −0.235263 at radius 0.03 to −0.195871 at radius 0.10, while NN predictions improve from −0.253795 to −0.229762. Candidate distance from the current best increases from 0.048149 to 0.110780.  
Function 7: All three radii remain within the established high-performing region. The GP remains close to the current best, while the NN predicts values above the current best at every radius. The radius 0.10 candidate produces the highest NN prediction of 3.028290 but also greater GP uncertainty and NN disagreement.  
Function 8: Both models predict continued potential improvement at all three radii. GP predictions range from 9.961961 to 9.976275, while NN predictions range from 9.993530 to 10.015223. The radius 0.10 candidate is substantially more novel at 0.190714 from the nearest observation while remaining supported by both models.  


Review:  
The multi-radius analysis generally reinforces the local-refinement strategy, but the appropriate degree of refinement differs substantially between functions.  
For Function 3, changing the search radius has almost no effect on the region ultimately selected. This convergence strengthens the evidence that the useful search area is immediately around the latest best. Because the NN fails LOOCV, the GP and observed improvement history should remain the primary evidence rather than the NN prediction.  
Function 4 shows similarly strong local stability. All three searches converge close to the current best and the GP predicts further improvement throughout this region. Given the consecutive recent improvements and validated NN support for the broader region, continued fine local refinement remains well supported.  
Function 6 differs because increasing the radius produces progressively stronger predictions from both models. The 0.10 search therefore provides evidence that useful refinement may extend somewhat beyond the immediate neighbourhood of the current best rather than requiring an extremely small step.  
For Function 7, all three searches remain promising. The increasingly optimistic NN prediction at the larger radius needs to be balanced against increasing disagreement and a slightly weaker GP prediction. A more conservative local candidate may therefore provide a better balance between novelty and cross-model support.  
Function 8 provides particularly interesting evidence. Despite the diminishing returns identified earlier, both models continue to predict improvement across the local search. The 0.10 candidate is predicted above the current best by both models and provides considerably greater novelty than the smaller-radius alternatives. This supports broadening the local refinement slightly rather than repeatedly making extremely small adjustments around the current best.  

**Function 5 novelty-constrained refinement**

The initial local-refinement search for Function 5 repeatedly converged extremely close to the existing optimum at [1,1,1,1], producing candidates with nearest-observation distances of only approximately 0.0001. Although technically novel, these points represent negligible movement within the search space and are unlikely to provide meaningful additional information.  
Function 5 was therefore examined separately using minimum novelty distances of 0.03, 0.05 and 0.10 from all existing observations. This allows the strength of the established upper-boundary region to be retained while testing whether meaningful movement away from the current optimum remains supported by the GP and validated NN.  

Minimum novelty 0.03: Candidate = [1.000000, 0.969998, 1.000000, 1.000000] | GP predicted = 8287.623017 | NN predicted = 8688.604318 | GP uncertainty = 247.689521 | NN disagreement = 118.482518 | Nearest distance = 0.030002.  

Minimum novelty 0.05: Candidate = [1.000000, 0.949994, 1.000000, 1.000000] | GP predicted = 8001.755491 | NN predicted = 8710.650195 | GP uncertainty = 410.333828 | NN disagreement = 197.196894 | Nearest distance = 0.050006.  

Minimum novelty 0.10: Candidate = [1.000000, 0.899981, 1.000000, 1.000000] | GP predicted = 7187.506528 | NN predicted = 8743.779604 | GP uncertainty = 802.299569 | NN disagreement = 369.571101 | Nearest distance = 0.100019.  
 
The novelty-constrained analysis confirms that Function 5 remains strongly concentrated around the [1,1,1,1] boundary optimum, but also reveals increasing disagreement between the surrogate models as candidates are forced further away. At a minimum novelty of 0.03, the GP still predicts a relatively strong output of 8287.62, while the NN predicts a small improvement over the current best of 8662.48. Both GP uncertainty and NN disagreement remain substantially lower than at the larger novelty thresholds.  
As the minimum distance increases to 0.05 and 0.10, the NN becomes progressively more optimistic, but the GP prediction deteriorates and uncertainty increases sharply. NN disagreement also rises from 118.48 at 0.03 to 369.57 at 0.10. The increasing NN prediction should therefore not be interpreted as stronger evidence for improvement, because it occurs alongside increasing extrapolation and model disagreement.  
Overall, the 0.03 region provides the strongest balance between meaningful novelty and model support. This is also consistent with the previous iteration, where a movement of approximately 0.03 from the optimum remained high-performing but did not improve on [1,1,1,1]. The candidate [1, 0.969998, 1, 1] therefore provides the strongest Function 5 option to carry forward into the final comparison.  

**Function 1 model-independent exploration**

Function 1 was treated separately because neither surrogate currently provides reliable predictive guidance. The GP length scale remains at its lower bound, the NN fails LOOCV validation, the models strongly disagree on their preferred regions and neighbourhood analysis does not distinguish between the alternatives. A model-independent space-filling strategy was therefore used to prioritise information gain rather than unreliable predicted performance. A large candidate set was generated across the two-dimensional search space and the candidate with the greatest distance from its nearest existing observation was selected.

Function 1: The maximum-distance search selected [0.682178, 0.001903]. Its nearest existing observation is 0.306754 away, with the next two nearest observations at distances of 0.308314 and 0.316118.  

The selected candidate occupies a substantially underexplored region of the Function 1 search space. Unlike the GP and NN recommendations, its selection does not depend on either surrogate producing reliable predictions and instead maximises distance from the existing observations.  
A nearest-observation distance of 0.306754 provides meaningful novelty compared with the model-generated alternatives examined earlier. Given the GP's persistent lower-bound length scale, failed NN validation, strong GP–NN disagreement and lack of useful neighbourhood differentiation, this provides a more defensible strategy for Function 1 at this stage. The purpose of the query is therefore primarily exploration and information gain, rather than an expectation of immediate improvement.  

**Function 2 NN-supported refinement**

Function 2 was also treated separately because its GP length scale remains at the lower bound, making GP-based acquisition recommendations unreliable. Unlike Function 1, however, the NN continues to outperform its LOOCV baseline and its preferred region has strong neighbourhood support, including the current best observation. NN-only local searches were therefore performed around the current best at radii of 0.03, 0.05 and 0.10. Predicted output, ensemble disagreement and candidate novelty were compared to determine whether controlled refinement provided a better-supported alternative to the unconstrained NN recommendation.  

Radius 0.03: Candidate = [0.711375, 0.108192] | NN predicted = 0.718309 | NN disagreement = 0.149730 | Distance from best = 0.030012 | Nearest existing observation = 0.018533.  

Radius 0.05: Candidate = [0.710141, 0.088175] | NN predicted = 0.743700 | NN disagreement = 0.211086 | Distance from best = 0.050067 | Nearest existing observation = 0.030706.  

Radius 0.10: Candidate = [0.704305, 0.038527] | NN predicted = 0.810157 | NN disagreement = 0.363055 | Distance from best = 0.100015 | Nearest existing observation = 0.077676.  

The Function 2 refinement shows a clear relationship between increasing distance from the current best and increasing predicted performance, but this is accompanied by increasing NN disagreement. The 0.03 candidate has the lowest disagreement but is predicted below the current best of 0.746962 and remains very close to existing observations.  
At radius 0.05, the prediction rises to 0.743700, approximately matching the current best, while disagreement increases moderately. At radius 0.10, the NN predicts 0.810157, exceeding the current best, and the candidate provides substantially greater novelty. However, disagreement also increases considerably to 0.363055.  
The larger-radius candidate therefore offers greater potential improvement and information gain, but with less reliable model agreement. This is consistent with the earlier unconstrained NN recommendation, which moved further in the same general direction towards lower values of the second input and predicted an even higher output, but with still greater disagreement. Overall, the analysis suggests that the promising region may extend away from the current best towards lower values of the second dimension, but confidence decreases as the search moves further into this less-observed area.  

**Final candidate selection**

One final candidate was selected for each function using the combined evidence from the preceding analyses. Selection prioritised observed optimisation performance and model reliability, followed by neighbourhood support, cross-model agreement, predicted performance and candidate novelty. Where model reliability differed between functions, the weighting of this evidence was adjusted accordingly rather than applying a single selection rule across all eight functions.  
For functions showing continued improvement and consistent local support, candidates were selected through controlled refinement around established high-performing regions. Where model reliability was poor, alternative strategies were used: model-independent exploration for Function 1 and NN-supported refinement for Function 2. The selections also considered evidence of diminishing returns, avoiding unnecessary exploration where an established region continued to produce improvements while allowing greater novelty where broader local searches remained well supported.  

Provisional selections

Function 1 – Model-independent exploration  
Candidate: [0.682178, 0.001903]  
- Neither surrogate provides reliable guidance, so the maximum-distance candidate is preferred. Its nearest-observation distance of 0.306754 gives substantial information gain without relying on unreliable model predictions.  

Function 2 – NN refinement, radius 0.10  
Candidate: [0.704305, 0.038527]  
- The 0.10 candidate is preferred over 0.05. It follows the same direction as the unconstrained NN optimum, predicts 0.810157 compared with the current best of 0.746962, and provides considerably greater novelty. NN disagreement increases to 0.363055, so the prediction itself should be treated cautiously, but the directional evidence from the NN and neighbourhood analysis supports testing further towards lower values of the second dimension.

Function 3 – radius 0.03  
Candidate: [0.473523, 0.500295, 0.456235]  
- The 0.03 candidate is preferred because it has the highest GP prediction (−0.000166), while the NN is not sufficiently validated to justify selecting between candidates based on its small prediction differences.

Function 4 – radius 0.05  
Candidate: [0.388825, 0.366595, 0.350533, 0.420635]  
- This produces the highest GP prediction of 0.563957, substantially above the current best of 0.387262. All three radii converge tightly on the same region, so this remains a controlled local refinement rather than a materially more exploratory choice.  

Function 5 – novelty 0.03  
Candidate: [1.000000, 0.969998, 1.000000, 1.000000]  
- This provides meaningful novelty while remaining closest to the established boundary optimum. Moving further away causes GP predictions to deteriorate sharply and uncertainty/disagreement to increase. The 0.03 candidate therefore provides the strongest balance between novelty and model support.  

Function 6 – radius 0.10  
Candidate: [0.440497, 0.372759, 0.525613, 0.791828, 0.000000]  
- Both models become more optimistic as the search radius expands. At 0.10, the GP predicts −0.195871 and the NN −0.229762, both above the current best of −0.274388. The candidate also provides meaningful novelty at 0.110780, making this a justified broader local move.  

Function 7 – radius 0.03  
Candidate: [0.199451, 0.330731, 0.460263, 0.255214, 0.346569, 0.603374]  
- A more conservative selection is appropriate for Function 7. The 0.10 candidate's higher NN prediction comes with increasing disagreement and a weaker GP prediction. The 0.03 candidate has the strongest GP support, the lowest NN disagreement and still has an NN prediction of 2.985400, above the current best of 2.929370.

Function 8 – radius 0.10  
Candidate: [0.031565, 0.175185, 0.160043, 0.130670, 0.742356, 0.487463, 0.194164, 0.622246]  
- Despite recent diminishing gains, this function established another new best last week. The broader local candidate is particularly attractive because both models predict improvement: GP = 9.976275 and NN = 10.009696 versus the current best of 9.957032. It also increases novelty to 0.190714 without moving into the poorly supported exploratory GP region.  


The final selections reflect increasingly function-specific optimisation strategies rather than applying a common acquisition rule across all eight functions. Function 1 remains focused on exploration because the available models provide little reliable predictive guidance, while Function 2 relies more heavily on the validated NN and observed neighbourhood structure.  
Functions 3, 4 and 7 favour relatively conservative refinement because strong regions have already been identified and the evidence does not justify moving substantially away from them. Function 5 similarly remains close to its boundary optimum, but with a deliberate novelty constraint to avoid submitting an effectively duplicate point.  
Functions 6 and 8 justify somewhat broader local refinement. Both recently established new bests and both GP and NN predictions continue to support improvement as the search moves further from the existing optimum. This is particularly relevant to the diminishing-returns analysis: smaller recent gains do not by themselves justify abandoning a region while new bests and model support continue to emerge.  

**Step 8. Final validation and submission**

The eight selected candidates were rounded to six decimal places and checked against the complete updated datasets before submission. Each candidate was tested for the correct dimensionality, compliance with the permitted [0,1] search space and duplication of an existing observation. The distance to the nearest existing observation was also calculated after rounding to confirm that each submitted candidate retained the intended level of novelty.  
This final check is particularly important for Function 5, where a deliberate minimum novelty constraint was introduced, and for Functions 1, 2, 6 and 8, where candidate novelty formed part of the selection rationale.  

All eight final candidates passed the validation checks. Each candidate had the correct dimensionality, remained within the permitted [0,1] search space and was distinct from all existing observations.

Nearest-observation distances ranged from 0.010539 for Function 3 to 0.306755 for Function 1, reflecting the different function-specific strategies used. Functions 3 and 4 remain particularly close to their latest best observations as part of the fine local-refinement strategy, while Function 5 retains the deliberately imposed novelty distance of approximately 0.03 from its boundary optimum. Function 7 similarly remains close to its established high-performing region.

Functions 6 and 8 make broader local moves, with nearest-observation distances of 0.110780 and 0.190714 respectively, reflecting the evidence that broader local refinement remained promising. Function 2 provides an intermediate case, moving 0.077676 from the nearest observation in the NN-supported direction without relying on its unreliable GP. Function 1 is the most exploratory selection, with a distance of 0.306755, consistent with the model-independent space-filling strategy.

Overall, the final validation confirms that the candidates preserve the intended balance between exploration and exploitation after formatting to six decimal places, while all eight satisfy the submission requirements.

**Module 21 (Week 10)**

**Methodology**

This round will build directly on the previous analysis by using the newly received outputs to assess how well the different modelling approaches performed in practice. Last round, GP and neural-network predictions were used alongside LOOCV, neighbourhood analysis and multi-radius local refinement to select candidates. Now that the actual outputs from those candidates are available, their realised performance can be compared with the model predictions.  
The main focus will therefore be on model reliability and function-specific refinement. Prediction errors will be used alongside the existing diagnostics to identify where the GP or NN provides useful guidance and where greater caution is required. The latest observations will then be added to each dataset and the models refitted before generating new candidates.  
The approach will remain different across the eight functions. Functions where recent local refinement has produced new bests can continue to exploit those regions, while functions where predicted improvements failed to materialise may require tighter refinement or reduced reliance on the model responsible. Function 1 will continue to require a more exploratory approach because the surrogate models have not yet provided reliable guidance.  
The aim is not simply to select whichever model produces the highest prediction, but to give greater weight to models and search strategies that have been supported by actual previous results.  

Step-by-step approach:  
Step 1 – Update datasets and assess Round 10 results  
- Add the eight new observations, compare each output with its previous best and identify which strategies produced improvements. This establishes the starting point for the new round.  
Step 2 – Assess realised model prediction accuracy  
- Compare the GP and NN predictions made for last week's submitted candidates with the actual outputs received. Calculate absolute prediction errors and identify where each model performed well or poorly. This will provide additional evidence for how much weight each model should receive this round.  
Step 3 – Refit and diagnose the GP models  
- Fit the RBF GP models to the expanded datasets and review the fitted length scales and log marginal likelihoods. Particular attention will remain on Functions 1 and 2 because their GP length scales previously collapsed to the lower bound.  
Step 4 – Revalidate the NN models  
- Repeat NN LOOCV using the updated datasets and compare the error against the baseline. This will show whether the additional observation changes which functions have sufficiently useful NN models to support candidate selection.  
Step 5 – Generate updated GP and NN candidates  
- Generate candidate points using the same core GP acquisition settings and NN approach used previously. Keeping these broadly consistent allows the effect of the additional data to be assessed rather than changing several parts of the methodology simultaneously.  
Step 6 – Compare candidate agreement and neighbourhoods  
- Compare where the GP and NN candidates lie and inspect their nearest existing observations. Agreement between models will provide stronger evidence for a region, while large disagreement will indicate greater uncertainty and the need for more careful selection.  
Step 7 – Apply function-specific refinement  
- Use the accumulated evidence to decide the appropriate search strategy for each function. Current expectations are tighter refinement for Functions 3 and 7, continued GP-led refinement for Function 4, broader local refinement for Functions 6 and 8, boundary-focused analysis for Function 5, and more cautious/exploratory treatment of Functions 1 and 2. These are starting hypotheses rather than fixed decisions and can change following Steps 2–6.  
Step 8 – Compare and select final candidates  
- Bring together recent improvement history, realised prediction accuracy, GP/NN diagnostics, predicted outputs, uncertainty, model agreement and candidate novelty. Select one candidate for each function based on the combined evidence rather than any single metric.  
Step 9 – Final validation and save  
- Check every selected candidate for the correct number of dimensions, [0,1] bounds, duplicates and distance from existing observations. Format each to six decimal places, save the Round 11 points and submit them through the portal.  

**Step 1. Update datasets and assess Round 10 results**

The eight Round 10 query points and their returned outputs will first be added to the existing datasets. Before any models are refitted, the new outputs will be compared with the previous best for each function. This provides a direct assessment of whether the strategy used for each function in the previous round produced an improvement.
The updated datasets will then be checked to confirm that exactly one new observation has been added to each function and saved separately for use throughout the rest of the analysis.

Function 1:  
Previous best: 0.0000000000  
New output:    0.0000000000  
Difference:    -0.0000000000  
New best:      False   

Function 2:  
Previous best: 0.7469622512  
New output:    0.5572172325  
Difference:    -0.1897450186  
New best:      False  

Function 3:  
Previous best: -0.0008627356  
New output:    -0.0111975944  
Difference:    -0.0103348588  
New best:      False  

Function 4:  
Previous best: 0.3872619363  
New output:    0.5837298257  
Difference:    0.1964678893  
New best:      True  

Function 5:  
Previous best: 8662.4825000000  
New output:    8115.4004465113  
Difference:    -547.0820534887  
New best:      False  

Function 6:  
Previous best: -0.2743879717  
New output:    -0.2397692962  
Difference:    0.0346186755  
New best:      True  

Function 7:  
Previous best: 2.9293698997  
New output:    2.8391956973  
Difference:    -0.0901742024  
New best:      False  

Function 8:  
Previous best: 9.9570315368  
New output:    9.9849814249  
Difference:    0.0279498881  
New best:      True  

The Round 10 results provide mixed but useful evidence about the strategies used in the previous round. Functions 4, 6 and 8 all produced new bests, supporting the decision to continue local refinement while allowing the size of the local step to vary between functions.  
Function 4 provides particularly strong support for continued local refinement, with another substantial improvement from 0.387262 to 0.583730. Function 6 also improved following the larger-radius local candidate, although the gain was smaller than some of its previous improvements. Function 8 produced another new best of 9.984981, continuing its recent pattern of smaller improvements as the output approaches 10.  
The unsuccessful queries are also informative. Function 3 performed considerably worse despite being extremely close to its previous best, suggesting that its strongest region may be narrow and that small movements do not necessarily produce similar outputs. Function 5 fell by around 547 after moving slightly away from [1,1,1,1], providing further evidence that the upper boundary remains particularly important. Function 7 remained relatively strong but failed to improve on its current best.  
Function 2 also failed to improve, with the output falling from the current best of 0.746962 to 0.557217. This is important because the NN had predicted improvement for the submitted candidate and provides a useful test of how much weight should be placed on its predictions this round. Function 1 remains unresolved, as the broader exploratory query again returned an output effectively equal to zero.  
Overall, the results reinforce the need for a function-specific strategy. The successful local searches for Functions 4, 6 and 8 support further refinement, while the results for Functions 2, 3, 5 and 7 suggest that model predictions and local movements need to be treated more cautiously.  

**Step 2. Assess realised model prediction accuracy**

The predictions made by the GP and NN models for the Round 10 submitted candidates will be compared with the actual outputs now received. Absolute prediction error will be calculated for each available model prediction, allowing the previous model recommendations to be assessed using genuinely unseen outputs.  
This adds another measure of model reliability alongside the GP diagnostics and NN LOOCV used previously. The results will be treated as supporting evidence rather than automatically selecting whichever model had the lowest error, as a single realised observation is not enough to establish that one model is consistently more reliable.  

Function 1 could not be assessed because the submitted point was generated using model-independent exploration rather than a GP or NN prediction.  
Function 2 had only an NN prediction available. The NN predicted 0.810157, compared with the actual output of 0.557217, giving an absolute error of 0.252939.  
Function 3 was predicted relatively similarly by both models. The GP had an absolute error of 0.011032, while the NN error was slightly lower at 0.010468.  
Function 4 showed a clear difference between the models. The GP predicted 0.563957 compared with the actual 0.583730, producing an error of only 0.019773. The NN error was considerably larger at 0.272725.  
Function 5 was also predicted more accurately by the GP. Its error was 172.222571, compared with an NN error of 573.203872.  
Function 6 was predicted more accurately by the NN, with an error of 0.010008 compared with 0.043899 for the GP.  
Function 7 favoured the GP, with an error of 0.089696 compared with 0.146204 for the NN.  
Function 8 was predicted closely by both models, although the GP was more accurate. The GP error was 0.008706, compared with 0.024715 for the NN.  

The realised prediction errors provide useful additional evidence about model reliability, but a single new observation for each function is not enough to decide that one model should always be preferred.  
The strongest result is Function 4. The GP prediction of 0.563957 was very close to the realised new best of 0.583730, supporting both the GP model and the local-refinement strategy used for this function. Function 8 provides similar evidence, with the GP predicting the new result to within 0.009. The GP was also more accurate for Functions 5 and 7, although neither candidate improved on the existing best.  
For Function 6, the NN performed particularly well, predicting the new output to within 0.010. This supports continuing to use the NN as useful secondary evidence for this function, especially as the submitted candidate also produced a new best.  
Function 3 is less clear. Both models predicted a value very close to zero, but the actual result was −0.011198. Although the NN had the slightly smaller absolute error, both models failed to capture how quickly performance deteriorated after moving only a small distance from the incumbent. This reinforces the earlier evidence that the high-performing region may be particularly narrow.  
Function 2 provides a warning against relying too heavily on the NN. Its predicted improvement to 0.810157 did not occur, with the actual result falling to 0.557217. This supports reducing the influence of the NN prediction for this function until the updated validation provides stronger evidence.  
Overall, the results support the planned function-specific weighting of model evidence. The GP currently has particularly strong realised support for Functions 4 and 8, while the NN remains useful for Function 6. For the other functions, prediction accuracy should be considered alongside LOOCV, GP diagnostics, improvement history and neighbourhood behaviour rather than used on its own.  

**Step 3. Refit and diagnose the GP models**

The Gaussian Process models will now be refitted using the expanded datasets. The fitted RBF length scales and log marginal likelihoods will be compared with the previous round to identify whether the additional observations have changed the model behaviour.  
Particular attention will be given to Functions 1 and 2, whose GP length scales previously remained at the lower bound of 1e-05, indicating that the models were struggling to identify useful smooth structure. For the remaining functions, changes in the fitted length scales will help assess whether the new observations have altered the estimated local behaviour before new candidates are generated.  

Function 1 continues to fit an RBF length scale of 1e-05, remaining at the specified lower bound. Its log marginal likelihood decreased from −25.540894 to −26.959832.  
Function 2 also remains at the lower bound of 1e-05, with a log marginal likelihood of −26.959832. The additional observation has therefore not resolved the GP fitting problem previously identified for this function.  
Function 3 shows the largest change, with its length scale falling from 0.201 to 0.0968. Its log marginal likelihood is now −29.946737.  
Function 4 remains very stable, with the length scale moving only slightly from 0.313 to 0.316. Its log marginal likelihood improved substantially from −23.818726 to −19.214108.  
Function 5's length scale increased from 0.303 to 0.338, while its log marginal likelihood improved from −23.970400 to −20.623561.  
Function 6 remains relatively stable, with its length scale changing slightly from 0.527 to 0.518. Its log marginal likelihood improved from −22.245983 to −20.566977.  
Function 7 also remains stable, with the length scale changing from 0.323 to 0.320. Its log marginal likelihood improved slightly from −34.151061 to −32.767946.  
Function 8 continues to have the largest fitted length scale, increasing slightly from 1.05 to 1.07. Its log marginal likelihood improved from −21.138922 to −18.081197.  

Functions 1 and 2 continue to show the same GP problem seen previously. Both remain at the minimum permitted length scale, suggesting that the RBF models are still unable to identify useful smooth structure from the observations. The latest data has not improved this, so GP-generated predictions for these two functions should continue to receive limited weight.  
Function 3 is the most significant change this round. The length scale has more than halved from 0.201 to 0.0968 following the latest observation. This fits well with the Round 10 result, where a very small movement from the best point caused the output to fall from −0.000863 to −0.011198. The GP is now modelling the function as varying over a much shorter distance, providing further evidence that the high-performing region may be narrow. This supports using particularly tight refinement for Function 3.  
Function 4 provides strong evidence in the opposite direction. Its length scale has remained almost unchanged despite the latest substantial improvement, while the log marginal likelihood has improved considerably. Combined with the very accurate realised GP prediction from Step 2, this strengthens the case for continuing GP-led local refinement.  
Functions 5, 6 and 7 remain relatively stable, with only modest changes in their fitted length scales. Function 6's stability is encouraging given that its latest candidate produced another new best. Function 5 continues to show relatively smooth structure despite the decline caused by moving away from [1,1,1,1], while Function 7 shows little evidence that the latest unsuccessful query fundamentally changed the GP's view of the function.  
Function 8 also remains highly stable. Its slightly larger length scale and improved log marginal likelihood, combined with another new best and a very accurate GP prediction last round, provide good support for continuing to use the GP as an important part of candidate selection.  
Overall, this strengthens the function-specific picture emerging from Steps 1 and 2 rather than changing it substantially. The most important new information is the sharp reduction in Function 3's length scale, which independently supports the idea that much tighter local behaviour needs to be considered.  

**Step 4. Updated NN LOOCV validation**

The neural-network models will now be revalidated using leave-one-out cross-validation (LOOCV) on the updated datasets. The resulting mean absolute error will again be compared with the simple baseline MAE used in the previous round.  
Keeping the same validation method allows a direct comparison with the previous results and shows whether the latest observation has strengthened or weakened the evidence for using the NN as a supporting model for each function. The results from Step 2 will also remain important, as LOOCV measures performance across the full dataset while realised prediction error measures performance on the actual candidate submitted last round.  

Function 1 continues to fail the NN validation, with the LOOCV MAE increasing from 0.001837 to 0.033792, compared with a baseline MAE of 0.000946.  
Function 2 continues to beat the baseline. Its NN MAE improved from 0.236724 to 0.193346, compared with a baseline of 0.238444.  
Function 3 again fails to beat the baseline, with its NN MAE increasing from 0.056881 to 0.076948, compared with a baseline of 0.054294.  
Function 4 continues to perform strongly, with the NN MAE improving from 1.385355 to 1.221417, substantially below the baseline of 7.632505.  
Function 5 also remains strongly validated, with its NN MAE improving from 255.307417 to 197.316315, compared with a baseline of 2556.291575.  
Function 6 improves from an NN MAE of 0.255144 to 0.161145, remaining comfortably below the baseline of 0.547373.  
Function 7 improves from 0.443789 to 0.349355, compared with a baseline of 0.777671.  
Function 8 continues to beat the baseline, although its NN MAE has increased from 0.309128 to 0.520098. It remains below the baseline of 0.999338.  

The overall validation pattern remains similar to last week, with the NN beating the baseline for six of the eight functions.  
Functions 1 and 3 remain the main exceptions. Function 1 has deteriorated considerably and, combined with the continued GP length-scale problem, there is still little evidence for relying on either model. Function 3 has also deteriorated and again fails to outperform the baseline. Alongside the much shorter GP length scale identified in Step 3, this strengthens the evidence that Function 3 is difficult to model and may contain a particularly narrow high-performing region.  
Function 2 is more mixed. Its overall LOOCV performance has improved and it continues to beat the baseline, but the realised prediction test in Step 2 showed that the NN substantially overestimated last round's submitted candidate. The NN can therefore remain useful as supporting evidence, but its individual predictions should be treated cautiously.  
Functions 4, 5, 6 and 7 all show improved NN validation. This supports continuing to use the NN alongside the GP for these functions rather than removing it based on a single unsuccessful candidate. Function 6 is particularly encouraging because its improved LOOCV performance is also supported by the accurate NN prediction and new best achieved last round.  
Function 8 is slightly different because its NN validation has weakened, despite still beating the baseline. Given that the GP was considerably more accurate on the latest submitted candidate and continues to show stable diagnostics, the GP should probably carry more weight than the NN for Function 8 this round.  

**Step 5. Generate updated GP acquisition candidates**

Updated GP acquisition candidates will be generated using the expanded Round 10 datasets. As in the previous round, several UCB exploration settings and Expected Improvement settings will be compared rather than relying on a single acquisition function.  
The purpose at this stage is not to select the final submission points, but to identify how the GP currently balances exploitation and exploration across the eight functions. The candidate locations, predicted outputs, uncertainty and distance from existing observations will then be compared with NN-supported and local-refinement candidates in later steps.  
The evidence from the earlier analysis will affect how these candidates are interpreted. GP candidates for Functions 1 and 2 will receive limited weight because both models remain at the minimum RBF length scale. Function 3 will require particular attention to very local candidates following the sharp reduction in its fitted length scale. Greater confidence can be placed in GP evidence for Functions 4 and 8 following their accurate realised predictions and stable diagnostics.  

**GP acquisition comparison**

Keeping the same acquisition settings established last week:
- UCB: κ = 0.5, 1.0, 1.5 and 2.5
- EI: improvement thresholds of 0%, 1%, 5% and 10%

Function 1 produces exactly the same candidate across every UCB and EI setting: [0.374540, 0.950714]. The GP predicts −0.000501 with uncertainty of 0.001508 and a nearest-observation distance of 0.195683.  
Function 2 behaves similarly, with every acquisition setting selecting [0.282588, 0.458677]. The predicted output is 0.349989, well below the current best of 0.746962, with relatively high uncertainty of 0.257464.    
Function 3 shows a clear exploration-exploitation trade-off. UCB 0.5 predicts 0.016134 at a distance of 0.055414, while UCB 2.5 moves further away to 0.142024 and predicts −0.046677 with much greater uncertainty. All EI settings select the same candidate, predicting 0.018737 at a distance of 0.049266.    
Function 4 also changes substantially as exploration increases. UCB 0.5 and all EI settings select the same candidate, with a predicted output of 0.763656, above the current best of 0.583730, at a distance of 0.156940. Higher UCB values move progressively further into uncertain regions and produce increasingly negative mean predictions.    
Function 5 shows a strong divide between local and exploratory candidates. UCB settings from 0.5 to 1.5 select a relatively local candidate with a prediction of 6049.596207, below the current best of 8662.482500. UCB 2.5 and all EI settings instead select a distant candidate with very high uncertainty and a much lower mean prediction of 1722.794150.   
Function 6's UCB 0.5 candidate is the most locally promising GP option, predicting −0.222140, slightly better than the current best of −0.239769, at a distance of 0.122735. Higher exploration settings move further away and produce lower predicted outputs.    
Function 7 produces no GP candidate with a predicted output above the current best of 2.929370. UCB 0.5 and 1.0 provide the strongest mean prediction at 2.759404, while increasing exploration produces progressively lower predictions and greater uncertainty.    
Function 8 also shows a strong trade-off. UCB 0.5 remains relatively close to the established region and predicts 9.938809, slightly below the current best of 9.984981. Higher exploration settings move considerably further away, with the most exploratory candidate lying 0.906832 from the nearest observation and having a lower predicted mean of 9.401114.  

The acquisition comparison reinforces the need for a function-specific strategy rather than selecting the same acquisition setting across all eight functions.  
Functions 1 and 2 again demonstrate the limitations of their GP models. Every acquisition setting collapses onto the same candidate, consistent with both fitted length scales remaining at the lower bound. Neither set of GP candidates provides convincing evidence for candidate selection, so alternative methods should remain the main focus for these functions.  
Function 3 is more interesting. The lower-exploration GP and EI candidates predict substantial improvement, but Step 1 showed that last round's extremely local candidate still performed worse than expected, while Step 3 showed the fitted length scale falling sharply. The GP predictions therefore provide useful evidence about the likely region, but tighter local investigation around the current best should be carried out before choosing a candidate.  
Function 4 provides stronger GP evidence. UCB 0.5 and EI identify the same region with a predicted output above the current best. Given the GP's very accurate prediction last round and the consecutive improvements achieved in this area, this region deserves further local refinement. The highly exploratory UCB candidates appear much less attractive.  
Function 5 again suggests that the known boundary optimum at [1,1,1,1] remains difficult to improve upon. Even the more local GP candidate is predicted well below the current best, while the exploratory alternatives rely almost entirely on uncertainty. Last round also demonstrated that moving away from the boundary reduced the realised output. This supports returning to very tight boundary-focused refinement rather than global exploration.  
Function 6 provides a useful candidate region. UCB 0.5 predicts a modest improvement while maintaining much greater proximity to the established high-performing area than the other acquisition settings. Combined with the new best achieved last round and improving NN validation, this supports further local cross-model refinement.  
Function 7 provides little evidence for broad exploration. The GP's best mean prediction remains below the current best and becomes progressively worse as exploration increases. Given the strong NN validation, the next stage should examine whether NN-supported local refinement around the established best region provides better options.  
Function 8 presents a similar case. The global GP acquisition search does not predict improvement, despite the previous local-refinement candidate producing another new best. This suggests that the global acquisition search is too coarse for the current stage of optimisation, and that another targeted local search around the improved incumbent is more appropriate.  
  
**Updated NN candidate recommendations**

Updated NN candidate recommendations will now be generated using the expanded datasets. As in the previous round, the NN will be used as a supporting model rather than as the sole basis for selection, with particular attention given to whether it passed the updated LOOCV validation.  
Candidate quality will be assessed using the predicted output, disagreement between the NN models and distance from the nearest existing observation. The realised prediction results from Step 2 will also be considered later when deciding how much weight to give the NN for each function.  
NN recommendations for Functions 1 and 3 will be treated cautiously because neither model passed LOOCV. Function 2 passed LOOCV but substantially overpredicted last round's submitted point, so its recommendation will also require additional validation. Greater weight can currently be given to NN evidence for Functions 4–7, particularly Function 6, where both LOOCV and the latest realised prediction were encouraging.  

Function 1 produces an NN candidate of [0.014679, 0.660492], with a predicted output of 0.015695 and a nearest-observation distance of 0.265804. However, the NN again fails LOOCV validation, so the prediction has limited reliability.  
Function 2 produces a candidate of [0.998959, 0.845004], with a predicted output of 1.004493, considerably above the current best of 0.746962. The candidate is 0.138158 from the nearest observation and the NN passes LOOCV.  
Function 3 recommends a very local candidate only 0.014162 from an existing observation, with a predicted output of −0.005204. The NN again fails LOOCV validation.  
Function 4 produces a relatively local candidate with a predicted output of 0.117631, substantially below the new best of 0.583730. The NN passes LOOCV but shows disagreement of 0.128332.  
Function 5 also remains relatively close to the established high-performing region. Its predicted output of 6187.342810 is well below the current best of 8662.482500, despite passing LOOCV.  
Function 6 produces a candidate 0.094106 from its nearest observation, with a predicted output of −0.273307. This is below the new best of −0.239769, although the NN passes LOOCV and disagreement remains low at 0.012744.  
Function 7 recommends a candidate relatively close to the established high-performing region. Its predicted output of 2.900738 is slightly below the current best of 2.929370, with relatively low disagreement of 0.051289.  
Function 8 is the most aggressive NN recommendation. It predicts 10.434491, substantially above the current best of 9.984981, but the candidate lies 0.677331 from its nearest observation and NN disagreement is relatively high at 0.272532.  

The NN results provide useful supporting evidence, but they also reinforce why validation and realised prediction accuracy need to be considered alongside the headline predictions.  
Functions 1 and 3 remain unsuitable for NN-led selection because both continue to fail LOOCV. Function 3's candidate is at least located very close to the established high-performing region, but its predicted output should not be given substantial weight.  
Function 2 requires particular caution. The NN again predicts a large improvement, this time above 1.0, and continues to pass overall LOOCV. However, last round it predicted 0.810157 and the actual output was only 0.557217. The new candidate also moves into a different region of the search space. The prediction is therefore interesting, but not strong enough on its own to justify submission.  
For Function 4, the NN does not identify an improvement despite its good overall validation. This contrasts with the GP, which both predicted last round's new best accurately and currently identifies another potentially improving region. The GP should therefore carry more weight for Function 4.  
Function 5 provides agreement at a broader level: neither the GP nor NN currently predicts that moving away from the known boundary region will improve on [1,1,1,1]. This strengthens the case for very targeted boundary analysis rather than following either global recommendation.  
Function 6 is slightly more mixed. The NN remains well validated and accurately predicted last round's result, but its new global candidate is predicted below the incumbent. The GP's lower-exploration candidate was more optimistic. Rather than choosing between them now, another local search around the new best should be more informative.  
Function 7 also provides little support for a global move. Both GP and NN candidates remain below the current best, while the NN candidate stays reasonably close to the established strong region. This supports another local refinement rather than broader exploration.  
Function 8's headline NN prediction should be treated cautiously. Although 10.434491 looks attractive, it is a substantial extrapolation: the candidate is 0.677331 from the nearest observation, disagreement is relatively high, NN LOOCV weakened this round, and the NN was less accurate than the GP on last round's submitted point. The result is therefore more useful as evidence that another potentially interesting region exists than as a submission candidate itself.  

**Step 6. GP–NN candidate agreement**

The updated NN recommendation for each function will now be compared with the GP acquisition candidates generated in Step 5a. Euclidean distance will be used to measure how closely the different models are targeting the same areas of the search space.  
The purpose is not to assume that agreement between the models makes a candidate correct, but to identify where independently generated recommendations converge on similar regions. Close agreement can provide additional support where both models have performed reliably, while large disagreement may indicate different interpretations of the available observations.  
The results will be interpreted alongside the earlier evidence. In particular, agreement will carry less weight for Functions 1 and 3 because their NNs fail LOOCV, and for Function 2 because of the NN's poor realised prediction last round. Similarly, agreement involving the unreliable GP models for Functions 1 and 2 will not be treated as strong evidence.  

Function 1 shows limited agreement between the two models. The NN candidate is 0.462308 from every GP candidate, with all GP acquisition methods continuing to select the same point.  
Function 2 shows even greater disagreement. The NN candidate is 0.813902 from the GP candidate, which again remains identical across all acquisition settings.  
Function 3 shows much stronger agreement. The NN candidate is only 0.051914 from the EI candidates and 0.061613 from UCB 0.5. The distance increases as the UCB exploration parameter rises, reaching 0.169131 for UCB 2.5.  
Function 4 shows moderate agreement. The NN candidate is 0.197230 from UCB 0.5 and all EI candidates. The distance increases to 0.657291 for the more exploratory UCB 2.5 candidate.  
Function 5 shows complete agreement between the NN and the lower-exploration GP approaches. The NN candidate exactly matches the UCB 0.5, 1.0 and 1.5 candidates, giving a distance of 0.000000. The more exploratory UCB 2.5 and EI candidates are substantially different, at a distance of 1.631172.  
Function 6 shows reasonable agreement between the NN and UCB 0.5, with a distance of 0.170772. The remaining candidates are further away, particularly UCB 2.5 at 1.207496.  
Function 7 shows moderate agreement, with the NN candidate 0.246767 from the closest GP candidates. This increases to 0.331133 for the more exploratory alternatives.  
Function 8 shows the greatest disagreement. The NN candidate is 0.949228 from even the closest GP candidate and approximately 1.39 from the majority of the GP recommendations.  

The GP–NN comparison reinforces the differences already emerging between the functions and helps identify where agreement between the models provides useful additional evidence.  
Functions 1 and 2 show little agreement. Combined with the GP length-scale problems identified earlier, this provides no strong reason to favour regions simply because one of the models recommends them. Function 2 is particularly uncertain because the NN continues to produce optimistic predictions despite substantially overpredicting the submitted point last round.  
Function 3 shows the strongest useful agreement, with the NN, EI and lower-exploration UCB approaches all targeting a similar area. However, the NN still fails LOOCV and the GP length scale has fallen considerably. The agreement therefore supports the region rather than any individual candidate, making a tighter local search the more appropriate next step.  
Function 4 also provides useful cross-model agreement. Although the NN predicts a lower output than the GP, both are targeting reasonably similar areas. Given the GP's strong realised accuracy last round and the latest new best, further local refinement around the established high-performing region remains well supported.  
Function 5 has the strongest numerical agreement, with the NN and several GP settings selecting exactly the same candidate. However, both models predict this candidate substantially below the current best. Agreement alone therefore does not make it attractive. The repeated evidence still points towards the [1,1,1,1] boundary as the important area to investigate.  
Function 6 shows enough agreement to suggest that both models are identifying a similar broader region. This is supported by the new best achieved last round, improved NN validation and relatively good realised NN accuracy. A targeted local search around the new best should therefore be more informative than either global recommendation alone.  
Function 7 again points towards the established high-performing region without either model predicting a clear improvement. This favours local refinement rather than increasing exploration.  
Function 8 provides a useful warning. The NN's predicted 10.434491 comes from a completely different area from the GP recommendations. Combined with its large distance from existing observations, relatively high disagreement, weaker LOOCV this round and poorer realised accuracy than the GP last round, there is not enough support to chase this candidate directly. Local refinement around the new best remains the more defensible approach.  

**Step 7. Candidate neighbourhood analysis**

The candidate neighbourhoods will now be examined by identifying the three nearest existing observations to the NN, UCB 0.5 and UCB 2.5 candidates for each function. This follows the same approach as last week.  
The purpose is to determine whether each model-generated candidate is supported by strong observed outputs in its surrounding region. This provides an additional check before local refinement, particularly where a model prediction is optimistic but the candidate is relatively far from previously sampled points.  
UCB 0.5 represents the more exploitative GP recommendation, while UCB 2.5 provides a more exploratory comparison. The NN candidate provides the alternative model-based recommendation. The distances and outputs of neighbouring observations will be considered alongside the model diagnostics, validation and realised prediction accuracy from the previous steps.  

Function 1 provides little useful neighbourhood evidence for either model. All three nearest observations around both the NN and GP candidates have outputs effectively equal to zero, while the candidates themselves remain relatively distant from existing observations.  
Function 2 provides limited support for the NN region. Its nearest observations have outputs of 0.420586, 0.293993 and −0.082041, all below the current best of 0.746962. The GP region is less encouraging, with neighbouring outputs of −0.065624, 0.214965 and −0.013858.  
Function 3 shows that the NN and UCB 0.5 candidates remain close to the established high-performing region. The NN candidate is only 0.016794 from the current best observation of −0.000863, while UCB 0.5 is 0.066263 from it. The more exploratory UCB 2.5 candidate is further away and surrounded by weaker outputs.  
Function 4 provides strong neighbourhood support for the NN candidate, which lies only 0.052981 from the latest best of 0.583730 and is also close to the previous bests of 0.387262 and 0.186912. UCB 0.5 remains connected to this improving region but at greater distances. UCB 2.5 is substantially further away and surrounded by strongly negative observations.  
Function 5 confirms that the NN and UCB 0.5 candidates remain in the established high-output boundary region, with nearby outputs of 6691.931674, 8115.400447 and 7121.151955. However, these remain below the best output of 8662.482500 at [1,1,1,1]. UCB 2.5 lies in a clearly weaker region, with neighbouring outputs below 234.  
Function 6 shows good neighbourhood support for both the NN and UCB 0.5 candidates. Both are closest to the new best of −0.239769, followed by the previous bests of −0.274388 and −0.379740. UCB 2.5 is much further away and surrounded by substantially weaker outputs.  
Function 7 also provides strong support for the NN region. Its three nearest observations are 2.929370, 2.839196 and 2.922059, all among the strongest observed outputs. The GP candidates remain connected to the same general high-performing region, although at greater distances.  
Function 8 provides little support for the optimistic NN candidate. Its two closest observations produce only 9.141639 and 9.183005, while the candidate itself is more than 0.67 from any existing observation. In contrast, UCB 0.5 is surrounded by strong observations of 9.984981, 9.840530 and 9.930811. The highly exploratory UCB 2.5 candidate is considerably more distant and less well supported.  

The neighbourhood analysis strengthens the case for local refinement rather than broad exploration across most of the functions.  
Function 1 remains unresolved. Neither model has demonstrated reliable predictive ability and the surrounding observations provide no meaningful differentiation. A model-independent exploration approach therefore remains more appropriate than relying on either recommendation.  
For Function 2, the NN region is somewhat better supported than the GP region, but there is still no observed evidence close to its predicted output of 1.004493. Combined with last round's NN overprediction, this reinforces the need for caution. Rather than submitting the global NN candidate directly, a more targeted investigation around the established best region is likely to be preferable.  
Function 3 provides further evidence that the useful region remains close to the existing best. The lower-exploration candidates are surrounded by the strongest observations, whereas moving further away produces weaker neighbourhoods. Combined with the reduced GP length scale, this supports very tight local refinement around the incumbent.  
Function 4 has one of the clearest patterns. The NN candidate sits close to a sequence of increasingly strong observations, while the lower-exploration GP candidate remains connected to the same region. The exploratory UCB 2.5 region is poorly supported. This provides strong evidence for continuing local refinement around the latest best.  
Function 5 again confirms the importance of the upper boundary. Although the NN and low-UCB candidate occupy a generally strong region, none of their neighbouring observations outperform [1,1,1,1]. The evidence therefore continues to favour tight boundary-focused refinement rather than moving substantially away from the known best.  
Function 6 also shows a clear high-performing local region. Both the NN and low-exploration GP candidate are surrounded by the recent sequence of improvements. This supports another local search around the latest best rather than following the much weaker exploratory candidate.  
Function 7 provides similar evidence. The NN candidate is particularly well supported by neighbouring observations close to the current best, while the GP candidates move progressively further away. Local refinement around the established high-performing cluster is therefore the stronger approach.  
Function 8 gives perhaps the clearest reason to reject a headline model prediction. The NN's predicted 10.434491 is not supported by its immediate neighbourhood, whereas the lower-exploration GP candidate is surrounded by the strongest observations in the dataset. This strengthens the case for ignoring the distant NN candidate for now and refining around the new best instead.  

**Step 8. Multi-radius local refinement**

The next stage will refine candidate selection around the strongest regions identified in the previous analysis. Rather than applying the same search method to every function, the approach will reflect the different levels of model reliability and optimisation behaviour now observed.  
For Functions 3, 4, 6, 7 and 8, local candidate searches will be performed around the current best point using radii of 0.03, 0.05 and 0.10. Candidates will be assessed using GP prediction, GP uncertainty, NN prediction, NN disagreement and distance from existing observations. This allows the effect of gradually widening the local search to be assessed before selecting a final candidate.  
Function 3 will be treated particularly cautiously following the reduction in GP length scale and the weaker Round 10 result. For Functions 4, 6, 7 and 8, the aim is to determine whether the improving regions identified in previous rounds can be refined further.  
Functions 1, 2 and 5 will be handled separately afterwards. Function 1 will continue to use model-independent exploration, Function 2 will use targeted local refinement around its established best, and Function 5 will use novelty-constrained boundary refinement around [1,1,1,1].  

**Generate local candidates**

Function 3 produces very similar candidates across all three search radii. The candidates remain approximately 0.032 from the current best, with GP predictions ranging from 0.011428 to 0.012920. In contrast, the NN remains more conservative, predicting between −0.000707 and −0.001088. The radius 0.10 candidate has the strongest GP prediction and lowest GP uncertainty, although the NN prediction is slightly below the current best.  
Function 4 also shows strong convergence across the three radii, with all candidates extremely close to the current best. GP predictions range from 0.604620 to 0.649909, all above the current best of 0.583730. The NN predictions are lower at 0.534004–0.555554. The radius 0.05 candidate produces the highest GP prediction of 0.649909, while remaining only 0.007193 from the current best.  
Function 6 shows progressively broader movement as the search radius increases. The GP prediction improves from −0.215407 at radius 0.03 to −0.204221 at radius 0.10, all above the current best of −0.239769. The NN remains relatively stable at approximately −0.230 to −0.232. GP uncertainty and NN disagreement both increase as the search becomes wider.  
Function 7 produces promising candidates at all three radii. Both models predict improvement over the current best for the 0.03 and 0.05 candidates. The radius 0.10 candidate produces the strongest GP prediction of 3.012495, but the NN prediction falls to 2.916688 and both GP uncertainty and NN disagreement increase. The smaller-radius candidates therefore show stronger agreement between the models.  
Function 8 shows a clearer difference between the GP and NN. The GP predicts values close to but slightly below the current best across all three radii, ranging from 9.972053 to 9.984110. The NN predicts improvement throughout, reaching 10.161809 at radius 0.10. However, NN disagreement increases substantially as the radius expands, while the candidate also becomes more distant from existing observations.  

The multi-radius analysis provides fairly clear direction for most of these functions, although the appropriate level of local refinement differs.  
For Function 3, all three searches converge on almost exactly the same region despite the different radius limits. This suggests that the candidate region is being selected because of the modelled local structure rather than simply because a wider search is available. However, the GP and NN disagree on whether this region actually improves on the current best. The GP is optimistic while the NN, which fails LOOCV, remains close to the existing best. Given the poor Round 10 result and the shorter GP length scale, this candidate should therefore be treated cautiously despite the apparently strong GP prediction.  
Function 4 provides much stronger evidence for continued fine refinement. All three searches converge extremely close to the latest best, and the GP predicts further improvement throughout this region. This is particularly encouraging because the GP predicted the previous submitted point accurately. The radius 0.05 candidate currently provides the strongest GP prediction while remaining very close to the established high-performing region.  
Function 6 also provides encouraging evidence. Both models predict improvement over the current best at all three radii, although the GP becomes increasingly optimistic as the search expands while uncertainty also rises. The radius 0.10 candidate offers the largest predicted GP improvement, but the smaller candidates provide greater model agreement and lower uncertainty. This creates a useful exploration–exploitation trade-off rather than an obvious reason to select the widest search.  
Function 7 shows perhaps the strongest cross-model evidence. Both models predict improvement at radii 0.03 and 0.05, with relatively low disagreement and uncertainty. Although the GP prediction rises above 3.0 at radius 0.10, the NN no longer predicts improvement and uncertainty increases considerably. The 0.03–0.05 region therefore appears better supported than the more ambitious 0.10 candidate.  
Function 8 requires more caution. The NN continues to predict improvement, but its optimism increases as the candidate moves further from observed data. The GP, which was considerably more accurate than the NN last round, does not predict an improvement at any radius. The smaller-radius candidates therefore appear more defensible than the 0.10 candidate, while there is currently little evidence for following the NN towards a substantially more exploratory point.  
Overall, this stage strengthens the case for fine local refinement for Functions 4 and 7, while Function 6 may justify a slightly broader local move. Functions 3 and 8 contain greater disagreement between the models and should therefore remain more conservative.  

**Function 5. Novelty-Constrained Boundary Refinement**

Function 5 will again be treated separately because its current best remains at the exact upper boundary [1, 1, 1, 1], with an output of 8662.482500. The standard local search is unsuitable here because candidate generation tends to collapse back towards the existing best point, providing almost no novelty.  
As last week, minimum novelty constraints of 0.03, 0.05 and 0.10 will therefore be applied. This forces each candidate to remain a meaningful distance from all existing observations while still searching around the high-performing boundary region.  
The candidates will be compared using GP prediction, GP uncertainty, NN prediction and NN disagreement. Particular attention will be given to whether this week's additional observation at [1.000000, 0.969998, 1.000000, 1.000000], which returned 8115.400447, provides stronger evidence about how quickly performance declines when moving away from the exact boundary.  

Function 5 shows a clear decline in predicted performance as the novelty requirement increases. At minimum novelty 0.03, both models select a candidate extremely close to the upper boundary, [1.000000, 1.000000, 1.000000, 0.969985]. The GP predicts 8156.394720 and the NN predicts 8144.043455, both below the current best of 8662.482500.  
At minimum novelty 0.05, the candidate moves further along the fourth dimension to approximately 0.95. The GP and NN predictions fall to 7793.218387 and 7800.066846 respectively, while uncertainty and disagreement both increase.  
At minimum novelty 0.10, the fourth dimension falls further to approximately 0.895. Predicted performance decreases substantially to 6717.968992 for the GP and 6887.668776 for the NN. GP uncertainty also rises to 219.056976 and NN disagreement to 73.315538.  

The novelty-constrained analysis provides stronger evidence than last week that the optimum for Function 5 is concentrated very close to the upper boundary.  
Both models now agree closely on the expected effect of moving away from [1,1,1,1]. Even the smallest novelty constraint produces predictions approximately 500 points below the current best, while larger movements cause progressively greater deterioration.  
This is also consistent with the latest realised result. Last week's candidate [1.000000, 0.969998, 1.000000, 1.000000] returned 8115.400447, compared with the best of 8662.482500. The new 0.03 candidate effectively performs a similar experiment on the fourth dimension rather than the second, and both models predict a similar reduction in output.  
The close agreement between the GP and NN is useful here because both models pass their respective validation checks for Function 5, while the GP was also considerably more accurate than the NN on last week's submitted candidate. However, neither model currently provides evidence that a meaningful move away from [1,1,1,1] is likely to improve the existing best.  
Overall, Function 5 now provides increasing evidence that the optimum may lie very close to the upper boundary. Further refinement may still help determine the sensitivity of individual dimensions, but repeating increasingly small movements away from [1,1,1,1] is unlikely to provide as much value as it did in earlier rounds.  

**Function 1. Model-Independent Exploration**

Function 1 will again use model-independent exploration rather than a GP- or NN-based recommendation. The GP remains unreliable, with its RBF length scale fixed at the lower bound of 1e-05, while the NN continues to fail LOOCV validation. The latest neighbourhood analysis also provides no useful evidence for either model's candidate, with surrounding outputs effectively equal to zero.  
The previous exploration point also returned an output extremely close to zero, so there is still no evidence of a promising region to exploit. The aim will therefore remain to gather information from an underexplored part of the search space.  
A large random candidate set will be generated and the point with the greatest distance from its nearest existing observation will be selected. This maximises novelty without relying on either predictive model.  

Function 1 produces a model-independent exploration candidate of [0.002495, 0.694432]. The point is 0.301746 from its nearest existing observation, with the next two closest observations at distances of 0.302489 and 0.324234.  
This provides a relatively novel point within the remaining unexplored search space without relying on either predictive model.  

The result remains consistent with the exploration strategy used for Function 1 in the previous round. The candidate is substantially separated from existing observations, providing an opportunity to gather information from a poorly sampled region.  
There is still no reliable evidence for switching back to model-led exploitation. The GP length scale remains at its lower bound, the NN fails LOOCV, and previous observations continue to provide almost no useful distinction between candidate regions.  
The candidate should therefore remain a strong option for Function 1 when the final submission points are selected. Its purpose is primarily information gathering rather than predicted improvement, which is appropriate given the continuing model uncertainty.  

**Function 2. Targeted local refinement**

Function 2 will again use targeted local refinement around the current best point rather than following either global model recommendation directly.  
The GP remains unreliable, with its RBF length scale at the lower bound of 1e-05, meaning its predictions should not be used to guide the local search. The NN continues to pass LOOCV and can therefore provide some useful information about the local structure. However, its influence will remain limited because last round it predicted 0.810157 for the submitted point, while the realised output was only 0.557217.  
The current best remains 0.746962 at [0.713092, 0.138155]. Candidates will be generated at approximately 0.03, 0.05 and 0.10 from this point, following the same approach as last week. The NN prediction, disagreement, distance from the current best and distance from existing observations will then be compared.  
This allows the search to investigate whether there is further improvement around the known high-performing region without relying on the NN's much more speculative global candidate.  

Function 2 produces increasingly distant candidates as the search radius expands, but the NN prediction declines at each stage.  
At radius 0.03, the candidate [0.707291, 0.167589] is predicted at 0.631267, with low NN disagreement of 0.018715. However, it is only 0.009635 from an existing observation.  
At radius 0.05, the candidate [0.702093, 0.186930] has a slightly lower predicted output of 0.622806. NN disagreement falls to 0.013365, while the candidate provides slightly more novelty with a nearest-observation distance of 0.029174.  
At radius 0.10, the candidate [0.705183, 0.237842] is predicted at 0.606641. This is the lowest prediction of the three, while NN disagreement increases to 0.036773 and the nearest-observation distance increases to 0.079817.  
None of the three candidates are predicted to outperform the current best of 0.746962.  

This week's local refinement provides a noticeably different pattern from last week. Previously, the NN became increasingly optimistic as the search moved away from the current best, eventually predicting 0.810157 at radius 0.10. That candidate subsequently produced only 0.557217.  
After incorporating that result into the dataset, the updated NN no longer predicts that moving further in this direction will improve performance. Instead, predicted output steadily declines from 0.631267 to 0.606641 as the radius increases.  
This provides useful evidence that the current best region remains preferable to further movement in the direction identified by the local NN search. The radius 0.03 candidate has the strongest prediction and lowest distance from the current best, but it is also only 0.009635 from an existing observation and is still predicted substantially below the incumbent.  
The result also weakens the case for following the global NN candidate from Step 5b. Although that candidate was predicted at 1.004493, it lies in a different region and its surrounding observations were substantially weaker than the current best. Given the NN's overprediction last round, there is insufficient evidence to rely on that prediction alone.  
Overall, Function 2 currently provides no clear model-supported improvement candidate. A conservative local candidate could still be used to learn more about the best region, but there is less reason than last week to make a relatively large move based on the NN.  

**Step 9. Final Candidate Selection**

The final candidate for each function will now be selected using the evidence gathered throughout the round rather than relying on a single model or acquisition method. Selection will consider the latest realised results, GP reliability, NN LOOCV performance, realised prediction accuracy, neighbourhood analysis, local refinement and candidate novelty.  
Where both models are reliable, preference will be given to candidates supported by both. Where one model has shown stronger realised accuracy, greater weight will be placed on that model. For Functions 1 and 2, where the GP remains unreliable, the function-specific approaches developed in Step 8 will be retained.  
The aim is to balance predicted improvement, model agreement and novelty, while avoiding unnecessarily aggressive candidates where the evidence favours continued local refinement.  

Function 1: [0.002495, 0.694432]
- Model-independent exploration; both predictive models remain unreliable.
Function 2: [0.702093, 0.186930]
- 0.05 local candidate gives a better novelty/reliability balance than 0.03, without making the larger 0.10 move.
Function 3: [0.440641, 0.482965, 0.435994]
- 0.05 candidate; very similar across all radii, with strong GP prediction but enough caution given model disagreement.
Function 4: [0.388750, 0.360865, 0.352431, 0.416723]
- 0.05 candidate has the strongest GP prediction and remains close to the newly established best region.
Function 5: [1.000000, 1.000000, 1.000000, 0.969985]
- Smallest novelty move; larger departures from the boundary are strongly predicted to reduce performance.
Function 6: [0.444874, 0.347937, 0.533470, 0.742972, 0.000000]
- 0.05 candidate balances predicted improvement with lower uncertainty than the wider 0.10 search.
Function 7: [0.145227, 0.296570, 0.489754, 0.259986, 0.333711, 0.644995]
- 0.05 candidate has strong agreement between GP and NN without the increased uncertainty of radius 0.10.
Function 8: [0.000000, 0.221175, 0.182914, 0.081049, 0.771543, 0.527149, 0.198086, 0.585768]
- 0.05 provides useful novelty while avoiding both the wider 0.10 local move and the much more speculative global NN candidate.

Review of selection:  
Function 4 is one of the strongest exploitation choices this round. The previous candidate produced a new best of 0.583730 and the GP prediction was particularly accurate. The selected candidate stays very close to this region while the GP predicts another improvement to approximately 0.649909.  
Function 6 also follows the improving region, but the 0.05 candidate is preferable to automatically taking the highest GP prediction at radius 0.10. Both models support the general direction and the selected point maintains a reasonable balance between improvement and uncertainty.  
Function 7 is similar. The 0.03 and 0.05 candidates receive support from both models, whereas the 0.10 candidate introduces noticeably greater disagreement. The 0.05 candidate provides slightly more novelty without moving into the less certain region.  
Function 8 is deliberately more conservative than the NN recommendation. The GP was the more accurate model last round and does not support the NN's much more aggressive global candidate. The 0.05 local candidate therefore keeps the search around the newly improved region while still providing meaningful novelty.  
Functions 2 and 3 are less clear-cut. Neither currently provides strong cross-model evidence of improvement, so these selections are intended to gather useful local information without making unnecessarily large moves.  
For Function 5, there is an important limitation: even the selected candidate is predicted below the current best by both models. Its value is therefore primarily in testing whether the fourth dimension behaves similarly to the second dimension near the upper boundary rather than because it is expected to produce a new best.  

**Results**

All eight final candidates have the correct dimensionality, fall within the permitted [0,1] range and are not duplicates of existing observations.  
The amount of novelty varies deliberately between functions. Function 1 is the most exploratory, with a nearest-observation distance of 0.301746, reflecting the continued lack of reliable model guidance. Functions 4 and 5 are much more exploitative, with distances of 0.007193 and 0.030015, reflecting stronger evidence that their optima lie close to the currently identified regions.  
Functions 2, 3, 6 and 7 make relatively small local adjustments, with nearest-observation distances between approximately 0.029 and 0.056. Function 8 makes a somewhat broader move at 0.099346, while still remaining considerably closer to known observations than the more speculative global NN candidate.  

The final selections reflect the function-specific strategy developed throughout this round rather than applying the same optimisation method to all eight functions.  
Function 1 remains focused on exploration because neither model currently provides reliable guidance. Function 2 stays close to its established high-performing region but avoids an extremely small adjustment or another aggressive NN-led move.  
Functions 3, 4, 6 and 7 primarily use local refinement. Function 4 is particularly exploitative following its new best in Round 10 and the strong realised accuracy of the GP. Function 6 also builds directly on its latest improvement, while Function 7 uses the region where the GP and NN provide relatively strong agreement. Function 3 remains more cautious because of weaker model reliability and the unsuccessful previous local refinement.  
Function 5 continues to test the upper boundary, but the latest realised result and updated model predictions provide increasing evidence that [1,1,1,1] is close to the optimum. The selected point therefore uses only the minimum novelty constraint.  
Function 8 continues refinement around the latest new best rather than following the NN's much more exploratory global recommendation. This gives greater weight to the GP, which was considerably more accurate on the previous submitted point.  
Overall, the final set maintains the same balance used throughout the project: exploitation where repeated observations and model performance provide evidence for it, and greater exploration where the models remain unreliable or the optimum is less clearly defined.  

**Module 22 (Week 11)**

**Methodology**

This week, I decided to introduce clustering as an additional way of analysing the structure within the observations collected so far. This follows naturally from the neighbourhood analysis and local refinement used in recent rounds, which have increasingly focused on whether strong individual observations are supported by other high-performing points nearby. Rather than only looking at the closest observations to a candidate, clustering can help identify whether there are broader regions of the search space that consistently perform well. K-Means will therefore be introduced as an additional diagnostic alongside the existing GP and neural-network models, rather than replacing them. The aim is to use the clusters to assess whether current high-performing points belong to genuinely promising regions or are more isolated results. This information can then be combined with previous model accuracy, neighbourhood performance and the existing exploration–exploitation approach to decide whether each function should continue with local refinement or move towards wider exploration. Clustering will only be given weight where the resulting groups appear meaningful, rather than assuming that every function must contain useful cluster structure

Step 1 – Incorporate and assess the latest results  
- Add the new observations to each function's dataset and compare the outputs with the previous best values. Identify which functions improved and where the latest candidates failed to improve on the existing best.  
Step 2 – Review realised model prediction accuracy  
- Compare the latest outputs with the GP and NN predictions made during the previous round. Calculate prediction errors where comparable and use these results as additional evidence of which model currently appears more reliable for each function.  
Step 3 – Update the existing model diagnostics  
- Refit the GP models using the expanded datasets and review the fitted kernels, length scales and any warnings. Repeat NN LOOCV validation to determine whether the NN continues to provide useful predictive information relative to the baseline.  
Step 4 – Analyse optimisation progress  
- Review the development of the best observed output for each function, including the location of the current best, number of observations since the last improvement and frequency of previous improvements. Use this to distinguish functions that are still responding to local refinement from those showing greater evidence of stagnation.  
Step 5 – Apply K-Means clustering to the accumulated observations  
- Apply K-Means to each function's input observations to investigate whether identifiable regions have developed within the search space. Test values of k from 2 to 5 and use silhouette scores to select the most appropriate clustering structure for each function rather than assuming the same number of clusters throughout.  
Step 6 – Assess cluster performance and relevance  
- Compare the size, mean output, best output and centroid of each identified cluster. Determine which cluster contains the current best observation and assess whether strong performance is concentrated within a broader region or associated with more isolated observations. Use the silhouette score to determine how much confidence should be placed in the clustering evidence.  
Step 7 – Generate cluster-guided candidate alternatives  
- Use the strongest-performing clusters as an additional indication of promising search regions. Generate candidates within these regions and evaluate them using the established GP and NN models, while retaining a minimum distance from previous observations. Treat clustering as supporting evidence rather than automatically selecting cluster centroids or cluster-generated candidates.  
Step 8 – Generate function-specific refined candidates  
- Produce a second set of candidate alternatives using strategies tailored to the accumulated evidence for each function. Retain model-independent exploration for F1, NN-supported cluster refinement for F2, local GP–NN refinement where appropriate, and boundary-focused refinement for F5. Adjust GP and NN influence where realised prediction accuracy provides evidence that one model should receive greater weight.  
Step 9 – Compare candidate alternatives and select provisional points  
- Compare the cluster-guided and function-specific candidates using predicted performance, model agreement, uncertainty, novelty, clustering evidence and previous optimisation behaviour. Select the stronger candidate for each function rather than applying a single selection rule across all eight.  
Step 10 – Carry out targeted refinement where required  
- Perform additional refinement only where the candidate comparison leaves a meaningful unresolved trade-off. For F5, investigate an intermediate novelty range close to the upper boundary to balance the strong evidence around [1,1,1,1] against the value of obtaining a sufficiently distinct new observation.  
Step 11 – Validate and submit the final candidates  
- Check each selected point for the correct dimensionality, [0,1] bounds, duplication and distance from existing observations. Once all candidates pass validation, format and submit the final point for each of the eight functions.  

**Step 1. Update datasets and assess Round 11 results**

The latest query results were compared with the previous best output for each function before being added to the datasets. This provides an initial assessment of whether the previous round's strategy produced an improvement and ensures that the new observations are evaluated against the existing data before the models are updated. The new input-output pairs were then added to their respective datasets ready for this week's analysis.  

Function 1:  
Previous best: 0.0000000000      
New output:    0.0000000000  
Difference:    -0.0000000000  
New best:      False  

Function 2:  
Previous best: 0.7469622512  
New output:    0.6661526891  
Difference:    -0.0808095621  
New best:      False  

Function 3:  
Previous best: -0.0008627356  
New output:    -0.0105822402  
Difference:    -0.0097195046  
New best:      False  

Function 4:  
Previous best: 0.5837298257  
New output:    0.6229498177  
Difference:    0.0392199920  
New best:      True  

Function 5:  
Previous best: 8662.4825000000  
New output:    8115.1776552646  
Difference:    -547.3048447354  
New best:      False  

Function 6:  
Previous best: -0.2397692962  
New output:    -0.2832282435  
Difference:    -0.0434589472  
New best:      False  

Function 7:  
Previous best: 2.9293698997  
New output:    3.0246097937  
Difference:    0.0952398940  
New best:      True  

Function 8:  
Previous best: 9.9849814249  
New output:    9.9606106544  
Difference:    -0.0243707705  
New best:      False  

Functions 4 and 7 produced new best results. Function 4 improved from 0.583730 to 0.622950, providing further support for the local refinement around the promising region identified in recent rounds. Function 7 also improved from 2.929370 to 3.024610, suggesting that the local region supported by the GP, NN and neighbourhood evidence remains productive.  
The remaining functions did not improve their existing bests. Function 2 still returned a relatively strong 0.666153, while Function 3 again produced a weaker result close to the current best region. Function 5 returned 8115.18, further supporting the previous evidence that moving away from its best boundary point reduces performance. Functions 6 and 8 remained relatively close to their existing bests but did not improve them. Function 1 again returned a value effectively equal to zero despite wider exploration, showing that it remains difficult to identify a useful high-performing region.  
Overall, the results support continuing with a function-specific strategy, while this week's clustering analysis should help determine whether the apparent high-performing areas represent consistent regions or are based mainly on isolated observations.  

**Step 2. Review realised model prediction accuracy**

Before refitting the GP and neural-network models with the latest observations, I compared the predictions made during the previous round with the outputs that were actually returned. This provides an out-of-sample check of how accurately each model predicted a point it had not yet observed, rather than assessing performance after the new result has already been included in the training data.  
For each submitted point, I used the GP and NN predictions recorded at the time the candidate was generated and calculated the absolute difference between the predicted and actual output. Where predictions from both models were available, their errors were compared to identify which model was closer. This builds on the realised prediction analysis introduced in the previous round and allows model reliability to be tracked over time rather than judging a model from a single prediction.  
Not every submitted point had predictions from both models because some candidates were selected using function-specific methods. Function 1, for example, was chosen through model-independent exploration, while Function 2 was selected through NN-supported local refinement. These missing predictions were therefore left as unavailable rather than estimating them retrospectively.  
The results will be used alongside model validation, neighbourhood evidence and this week's clustering analysis when deciding how much weight to give the GP and NN for each function.  

Results:  
For Function 2, only the NN prediction was available. Its prediction of 0.6228 was reasonably close to the actual result of 0.6662, with an absolute error of 0.0433. This provides some support for continuing to consider the NN for F2, although it cannot be directly compared with the GP for this round.  
For Function 3, the NN was more accurate, with an error of 0.0099 compared with 0.0221 for the GP. Importantly, the GP predicted a positive output while the actual result remained negative, so the NN better reflected the behaviour of the local region.  
For Function 4, the GP was clearly more accurate. It predicted 0.6499 compared with the actual new best of 0.6229, giving an error of only 0.0270. The NN underestimated the result with a larger error of 0.0889. This strengthens the case for continuing to give the GP considerable weight for F4.  
For Function 5, both models predicted the result well considering the much larger scale of its outputs. The NN was slightly closer, with an error of 28.87, compared with 41.22 for the GP. The agreement between the two models also supports the evidence that moving slightly away from the [1,1,1,1] boundary would reduce the output.  
For Function 6, the NN was closer, although both models overestimated the performance of the submitted point. The NN error was 0.0526, compared with 0.0741 for the GP. This suggests some continued value in the NN for F6, but the weaker-than-predicted actual result means the local region should not automatically be exploited further.  
For Function 7, the GP was more accurate, predicting 2.9806 against the new best of 3.0246, with an error of 0.0440. The NN was also relatively close at 0.0707, meaning both models provided useful evidence for the region that ultimately produced the improvement.  
For Function 8, the GP was substantially more accurate, with an error of only 0.0235 compared with 0.1246 for the NN. The NN again overestimated the likely output, while the GP closely represented the performance of the submitted point.  

Overall, the realised results support continuing with the function-specific weighting of the two models rather than choosing one model for all eight functions. The GP currently appears particularly useful for F4, F7 and F8, while the NN provided the closer predictions for F3, F5 and F6. F1 remains outside this comparison because its point was deliberately selected through model-independent exploration.  

**Step 3. Updated model diagnostics**

The Gaussian Process models were refitted using the updated datasets containing the latest Round 11 observations. The fitted RBF length scale and log marginal likelihood were then reviewed for each function, together with any optimisation warnings produced during fitting.  
The length scale provides an indication of how quickly the GP believes each function changes across the input space. Very small values can indicate that the model is struggling to identify a smooth relationship between nearby observations, which has previously been an issue for Functions 1 and 2. The log marginal likelihood is also recorded to provide a consistent model diagnostic across rounds, although it is not used alone to judge whether a model is reliable.  
This step allows the current GP behaviour to be compared with previous rounds before its predictions are used to generate new candidates. In particular, I want to see whether the additional observations have improved the previously problematic models and whether the GPs for the functions that have responded well to local refinement remain stable.  

Results:  
Functions 1 and 2 again fitted at the minimum permitted RBF length scale of 1e-05, with the same warning as the previous round. Adding the latest observations has therefore not resolved the GP issue for either function. The extremely small length scale suggests that the GP is still struggling to identify a useful smooth relationship between neighbouring observations, so its predictions should continue to be treated cautiously for these two functions.  
Function 3's length scale decreased from 0.0968 to 0.0642. The model is therefore representing the function as somewhat more locally variable following the latest observation. There is no optimisation warning, but the smaller length scale provides another reason not to rely too heavily on broad GP predictions for F3.  
Function 4 decreased slightly from 0.316 to 0.272. This remains a reasonable fitted length scale with no warning, and the GP also predicted the latest new best relatively accurately. Together, this continues to support the GP as useful evidence for F4, particularly around the promising local region.  
Function 5 increased slightly from 0.338 to 0.359 and remains stable with no warning. The GP continues to appear capable of modelling the strong boundary behaviour observed for this function.  
Function 6 decreased from 0.518 to 0.470, but remains well away from the kernel bounds with no warning. The GP therefore remains usable, although the latest realised prediction error suggests its recommendations should still be considered alongside the NN and neighbourhood evidence.  
Function 7 increased slightly from 0.320 to 0.338 and remains stable. This is encouraging given that the latest local refinement produced a new best and the GP predicted that result reasonably accurately.  
Function 8 is almost unchanged, moving from 1.07 to 1.08. It continues to have the largest fitted length scale and appears to be the smoothest of the functions from the GP's perspective. Combined with its strong realised prediction accuracy, the GP remains particularly useful for F8.  

Overall, the additional observations have not fundamentally changed the GP picture from the previous round. F1 and F2 remain the clear exceptions where the GP is unreliable, while F3–F8 have usable fitted models without optimisation warnings. This supports continuing with the function-specific approach rather than changing the GP methodology across all eight functions.  

**Updated NN LOOCV validation**

The neural-network surrogate was re-evaluated using Leave-One-Out Cross-Validation (LOOCV) after adding the latest observations. For each function, one observation is removed at a time, the NN is trained on the remaining data, and a prediction is made for the excluded observation. The Mean Absolute Error (MAE) across all excluded observations is then calculated.  
As in the previous round, this is compared with a simple baseline that predicts the mean of the remaining outputs. The NN is considered to add useful predictive information where its LOOCV MAE is lower than the baseline MAE. Repeating this validation after each new round is important because the reliability of the NN may change as more observations are collected and the datasets become increasingly concentrated around promising regions.  
This validation will not determine candidate selection by itself. Instead, it provides another measure of model reliability to combine with realised prediction accuracy, GP diagnostics, neighbourhood evidence and the clustering analysis introduced this week.  

Results:
Function 1 remains the clear exception. Although its NN LOOCV MAE improved from 0.0338 to 0.0278, this is still substantially worse than the baseline MAE of only 0.0009. The NN therefore continues to provide little useful predictive information for F1, supporting the decision to keep relying on alternative exploration methods.  
Function 2 still beats the baseline, although only narrowly. Its NN MAE increased from 0.1933 to 0.2369, compared with a baseline of 0.2392. Combined with the reasonably accurate prediction of the latest result, the NN remains useful for F2, but the small margin over the baseline suggests it should not be relied upon too heavily.  
Function 3 shows the most important change. Its NN MAE improved from 0.0769 to 0.0460 and it now beats the baseline for the first time in the latest validation. This supports the latest realised result, where the NN was also more accurate than the GP. The NN can therefore be given more consideration for F3 this round, although this improvement is still based on a relatively small dataset.  
Functions 4 and 5 continue to show clear evidence that the NN is learning useful structure. F4's MAE of 1.4783 remains far below its baseline of 7.7991, while F5 improved substantially from 197.32 to 160.03, compared with a much larger baseline error of 2752.03.  
Function 6 continues to beat its baseline, although its NN MAE increased from 0.1611 to 0.2398. This reduction in validation performance, together with the latest local refinement failing to improve the best result, suggests some caution is appropriate when using the NN for F6 this week.  
Function 7 also continues to beat the baseline, although its MAE increased from 0.3494 to 0.4733. The NN therefore remains useful, but the GP was more accurate on the latest submitted point that produced the new best.  
Function 8 improved noticeably, with its NN MAE falling from 0.5201 to 0.3672, while the baseline remains much higher at 1.0152. The NN therefore continues to capture useful structure, even though the GP was substantially more accurate for the latest submitted point.  

Overall, the NN now beats the baseline for seven of the eight functions, compared with six last round. However, the results reinforce the importance of not treating this as a simple pass/fail test. F2 only narrowly beats its baseline, while F4, F5 and F8 show much clearer improvements over it. These validation results will therefore be considered alongside realised prediction accuracy and the GP diagnostics rather than being used alone.  

**Step 4. Analyse recent optimisation progress**

Before applying clustering, I reviewed the progression of the best observed output for each function. The purpose is to understand whether each optimisation process is still producing meaningful improvements or whether it is beginning to plateau.  
For each function, the running best is calculated across the observations in the order they were collected. I then compare the current best with the previous improvements and record how many observations have passed since the last new best. This provides a simple measure of optimisation progress without relying on either surrogate model.  
This is particularly useful for deciding how clustering evidence should later be interpreted. If recent improvements are occurring within a consistent area of the search space, this would support further exploitation of that region. In contrast, if the best value has remained unchanged despite several nearby searches, this may suggest that the current region is becoming exhausted and that wider exploration should receive more weight.  
The analysis will therefore be used as an additional diagnostic rather than as a fixed rule for candidate selection.  

Results:
Function 1 shows the clearest plateau. Its best result was found at observation 2, with 17 subsequent observations failing to improve it. There has also been no improvement within the recent window. This reinforces the evidence that neither repeated exploration nor the current surrogate models have identified a reliable route towards improvement.  
Function 2 also appears to be plateauing. Its current best of 0.746962 was found five observations ago, with no improvement in the recent window. However, the latest result of 0.666153 was still relatively strong, so the promising region should not necessarily be abandoned. Clustering may help establish whether the best point belongs to a broader high-performing region.  
Function 3 has shown relatively frequent improvement across the optimisation, with seven new bests. Its current best was found only two observations ago, and the recent improvement remains positive. However, the two subsequent local searches have failed to improve it, suggesting that very tight refinement around the current best may now be producing diminishing returns.  
Function 4 shows the strongest evidence of continued progress. It has produced 12 improvements, including a new best from the latest observation. The recent improvement is also substantial. This strongly supports the existence of a productive region and makes F4 particularly interesting for the clustering analysis.  
Function 5 has also shown strong historical progress, with 10 improvements and a large increase in the best value over the recent window. However, its best was found three observations ago and subsequent moves away from [1,1,1,1] have performed worse. This increasingly suggests that the upper boundary itself is important rather than simply being part of a broad high-performing region.  
Function 6 remains relatively active. Its best was found only one observation ago and there has been a positive improvement over the recent window. The latest point failed to improve it, so the evidence supports retaining the promising region while avoiding excessively narrow exploitation until we better understand its structure.  
Function 7 continues to progress, with the latest observation producing a new best of 3.024610. Although it has recorded fewer total improvements than F4, several have occurred later in the optimisation. This suggests that the current local region remains productive and provides another strong candidate for identifying a meaningful high-performing cluster.  
Function 8 also remains relatively active. Its current best was found only one observation ago, with positive recent improvement. The latest point was slightly weaker but still strong, suggesting that the region remains promising even though the most recent refinement did not improve the incumbent.  
Overall, there is a useful distinction emerging. F4 and F7 are still actively improving; F3, F5, F6 and F8 have relatively recent bests but may be approaching local plateaus; F2 has gone longer without improvement; and F1 is clearly the most stagnant.  

**Step 5. Apply K-Means clustering**

K-Means clustering was introduced to investigate whether the accumulated observations form identifiable regions within each function's input space. This develops the neighbourhood analysis used in previous rounds by considering groups of observations rather than only the points immediately surrounding individual candidates.  
The clustering is applied to the input variables only, rather than including the output as a clustering feature. This allows the algorithm to identify regions of the search space independently of their performance. The outputs can then be compared afterwards to determine whether any of the identified regions are consistently stronger than others. This avoids effectively telling the clustering algorithm which observations are good before assessing the resulting groups.  
Rather than choosing the number of clusters arbitrarily, I tested values of k from 2 to 5 for each function. The silhouette score is used to compare these alternatives, with a higher score indicating that observations are generally more similar to their own cluster than to neighbouring clusters. The value of k with the highest silhouette score is retained as an initial choice.  
The selected number of clusters will not automatically be assumed to represent meaningful optimisation structure. In the next stage, the outputs within each cluster will be examined to determine whether the clustering actually separates high- and low-performing regions. This is particularly important because K-Means will divide the observations into clusters even when the underlying structure is weak.  

Results:
The clustering strength varies considerably between the functions. Function 2 shows the clearest cluster structure, selecting three clusters with a silhouette score of 0.6462. Function 5 also shows reasonably clear separation, selecting two clusters with a score of 0.4331, while Function 1 selects four clusters with a moderate score of 0.4066.  
Functions 3, 4 and 7 show weaker separation, with their best silhouette scores around 0.30. This suggests that K-Means can divide the observations into groups, but those groups are not particularly strongly separated in the input space.  
Functions 6 and particularly 8 show the weakest clustering structure. F6's best score is 0.2572, while F8 reaches only 0.1853. The F8 result is especially important because it suggests that forcing its 50 observations into clusters may not provide particularly meaningful information.  

Overall, the results support exactly the cautious approach we planned: clustering appears potentially informative for some functions, especially F2 and F5, but should not automatically be given equal weight across all eight functions.  

**Step 6. Analyse cluster performance**

Having identified the clustering structure for each function, I next compared the performance of the resulting clusters. For each cluster, I recorded its size, mean output, best output and centroid, and identified the cluster containing the current best observation. This allowed me to assess whether the strongest individual observations were supported by a broader high-performing region.  
The results were considered alongside the silhouette scores from the previous step. Where cluster separation was relatively strong and one cluster also showed better output performance, I placed greater confidence in the clustering evidence. Where separation was weak, the results were treated more cautiously and used only as supporting evidence alongside the existing GP, NN and neighbourhood analysis.    

Function 1  
- Selected k: 4  
- Silhouette score: 0.4066  
- Best observation: 2  
- Best cluster: 0  
- Cluster mean: -0.000515  
- Cluster best output: 0.000000  
- Interpretation: Moderate cluster separation, but no clearly superior region is identified. Clustering adds limited new information for F1.  

Function 2  
- Selected k: 3  
- Silhouette score: 0.6462  
- Best observation: 14  
- Best cluster: 2  
- Cluster mean: 0.591846  
- Cluster best output: 0.746962  
- Interpretation: Strong cluster separation and the best observation lies within the strongest-performing cluster. This provides good evidence for continuing to refine this region.  
  
Function 3  
- Selected k: 5  
- Silhouette score: 0.3073  
- Best observation: 22  
- Best cluster: 2  
- Cluster mean: -0.046103  
- Cluster best output: -0.000863  
- Interpretation: The best observation is within the strongest cluster, supporting the current search region, although the relatively weak separation means clustering should remain secondary evidence.  
  
Function 4  
- Selected k: 5  
- Silhouette score: 0.3068  
- Best observation: 39  
- Best cluster: 0  
- Cluster mean: -5.322216  
- Cluster best output: 0.622950  
- Interpretation: Cluster separation is fairly weak, but the best cluster performs substantially better than the alternatives. This supports continued local refinement around the current high-performing region.  
  
Function 5  
- Selected k: 2  
- Silhouette score: 0.4331  
- Best observation: 26  
- Best cluster: 1  
- Cluster mean: 5678.018467  
- Cluster best output: 8662.482500  
- Interpretation: Clear performance separation between the two clusters. The strongest cluster is concentrated towards high input values, strongly supporting the existing boundary-focused strategy.  
  
Function 6  
- Selected k: 4  
- Silhouette score: 0.2572  
- Best observation: 28  
- Best cluster: 2  
- Cluster mean: -0.596464  
- Cluster best output: -0.239769  
- Interpretation: The best observation lies within the strongest-performing cluster, but weak cluster separation means this should mainly be used as supporting evidence.  
  
Function 7  
- Selected k: 5  
- Silhouette score: 0.3002  
- Best observation: 39  
- Best cluster: 1  
- Cluster mean: 1.983345  
- Cluster best output: 3.024610  
- Interpretation: Despite relatively weak cluster separation, the best cluster performs substantially better than all others. This strongly supports the promising region identified in recent rounds.  
  
Function 8  
- Selected k: 4  
- Silhouette score: 0.1853  
- Best observation: 48  
- Best cluster: 0  
- Cluster mean: 9.669458  
- Cluster best output: 9.984981  
- Interpretation: The best observation lies within the strongest-performing cluster, but the very low silhouette score means clustering should have limited influence on the next selection.  

Overall Interpretation
- The clustering analysis provides useful additional evidence that several of the high-performing observations found in previous rounds are part of broader promising regions rather than isolated good results.  
- F2 and F5 provide the strongest clustering evidence. Both have relatively good silhouette scores and clear differences in output between their clusters, giving greater confidence in their identified high-performing regions.  
- F4 and F7 also show useful patterns. Their silhouette scores are weaker, but the clusters containing the current best observations perform substantially better than the alternatives.  
- F3 and F6 provide supporting evidence for regions already identified through the GP, NN and local-refinement methods, although their weaker cluster separation means K-Means should not be relied upon independently.
- F1 provides limited additional information, as several clusters produce outputs effectively equal to zero and no clearly superior region emerges.  
- F8 should be treated most cautiously, as its silhouette score of 0.1853 indicates weak separation despite the best-performing observations being concentrated in one cluster.  
- Overall, the results suggest that clustering should be used as another source of evidence alongside the GP, NN, neighbourhood analysis and previous optimisation results, rather than replacing those methods.  
- For the next candidate-selection stage, the influence of clustering can therefore be function-specific: stronger for F2 and F5, useful supporting evidence for F3, F4, F6 and F7, and relatively limited for F1 and F8.

How the clustering results will be used:
The clustering results will be incorporated into the existing function-specific optimisation strategy rather than used as a standalone method. Where the clustering evidence is relatively strong, particularly for Functions 2 and 5, the identified high-performing regions will be given greater consideration when generating the next candidates. For Functions 3, 4, 6 and 7, the clusters provide additional support for promising regions already identified through the GP, NN and neighbourhood analysis, but will be considered alongside these existing methods. Clustering will have less influence on Functions 1 and 8, where the results provide weaker evidence of clearly separated regions.  
The cluster centroids will not be treated as optimal points themselves. Instead, the clusters will help identify areas of the input space in which candidate searches can be focused. This allows the new clustering analysis to contribute to candidate selection while retaining the exploration, local refinement and model-comparison methods developed in previous rounds.  

**Step 7. Cluster-guided candidate generation**

The clustering results were incorporated into candidate generation by using the strongest-performing clusters as an additional indication of promising areas within each function's search space. Rather than treating cluster centroids as candidate points directly, the clusters were used to define regions in which further optimisation could be investigated.  
The influence of clustering was varied by function according to the strength of the evidence identified in the previous step. For functions such as F2 and F5, where relatively clear clusters corresponded with substantial differences in output performance, cluster membership was given greater weight. For functions with weaker silhouette scores, clustering was treated as supporting evidence alongside the existing GP, NN and neighbourhood results.  
Candidate points within promising regions can then be evaluated using the established models. GP predictions and uncertainty provide the existing exploration–exploitation evidence, while NN predictions are considered where validation suggests that the model is useful. Distance from previous observations is also retained to avoid repeatedly selecting almost identical points.  
This approach allows the new unsupervised-learning method to contribute to the optimisation without replacing the function-specific strategy developed over previous rounds.  

Cluster-Guided Candidate Results:

Function 2  
- Candidate: [0.729306, 0.186865]  
- GP prediction: 0.365797  
- NN prediction: 0.661354  
- Nearest distance: 0.027213  
- Interpretation: The NN gives a relatively strong prediction within the best-performing cluster, while the GP remains much less optimistic and highly uncertain. Given the continuing GP length-scale problem for F2, the NN and strong clustering evidence are more informative here. However, the candidate is fairly close to an existing observation.  

Function 3  
- Candidate: [0.428667, 0.516401, 0.455178]  
- GP prediction: 0.037513  
- NN prediction: -0.012296  
- Nearest distance: 0.039857  
- Interpretation: The GP predicts a substantial improvement over the current best of -0.000863, but the NN does not support this. This disagreement makes the candidate interesting but relatively uncertain and suggests it should be compared with a more conservative local alternative.  

Function 4  
- Candidate: [0.397404, 0.357252, 0.361581, 0.434077]  
- GP prediction: 0.722666  
- NN prediction: 0.412112  
- Nearest distance: 0.021532  
- Interpretation: The GP predicts another improvement over the new best of 0.622950, while the NN is considerably less optimistic. The point lies close to the productive region already identified, so it provides a plausible exploitation candidate, although its low novelty should be considered.  

Function 5  
- Candidate: [0.989719, 0.999174, 0.994916, 0.997164]  
- GP prediction: 8321.19  
- NN prediction: 8326.64  
- Nearest distance: 0.011843  
- Interpretation: Excellent GP–NN agreement and the candidate lies exactly where the clustering analysis suggested performance should be strongest. However, both models predict an output below the existing best of 8662.48 and the point is extremely close to previously sampled observations. This supports the boundary hypothesis but does not yet provide a compelling next query.  

Function 6  
- Candidate: [0.410753, 0.388161, 0.491811, 0.803802, 0.010713]  
- GP prediction: -0.228475  
- NN prediction: -0.216792  
- Nearest distance: 0.050226  
- Interpretation: This is one of the more convincing candidates. Both models agree closely and both predict an improvement over the current best of -0.239769. It also maintains reasonable novelty while remaining within the strongest-performing cluster.  

Function 7  
- Candidate: [0.152047, 0.265662, 0.514461, 0.311439, 0.322715, 0.733203]  
- GP prediction: 3.087322  
- NN prediction: 3.020521  
- Nearest distance: 0.110278  
- Interpretation: Particularly promising. Both models broadly agree, the GP predicts an improvement over the current best of 3.024610, and the candidate has good novelty. This provides strong support for further refinement within the high-performing cluster.  

Function 8  
- Candidate: [0.046424, 0.134033, 0.255486, 0.042918, 0.790873, 0.568371, 0.062756, 0.628240]  
- GP prediction: 9.933104  
- NN prediction: 10.252357  
- Nearest distance: 0.196623  
- Interpretation: The NN predicts an improvement over the current best of 9.984981, whereas the GP predicts a slightly weaker result. The candidate has substantial novelty, which is useful in eight dimensions, but the weak clustering structure for F8 means this should remain an alternative rather than becoming the preferred candidate solely because of the NN prediction.  

Overall Interpretation  
The cluster-guided search has produced several useful candidates, but the evidence varies considerably between functions. F6 and F7 stand out because the GP and NN broadly agree on strong candidates that also retain reasonable distance from previous observations. F2 is supported by the NN and particularly strong clustering evidence, although its GP remains unreliable.  
F3 and F4 produce potentially strong candidates according to the GP, but disagreement with the NN suggests that further comparison is worthwhile before selecting them. F5 provides further confirmation of the strong upper-boundary region, but the candidate is extremely close to existing observations and is predicted to remain below the current best. F8 offers considerably greater novelty and a strong NN prediction, although its weak cluster separation means clustering itself provides limited additional confidence.  
Overall, these results support using the cluster-guided candidates as additional alternatives rather than automatically selecting them for submission, as planned.  

**Step 8. Function-specific candidate refinement**

Following the cluster-guided candidate generation, additional candidate alternatives were produced using a function-specific strategy. This reflects the fact that the eight functions now show substantially different behaviour and that no single optimisation method has performed consistently best across all of them.  
The strategy for each function was based on the accumulated evidence from previous rounds, including GP diagnostics, NN validation, realised prediction accuracy, optimisation progress, neighbourhood behaviour and the new clustering results. Functions with reliable surrogate models and clearly identified high-performing regions were refined locally, while functions with weaker models or evidence of stagnation retained a greater emphasis on exploration.  
The relative influence of the models was also adjusted where there was evidence to support doing so. For F2, the GP was given little weight in candidate selection because its RBF length scale remained at the lower bound of 1e-05, while the NN continued to pass LOOCV validation and the clustering analysis identified relatively strong regional structure. F2 was therefore refined using the NN within its strongest-performing cluster.  
For F4 and F8, the GP was given greater weight because its predictions were more accurate than the NN when compared with the realised Round 11 outputs. For F4 and F8, greater weight was given to the GP because it was more accurate than the NN on the latest realised outputs. The GP received particularly strong weighting for F8 because the difference in prediction error between the two models was larger. The remaining locally refined functions retained a balanced GP–NN weighting where there was not sufficient evidence to favour one model strongly over the other.  
A minimum distance from previous observations was also retained to avoid repeatedly sampling almost identical points. This was particularly relevant for F5, where the cluster-guided candidate was only 0.011843 from an existing observation. The boundary-refinement search therefore required a minimum nearest-observation distance of 0.03, allowing the search to remain focused on the strongly performing upper-boundary region while providing greater novelty.  
The purpose of this stage was not to automatically select the candidate with the highest predicted output. Instead, the function-specific candidates were generated as alternatives to the cluster-guided candidates so that predicted performance, model agreement, uncertainty, novelty and regional evidence could be compared before the final submission points were chosen.  

The following strategies were used:  
- F1: model-independent exploration due to continued GP failure, poor NN validation and long-term stagnation.  
- F2: NN-supported refinement within the strongly identified high-performing cluster, with the GP given little influence.  
- F3: local GP/NN refinement around the current best and high-performing cluster.  
- F4: local refinement around the latest new best, with greater emphasis on the GP following its recent predictive accuracy.  
- F5: boundary refinement around [1,1,1,1], while enforcing greater novelty than the cluster-generated candidate.  
- F6: GP–NN local refinement within the identified high-performing region.  
- F7: GP–NN local refinement around the new best and strongest-performing cluster.  
- F8: GP-led refinement around the current high-performing region, with clustering receiving relatively little weight due to its weak silhouette score.

Results:

Function 1  
- Strategy: Model-independent exploration  
- Current best: 0.000000  
- Candidate: [0.249506, 0.999107]  
- Nearest distance: 0.246275  
- Interpretation: The candidate provides substantial novelty in an unexplored part of the search space. With the GP remaining unreliable, the NN failing validation and no recent improvement, continued model-independent exploration remains the most appropriate strategy.  
  
Function 2  
- Strategy: NN-supported cluster refinement  
- Current best: 0.746962  
- Candidate: [0.665824, 0.186906]  
- NN prediction: 0.619287  
- Nearest distance: 0.036269  
- Interpretation: The candidate remains within the strongest-performing cluster while providing slightly greater novelty than the earlier cluster-guided candidate. Although the predicted output is below the current best, the strong clustering evidence and validated NN continue to support refinement within this region.  

Function 3  
- Strategy: Local GP–NN refinement  
- Current best: -0.000863  
- Candidate: [0.428706, 0.513778, 0.457382]  
- GP prediction: 0.037818  
- NN prediction: -0.012456  
- GP uncertainty: 0.020776  
- Nearest distance: 0.038875  
- Interpretation: The GP predicts a meaningful improvement over the current best, while the NN remains more cautious. This disagreement means the candidate is less certain, although both the clustering and previous local analysis support continued investigation of this region.  

Function 4  
- Strategy: GP-weighted local refinement  
- Current best: 0.622950   
- Candidate: [0.400697, 0.362771, 0.375829, 0.441739]  
- GP prediction: 0.744666  
- NN prediction: 0.352589  
- GP uncertainty: 0.118732  
- Nearest distance: 0.035225  
- Interpretation: The GP predicts another improvement over the latest best and the candidate provides greater novelty than the cluster-guided alternative. The NN is considerably more cautious, but the GP has recently been the more accurate model for F4 and the candidate remains within the strongly performing region.  

Function 5  
- Strategy: Novel boundary refinement  
- Current best: 8662.482500  
- Candidate: [0.967622, 0.993026, 0.997140, 0.999716]  
- GP prediction: 7899.378880  
- NN prediction: 7935.631782  
- GP uncertainty: 6.712195  
- Nearest distance: 0.033245  
- Interpretation: Both models agree that this remains a very high-performing region, but neither predicts an improvement over the existing best at [1,1,1,1]. The candidate provides more novelty than the earlier cluster candidate while retaining the boundary-focused strategy. The evidence increasingly suggests that the optimum may lie directly at or extremely close to the upper boundary.  

Function 6  
- Strategy: Local GP–NN refinement  
- Current best: -0.239769  
- Candidate: [0.400874, 0.395758, 0.500319, 0.800539, 0.004733]  
- GP prediction: -0.228270  
- NN prediction: -0.211156  
- GP uncertainty: 0.015295  
- Nearest distance: 0.053264  
- Interpretation: Both models predict an improvement over the current best and show relatively close agreement. The candidate also maintains reasonable novelty while remaining within the high-performing region identified by the clustering analysis. This is one of the stronger candidates produced in this stage.  

Function 7  
- Strategy: Local GP–NN refinement  
- Current best: 3.024610  
- Candidate: [0.176017, 0.276563, 0.482241, 0.296795, 0.315358, 0.706009]  
- GP prediction: 3.128395  
- NN prediction: 3.071368  
- GP uncertainty: 0.095238  
- Nearest distance: 0.082579  
- Interpretation: Both models predict an improvement over the new current best and show good agreement. The candidate also retains reasonable novelty and remains within the high-performing region identified through clustering and previous local refinement. This provides particularly strong evidence for continued exploitation of this region.  

Function 8  
- Strategy: GP-weighted local refinement  
- Current best: 9.984981  
- Candidate: [0.077042, 0.136953, 0.185881, 0.169530, 0.768770, 0.510795, 0.209002, 0.691691]  
- GP prediction: 9.998146  
- NN prediction: 9.986669  
- GP uncertainty: 0.034109  
- Nearest distance: 0.109510  
- Interpretation: Both models give predictions close to or slightly above the current best and show much stronger agreement than for the earlier cluster-guided candidate. Given the GP's stronger recent accuracy and the weak clustering evidence for F8, this candidate provides a more convincing balance of predicted performance and novelty.  

Overall Interpretation  
- The function-specific refinement has produced particularly promising candidates for F6, F7 and F8, where model predictions are relatively consistent and the candidates retain useful novelty.  
- F4 also remains promising, although there is considerably greater disagreement between the GP and NN. Recent realised prediction accuracy provides a reason to give the GP greater weight.  
- F3 shows a similar GP–NN disagreement, so further comparison is warranted before selecting the final point.  
- F2 continues to support the same high-performing cluster identified previously, although neither the latest NN candidate nor the previous cluster candidate is predicted to exceed the current best.  
- F5 increasingly supports the conclusion that performance improves towards the extreme upper boundary. Both models predict the more novel candidate below the existing [1,1,1,1] result, so further movement away from the boundary may have limited value.  
- F1 remains fundamentally different from the other functions. With no reliable surrogate signal or recent improvement, the new point deliberately prioritises exploration rather than predicted performance.  
- Overall, the results reinforce the value of a function-specific approach. The new clustering analysis has added useful regional information, but the strongest candidates emerge when that information is combined with the GP, NN, realised model accuracy, novelty and previous optimisation behaviour.  
 
**Step 9. Final Candidate Comparison**

The final candidate comparison combined the evidence generated throughout the current and previous rounds rather than selecting points solely according to their predicted output. For each function, the cluster-guided and function-specific candidates were compared using model predictions, model agreement, uncertainty, distance from previous observations, clustering evidence and the historical behaviour of the function.  
Greater weight was placed on models that had demonstrated stronger validation or realised prediction accuracy for the individual function. Novelty was also considered to ensure that additional queries provided useful new information rather than repeatedly sampling almost identical locations. Where the available evidence remained conflicting, the more conservative candidate was preferred or further targeted refinement was considered before final selection.  
This allowed the new clustering evidence to be incorporated without changing the function-specific approach developed over previous rounds.  

Function 1  
- Current best: 0.000000  
- Refined candidate: [0.249506, 0.999107]  
- Nearest distance: 0.246275  
- Preferred approach: Refined candidate  
- Reason: There is still no reliable GP or NN signal and clustering did not identify a clearly superior region. The candidate therefore deliberately targets a substantially unexplored part of the space. This is consistent with the model-independent exploration strategy already established for F1.  

Function 2  
Cluster candidate  
- Candidate: [0.729306, 0.186865]  
- NN prediction: 0.661354  
- Nearest distance: 0.027213  
Refined candidate  
- Candidate: [0.665824, 0.186906]  
- NN prediction: 0.619287  
- Nearest distance: 0.036269  
Current best: 0.746962  
- Initial preference: Cluster candidate  
- Reason: Neither candidate is predicted to improve on the current best, but the cluster candidate has the stronger NN prediction. Its slightly lower novelty is still sufficient to provide a distinct query. The GP should not influence this comparison because its fitted length scale remains unreliable.  

Function 3  
Cluster candidate  
- Candidate: [0.428667, 0.516401, 0.455178]  
- GP prediction: 0.037513  
- NN prediction: -0.012296  
- Nearest distance: 0.039857  
Refined candidate  
- Candidate: [0.428706, 0.513778, 0.457382]  
- GP prediction: 0.037818  
- NN prediction: -0.012456  
- Nearest distance: 0.038875  
Current best: -0.000863  
- Initial preference: Refined candidate, but only marginally  
- Reason: The two candidates are extremely similar and provide almost identical evidence. The refined point has the slightly stronger GP prediction, while the cluster point has marginally stronger NN and novelty evidence. There is effectively no meaningful strategic difference between them, so I would not spend another stage optimising between these two.  

Function 4  
Cluster candidate  
- Candidate: [0.397404, 0.357252, 0.361581, 0.434077]  
- GP prediction: 0.722666  
- NN prediction: 0.412112  
- Nearest distance: 0.021532  
Refined candidate  
- Candidate: [0.400697, 0.362771, 0.375829, 0.441739]  
- GP prediction: 0.744666  
- NN prediction: 0.352589  
- Nearest distance: 0.035225  
Current best: 0.622950  
- Initial preference: Refined candidate  
- Reason: The refined candidate has the stronger GP prediction and substantially greater novelty. Although its NN prediction is weaker, the GP was considerably more accurate on the realised Round 11 result and was deliberately given greater influence for this function.  

Function 5  
Cluster candidate  
- Candidate: [0.989719, 0.999174, 0.994916, 0.997164]  
- GP prediction: 8321.19  
- NN prediction: 8326.64  
- Nearest distance: 0.011843  
Refined candidate  
- Candidate: [0.967622, 0.993026, 0.997140, 0.999716]  
- GP prediction: 7899.38  
- NN prediction: 7935.63  
- Nearest distance: 0.033245  
Current best: 8662.48 at [1,1,1,1]  
- Initial preference: Cluster candidate if choosing strictly between these two.  
- Reason: It has considerably stronger predicted performance and remains extremely close to the known optimum region. However, its novelty is very low and neither model expects it to beat [1,1,1,1]. F5 is one function where I think one final targeted check is justified before submission, because we may be able to find a better compromise between the two candidates: closer to the boundary than the refined candidate but with more novelty than 0.0118.  

Function 6  
Cluster candidate  
- Candidate: [0.410753, 0.388161, 0.491811, 0.803802, 0.010713]  
- GP prediction: -0.228475  
- NN prediction: -0.216792  
- Nearest distance: 0.050226  
Refined candidate  
- Candidate: [0.400874, 0.395758, 0.500319, 0.800539, 0.004733]  
- GP prediction: -0.228270  
- NN prediction: -0.211156  
- Nearest distance: 0.053264  
Current best: -0.239769  
- Initial preference: Refined candidate  
- Reason: This is a fairly clear comparison. Both models predict an improvement, both slightly favour the refined point, and it also has marginally greater novelty.  

Function 7  
Cluster candidate  
- Candidate: [0.152047, 0.265662, 0.514461, 0.311439, 0.322715, 0.733203]  
- GP prediction: 3.087322  
- NN prediction: 3.020521  
- Nearest distance: 0.110278  
Refined candidate  
- Candidate: [0.176017, 0.276563, 0.482241, 0.296795, 0.315358, 0.706009]  
- GP prediction: 3.128395  
- NN prediction: 3.071368  
- Nearest distance: 0.082579  
Current best: 3.024610  
- Initial preference: Refined candidate  
- Reason: Both GP and NN predict an improvement over the current best and both favour the refined candidate. Although the cluster candidate has greater novelty, 0.0826 is already a substantial distance from the existing observations.  

Function 8  
Cluster candidate  
- Candidate: [0.046424, 0.134033, 0.255486, 0.042918, 0.790873, 0.568371, 0.062756, 0.628240]  
- GP prediction: 9.933104  
- NN prediction: 10.252357  
- Nearest distance: 0.196623  
Refined candidate  
- Candidate: [0.077042, 0.136953, 0.185881, 0.169530, 0.768770, 0.510795, 0.209002, 0.691691]  
- GP prediction: 9.998146  
- NN prediction: 9.986669  
- Nearest distance: 0.109510  
Current best: 9.984981  
- Initial preference: Refined candidate  
- Reason: The refined candidate has much stronger model agreement and the GP predicts a slight improvement over the current best. This is more persuasive than the very optimistic NN prediction for the cluster candidate, particularly because the GP substantially outperformed the NN on the latest realised result and F8's clustering structure was weak.  

**Step 10. F5 Targeted Boundary Check**

F5 was given one final targeted refinement because both the clustering and previous optimisation results provide strong evidence that performance increases towards the upper boundary of the search space. The current best remains [1,1,1,1] with an output of 8662.4825, while both the cluster-guided and more novel candidates were predicted to perform below this value.  
Rather than repeating either search, an intermediate novelty range was examined to determine whether a better balance could be found between remaining close to the high-performing boundary and obtaining useful new information. Candidates were therefore generated close to [1,1,1,1] and required to be between 0.02 and 0.04 from the nearest existing observation.  
The candidates were evaluated using both the GP and NN models. As both models have performed well for F5 and the NN substantially outperforms its baseline under LOOCV, their standardised predictions were weighted equally. The resulting candidate can then be compared with the previous cluster-guided and novelty-constrained alternatives before the final F5 point is selected.  

Current best: 8662.482500 at [1.000000, 1.000000, 1.000000, 1.000000]  
Targeted candidate: [0.981966, 0.999701, 0.990442, 0.998795]  
GP prediction: 8141.386269  
NN prediction: 8155.828807  
GP uncertainty: 4.961240  
Nearest distance: 0.020448  
Interpretation: The targeted candidate provides an intermediate balance between predicted performance and novelty. Both models agree closely, but both still predict performance below the current best. Together with the previous alternatives, the results show a consistent pattern in which predicted performance decreases as candidates move further away from [1,1,1,1]. This provides further evidence that the best-performing region identified so far appears to lie at or very close to the upper boundary.  

**Step 11. Final Candidate Validation**

Before submission, the eight selected candidates were subjected to a final validation check. Each point was checked to ensure that it contained the correct number of dimensions for its function, all values remained within the permitted [0,1] range, and the candidate did not duplicate an existing observation.  
The Euclidean distance to the nearest existing observation was also calculated. This provides a final measure of novelty and confirms how closely each candidate samples an area that has already been evaluated. This final stage does not alter the optimisation strategy, but acts as a safeguard against invalid or unnecessarily duplicated submissions.  

Final Validation Results  
- All eight candidates have the correct number of dimensions.  
- All values are within the permitted [0,1] range.  
- No duplicate observations were identified.  
- Nearest-observation distances range from 0.020448 for F5 to 0.246275 for F1.  
- The relatively small distance for F5 is intentional, reflecting the strong evidence that performance increases towards [1,1,1,1].  
- F1 has the greatest novelty, consistent with the model-independent exploration strategy.  
- F6, F7 and F8 retain increasingly substantial novelty of 0.053264, 0.082578 and 0.109510, respectively, despite remaining focused on their identified high-performing regions.  
- Overall validation: PASSED. The eight candidates can therefore be carried forward for submission.  

**Module 23 (Week 12)**

**Methodology**

This week builds on the increasingly function-specific approach developed over previous rounds while introducing Principal Component Analysis (PCA) as an additional way of understanding the accumulated observations. Last week's K-Means analysis helped identify whether high-performing observations were concentrated within particular regions of the input space. PCA will now be used to investigate whether the variation within those observations can be explained by a smaller number of underlying directions, which may be particularly useful for the higher-dimensional functions.  
PCA will be applied separately to each function rather than combining the eight datasets. The amount of variation explained by each principal component will be reviewed alongside the component loadings to understand which combinations of input dimensions contribute most strongly to the observed structure. The principal-component scores will also be compared with the corresponding outputs. This is important because PCA identifies variation in the inputs rather than directly identifying which variables produce high outputs. A direction that explains substantial input variation is therefore only particularly useful for optimisation if there is also evidence that movement along that direction is associated with changes in performance.  
The PCA results will then be compared with the K-Means and neighbourhood evidence from the previous round. Where PCA identifies a clear direction within an already promising cluster, this can provide additional evidence for how that region should be refined. Where the PCA structure is weak or has little relationship with output, it will remain a diagnostic rather than being allowed to influence candidate selection heavily.  
The existing GP and NN models will therefore remain the main predictive tools. Realised prediction accuracy, model diagnostics, recent optimisation progress, clustering, neighbourhood performance, PCA and candidate novelty will be considered together. As in previous rounds, the final strategy will remain function-specific rather than applying the same optimisation rule across all eight functions.  

Steps:
Step 1 – Incorporate and assess the latest results    
- Add the new observations to each function's dataset and compare the outputs with the previous best values. Identify which functions improved and where the latest candidate underperformed.
Step 2 – Review realised model prediction accuracy  
- Compare the new outputs with the GP and NN predictions recorded before submission. Use the prediction errors to update the evidence for which model is currently more reliable for each function.  
Step 3 – Update model diagnostics and validation  
- Refit the GP models using the expanded datasets and review their fitted kernels, length scales and any warnings. Repeat NN validation to check whether its performance remains useful following the latest observations.  
Step 4 – Review optimisation progress    
- Examine how recently each function has improved and how many observations have passed since its current best was found. Use this to identify functions where local refinement is still working and those where progress appears to be slowing.  
Step 5 – Apply PCA to each function    
- Apply PCA separately to the accumulated input observations for each function. Examine how much variance is explained by each principal component and how many components are required to explain most of the variation in the data.  
Step 6 – Examine PCA loadings and relationships with output    
- Review the principal-component loadings to understand which input dimensions contribute to each component. Compare the component scores with observed outputs to identify whether any PCA directions also show a meaningful relationship with performance.  
Step 7 – Examine high-performing observations in PCA space    
- Compare the positions of the highest-performing observations with the overall PCA distribution. Use this to determine whether strong outputs are concentrated towards a particular direction and whether the current best observation supports the same pattern.  
Step 8 – Generate and evaluate PCA-guided candidates    
- Generate candidates by moving from the current best observation in promising PCA directions where the evidence is sufficiently strong. Test several step sizes and compare the resulting candidates using GP predictions, GP uncertainty, NN predictions and distance from previous observations.  
Step 9 – Carry out function-specific candidate refinement    
- Refine the search separately for each function using the strongest available evidence. Retain model-independent exploration where the models remain unreliable, local refinement where recent progress supports it, boundary searches where appropriate, and PCA-guided refinement where PCA provides useful additional information.  
Step 10 – Compare model disagreement and select final candidates    
- Compare the most promising candidates using GP and NN predictions, PCA evidence, recent results and distance from previous observations. Where GP and NN predictions increasingly disagree, treat this as additional risk and consider intermediate candidates rather than automatically selecting the most aggressive option. Validate the final points for correct dimensions, input bounds and duplicates before submission.  

**Step 1. Incorporate and assess the latest results**

The Round 12 input and output values were first added to the existing datasets for all eight functions. The new output for each function was then compared with the best value observed before Round 12. This provides an immediate assessment of whether the latest function-specific strategy produced an improvement and identifies which functions should continue to be locally refined or may require a change in approach.  
This comparison is particularly useful this round because the submitted candidates incorporated the K-Means evidence introduced in the previous module alongside the existing GP, NN, neighbourhood and novelty evidence. The realised outputs therefore provide the first indication of whether candidates informed by these identified regions continued to improve the optimisation.  

Function 1:  
Previous best: 0.0000000000
New output:    0.0000000000
Difference:    -0.0000000000
New best:      False

Function 2:  
Previous best: 0.7469622512
New output:    0.5619276587
Difference:    -0.1850345924
New best:      False

Function 3:  
Previous best: -0.0008627356
New output:    -0.0003519729
Difference:    0.0005107628
New best:      True

Function 4:  
Previous best: 0.6229498177
New output:    0.2450446621
Difference:    -0.3779051556
New best:      False

Function 5:  
Previous best: 8662.4825000000
New output:    8119.4075501150
Difference:    -543.0749498850
New best:      False

Function 6:  
Previous best: -0.2397692962
New output:    -0.2629349273
Difference:    -0.0231656311
New best:      False

Function 7:  
Previous best: 3.0246097937
New output:    3.0700969378
Difference:    0.0454871441
New best:      True

Function 8:  
Previous best: 9.9849814249
New output:    9.9874191724
Difference:    0.0024377476
New best:      True

Round 12 produced three new best values: F3, F7 and F8. This is encouraging because all three came from local refinement of regions already supported by previous evidence. F3 and F8 produced relatively small improvements, suggesting they may be approaching strong local solutions, while F7 continues to show clearer progress. F4 demonstrates the opposite result, with a nearby candidate performing substantially worse than the previous best, suggesting that its high-performing region may be relatively narrow. F2 and F5 also failed to improve despite targeting previously promising regions, so their next selections should consider whether further local exploitation remains worthwhile.

**Step 2. Review realised model prediction accuracy**

The Round 12 realised outputs were compared with the GP and NN predictions recorded when the candidates were selected. Absolute prediction error was calculated for each available model. This provides an out-of-sample check of how accurately the models predicted the newly observed points and helps determine how much weight each model should receive during this week's candidate selection.  
The comparison is treated as supporting evidence rather than a definitive measure of model quality, as it is based on a single new observation for each function. It is particularly useful when combined with the longer-term validation and model diagnostics updated in the next step.  

Results:  
- F1: Not directly comparable because the candidate was selected using model-independent exploration rather than GP or NN predictions.  
- F2: The NN was more accurate, with an error of 0.0994 compared with 0.1961 for the GP. This supports the decision to give the NN greater weight for F2.  
- F3: The NN was more accurate, with an error of 0.0121 compared with 0.0382 for the GP. The NN also correctly predicted that the result would remain close to zero.  
- F4: The NN was considerably more accurate, with an error of 0.1075 compared with 0.4996 for the GP. The GP substantially overestimated the improvement available around this point.  
- F5: The GP was more accurate, with an error of 21.98 compared with 36.42 for the NN. Both models were reasonably close given the scale of the function.  
- F6: The GP was slightly more accurate, with an error of 0.0347 compared with 0.0518 for the NN. Both models predicted the result reasonably closely.  
- F7: The NN was extremely accurate, predicting 3.0714 against the realised 3.0701, an error of only 0.0013. This provides particularly strong recent evidence in favour of the NN around the current high-performing region.  
- F8: The NN was again extremely accurate, with an error of only 0.00075, compared with 0.0107 for the GP. Both predicted performance close to the realised new best, but the NN was substantially closer.  

The NN was closer for five of the seven directly comparable functions, while the GP was closer for F5 and F6. The strongest results were for F7 and F8, where the NN predictions were extremely close to the realised outputs. F4 is also important because the GP considerably overestimated the candidate while the NN gave a more cautious and more accurate prediction.  
These results suggest increasing the influence of the NN for F2, F3, F4, F7 and F8, while retaining stronger GP evidence for F5 and F6. However, these are still individual realised observations, so the results should be considered alongside the updated GP diagnostics and NN validation before changing the strategy.  

**Step 3. Update model diagnostics and validation**

The GP models were refitted using the expanded datasets, including the new Round 12 observations. The fitted RBF length scales, log marginal likelihoods and any optimisation warnings were recorded to check whether the behaviour of each GP had changed as more observations became available.  
NN performance was then reassessed using leave-one-out cross-validation (LOOCV). The NN MAE was compared with the same simple baseline used previously. This provides a broader measure of NN performance across the accumulated dataset and complements the realised prediction accuracy from Step 2, which only tests the latest observation.  

**Updated GP diagnostics**

Results:  
- F1: The length scale remains at the lower bound of 1e-05, with the same convergence warning. The additional observation has therefore not resolved the GP issue, and it remains unreliable for candidate selection.  
- F2: The same problem remains, with a length scale of 1e-05 and a lower-bound warning. This supports continuing to give the GP little weight for F2.  
- F3: The fitted length scale decreased slightly from 0.0642 to 0.0540. There are no warnings, although the relatively short length scale suggests the function continues to vary over fairly small distances.  
- F4: The length scale increased from 0.272 to 0.307 with no warning. The GP remains well behaved despite its poor realised prediction in Round 12, so it should not necessarily be discarded based on that single result.  
- F5: The GP remains very stable, with the length scale changing only from 0.359 to 0.360. This supports continuing to use the GP around the strong upper-boundary region.  
- F6: The length scale decreased slightly from 0.470 to 0.444, with no warnings. The model remains stable and was also the more accurate model for the latest realised observation.  
- F7: The length scale increased slightly from 0.338 to 0.347. There are no warnings, suggesting the GP remains useful alongside the increasingly strong NN evidence.  
- F8: The length scale remains comparatively large at 1.09, up slightly from 1.08. The model remains stable and suggests a relatively smooth response across the observed region.  

The additional observations have not substantially changed the GP behaviour. F1 and F2 remain the clear exceptions, with both models still reaching the minimum permitted length scale and producing convergence warnings. Their GP predictions should therefore continue to receive limited weight.  
The GPs for F3–F8 remain stable and warning-free. F5 and F8 show particularly little change from the previous round, while F4 remains technically well behaved despite substantially overpredicting its latest candidate. This reinforces the value of considering both model diagnostics and realised prediction accuracy rather than relying on either measure alone.  

**Updated NN validation**

The NN was reassessed using leave-one-out cross-validation on the expanded datasets. Each observation was removed in turn, the model was trained on the remaining observations, and a prediction was made for the excluded point. The resulting mean absolute error (MAE) was compared with the same baseline prediction used in previous rounds.  
This provides a broader assessment of NN reliability across all accumulated observations, rather than judging the model only on its latest Round 12 prediction. The results will be considered alongside the realised prediction accuracy and GP diagnostics when deciding how much weight to give each model.  

Results  
- F1: The NN still fails to beat the baseline, with an MAE of 0.00114 compared with 0.00086. Although this is a substantial improvement from last week's NN MAE of 0.0278, there is still no strong reason to rely on the NN for F1.  
- F2: NN performance improved considerably, with MAE falling from 0.2369 to 0.1463. It now beats the baseline by a clearer margin, supporting the strong recent evidence in favour of the NN for this function.  
- F3: NN MAE increased from 0.0460 to 0.0588 and now slightly exceeds the baseline of 0.0556. This contrasts with the latest realised result, where the NN was more accurate than the GP, so the evidence for F3 is now mixed rather than clearly favouring either model.  
- F4: NN validation improved from 1.4783 to 1.2513, remaining substantially better than the baseline of 7.9456. Combined with its much better Round 12 prediction, the NN should receive greater weight this round.  
- F5: NN MAE increased slightly from 160.03 to 169.15, but remains dramatically better than the baseline of 2917.87. The NN remains useful, although the GP was more accurate on the latest observation.  
- F6: NN MAE improved from 0.2398 to 0.2167 and remains comfortably below the baseline. Both models therefore remain useful, with the GP having been slightly more accurate in Round 12.  
- F7: NN MAE improved from 0.4733 to 0.4157 and remains well below the baseline. Together with the extremely accurate latest prediction, this strengthens the case for giving the NN substantial weight for F7.  
- F8: NN MAE improved from 0.3672 to 0.2743, substantially outperforming the baseline of 1.0295. Combined with its very accurate Round 12 prediction, the NN currently provides strong evidence for local refinement.  

The NN now beats the baseline for six of the eight functions. F2, F4, F6, F7 and F8 all showed improved LOOCV performance, while F5 remains strongly validated despite a small increase in error.  
The main exceptions are F1 and F3. F1 continues to lack a reliable predictive model, supporting the existing model-independent approach. F3 is more interesting: its latest NN prediction was relatively accurate, but its overall LOOCV performance has fallen slightly below the baseline. The NN should therefore not be given too much weight based on one successful prediction.  
Taken together with the GP diagnostics and realised prediction errors, we now have a clearer function-specific picture: NN evidence is particularly strong for F2, F4, F7 and F8; GP evidence remains useful for F5 and F6; F3 should use both cautiously; and F1 remains primarily model-independent.  

**Step 4. Review optimisation progress**

The optimisation history was reviewed for each function using the expanded datasets. The current best output, location of the best observation, number of observations since that best, and total number of improvements were recorded. This helps distinguish functions that are still responding to local refinement from those where progress has slowed or stalled.  
This is particularly useful before introducing PCA because the new technique should support the existing function-specific strategy rather than automatically changing it. Functions that continue to improve around established regions may justify further exploitation, while those showing little recent progress may benefit from wider exploration or a different search direction.  

Results:  
- F1: No meaningful improvement for 18 observations. The function remains effectively flat around its best observed value, reinforcing the need for exploration rather than local refinement.  
- F2: The best result has now stood for 6 observations. Although the identified region continues to produce relatively strong outputs, recent searches have not exceeded the best of 0.7470, suggesting some degree of plateau.  
- F3: New best at the latest observation, improving to -0.000352. This is the eighth improvement overall and shows that local refinement is still producing progress, although the latest gain is small.  
- F4: The current best of 0.62295 was found only one observation ago. The latest candidate performed worse, but the recent sequence of improvements suggests that the current region remains promising.  
- F5: No improvement for 4 observations, although its most recent record improvement was substantial. The strong upper-boundary region remains clear, but progress now appears to be slowing.  
- F6: The best result has stood for 2 observations. This is still relatively recent, so there is not yet strong evidence of a plateau.  
- F7: New best at the latest observation, reaching 3.07010. The function has now improved in several recent rounds, providing strong evidence that continued refinement of the current region is working.  
- F8: New best at the latest observation, reaching 9.98742. The improvement is very small, however, suggesting that the function may now be approaching a plateau within the current region.  

The optimisation histories support increasingly different strategies across the eight functions. F3, F7 and F8 are still producing new best values, making further refinement reasonable, particularly for F7 where the improvement pattern remains consistent.  
F4 and F6 have relatively recent best observations and therefore still justify investigation around their current regions, although F4's latest result shows that its strongest area may be quite narrow.  
F2 and F5 show clearer signs of slowing, with 6 and 4 observations respectively since their best results. This makes them particularly interesting for the PCA analysis: rather than simply continuing to search around the same points, we can investigate whether the accumulated observations contain a dominant direction of variation that could suggest a different way of moving through those promising regions.  
F1 remains the clearest case for exploration, with 18 observations since its best and neither the GP nor NN providing convincing guidance.  

**Current position before PCA**
Before introducing PCA, the latest results, model diagnostics and optimisation history suggest the following position for each function:  
- F1: Long-term plateau with unreliable GP and NN models. Wider, model-independent exploration remains the most appropriate approach.  
- F2: No improvement for six observations despite a previously identified promising region. The NN evidence is relatively strong, but PCA may help identify whether there are alternative directions worth investigating.  
- F3: New best in the latest round, although the improvement was small. Continued local refinement appears reasonable, but the GP and NN evidence remains mixed.  
- F4: The best was found recently, although the latest nearby candidate performed considerably worse. Local refinement remains appropriate but should be approached cautiously.  
- F5: A clear high-performing region remains close to the upper boundary, but there has been no improvement for four observations. PCA may help identify whether there are useful directions within this region that have not yet been explored.  
- F6: The current best was found relatively recently and both models remain useful. GP/NN-supported local refinement therefore remains appropriate.  
- F7: A new best was achieved in the latest round and recent local refinement continues to produce improvements. This remains one of the strongest cases for continued exploitation of the current region.  
- F8: A new best was also achieved, but the improvement was very small. Further fine local refinement is reasonable, although the function may be approaching a plateau.  
PCA will now be introduced as an additional source of evidence to investigate the main directions of variation within these accumulated observations, rather than replacing the function-specific strategies already developed.  

**Step 5. Apply PCA to each function**

PCA was applied separately to the accumulated input observations for each function. Although all inputs were already on the same 0–1 scale, standardisation was retained so that PCA focused on relationships between dimensions rather than allowing dimensions with greater sampled variance to dominate. The proportion of variance explained by each principal component was then calculated, together with the cumulative explained variance.  
The purpose of this stage was to determine whether the variation within each function's observations could be represented by a smaller number of underlying directions. The number of components required to explain approximately 80%, 90% and 95% of the observed variation was also recorded. At this stage, PCA was treated only as a description of the structure within the sampled inputs; a component explaining substantial input variation does not necessarily mean that it is important for achieving higher outputs.  

Results  
- F1: PC1 explains 52.33% of the variance and PC2 explains 47.67%. Both components are required to reach 80% explained variance, so PCA does not identify a clear lower-dimensional structure.  
- F2: PC1 explains 65.99% of the variance, showing a stronger main direction than F1. However, both components are still required to reach 80%, so neither dimension can simply be ignored.  
- F3: PC1 explains 57.77%, while the first two components together explain 84.20%. This suggests that most of the variation in the three-dimensional search space can be represented using two components.  
- F4: Variance is relatively spread across the four components. Three components are required to explain 86.34%, suggesting only limited scope for dimensionality reduction.  
- F5: PCA identifies a much clearer dominant direction, with PC1 alone explaining 61.55% of the variance. Three components explain 90.74%. This is potentially useful given the strong pattern already identified around the upper boundary.  
- F6: The variance is fairly evenly distributed, with four of the five components required to exceed 80%. PCA therefore provides relatively little dimensionality reduction.  
- F7: The first three components explain 72.89%, with four required to exceed 80%. There is some structure within the six-dimensional inputs, although there is not one dominant component.  
- F8: Despite having eight dimensions, the first five components explain 80.11% of the variance. However, seven are required to exceed 90%, suggesting that the observed inputs still contain substantial variation across several dimensions.  

PCA shows the clearest potential structure for F3 and F5, where a smaller number of components captures a relatively large proportion of the observed variation. F5 is particularly interesting because PC1 alone accounts for over 60%.  
For F4, F6, F7 and F8, variance is distributed more widely across the components. PCA may still help us understand relationships between dimensions, but there is not enough evidence to simply reduce these functions to one or two dimensions.  
For F1 and F2, the original problems are already only two-dimensional, so PCA is more useful for identifying the main direction in which the existing observations vary than for reducing dimensionality.  

**Step 6. Examine PCA loadings and relationship with output**

The PCA results were examined in more detail by looking at the loadings of each original input dimension on the principal components. This shows which dimensions contribute most strongly to each PCA direction.  
The PCA scores for each observation were then compared with the corresponding function outputs using simple correlations. This provides an indication of whether movement along any of the main PCA directions is associated with higher or lower performance. The correlations are treated as supporting evidence only, as PCA itself is unsupervised and the relationships between inputs and outputs may not be linear. The correlations were also interpreted cautiously given the relatively small and adaptively sampled datasets.  

For each function, the analysis therefore records:  
- the dimensions contributing most strongly to the main principal components;  
- the correlation between each component score and output;  
- which component has the strongest relationship with performance.  

Results  
- F1: Neither component shows a meaningful relationship with output. PC1 has the strongest correlation, but at only 0.038 this is effectively negligible. PCA therefore provides little useful optimisation guidance for F1.  
- F2: PC2 has the strongest relationship with output, with a correlation of -0.573. Its loadings are negative for Dimension 1 and positive for Dimension 2, suggesting that better outputs tend to occur towards the opposite direction: relatively higher Dimension 1 and lower Dimension 2. This fits reasonably well with the location of the current strong region.  
- F3: PC3 has the strongest relationship with output at -0.335, although this is only moderate and PC3 explains just 15.8% of input variance. PCA therefore provides some additional information, but not enough to drive candidate selection on its own.  
- F4: PC1 has a relatively strong negative relationship with output at -0.652. All four dimensions contribute positively to PC1, particularly Dimensions 1 and 2. Moving towards lower PC1 scores therefore appears to be associated with better performance. This may help explain why relatively small movements within the current region can have a noticeable effect on output.  
- F5: This is the clearest PCA result so far. PC1 explains 61.55% of input variance and has a very strong positive correlation with output of 0.906. All four dimensions load positively and fairly evenly onto PC1. This strongly supports the previous evidence that increasing the dimensions together towards the upper boundary is associated with substantially better performance.  
- F6: PC1 has a strong negative correlation with output of -0.741. The largest contributions come from Dimensions 5 and 4, in opposite directions. This suggests that the combination represented by PC1 may provide useful additional guidance when refining the current region.  
- F7: PC1 has a moderately strong negative correlation with output of -0.606. Dimensions 1 and 6 contribute particularly strongly but in opposite directions. Given that the latest local searches are already producing new bests, PCA can be used as supporting evidence rather than replacing the current strategy.  
- F8: This provides the strongest relationship of all eight functions. PC1 has a correlation of -0.918 with output. Dimensions 1, 3 and 7 contribute strongly and positively to PC1, while Dimension 5 contributes negatively. Lower PC1 scores are therefore very strongly associated with higher observed outputs. Although PC1 explains only 25.53% of total input variance, its relationship with performance makes it potentially useful for further refinement.  

The PCA analysis is most informative for F5 and F8. F5's first component both explains a large amount of input variation and is very strongly related to output, reinforcing the upper-boundary pattern already identified through previous methods. F8's PC1 explains less overall variance but has an even stronger relationship with output, suggesting that PCA has identified a combination of dimensions closely associated with performance.  
F4 and F6 also show reasonably strong relationships between PC1 and output, while F2 and F7 provide moderate additional evidence. PCA appears less useful for F1 and F3, where the relationships are weak or spread across less important components.  
These results also demonstrate why explained variance alone was not enough. For example, F8's PC1 explains only around a quarter of input variation, but has the strongest relationship with output of any component examined. PCA should therefore be used alongside the existing GP, NN, clustering and neighbourhood evidence rather than simply selecting the component explaining the most variance.  

**Step 7. Compare high-performing observations in PCA space**

The PCA scores of the highest-performing observations were compared with the rest of the dataset. For each function, the top 20% of observations by output were identified and their average principal-component scores were compared with the overall average and the current best observation.  
This helps determine whether the strongest observations are concentrated towards a particular part of PCA space. Particular attention was given to the components identified in the previous step as having the strongest relationship with output. If the best observations consistently appear towards one end of a component, this provides stronger evidence that the PCA direction may be useful for candidate refinement.  
PCA is still treated as supporting evidence rather than a direct optimisation rule. A relationship observed within the existing data does not necessarily mean that continuing indefinitely in the same PCA direction will improve performance.  

Results  
- F1: There is no clear PCA pattern. The top observations are centred close to the overall mean and the current best has a PC1 z-position of only -0.04. Combined with the negligible correlation from Step 6, PCA provides little useful guidance for F1.  
- F2: The current best has a PC2 z-position of -1.23, which is consistent with the negative relationship between PC2 and output. However, the top 20% cover quite a wide range of PC2 scores, so the evidence is useful but not especially strong.  
- F3: There is little separation between the strongest observations and the rest of the dataset. The top 20% mean is almost exactly at the centre of PC3, while the current best actually lies slightly in the opposite direction to that suggested by the correlation. PCA should therefore receive little weight for F3.  
- F4: The top 20% are consistently located towards lower PC1 scores, with a mean z-position of -0.45. The current best is further in the same direction at -0.55. This supports the negative PC1 relationship identified previously and provides useful additional evidence for local refinement.  
- F5: There is a very clear pattern. The top 20% are tightly concentrated at high PC1 scores, with an average z-position of +1.40, while the current best is at +1.45. This strongly confirms that the high-performing upper-boundary region corresponds to movement in the positive PC1 direction.  
- F6: The top observations are strongly concentrated towards lower PC1 scores, averaging -0.88 standard deviations from the centre. The current best is even further in this direction at -1.01. PCA therefore provides strong supporting evidence for the existing promising region.  
- F7: A similarly clear pattern is present. The top 20% have an average PC1 z-position of -0.90, while the current best is at -0.99. Combined with the latest new best, this suggests that the current local search is moving through a genuinely high-performing part of the input space.  
- F8: This provides the strongest PCA evidence. The top 20% are concentrated at very low PC1 scores, averaging -1.59 standard deviations from the centre. The current best is almost exactly in this region at -1.60. Together with the -0.918 output correlation, this provides strong evidence that PC1 captures an important performance direction.  

The combination of Steps 5–7 suggests that PCA should not be applied equally across all eight functions.  
There is strong evidence for F5, F6, F7 and F8, where the high-performing observations are clearly concentrated towards one end of PC1 and the current best points follow the same pattern. F4 also shows a useful, although less pronounced, relationship.  
For F2, PCA provides some supporting evidence but the top-performing region is less tightly concentrated. For F3, the evidence is weak and somewhat inconsistent, while F1 shows essentially no useful PCA structure related to performance.  

**Step 8. Generate PCA-guided candidates**

PCA-guided candidates were generated only for functions where the previous analysis showed a meaningful relationship between principal-component position and output performance. For these functions, candidate points were generated by making small movements from the current best observation in the PCA direction associated with higher outputs.  
The size of the movement was kept relatively small because the PCA relationships are based only on the observations collected so far and do not prove that performance will continue improving indefinitely in the same direction. All generated candidates were constrained to the original [0,1] input space.  
The candidates were then assessed using the existing GP and NN models where these had demonstrated useful predictive performance. Distance from previous observations was also calculated to avoid selecting points that provide almost no new information. PCA was therefore used as an additional candidate-generation method rather than as a replacement for the existing optimisation strategy.  
For F1 and F3, where the PCA evidence was weak, no PCA-guided candidate was generated.  

Results  
- F2: Moving towards lower PC2 produces sensible candidates, but the 0.10 step is only 0.0037 from an existing observation. The 0.20 and 0.30 steps provide more novelty at distances of 0.0263 and 0.0528 respectively. Given the moderate PCA evidence, these should be compared with an NN-led candidate rather than selected directly.  
- F4: All three candidates move progressively away from the current observations while following the favourable lower-PC1 direction. The 0.20 candidate provides a useful balance, with a nearest distance of 0.0510, while the 0.30 candidate offers greater exploration at 0.0765.  
- F5: All PCA movements return [1, 1, 1, 1], which is already an observed point. This is actually informative: PCA independently points towards the same upper boundary identified through the previous optimisation methods. However, it cannot provide a novel PCA candidate, so F5 should continue to use targeted boundary refinement instead.  
- F6: The PCA candidates provide a clear progression towards lower PC1. The 0.20 and 0.30 steps have useful distances of 0.0406 and 0.0610. PCA therefore provides a sensible alternative direction to compare with the existing GP/NN local search.  
- F7: The candidates move smoothly through the promising region identified by PCA. The 0.30 step has a nearest distance of 0.0834, making it sufficiently different from previous observations while remaining close to the current high-performing region.  
- F8: PCA produces particularly useful candidates. The 0.20 and 0.30 steps have distances of 0.0599 and 0.0898, while continuing in the strongly supported lower-PC1 direction. Given the very strong PCA-output relationship and the latest new best, these are strong candidates for further assessment.  

The candidate generation confirms that PCA is useful selectively rather than universally. F4, F6, F7 and F8 now have novel PCA-guided alternatives that remain reasonably close to established high-performing regions. F2 also produces usable candidates, although its PCA evidence is weaker.  
For F5, PCA strengthens the existing conclusion that performance improves towards the extreme upper boundary, but because the favourable direction is already constrained by [1,1,1,1], it cannot generate a new point. F1 and F3 remain deliberately excluded because their PCA evidence was not strong enough.  

**Step 9. Evaluate PCA-guided candidates**

The PCA-guided candidates were evaluated using the updated GP and NN models. Predicted output, GP uncertainty and distance from previous observations were compared across the three PCA step sizes.  
The model predictions were interpreted according to the function-specific evidence established earlier. The NN was given greater consideration for F2, F4, F7 and F8, while both models remained relevant for F6. For F5, no PCA candidate was assessed because movement in the favourable PCA direction returned the existing boundary point.  
The purpose was not simply to select the candidate with the highest predicted output, but to determine whether the existing predictive models support the direction identified independently by PCA.  

Results  
- F2: The NN becomes progressively less positive as we move further in the PCA direction, falling from 0.530 at the 0.10 step to 0.271 at 0.30. The GP gives identical predictions because of its continuing length-scale problem and should not influence the decision. The 0.10 point is also almost a duplicate, at only 0.0037 from an existing observation. Overall, PCA identifies a general performance relationship for F2 but does not currently provide a particularly attractive new candidate.  
- F4: The GP initially supports movement along the PCA direction, predicting around 0.73 for the 0.10 and 0.20 steps. However, the NN strongly disagrees and deteriorates rapidly from 0.108 to -1.042. Given that the NN was considerably more accurate on the latest realised observation, this disagreement makes a purely PCA-guided candidate too risky. PCA should remain supporting evidence rather than drive the F4 selection.  
- F6: The GP predicts gradual improvement as the PCA step increases, reaching -0.203 at 0.30 compared with the current best of -0.240. The NN is more cautious and predicts deterioration as the step increases. Since both models have useful validation and the GP was slightly more accurate in Round 12, PCA remains relevant, but a moderate movement is preferable to simply taking the largest step.  
- F7: This is the strongest overall agreement. At the 0.10 step, both models predict performance around or above the current best, with the GP at 3.074 and NN at 3.117. At 0.20 the NN improves further to 3.137, although the GP falls slightly below the current best. This supports continued movement in the PCA direction, but suggests that a relatively small-to-moderate step is preferable.  
- F8: The models disagree systematically. The NN predicts continued improvement as the PCA step increases, reaching 10.098 at 0.30, whereas the GP predicts a gradual decline from the current best. The NN has strong LOOCV performance and was extremely accurate in Round 12, so its evidence is important. However, the GP is also stable and accurate, meaning the disagreement argues for a cautious PCA movement rather than automatically selecting the 0.30 step.  

PCA has added useful information, but the model evaluation shows why it should not be used independently. F7 provides the strongest combined evidence for a PCA-guided candidate, as PCA, the recent optimisation history and both models broadly support the same direction.  
F6 and F8 also provide useful PCA directions, but model disagreement means the size of the movement should be conservative. F2 and F4 show that a relationship between PCA position and historical output does not necessarily mean that continuing further in that direction will produce a better candidate.  
This leaves PCA as a useful additional method rather than a replacement for the existing GP, NN and local-search approaches.  

**Current position before final candidate refinement**  
- F1: PCA provides no useful additional evidence. Continue with model-independent exploration.  
- F2: PCA provides some supporting evidence, but the PCA-guided candidates were not strongly supported by the NN. Continue with an NN-led search around the established high-performing region.  
- F3: PCA evidence is weak and the latest observation produced a new best. Continue careful local refinement around the current best.  
- F4: PCA identifies a potentially useful direction, but the GP and NN predictions disagree substantially. Continue local refinement, giving greater consideration to the NN based on its recent accuracy.  
- F5: PCA provides very strong support for movement towards the upper boundary, but this direction has already reached [1,1,1,1]. Continue targeted boundary refinement while maintaining some distance from previous observations.  
- F6: PCA identifies a strong performance direction, although the GP and NN disagree somewhat about how far to move. Use a conservative search combining GP and PCA evidence.  
- F7: PCA, GP, NN and recent optimisation progress broadly support the same region. Continue local refinement using the combined evidence.  
- F8: PCA and the NN strongly support further movement in the current direction, while the GP is more cautious. Use a conservative PCA/NN-supported refinement while retaining some GP influence.  

**Step 10. Function-specific candidate refinement**

Final candidate searches were carried out separately for each function using the evidence developed throughout the round. Rather than applying one optimisation rule across all eight functions, the search method was adapted according to recent improvement, GP and NN reliability, PCA evidence and distance from previous observations.  
For functions where PCA identified a meaningful performance direction, this information was incorporated alongside the existing models rather than used independently. Where PCA evidence was weak or contradicted by the predictive models, the previous local or exploratory strategies were retained. Candidate novelty was also considered to avoid selecting points that were effectively duplicates of existing observations.  

F1: The model-independent search selected [0.585780, 0.301450], which is 0.233 from the nearest previous observation. Given the continuing GP failure and lack of useful PCA evidence, this is a reasonable exploratory candidate.  

F2: The NN-led search selected [0.708357, 0.238021]. The NN predicts 0.695, reasonably close to the current best of 0.747, while maintaining a useful distance of 0.051. This is more attractive than the PCA-guided candidates and provides a strong option for final selection. 

F3: The balanced search selected [0.451202, 0.516398, 0.469787]. Both models predict an improvement over the current best of -0.000352, with the GP predicting 0.00777 and NN 0.00045. Although the distance is fairly small at 0.026, this is appropriate for a function that has just produced another new best through local refinement. The agreement between both models provides reasonable support for continued local refinement.  

F4: The selected point [0.379411, 0.353227, 0.324930, 0.420568] is supported particularly strongly by the GP, which predicts 0.789 against a current best of 0.623. The NN is more conservative at 0.402, but importantly it is nowhere near as negative as it was for the more aggressive PCA candidates. This provides a better balance between the model predictions than following the PCA direction directly.   

F5: The candidate [0.997941, 0.995996, 0.984983, 0.987126] stays very close to the known upper-boundary optimum while remaining a novel observation. Neither model predicts a new best, but that is unsurprising because The search has already reached the extreme upper-boundary region, so the candidate focuses on fine refinement close to this area. This provides a sensible continuation of the boundary strategy, although alternative near-boundary candidates should be compared before final selection.    

F6: This candidate requires greater caution because the two models disagree substantially. The GP predicts a substantial improvement to -0.138, but the NN predicts deterioration to -0.317. The point is also relatively far from the existing observations at 0.102. This is also a more aggressive movement than the PCA-guided candidates. Given the model disagreement, a more conservative intermediate candidate may provide a better balance between predicted improvement and risk. 

F7: The models again show substantial disagreement. The NN predicts an excellent 3.238, but the GP predicts 2.987, below the current best of 3.070. At 0.093 from previous observations, this is fairly exploratory. The PCA 0.10 candidate was considerably more conservative and had both models predicting performance around or above the current best. An intermediate candidate may therefore provide a better compromise between the two approaches.  

F8: This represents the most aggressive candidate movement of the eight functions. The NN predicts 10.212, which is very attractive, but the GP has fallen to 9.933 and the point is a substantial 0.181 from previous observations. Given that the PCA 0.10–0.30 candidates provide more controlled movements with distances of approximately 0.030–0.090, the NN prediction alone does not provide sufficient evidence to justify such a large movement.  

**Step 11. Final targeted candidate comparison**

A final targeted comparison was carried out for F5–F8, where the previous stage left some uncertainty over how aggressively to refine the promising regions.  
For F5, several near-boundary candidates were compared because PCA confirmed the direction towards [1,1,1,1], but the exact boundary point had already been observed.  
For F6–F8, intermediate candidates were generated between the conservative PCA-guided alternatives and the more aggressive model-selected candidates from Step 10. These were evaluated using GP and NN predictions and distance from previous observations. This allowed the final choice to balance predicted performance, model agreement and novelty rather than automatically selecting either extreme.  

Results  
- F5: Boundary Option 1 remains the strongest choice. It has the highest GP prediction (8054.28) and NN prediction (8072.19) of the four alternatives, while maintaining a nearest distance of 0.0203. Moving further away from the boundary causes both predictions to fall, reinforcing the evidence that performance is concentrated very close to [1,1,1,1].  
- F6: The GP becomes increasingly optimistic as the candidate moves towards the model-selected point, while the NN becomes increasingly pessimistic. The 0.33 intermediate candidate provides a sensible compromise: the GP predicts -0.1861, above the current best of -0.2398, while the NN predicts -0.2652. Its distance of 0.0571 also provides useful novelty without moving as far into an uncertain region as the more aggressive alternatives.  
- F7: The NN becomes increasingly positive as the search moves towards the model-selected point, but the GP moves progressively in the opposite direction. The 0.33 intermediate candidate again provides a reasonable balance, with predictions of 3.0450 from the GP and 3.1756 from the NN. It remains relatively close to the established high-performing region with a distance of 0.0626.  
- F8: The same trade-off is particularly clear. The NN increasingly favours the more aggressive candidates while the GP becomes progressively less positive and more uncertain. The 0.33 intermediate candidate offers a useful compromise, with an NN prediction of 10.1110, a GP prediction of 9.9691, and a distance of 0.0953. This retains the strong PCA/NN direction without relying completely on the much more aggressive NN-led candidate.  

Model disagreement was also used as an indication of candidate risk. Where the GP and NN gave similar predictions, there was greater confidence that the candidate remained within a well-understood region. For F6–F8, the models increasingly disagreed as the candidates moved further from previous observations. Rather than selecting the most aggressive candidates, intermediate points were therefore chosen to retain the promising PCA direction while limiting the risk of relying too heavily on a single model.  

The final comparison shows that the most aggressive model-selected candidates for F6–F8 are not necessarily the strongest choices once model disagreement and novelty are considered. Intermediate candidates allow the PCA evidence to influence the search while avoiding excessive movement away from regions already supported by observations.  

**Step 12. Final Round 13 candidate validation**

The final candidates were checked before submission to confirm that each point had the correct number of dimensions, remained within the permitted [0,1] input range and did not duplicate an existing observation. The distance to the nearest previous observation was also recorded as a final check that each candidate provided some new information.

All eight candidates passed the validation checks.  
No candidates duplicated an existing observation.  
F1 remained the most exploratory candidate, with a nearest-observation distance of 0.233.  
F3–F5 remained relatively close to known high-performing regions, reflecting the more local refinement strategy used for these functions.  
F6–F8 maintained greater distance from previous observations, reflecting the additional influence of PCA and the aim of exploring promising new directions.  
The validated candidates were therefore selected as the final Round 13 submissions.  

Overall, this round showed that PCA was most useful as an additional source of evidence rather than as a standalone optimisation method. It provided particularly useful guidance for F5–F8, while adding little for F1 and F3. Combining PCA with realised model accuracy, GP uncertainty, NN validation and candidate novelty allowed the final selections to remain function-specific while becoming increasingly focused on exploitation as the query budget reduces.  





So far:  
 
Week 1: Establish a Gaussian Process + UCB baseline for all eight functions.  
Week 2: Diagnose function-specific GP behaviour and adapt the strategy where the surrogate model struggles, including testing alternative kernels and greater exploration.  
Week 3: Introduce SVM classification to identify high- and low-performing regions and use it selectively where it adds useful information.  
Week 4: Investigate the exploration–exploitation trade-off by comparing different UCB κ values and Expected Improvement, leading to more function-specific query selection.  
Week 5: Introduce a simple neural-network surrogate as a secondary model and compare its recommendations with the established GP-based approach.  
Week 6: Use accumulated results to assess which methods are actually working for each function and make candidate selection more evidence-driven.  
Week 7: Systematically tune UCB and EI hyperparameters, assess candidate stability, and combine GP–NN agreement, validation and neighbourhood evidence to refine function-specific selections.  
Week 8: Strengthen the function-specific approach by comparing GP and NN recommendations with the performance of nearby observations, using model diagnostics and validation to decide which model should receive greater weight for each function.  
Week 9: Introduce multi-radius local refinement around established high-performing regions, alongside novelty constraints and model-independent exploration where appropriate, to balance small improvements against gathering new information.  
Week 10: Evaluate the realised accuracy of the previous round's GP and NN predictions, update model diagnostics and NN validation, and use this evidence to refine the function-specific strategy. Local searches, novelty constraints and model-independent exploration are then used selectively rather than applying a single optimisation method across all functions.  
Week 11: Introduce K-Means clustering to identify broader high-performing regions within each function’s search space and assess how meaningful those regions are using silhouette scores and cluster performance. Combine this new regional evidence with GP and NN predictions, realised model accuracy, local refinement and novelty to generate and compare function-specific candidates before final selection.  
Week 12: Introduce PCA to identify the main directions of variation within each function and assess how these relate to output performance. Use this alongside GP and NN predictions to guide candidate refinement, while using model disagreement to avoid overly aggressive moves in uncertain regions.  
