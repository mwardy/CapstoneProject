# CapstoneProject
**Capstone project for Imperial AI/ML Course**

**Objective**

The capstone project involves using Bayesian Optimisation to find the maximum values of eight unknown functions. It begins with 10 known data points for each function, which are used to build an initial model and select the first new query point. The functions have different numbers of input dimensions, meaning that some have relatively simple search spaces while others have many more possible combinations of inputs to explore. As the number of dimensions increases, the search space becomes more complex, making it more difficult to identify promising areas from the limited observations available. This means that the balance between exploring uncertain areas and exploiting areas that already appear promising may differ between functions. Each week, one additional data point is received for each function based on the queries submitted, allowing the models to be updated with more information. As the project progresses, the aim is to use this growing dataset to improve the models and make increasingly informed decisions about where to query next. Overall, the project demonstrates how an iterative optimisation approach can gradually improve its understanding of unknown problems of varying complexity and move towards better solutions.

**Methodology**

- Examine the initial data — Check the inputs and outputs for each function, including the number of dimensions, highest and lowest observed outputs, and output range.
- Fit a Gaussian Process — Train an RBF Gaussian Process for each function using the existing input-output observations.
- Generate candidate points — Create a large number of possible new input combinations between 0 and 1 for each function.
- Predict output and uncertainty — Use each GP to calculate the predicted mean (μ) and uncertainty (σ) for the candidate points.
- Calculate UCB — Score each candidate using Upper Confidence Bound, combining predicted performance with uncertainty to balance exploitation and exploration.
- Select and submit the next point — Choose the candidate with the highest UCB for each function, check whether the choice appears sensible, and format each query to six decimal places for submission.
- Reflect and update — Explain the reasoning behind the selections and, once the new outputs are received, add them to the existing data and repeat the optimisation process.

**Module 12 (Week 1)**

**Methodology**

**1. Search for highest and lowest observed output and note the corresponding input**

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

**2. Fit RBF Gaussian Processes**

Functions 1, 5 and 6 produced convergence warnings because the optimised RBF length scale reached its lower bound, suggesting the GP was modelling very rapid changes in output; the model predictions and uncertainty will therefore be checked before using them to select new query points.
- If we relax the lower bound, does the model simply push the length scale even closer to zero, or does it settle at a sensible value?

After checking:
- Function 5 behaves much more sensibly once I let the optimiser search a wider range. The GP settled on 0.269, which is a perfectly reasonable length scale, suggesting the original warning was artificial, caused by the optimiser hitting the default limit.
- Functions 1 & 6: Even after allowing much smaller values, both models immediately pushed the length scale down to 1e-10 and produced the same warning again. Telling me the data itself is making the GP believe the function changes extremely rapidly.
- Because of this, I'll keep the original GP models for Functions 1 and 6 rather than chasing an even smaller length scale.

The convergence warnings were investigated by relaxing the RBF length-scale bounds. Function 5 converged to a stable length scale, while Functions 1 and 6 continued to converge to the minimum value, so their original Gaussian Process models were retained for the initial optimisation.

**3. Generate Candidate Points and Predict Mean + Uncertainty**

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

**Module 13**

**Methodology**
