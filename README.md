# CapstoneProject
**Capstone project for Imperial AI/ML Course**

**Objective**

The capstone project involves using Bayesian Optimisation to find the maximum values of eight unknown functions. It begins with 10 known data points for each function, which are used to build an initial model and select the first new query point. The functions have different numbers of input dimensions, meaning that some have relatively simple search spaces while others have many more possible combinations of inputs to explore. As the number of dimensions increases, the search space becomes more complex, making it more difficult to identify promising areas from the limited observations available. This means that the balance between exploring uncertain areas and exploiting areas that already appear promising may differ between functions. Each week, one additional data point is received for each function based on the queries submitted, allowing the models to be updated with more information. As the project progresses, the aim is to use this growing dataset to improve the models and make increasingly informed decisions about where to query next. Overall, the project demonstrates how an iterative optimisation approach can gradually improve its understanding of unknown problems of varying complexity and move towards better solutions.


**Module 12 (Week 1)**

**Methodology**

For the first week, I will begin by examining the initial observations for each of the eight functions, looking at the input combinations and their corresponding outputs to gain a basic understanding of the available data. I will then fit a Gaussian Process surrogate model using an RBF kernel for each function. The models will be used to estimate the predicted outputs and uncertainty across unobserved areas of the search space. I will then use the Upper Confidence Bound (UCB) acquisition function to balance exploration of uncertain areas with exploitation of areas predicted to produce high outputs, and select one new query point for each function. These points will then be submitted and, once their outputs are received, added to the existing observations so that the models can be updated and the optimisation process repeated in the following week. All of this will be to try and get a simple baseline that I can improve upon in future weeks.

In subsequent weeks, I plan to refine this approach as additional observations become available. This will include comparing alternative acquisition functions such as Expected Improvement and Probability of Improvement, and experimenting with the balance between exploration and exploitation to determine how these changes affect the selection of new query points.

**Module 13**

**Methodology**
