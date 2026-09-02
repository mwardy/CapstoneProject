# CapstoneProject
Capstone project for Imperial AI/ML Course

Objective

The capstone project involves using Bayesian Optimisation to find the maximum values of eight unknown functions. It begins with 10 known data points for each function, which are used to build an initial model and select the first new query point. Each week, one additional data point is received for each function based on the queries submitted, allowing the models to be updated with more information. As the project progresses, the aim is to use this growing dataset to improve the models and make increasingly informed decisions about where to query next. Overall, the project demonstrates how an iterative optimisation approach can gradually improve its understanding of an unknown problem and move towards better solutions.

Module 12 (Week 1)

For the first week, I plan to begin by exploring and visualising the 10 initial observations for each of the eight functions to understand the distribution of the inputs and outputs. I will then build Gaussian Process surrogate models and test different optimisation setups, comparing approaches such as RBF and Matérn kernels combined with UCB and Expected Improvement (EI) acquisition functions. This will allow me to assess how sensitive the suggested query points are to different modelling choices and identify which approaches appear most suitable for each function. I will then use these results, alongside the balance between exploration and exploitation, to select the first new query point for each function. The outputs received from these queries will then be added to the existing data and used to update and refine the models in the following week.
