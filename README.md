
# Bayesian Black-Box Optimisation Capstone

## Project Overview

This repository documents my Bayesian Black-Box Optimisation (BBO) capstone project, completed as part of the Imperial College London Professional Certificate in Machine Learning and Artificial Intelligence.

The objective was to maximise eight unknown black-box functions using a limited number of evaluations. Each function had a different input dimensionality, ranging from two to eight, with all input variables bounded between 0 and 1.

The challenge was to identify promising input combinations without knowing the underlying mathematical functions or having access to their gradients. Each new evaluation provided additional information that could be used to improve subsequent optimisation decisions.

The project therefore focused on balancing **exploration** of uncertain regions with **exploitation** of areas where strong results had already been observed.

## Methodology

The initial approach used Gaussian Process (GP) regression with a Radial Basis Function (RBF) kernel and Upper Confidence Bound (UCB) acquisition to identify promising candidate points.

As the project progressed, the optimisation strategy evolved to incorporate additional techniques, including:

- **Expected Improvement (EI):** An alternative acquisition function for identifying candidates likely to improve on the current best observation.
- **Candidate refinement:** More focused searches around promising regions of the input space.
- **Neural-network ensembles:** Alternative surrogate predictions used to compare against GP recommendations.
- **Leave-One-Out Cross-Validation (LOOCV):** Evaluation of neural-network predictive performance against a simple baseline.
- **GP–NN comparisons:** Assessment of agreement and disagreement between surrogate-model recommendations.
- **Candidate novelty checks:** Avoidance of unnecessary evaluations at or very close to previously sampled points.
- **Function-specific exploration:** Adjusting candidate selection according to the observed behaviour and modelling reliability of each function.

Gaussian Processes remained the principal modelling approach, while additional techniques provided supporting evidence for candidate selection.

## Project Development

The project was completed through sequential optimisation rounds. Each round followed a broadly consistent process:

1. Incorporate the latest returned observations into the relevant datasets.
2. Fit or update the surrogate models for each function.
3. Generate and evaluate potential candidate locations.
4. Compare predictions, acquisition values and relevant diagnostics.
5. Select one new query for each of the eight functions.
6. Record the submitted coordinates and subsequently assess the returned results.

Over time, the strategy became more function-specific, particularly where surrogate predictions proved unreliable or the search space presented different challenges.

The weekly documentation records the modelling decisions, results, limitations and changes made throughout the project.

## Evaluation and Results

The primary performance metric was the **best observed objective value** for each function.

Additional diagnostics included:

- Improvement over previous best observations.
- GP predictive mean and uncertainty.
- Acquisition-function values.
- Neural-network cross-validation error.
- Agreement between alternative model recommendations.
- Candidate distance and search-space coverage.
- GP kernel behaviour and hyperparameter stability.

The eight functions presented different optimisation challenges. Some benefited from local refinement around promising regions, while others required greater exploration because model predictions were less reliable.

The project also highlighted the importance of validating surrogate models rather than assuming that increasingly complex modelling would necessarily improve optimisation performance.

Detailed numerical results and individual query decisions are documented in the weekly folders.

## Repository Structure

The repository is organised into weekly folders, each documenting a stage of the sequential optimisation process.

Each weekly folder contains:

- **Python files:** The modelling code used to generate and evaluate candidate points.
- **Data:** The observations available at that stage of the project, including newly returned results.
- **Weekly write-up:** An explanation of the methodology, modelling decisions, submitted coordinates, results and lessons learned.

This structure preserves the development of the optimisation approach and allows each week's work to be reviewed independently.

Additional repository-level documentation includes:

- **README.md:** Project overview, methodology and navigation.
- **Final Capstone Model Card:** Summary of the optimisation framework, performance and limitations.
- **Dataset Datasheet:** Documentation of the data, its collection and its use throughout the project.

## Tools and Libraries

The project primarily uses Python, with libraries including:

- **NumPy:** Numerical operations and candidate generation.
- **Pandas:** Data handling and analysis.
- **scikit-learn:** Gaussian Process regression, kernels, neural-network modelling and validation.

The weekly Python files document the specific methods, configurations and additional dependencies used in each round.

## Reproducibility

Each weekly folder contains the Python code, data and write-up associated with that optimisation round, allowing the modelling decisions and available observations to be traced throughout the project.

Where stochastic methods are used, the code includes random seeds to support reproducibility.

Because each optimisation round depends on observations returned after the previous submission, the project should be interpreted as a sequential process rather than a single optimisation run.

The weekly files provide the detailed evidence supporting the methodology and results summarised here.

## Documentation

Further information is available in:

- **Final Capstone Model Card:** An overview of the optimisation approach, its evolution, performance and limitations.
- **Dataset Datasheet:** A description of the data used throughout the project.
- **Weekly Write-ups:** Detailed explanations of modelling choices, submitted candidates, observed results and lessons learned.

## Key Learnings

The project demonstrated that successful black-box optimisation requires more than selecting the highest predicted value from a surrogate model.

Important considerations included balancing exploration and exploitation, identifying unreliable model behaviour, comparing alternative modelling approaches and adapting candidate selection to the characteristics of individual functions.

A central lesson was that additional model complexity should be justified by evidence of improved reliability or decision-making, particularly when working with very small datasets.
