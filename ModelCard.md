# Final Capstone Model Card

## Bayesian Black-Box Optimisation Framework

## 1. Overview

**Model name:** Adaptive Bayesian Black-Box Optimisation Framework  
**Model type:** Sequential surrogate-based optimisation  
**Version:** Final Capstone Model Card

This framework was developed to maximise eight unknown black-box functions under a restricted evaluation budget. Each function had a different number of input dimensions, ranging from two to eight, with all inputs bounded between 0 and 1.

The approach used Gaussian Process (GP) regression as its primary surrogate model, supported by acquisition functions, neural-network ensembles and additional diagnostic techniques. Rather than applying a single optimisation strategy to every function, the framework evolved to make function-specific decisions based on observed performance, model reliability and the available evaluation budget.

## 2. Intended Use

### Suitable applications

The framework is designed for optimisation problems where:

- Objective-function evaluations are expensive or limited.
- Gradients and analytical expressions are unavailable.
- Input variables are continuous and bounded.
- Only a relatively small number of observations are available.
- Balancing exploration and exploitation is important.

Potential applications include hyperparameter tuning, engineering design and scientific experimentation.

### Unsuitable applications

The approach is less suitable for very high-dimensional or discrete optimisation problems, highly discontinuous objectives, or situations requiring guaranteed global optima.

It should also be used cautiously when objective functions are highly noisy or surrogate-model uncertainty is unreliable.

## 3. Model Details and Strategy Evolution

### Initial approach

The project began with Gaussian Process regression using a Radial Basis Function (RBF) kernel. Upper Confidence Bound (UCB) was used to balance predicted objective values against uncertainty, allowing candidate selection to consider both promising regions and areas requiring further exploration.

Each of the eight functions was modelled independently, with newly returned observations incorporated into its cumulative dataset before subsequent optimisation rounds.

### Development of the optimisation strategy

As more observations became available, the framework developed beyond its initial GP and UCB approach.

Key developments included:

- **Additional acquisition strategies:** Expected Improvement (EI) was introduced alongside UCB to compare different approaches to selecting promising candidates.
- **Candidate refinement:** Broader searches were supplemented by more concentrated searches around promising regions.
- **Neural-network ensembles:** Multiple neural networks provided an alternative set of predictions and helped identify disagreement with the GP.
- **Model validation:** Leave-One-Out Cross-Validation (LOOCV) compared neural-network prediction errors against a simple mean-prediction baseline.
- **GP–NN comparisons:** Differences between GP and neural-network recommendations provided additional evidence when evaluating candidate locations.
- **Novelty constraints:** Minimum-distance checks helped prevent repeated evaluations of points that were too similar to existing observations.
- **Function-specific decisions:** Different functions received different combinations of exploration, exploitation and local refinement, depending on their observed behaviour.

The later stages placed greater emphasis on assessing whether the models could be trusted, rather than automatically selecting whichever candidate had the highest acquisition score.

### Responding to model limitations

An important finding was that the GP did not behave equally reliably across all functions.

For example, fitted RBF length scales for Functions 1 and 2 sometimes approached the lower optimisation bound, indicating potentially unstable surrogate behaviour. Function 1 was particularly challenging because observed outputs remained close to zero.

Alternative GP kernels were investigated for Function 1, but these did not establish a clearly superior model. Space-filling exploration therefore remained a useful alternative to relying entirely on model predictions.

For other functions, neural-network validation and GP–NN comparisons helped determine whether additional model recommendations were sufficiently credible to influence the next query.

The final framework was consequently adaptive: model predictions informed decisions, but diagnostics, previous observations and candidate diversity also influenced selection.

## 4. Performance

### Evaluation metrics

The primary performance measure was the **best observed objective value** for each function, rather than the highest value predicted by a surrogate model.

Supporting metrics and diagnostics included:

- Improvement over the previous best observation.
- GP predicted mean and standard deviation.
- UCB and EI acquisition values.
- Neural-network LOOCV mean absolute error (MAE).
- Comparison against a mean-prediction baseline.
- Agreement or disagreement between surrogate models.
- Distance from existing observations and promising regions.
- GP kernel behaviour and fitted length scales.

### Performance across the eight functions

| Function | Dimensions | Main observations |
|----------|------------|-------------------|
| F1 | 2 | Persistent difficulty identifying non-zero outputs; GP reliability concerns led to alternative kernel testing and space-filling exploration. |
| F2 | 2 | Mixed performance across queries, with later evidence of unstable GP length-scale fitting. |
| F3 | 3 | Improvements were identified through iterative refinement, although neural-network validation was not consistently stronger than the baseline. |
| F4 | 4 | Substantial improvements were achieved in some rounds, with neural-network validation providing useful supporting evidence. |
| F5 | 4 | Strong improvements were observed near the upper input boundaries, making local refinement and duplicate avoidance particularly important. |
| F6 | 5 | Neural-network validation provided useful evidence, although finding consistently improving candidates remained challenging. |
| F7 | 6 | The framework identified improvements while managing a larger search space and limited observations. |
| F8 | 8 | Improvements demonstrated the potential of the approach in higher dimensions, although sparse coverage remained a significant limitation. |

Performance varied considerably between functions. Some queries produced new best observations, while others primarily provided information about poorly understood regions or exposed limitations in the surrogate models.

A key lesson was that more complex modelling did not automatically produce better recommendations. Additional models were most useful when their predictions could be validated and compared against simpler alternatives.

## 5. Assumptions and Limitations

### Assumptions

The framework assumes that:

- Nearby input locations may have related objective values.
- The available observations contain useful information about promising regions.
- GP regression can provide meaningful predictions for at least part of the search space.
- Candidate generation offers sufficient coverage of relevant regions.
- Returned objective values can be compared consistently across optimisation rounds within each function.

### Limitations

The principal limitations are:

**Limited observations:** The evaluation budget was small relative to the dimensionality and complexity of the search spaces. This restricted the reliability of both GP and neural-network predictions.

**Surrogate-model instability:** GP hyperparameters could become unstable, particularly when fitted length scales approached their bounds. Predicted uncertainty was not necessarily a reliable measure of actual prediction error.

**Neural-network overfitting:** Neural networks trained on small datasets could produce misleading predictions. LOOCV provided a useful check but was itself sensitive to the limited sample size.

**Adaptive sampling bias:** Concentrating evaluations around successful regions could leave other potentially valuable areas unexplored.

**Computational and methodological complexity:** Comparing multiple models and candidate-generation methods increased the effort required to reproduce and interpret decisions.

**No global-optimum guarantee:** The best observed result represents the strongest evaluated point, not proof that the global maximum was found.

## 6. Ethical Considerations and Transparency

The capstone used numerical benchmark functions rather than personal or sensitive information, so privacy and direct social-impact risks were limited.

Nevertheless, transparency is important because optimisation decisions depend on assumptions about model behaviour and uncertainty.

The project documents the observations used, modelling techniques, acquisition strategies, candidate-selection decisions and submitted coordinates. Retaining weekly code and results makes it possible to examine why particular decisions were made and whether the resulting observations supported them.

In real-world applications, the same principles would help prevent misleading confidence in model predictions, particularly where optimisation outcomes could affect cost, safety or resource allocation.

## 7. Decision-Making and Reflection

The main strength of the framework is its ability to adapt as evidence accumulates. The project began with a relatively simple GP-based strategy and developed into a more comprehensive process that considered model agreement, validation performance, candidate novelty and the behaviour of individual functions.

An important lesson was that the most sophisticated model was not always the most reliable. For some functions, local refinement or space-filling exploration was more defensible than following an unstable surrogate prediction.

The principal weakness is the limited evaluation budget, which makes reliable model comparison difficult and increases the risk of drawing conclusions from very few observations.

This model card provides an overview of the framework's purpose, evolution, performance assessment and limitations. Exact parameters, intermediate outputs and individual query decisions are better documented in the weekly notebooks and results files, where they can be inspected alongside the code that generated them.

Together, these records provide a more complete account of both successful optimisation decisions and unsuccessful experiments, supporting reproducibility and future improvements.
