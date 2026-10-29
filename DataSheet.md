
# Dataset Datasheet
## Bayesian Black-Box Optimisation (BBO) Capstone

### 1. Motivation

#### Why was this dataset created?

This dataset was created as part of the Bayesian Black-Box Optimisation (BBO) capstone for the Imperial College Professional Certificate in Machine Learning and Artificial Intelligence.

The objective was to maximise eight unknown black-box functions using a limited number of evaluations. The dataset provides the observations needed to develop surrogate models, assess their predictions and identify promising input combinations.

Unlike a conventional machine learning dataset, the observations are collected sequentially. Each new result informs the next optimisation decision, requiring a balance between exploring unfamiliar regions and refining areas that have already produced strong results.

#### What task does it support?

The dataset supports:

- Bayesian optimisation and sequential decision-making
- Gaussian Process (GP) surrogate modelling
- Neural-network surrogate modelling
- Acquisition function evaluation, including UCB and Expected Improvement
- Comparison of exploration and exploitation strategies
- Evaluation of optimisation performance across successive rounds

### 2. Composition

#### Dataset contents

The dataset contains observations for eight independent black-box functions, with input dimensionality ranging from two to eight.

| Function | Input dimensions |
|---|---:|
| Function 1 | 2 |
| Function 2 | 2 |
| Function 3 | 3 |
| Function 4 | 4 |
| Function 5 | 4 |
| Function 6 | 5 |
| Function 7 | 6 |
| Function 8 | 8 |

Each observation consists of:

- An input vector containing continuous numerical variables
- A corresponding scalar output returned by the black-box function

All input values are bounded between 0 and 1. The output scales vary considerably between functions, reflecting differences in their underlying objective surfaces.

The data is organised into separate CSV files for each function, with input and output observations used by the accompanying Python scripts and Jupyter notebooks.

#### Dataset size and evolution

The project began with an initial set of observations supplied for each function. The number of initial observations may differ between functions.

The dataset then grows cumulatively throughout the optimisation process, with one additional observation per function following each weekly evaluation.

Each new observation consists of the submitted input coordinates and the corresponding output returned by the evaluation platform. These observations are appended to the existing data rather than replacing previous results.

As the dataset expands, the surrogate models are retrained using all observations available at that stage. This allows new results to influence predictions, uncertainty estimates and subsequent query selection.

The distribution of observations also changes over time. Some functions become more densely sampled around promising regions, while others continue to require broader exploration.

Each weekly dataset therefore represents a snapshot of the information available at that point in the project. Observations from later rounds are not used when generating earlier recommendations.

#### Gaps and limitations

The dataset is intentionally sparse because the number of permitted function evaluations is limited.

This is particularly significant for higher-dimensional functions, where the number of possible input combinations increases substantially.

Although the datasets are intended to contain complete input-output observations, missing values and duplicate records should be checked when loading the files.

The limited and adaptively selected observations mean that some regions of the search space remain poorly represented.

### 3. Collection Process

#### How was the data collected?

The initial observations were provided as part of the capstone project.

Additional observations were generated through successive weekly optimisation rounds, with one new query submitted for each of the eight functions.

The general process involved:

1. Loading the observations available for each function.
2. Training surrogate models to approximate the unknown objective function.
3. Generating candidate input combinations within the permitted search space.
4. Evaluating candidates using acquisition functions, model predictions and uncertainty estimates.
5. Selecting one query point per function.
6. Submitting the selected points to the evaluation platform.
7. Recording the returned outputs and adding them to the datasets for the next round.

#### Optimisation strategy

The collection strategy evolved as additional observations became available.

Earlier rounds focused on Gaussian Process modelling and balancing exploration against exploitation through acquisition functions such as Upper Confidence Bound (UCB).

Later rounds introduced additional methods, including Expected Improvement (EI), neural-network ensembles, leave-one-out cross-validation (LOOCV), comparisons between GP and NN recommendations, and local refinement around successful observations.

Where surrogate models demonstrated poor predictive reliability, alternative approaches were investigated. For example, space-filling exploration was used for Function 1 when neither GP nor NN models provided reliable validation results.

The approach was therefore adaptive, with different functions requiring different balances between exploration and exploitation.

#### Time frame

The observations were collected across successive weekly rounds during the capstone project.

Each round used the observations available at that point, with the returned results incorporated into the following week's dataset.

The accompanying weekly methodology files document the modelling decisions, submitted queries and observed improvements.

### 4. Preprocessing and Uses

#### Preprocessing

The datasets were loaded and processed using Python, primarily with NumPy, pandas and scikit-learn.

Processing included:

- Separating input variables from objective outputs
- Checking dimensionality and input bounds
- Identifying existing observations to avoid duplicate submissions
- Calculating distances between candidate points and observed inputs
- Preparing data for Gaussian Process and neural-network training
- Applying output standardisation where required for neural-network modelling

Transformations used during model training were separate from the underlying stored observations. Original function outputs were retained for evaluating optimisation performance and comparing observed results.

GP hyperparameter optimisation, acquisition calculations and candidate filtering formed part of the modelling process rather than permanent changes to the raw dataset.

#### Intended uses

The dataset is intended for:

- Training and evaluating surrogate models
- Comparing Bayesian optimisation strategies
- Investigating exploration-exploitation trade-offs
- Assessing optimisation performance over successive rounds
- Analysing the effects of additional observations on model predictions
- Educational assessment and reproducibility of the capstone project

#### Inappropriate uses and limitations

The dataset should not be used to draw conclusions about real-world physical, financial or scientific systems, as the underlying benchmark functions are unknown.

It is also unsuitable as a general-purpose supervised learning benchmark because the observations are limited in number and selected adaptively rather than sampled independently.

Strong performance on the observed data does not guarantee accurate predictions across the full search space or identification of the global optimum.

The results should therefore be interpreted within the constraints of the capstone's evaluation budget and optimisation objectives.

### 5. Distribution and Maintenance

#### Availability

The dataset is organised within the capstone project's GitHub repository, alongside the Python scripts, Jupyter notebooks and weekly methodology write-ups.

The data files are stored separately for each function, allowing the modelling process to be reviewed and reproduced.

#### Terms of use

The original benchmark observations were supplied for educational purposes as part of the Imperial College capstone.

The datasets and subsequent optimisation results are documented for academic assessment and project review. Any redistribution or reuse remains subject to the applicable course and data-provider terms.

No separate permission for unrestricted commercial reuse is implied.

#### Maintenance

The dataset is maintained by the project author.

During the capstone, maintenance consists primarily of appending new query-response observations while retaining the original data.

GitHub version control and the accompanying weekly methodology files provide a record of how the dataset and optimisation strategy evolved.

Following completion of the capstone, the final datasets and supporting analysis are intended to remain available as a record of the project, subject to applicable sharing permissions.

### 6. Assumptions and Reflections

The optimisation process assumes that observations from each black-box function can provide useful information about outputs at nearby, previously untested input combinations.

However, the reliability of this assumption varies between functions. Some demonstrate relatively smooth and predictable behaviour, while others are difficult to model using the available observations.

The limited evaluation budget also creates a trade-off between gathering information about unexplored regions and refining promising areas.

Throughout the project, model predictions were therefore assessed alongside cross-validation results, observed outputs, uncertainty estimates and candidate novelty.

An important limitation is that the data collection process is adaptive. Later observations are influenced by earlier results and modelling decisions, meaning the final dataset is not a uniformly sampled representation of the search space.

The dataset should consequently be understood as both a collection of optimisation observations and a record of the sequential decisions made throughout the capstone.

