
# Week 12 — Investigating Principal Component Analysis for Candidate Selection

**Imperial College AI/ML Capstone Project | Module 23**

## Objective

The objective of Week 12 was to investigate whether Principal Component Analysis (PCA) could provide additional information about the structure of the eight unknown functions and improve candidate selection.

Previous rounds had increasingly relied on function-specific optimisation strategies combining Gaussian Process (GP) regression, neural-network (NN) ensembles, acquisition functions, historical results, clustering and local refinement.

K-Means clustering had provided useful supplementary information in Week 11, although its effectiveness varied considerably by function. PCA was therefore introduced to examine whether combinations of input dimensions were associated with stronger function outputs.

The analysis focused on understanding the variation within the accumulated observations, identifying principal components associated with better performance, and generating additional candidates along promising directions.

PCA was treated as an additional source of evidence rather than a replacement for the existing optimisation framework.

The final candidate selections continued to consider model reliability, realised prediction accuracy, uncertainty, historical performance and novelty.

## 1. Incorporate and Assess the Latest Observations

The results returned from the Week 11 queries were incorporated into the datasets and compared with the previous highest observed outputs.

| Function | Previous Best | New Output | Updated Best | Result |
|---|---:|---:|---:|---|
| 1 | 0.000000 | ~0.000000 | 0.000000 | No improvement |
| 2 | 0.746962 | 0.561928 | 0.746962 | No improvement |
| 3 | -0.000863 | -0.000352 | -0.000352 | New best |
| 4 | 0.622950 | 0.245045 | 0.622950 | No improvement |
| 5 | 8662.482500 | 8119.407550 | 8662.482500 | No improvement |
| 6 | -0.239769 | -0.262935 | -0.239769 | No improvement |
| 7 | 3.024610 | 3.070097 | 3.070097 | New best |
| 8 | 9.984981 | 9.987419 | 9.987419 | New best |

Three of the eight queries produced new best observations.

**Function 3** improved from approximately -0.000863 to -0.000352, continuing to approach zero through local refinement.

**Function 7** increased from approximately 3.0246 to 3.0701, demonstrating further improvement within its established high-performing region.

**Function 8** achieved another new best of approximately 9.9874. However, the improvement was relatively small at 0.00244, suggesting that the optimisation might be approaching a plateau.

The unsuccessful results were also informative.

Function 4 returned 0.2450, substantially below its previous best of 0.6230, despite selecting a point relatively close to its strongest observations. This suggested that its high-performing region might be comparatively narrow.

Function 5 again failed to improve after moving away from its established upper-boundary optimum.

Function 6 remained relatively close to its best, while Function 2 returned 0.5619 after following an NN-supported region.

Function 1 continued to produce effectively zero outputs.

Overall, the results reinforced the importance of local refinement for several functions while highlighting the limitations of relying exclusively on surrogate predictions.

## 2. Evaluate Realised Prediction Accuracy

As in previous rounds, the GP and NN predictions recorded during candidate selection were compared with the outputs actually received.

Absolute prediction error was calculated as:

Absolute Error = |Predicted Output - Actual Output|

This provided an out-of-sample assessment of model reliability, complementing GP diagnostics and NN Leave-One-Out Cross-Validation (LOOCV).

### 2.1 Realised Prediction Errors

| Function | GP Absolute Error | NN Absolute Error | More Accurate Model |
|---|---:|---:|---|
| 1 | — | — | Not applicable |
| 2 | 0.1961 | 0.0994 | NN |
| 3 | 0.0382 | 0.0121 | NN |
| 4 | 0.4996 | 0.1075 | NN |
| 5 | 21.98 | 36.42 | GP |
| 6 | 0.0347 | 0.0518 | GP |
| 7 | 0.0713 | 0.0013 | NN |
| 8 | 0.0107 | 0.00075 | NN |

Function 1 was excluded because its candidate was selected through model-independent exploration.

The neural network was more accurate for five of the seven directly comparable functions.

**Function 7** provided the clearest example of NN accuracy. Its predicted output of approximately 3.0714 was extremely close to the realised result of 3.0701.

**Function 8** showed similarly encouraging performance, with an NN prediction error of only 0.00075.

Function 4 was particularly important because the GP substantially overestimated the result. Its error of approximately 0.4996 contrasted with the NN error of 0.1075, suggesting that greater weight should be given to the NN when assessing further local movements.

Function 3 also favoured the NN, although both models continued to experience difficulties predicting small output changes near zero.

The GP was more accurate for Functions 5 and 6.

For Function 5, both models performed reasonably well relative to the much larger scale of its outputs.

Overall, the results supported a function-specific approach to model weighting.

However, these comparisons were based on one new observation per function, so they were treated as supporting evidence rather than definitive measures of overall model quality.

## 3. Update Surrogate Models and Validation

The GP and NN models were refitted using the expanded datasets.

GP diagnostics continued to include the fitted RBF length scales, log marginal likelihood and any optimisation warnings.

Neural-network performance was reassessed using LOOCV, with Mean Absolute Error (MAE) compared against the mean-prediction baseline.

### 3.1 Neural-Network Validation

| Function | NN LOOCV MAE | Mean Baseline MAE | Assessment |
|---|---:|---:|---|
| 1 | 0.00114 | 0.00086 | Fails baseline |
| 2 | 0.1463 | Above NN MAE | Improved |
| 3 | 0.0588 | 0.0556 | Fails baseline |
| 4 | 1.2513 | 7.9456 | Strong |
| 5 | 169.15 | 2917.87 | Strong |
| 6 | 0.2167 | Above NN MAE | Improved |
| 7 | 0.4157 | Above NN MAE | Improved |
| 8 | 0.2743 | 1.0295 | Strong |

The NN outperformed the baseline for **six of the eight functions**.

Function 2 improved considerably, with MAE decreasing from approximately 0.2369 to 0.1463.

Function 4 improved from 1.4783 to 1.2513, reinforcing the evidence from its more accurate realised prediction.

Functions 6 and 7 also showed improved validation performance.

**Function 8** demonstrated particularly encouraging results, with MAE decreasing from approximately 0.3672 to 0.2743.

Function 5 remained strongly validated despite a slight increase in error.

Functions 1 and 3 continued to fail the baseline comparison.

Function 3's result was particularly interesting because its NN had accurately predicted the latest observation, but its overall LOOCV performance had deteriorated.

This demonstrated why individual successful predictions should not be treated as proof of general model reliability.

The updated evidence suggested:

- Greater NN influence for Functions 2, 4, 7 and 8.
- Continued use of both models for Functions 5 and 6.
- Cautious GP/NN comparison for Function 3.
- Continued model-independent exploration for Function 1.

## 4. Review Optimisation Progress

The running-best histories were examined to establish which functions continued to improve and which appeared to be approaching stable regions.

| Function | Current Best | Observations Since Best | Interpretation |
|---|---:|---:|---|
| 1 | ~0.000000 | 18 | Effectively stagnant |
| 2 | 0.746962 | 6 | No recent improvement |
| 3 | -0.000352 | 0 | Latest query improved |
| 4 | 0.622950 | 1 | Recently improved |
| 5 | 8662.482500 | 4 | Strong boundary region, no recent improvement |
| 6 | -0.239769 | 2 | Recent progress, now stable |
| 7 | 3.070097 | 0 | Latest query improved |
| 8 | 9.987419 | 0 | Latest query improved, smaller gains |

Functions 3, 7 and 8 had just produced new best observations, supporting continued refinement of their current regions.

Function 4's best was also relatively recent, although the latest result demonstrated that small movements could produce substantially weaker outcomes.

Function 5 remained concentrated near the upper input boundary, but had not improved for four observations.

Function 2 had gone six observations without improving, suggesting that alternative information might be useful when choosing its next candidate.

Function 6 remained promising, with both GP and NN models providing useful guidance.

Function 1 continued to show almost no meaningful progress.

This analysis provided the context for investigating whether PCA could reveal useful directions within the accumulated observations.

## 5. Apply Principal Component Analysis

PCA was applied separately to the input observations for each function.

The inputs were standardised before fitting PCA so that dimensions with greater sampled variation did not disproportionately influence the analysis.

For each function, the explained variance ratio of each principal component was calculated, together with the cumulative variance explained.

The number of components required to explain approximately 80%, 90% and 95% of the observed input variation was also examined.

### 5.1 Explained Variance Results

| Function | PC1 Explained Variance | Components for 80% | Key Finding |
|---|---:|---:|---|
| 1 | 52.33% | 2 of 2 | Little dimensionality reduction |
| 2 | 65.99% | 2 of 2 | Relatively strong main direction |
| 3 | 57.77% | 2 of 3 | Moderate dimensionality reduction |
| 4 | 36.51% | 3 of 4 | Variance spread across dimensions |
| 5 | 61.55% | 3 of 4 | Strong main direction |
| 6 | 30.09% | 4 of 5 | Limited dimensionality reduction |
| 7 | 28.24% | 4 of 6 | Several components remain important |
| 8 | 25.53% | 5 of 8 | Variance distributed across several dimensions |

**Function 5** showed one of the clearest dominant directions, with PC1 explaining approximately 61.6% of input variation.

Function 3 also showed useful compression, with the first two principal components explaining 84.2% of variance.

Functions 4, 6, 7 and 8 displayed more widely distributed variation.

For Function 8, the first five components explained approximately 80.1%, but seven were required to exceed 90%.

This suggested that PCA might be useful for understanding relationships between dimensions without necessarily providing substantial dimensionality reduction.

Importantly, explained variance alone did not indicate whether a component was relevant for optimisation.

A principal component could explain considerable variation within the inputs without having a meaningful relationship with function outputs.

The next stage therefore examined how PCA scores related to observed performance.

## 6. Examine PCA Loadings and Output Relationships

The principal-component loadings were examined to identify which original input dimensions contributed most strongly to each direction.

PCA scores were then compared with the corresponding outputs using correlations.

This provided an indication of whether movement along particular principal-component directions was associated with stronger performance.

Because PCA is unsupervised, these correlations were treated as observational evidence rather than proof of a direct causal relationship.

### 6.1 Strongest PCA–Output Relationships

| Function | Strongest Component | Output Correlation | Assessment |
|---|---|---:|---|
| 1 | PC1 | 0.038 | Negligible |
| 2 | PC2 | -0.573 | Moderate |
| 3 | PC3 | -0.335 | Relatively weak |
| 4 | PC1 | -0.652 | Strong |
| 5 | PC1 | +0.906 | Very strong |
| 6 | PC1 | -0.741 | Strong |
| 7 | PC1 | -0.606 | Moderate to strong |
| 8 | PC1 | -0.918 | Very strong |

**Function 1**

Neither principal component showed a meaningful relationship with output.

The strongest correlation was only 0.038, providing little evidence that PCA could improve candidate selection.

**Function 2**

PC2 showed a moderate negative correlation of approximately -0.573.

The corresponding loadings suggested that higher outputs tended to occur with relatively higher values of Dimension 1 and lower values of Dimension 2.

This broadly agreed with the location of the current high-performing region.

**Function 3**

PC3 had the strongest relationship with output, although its correlation of -0.335 was relatively weak.

PC3 also explained only 15.8% of input variance.

PCA therefore offered limited additional guidance.

**Function 4**

PC1 showed a relatively strong negative correlation of -0.652.

All four dimensions contributed positively to PC1, particularly Dimensions 1 and 2.

Movement towards lower PC1 scores appeared to be associated with higher observed outputs.

**Function 5**

PC1 explained 61.55% of input variation and had a very strong positive correlation with output of 0.906.

All four dimensions contributed positively and fairly evenly.

This reinforced the existing evidence that increasing the inputs together towards the upper boundary was associated with substantially stronger performance.

**Function 6**

PC1 had a strong negative correlation of approximately -0.741.

Dimensions 4 and 5 made particularly important contributions in opposing directions.

This suggested that PCA might help identify useful combinations of input changes for further refinement.

**Function 7**

PC1 showed a negative correlation of -0.606.

Dimensions 1 and 6 contributed strongly in opposite directions.

Combined with the latest new best, this provided useful supporting evidence for continued refinement.

**Function 8**

PC1 demonstrated the strongest relationship with output of all eight functions, with a correlation of approximately -0.918.

Dimensions 1, 3 and 7 contributed positively, while Dimension 5 contributed negatively.

Lower PC1 scores were therefore strongly associated with higher observed outputs.

This was particularly interesting because PC1 explained only 25.53% of total input variance.

It demonstrated that a component did not need to explain most of the input variation to be strongly associated with performance.

### 6.2 Interpretation

PCA appeared particularly informative for Functions 5 and 8.

Functions 4, 6 and 7 also demonstrated meaningful relationships between principal-component scores and output.

Function 2 showed moderate evidence, while Functions 1 and 3 gained relatively little additional information.

The results suggested that PCA should be incorporated selectively rather than applied identically to all eight functions.

## 7. Examine High-Performing Observations in PCA Space

The highest-performing 20% of observations were identified for each function.

Their principal-component scores were compared with those of the full dataset and the current best observation.

The purpose was to determine whether strong outputs were concentrated in particular parts of PCA space.

### 7.1 Key Findings

**Function 1**

No clear concentration was identified.

The current best observation had a PC1 standardised position of approximately -0.04, close to the overall mean.

This reinforced the conclusion that PCA provided little useful guidance.

**Function 2**

The current best observation had a PC2 z-position of approximately -1.23, consistent with the negative correlation between PC2 and output.

However, the highest-performing observations occupied a relatively broad range.

**Function 3**

The strongest observations showed limited separation from the rest of the dataset.

The current best also lay slightly opposite to the direction suggested by the overall correlation.

PCA therefore remained unreliable for candidate selection.

**Function 4**

The top 20% of observations occupied lower PC1 positions, with an average z-position of approximately -0.45.

This supported the negative relationship between PC1 and output.

**Function 5**

The top 20% of observations had an average PC1 z-position of approximately +1.40.

The current best was at approximately +1.45.

This provided particularly strong evidence that the best observations were concentrated towards the upper end of PC1.

**Function 6**

The strongest observations had an average PC1 z-position of approximately -0.88.

The current best was at approximately -1.01.

This supported the relationship between lower PC1 scores and stronger outputs.

**Function 7**

The top 20% had an average PC1 z-position of approximately -0.90, while the current best was approximately -0.99.

This indicated that the recent successful local refinements were concentrated in a consistently high-performing area of PCA space.

**Function 8**

The strongest observations were especially concentrated towards lower PC1 scores.

The top 20% had an average PC1 z-position of approximately -1.59, with the current best at approximately -1.60.

Combined with the correlation of -0.918, this provided very strong evidence that PC1 captured an important performance-related direction.

### 7.2 Overall Assessment

Functions 5, 6, 7 and 8 showed the clearest concentration of high-performing observations along a principal-component direction.

Function 4 provided useful but less pronounced evidence.

Function 2 showed some support, while Functions 1 and 3 remained unsuitable for PCA-led selection.

However, the relationships were derived from the observations collected so far.

There was no guarantee that continuing further in a favourable PCA direction would produce higher outputs, particularly beyond previously sampled regions.

## 8. Generate PCA-Guided Candidates

PCA-guided candidates were generated for functions where the previous analysis identified a meaningful relationship between principal-component position and output.

Candidate points were constructed by moving from the current best observation in the direction associated with stronger historical performance.

Three movement sizes were evaluated:

- 0.10 — conservative movement.
- 0.20 — moderate movement.
- 0.30 — broader movement.

All points were constrained to the original `[0, 1]` search space.

Distances to existing observations were also calculated to assess candidate novelty.

### 8.1 Candidate Generation Findings

| Function | PCA Candidate Assessment |
|---|---|
| 1 | No candidate generated; insufficient PCA evidence |
| 2 | Lower PC2 direction, but limited NN support |
| 3 | No candidate generated; PCA evidence weak |
| 4 | Lower PC1 direction provides novel alternatives |
| 5 | Favourable direction returns existing boundary optimum |
| 6 | Lower PC1 direction offers useful local alternatives |
| 7 | PCA direction aligns with recent high-performing region |
| 8 | Lower PC1 direction offers particularly promising alternatives |

**Function 2**

Moving towards lower PC2 scores generated plausible candidates.

However, the 0.10 candidate was only 0.0037 from an existing observation.

The 0.20 and 0.30 candidates provided greater novelty, with nearest distances of approximately 0.0263 and 0.0528.

**Function 4**

The 0.20 candidate had a nearest distance of approximately 0.0510, while the 0.30 candidate increased this to 0.0765.

These offered controlled alternatives to the existing local search.

**Function 5**

Every movement along the favourable PCA direction returned `[1, 1, 1, 1]`.

This was informative because PCA independently identified the same upper-boundary region favoured by previous optimisation methods.

However, the point had already been observed, meaning PCA could not directly generate a novel candidate.

**Function 6**

The 0.20 and 0.30 movements produced nearest distances of approximately 0.0406 and 0.0610.

These provided useful alternatives to the existing GP/NN local search.

**Function 7**

The generated candidates remained relatively close to the current high-performing region.

The 0.30 candidate provided a nearest distance of approximately 0.0834.

**Function 8**

The PCA candidates showed a useful progression towards lower PC1 scores.

The 0.20 and 0.30 candidates had nearest distances of approximately 0.0599 and 0.0898.

Given the strong PCA–output relationship, these were particularly interesting candidates for further evaluation.

Overall, PCA generated potentially useful alternatives for Functions 4, 6, 7 and 8.

However, model predictions were required to determine whether the proposed directions were likely to improve performance.

## 9. Evaluate PCA-Guided Candidates

The PCA-guided candidates were evaluated using the updated GP and NN models.

For each movement size, predicted output, GP uncertainty and distance from existing observations were considered.

The objective was to determine whether the surrogate models supported the promising directions identified by PCA.

### 9.1 Function-Specific Results

**Function 2**

The NN became progressively less optimistic as the PCA movement increased.

Its predicted output declined from approximately 0.530 at a 0.10 step to 0.271 at 0.30.

The GP remained unreliable because of its length-scale problem.

PCA therefore provided some information about the general high-performing region but did not identify a particularly attractive candidate.

**Function 4**

The GP predicted approximately 0.73 for the 0.10 and 0.20 candidates.

However, the NN predictions deteriorated sharply, from approximately 0.108 to -1.042.

Given that the NN had been considerably more accurate in the latest realised comparison, this disagreement reduced confidence in following the PCA direction directly.

**Function 6**

The GP became increasingly optimistic as the PCA movement increased, predicting approximately -0.203 at a 0.30 step.

This exceeded the current best of -0.2398.

However, the NN predicted deterioration as the candidate moved further away.

This suggested that a more conservative intermediate movement would be preferable.

**Function 7**

This function showed the strongest overall agreement.

At the 0.10 step, the GP predicted approximately 3.074 and the NN predicted 3.117.

Both exceeded the current best of 3.0701.

At 0.20, the NN prediction increased to approximately 3.137, although the GP became slightly less optimistic.

This supported a relatively conservative PCA-guided movement.

**Function 8**

The NN predicted increasing performance as PCA movement increased, reaching approximately 10.098 at a 0.30 step.

The GP moved in the opposite direction, predicting a gradual decline from the current best.

Both models had useful validation, while the NN had demonstrated extremely accurate recent predictions.

PCA therefore provided a promising direction, but the disagreement justified selecting a more cautious candidate rather than automatically choosing the most aggressive movement.

### 9.2 Interpretation

PCA added useful information but was not sufficiently reliable to determine final candidate selection independently.

Function 7 provided the strongest agreement between PCA, GP, NN and recent optimisation progress.

Functions 6 and 8 also benefited from PCA, although model disagreement suggested that intermediate movements would be preferable.

Function 4 demonstrated that a historically favourable PCA direction could still be associated with substantial predictive disagreement.

For Function 5, PCA reinforced the established upper-boundary strategy without providing a novel candidate.

The next step was therefore to incorporate the PCA evidence into the existing function-specific refinement framework.

## 10. Function-Specific Candidate Refinement

Additional searches were performed using different strategies for each function.

These incorporated recent performance, GP and NN reliability, PCA evidence and candidate novelty.

### 10.1 Initial Refined Candidates

**Function 1 — Model-Independent Exploration**

The GP and NN remained unreliable, while PCA provided no meaningful additional information.

A maximum-distance exploration strategy was retained.

Candidate:

`[0.585780, 0.301450]`

Nearest-observation distance: approximately 0.2330.

**Function 2 — NN-Led Refinement**

The NN-led candidate was:

`[0.708357, 0.238021]`

The NN predicted approximately 0.6954, relatively close to the current best of 0.7470.

The nearest-observation distance was 0.0515.

This provided a more attractive option than the PCA-guided alternatives.

**Function 3 — Balanced GP/NN Refinement**

Candidate:

`[0.451202, 0.516398, 0.469787]`

The GP predicted approximately 0.00777 and the NN approximately 0.00045.

Both exceeded the current best of -0.000352.

The nearest-observation distance was approximately 0.0258.

Given the latest new best and mixed validation evidence, cautious local refinement remained appropriate.

**Function 4 — Model-Supported Local Refinement**

Candidate:

`[0.379411, 0.353227, 0.324930, 0.420568]`

The GP predicted approximately 0.7893, exceeding the current best of 0.6230.

The NN was more conservative at approximately 0.4022, but considerably less negative than its predictions for the more aggressive PCA candidates.

This provided a more balanced alternative.

**Function 5 — Targeted Boundary Refinement**

Candidate:

`[0.997941, 0.995996, 0.984983, 0.987126]`

The GP predicted approximately 8054.28 and the NN approximately 8072.19.

Neither exceeded the current best of 8662.48.

However, the candidate remained extremely close to the known upper-boundary region, with a nearest distance of 0.0203.

**Function 6 — GP/PCA-Supported Refinement**

The more aggressive candidate was:

`[0.476308, 0.331583, 0.522296, 0.869116, 0.036876]`

The GP predicted approximately -0.1382, suggesting substantial improvement.

However, the NN predicted approximately -0.3167, below the current best.

The nearest distance of 0.1016 also represented a relatively large movement.

A less aggressive alternative was therefore considered preferable.

**Function 7 — PCA/NN-Supported Refinement**

The more aggressive candidate was:

`[0.100031, 0.313128, 0.513590, 0.281910, 0.315782, 0.722704]`

The NN predicted a strong output of approximately 3.2383.

However, the GP predicted only 2.9871, below the current best.

The substantial disagreement justified considering an intermediate candidate.

**Function 8 — PCA/NN-Supported Refinement**

The more aggressive candidate was:

`[0.002489, 0.060489, 0.108976, 0.166484, 0.831647, 0.561007, 0.131069, 0.639344]`

The NN predicted approximately 10.2116.

However, the GP predicted only 9.9328, while the candidate was 0.1806 from its nearest observation.

Given the much smaller distances of the PCA-generated alternatives, the NN prediction alone was insufficient justification for such a large movement.

## 11. Final Targeted Candidate Comparison

A final targeted comparison was performed for Functions 5–8.

Function 5 required additional boundary alternatives, while Functions 6–8 required a compromise between conservative PCA movements and more aggressive model-selected points.

Intermediate candidates were generated and compared using GP predictions, NN predictions and nearest-observation distance.

### 11.1 Function 5 — Boundary Alternatives

Four near-boundary candidates were considered.

The strongest option retained the highest predicted output from both models:

- GP prediction: 8054.28.
- NN prediction: 8072.19.
- Nearest-observation distance: 0.0203.

Moving further from the upper boundary caused both model predictions to deteriorate.

The initial targeted boundary candidate was therefore retained.

### 11.2 Function 6 — Intermediate Refinement

The GP became increasingly optimistic as the candidate moved towards the aggressive model-selected point.

However, the NN became increasingly pessimistic.

An intermediate candidate using a 0.33 weighting towards the aggressive alternative provided a useful compromise.

Selected candidate:

`[0.448352, 0.342492, 0.517316, 0.837227, 0.012169]`

GP prediction: -0.1861.

NN prediction: -0.2652.

Nearest-observation distance: 0.0571.

The GP predicted an improvement over the current best of -0.2398, while the NN was only moderately less optimistic.

The point also provided greater novelty than a very conservative local movement without requiring the more substantial move proposed initially.

### 11.3 Function 7 — Intermediate Refinement

The NN became increasingly positive towards the more aggressive candidate, while the GP became less optimistic.

The 0.33 intermediate candidate was selected.

Selected candidate:

`[0.126721, 0.281455, 0.505735, 0.281039, 0.301164, 0.727393]`

GP prediction: 3.0450.

NN prediction: 3.1756.

Nearest-observation distance: 0.0626.

The candidate retained the promising PCA direction while limiting model disagreement relative to the more aggressive alternative.

### 11.4 Function 8 — Intermediate Refinement

Function 8 displayed a similar trade-off.

The NN increasingly favoured more aggressive movements, while the GP predictions declined and uncertainty increased.

The 0.33 intermediate candidate was selected.

Selected candidate:

`[0.028951, 0.099216, 0.141909, 0.156573, 0.800039, 0.529150, 0.166737, 0.679484]`

GP prediction: 9.9691.

NN prediction: 10.1110.

Nearest-observation distance: 0.0953.

This preserved the strong PCA/NN-supported direction without relying entirely on the optimistic NN prediction for the more aggressive candidate.

### 11.5 Overall Assessment

The final comparisons demonstrated that the candidate with the highest individual model prediction was not necessarily the most appropriate selection.

For Functions 6–8, model disagreement increased as candidates moved further from the observed high-performing regions.

Intermediate candidates therefore provided a balance between promising new directions, predicted performance and the risk associated with greater exploration.

This approach allowed PCA to influence the final selections while maintaining the established evidence-based optimisation methodology.

## 12. Final Week 12 Query Selections

All eight final candidates were validated for correct dimensionality, permitted input bounds and duplication against existing observations.

The nearest-observation distance was also recorded to confirm that each candidate provided some new information.

### Final Proposed Queries

| Function | Final Proposed Point | Nearest Distance |
|---|---|---:|
| 1 | `[0.585780, 0.301450]` | 0.233016 |
| 2 | `[0.708357, 0.238021]` | 0.051474 |
| 3 | `[0.451202, 0.516398, 0.469787]` | 0.025823 |
| 4 | `[0.379411, 0.353227, 0.324930, 0.420568]` | 0.030276 |
| 5 | `[0.997941, 0.995996, 0.984983, 0.987126]` | 0.020261 |
| 6 | `[0.448352, 0.342492, 0.517316, 0.837227, 0.012169]` | 0.057059 |
| 7 | `[0.126721, 0.281455, 0.505735, 0.281039, 0.301164, 0.727393]` | 0.062554 |
| 8 | `[0.028951, 0.099216, 0.141909, 0.156573, 0.800039, 0.529150, 0.166737, 0.679484]` | 0.095332 |

All eight candidates passed the saved validation checks.

Every candidate had the correct dimensionality, remained within the `[0, 1]` input space and did not duplicate an existing observation.

Function 1 remained the most exploratory selection.

Functions 3–5 retained relatively local refinement strategies, while Functions 6–8 selected somewhat broader movements incorporating PCA evidence.

### Submission Format

    Function 1: 0.585780-0.301450
    Function 2: 0.708357-0.238021
    Function 3: 0.451202-0.516398-0.469787
    Function 4: 0.379411-0.353227-0.324930-0.420568
    Function 5: 0.997941-0.995996-0.984983-0.987126
    Function 6: 0.448352-0.342492-0.517316-0.837227-0.012169
    Function 7: 0.126721-0.281455-0.505735-0.281039-0.301164-0.727393
    Function 8: 0.028951-0.099216-0.141909-0.156573-0.800039-0.529150-0.166737-0.679484

## 13. Week 12 Reflection

Week 12 demonstrated how PCA could provide additional insight into the structure of the accumulated observations and support candidate refinement.

Three of the latest queries produced new best observations, with Functions 3, 7 and 8 continuing to improve within previously identified promising regions.

The realised prediction analysis reinforced the importance of evaluating surrogate models separately for each function.

The NN was particularly accurate for Functions 7 and 8, while the GP remained useful for Functions 5 and 6.

The PCA analysis provided several important findings.

Function 5 demonstrated a strong dominant principal component that was highly correlated with output. This reinforced the existing evidence that its strongest performance was concentrated near the upper input boundary.

Function 8 showed an even stronger PCA–output correlation of approximately -0.918, despite its first component explaining only 25.53% of the input variance.

This demonstrated that explained variance and optimisation relevance were not necessarily the same.

Functions 4, 6 and 7 also showed useful performance-related directions, while Function 2 provided moderate evidence.

Functions 1 and 3 gained relatively little useful information from PCA.

The candidate-generation stage demonstrated that movement along favourable PCA directions could produce novel alternatives, particularly for Functions 6–8.

However, the surrogate models did not always agree with the PCA evidence.

In Functions 6, 7 and 8, more aggressive movements produced increasingly optimistic predictions from one model while the other became less positive.

Rather than automatically selecting the highest individual prediction, intermediate candidates were generated to balance the competing evidence.

This allowed PCA to contribute to the search without overriding GP uncertainty, NN validation, realised prediction accuracy or candidate novelty.

Function 5 also demonstrated an important limitation of PCA-guided optimisation. Although PCA correctly identified the favourable direction towards `[1, 1, 1, 1]`, this point had already been observed, so PCA could not directly generate a novel candidate.

Overall, Week 12 expanded the optimisation framework by introducing PCA as another source of evidence.

Its usefulness varied considerably between functions, reinforcing the importance of adapting the search strategy to each individual problem.

With the query budget approaching its conclusion, the final selections continued to prioritise established high-performing regions while retaining controlled exploration where supported by the accumulated evidence.

The following round will evaluate these latest selections and provide a final assessment of the optimisation process, surrogate reliability and lessons learned across the capstone.

---

**Associated code:** `Week_12.ipynb`

