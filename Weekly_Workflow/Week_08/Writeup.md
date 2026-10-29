
# Module 19 – Week 8

## Methodology

The Week 8 optimisation process followed eight main steps, building on the results and modelling decisions from previous rounds.

1. **Add and assess the Week 7 results:** Incorporate the eight new observations, compare them with the previous best outputs and assess which strategies were successful.
2. **Refit and diagnose the models:** Update the Gaussian Process (GP) and neural-network (NN) surrogates, examine GP length scales and uncertainty, and repeat NN leave-one-out cross-validation (LOOCV).
3. **Generate and compare GP candidates:** Evaluate Upper Confidence Bound (UCB) and Expected Improvement (EI) candidates using representative exploration settings.
4. **Generate and compare NN candidates:** Obtain recommendations from the updated NN ensembles and assess their predictions, model disagreement and novelty.
5. **Compare GP and NN candidates:** Measure the distances between model recommendations to identify areas of agreement and disagreement.
6. **Conduct neighbourhood and function-specific analysis:** Examine nearby observed outputs, recent optimisation performance and model reliability. Investigate Function 1 further because neither surrogate provides reliable predictions.
7. **Select final candidates:** Choose one point per function using the combined evidence, including additional local searches where necessary.
8. **Validate and format submissions:** Check dimensionality, bounds, duplicates and distances from existing observations before formatting the eight final queries.

---

## Step 1 – Add and Assess the Week 7 Results

The eight Week 7 query points and their returned outputs were added to the datasets and compared with the previous best result for each function. This provided an initial assessment of which function-specific strategies had been successful before the models were updated.

| Function | Previous Best | Week 7 Output | Result |
|---|---:|---:|---|
| F1 | 0.000000 | 0.000000 | No improvement |
| F2 | 0.746962 | 0.569904 | No improvement |
| F3 | -0.015498 | -0.010190 | New best |
| F4 | -0.200431 | 0.186912 | New best |
| F5 | 7121.151955 | 8662.482500 | New best |
| F6 | -0.379740 | -0.388687 | No improvement |
| F7 | 2.929370 | 2.369289 | No improvement |
| F8 | 9.911249 | 9.930811 | New best |

The Week 7 submissions produced **four new best results across the eight functions**.

Functions 3 and 8 achieved relatively small improvements, while Functions 4 and 5 produced much more substantial gains. Function 4 improved from -0.200431 to 0.186912, moving into positive output values for the first time. Function 5 increased from 7121.151955 to 8662.482500 following the selection of the upper-boundary point [1, 1, 1, 1].

Among the unsuccessful queries, Function 6 remained close to its previous best, whereas Functions 2 and 7 showed more noticeable declines. Function 2 was particularly informative because its previous recommendation had relatively strong GP–NN agreement, demonstrating that agreement between surrogate models does not necessarily guarantee an improvement.

Function 1 again returned an effectively zero output and remains difficult to optimise reliably.

Overall, these mixed results reinforced the importance of assessing optimisation performance separately for each function rather than applying the same acquisition strategy across all eight problems.

---

## Step 2 – Refit and Diagnose the Models

The GP and NN surrogate models were refitted using the expanded datasets.

The existing modelling configurations were retained for this round to maintain consistency with the preceding analysis. This allowed changes in model behaviour to be assessed primarily in relation to the additional observations.

For the GP models, fitted length scales, prediction ranges and uncertainty ranges were examined. The models were evaluated across 100,000 candidate points per function.

The NN ensembles were also retrained, and their performance was assessed using LOOCV against a mean-prediction baseline. NN recommendations were given greater weight only where the validation results suggested that the models could generalise beyond the training observations.

### Step 2a – GP Refitting and Diagnosis

The fitted RBF length scales were:

| Function | RBF Length Scale |
|---|---:|
| F1 | 0.00001 |
| F2 | 0.00001 |
| F3 | 0.199 |
| F4 | 0.416 |
| F5 | 0.273 |
| F6 | 0.581 |
| F7 | 0.302 |
| F8 | 1.04 |

The fitted GP models remained relatively stable for Functions 3–8. Functions 3, 4, 6, 7 and 8 showed only modest changes following the additional observations.

Function 5's length scale decreased from approximately 0.363 to 0.273 after the strong result at the upper boundary. This suggests that the model is responding to more localised variation around the newly identified high-performing region.

The most significant change occurred for Function 2, where the fitted length scale fell from approximately 0.0431 to the lower bound of 1e-05. This produced the same convergence warning previously observed for Function 1.

Such a small length scale suggests that the GP is modelling variation over extremely short distances and may struggle to generalise between observations.

Function 1 remained at the lower bound for another week, continuing the reliability concerns identified in earlier rounds.

### GP Prediction and Uncertainty Ranges

| Function | Prediction Range | Uncertainty Range |
|---|---:|---:|
| F1 | 0.000560 | 0.001570 |
| F2 | 0.651312 | 0.244076 |
| F3 | 0.388723 | 0.078255 |
| F4 | 35.821887 | 7.855238 |
| F5 | 7847.668688 | 2510.829993 |
| F6 | 2.303126 | 0.523703 |
| F7 | 3.026197 | 0.818402 |
| F8 | 4.878650 | 0.653663 |

The diagnostics showed meaningful variation in predicted outputs and uncertainty across most functions.

Function 1 continued to produce extremely small predicted values, with little evidence that its GP could reliably distinguish promising regions.

Function 2 was more interesting. Despite its length scale reaching the lower bound, the GP continued to produce a relatively wide prediction range, approximately -0.0817 to 0.5696. However, its maximum prediction was below the current best observation of 0.746962.

The lower-bound length scale means these predictions should be treated cautiously rather than relied upon as the primary basis for candidate selection.

Functions 3–8 showed more substantial variation in both predictions and uncertainty, suggesting that their GPs remained useful for distinguishing between candidate regions.

Functions 4 and 5 had particularly large uncertainty ranges, meaning that acquisition-based selection remained important rather than simply choosing the highest predicted mean.

Overall, the GP evidence was considered more reliable for Functions 3–8, while Functions 1 and 2 required additional scrutiny.

### Step 2b – NN Refitting and Validation

The five-model NN ensembles were retrained using the updated datasets.

The mean training losses were:

| Function | Mean Training Loss |
|---|---:|
| F1 | 0.000963 |
| F2 | 0.010989 |
| F3 | 0.000408 |
| F4 | 0.000284 |
| F5 | 0.000230 |
| F6 | 0.000370 |
| F7 | 0.000338 |
| F8 | 0.000381 |

The ensembles continued to achieve low training losses for most functions, indicating that they could fit the observed data closely.

Function 2 was the main exception. Its mean training loss increased from 0.006460 in the previous round to 0.010989.

This was notable because its GP length scale also reached the lower bound, suggesting that Function 2 had become more difficult for both surrogate approaches to represent.

However, training loss alone is not sufficient evidence of reliable prediction on unseen observations. LOOCV was therefore used to assess generalisation.

### NN LOOCV Results

| Function | NN LOOCV MAE | Mean Baseline MAE | Beats Baseline |
|---|---:|---:|---|
| F1 | 0.001704 | 0.001050 | No |
| F2 | 0.219676 | 0.237993 | Yes |
| F3 | 0.062478 | 0.051662 | No |
| F4 | 1.825570 | 7.284354 | Yes |
| F5 | 230.163312 | 2049.255132 | Yes |
| F6 | 0.241315 | 0.513894 | Yes |
| F7 | 0.399610 | 0.639930 | Yes |
| F8 | 0.354980 | 0.962395 | Yes |

The NN ensembles outperformed the mean-prediction baseline for **six of the eight functions**.

Functions 4, 5, 6 and 8 showed particularly clear improvements over the baseline, providing continued support for using their NN models as additional evidence during candidate selection.

Function 7 also remained validated.

Function 2 narrowly outperformed its baseline, with an MAE of 0.219676 compared with 0.237993. Although this was not a particularly strong validation result, it suggested that the NN retained some predictive value despite the deterioration in its GP model.

Functions 1 and 3 again failed to outperform their baselines. Their NN recommendations were therefore assigned limited weight.

For Function 1, both the GP and NN now provided weak modelling evidence. Function 3 had a more useful GP and recent successful observations, allowing greater reliance on GP-based local refinement.

Overall, the reliability pattern remained broadly consistent with the previous round: the NN models were useful for Functions 2 and 4–8, while Functions 1 and 3 required greater reliance on alternative evidence.

---

## Step 3 – Generate and Compare GP Candidates

The refitted GP models were used to generate candidates using UCB and EI.

Rather than repeating the full hyperparameter investigation from the previous round, representative settings were selected:

- UCB with κ = 0.5
- UCB with κ = 1.5
- EI with ξ = 0% of the observed output standard deviation
- EI with ξ = 5% of the observed output standard deviation

These settings provided a comparison between relatively exploitative and more exploratory acquisition behaviour.

For each candidate, the predicted output, uncertainty, acquisition score and distance from the nearest existing observation were recorded.

Candidates were evaluated using recent optimisation performance and model reliability rather than acquisition scores alone.

### Function 1

All four acquisition settings selected the same point:

[0.094177, 0.975622]

This showed complete stability across the tested UCB and EI settings.

However, because the fitted GP length scale remained at the lower bound, this agreement was not treated as strong evidence of a reliable optimum.

### Function 2

UCB with κ = 0.5 returned the previously submitted point, making it unsuitable as a new query.

UCB with κ = 1.5 and both EI settings selected the same more exploratory candidate:

[0.403694, 0.391789]

The disagreement between exploitative and exploratory settings, combined with the unreliable GP length scale, meant that GP evidence alone was insufficient to determine the next query.

### Function 3

UCB with κ = 0.5 remained relatively local, with a nearest-observation distance of approximately 0.0753.

Increasing the exploration parameter moved the recommendations progressively further from existing observations.

EI with ξ = 0% provided an intermediate candidate.

The recent new best provided some justification for prioritising the more local recommendation.

### Function 4

Both UCB candidates remained close to the newly successful region and predicted outputs above the current best.

In contrast, both EI settings selected a highly exploratory point approximately 0.5460 from the nearest observation.

This exploratory candidate had a very poor predicted mean but high uncertainty.

Given the substantial improvement achieved in the previous round, continued local refinement appeared better supported than moving to a distant, uncertain region.

### Function 5

Both UCB settings remained close to the successful upper-boundary region.

The EI recommendations moved to a much more distant and uncertain region.

None of the GP candidates predicted an improvement over the new best of 8662.482500.

However, the strong recent boundary result provided a clear reason to investigate nearby points further.

### Function 6

UCB with κ = 0.5 produced the most locally focused candidate and predicted a small improvement over the current best.

Increasing κ or ξ progressively increased candidate distance and uncertainty.

The locally focused recommendation was consistent with the strong observations already identified in this region.

### Function 7

UCB with κ = 0.5 effectively returned the existing best observation and was therefore unsuitable as a new query.

UCB with κ = 1.5 and both EI settings agreed on the same novel, more exploratory candidate.

This indicated that a novel point was required, although the model continued to favour the established high-performing region.

### Function 8

UCB with κ = 0.5 remained substantially more local than the other recommendations and predicted a small improvement over the new best.

Higher κ and both EI settings moved much further into unexplored regions.

The recent improvement provided support for considering the more conservative UCB recommendation.

### Interpretation

The GP results showed that the appropriate exploration level varied substantially between functions.

Functions 4 and 5 provided the clearest support for local refinement because both had recently achieved major improvements and their exploratory EI candidates had substantially poorer predicted means.

Functions 6 and 8 also showed evidence favouring local refinement.

Functions 1 and 2 required greater caution because their GP length scales had reached the lower bound.

Function 7 required a novel alternative because its most exploitative recommendation duplicated an existing observation.

Overall, acquisition scores were treated as one source of evidence rather than an automatic basis for final selection.

---

## Step 4 – Generate and Compare NN Candidates

The updated five-model NN ensembles were used to generate alternative candidates from the same 100,000-point candidate sets evaluated by the GPs.

Each NN recommendation was selected using the highest mean prediction across the ensemble.

The disagreement between ensemble members was recorded to assess prediction consistency, alongside the distance from the nearest existing observation.

Greater weight was assigned to NN recommendations for Functions 2 and 4–8 because these models passed LOOCV validation.

### Function 1

The NN selected:

[0.724967, 0.999094]

Predicted output: 0.001008  
Model disagreement: 0.000982

This candidate differed substantially from the GP recommendation.

Because the NN failed LOOCV validation, its prediction was not considered sufficiently reliable to justify selecting this point independently.

### Function 2

The NN selected:

[0.697103, 0.994806]

Predicted output: 0.696942  
Model disagreement: 0.049182

The candidate was approximately 0.0685 from an existing observation and differed substantially from the GP recommendations.

Although its prediction remained below the current best of 0.746962, the NN passed LOOCV and therefore provided useful alternative evidence given the GP's reliability problems.

### Function 3

The NN selected:

[0.437451, 0.559044, 0.487388]

Predicted output: -0.009572

This point was only approximately 0.0135 from the new best observation and suggested further local refinement.

However, because the NN again failed LOOCV, its recommendation was retained primarily for comparison with the GP.

### Function 4

The NN returned the existing new best observation:

[0.384332, 0.391111, 0.336887, 0.416488]

The ensemble disagreement was extremely low.

This reinforced the NN's preference for the successful region, but the exact point could not be submitted again.

A nearby novel candidate would therefore be required.

### Function 5

The NN selected:

[0.995347, 0.980915, 0.954873, 0.970449]

This exactly matched both UCB candidates.

The point was approximately 0.0574 from the successful [1, 1, 1, 1] observation.

The agreement between the GP and validated NN provided strong support for continued refinement of the upper-boundary region.

### Function 6

The NN selected:

[0.434907, 0.336577, 0.469677, 0.823758, 0.051909]

Predicted output: -0.345084

This would improve on the current best of -0.379740.

The candidate remained relatively local and was broadly consistent with the lower-exploration GP recommendations.

### Function 7

The NN returned the existing best observation, with very low model disagreement.

As with UCB κ = 0.5, this confirmed the model's preference for the established high-performing region.

However, a novel nearby point was required.

### Function 8

The NN selected a more exploratory point approximately 0.4480 from the current best.

Predicted output: 10.207430

This exceeded the current best of 9.930811.

The NN passed LOOCV, but its recommendation differed from the GP candidates.

Further comparison was therefore needed before deciding whether to favour the NN or the more local GP region.

### Interpretation

The NN results provided particularly strong support for the established successful regions of Functions 4, 5 and 7.

Functions 4 and 7 returned existing best observations, while Function 5 showed exact GP–NN agreement on a nearby novel point.

Function 6 also showed encouraging evidence, with the validated NN predicting an improvement in a region broadly consistent with the GP.

Function 8's NN predicted an improvement but selected a different region from the GP, requiring further comparison.

Functions 1 and 3 continued to have unvalidated NN models.

Function 2 remained difficult because its NN was validated but disagreed substantially with the unreliable GP.

The next step was therefore to quantify GP–NN agreement before examining neighbourhood evidence.

---

## Step 5 – GP–NN Candidate Comparison

The Euclidean distance between each NN recommendation and the four representative GP candidates was calculated.

Smaller distances indicated that the two surrogate approaches favoured similar regions, while larger distances indicated greater disagreement.

Agreement was interpreted alongside model reliability. Close recommendations were considered stronger evidence only where the underlying models had demonstrated useful predictive performance.

| Function | UCB κ=0.5 | UCB κ=1.5 | EI ξ=0% | EI ξ=5% |
|---|---:|---:|---:|---:|
| F1 | 0.6312 | 0.6312 | 0.6312 | 0.6312 |
| F2 | 0.8368 | 0.6706 | 0.6706 | 0.6706 |
| F3 | 0.0751 | 0.5999 | 0.5790 | 0.5999 |
| F4 | 0.0419 | 0.0591 | 0.9254 | 0.9254 |
| F5 | 0.0000 | 0.0000 | 1.3683 | 1.3683 |
| F6 | 0.0920 | 0.2137 | 0.1762 | 0.2137 |
| F7 | 0.0000 | 0.2220 | 0.2220 | 0.2220 |
| F8 | 0.3396 | 1.0053 | 1.0736 | 1.0736 |

### Interpretation by Function

**Function 1:** The closest GP–NN distance was 0.6312, indicating substantial disagreement. Because neither surrogate passed validation, this disagreement did little to resolve candidate selection.

**Function 2:** The closest distance was 0.6706, again indicating strong disagreement. The NN remained validated while the GP length scale had reached its lower bound, suggesting that the NN should receive relatively greater weight.

**Function 3:** The closest distance was 0.0751, showing strong local agreement with UCB κ = 0.5. However, the NN failed LOOCV, so this agreement was treated cautiously.

**Function 4:** The closest distance was 0.0419, showing strong agreement between the NN and locally focused UCB candidates. This supported continued refinement of the successful region.

**Function 5:** The NN exactly matched both UCB candidates, providing the clearest useful cross-model agreement of the round. The shared recommendation was novel and close to the successful upper boundary.

**Function 6:** The closest distance was 0.0920, indicating strong agreement between the validated NN and UCB κ = 0.5.

**Function 7:** The NN exactly matched UCB κ = 0.5, but both returned the existing best. This supported the region rather than providing a usable new query.

**Function 8:** The closest distance was 0.3396, indicating moderate disagreement. The NN was closest to UCB κ = 0.5, but the two approaches still preferred different points.

### Overall Interpretation

The comparison strengthened the case for local refinement in Functions 4–7.

Function 5 provided particularly strong evidence because the GP and NN independently selected exactly the same novel candidate near the recently successful boundary.

Function 6 also showed good cross-model support.

Function 3 showed local agreement, but the NN's validation failure limited the importance of this evidence.

Functions 1 and 2 showed substantial disagreement and remained unresolved.

Function 8 showed weaker agreement, suggesting that the final decision should consider recent performance and observed neighbourhood outputs.

---

## Step 6 – Neighbourhood Analysis

The three nearest existing observations to each representative GP and NN candidate were examined.

This allowed the model recommendations to be compared with actual observed performance in their surrounding regions.

The analysis included UCB κ = 0.5, UCB κ = 1.5, EI ξ = 0%, EI ξ = 5% and the NN candidate.

Neighbourhood evidence was considered alongside model validation, candidate novelty and recent optimisation performance.

### Function 1

Neither the GP nor NN region had meaningful local support.

The nearby observations were approximately zero, providing little evidence for distinguishing between the candidates.

This reinforced the need for additional investigation.

### Function 2

The nearest observed outputs around the NN candidate were:

- 0.611205
- 0.366010
- 0.507418

The nearest observed outputs around the exploratory GP candidate were:

- 0.214965
- -0.013858
- -0.065624

The NN region had substantially stronger observed support.

Given that the GP length scale had collapsed to the lower bound while the NN continued to outperform its LOOCV baseline, the NN candidate appeared better supported.

However, its predicted output remained below the existing best, so it represented refinement of a promising region rather than a confidently predicted improvement.

### Function 3

The nearest observed outputs around the local UCB candidate were:

- -0.010190
- -0.016451
- -0.020167

These included the newly established best result.

The more exploratory GP regions were associated with weaker observations.

This supported continued local refinement using UCB κ = 0.5.

### Function 4

The nearest observed outputs around the local UCB candidates were:

- 0.186912
- -0.200431
- -0.300645

The nearest observed outputs around the exploratory EI candidates were:

- -16.507159
- -7.966775
- -6.702089

The difference was substantial.

The local candidates remained close to the strongest observed outputs, while the exploratory EI candidates were associated with much poorer performance.

Continued local refinement was therefore strongly preferred.

### Function 5

The nearest observed outputs around the local UCB and NN candidates were:

- 8662.482500
- 6691.931674
- 7121.151955

The nearest observed outputs around the exploratory EI candidates were:

- 64.443440
- 78.434389
- 9.972332

This provided very strong evidence for continued local refinement.

The upper-boundary region contained the highest observed outputs, while the exploratory EI region had substantially weaker historical performance.

Combined with exact GP–NN agreement, the neighbourhood evidence strongly supported further investigation near [1, 1, 1, 1].

### Function 6

The nearest observed outputs around the local UCB candidate were:

- -0.379740
- -0.388687
- -0.458511

These included the current best and the recent Week 7 result.

The more exploratory candidates moved further from the strongest observed region.

The evidence therefore supported continued local GP/NN refinement.

### Function 7

The leading neighbourhood outputs were:

- 2.929370
- 2.536939
- 2.369289

Both local and moderately exploratory candidates remained associated with the established high-performing region.

However, the leading exploitative GP and NN recommendations duplicated the existing best, meaning a novel nearby alternative was required.

### Function 8

The nearest observed outputs around UCB κ = 0.5 were:

- 9.930811
- 9.821986
- 9.840530

The nearest observed outputs around the NN candidate were:

- 9.930811
- 9.911249
- 9.821986

Both candidates had strong neighbourhood evidence.

However, UCB κ = 0.5 remained closer to existing observations and represented a more conservative continuation of the recently successful region.

### Overall Interpretation

The neighbourhood analysis provided strong support for a locally focused strategy across most functions.

Functions 4 and 5 had the clearest evidence because their recent queries produced major improvements and their local candidates were surrounded by the strongest observed outputs.

Functions 3 and 6 also favoured continued refinement.

Function 7 required a novel local alternative because the strongest model recommendations returned existing observations.

Function 8 had good support for both the NN and local UCB regions, but the lower-κ UCB candidate was more conservative and better aligned with the recent successful query.

Function 2 favoured the NN region because its surrounding observations were stronger than those around the exploratory GP candidate.

Function 1 remained unresolved, as neither surrogate nor the nearby observations provided meaningful evidence for selecting a promising point.

---

## Step 6b – Additional Function 1 Model Diagnosis

Function 1 was investigated separately because neither surrogate model provided strong evidence for candidate selection.

The RBF GP again fitted a length scale at the lower bound of 1e-05, suggesting difficulty generalising between observations.

The NN also failed to outperform the mean-prediction baseline under LOOCV.

Furthermore, the GP and NN recommendations differed substantially, with a distance of 0.6312 between them.

Although all tested UCB and EI settings selected the same GP candidate, this stability was not considered reliable evidence because all four acquisition functions depended on the same problematic GP model.

### Alternative GP Kernel Investigation

To determine whether the poor behaviour was specific to the RBF kernel, four GP configurations were compared:

- Standard RBF
- Matérn 3/2
- Matérn 5/2
- RBF with a WhiteKernel noise component

Each model was assessed using LOOCV MAE against the same mean-prediction baseline.

| Kernel | LOOCV MAE | Baseline MAE | Beats Baseline |
|---|---:|---:|---|
| RBF | 0.001099 | 0.001050 | No |
| Matérn 3/2 | 0.001106 | 0.001050 | No |
| Matérn 5/2 | 0.001107 | 0.001050 | No |
| RBF + noise | 0.001099 | 0.001050 | No |

None of the alternative GP configurations outperformed the baseline.

The standard RBF produced the lowest validation error of the tested kernels, but its MAE of 0.001099 remained above the baseline value of 0.001050.

The Matérn alternatives performed slightly worse.

Adding a WhiteKernel noise component produced almost identical validation performance to the standard RBF.

The fitted kernels also continued to suggest very short-range behaviour.

Although the RBF fitted during this separate investigation produced a length scale of approximately 0.00409 rather than the lower-bound value observed earlier, this remained extremely small relative to the [0,1] search space.

The RBF + noise model also fitted an extremely small noise level, providing little evidence that observational noise was the main explanation for the modelling difficulty.

### Interpretation

The poor GP behaviour did not appear to be caused simply by the choice of RBF kernel.

Changing the assumed smoothness or explicitly modelling noise did not improve generalisation.

Combined with the NN's failure to outperform its baseline, neither surrogate provided reliable validation evidence for Function 1.

Consequently, the next query would not be selected solely from the predicted optimum of either model.

A model-independent space-filling investigation was therefore conducted to identify underexplored regions.

---

## Function 1 – Space-Filling Exploration

Because neither surrogate demonstrated reliable predictive performance, a space-filling approach was used as an additional source of evidence.

A set of 100,000 candidate points was generated across the [0,1]² search space.

For each candidate, the Euclidean distance to its nearest existing observation was calculated.

Candidates with the greatest minimum distance were identified as the most underexplored locations.

The five leading candidates were retained, with a minimum separation of 0.10 imposed between selected candidates to avoid returning several nearly identical points.

This approach did not attempt to predict where the function would be highest.

Instead, it prioritised sampling locations that would provide additional coverage of the search space when the surrogate models were unreliable.

### Space-Filling Results

| Candidate | Coordinates | Nearest-Observation Distance |
|---|---|---:|
| 1 | [0.998294, 0.000709] | 0.307658 |
| 2 | [0.684654, 0.001334] | 0.305975 |
| 3 | [0.006886, 0.695109] | 0.301267 |
| 4 | [0.786713, 0.000852] | 0.269276 |
| 5 | [0.898148, 0.005297] | 0.265794 |

The leading candidates were approximately 0.30 from their nearest existing observation.

By comparison, the GP recommendation was only approximately 0.0953 from an existing observation, while the NN recommendation was approximately 0.1442 away.

The space-filling alternatives therefore offered substantially greater novelty.

The strongest unexplored regions were concentrated around the boundaries of the search space, particularly where the second input was close to zero.

Candidate 1 had the greatest minimum distance, although Candidates 2 and 3 offered similar levels of novelty in different regions.

These points should not be interpreted as having higher predicted outputs.

Their value was that they would provide observations in regions about which the existing dataset contained relatively little information.

### Comparison of the Three Leading Candidates

The three strongest space-filling candidates were examined further.

Because their nearest-observation distances were similar, each was compared using nearby observed outputs and predictions from the existing GP and NN models.

The model predictions were treated as secondary evidence because neither surrogate passed validation.

| Candidate | Nearest Distance | GP Prediction | NN Prediction |
|---|---:|---:|---:|
| [0.998294, 0.000709] | 0.307658 | -0.000560 | 0.000593 |
| [0.684654, 0.001334] | 0.305975 | -0.000560 | 0.000432 |
| [0.006886, 0.695109] | 0.301267 | -0.000560 | -0.000247 |

The three candidates provided almost identical levels of space-filling exploration.

Their three nearest observed outputs were also approximately zero, meaning historical observations offered little basis for distinguishing between the regions.

The GP assigned the same predicted output and uncertainty to all three candidates.

This was consistent with its extremely short fitted length scale and reinforced its limited ability to discriminate between unexplored regions.

The NN provided somewhat greater differentiation, favouring Candidate 1, followed by Candidate 2.

However, these predictions remained unreliable because the NN failed LOOCV.

Candidate 1 also had greater NN disagreement than Candidates 2 and 3.

### Function 1 Decision

No candidate was clearly superior based on the combined evidence.

However, Candidate 1:

[0.998294, 0.000709]

provided the greatest model-independent space-filling value.

It also received the highest NN prediction of the three leading alternatives, offering limited secondary support.

The selection was therefore justified primarily by exploration rather than confidence in the surrogate predictions.

Candidate 1 was carried forward for final selection.

---

## Step 7 – Final Candidate Selection

One final point was selected for each function using the combined evidence from the Week 8 analysis.

The decisions considered:

- Recent optimisation performance
- GP reliability and fitted length scales
- UCB and EI acquisition behaviour
- NN LOOCV validation
- GP–NN agreement
- Candidate novelty
- Observed neighbourhood performance
- Function-specific exploration or exploitation requirements

Greater emphasis was placed on local refinement where recent observations had improved the objective and surrounding observations supported the same region.

More exploratory candidates were considered where the modelling evidence was weak.

Existing observations were excluded from final submission.

### Additional Local Searches

For Functions 4, 5 and 7, the evidence strongly favoured established high-performing regions.

However, some of the leading model recommendations corresponded to existing observations.

Additional local searches were therefore performed around the current best observation for these functions.

Novel candidate points were generated within small neighbourhoods, constrained to the [0,1] search space and evaluated using the GP and validated NN models.

#### Function 4

The local search selected:

[0.386248, 0.381010, 0.343907, 0.423092]

This was approximately 0.0141 from the current best observation.

Both the GP and NN continued to predict strong performance in this region.

The candidate therefore allowed further refinement without repeating the previous query.

#### Function 5

The initial local search selected:

[1.000000, 1.000000, 0.999995, 1.000000]

This point was technically novel but only 0.000005 from the existing best [1, 1, 1, 1].

It was considered too close to provide meaningful additional information.

A minimum nearest-observation distance of 0.03 was therefore introduced, and the local search was repeated.

The constrained search selected:

[1.000000, 1.000000, 0.970000, 1.000000]

GP predicted output: 8391.49  
NN predicted output: 8342.42  
Current best: 8662.48

Both models continued to predict strong performance near the upper boundary.

The novelty constraint ensured that the query was more informative than an almost identical resubmission of the current best.

#### Function 7

The local search selected:

[0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025]

This was approximately 0.0479 from the current best observation.

GP predicted output: 2.920647  
NN predicted output: 2.936110

Both models predicted performance close to the existing best of 2.929370.

The candidate therefore provided a novel opportunity to refine the established high-performing region.

---

## Final Candidate Decisions

### Function 1 – Space-Filling Exploration

**Selected candidate:**

[0.998294, 0.000709]

Neither the GP nor NN demonstrated reliable predictive performance, and alternative GP kernels also failed to outperform the baseline.

The selected point provided the greatest space-filling value, with a nearest-observation distance of approximately 0.3077.

The NN provided limited secondary support, but the primary justification was model-independent exploration.

### Function 2 – NN-Supported Region

**Selected candidate:**

[0.697103, 0.994806]

The GP was given reduced weight because its fitted length scale reached the lower bound.

The NN continued to outperform the mean baseline under LOOCV and identified this candidate as its preferred region.

Nearby observed outputs also provided stronger support than those surrounding the exploratory GP alternatives.

The selection therefore prioritised the better-supported NN region.

### Function 3 – Local GP Refinement

**Selected candidate:**

[0.463879, 0.497888, 0.452731]

The previous query produced a new best of -0.010190, supporting continued local refinement.

Because the NN failed LOOCV, greater weight was given to the GP.

The UCB κ = 0.5 candidate remained relatively close to the successful region and had strong nearby observed outputs.

### Function 4 – Local Refinement

**Selected candidate:**

[0.386248, 0.381010, 0.343907, 0.423092]

The previous observation produced a substantial new best of 0.186912.

The selected local candidate remained close to this successful point while avoiding an exact resubmission.

Both models continued to predict strong performance in the region.

### Function 5 – Local Boundary Refinement

**Selected candidate:**

[1.000000, 1.000000, 0.970000, 1.000000]

The previous query at [1, 1, 1, 1] produced a major new best of 8662.482500.

The selected point remained close to the successful upper boundary while maintaining a minimum distance of 0.03 from the existing best.

Both GP and NN continued to predict high outputs, supporting further local investigation.

### Function 6 – Local GP/NN Refinement

**Selected candidate:**

[0.426973, 0.384648, 0.468006, 0.752822, 0.084308]

The UCB κ = 0.5 candidate remained within the strongest observed region and had a GP prediction slightly above the current best.

The validated NN also favoured a nearby region.

The combined evidence supported continued local refinement rather than substantially greater exploration.

### Function 7 – Local Novel Refinement

**Selected candidate:**

[0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025]

Both models supported remaining close to the current best, but their leading recommendations included an existing observation.

The additional local search identified a novel candidate approximately 0.0479 from the current best.

The GP and NN predicted outputs of 2.920647 and 2.936110 respectively, providing good cross-model support.

### Function 8 – Local UCB Refinement

**Selected candidate:**

[0.026226, 0.080912, 0.128250, 0.056836, 0.748250, 0.399531, 0.253527, 0.720750]

The previous query produced a new best of 9.930811.

The UCB κ = 0.5 candidate provided a more local continuation of the successful region than the more exploratory alternatives.

Its surrounding observations included several of the strongest outputs recorded for Function 8.

The candidate was therefore preferred over the more exploratory NN and GP alternatives.

---

## Step 8 – Final Validation and Submission

Before submission, the eight selected candidates were checked for:

1. Correct dimensionality for each function.
2. Input values within the permitted [0,1] bounds.
3. Exact duplicates of existing observations.
4. Distance from the nearest existing observation.
5. Correct formatting to six decimal places.

All eight final candidates passed the validation checks.

Each candidate had the correct dimensionality, remained within the permitted search space and was distinct from the existing observations.

Nearest-observation distances ranged from approximately 0.0141 for Function 4 to 0.3077 for Function 1.

This reflected the different strategies adopted across the functions.

Functions 4, 5 and 7 deliberately remained close to established high-performing observations because local refinement was strongly supported.

Function 1 was much further from its existing observations because model-independent exploration was prioritised.

### Final Week 8 Submission Points

```text
Function 1: 0.998294-0.000709
Function 2: 0.697103-0.994806
Function 3: 0.463879-0.497888-0.452731
Function 4: 0.386248-0.381010-0.343907-0.423092
Function 5: 1.000000-1.000000-0.970000-1.000000
Function 6: 0.426973-0.384648-0.468006-0.752822-0.084308
Function 7: 0.157801-0.305871-0.468629-0.230835-0.339422-0.619025
Function 8: 0.026226-0.080912-0.128250-0.056836-0.748250-0.399531-0.253527-0.720750
```

---

## Week 8 Reflection

Week 8 produced several useful findings about the behaviour of the surrogate models and the importance of function-specific optimisation strategies.

The four improvements achieved in Week 7 provided strong evidence for continued local refinement in several functions, particularly Functions 4 and 5.

Function 5 was especially informative because the previous boundary query produced a substantial improvement, while both GP and NN models continued to identify the same nearby high-performing region.

The additional novelty constraint introduced during local refinement was also important. Although a point differing from the current best by only 0.000005 would technically satisfy the no-duplicate requirement, it would provide very little additional information. Requiring a distance of 0.03 produced a more meaningful query while preserving the focus on the successful boundary region.

Function 1 remained the most challenging optimisation problem. Neither the GP nor NN passed validation, and changing the GP kernel did not improve generalisation. The additional space-filling analysis therefore provided a more defensible basis for candidate selection than relying on unreliable surrogate predictions.

Function 2 also became more difficult after its GP length scale reached the lower bound. However, its NN continued to outperform the LOOCV baseline, and the NN candidate had stronger surrounding observations than the exploratory GP alternative. This demonstrated the value of using independent model evidence when one surrogate becomes unreliable.

For Functions 3, 4, 6, 7 and 8, recent results and neighbourhood evidence generally favoured continued refinement of established promising regions rather than substantial exploration.

An important methodological consideration remains the interpretation of GP–NN agreement. Although agreement can strengthen confidence when both models are reliable, Function 2's previous unsuccessful query demonstrated that agreement does not guarantee improvement. Similarly, exact agreement on an existing observation does not automatically provide a suitable novel query, as seen for Functions 4 and 7.

The NN architecture also remains a consistency consideration. The Week 7 implementation used two hidden layers of 20 neurons rather than the 16-neuron layers used in earlier rounds, and this configuration was retained in Week 8. Consequently, comparisons of NN performance across the full capstone should acknowledge that the architecture was not identical throughout. Keeping the configuration consistent from Week 7 into Week 8 nevertheless supports a more direct comparison between these two rounds.

Overall, Week 8 reinforced the importance of combining model predictions with validation, observed neighbourhood performance, candidate novelty and recent optimisation outcomes. The final selections reflected the evidence available for each function rather than applying a single acquisition strategy across all eight problems.

