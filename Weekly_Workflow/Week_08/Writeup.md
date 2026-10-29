
# Module 19 – Week 8

## Methodology

Week 8 built on the eight new observations from Week 7. The process was to (1) assess improvements, (2) refit and validate GP and NN surrogates, (3) generate UCB and EI candidates, (4) compare NN recommendations, (5) measure GP–NN agreement, (6) inspect nearby observed results and investigate Function 1 separately, (7) select one novel candidate per function, and (8) validate and format submissions.

The GP analysis used 100,000 candidate points per function. UCB was assessed at κ = 0.5 and 1.5, and EI at ξ = 0% and 5% of each function's observed output standard deviation. Five-model NN ensembles supplied alternative recommendations; their reliability was assessed with leave-one-out cross-validation (LOOCV) against a mean-prediction baseline. Recommendations were judged using predicted output, uncertainty, model reliability, proximity to existing observations and recent performance, rather than acquisition scores alone.

## 1. Week 7 Results

| Function | Previous best | New output | Improved? |
|---|---:|---:|---|
| F1 | 0.000000 | 0.000000 | No |
| F2 | 0.746962 | 0.569904 | No |
| F3 | -0.015498 | -0.010190 | Yes |
| F4 | -0.200431 | 0.186912 | Yes |
| F5 | 7121.151955 | 8662.482500 | Yes |
| F6 | -0.379740 | -0.388687 | No |
| F7 | 2.929370 | 2.369289 | No |
| F8 | 9.911249 | 9.930811 | Yes |

Four functions improved. F4 and F5 made the largest gains, supporting further refinement of their successful regions; F3 and F8 improved more modestly. F6 remained close to its best, while F2 and F7 declined. F1 again returned effectively zero. These differences reinforced the need for function-specific strategies.

## 2. Surrogate Diagnostics

### Gaussian Processes

| Function | RBF length scale | Prediction range | Uncertainty range |
|---|---:|---:|---:|
| F1 | 0.00001 | 0.000560 | 0.001570 |
| F2 | 0.00001 | 0.651312 | 0.244076 |
| F3 | 0.199 | 0.388723 | 0.078255 |
| F4 | 0.416 | 35.821887 | 7.855238 |
| F5 | 0.273 | 7847.668688 | 2510.829993 |
| F6 | 0.581 | 2.303126 | 0.523703 |
| F7 | 0.302 | 3.026197 | 0.818402 |
| F8 | 1.04 | 4.878650 | 0.653663 |

F1 remained at the length-scale lower bound of 1e-05. F2 also reached this bound, falling from 0.0431 the previous week, making both GPs questionable for generalisation. Although F2 retained a broad prediction range, its maximum predicted output was below the observed best of 0.746962. The other fitted length scales remained more stable. F5's decreased from 0.363 to 0.273 following its strong boundary observation. F4 and F5 had substantial uncertainty ranges, making acquisition-based selection relevant.

### Neural Networks

| Function | Mean training loss | NN LOOCV MAE | Mean baseline MAE | NN beats baseline? |
|---|---:|---:|---:|---|
| F1 | 0.000963 | 0.001704 | 0.001050 | No |
| F2 | 0.010989 | 0.219676 | 0.237993 | Yes |
| F3 | 0.000408 | 0.062478 | 0.051662 | No |
| F4 | 0.000284 | 1.825570 | 7.284354 | Yes |
| F5 | 0.000230 | 230.163312 | 2049.255132 | Yes |
| F6 | 0.000370 | 0.241315 | 0.513894 | Yes |
| F7 | 0.000338 | 0.639930 | 0.639930 | Yes |
| F8 | 0.000381 | 0.354980 | 0.962395 | Yes |

Six of eight NN ensembles beat the baseline. F4–F8 generally offered useful supporting evidence, and F2 passed narrowly despite an increased training loss (0.010989 versus 0.006460 previously). F1 and F3 failed validation, so their NN recommendations received limited weight. Low training losses were not treated as evidence of generalisation without LOOCV support.

## 3. Candidate Comparison

The main UCB, EI and NN findings were:

| Function | Main evidence from GP and NN candidates |
|---|---|
| F1 | All four GP settings selected [0.094177, 0.975622], but the GP was unreliable. NN preferred [0.724967, 0.999094] and failed LOOCV. |
| F2 | UCB κ=0.5 repeated an existing query; UCB κ=1.5 and both EI settings selected [0.403694, 0.391789]. Validated NN preferred [0.697103, 0.994806]. |
| F3 | UCB κ=0.5 stayed local (nearest-observation distance 0.0753). NN favoured a nearby point but failed LOOCV. |
| F4 | Both UCB settings remained near the new best; EI explored a distant region (distance 0.5460) with poor predicted output. NN returned the existing best. |
| F5 | Both UCB settings and the NN selected [0.995347, 0.980915, 0.954873, 0.970449]. EI favoured a distant, uncertain region. |
| F6 | UCB κ=0.5 predicted a small improvement locally; validated NN also favoured a nearby region. |
| F7 | UCB κ=0.5 and NN returned the existing best; higher-exploration GP settings agreed on a novel alternative. |
| F8 | UCB κ=0.5 remained local and predicted a small improvement. Validated NN predicted 10.207430 but favoured a more exploratory point. |

### GP–NN Agreement

Euclidean distances between the NN candidate and each GP candidate were:

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

F4–F6 showed strong useful agreement around successful local regions, especially F5's exact agreement on a novel candidate. F7's exact agreement was less actionable because both methods returned an existing observation. F3 showed local agreement but its NN failed LOOCV. F1 and F2 disagreed substantially; F8 showed moderate disagreement. Agreement was therefore treated as supporting evidence, not proof that a candidate would improve the objective.

## 4. Neighbourhood Evidence

The three nearest observed outputs to the leading candidates helped distinguish local refinement from more exploratory alternatives:

| Function | Key nearby observations and interpretation |
|---|---|
| F1 | Nearby observations were approximately zero around both GP and NN recommendations; neither region was convincingly supported. |
| F2 | NN region: 0.611205, 0.366010, 0.507418; exploratory GP region: 0.214965, -0.013858, -0.065624. NN region had stronger observed support. |
| F3 | Local UCB: -0.010190, -0.016451, -0.020167, supporting refinement around the new best. |
| F4 | Local UCB: 0.186912, -0.200431, -0.300645; exploratory EI: -16.507159, -7.966775, -6.702089. Strong preference for local refinement. |
| F5 | Local UCB/NN: 8662.482500, 6691.931674, 7121.151955; exploratory EI: 64.443440, 78.434389, 9.972332. Very strong local evidence. |
| F6 | Local UCB: -0.379740, -0.388687, -0.458511. Strongest observations remained nearby. |
| F7 | Leading neighbourhood: 2.929370, 2.536939, 2.369289. A novel point near the best was preferable. |
| F8 | Local UCB: 9.930811, 9.821986, 9.840530; NN: 9.930811, 9.911249, 9.821986. Both supported, but UCB was more local. |

The strongest case for exploitation was in F4 and F5, where recent improvements, cross-model evidence and nearby observed outputs aligned. F2's validated NN and neighbourhood results were more informative than its unreliable GP. F1 remained unresolved and required a separate investigation.

## 5. Additional Function 1 Investigation

Because F1's GP length scale again reached the lower bound and its NN failed LOOCV, alternative GP kernels were tested to see whether the RBF choice explained the problem.

| Kernel | LOOCV MAE | Mean baseline MAE | Beats baseline? |
|---|---:|---:|---|
| RBF | 0.001099 | 0.001050 | No |
| Matérn 3/2 | 0.001106 | 0.001050 | No |
| Matérn 5/2 | 0.001107 | 0.001050 | No |
| RBF + WhiteKernel | 0.001099 | 0.001050 | No |

None outperformed the baseline. The separate RBF fit produced a length scale of approximately 0.00409, still very short relative to the [0,1] domain; the noise component was also negligible. Changing the kernel did not resolve the generalisation problem.

### Space-Filling Search

A model-independent search across 100,000 points in [0,1]² identified candidates maximising distance from the nearest existing observation. Five leading points were retained with a minimum separation of 0.10 between candidates.

| Rank | Candidate | Nearest-observation distance |
|---|---|---:|
| 1 | [0.998294, 0.000709] | 0.307658 |
| 2 | [0.684654, 0.001334] | 0.305975 |
| 3 | [0.006886, 0.695109] | 0.301267 |
| 4 | [0.786713, 0.000852] | 0.269276 |
| 5 | [0.898148, 0.005297] | 0.265794 |

The leading space-filling points were much more novel than the GP (distance 0.0953) and NN (0.1442) recommendations. The three strongest were compared further:

| Candidate | GP prediction | NN prediction |
|---|---:|---:|
| [0.998294, 0.000709] | -0.000560 | 0.000593 |
| [0.684654, 0.001334] | -0.000560 | 0.000432 |
| [0.006886, 0.695109] | -0.000560 | -0.000247 |

All three had approximately zero nearby observed outputs, and the GP could not distinguish between them. The NN weakly favoured the first candidate, but its validation failure limited confidence in that prediction. **[0.998294, 0.000709]** was selected primarily because it maximised space-filling coverage, with only limited secondary NN support.

## 6. Final Candidate Selection

| Function | Selected point | Decision and justification |
|---|---|---|
| F1 | [0.998294, 0.000709] | Space-filling exploration; neither surrogate validated, and alternative kernels did not help. |
| F2 | [0.697103, 0.994806] | NN-supported region; NN passed LOOCV and had stronger nearby outputs than the unreliable GP alternatives. |
| F3 | [0.463879, 0.497888, 0.452731] | Local UCB κ=0.5 refinement after a new best; NN failed LOOCV. |
| F4 | [0.386248, 0.381010, 0.343907, 0.423092] | Novel local refinement near the substantial new best, supported by both models. |
| F5 | [1.000000, 1.000000, 0.970000, 1.000000] | Upper-boundary refinement with a 0.03 novelty constraint to avoid a near-duplicate. |
| F6 | [0.426973, 0.384648, 0.468006, 0.752822, 0.084308] | Local UCB κ=0.5 selection supported by validated NN and nearby observations. |
| F7 | [0.157801, 0.305871, 0.468629, 0.230835, 0.339422, 0.619025] | Novel local search around the best after GP and NN returned an existing point. |
| F8 | [0.026226, 0.080912, 0.128250, 0.056836, 0.748250, 0.399531, 0.253527, 0.720750] | Local UCB κ=0.5 continuation after a new best, supported by strong neighbourhood observations. |

### Local-Search Checks

For F4, the additional local search found a candidate approximately 0.0141 from the current best. For F7, the selected candidate was approximately 0.0479 from the best, with GP and NN predictions of 2.920647 and 2.936110 respectively.

For F5, an initial candidate [1.000000, 1.000000, 0.999995, 1.000000] was technically novel but only 0.000005 from the existing [1,1,1,1] observation. A minimum distance of 0.03 was imposed, giving the final point [1.000000, 1.000000, 0.970000, 1.000000]. The GP and NN predicted 8391.49 and 8342.42 respectively, below the best of 8662.48 but still strongly supporting the boundary region.

## 7. Validation and Submission

All eight candidates were checked for correct dimensionality, [0,1] bounds, duplicates and nearest-observation distances, then formatted to six decimal places. All passed. Distances ranged from approximately 0.0141 (F4, local refinement) to 0.3077 (F1, exploration).

    Function 1: 0.998294-0.000709
    Function 2: 0.697103-0.994806
    Function 3: 0.463879-0.497888-0.452731
    Function 4: 0.386248-0.381010-0.343907-0.423092
    Function 5: 1.000000-1.000000-0.970000-1.000000
    Function 6: 0.426973-0.384648-0.468006-0.752822-0.084308
    Function 7: 0.157801-0.305871-0.468629-0.230835-0.339422-0.619025
    Function 8: 0.026226-0.080912-0.128250-0.056836-0.748250-0.399531-0.253527-0.720750

## Reflection

Week 8 reinforced the value of adapting the exploration–exploitation balance to each function. Recent gains and local observations favoured refinement for F3–F8, particularly F4 and F5. The F5 novelty constraint was an important practical improvement: a technically new point almost identical to the best observation would add little information.

F1 was the clearest exception. Both surrogates failed validation, alternative GP kernels offered no improvement, and space-filling exploration provided a more defensible basis for the next query. F2 showed why a validated alternative surrogate and neighbourhood evidence can be useful when a GP becomes unreliable.

Cross-model agreement was informative but not decisive: it could identify a promising region without guaranteeing an improvement, and sometimes both models simply returned an existing observation. One consistency limitation also remains: the NN implementation used two 20-neuron hidden layers in Weeks 7–8, compared with 16-neuron layers in earlier weeks. Consequently, NN performance differences across the entire capstone cannot be attributed solely to additional observations; the architecture should be standardised in subsequent rounds.

