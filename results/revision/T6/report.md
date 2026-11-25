# CADR T6 P2a paired evaluation

P2a compares frozen controllers. The manifest lists completed batches, source and input hashes, and retry counts.

| Task | Training seed | Receipt | GPU |
| --- | ---: | --- | --- |
| hopper | 101 | complete | Tesla T4 |
| hopper | 102 | complete | Tesla T4 |
| hopper | 103 | complete | Tesla T4 |
| hopper | 104 | complete | Tesla T4 |
| hopper | 105 | complete | Tesla T4 |
| pickcube | 101 | complete | Tesla T4 |
| pickcube | 102 | complete | Tesla T4 |

The absolute tables show unscaled Hopper return and PickCube success percentages. The paired tables include
every seed-specific difference, secondary PickCube endpoints and the three complementary oracle comparisons.
Hopper core comparisons use five training seeds. Hopper extras and PickCube use two. Each seed has 20 paired episodes.
Equal-weight seed means and 95% percentile intervals use 10,000 seed-first resamples with seed 20261124.
Incomplete or incompatible cohorts appear as separate strata. Single-seed strata have descriptive values only.
The oracle receives true age during evaluation. Inspect negative oracle gaps as recorded.

The clean gate compares saved return, length and applicable success arrays bitwise. The documented exceptions
are listed in `clean_checks.csv`. The evaluator's separate entry-point equivalence check covers actions.
`timing.csv` gives per-method, scenario and training seed synchronized CUDA `controller.act` p50/p95;
`fixed_prior_matched` uses a static reference distribution and its fallback rate is in that file.

`historical_d4rl_hopper.csv` rescales raw return using the old D4RL reference values −20.272305 and 3234.3.
Hopper-v5 same-environment normalization remains NA. Empty CSV cells denote unavailable estimates.

## Decision status — reviewable

- Route A criteria met: False.
- Long-delay censoring title criterion met: False.
- Route B discussion flag: False.
- Oracle gaps are descriptive. The design did not specify a numerical threshold for a large gap.

The following five-seed Hopper contrasts use raw return units.

| Priority scenario | Hazard − legacy | Hazard − in-order | Hazard − uncensored | Oracle − hazard |
| --- | ---: | ---: | ---: | ---: |
| drifting_to_long_delay | -34.782 [-203.034, 123.507] | 125.149 [-264.754, 464.437] | 59.148 [-83.591, 189.178] | 514.069 [289.659, 757.225] |
| markov_switching | 70.397 [-82.874, 219.191] | -985.499 [-1270.335, -686.205] | 26.228 [-100.319, 146.268] | 1531.419 [1270.703, 1771.404] |
| ood_support_shift | 230.973 [-22.805, 462.753] | -152.773 [-367.250, 81.722] | 2.454 [-123.644, 109.900] | 925.404 [351.285, 1490.789] |
| ood_changepoint | 435.664 [224.717, 640.853] | -412.694 [-663.679, -177.532] | 16.465 [-116.432, 134.582] | 1320.680 [1147.458, 1484.424] |

PickCube supports directional comparisons only. See `decision.json` and the full paired tables.
- hopper: 168 absolute rows and 312 paired rows.
- pickcube: 168 absolute rows and 261 paired rows.
