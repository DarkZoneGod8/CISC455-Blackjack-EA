# Evolving a Blackjack Strategy with an EA

CISC 455/851 Evolutionary Optimization group project

The evolutionary algorithm (EA) learns card-counting tags and bet sizes for six-deck Blackjack. Fixed Basic Strategy governs every playing decision. The experiments compare evolved policies with one-unit flat betting and standard Hi-Lo, and test how the risk penalty and crossover operator affect performance.

## Project structure

| Path | Purpose |
| --- | --- |
| `blackjack_env.py` | Seeded shoe generation, Basic Strategy, true count, betting, settlement, and bankroll tracking |
| `ea_solver.py` | Chromosome decoding, fitness, tournament selection, crossover, mutation, and elitism |
| `experiments.py` | Multi-seed training, validation, held-out comparisons, and output generation |
| `requirements.txt` | NumPy and Matplotlib dependencies |
| [`results/`](results/) | Committed CSVs, PNG plots, and the experiment configuration |

## Install and reproduce

Use Python 3.10 or newer. Install NumPy and Matplotlib from the repository root.

```bash
pip install -r requirements.txt
```

Run the full experiment suite to regenerate the CSVs and PNGs in `results/`.

```bash
python experiments.py --output results
```

The full run uses seeds 0–9, 50 generations, 60 individuals, and 150 fresh shoes per generation. The three configurations compare risk-neutral and risk-averse fitness under arithmetic crossover, then compare arithmetic and discrete crossover under the same risk-averse objective. Validation shoes select the champion before testing on 2,000 unseen shoes, grouped into ten sessions of 200 shoes.

Use separate output directories for a quick pipeline check or an optional self-adaptive run:

```bash
python experiments.py --quick --output quick-results
python experiments.py --self-adaptive --output adaptive-results
```

## EA design

| Component | Implementation |
| --- | --- |
| Representation | 14 real genes encode 10 rank tags and four bet multipliers. Nonzero tags are normalized to an L1 norm of 10. Bets are rounded to units and mapped to true-count brackets below 1, [1, 2), [2, 3), and 3 or above. |
| Fitness evaluation | Mean net profit per allocated shoe minus λ times its standard deviation and a 5-unit penalty for reaching zero bankroll. The ablation uses λ = 0 or 0.25. Each session starts at 200 units and stops below one unit. Common Random Numbers give all candidates in a generation the same shuffled shoes. |
| Selection | Size-three tournaments sampled with replacement select parents. The two highest-scoring individuals survive unchanged. |
| Crossover | Whole arithmetic crossover uses one interpolation coefficient for all genes in a child. Discrete crossover independently chooses a parent for each gene. |
| Mutation | Gaussian noise perturbs each gene before clipping to its bounds. The default standard deviation is 0.35. Self-adaptive mode evolves a separate step size for each gene. |

The simulator reshuffles between hands once 78 cards or fewer remain. The dealer stands on soft 17, naturals pay 3:2, doubling after a split is allowed, and only one split is permitted per round. Each split ace receives one extra card. [Multi-deck Basic Strategy reference](https://thirdbase21.com/basic-strategy-chart/)

## Results

- [Convergence plot](results/convergence.png) and [CSV](results/convergence.csv) show the mean and standard deviation of best and average training fitness across seeds.
- [Baseline plot](results/baseline.png) and [CSV](results/baseline.csv) compare the evolved champion, flat betting, and Hi-Lo. The [policy table](results/policies.csv) records the tags and bet ramps.
- [Ablation plot](results/ablation.png) and [per-seed CSV](results/ablation_runs.csv) show how risk and crossover choices affect held-out profit and ruin. [Run configuration](results/run_config.json) records the seeds and selected genotype.

Profit statistics include zero profit for shoes left unplayed after ruin. Win rate is winning hands divided by settled hands, including pushes and split hands. Ruin rate is the fraction of sessions ending below one unit.
