# Evolving a Blackjack Strategy with an EA

CISC 455/851 Evolutionary Algorithms course project

This project evolves the playing decisions in a Blackjack strategy from a population of random lookup tables. Each candidate starts out choosing Hit or Stand with equal probability in every table cell. Selection favors candidates with higher simulated net return per hand. A fixed, Hit/Stand-only Basic Strategy table provides a benchmark and a match-rate diagnostic.

| File | Purpose |
| --- | --- |
| `blackjack_env.py` | Finite-shoe simulator, rule configuration, shared-shoe evaluation and Basic Strategy reference |
| `ea_solver.py` | Random initialization, selection, crossover, mutation, elitism and optional validation |
| `experiments.py` | Independent seeded runs, held-out comparisons, CSV exports and PNG plots |
| `requirements.txt` | NumPy and Matplotlib dependencies |

## Run locally

Use Python 3.10 or later. Run these commands from the project directory.

```bash
pip install -r requirements.txt
python experiments.py
```

The default experiment uses 256 candidates, 35 generations, 1,500 training hands per candidate per generation, three independent runs, 12,000 validation hands and 50,000 held-out hands per run. Each run writes to its own folder under `results/`.

For a smaller experiment or a single training run, use the following commands.

```bash
python experiments.py --pop-size 32 --num-generations 5 --num-hands 300 --num-runs 2 --validation-hands 1000 --test-hands 2500 --output results/quick
python ea_solver.py --seed 42 --num-generations 10 --print-pair-tables
python experiments.py --help
```

`ea_solver.py` uses the same experiment pipeline and defaults to one seeded run.

## Run in Google Colab

Upload or extract all five project files into the same Colab directory and use `%cd` to enter it. Run this cell.

```python
%pip install -r requirements.txt
!python experiments.py
```

Download the output folder from Colab before ending the session. Matplotlib saves figures with a noninteractive backend.

## Evolution and evaluation

| Component | Implementation |
| --- | --- |
| Representation | A 26 × 10 binary table. Rows cover hard totals 4–20 and soft totals 12–20. Columns cover dealer upcards 2–10 and Ace. Each bit independently starts as Hit (0) with probability `--initial-hit-probability` (default 0.5) and Stand (1) otherwise. Totals of 21 stand automatically, and busts and naturals settle without a policy decision. |
| Fitness | Default `mean_return` uses net return in wager units per hand. `risk_adjusted` subtracts `--risk-penalty` (default 0.1) times the standard deviation of per-hand payouts in the evaluation batch. Pushes are worth zero. Common Random Numbers (CRN) give every candidate in a generation the same shuffled shoe sequence. Different decisions can shift later hand boundaries. |
| Parent selection | Default `tournament` picks the best of `--tournament-size` draws with replacement (default four) for each parent. `rank` samples parents with probability proportional to their fitness rank, giving weight 1 to the worst and population size to the best. |
| Crossover | With probability 0.9 per offspring, uniform crossover chooses a parent independently per table entry. With `--crossover-type one_point`, a random cut in the flattened table joins a prefix from the first parent with a suffix from the second. Without crossover, the child starts as a copy of the first parent. |
| Mutation | Default `bit_flip` flips each child entry independently with probability `--mutation-rate` (default 0.01). `fixed_count` flips distinct random entries per child. Its count is the chromosome length times the mutation rate, rounded randomly between the adjacent integers to preserve the expected count. |
| Survivor selection | Default `elitism` carries the `--elitism-count` highest-scoring strategies forward unchanged (default six) and fills the remaining slots with offspring. `offspring_only` fills every slot with a child. The entire next generation faces a fresh shoe batch. |
| Validation | Each generation's training leader is saved. A separate shared batch scores these leaders with the selected fitness objective and chooses the final table. Use `--validation-hands 0` to use the last generation's leader. |
| Held-out test | Each run compares its selected table with Basic Strategy and Random Play on an independent set of shared shoes. Random Play chooses Hit or Stand independently and uniformly at each decision. |

Generation 1 measures the random starting population. Training scores can fluctuate because each generation gets a fresh batch. Agreement gives each table cell equal weight, including rare states. The reference and agreement scores never affect parent selection, mutation, or validation. Held-out baseline comparisons always report mean net return per hand, giving both fitness objectives the same evaluation metric.

## Rules and configuration

The default game uses six decks, a one-unit flat wager, a 75% penetration cutoff and dealer stand on soft 17. Naturals pay 3:2, dealer blackjack settles before player actions, and ordinary wins and losses are one unit. The game offers Hit and Stand, with no Split, Double, insurance or surrender. A shoe is replaced between rounds once the cutoff is reached. At high penetration, the simulator can reshuffle earlier to reserve enough cards to finish a round.

Set any EA parameter with its corresponding flag, or run `python experiments.py --help` to see the complete list. Each operator and fitness selector has two options.

| Flag | Default | Second option |
| --- | --- | --- |
| `--parent-selection` | `tournament` | `rank` |
| `--crossover-type` | `uniform` | `one_point` |
| `--mutation-type` | `bit_flip` | `fixed_count` |
| `--survivor-selection` | `elitism` | `offspring_only` |
| `--fitness-objective` | `mean_return` | `risk_adjusted` |

Numeric settings include `--pop-size`, `--num-generations`, `--tournament-size`, `--crossover-rate`, `--mutation-rate`, `--elitism-count`, `--num-hands`, `--validation-hands`, `--initial-hit-probability` and `--risk-penalty`. Use `--num-runs` for independent runs, `--seed` to reproduce the seed tree and `--test-hands` for the held-out sample size. Rerun with the same settings and dependency versions to reproduce the seeded results.

Use `--num-decks`, `--penetration` (a fraction of the shoe), `--hit-soft-17`, `--blackjack-payout`, `--win-payout` and `--loss-payout` to change the rules. Payout flags take nonnegative magnitudes. Basic Strategy stays fixed as a multi-deck benchmark under custom rules. Python callers can pass `BlackjackConfig` and `EAConfig` directly to `evolve()`.

## Parameter sweeps

Every EA flag has a corresponding `--sweep-...` flag. Pass values as space-separated lists. By default, the experiment runs the base configuration and then changes one parameter at a time. This command compares offspring-only replacement against the base of six elites and also tests keeping 12 elites. Each of the three configurations runs on the same three seeds.

```bash
python experiments.py --pop-size 60 --num-runs 3 --sweep-survivor-selection elitism offspring_only --sweep-elitism-count 6 12
```

Add `--sweep-mode grid` to test every combination of the supplied values. Parameters without a list retain their base settings, and duplicate configurations run once. Choose the relevant base mode when sweeping a mode-specific parameter. For example, use `--fitness-objective risk_adjusted --sweep-risk-penalty 0.05 0.1 0.2` to compare risk penalties. Tournament size applies to `tournament`, and elite count applies to `elitism`. Python callers can build the same configurations using `experiments.build_sweep(base_config, {"pop_size": [40, 60, 120]})`.

## Champion decision tables

Use `--print-pair-tables` to print the champion immediately after every generation and once after final champion selection. From Python, pass `log_pair_tables=True` to `evolve()`. Experiment runs always export the same tables to `champion_pair_decisions.csv`.

The main table lists every hard or soft total against every dealer upcard. The equal-rank section shows how two-card starting pairs map to that table, such as A+A to soft 12 and 8+8 to hard 16. Tens and face cards share the same card value. The actions in both sections are Hit or Stand. CSV records are marked `generation` or `final`, so a validation-selected champion can be traced to its original generation.

## Results

Run folders have names like `results/run_pop60_k3_h1500_r1_20261119_150000_a1b2c3d4/`. A UTC timestamp and a unique suffix keep repeated launches from overwriting earlier runs. Use `--output` to choose a different parent folder.

| Output | Contents |
| --- | --- |
| `champion_pair_decisions.csv` | All generation champions and the final champion, including the state rows and the two-card pair rows |
| `convergence_runs.csv` | Generation scores and Basic Strategy agreement for the leader and population |
| `convergence.csv` and `convergence.png` | Generation metrics summarized across runs, with mean and one standard deviation |
| `baseline_runs.csv` | Held-out returns for all three policies in each run |
| `baseline.csv` and `baseline.png` | Held-out mean and standard deviation of return per hand across runs |
| `policies.csv` | Selected strategy table, generation, validation score and agreement for each run |
| `run_config.json` | Settings, root seed, run seed key and dependency versions |

Each invocation also creates a `results/batch_.../` folder. Its `run_index.csv` maps configurations and replicate indices to the individual run directories. Its `sweep_summary.csv` compares held-out return across settings, and each `config_.../` subfolder holds pooled convergence and baseline CSVs and plots. The summaries use population standard deviation across runs. A single run has a reported standard deviation of zero.

The CSVs and plots already checked into the root of the results folder are from the earlier card-counting and bet-sizing experiment. Run the commands above to generate results for the Hit/Stand strategy.
