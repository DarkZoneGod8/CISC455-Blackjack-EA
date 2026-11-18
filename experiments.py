"""Run paired-seed Blackjack experiments and write presentation-ready results."""

import argparse
import csv
import json
import os
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter

# Matrix products have only ten features. A single BLAS thread reduces overhead.
for variable in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(variable, "1")

import matplotlib
import numpy as np

from blackjack_env import evaluate, generate_shoes, prepare_shoes
from ea_solver import EAConfig, EvolutionaryAlgorithm, decode, fitness

matplotlib.use("Agg")
import matplotlib.pyplot as plt


RANK_NAMES = ("2", "3", "4", "5", "6", "7", "8", "9", "10/J/Q/K", "A")
HI_LO = np.array([1, 1, 1, 1, 1, 0, 0, 0, -1, -1])
METRICS = ("Mean Profit", "Win Rate", "Std Dev", "Sharpe-like Ratio", "Ruin Rate")
REFERENCE = "Risk-averse / arithmetic"
NEUTRAL = "Risk-neutral / arithmetic"
DISCRETE = "Risk-averse / discrete"


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def run_seed(configs, seed, validation):
    """Pair the variants on one initial population and one shoe stream per seed."""
    solvers = {name: EvolutionaryAlgorithm(config, seed) for name, config in configs.items()}
    shoe_rng = next(iter(solvers.values())).shoe_rng
    reference = configs[REFERENCE]
    for _ in range(reference.generations):
        batch = prepare_shoes(generate_shoes(reference.shoes_per_generation, shoe_rng))
        for solver in solvers.values():
            solver.step(batch)

    histories, champions = {}, {}
    for name, solver in solvers.items():
        result = solver.result()
        histories[name] = result.history
        tags, ramps = decode(result.leaders)
        config = configs[name]
        scores = np.array([
            fitness(evaluate(batch, tags, ramps), config.lambda_risk, config.ruin_penalty)
            for batch in validation
        ]).mean(axis=0)
        best = int(scores.argmax())
        champions[name] = (result.leaders[best], float(scores[best]), best + 1)
    return histories, champions


def test_metrics(test_batches, tags, ramps):
    """Pool shoe profit and played hands, and count ruin per independent session."""
    results = [evaluate(batch, tags, ramps) for batch in test_batches]
    profits = np.concatenate([result.profits for result in results], axis=1)
    means, spreads = profits.mean(axis=1), profits.std(axis=1)
    wins = np.sum([result.wins for result in results], axis=0)
    hands = np.sum([result.hands for result in results], axis=0)
    ratios = np.divide(means, spreads, out=np.full_like(means, np.nan), where=spreads > 0)
    ruin = np.mean([result.ruined for result in results], axis=0)
    return np.column_stack((means, wins / hands, spreads, ratios, ruin))


def convergence_outputs(output, histories, variants):
    labels = ("Best fitness", "Average fitness")
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    summaries = []
    members = []
    for name in variants:
        runs = histories[name]
        raw = np.array([[(row.best, row.average, row.ruin_rate) for row in run] for run in runs])
        x = np.arange(1, raw.shape[1] + 1)
        mean, spread = raw.mean(axis=0), raw.std(axis=0)
        members.extend((name, seed, generation, *row)
                       for seed, run in enumerate(raw) for generation, row in enumerate(run, 1))
        summaries.extend((name, gen, m[0], s[0], m[1], s[1], m[2], s[2])
                         for gen, m, s in zip(x, mean, spread))
        for column, ax in enumerate(axes):
            line, = ax.plot(x, mean[:, column], label=name)
            ax.fill_between(x, mean[:, column] - spread[:, column],
                            mean[:, column] + spread[:, column], color=line.get_color(), alpha=0.15)
            ax.set(xlabel="Generation", ylabel=labels[column] + " (units per shoe)")
            ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.suptitle("Training convergence across seeds (mean ± 1 SD)")
    figure.savefig(output / "convergence.png", dpi=170)
    plt.close(figure)
    write_csv(output / "convergence_runs.csv",
              ("variant", "seed", "generation", "best_fitness", "average_fitness", "population_ruin"),
              members)
    write_csv(output / "convergence.csv",
              ("variant", "generation", "best_mean", "best_sd", "average_mean", "average_sd",
               "ruin_mean", "ruin_sd"), summaries)


def ablation_outputs(output, histories, champions, results):
    figure, axes = plt.subplots(2, 2, figsize=(9, 7), layout="constrained")
    comparisons = [
        ("Risk coefficient", (NEUTRAL, REFERENCE), ("λ = 0", "λ = 0.25")),
        ("Crossover", (REFERENCE, DISCRETE), ("Arithmetic", "Discrete")),
    ]
    for column, (title, variants, labels) in enumerate(comparisons):
        for row, metric in enumerate((0, 4)):
            ax = axes[row, column]
            values = np.stack([results[name][:, metric] for name in variants], axis=1)
            for pair in values:
                ax.plot([0, 1], pair, "o-", alpha=0.3, color="#247c87")
            ax.plot([0, 1], values.mean(axis=0), "o-", linewidth=3, color="#17254a")
            ax.set_xticks([0, 1], labels)
            ax.set_ylabel("Mean profit (units per shoe)" if metric == 0 else "Session ruin rate")
            ax.grid(axis="y", alpha=0.25)
            if row == 0:
                ax.axhline(0, color="0.55", linewidth=1)
                ax.set_title(title)
            else:
                ax.set_ylim(-0.04, 1.04)
    figure.suptitle("Held-out ablations by paired seed (dark line shows the mean)")
    figure.savefig(output / "ablation.png", dpi=170)
    plt.close(figure)

    write_csv(output / "ablation_runs.csv",
              ("variant", "seed", "leader_generation", "training_fitness",
               "validation_fitness", "training_minus_validation", *METRICS),
              ((name, seed, generation, histories[name][seed][generation - 1].best, score,
                histories[name][seed][generation - 1].best - score, *results[name][seed])
               for name, entries in champions.items()
               for seed, (_, score, generation) in enumerate(entries)))
    write_csv(output / "ablation.csv", ("variant", "statistic", *METRICS),
              ((name, statistic, *values) for name, runs in results.items()
               for statistic, values in (("mean", runs.mean(axis=0)), ("sd", runs.std(axis=0)))))


def baseline_outputs(output, entries, tags, ramps, results):
    print("\nHeld-out baseline comparison")
    print(f"{'Policy':<18}{'Mean profit':>13}{'Win rate':>12}{'Std dev':>11}"
          f"{'Sharpe-like':>13}{'Ruin rate':>12}")
    for name, (mean, win, sd, ratio, ruin) in zip(entries, results):
        print(f"{name:<18}{mean:>13.3f}{win:>12.2%}{sd:>11.3f}{ratio:>13.3f}{ruin:>12.2%}")
    write_csv(output / "baseline.csv", ("policy", *METRICS),
              ((name, *row) for name, row in zip(entries, results)))

    print(f"\n{'Rank':<10}{'Evolved':>12}{'Hi-Lo':>12}")
    for rank, learned, standard in zip(RANK_NAMES, tags[0], HI_LO):
        print(f"{rank:<10}{learned:>+12.3f}{standard:>+12d}")
    print(f"Bet ramps (TC <1 / [1, 2) / [2, 3) / >=3)  "
          f"evolved {ramps[0].tolist()}, Hi-Lo {ramps[2].tolist()}")
    write_csv(output / "policies.csv", ("policy", *RANK_NAMES, "b0", "b1", "b2", "b3"),
              ((name, *row, *ramp) for name, row, ramp in zip(entries, tags, ramps)))

    figure, axes = plt.subplots(1, 3, figsize=(10, 3.8), layout="constrained")
    for axis, index in zip(axes, (0, 2, 4)):
        axis.bar(entries, results[:, index], color=("#247c87", "#aab5c8", "#485a84"))
        axis.set_title(METRICS[index])
        axis.tick_params(axis="x", labelrotation=20)
        axis.grid(axis="y", alpha=0.25)
        if index == 4:
            axis.set_ylim(0, 1)
    figure.suptitle("Held-out policy comparison")
    figure.savefig(output / "baseline.png", dpi=170)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--quick", action="store_true", help="run a small pipeline smoke test")
    parser.add_argument("--self-adaptive", action="store_true", help="evolve per-gene mutation scales")
    args = parser.parse_args()
    started = perf_counter()

    config = EAConfig(self_adaptive=args.self_adaptive)
    seeds, test_sessions, shoes_per_session = 10, 10, 200
    if args.quick:
        config = replace(config, pop_size=12, generations=4, shoes_per_generation=8)
        seeds, test_sessions, shoes_per_session = 2, 3, 8
    configs = {REFERENCE: config, NEUTRAL: replace(config, lambda_risk=0.0),
               DISCRETE: replace(config, crossover="discrete")}
    args.output.mkdir(parents=True, exist_ok=True)
    # Independent random streams for validation and test, separate from training seeds.
    validation_rng, test_rng = [
        np.random.default_rng(seed) for seed in np.random.SeedSequence(455851).spawn(2)
    ]
    validation = [
        prepare_shoes(generate_shoes(config.shoes_per_generation, validation_rng)) for _ in range(2)
    ]
    histories = {name: [] for name in configs}
    champions = {name: [] for name in configs}
    for seed in range(seeds):
        seed_histories, seed_champions = run_seed(configs, seed, validation)
        for name in configs:
            histories[name].append(seed_histories[name])
            champions[name].append(seed_champions[name])
        print(f"Completed seed {seed + 1}/{seeds} ({perf_counter() - started:.1f} s)", flush=True)

    # Choose the primary champion entirely on validation, before looking at test results.
    winning_seed = max(range(seeds), key=lambda seed: champions[REFERENCE][seed][1])
    winning_genes, _, winning_generation = champions[REFERENCE][winning_seed]
    test = [prepare_shoes(generate_shoes(shoes_per_session, test_rng))
            for _ in range(test_sessions)]
    all_tags, all_ramps = decode(np.array([entry[0] for runs in champions.values() for entry in runs]))
    all_results = test_metrics(
        test, np.vstack((all_tags, np.zeros(10), HI_LO)),
        np.vstack((all_ramps, [1, 1, 1, 1], [1, 2, 4, 8])),
    )
    baseline_indices = [winning_seed, len(all_tags), len(all_tags) + 1]
    baseline_outputs(
        args.output, ("Evolved", "Flat", "Hi-Lo"), np.vstack((all_tags[winning_seed], np.zeros(10), HI_LO)),
        np.vstack((all_ramps[winning_seed], [1, 1, 1, 1], [1, 2, 4, 8])), all_results[baseline_indices],
    )
    convergence_outputs(args.output, histories, configs)
    ablation_outputs(args.output, histories, champions, {
        name: all_results[index * seeds:(index + 1) * seeds]
        for index, name in enumerate(configs)
    })
    (args.output / "run_config.json").write_text(json.dumps({
        "variants": {name: asdict(variant) for name, variant in configs.items()},
        "training_seeds": list(range(seeds)), "validation_sessions": 2,
        "holdout_seed": 455851, "test_sessions": test_sessions,
        "test_shoes_per_session": shoes_per_session, "winning_seed": winning_seed,
        "winning_generation": winning_generation, "winning_genes": winning_genes.tolist(),
    }, indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote CSVs and PNGs to {args.output} in {perf_counter() - started:.1f} s.")


if __name__ == "__main__":
    main()
