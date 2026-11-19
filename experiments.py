"""Train independent Blackjack runs and compare policies on held-out shoes."""

import argparse
import csv
import json
from dataclasses import asdict, replace
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from tempfile import mkdtemp
from time import perf_counter

import matplotlib
import numpy as np

from blackjack_env import ShoeBatch, add_rule_arguments, basic_strategy_table
from ea_solver import (
    EA_CHOICES, EA_PARAMETER_TYPES, UPCARD_FIELDS, add_ea_arguments, champion_pair_rows, configs_from_args,
    evolve, strategy_match_pct,
)

matplotlib.use("Agg")
import matplotlib.pyplot as plt


POLICIES = ("Evolved", "Random Play", "Basic Strategy")
METRICS = ("best_fitness", "average_fitness", "best_match_pct", "population_match_pct")
POLICY_FIELDS = ("run", "selected_generation", "validation_fitness", "match_pct",
                 "hand_type", "player_total", *UPCARD_FIELDS)


def write_csv(path, fields, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(rows)


def save_convergence(output, histories, fitness_objective):
    generations = np.arange(1, histories.shape[1] + 1)
    mean = histories.mean(axis=0)
    sd = histories.std(axis=0)
    fields = tuple(f"{metric}_{stat}" for metric in METRICS for stat in ("mean", "sd"))
    write_csv(output / "convergence.csv", ("generation", *fields),
              ((gen, *np.column_stack((m, s)).ravel()) for gen, m, s in zip(generations, mean, sd)))

    figure, (fitness_ax, match_ax) = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    series = (
        (fitness_ax, "Generation leader"), (fitness_ax, "Population mean"),
        (match_ax, "Generation leader"), (match_ax, "Population mean"),
    )
    for column, (axis, label) in enumerate(series):
        values, spread = mean[:, column], sd[:, column]
        line, = axis.plot(generations, values, label=label)
        axis.fill_between(generations, values - spread, values + spread,
                          color=line.get_color(), alpha=0.15)
    fitness_label = ("Risk-adjusted fitness" if fitness_objective == "risk_adjusted"
                     else "Mean net return per hand (units)")
    fitness_ax.set(xlabel="Generation", ylabel=fitness_label)
    fitness_ax.axhline(0, color="0.55", linewidth=1)
    match_ax.set(xlabel="Generation", ylabel="Basic Strategy agreement (%)", ylim=(0, 100))
    for axis in (fitness_ax, match_ax):
        axis.grid(alpha=0.25)
        axis.legend()
    figure.suptitle("Training across independent runs · mean ± 1 SD")
    figure.savefig(output / "convergence.png", dpi=170)
    plt.close(figure)


def save_comparison(output, returns):
    means = returns.mean(axis=0)
    spreads = returns.std(axis=0)
    write_csv(output / "baseline.csv", ("policy", "mean_return_per_hand", "sd_return_per_hand"),
              zip(POLICIES, means, spreads))

    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    positions = np.arange(len(POLICIES))
    axis.errorbar(positions, means, yerr=spreads, fmt="o", capsize=5, color="#17626b")
    axis.set_xticks(positions, POLICIES)
    axis.set_xlim(-0.5, len(POLICIES) - 0.5)
    axis.set(ylabel="Mean net return per hand (units)", title="Held-out policy comparison")
    axis.axhline(0, color="0.55", linewidth=1)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output / "baseline.png", dpi=170)
    plt.close(figure)


def build_sweep(base, sweep_values, mode="one-at-a-time"):
    """Build distinct EA configurations starting from the base settings."""
    if mode not in ("one-at-a-time", "grid"):
        raise ValueError("sweep mode must be one-at-a-time or grid")
    if not sweep_values:
        return [base]
    for name, values in sweep_values.items():
        if len(values) == 0:
            raise ValueError(f"sweep values for {name} cannot be empty")
    if mode == "grid":
        changes = (dict(zip(sweep_values, values)) for values in product(*sweep_values.values()))
    else:
        changes = ({name: value} for name, values in sweep_values.items() for value in values)
    configs = dict.fromkeys([base] if mode == "one-at-a-time" else [])
    for overrides in changes:
        try:
            configs[replace(base, **overrides)] = None
        except ValueError as error:
            settings = ", ".join(f"{key}={value}" for key, value in overrides.items())
            raise ValueError(f"invalid sweep configuration ({settings}): {error}") from error
    return list(configs)


def unique_output(root, label):
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return Path(mkdtemp(prefix=f"{label}_{timestamp}_", dir=root))


def save_champion_tables(output, result):
    def rows():
        for generation, entry in enumerate(result.history, start=1):
            for row in champion_pair_rows(entry.leader):
                yield "generation", generation, entry.best_fitness, None, *row
        selected = result.selected_generation
        training_fitness = result.history[selected - 1].best_fitness
        for row in champion_pair_rows(result.policy):
            yield "final", selected, training_fitness, result.validation_fitness, *row

    write_csv(output / "champion_pair_decisions.csv",
              ("stage", "generation", "training_fitness", "validation_fitness",
               "row_kind", "player_hand", "hand_type", "player_total", *UPCARD_FIELDS), rows())


def save_outputs(output, histories, members, returns, baseline_runs, policy_rows, fitness_objective):
    write_csv(output / "convergence_runs.csv", ("run", "generation", *METRICS), members)
    write_csv(output / "baseline_runs.csv", ("run", "policy", "return_per_hand"), baseline_runs)
    write_csv(output / "policies.csv", POLICY_FIELDS, policy_rows)
    save_convergence(output, histories, fitness_objective)
    save_comparison(output, returns)


def run_configuration(args, config, rules, reference, batch, index):
    config_id = f"config_{index:03d}"
    summary_dir = batch / config_id
    summary_dir.mkdir()
    metadata = {
        "batch_id": batch.name, "config_id": config_id, "ea": asdict(config),
        "blackjack": asdict(rules), "seed": args.seed, "num_runs": args.num_runs,
        "test_hands": args.test_hands, "numpy_version": np.__version__,
        "matplotlib_version": matplotlib.__version__,
    }
    histories, members, returns = [], [], []
    baseline_runs, policy_rows, run_index = [], [], []
    # Each config gets the same run seed tree so matching indices can be compared.
    for run, run_seed in enumerate(np.random.SeedSequence(args.seed).spawn(args.num_runs), start=1):
        label = (
            f"run_c{index:03d}_r{run:02d}_m{config.mutation_rate}"
            f"_{config.parent_selection}_{config.crossover_type}_{config.mutation_type}"
            f"_{config.survivor_selection}_{config.fitness_objective}"
        )
        output = unique_output(args.output, label)
        print(f"{config_id} run {run}/{args.num_runs} in {output.resolve()}", flush=True)
        started = perf_counter()
        train_seed, test_seed, random_seed = run_seed.spawn(3)
        result = evolve(config, rules, train_seed, reference, log_pair_tables=args.print_pair_tables)
        history = [
            (entry.best_fitness, entry.average_fitness, strategy_match_pct(entry.leader, reference),
             entry.population_match_pct)
            for entry in result.history
        ]
        test_batch = ShoeBatch(rules, args.test_hands, test_seed)
        evolved_return, reference_return = test_batch.evaluate(np.stack((result.policy, reference)))
        random_return = test_batch.evaluate_random(random_seed)
        scores = (evolved_return, random_return, reference_return)
        match = strategy_match_pct(result.policy, reference)
        generation_rows = [(run, gen, *row) for gen, row in enumerate(history, start=1)]
        baseline_rows = [(run, name, score) for name, score in zip(POLICIES, scores)]
        final_policy_rows = [
            (run, result.selected_generation, result.validation_fitness, match, hand_type, total, *actions)
            for kind, _, hand_type, total, *actions in champion_pair_rows(result.policy)
            if kind == "state"
        ]
        save_champion_tables(output, result)
        save_outputs(output, np.asarray([history]), generation_rows, np.asarray([scores]),
                     baseline_rows, final_policy_rows, config.fitness_objective)
        (output / "run_config.json").write_text(json.dumps({
            **metadata, "run": run, "run_id": output.name, "run_spawn_key": list(run_seed.spawn_key),
            "selected_generation": result.selected_generation,
        }, indent=2) + "\n", encoding="utf-8")

        histories.append(history)
        members.extend(generation_rows)
        returns.append(scores)
        baseline_runs.extend(baseline_rows)
        policy_rows.extend(final_policy_rows)
        run_index.append((config_id, run, output.name))
        print(f"  Selected generation {result.selected_generation}  "
              f"held-out {evolved_return:+.4f} units/hand  ({perf_counter() - started:.1f} s)", flush=True)

    returns = np.asarray(returns)
    save_outputs(summary_dir, np.asarray(histories), members, returns, baseline_runs, policy_rows,
                 config.fitness_objective)
    (summary_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return config_id, returns.mean(axis=0), returns.std(axis=0), run_index


def main(default_runs=3):
    parser = argparse.ArgumentParser(description=__doc__)
    add_ea_arguments(parser)
    add_rule_arguments(parser)
    parser.add_argument("--num-runs", type=int, default=default_runs)
    parser.add_argument("--seed", type=int, default=455851, help="root seed for independent runs")
    parser.add_argument("--test-hands", type=int, default=50000, help="held-out hands per run")
    parser.add_argument("--output", type=Path, default=Path("results"), help="parent for new run folders")
    parser.add_argument("--print-pair-tables", action="store_true", help="print generation and final champions")
    group = parser.add_argument_group("parameter sweep")
    group.add_argument("--sweep-mode", choices=("one-at-a-time", "grid"), default="one-at-a-time")
    for name, value_type in EA_PARAMETER_TYPES.items():
        flag = f"--sweep-{name.replace('_', '-')}"
        group.add_argument(flag, nargs="+", type=value_type, choices=EA_CHOICES.get(name))
    args = parser.parse_args()
    try:
        base, rules = configs_from_args(args)
        if args.num_runs < 1 or args.test_hands < 1 or args.seed < 0:
            raise ValueError("run and test hand counts must be positive and seed nonnegative")
        sweep_values = {
            name: getattr(args, f"sweep_{name}") for name in EA_PARAMETER_TYPES
            if getattr(args, f"sweep_{name}") is not None
        }
        configs = build_sweep(base, sweep_values, args.sweep_mode)
    except ValueError as error:
        parser.error(str(error))

    args.output.mkdir(parents=True, exist_ok=True)
    batch = unique_output(args.output, "batch")
    reference = basic_strategy_table()
    summaries, run_index = [], []
    for index, config in enumerate(configs, start=1):
        config_id, means, spreads, runs = run_configuration(args, config, rules, reference, batch, index)
        run_index.extend(runs)
        values = tuple(getattr(config, name) for name in EA_PARAMETER_TYPES)
        summaries.extend((config_id, *values, args.num_runs, name, mean, sd)
                         for name, mean, sd in zip(POLICIES, means, spreads))
        print(f"{config_id} held-out return per hand (mean ± SD)")
        for name, mean, sd in zip(POLICIES, means, spreads):
            print(f"  {name:>14}  {mean:+.4f} ± {sd:.4f}")

    write_csv(batch / "sweep_summary.csv",
              ("config_id", *EA_PARAMETER_TYPES, "num_runs", "policy",
               "mean_return_per_hand", "sd_return_per_hand"), summaries)
    write_csv(batch / "run_index.csv", ("config_id", "run", "run_id"), run_index)
    print(f"\nWrote configuration summaries to {batch.resolve()}")


if __name__ == "__main__":
    main()
