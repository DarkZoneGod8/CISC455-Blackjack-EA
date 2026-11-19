"""Evolve binary Blackjack policies with a generational genetic algorithm."""

from dataclasses import dataclass, fields
from math import isfinite

import numpy as np

from blackjack_env import HIT, BlackjackConfig, POLICY_SHAPE, STAND, STATES, UPCARDS, ShoeBatch


EA_CHOICES = {
    "parent_selection": ("tournament", "rank"),
    "crossover_type": ("uniform", "one_point"),
    "mutation_type": ("bit_flip", "fixed_count"),
    "survivor_selection": ("elitism", "offspring_only"),
    "fitness_objective": ("mean_return", "risk_adjusted"),
}
CROSSOVER_TYPES = EA_CHOICES["crossover_type"]
UPCARD_LABELS = tuple("A" if card == 1 else str(card) for card in UPCARDS)
UPCARD_FIELDS = tuple(f"vs_{label}" for label in UPCARD_LABELS)


@dataclass(frozen=True)
class EAConfig:
    pop_size: int = 256
    num_generations: int = 35
    tournament_size: int = 4
    crossover_rate: float = 0.9
    mutation_rate: float = 0.01
    elitism_count: int = 6
    num_hands: int = 1500
    validation_hands: int = 12000
    crossover_type: str = "uniform"
    parent_selection: str = "tournament"
    survivor_selection: str = "elitism"
    mutation_type: str = "bit_flip"
    fitness_objective: str = "mean_return"
    risk_penalty: float = 0.1
    initial_hit_probability: float = 0.5

    def __post_init__(self):
        if self.pop_size < 2 or self.num_generations < 1 or self.num_hands < 1:
            raise ValueError("require population >= 2 and positive generation and hand counts")
        if self.tournament_size < 1 or (
            self.parent_selection == "tournament" and self.tournament_size > self.pop_size
        ):
            raise ValueError("tournament_size must be positive and at most pop_size in tournament mode")
        if self.elitism_count < 0 or (
            self.survivor_selection == "elitism" and self.elitism_count >= self.pop_size
        ):
            raise ValueError("elitism_count must be nonnegative and below pop_size in elitism mode")
        if not 0 <= self.crossover_rate <= 1 or not 0 <= self.mutation_rate <= 1:
            raise ValueError("crossover_rate and mutation_rate must be in [0, 1]")
        if self.validation_hands < 0:
            raise ValueError("validation_hands must be nonnegative")
        if not 0 <= self.initial_hit_probability <= 1:
            raise ValueError("initial_hit_probability must be in [0, 1]")
        if not isfinite(self.risk_penalty) or self.risk_penalty < 0:
            raise ValueError("risk_penalty must be finite and nonnegative")
        for name, choices in EA_CHOICES.items():
            if getattr(self, name) not in choices:
                raise ValueError(f"{name} must be one of {choices}")


_EA_DEFAULTS = EAConfig()
EA_PARAMETER_TYPES = {name: type(value) for name, value in vars(_EA_DEFAULTS).items()}


@dataclass(frozen=True)
class Generation:
    best_fitness: float
    average_fitness: float
    leader: np.ndarray
    population_match_pct: float | None


@dataclass(frozen=True)
class RunResult:
    history: tuple[Generation, ...]
    policy: np.ndarray
    selected_generation: int
    validation_fitness: float | None


def strategy_match_pct(policy: np.ndarray, reference: np.ndarray) -> float:
    """Unweighted percentage of strategy cells matching the reference."""
    return float(100 * np.mean(policy == reference))


def _action_labels(actions):
    return tuple("Stand" if value == STAND else "Hit" for value in actions)


def champion_pair_rows(policy: np.ndarray):
    """Yield both policy-state rows and equal-rank two-card starting hands."""
    for (soft, total), actions in zip(STATES, policy):
        hand_type = "soft" if soft else "hard"
        yield "state", f"{hand_type.title()} {total}", hand_type, total, *_action_labels(actions)

    state_rows = dict(zip(STATES, policy))
    for rank in (*range(2, 11), 1):
        label = "A" if rank == 1 else str(rank)
        state = (True, 12) if rank == 1 else (False, 2 * rank)
        soft, total = state
        yield ("pair", f"{label}+{label}", "soft" if soft else "hard",
               total, *_action_labels(state_rows[state]))


def format_champion_pair_table(policy: np.ndarray) -> str:
    """Format the action for every player state or starting pair vs dealer upcard."""
    header = f"{'Player hand':<13}" + "".join(f"{card:>6}" for card in UPCARD_LABELS)
    lines = [header, "-" * len(header)]
    for index, (_, hand, _, _, *actions) in enumerate(champion_pair_rows(policy)):
        if index == len(STATES):
            lines.extend(("", "Equal-rank starting hands", header, "-" * len(header)))
        lines.append(f"{hand:<13}" + "".join(f"{action:>6}" for action in actions))
    return "\n".join(lines)


def _select_parents(scores, child_count, config, rng):
    if config.parent_selection == "tournament":
        contests = rng.integers(config.pop_size, size=(2, child_count, config.tournament_size))
        winners = np.take_along_axis(contests, scores[contests].argmax(axis=-1)[..., None], axis=-1)
        return winners[..., 0]

    # Randomizing tie order gives equally fit candidates equal chances in expectation.
    shuffled = rng.permutation(config.pop_size)
    ranked = shuffled[np.argsort(scores[shuffled], kind="stable")]
    weights = np.arange(1, config.pop_size + 1, dtype=float)
    return rng.choice(ranked, size=(2, child_count), p=weights / weights.sum())


def _mutate(children, config, rng):
    if config.mutation_type == "bit_flip":
        children ^= rng.random(children.shape) < config.mutation_rate
        return

    # Round stochastically so both operators flip the same expected number of genes.
    flattened = children.reshape(len(children), -1)
    gene_count = flattened.shape[1]
    expected = gene_count * config.mutation_rate
    whole = int(expected)
    counts = whole + (rng.random(len(children)) < expected - whole)
    for child, count in zip(flattened, counts):
        if count:
            child[rng.choice(gene_count, size=count, replace=False)] ^= 1


def _next_population(population, scores, config, rng):
    elite_count = config.elitism_count if config.survivor_selection == "elitism" else 0
    elite_indices = np.argsort(-scores, kind="stable")[:elite_count]
    child_count = config.pop_size - elite_count
    first_parents, second_parents = population[_select_parents(scores, child_count, config, rng)]
    crossover = rng.random((child_count, 1, 1)) < config.crossover_rate
    if config.crossover_type == "uniform":
        take_second = rng.random(first_parents.shape) < 0.5
    else:
        gene_count = POLICY_SHAPE[0] * POLICY_SHAPE[1]
        cuts = rng.integers(1, gene_count, size=(child_count, 1, 1))
        gene_indices = np.arange(gene_count).reshape(POLICY_SHAPE)
        take_second = gene_indices >= cuts
    children = np.where(crossover & take_second, second_parents, first_parents)
    _mutate(children, config, rng)
    return np.concatenate((population[elite_indices], children), axis=0)


def evolve(
    config: EAConfig = EAConfig(), rules: BlackjackConfig = BlackjackConfig(),
    seed: int | np.random.SeedSequence = 0, reference: np.ndarray | None = None,
    *, log_pair_tables: bool = False,
) -> RunResult:
    """Evolve a randomly initialized strategy population and optionally log its champions."""
    root = seed if isinstance(seed, np.random.SeedSequence) else np.random.SeedSequence(seed)
    operator_seed, training_seed, validation_seed = root.spawn(3)
    rng = np.random.default_rng(operator_seed)
    population_shape = (config.pop_size, *POLICY_SHAPE)
    if config.initial_hit_probability == 0.5:
        population = rng.integers(0, 2, size=population_shape, dtype=np.uint8)
    else:
        population = np.where(rng.random(population_shape) < config.initial_hit_probability,
                              HIT, STAND).astype(np.uint8)
    if reference is not None and reference.shape != POLICY_SHAPE:
        raise ValueError("reference must have the policy table shape")

    risk_penalty = config.risk_penalty if config.fitness_objective == "risk_adjusted" else 0.0
    history = []
    for generation, batch_seed in enumerate(training_seed.spawn(config.num_generations), start=1):
        scores = ShoeBatch(rules, config.num_hands, batch_seed).evaluate(population, risk_penalty=risk_penalty)
        leader_index = int(scores.argmax())
        # This diagnostic has no influence on training, validation or selection.
        match = None if reference is None else strategy_match_pct(population, reference)
        entry = Generation(float(scores[leader_index]), float(scores.mean()),
                           population[leader_index].copy(), match)
        history.append(entry)
        if log_pair_tables:
            print(f"\nGeneration {generation} champion ({entry.best_fitness:+.4f} fitness)\n"
                  f"{format_champion_pair_table(entry.leader)}", flush=True)
        if generation < config.num_generations:
            population = _next_population(population, scores, config, rng)

    selected = len(history) - 1
    validation_fitness = None
    if config.validation_hands:
        leaders = np.stack([entry.leader for entry in history])
        scores = ShoeBatch(rules, config.validation_hands, validation_seed).evaluate(leaders, risk_penalty=risk_penalty)
        selected = int(scores.argmax())
        validation_fitness = float(scores[selected])
    policy = history[selected].leader
    if log_pair_tables:
        score = history[selected].best_fitness if validation_fitness is None else validation_fitness
        sample = "training" if validation_fitness is None else "validation"
        print(f"\nFinal champion from generation {selected + 1} "
              f"({sample} fitness {score:+.4f})\n{format_champion_pair_table(policy)}", flush=True)
    return RunResult(tuple(history), policy, selected + 1, validation_fitness)


def add_ea_arguments(parser):
    group = parser.add_argument_group("evolution")
    descriptions = {
        "tournament_size": "number of candidates per tournament",
        "elitism_count": "number of intact parents copied in elitism mode",
        "validation_hands": "hands for choosing among generation champions (0 disables validation)",
        "risk_penalty": "multiplier for per-hand payout deviation in risk_adjusted mode",
    }
    for name, value_type in EA_PARAMETER_TYPES.items():
        group.add_argument(f"--{name.replace('_', '-')}", type=value_type, default=getattr(_EA_DEFAULTS, name),
                           choices=EA_CHOICES.get(name), help=descriptions.get(name))


def configs_from_args(args):
    return tuple(
        config_type(**{field.name: getattr(args, field.name) for field in fields(config_type)})
        for config_type in (EAConfig, BlackjackConfig)
    )


def main():
    # Keep the short single-run entry point on the same output pipeline.
    from experiments import main as run_experiments

    run_experiments(default_runs=1)


if __name__ == "__main__":
    main()
