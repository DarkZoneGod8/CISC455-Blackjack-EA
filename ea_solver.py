"""Real-valued evolutionary optimization for counting and bet sizing.

The genotype is the inherited vector of 14 real genes. The phenotype is
the playable normalized count and rounded bet ramp. Tag and bet genes
exhibit epistasis: moving a tag across a count boundary only matters when
the bets on the two sides differ.
"""

from dataclasses import dataclass

import numpy as np

from blackjack_env import Evaluation, ShoeBatch, evaluate, generate_shoes, prepare_shoes


LOWER = np.array([-2.0] * 10 + [1.0] * 4)
UPPER = np.array([2.0] * 10 + [12.0] * 4)
GENES = len(LOWER)


def decode(genes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return L1-normalized tags and nearest-unit wagers (ties round up)."""
    genes = np.asarray(genes, dtype=float)
    if genes.ndim == 0 or genes.shape[-1] != GENES:
        raise ValueError("expected 14 genes per individual")
    weights = genes[..., :10]
    norm = np.abs(weights).sum(axis=-1, keepdims=True)
    tags = np.divide(10 * weights, norm, out=np.zeros_like(weights), where=norm != 0)
    return tags, np.floor(genes[..., 10:] + 0.5).astype(np.int8)


def fitness(result: Evaluation, lambda_risk: float, ruin_penalty: float) -> np.ndarray:
    """Score per-shoe profit, its population standard deviation, and zero cash."""
    profits = result.profits
    return profits.mean(axis=1) - lambda_risk * profits.std(axis=1) - ruin_penalty * result.zero_bankroll


@dataclass(frozen=True)
class EAConfig:
    pop_size: int = 60
    generations: int = 50
    shoes_per_generation: int = 150
    lambda_risk: float = 0.25
    ruin_penalty: float = 5.0
    tournament_size: int = 3
    elites: int = 2
    crossover: str = "arithmetic"
    sigma: float = 0.35
    self_adaptive: bool = False
    tau: float = 1 / np.sqrt(2 * GENES)

    def __post_init__(self):
        if not 0 <= self.elites < self.pop_size or self.generations < 1:
            raise ValueError("require 0 <= elites < pop_size and positive generations")
        if self.shoes_per_generation < 1 or self.tournament_size < 1:
            raise ValueError("shoe count and tournament size must be positive")
        if self.crossover not in ("arithmetic", "discrete"):
            raise ValueError("crossover must be arithmetic or discrete")
        if self.sigma <= 0 or self.tau < 0 or self.lambda_risk < 0 or self.ruin_penalty < 0:
            raise ValueError("invalid mutation or risk parameters")


@dataclass(frozen=True)
class Generation:
    best: float
    average: float
    ruin_rate: float


@dataclass(frozen=True)
class RunResult:
    history: tuple[Generation, ...]
    leaders: np.ndarray


class EvolutionaryAlgorithm:
    """Tournament selection, recombination, mutation and generational elitism.

    Larger tournaments increase selection pressure by giving highly ranked
    parents more chances to win. Elites survive intact, and CRN compares them
    with their offspring on a fresh common shoe batch next generation.
    """

    def __init__(self, config: EAConfig = EAConfig(), seed: int = 0):
        self.config = config
        operator_seed, shoe_seed = np.random.SeedSequence(seed).spawn(2)
        self.rng = np.random.default_rng(operator_seed)
        self.shoe_rng = np.random.default_rng(shoe_seed)
        self.genes = self.rng.uniform(LOWER, UPPER, (config.pop_size, GENES))
        self.steps = np.full(self.genes.shape, config.sigma)
        self.history: list[Generation] = []
        self.leaders: list[np.ndarray] = []

    def _next_population(self, scores: np.ndarray) -> None:
        config = self.config
        elites = np.argsort(-scores, kind="stable")[:config.elites]
        count = config.pop_size - config.elites
        contests = self.rng.integers(config.pop_size, size=(2, count, config.tournament_size))
        winners = np.take_along_axis(contests, scores[contests].argmax(axis=2, keepdims=True), axis=2)
        first, second = winners[..., 0]

        if config.crossover == "arithmetic":
            # One alpha per child applies to the entire vector.
            alpha = self.rng.random((count, 1))
            children = alpha * self.genes[first] + (1 - alpha) * self.genes[second]
            steps = alpha * self.steps[first] + (1 - alpha) * self.steps[second]
        else:
            mask = self.rng.integers(2, size=(count, GENES), dtype=bool)
            children = np.where(mask, self.genes[first], self.genes[second])
            steps = np.where(mask, self.steps[first], self.steps[second])

        if config.self_adaptive:
            # Each gene inherits its own step size and perturbs it independently.
            steps *= np.exp(config.tau * self.rng.standard_normal(steps.shape))
            np.clip(steps, 1e-3, UPPER - LOWER, out=steps)
        children += steps * self.rng.standard_normal(children.shape)
        np.clip(children, LOWER, UPPER, out=children)
        self.genes = np.concatenate((self.genes[elites], children))
        self.steps = np.concatenate((self.steps[elites], steps))

    def step(self, batch: ShoeBatch | None = None) -> Generation:
        """Evaluate the current generation and prepare its successor."""
        if len(self.history) >= self.config.generations:
            raise RuntimeError("all configured generations are complete")
        if batch is None:
            batch = prepare_shoes(generate_shoes(self.config.shoes_per_generation, self.shoe_rng))
        result = evaluate(batch, *decode(self.genes))
        scores = fitness(result, self.config.lambda_risk, self.config.ruin_penalty)
        report = Generation(float(scores.max()), float(scores.mean()), float(result.ruined.mean()))
        self.history.append(report)
        self.leaders.append(self.genes[scores.argmax()].copy())
        if len(self.history) < self.config.generations:
            self._next_population(scores)
        return report

    def result(self) -> RunResult:
        """Return the fitness history and each generation's training leader."""
        return RunResult(tuple(self.history), np.asarray(self.leaders))

    def run(self) -> RunResult:
        """Complete all configured generations using the internal shoe stream."""
        while len(self.history) < self.config.generations:
            self.step()
        return self.result()
