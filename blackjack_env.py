"""Seeded, flat-wager Blackjack with Hit/Stand policies and finite shoes."""

from dataclasses import dataclass
from math import ceil, isfinite, sqrt

import numpy as np


HIT, STAND = 0, 1
UPCARDS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 1)
UPCARD_COLUMN = (0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
STATES = tuple((False, total) for total in range(4, 21)) + tuple(
    (True, total) for total in range(12, 21)
)
POLICY_SHAPE = (len(STATES), len(UPCARDS))
_DECK = bytes([1, 2, 3, 4, 5, 6, 7, 8, 9] * 4 + [10] * 16)


@dataclass(frozen=True)
class BlackjackConfig:
    num_decks: int = 6
    penetration: float = 0.75
    hit_soft_17: bool = False
    blackjack_payout: float = 1.5
    win_payout: float = 1.0
    loss_payout: float = 1.0

    def __post_init__(self):
        if self.num_decks < 1 or not 0 < self.penetration <= 1:
            raise ValueError("num_decks must be positive and penetration must be in (0, 1]")
        for value in (self.blackjack_payout, self.win_payout, self.loss_payout):
            if not isfinite(value) or value < 0:
                raise ValueError("payouts must be finite and nonnegative")


def basic_strategy_table() -> np.ndarray:
    """Return the conventional multi-deck Hit/Stand reference as a fresh table.

    Pairs use their corresponding hard or soft total. The chart is a fixed
    benchmark even when the simulator uses nonstandard rules or payouts.
    """
    table = np.full(POLICY_SHAPE, HIT, dtype=np.uint8)
    table[12 - 4, 2:5] = STAND                 # Hard 12 vs 4 through 6.
    table[13 - 4:17 - 4, :5] = STAND           # Hard 13–16 vs 2 through 6.
    table[17 - 4:21 - 4, :] = STAND            # Hard 17–20.
    soft_start = 17
    table[soft_start + 18 - 12, :7] = STAND    # Soft 18 vs 2 through 8.
    table[soft_start + 19 - 12:, :] = STAND     # Soft 19–20.
    return table


def _play_round(
    shoe: bytes, start: int, actions: bytes | None, random_bits: int,
    rules: BlackjackConfig,
) -> tuple[int, float]:
    first, upcard, second, hole = shoe[start:start + 4]
    position = start + 4
    player_low = first + second
    dealer_low = upcard + hole
    player_ace = first == 1 or second == 1
    dealer_ace = upcard == 1 or hole == 1

    player_natural = player_ace and player_low == 11
    dealer_natural = dealer_ace and dealer_low == 11
    if dealer_natural:
        return position, 0.0 if player_natural else -rules.loss_payout
    if player_natural:
        return position, rules.blackjack_payout

    column = UPCARD_COLUMN[upcard]
    while True:
        soft = player_ace and player_low <= 11
        player_total = player_low + 10 if soft else player_low
        if player_total > 21:
            return position, -rules.loss_payout
        if player_total == 21:
            break
        if actions is None:
            action = random_bits & 1
            random_bits >>= 1
        else:
            row = player_total + 5 if soft else player_total - 4
            action = actions[row * 10 + column]
        if action == STAND:
            break
        card = shoe[position]
        position += 1
        player_low += card
        player_ace |= card == 1

    while True:
        soft = dealer_ace and dealer_low <= 11
        dealer_total = dealer_low + 10 if soft else dealer_low
        if dealer_total > 17 or (dealer_total == 17 and not (soft and rules.hit_soft_17)):
            break
        card = shoe[position]
        position += 1
        dealer_low += card
        dealer_ace |= card == 1

    if dealer_total > 21 or player_total > dealer_total:
        return position, rules.win_payout
    if player_total < dealer_total:
        return position, -rules.loss_payout
    return position, 0.0


class ShoeBatch:
    """Shared shuffled shoes for exactly num_hands of evaluation per policy.

    Every policy starts at shoe zero. Different actions can consume different
    cards per round, so hand boundaries and reshuffle times can diverge.
    """

    def __init__(self, rules: BlackjackConfig, num_hands: int, seed: int | np.random.SeedSequence):
        if num_hands < 1:
            raise ValueError("num_hands must be positive")
        self.rules = rules
        self.num_hands = num_hands
        self._deck = np.frombuffer(_DECK * rules.num_decks, dtype=np.uint8)
        self._shoes: list[bytes] = []
        self._rng = np.random.default_rng(seed)

        # Before the final card of each hand, combined low totals are at most
        # 20 (player) + 16 (dealer). If a hand takes no hits, count only its
        # first card here. The cheapest ranks bound how many cards can fit.
        cheapest = np.sort(self._deck)[:36]
        reserve = 2 + int(np.searchsorted(np.cumsum(cheapest), 36, side="right"))
        self._cut = min(ceil(len(self._deck) * rules.penetration), len(self._deck) - reserve + 1)

    def _shoe(self, index: int) -> bytes:
        if index == len(self._shoes):
            self._shoes.append(self._rng.permutation(self._deck).tobytes())
        return self._shoes[index]

    def _score(self, actions: bytes | None, random_bits: tuple[int, ...] = (),
               risk_penalty: float = 0.0) -> float:
        shoe_index = position = 0
        shoe = self._shoe(0)
        net = 0.0
        squared_net = 0.0
        cut = self._cut
        rules = self.rules
        for hand in range(self.num_hands):
            if position >= cut:
                shoe_index += 1
                shoe = self._shoe(shoe_index)
                position = 0
            bits = random_bits[hand] if random_bits else 0
            position, payout = _play_round(shoe, position, actions, bits, rules)
            net += payout
            if risk_penalty:
                squared_net += payout * payout
        mean = net / self.num_hands
        if risk_penalty:
            deviation = sqrt(max(0.0, squared_net / self.num_hands - mean * mean))
            return mean - risk_penalty * deviation
        return mean

    def evaluate(self, policies: np.ndarray, *, risk_penalty: float = 0.0) -> np.ndarray:
        """Score policies by mean return minus an optional penalty for payout deviation."""
        if not isfinite(risk_penalty) or risk_penalty < 0:
            raise ValueError("risk_penalty must be finite and nonnegative")
        policies = np.asarray(policies)
        if policies.ndim == 2:
            policies = policies[np.newaxis, ...]
        if policies.ndim != 3 or policies.shape[1:] != POLICY_SHAPE:
            raise ValueError(f"expected policies with shape (n, {POLICY_SHAPE[0]}, 10)")
        if not np.all((policies == HIT) | (policies == STAND)):
            raise ValueError("policies may contain only HIT or STAND")
        scores = np.empty(len(policies))
        cache: dict[bytes, float] = {}
        for index, policy in enumerate(policies.astype(np.uint8, copy=False)):
            actions = policy.tobytes()
            if actions not in cache:
                cache[actions] = self._score(actions, risk_penalty=risk_penalty)
            scores[index] = cache[actions]
        return scores

    def evaluate_random(self, seed: int | np.random.SeedSequence) -> float:
        """Score independent fair Hit/Stand choices on the same shared shoes."""
        rng = np.random.default_rng(seed)
        bits = rng.integers(0, 1 << 32, size=self.num_hands, dtype=np.uint32)
        return self._score(None, tuple(bits.tolist()))


def add_rule_arguments(parser):
    defaults = BlackjackConfig()
    group = parser.add_argument_group("Blackjack rules")
    group.add_argument("--num-decks", type=int, default=defaults.num_decks)
    group.add_argument("--penetration", type=float, default=defaults.penetration)
    group.add_argument("--hit-soft-17", action="store_true", help="dealer hits soft 17")
    group.add_argument("--blackjack-payout", type=float, default=defaults.blackjack_payout)
    group.add_argument("--win-payout", type=float, default=defaults.win_payout)
    group.add_argument("--loss-payout", type=float, default=defaults.loss_payout)
