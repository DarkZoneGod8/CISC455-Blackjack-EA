"""Seedable single-player six-deck Blackjack with fixed total-dependent play.

Card values are 2 through 10, with aces encoded as 1. Bets use true count
before the opening deal. All cards consumed in the round, including the
dealer hole card revealed at settlement, contribute to the next count.
"""

from dataclasses import dataclass

import numpy as np


SHOE_SIZE = 312
CUT_REMAINING = 78
START_BANKROLL = 200.0
RANKS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 1)
_DECK = np.repeat(RANKS, [4, 4, 4, 4, 4, 4, 4, 4, 16, 4]).astype(np.uint8)
_RANK_INDEX = (0, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8)
_ONE_HOT = np.eye(10, dtype=np.int16)[list(_RANK_INDEX)]

# Columns: 2 3 4 5 6 7 8 9 T A. D = double or hit, F = double or stand.
HARD = {
    **dict.fromkeys(range(4, 9), "HHHHHHHHHH"),
    9:  "HDDDDHHHHH",
    10: "DDDDDDDDHH",
    11: "DDDDDDDDDH",
    12: "HHSSSHHHHH",
    **dict.fromkeys(range(13, 17), "SSSSSHHHHH"),
    **dict.fromkeys(range(17, 22), "SSSSSSSSSS"),
}
SOFT = {
    12: "HHHHHHHHHH",
    13: "HHHDDHHHHH",
    14: "HHHDDHHHHH",
    15: "HHDDDHHHHH",
    16: "HHDDDHHHHH",
    17: "HDDDDHHHHH",
    18: "SFFFFSSHHH",
    **dict.fromkeys(range(19, 22), "SSSSSSSSSS"),
}
# A non-split entry delegates to the hard/soft table.
PAIRS = {
    1:  "PPPPPPPPPP",
    2:  "PPPPPP----",
    3:  "PPPPPP----",
    4:  "---PP-----",
    5:  "----------",
    6:  "PPPPP-----",
    7:  "PPPPPP----",
    8:  "PPPPPPPPPP",
    9:  "PPPPP-PP--",
    10: "----------",
}


def _total(low_total: int, aces: int) -> int:
    return low_total + (10 if aces and low_total <= 11 else 0)


def _play_hand(cards, position, low_total, aces, up_column, limit, stand):
    """Return final position, total and wager multiple for one hand."""
    total = _total(low_total, aces)
    if stand:
        return position, total, 1
    while total < 21:
        if total != low_total:
            action = SOFT[total][up_column]
        else:
            action = HARD[total][up_column]
        can_double = limit >= 2 and action in ("D", "F")
        if action == "S" or (action == "F" and not can_double):
            break
        card = cards[position]
        position += 1
        low_total += card
        aces += card == 1
        total = _total(low_total, aces)
        if can_double:
            return position, total, 2
        limit = 1
    return position, total, 1


def _play_round(cards, start, limit=4):
    """Play one round with at most `limit` base wagers available.

    Return the next card offset, net in half-wagers, peak committed wagers,
    and settled hand wins and count. The dealer peeks before player actions.
    """
    first, up, second, hole = cards[start:start + 4]
    position = start + 4
    player_natural = first + second == 11 and (first == 1 or second == 1)
    dealer_natural = up + hole == 11 and (up == 1 or hole == 1)
    if player_natural or dealer_natural:
        profit = 0 if player_natural and dealer_natural else (3 if player_natural else -2)
        return position, profit, 1, int(profit > 0), 1

    column = _RANK_INDEX[up]
    if first == second and limit >= 2 and PAIRS[first][column] == "P":
        # Play the first split hand before dealing the second hand's extra card.
        extra = cards[position]
        position, total1, wager1 = _play_hand(
            cards, position + 1, first + extra, int(first == 1) + (extra == 1),
            column, limit - 1, first == 1,
        )
        extra = cards[position]
        position, total2, wager2 = _play_hand(
            cards, position + 1, second + extra, int(second == 1) + (extra == 1),
            column, limit - wager1, second == 1,
        )
        hands = ((total1, wager1), (total2, wager2))
    else:
        position, total, wager = _play_hand(
            cards, position, first + second, int(first == 1) + (second == 1),
            column, limit, False,
        )
        hands = ((total, wager),)

    dealer_low = up + hole
    dealer_aces = int(up == 1) + (hole == 1)
    dealer = _total(dealer_low, dealer_aces)
    if any(total <= 21 for total, _ in hands):
        while dealer < 17:
            card = cards[position]
            position += 1
            dealer_low += card
            dealer_aces += card == 1
            dealer = _total(dealer_low, dealer_aces)

    profit = exposure = wins = 0
    for total, wager in hands:
        exposure += wager
        if total > 21:
            profit -= 2 * wager
        elif dealer > 21 or total > dealer:
            profit += 2 * wager
            wins += 1
        elif total < dealer:
            profit -= 2 * wager
    return position, profit, exposure, wins, len(hands)


def generate_shoes(n_shoes: int, seed: int | np.random.Generator) -> np.ndarray:
    """Generate complete shuffled shoes for Common Random Numbers evaluation."""
    if n_shoes < 1:
        raise ValueError("n_shoes must be positive")
    rng = np.random.default_rng(seed)
    decks = np.tile(_DECK, 6)
    return np.array([rng.permutation(decks) for _ in range(n_shoes)])


@dataclass(frozen=True)
class ShoeBatch:
    """Raw shoes and cached round paths when every Basic Strategy action is affordable."""

    shoes: tuple[tuple[int, ...], ...]
    offsets: np.ndarray
    counts: np.ndarray
    remaining_decks: np.ndarray
    payout: np.ndarray
    exposure: np.ndarray
    wins: np.ndarray
    hands: np.ndarray


def prepare_shoes(shoes: np.ndarray) -> ShoeBatch:
    """Cache fixed-play outcomes, with one count feature per observed rank."""
    if shoes.ndim != 2 or shoes.shape[1] != SHOE_SIZE or not len(shoes):
        raise ValueError("expected one or more 312-card shoes")
    cards = tuple(tuple(row) for row in shoes.tolist())
    offsets = [0]
    positions, shoe_indices, payout, exposure, wins, hands = [], [], [], [], [], []
    for shoe_index, shoe in enumerate(cards):
        position = 0
        while SHOE_SIZE - position > CUT_REMAINING:
            positions.append(position)
            shoe_indices.append(shoe_index)
            position, net, risk, won, played = _play_round(shoe, position)
            payout.append(net)
            exposure.append(risk)
            wins.append(won)
            hands.append(played)
        offsets.append(len(positions))
    # prefix[:, p] describes exactly the p cards observed before a new round.
    prefix = np.zeros((len(cards), SHOE_SIZE + 1, 10), dtype=np.int16)
    np.cumsum(_ONE_HOT[shoes], axis=1, dtype=np.int16, out=prefix[:, 1:])
    starts = np.array(offsets[:-1])
    remaining = np.maximum((SHOE_SIZE - np.array(positions)) / 52.0, 0.5)
    return ShoeBatch(
        cards, np.array(offsets), prefix[shoe_indices, positions].astype(float),
        remaining, np.array(payout, dtype=np.int8),
        2 * np.array(exposure, dtype=np.int8),
        np.add.reduceat(wins, starts), np.add.reduceat(hands, starts),
    )


@dataclass(frozen=True)
class Evaluation:
    """One row per policy, with per-allocated-shoe profit and session outcomes."""

    profits: np.ndarray
    wins: np.ndarray
    hands: np.ndarray
    ruined: np.ndarray
    zero_bankroll: np.ndarray


def _replay_shoe(shoe, weights, ramp, cash):
    """Replay a shoe when the cached wagers cannot be fully funded."""
    position = wins = hands = 0
    seen = np.zeros(10, dtype=float)
    while SHOE_SIZE - position > CUT_REMAINING and cash >= 2:
        remaining = max((SHOE_SIZE - position) / 52.0, 0.5)
        true_count = float(seen @ weights) / remaining
        bracket = int(true_count >= 1) + int(true_count >= 2) + int(true_count >= 3)
        bet = min(int(ramp[bracket]), cash // 2)
        end, payout, _, won, played = _play_round(shoe, position, min(cash // (2 * bet), 4))
        cash += bet * payout
        wins += won
        hands += played
        for index in range(position, end):
            seen[_RANK_INDEX[shoe[index]]] += 1
        position = end
    return cash, wins, hands


def evaluate(batch: ShoeBatch, weights: np.ndarray, ramps: np.ndarray,
             bankroll: float = START_BANKROLL) -> Evaluation:
    """Evaluate policies side by side, carrying bankroll across the whole batch.

    Common Random Numbers (CRN) give each policy the same raw shuffled shoes.
    Bet-independent play usually shares every observed round as well. When
    cash constrains play, replay that shoe and resume the shared path at the
    next shuffle. Money is tracked in exact half units throughout settlement.
    """
    weights = np.atleast_2d(np.asarray(weights, dtype=float))
    ramps = np.atleast_2d(np.floor(np.asarray(ramps) + 0.5))
    if weights.shape[1] != 10 or ramps.shape != (len(weights), 4):
        raise ValueError("expected ten weights and four ramp entries per policy")
    if not np.all((ramps >= 1) & (ramps <= 12)):
        raise ValueError("rounded bets must be between 1 and 12")
    ramps = ramps.astype(np.int8)
    bankroll2 = round(2 * bankroll)
    if bankroll2 < 2 or bankroll2 != 2 * bankroll:
        raise ValueError("bankroll must be at least one unit in half-unit increments")

    # Row-major policy paths make each cumulative sum contiguous.
    true_count = (weights @ batch.counts.T) / batch.remaining_decks
    brackets = (true_count >= 1).astype(np.int8)
    brackets += true_count >= 2
    brackets += true_count >= 3
    bets = np.take_along_axis(ramps, brackets, axis=1)
    increments = bets * batch.payout
    after = np.cumsum(increments, axis=1, dtype=np.int32)
    starts = batch.offsets[:-1]
    at_end = after[:, batch.offsets[1:] - 1]
    at_start = np.pad(at_end[:, :-1], ((0, 0), (1, 0)))
    raw_profit = at_end - at_start
    # Minimum starting cash to afford every cached action in a shoe.
    needed = np.maximum.reduceat(bets * batch.exposure - (after - increments), starts, axis=1)

    profits = raw_profit.copy()
    total_wins = np.full(len(weights), batch.wins.sum())
    total_hands = np.full(len(weights), batch.hands.sum())
    end_cash = bankroll2 + at_end[:, -1]
    risky = np.flatnonzero((needed > bankroll2).any(axis=1))
    for policy in risky:
        cash_offset = bankroll2
        shoe_index = 0
        while shoe_index < len(starts):
            failures = np.flatnonzero(needed[policy, shoe_index:] > cash_offset)
            if not failures.size:
                break
            shoe_index += int(failures[0])
            cash = int(cash_offset + at_start[policy, shoe_index])
            end, wins, hands = _replay_shoe(
                batch.shoes[shoe_index], weights[policy], ramps[policy], cash,
            )
            profits[policy, shoe_index] = end - cash
            total_wins[policy] += wins - batch.wins[shoe_index]
            total_hands[policy] += hands - batch.hands[shoe_index]
            if end < 2:
                profits[policy, shoe_index + 1:] = 0
                total_wins[policy] -= batch.wins[shoe_index + 1:].sum()
                total_hands[policy] -= batch.hands[shoe_index + 1:].sum()
                end_cash[policy] = end
                break
            cash_offset = end - at_end[policy, shoe_index]
            end_cash[policy] = cash_offset + at_end[policy, -1]
            shoe_index += 1
    return Evaluation(profits * 0.5, total_wins, total_hands, end_cash < 2, end_cash == 0)
