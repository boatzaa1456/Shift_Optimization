"""
Cycle-based Genetic Algorithm for Shift Scheduling
- Evolve a repeating cycle of length CYCLE_LEN, then tile to full 365 days
- Supports arbitrary number of groups (A..), shifts (e.g., ['M','A','N']) and Holiday 'H'
- Hard/Soft Constraints (penalized):
    * Per day: exactly one group per shift (others = 'H')
    * Max consecutive SAME shift per group <= MAX_RUN (e.g., 4)
    * Max consecutive WORKING days (any shift) per group <= MAX_WORK_STREAK (e.g., 6)  <-- NEW RULE
    * Run/Work-streak checked inside cycle and across cycle boundary (wrap-around) when tiled
- Objectives:
    * Yearly fairness  : sum_g (work_g - T_year)^2
    * Monthly fairness : sum_{m,g} (work_{g,m} - T_m)^2   (weighted)
- Prints MeanAvgPerMonth (average of groups' Average Work/Month) each generation.

Copy–paste to PyCharm and run.
"""

import random
from collections import Counter
from datetime import date, timedelta
from typing import List, Tuple, Dict

# ==============
# CONFIG
# ==============
SEED = 555
DAYS = 365
START_DATE = date(2025, 1, 1)

# Groups & Shifts
NUM_GROUPS = 3
GROUPS = [chr(ord('A') + i) for i in range(NUM_GROUPS)]   # ['A','B','C','D','E','F']
SHIFTS = ['M', 'N']                                # any set, e.g. ['M','N'] or ['D','A','N']
HOLIDAY = 'H'
GROUP_INDEX = {g: idx for idx, g in enumerate(GROUPS)}

# Cycle length pattern: keep a repeating weekly/bi-weekly template
CYCLE_LEN = 14                                     # choose 7 or 14 for realistic repeating pattern

# Hard/soft constraints
MAX_RUN = 4              # max consecutive SAME shift per group
MAX_WORK_STREAK = 6      # NEW RULE: max consecutive working days (any shift)
MIN_REST_DAYS_BETWEEN_SHIFTS = 1  # require at least this many holidays before changing shift type

# GA hyper-parameters
POP_SIZE = 320
GENERATIONS = 1500
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE_BASE = 0.3      # baseline per-cycle-day mutation probability
MUTATION_RATE_MAX = 0.55      # adaptive ceiling when exploration is needed
ELITE_KEEP = 2
IMMIGRANTS_FRAC = 0.10       # % random immigrants each generation
NO_IMPROVE_RESET = 200       # generations without progress before re-diversifying

# Objective/penalty weights
PENALTY_SHIFT_RUN_W = 80.0       # weight for SAME-shift run-length violations
PENALTY_WORK_STREAK_W = 120.0    # weight for > MAX_WORK_STREAK violations (stronger)
PENALTY_SHIFT_TRANSITION_W = 160.0  # weight for unrealistic day-to-day shift changes
YEAR_FAIRNESS_WEIGHT = 0.05
MONTH_OBJECTIVE_WEIGHT = 0.01    # quadratic monthly fairness guardrail

random.seed(SEED)


# ==============
# CALENDAR
# ==============
def make_calendar():
    days = [START_DATE + timedelta(days=i) for i in range(DAYS)]
    months = [d.strftime("%Y-%m") for d in days]
    month_list = []
    seen = set()
    for m in months:
        if m not in seen:
            month_list.append(m)
            seen.add(m)
    month_days = {m: months.count(m) for m in month_list}
    return days, months, month_list, month_days


def precompute_cycle_coverage(months: List[str], month_list: List[str]) -> Tuple[List[int], List[Dict[str, int]]]:
    """Pre-compute how often each cycle index appears in the year and per-month."""
    year_freq = [0 for _ in range(CYCLE_LEN)]
    month_freq = [{m: 0 for m in month_list} for _ in range(CYCLE_LEN)]
    for day_idx, month in enumerate(months):
        cyc_idx = day_idx % CYCLE_LEN
        year_freq[cyc_idx] += 1
        month_freq[cyc_idx][month] += 1
    return year_freq, month_freq


# ==============
# HELPERS
# ==============
def assert_params():
    if NUM_GROUPS < len(SHIFTS):
        raise ValueError("NUM_GROUPS must be >= number of shifts.")
    if CYCLE_LEN < len(SHIFTS):
        raise ValueError("CYCLE_LEN must be >= number of shifts.")
    if len(set(GROUPS)) != NUM_GROUPS:
        raise ValueError("GROUPS must be unique and match NUM_GROUPS.")
    if len(set(SHIFTS)) != len(SHIFTS):
        raise ValueError("SHIFTS must be unique.")


def target_year() -> float:
    # ideal workdays per group per YEAR
    return DAYS * len(SHIFTS) / float(NUM_GROUPS)


def target_months(month_days: Dict[str, int]) -> Dict[str, float]:
    # ideal workdays per group per MONTH m
    return {m: (month_days[m] * len(SHIFTS)) / float(NUM_GROUPS) for m in month_days}


def mean_avg_per_month_from_counts(counts_year: dict) -> float:
    # mean( AverageWorkMonth(group) ) = (mean(workdays_year[group]) / 12)
    return (sum(counts_year.values()) / float(len(counts_year))) / 12.0


def mean_avg_per_month_gap(work_m: Dict[str, Dict[str, int]], month_list: List[str], target_month: Dict[str, float]) -> float:
    """Average absolute deviation from the ideal workdays per month across groups."""
    total = 0.0
    count = 0
    for g in GROUPS:
        for m in month_list:
            total += abs(work_m[g][m] - target_month[m])
            count += 1
    return total / float(count if count else 1)


def compute_month_counts_for_cycle(cyc: List[List[str]], month_freq: List[Dict[str, int]], month_list: List[str]) -> Dict[str, Dict[str, int]]:
    counts = {g: {m: 0 for m in month_list} for g in GROUPS}
    for cyc_idx in range(CYCLE_LEN):
        month_counts = month_freq[cyc_idx]
        row = cyc[cyc_idx]
        for gi, g in enumerate(GROUPS):
            if row[gi] in SHIFTS:
                for m, mc in month_counts.items():
                    if mc:
                        counts[g][m] += mc
    return counts


def balance_cycle_monthly(
    cyc: List[List[str]],
    month_freq: List[Dict[str, int]],
    month_list: List[str],
    target_month: Dict[str, float],
    attempts: int = 3,
) -> None:
    """Heuristic post-mutation adjuster to reduce large monthly imbalances."""
    for _ in range(attempts):
        work_m = compute_month_counts_for_cycle(cyc, month_freq, month_list)
        # locate the largest surplus and deficit
        max_diff = (0.0, None, None)
        min_diff = (0.0, None, None)
        for g in GROUPS:
            for m in month_list:
                diff = work_m[g][m] - target_month[m]
                if diff > max_diff[0]:
                    max_diff = (diff, g, m)
                if diff < min_diff[0]:
                    min_diff = (diff, g, m)

        if max_diff[1] is None or min_diff[1] is None or max_diff[0] <= 0 or min_diff[0] >= 0:
            break

        surplus_group, surplus_month = max_diff[1], max_diff[2]
        deficit_group, _ = min_diff[1], min_diff[2]
        gi_surplus = GROUP_INDEX[surplus_group]
        gi_deficit = GROUP_INDEX[deficit_group]

        candidate_days = [d for d in range(CYCLE_LEN) if month_freq[d][surplus_month] > 0]
        random.shuffle(candidate_days)
        improved = False
        for d in candidate_days:
            row = cyc[d]
            val_surplus = row[gi_surplus]
            if val_surplus not in SHIFTS:
                continue
            val_deficit = row[gi_deficit]
            if val_deficit == HOLIDAY:
                row[gi_deficit] = val_surplus
                row[gi_surplus] = HOLIDAY
                repair_day_row(row)
                improved = True
                break
            elif val_deficit in SHIFTS and val_deficit != val_surplus:
                row[gi_surplus], row[gi_deficit] = val_deficit, val_surplus
                repair_day_row(row)
                improved = True
                break
        if not improved:
            break
def make_empty_cycle() -> List[List[str]]:
    return [[HOLIDAY for _ in GROUPS] for _ in range(CYCLE_LEN)]


def init_random_cycle() -> List[List[str]]:
    cyc = make_empty_cycle()
    for d in range(CYCLE_LEN):
        idxs = list(range(NUM_GROUPS))
        random.shuffle(idxs)
        chosen = idxs[:len(SHIFTS)]
        perm = SHIFTS[:]
        random.shuffle(perm)
        # reset to H
        for gi in range(NUM_GROUPS):
            cyc[d][gi] = HOLIDAY
        # assign shifts
        for s_i, s in enumerate(perm):
            cyc[d][chosen[s_i]] = s
    return cyc


def repair_day_row(row: List[str]) -> None:
    # Ensure exactly one per shift; others H
    pos = {s: [] for s in SHIFTS}
    for gi, v in enumerate(row):
        if v in SHIFTS:
            pos[v].append(gi)
    # release extras to H
    for s, positions in pos.items():
        while len(positions) > 1:
            gi = positions.pop()
            row[gi] = HOLIDAY
    # recompute; fill missing shifts
    pos = {s: [] for s in SHIFTS}
    for gi, v in enumerate(row):
        if v in SHIFTS:
            pos[v].append(gi)
    need = [s for s in SHIFTS if len(pos[s]) == 0]
    if need:
        h_idx = [gi for gi, v in enumerate(row) if v == HOLIDAY]
        random.shuffle(h_idx)
        for s in need:
            gi = h_idx.pop() if h_idx else None
            if gi is None:
                # steal from a shift that exists
                donors = [(ss, pos[ss][0]) for ss in SHIFTS if pos[ss]]
                if donors:
                    _, donor = random.choice(donors)
                    row[donor] = s
            else:
                row[gi] = s
    # clean leftovers
    for gi in range(len(row)):
        if row[gi] not in SHIFTS:
            row[gi] = HOLIDAY


def copy_cycle(cyc: List[List[str]]) -> List[List[str]]:
    return [r[:] for r in cyc]


# ==============
# TILING
# ==============
def tile_cycle(cyc: List[List[str]]) -> List[List[str]]:
    out = []
    i = 0
    while len(out) < DAYS:
        out.append(cyc[i % CYCLE_LEN][:])
        i += 1
    return out


# ==============
# METRICS
# ==============
def work_counts_year(sched: List[List[str]]) -> Dict[str, int]:
    cnt = {g: 0 for g in GROUPS}
    for d in range(DAYS):
        for gi, g in enumerate(GROUPS):
            if sched[d][gi] != HOLIDAY:
                cnt[g] += 1
    return cnt


def work_counts_monthly(sched: List[List[str]], months: List[str], month_list: List[str]) -> Dict[str, Dict[str, int]]:
    w = {g: {m: 0 for m in month_list} for g in GROUPS}
    for d in range(DAYS):
        m = months[d]
        for gi, g in enumerate(GROUPS):
            if sched[d][gi] != HOLIDAY:
                w[g][m] += 1
    return w


def runlength_penalty_cycle_same_shift(cyc: List[List[str]]) -> float:
    """
    Penalty for SAME-shift run-length > MAX_RUN (wrap-around aware).
    Scan two concatenated cycles to catch boundary runs; divide by 2 at end.
    """
    penalty = 0.0
    two = cyc + cyc
    L = len(two)
    for gi, _ in enumerate(GROUPS):
        cur = None
        run = 0
        for d in range(L):
            s = two[d][gi]
            if s == HOLIDAY:
                cur = None; run = 0; continue
            if s == cur:
                run += 1
            else:
                cur = s; run = 1
            if run > MAX_RUN:
                penalty += (run - MAX_RUN)
    return penalty / 2.0


def work_streak_penalty_cycle_any_shift(cyc: List[List[str]]) -> float:
    """
    NEW RULE: Penalty for consecutive WORKING days (any shift) > MAX_WORK_STREAK.
    Also wrap-around aware using two concatenated cycles; divide by 2.
    """
    penalty = 0.0
    two = cyc + cyc
    L = len(two)
    for gi, _ in enumerate(GROUPS):
        streak = 0
        for d in range(L):
            s = two[d][gi]
            if s == HOLIDAY:
                streak = 0
            else:
                streak += 1
                if streak > MAX_WORK_STREAK:
                    penalty += (streak - MAX_WORK_STREAK)
    return penalty / 2.0


def shift_transition_penalty_cycle(cyc: List[List[str]]) -> float:
    """Penalty when a group changes to a different shift without the required rest days."""
    penalty = 0.0
    for gi, _ in enumerate(GROUPS):
        for d in range(CYCLE_LEN):
            current = cyc[d][gi]
            if current not in SHIFTS:
                continue

            # scan forward until we encounter the next working day or looped once around the cycle
            rest_days = 0
            steps = 0
            idx = (d + 1) % CYCLE_LEN
            while steps < CYCLE_LEN:
                nxt = cyc[idx][gi]
                if nxt == HOLIDAY:
                    rest_days += 1
                else:
                    if nxt != current and rest_days < MIN_REST_DAYS_BETWEEN_SHIFTS:
                        penalty += 1.0
                    break
                idx = (idx + 1) % CYCLE_LEN
                steps += 1
    return penalty


def day_feasibility_penalty_cycle(cyc: List[List[str]]) -> float:
    p = 0.0
    for d in range(CYCLE_LEN):
        c = Counter(cyc[d])
        for s in SHIFTS:
            p += abs(c.get(s, 0) - 1) * 10.0
    return p


def fitness_cycle(
    cyc: List[List[str]],
    year_freq: List[int],
    month_freq: List[Dict[str, int]],
    month_list: List[str],
    target_year_value: float,
    target_month: Dict[str, float],
) -> Tuple[float, Dict]:
    """
    Compute fitness on the *tiled* 365-day schedule produced by the cycle.
    """
    # penalties on cycle itself
    p_day = day_feasibility_penalty_cycle(cyc)
    p_run_same = runlength_penalty_cycle_same_shift(cyc) * PENALTY_SHIFT_RUN_W
    p_work_streak = work_streak_penalty_cycle_any_shift(cyc) * PENALTY_WORK_STREAK_W
    p_shift_trans = shift_transition_penalty_cycle(cyc) * PENALTY_SHIFT_TRANSITION_W

    # if infeasible, these penalties will dominate and GA will move away
    counts_y = {g: 0 for g in GROUPS}
    work_m = {g: {m: 0 for m in month_list} for g in GROUPS}

    for cyc_idx in range(CYCLE_LEN):
        freq_year = year_freq[cyc_idx]
        if not freq_year:
            continue
        row = cyc[cyc_idx]
        month_counts = month_freq[cyc_idx]
        for gi, g in enumerate(GROUPS):
            if row[gi] in SHIFTS:
                counts_y[g] += freq_year
                for m, mc in month_counts.items():
                    if mc:
                        work_m[g][m] += mc

    fairness_year = sum((counts_y[g] - target_year_value) ** 2 for g in GROUPS)
    fairness_month = 0.0
    for m in month_list:
        ideal_m = target_month[m]
        for g in GROUPS:
            fairness_month += (work_m[g][m] - ideal_m) ** 2

    mean_avg_gap = mean_avg_per_month_gap(work_m, month_list, target_month)
    mean_avg_workdays = mean_avg_per_month_from_counts(counts_y)

    # fitness emphasises the monthly gap first, with softer fairness + feasibility penalties
    year_abs_gap = sum(abs(counts_y[g] - target_year_value) for g in GROUPS) / float(NUM_GROUPS)

    f = (
        mean_avg_gap
        + YEAR_FAIRNESS_WEIGHT * year_abs_gap
        + MONTH_OBJECTIVE_WEIGHT * fairness_month
        + p_day
        + p_run_same
        + p_work_streak
        + p_shift_trans
    )
    meta = {
        "fairness_year": fairness_year,
        "fairness_month": fairness_month,
        "day_pen": p_day,
        "run_pen_same": p_run_same,
        "work_streak_pen": p_work_streak,
        "shift_transition_pen": p_shift_trans,
        "counts_year": counts_y,
        "work_month": work_m,
        "mean_avg_gap": mean_avg_gap,
        "mean_avg_workdays": mean_avg_workdays,
        "year_abs_gap": year_abs_gap,
    }
    return f, meta


# ==============
# GA OPERATORS (on the cycle)
# ==============
def crossover_cycle(p1: List[List[str]], p2: List[List[str]]) -> Tuple[List[List[str]], List[List[str]]]:
    if random.random() > CROSSOVER_RATE:
        return copy_cycle(p1), copy_cycle(p2)
    cut = random.randint(1, CYCLE_LEN - 1)
    c1, c2 = copy_cycle(p1), copy_cycle(p2)
    for d in range(cut, CYCLE_LEN):
        c1[d], c2[d] = c2[d][:], c1[d][:]
    for d in range(CYCLE_LEN):
        repair_day_row(c1[d]); repair_day_row(c2[d])
    return c1, c2


def mutate_cycle(indiv: List[List[str]], mutation_rate: float) -> None:
    # day-wise local mutation
    for d in range(CYCLE_LEN):
        if random.random() < mutation_rate:
            chosen = random.sample(range(NUM_GROUPS), len(SHIFTS))
            random.shuffle(chosen)
            for gi in range(NUM_GROUPS):
                indiv[d][gi] = HOLIDAY
            for s_i, s in enumerate(SHIFTS):
                indiv[d][chosen[s_i]] = s
            repair_day_row(indiv[d])
    # block-swap two groups across random sub-block (helps large moves but keeps per-day feasible)
    if random.random() < 0.15:
        g1, g2 = random.sample(range(NUM_GROUPS), 2)
        length = random.randint(7, min(14, CYCLE_LEN))
        lo = random.randint(0, CYCLE_LEN - length)
        hi = lo + length
        for d in range(lo, hi):
            indiv[d][g1], indiv[d][g2] = indiv[d][g2], indiv[d][g1]


def tournament_select(pop, fits):
    best = None; best_f = float('inf')
    for _ in range(TOURNAMENT_K):
        i = random.randrange(len(pop))
        if fits[i] < best_f:
            best_f = fits[i]; best = pop[i]
    return copy_cycle(best)


# ==============
# GA MAIN
# ==============
def run_ga_cycle():
    assert_params()
    _, months, month_list, month_days = make_calendar()
    year_freq, month_freq = precompute_cycle_coverage(months, month_list)
    target_year_value = target_year()
    target_month = target_months(month_days)

    # init population
    population = [init_random_cycle() for _ in range(POP_SIZE)]
    fit_cache = [None] * POP_SIZE
    mutation_rate = MUTATION_RATE_BASE

    def eval_pop():
        for i in range(POP_SIZE):
            if fit_cache[i] is None:
                fit_cache[i] = fitness_cycle(
                    population[i],
                    year_freq,
                    month_freq,
                    month_list,
                    target_year_value,
                    target_month,
                )

    best_f = float('inf'); best_ind = None; best_meta = None
    no_improve = 0

    for gen in range(GENERATIONS):
        eval_pop()
        fits = [fit_cache[i][0] for i in range(POP_SIZE)]

        # track best
        improved_this_gen = False
        for i in range(POP_SIZE):
            if fits[i] < best_f:
                best_f = fits[i]
                best_ind = copy_cycle(population[i])
                best_meta = fit_cache[i][1]
                improved_this_gen = True

        if improved_this_gen:
            no_improve = 0
            mutation_rate = max(MUTATION_RATE_BASE, mutation_rate * 0.92)
        else:
            no_improve += 1
            if no_improve % 40 == 0:
                mutation_rate = min(MUTATION_RATE_MAX, mutation_rate * 1.08)

        # elitism
        order = sorted(range(POP_SIZE), key=lambda i: fits[i])
        elites = [copy_cycle(population[i]) for i in order[:ELITE_KEEP]]

        # next generation
        new_pop = elites[:]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            c1, c2 = crossover_cycle(p1, p2)
            mutate_cycle(c1, mutation_rate)
            mutate_cycle(c2, mutation_rate)
            balance_cycle_monthly(c1, month_freq, month_list, target_month)
            balance_cycle_monthly(c2, month_freq, month_list, target_month)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        # random immigrants to maintain diversity
        num_imm = int(POP_SIZE * IMMIGRANTS_FRAC)
        for k in range(num_imm):
            new_pop[-1 - k] = init_random_cycle()

        if no_improve >= NO_IMPROVE_RESET:
            # forcefully diversify by refreshing half of the non-elite population
            for idx in range(ELITE_KEEP, POP_SIZE):
                if random.random() < 0.5:
                    new_pop[idx] = init_random_cycle()
            mutation_rate = min(MUTATION_RATE_MAX, mutation_rate * 1.15)
            no_improve = 0

        population = new_pop
        fit_cache = [None] * POP_SIZE

        # progress log with MeanAvgPerMonth
        if (gen + 1) % 50 == 0 or gen == 0:
            yr = best_meta["fairness_year"] if best_meta else float('nan')
            mo = best_meta["fairness_month"] if best_meta else float('nan')
            mean_avg_gap = best_meta["mean_avg_gap"] if best_meta else float('nan')
            mean_avg = best_meta["mean_avg_workdays"] if best_meta else float('nan')
            year_abs_gap = best_meta["year_abs_gap"] if best_meta else float('nan')
            print(
                f"Gen {gen+1:4d} | best_f={best_f:.4f} | MeanAvgPerMonthGap={mean_avg_gap:.4f} "
                f"| meanWorkMonth={mean_avg:.2f} | year_sq={yr:.2f} | year_abs={year_abs_gap:.2f} "
                f"| monthW*={MONTH_OBJECTIVE_WEIGHT*mo:.2f} "
                f"| run_same_pen={best_meta['run_pen_same']:.2f} | work_streak_pen={best_meta['work_streak_pen']:.2f} "
                f"| shift_change_pen={best_meta['shift_transition_pen']:.2f} | day_pen={best_meta['day_pen']:.2f} "
                f"| mut_rate={mutation_rate:.2f} | CYCLE_LEN={CYCLE_LEN}"
            )

        if best_meta and best_meta["mean_avg_gap"] <= 0.01 and best_meta["year_abs_gap"] <= 0.5:
            print(
                f"Early stop at generation {gen+1}: MeanAvgPerMonthGap={best_meta['mean_avg_gap']:.4f}, "
                f"year_abs_gap={best_meta['year_abs_gap']:.2f}"
            )
            break

    return best_ind, best_meta


def cycle_to_shift_table(cyc: List[List[str]]) -> Dict[str, List[str]]:
    table = {s: [] for s in SHIFTS}
    table[HOLIDAY] = []
    for d in range(CYCLE_LEN):
        shift_assignment = {s: '-' for s in SHIFTS}
        holiday_groups = []
        for gi, g in enumerate(GROUPS):
            val = cyc[d][gi]
            if val == HOLIDAY:
                holiday_groups.append(g)
            elif val in SHIFTS:
                if shift_assignment[val] == '-':
                    shift_assignment[val] = g
                else:
                    shift_assignment[val] += f"/{g}"
        for s in SHIFTS:
            table[s].append(shift_assignment[s])
        table[HOLIDAY].append(",".join(holiday_groups) if holiday_groups else '-')
    return table


def summarize_cycle(cyc, meta):
    print("\n=== BEST (Cycle) SUMMARY ===")
    print(f"CYCLE_LEN = {CYCLE_LEN} days")
    T = target_year()
    for g in GROUPS:
        w = meta["counts_year"][g]
        print(f"  {g}: workdays/year = {w}  (diff {w - T:+.2f})")
    print(f"Year fairness: {meta['fairness_year']:.2f}")
    print(f"Month fairness(weighted): {MONTH_OBJECTIVE_WEIGHT*meta['fairness_month']:.2f}")
    print(f"Run same-shift penalty: {meta['run_pen_same']:.2f}")
    print(f"Work-streak (> {MAX_WORK_STREAK}) penalty: {meta['work_streak_pen']:.2f}")
    print(f"Shift change penalty: {meta['shift_transition_pen']:.2f}")
    print(f"Day feasibility penalty: {meta['day_pen']:.2f}")
    mean_avg = meta.get("mean_avg_workdays", mean_avg_per_month_from_counts(meta["counts_year"]))
    mean_gap = meta.get("mean_avg_gap")
    year_abs = meta.get("year_abs_gap")
    print(f"\nMean Average Work Days per Month (across groups): {mean_avg:.2f}")
    if mean_gap is not None:
        print(f"MeanAvgPerMonthGap (objective): {mean_gap:.4f}")
    if year_abs is not None:
        print(f"Yearly average absolute gap: {year_abs:.2f}")

    # Preview first 14 days of the cycle
    print("\nCycle pattern (group per shift each day):")
    table = cycle_to_shift_table(cyc)
    day_labels = [f"{d+1:>3}" for d in range(CYCLE_LEN)]
    header = ["Shift"] + day_labels
    col_width = max(5, max(len(g) for row in table.values() for g in row))
    print(" ".join(h.rjust(col_width) for h in header))
    for shift_name in SHIFTS + [HOLIDAY]:
        row = [shift_name] + table[shift_name]
        print(" ".join(cell.rjust(col_width) for cell in row))


if __name__ == "__main__":
    best_cycle, info = run_ga_cycle()
    summarize_cycle(best_cycle, info)
