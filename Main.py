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
SHIFTS = ['M','N']                                   # any set, e.g. ['M','N'] or ['D','A','N']
HOLIDAY = 'H'

# Cycle length (try 28 / 33 / 35 / 42 ...)
CYCLE_LEN = 42

# Hard/soft constraints
MAX_RUN = 4              # max consecutive SAME shift per group
MAX_WORK_STREAK = 6      # NEW RULE: max consecutive working days (any shift)

# GA hyper-parameters
POP_SIZE = 480
GENERATIONS = 3000
TOURNAMENT_K = 3
CROSSOVER_RATE = 0.9
MUTATION_RATE = 0.35         # per-cycle-day mutation probability
ELITE_KEEP = 2
IMMIGRANTS_FRAC = 0.10       # % random immigrants each generation

# Objective/penalty weights
PENALTY_SHIFT_RUN_W = 80.0       # weight for SAME-shift run-length violations
PENALTY_WORK_STREAK_W = 120.0    # weight for > MAX_WORK_STREAK violations (stronger)
MONTH_OBJECTIVE_WEIGHT = 0.07    # weight for monthly fairness term

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


def day_feasibility_penalty_cycle(cyc: List[List[str]]) -> float:
    p = 0.0
    for d in range(CYCLE_LEN):
        c = Counter(cyc[d])
        for s in SHIFTS:
            p += abs(c.get(s, 0) - 1) * 10.0
    return p


def fitness_cycle(cyc: List[List[str]], months: List[str], month_list: List[str], month_days: Dict[str, int]) -> Tuple[float, Dict]:
    """
    Compute fitness on the *tiled* 365-day schedule produced by the cycle.
    """
    # penalties on cycle itself
    p_day = day_feasibility_penalty_cycle(cyc)
    p_run_same = runlength_penalty_cycle_same_shift(cyc) * PENALTY_SHIFT_RUN_W
    p_work_streak = work_streak_penalty_cycle_any_shift(cyc) * PENALTY_WORK_STREAK_W

    # if infeasible, these penalties will dominate and GA will move away
    sched = tile_cycle(cyc)

    # yearly fairness
    counts_y = work_counts_year(sched)
    T_year = target_year()
    fairness_year = sum((counts_y[g] - T_year) ** 2 for g in GROUPS)

    # monthly fairness
    work_m = work_counts_monthly(sched, months, month_list)
    Tm = target_months(month_days)
    fairness_month = 0.0
    for m in month_list:
        for g in GROUPS:
            fairness_month += (work_m[g][m] - Tm[m]) ** 2

    f = fairness_year + MONTH_OBJECTIVE_WEIGHT * fairness_month + p_day + p_run_same + p_work_streak
    meta = {
        "fairness_year": fairness_year,
        "fairness_month": fairness_month,
        "day_pen": p_day,
        "run_pen_same": p_run_same,
        "work_streak_pen": p_work_streak,
        "counts_year": counts_y
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


def mutate_cycle(indiv: List[List[str]]) -> None:
    # day-wise local mutation
    for d in range(CYCLE_LEN):
        if random.random() < MUTATION_RATE:
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

    # init population
    population = [init_random_cycle() for _ in range(POP_SIZE)]
    fit_cache = [None] * POP_SIZE

    def eval_pop():
        for i in range(POP_SIZE):
            if fit_cache[i] is None:
                fit_cache[i] = fitness_cycle(population[i], months, month_list, month_days)

    best_f = float('inf'); best_ind = None; best_meta = None

    for gen in range(GENERATIONS):
        eval_pop()
        fits = [fit_cache[i][0] for i in range(POP_SIZE)]

        # track best
        for i in range(POP_SIZE):
            if fits[i] < best_f:
                best_f = fits[i]; best_ind = copy_cycle(population[i]); best_meta = fit_cache[i][1]

        # elitism
        order = sorted(range(POP_SIZE), key=lambda i: fits[i])
        elites = [copy_cycle(population[i]) for i in order[:ELITE_KEEP]]

        # next generation
        new_pop = elites[:]
        while len(new_pop) < POP_SIZE:
            p1 = tournament_select(population, fits)
            p2 = tournament_select(population, fits)
            c1, c2 = crossover_cycle(p1, p2)
            mutate_cycle(c1); mutate_cycle(c2)
            new_pop.append(c1)
            if len(new_pop) < POP_SIZE:
                new_pop.append(c2)

        # random immigrants to maintain diversity
        num_imm = int(POP_SIZE * IMMIGRANTS_FRAC)
        for k in range(num_imm):
            new_pop[-1 - k] = init_random_cycle()

        population = new_pop
        fit_cache = [None] * POP_SIZE

        # progress log with MeanAvgPerMonth
        if (gen + 1) % 50 == 0 or gen == 0:
            yr = best_meta["fairness_year"]
            mo = best_meta["fairness_month"]
            mean_avg = mean_avg_per_month_from_counts(best_meta["counts_year"])
            print(
                f"Gen {gen+1:4d} | best_f={best_f:.2f} | year={yr:.2f} | monthW*={MONTH_OBJECTIVE_WEIGHT*mo:.2f} "
                f"| run_same_pen={best_meta['run_pen_same']:.2f} | work_streak_pen={best_meta['work_streak_pen']:.2f} "
                f"| day_pen={best_meta['day_pen']:.2f} | MeanAvgPerMonth={mean_avg:.2f} | CYCLE_LEN={CYCLE_LEN}"
            )

    return best_ind, best_meta


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
    print(f"Day feasibility penalty: {meta['day_pen']:.2f}")
    mean_avg = mean_avg_per_month_from_counts(meta["counts_year"])
    print(f"\nMean Average Work Days per Month (across groups): {mean_avg:.2f}")

    # Preview first 14 days of the cycle
    print("\nFirst cycle block preview:")
    header = "Day  " + "  ".join(GROUPS)
    print(header)
    for d in range(min(CYCLE_LEN, 14)):
        print(f"{d+1:>3}  " + "  ".join(cyc[d]))


if __name__ == "__main__":
    best_cycle, info = run_ga_cycle()
    summarize_cycle(best_cycle, info)
