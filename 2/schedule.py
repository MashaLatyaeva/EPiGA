"""Discrete genetic algorithm for a constrained weekly duty roster (variant 10)."""

from __future__ import annotations

import argparse
import csv
import html
import json
import random
import statistics
import time
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data.json"
DEFAULT_SEED = 2026
DEFAULT_POPULATION = 60
DEFAULT_GENERATIONS = 80
DEFAULT_RUNS = 20
MAX_DUTIES_PER_EMPLOYEE = 4
HARD_PENALTY = 1000.0


def build_data(seed: int = DEFAULT_SEED) -> dict[str, object]:
    employees = [f"Сотрудник {index + 1:02}" for index in range(10)]
    shifts = [
        {
            "id": f"d{day + 1}_{kind}",
            "day": day,
            "day_name": ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"][day],
            "kind": kind,
            "label": f"{['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'][day]} {kind}",
        }
        for day in range(7)
        for kind in ("Утро", "Вечер", "Ночь")
    ]
    conflicts: set[tuple[int, int]] = set()
    for day in range(7):
        indexes = [day * 3, day * 3 + 1, day * 3 + 2]
        for left in range(3):
            for right in range(left + 1, 3):
                conflicts.add((indexes[left], indexes[right]))
        if day < 6:
            conflicts.add((day * 3 + 2, (day + 1) * 3))

    rng = random.Random(seed)
    preferences = [
        [rng.randint(0, 12) for _ in shifts]
        for _ in employees
    ]
    return {
        "description": "Синтетический недельный план дежурств; предпочтения 0 (лучше) … 12",
        "seed": seed,
        "employees": employees,
        "shifts": shifts,
        "max_duties_per_employee": MAX_DUTIES_PER_EMPLOYEE,
        "hard_conflicts": [list(pair) for pair in sorted(conflicts)],
        "preference_costs": preferences,
    }


def load_data() -> dict[str, object]:
    if not DATA_FILE.exists():
        DATA_FILE.write_text(
            json.dumps(build_data(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return json.loads(DATA_FILE.read_text(encoding="utf-8"))


def violations(
    chromosome: Sequence[int], conflicts: Sequence[Sequence[int]], employee_count: int
) -> int:
    count = sum(
        chromosome[left] == chromosome[right]
        for left, right in conflicts
    )
    loads = [chromosome.count(employee) for employee in range(employee_count)]
    count += sum(max(0, load - MAX_DUTIES_PER_EMPLOYEE) for load in loads)
    return int(count)


def repair(
    chromosome: Sequence[int], data: dict[str, object], rng: random.Random
) -> list[int]:
    shifts = data["shifts"]
    employees = data["employees"]
    conflicts = data["hard_conflicts"]
    preferences = data["preference_costs"]
    conflict_map: list[set[int]] = [set() for _ in shifts]
    for left, right in conflicts:
        conflict_map[left].add(right)
        conflict_map[right].add(left)

    repaired: list[int] = []
    loads = [0] * len(employees)
    night_counts = [0] * len(employees)
    for shift_index, shift in enumerate(shifts):
        forbidden = {
            repaired[previous]
            for previous in conflict_map[shift_index]
            if previous < len(repaired)
        }
        candidates = [
            employee
            for employee in range(len(employees))
            if employee not in forbidden and loads[employee] < MAX_DUTIES_PER_EMPLOYEE
        ]
        if not candidates:
            raise RuntimeError(f"No feasible employee for shift {shift['id']}")

        preferred = min(len(employees) - 1, max(0, int(chromosome[shift_index])))
        scored: list[tuple[float, int]] = []
        for employee in candidates:
            marginal = float(preferences[employee][shift_index])
            marginal += 2.0 * (loads[employee] + 1 - len(shifts) / len(employees)) ** 2
            if shift["kind"] == "Ночь":
                marginal += 1.5 * (night_counts[employee] + 1 - 0.7) ** 2
            if employee != preferred:
                marginal += 0.15
            scored.append((marginal + rng.random() * 0.8, employee))
        chosen = min(scored)[1]
        repaired.append(chosen)
        loads[chosen] += 1
        if shift["kind"] == "Ночь":
            night_counts[chosen] += 1
    return repaired


def soft_cost(chromosome: Sequence[int], data: dict[str, object]) -> float:
    employees = data["employees"]
    shifts = data["shifts"]
    preferences = data["preference_costs"]
    loads = [chromosome.count(employee) for employee in range(len(employees))]
    nights = [
        sum(
            chromosome[index] == employee and shift["kind"] == "Ночь"
            for index, shift in enumerate(shifts)
        )
        for employee in range(len(employees))
    ]
    target_load = len(shifts) / len(employees)
    target_nights = 7 / len(employees)
    preference_cost = sum(
        preferences[employee][shift_index]
        for shift_index, employee in enumerate(chromosome)
    )
    workload_cost = 2.0 * sum((load - target_load) ** 2 for load in loads)
    night_balance_cost = 1.5 * sum((count - target_nights) ** 2 for count in nights)
    return preference_cost + workload_cost + night_balance_cost


def objective(
    chromosome: Sequence[int], data: dict[str, object], strategy: str
) -> tuple[float, int]:
    violation_count = violations(
        chromosome, data["hard_conflicts"], len(data["employees"])
    )
    penalty = HARD_PENALTY * violation_count if strategy == "penalty" else 0.0
    return soft_cost(chromosome, data) + penalty, violation_count


def _tournament(
    population: Sequence[Sequence[int]],
    scores: Sequence[float],
    rng: random.Random,
    size: int = 3,
) -> list[int]:
    contestants = rng.sample(range(len(population)), size)
    return list(population[min(contestants, key=scores.__getitem__)])


def _crossover(first: Sequence[int], second: Sequence[int], rng: random.Random) -> list[int]:
    return [left if rng.random() < 0.5 else right for left, right in zip(first, second)]


def _mutate(
    chromosome: list[int],
    rng: random.Random,
    mutation: str,
    probability: float = 0.12,
) -> None:
    if mutation == "reassign":
        for index in range(len(chromosome)):
            if rng.random() < probability:
                old = chromosome[index]
                chromosome[index] = rng.randrange(10)
                if chromosome[index] == old:
                    chromosome[index] = (old + rng.randrange(1, 10)) % 10
    elif mutation == "swap":
        if rng.random() < probability:
            first, second = rng.sample(range(len(chromosome)), 2)
            chromosome[first], chromosome[second] = chromosome[second], chromosome[first]
    else:
        raise ValueError(f"Unknown mutation operator: {mutation}")


def run_ga(
    *,
    data: dict[str, object],
    seed: int,
    population_size: int,
    generations: int,
    strategy: str,
    mutation: str,
) -> tuple[list[int], list[float], int]:
    rng = random.Random(seed)
    shift_count = len(data["shifts"])

    def prepare(chromosome: Sequence[int]) -> list[int]:
        if strategy == "repair":
            return repair(chromosome, data, rng)
        return list(chromosome)

    population = [
        prepare([rng.randrange(len(data["employees"])) for _ in range(shift_count)])
        for _ in range(population_size)
    ]
    history: list[float] = []
    evaluations = len(population)
    scores = [objective(item, data, strategy)[0] for item in population]

    for generation in range(generations + 1):
        best_index = min(range(population_size), key=scores.__getitem__)
        history.append(scores[best_index])
        if generation == generations:
            break
        next_population = [list(population[best_index])]
        while len(next_population) < population_size:
            first = _tournament(population, scores, rng)
            second = _tournament(population, scores, rng)
            child = _crossover(first, second, rng)
            _mutate(child, rng, mutation)
            next_population.append(prepare(child))
        population = next_population
        scores = [objective(item, data, strategy)[0] for item in population]
        evaluations += len(population)

    best_index = min(range(population_size), key=scores.__getitem__)
    return list(population[best_index]), history, evaluations


def run_random_search(
    *, data: dict[str, object], seed: int, candidates: int
) -> tuple[list[int], list[float], int]:
    rng = random.Random(seed)
    best: list[int] | None = None
    best_score = float("inf")
    history: list[float] = []
    for _ in range(candidates):
        candidate = repair(
            [rng.randrange(len(data["employees"])) for _ in data["shifts"]], data, rng
        )
        score = soft_cost(candidate, data)
        if score < best_score:
            best, best_score = candidate, score
        history.append(best_score)
    assert best is not None
    return best, history, candidates


def _summary(rows: Sequence[dict[str, object]]) -> dict[str, float]:
    costs = [float(row["cost"]) for row in rows]
    return {
        "best": min(costs),
        "mean": statistics.fmean(costs),
        "median": statistics.median(costs),
        "stdev": statistics.stdev(costs) if len(costs) > 1 else 0.0,
        "worst": max(costs),
        "feasible_rate": statistics.fmean(float(row["feasible"]) for row in rows),
        "mean_violations": statistics.fmean(float(row["violations"]) for row in rows),
    }


def _write_convergence_svg(
    path: Path, series: dict[str, list[list[float]]]
) -> None:
    width, height = 900, 520
    left, top, plot_width, plot_height = 78, 60, 780, 360
    all_values = [value for runs in series.values() for run in runs for value in run]
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * 0.07, 0.5)
    low, high = low - padding, high + padding
    colors = ["#2563eb", "#dc2626", "#059669", "#6b7280"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="450" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">План дежурств: лучшее значение целевой функции</text>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = top + plot_height * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-8}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.1f}</text>')
    for series_index, (name, runs) in enumerate(series.items()):
        color = colors[series_index]
        length = len(runs[0])
        mean = [statistics.fmean(run[index] for run in runs) for index in range(length)]
        x_values = [left + index / max(length - 1, 1) * plot_width for index in range(length)]
        y_values = [top + (high - value) / (high - low) * plot_height for value in mean]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(x_values, y_values))
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = 48
        legend_x = left + series_index * 190
        parts.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x+22}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+28}" y="{legend_y+4}" font-family="sans-serif" font-size="11">{html.escape(name)}</text>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#111827"/>',
            f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">Поколение / блок случайных кандидатов</text>',
            '</svg>',
        ]
    )
    path.write_text("\n".join(parts), encoding="utf-8")


def run_experiments(
    *, runs: int, seed: int, population_size: int, generations: int
) -> dict[str, dict[str, float]]:
    data = load_data()
    configurations = [
        ("Ремонт + переназначение", "repair", "reassign"),
        ("Штраф + переназначение", "penalty", "reassign"),
        ("Ремонт + обмен", "repair", "swap"),
    ]
    rows: list[dict[str, object]] = []
    convergence: dict[str, list[list[float]]] = {}
    best_schedules: dict[str, tuple[float, list[int]]] = {}
    budget = population_size * (generations + 1)

    for config_name, strategy, mutation in configurations:
        convergence[config_name] = []
        for run_index in range(runs):
            run_seed = seed + run_index
            started = time.perf_counter()
            schedule, history, evaluations = run_ga(
                data=data,
                seed=run_seed,
                population_size=population_size,
                generations=generations,
                strategy=strategy,
                mutation=mutation,
            )
            elapsed = time.perf_counter() - started
            cost, violation_count = objective(schedule, data, strategy)
            convergence[config_name].append(history)
            row = {
                "configuration": config_name,
                "run": run_index + 1,
                "seed": run_seed,
                "cost": cost,
                "feasible": int(violation_count == 0),
                "violations": violation_count,
                "evaluations": evaluations,
                "seconds": elapsed,
            }
            rows.append(row)
            if config_name not in best_schedules or cost < best_schedules[config_name][0]:
                best_schedules[config_name] = (cost, schedule)

    random_name = "Случайный допустимый поиск"
    convergence[random_name] = []
    for run_index in range(runs):
        run_seed = seed + run_index + 1_000_000
        started = time.perf_counter()
        schedule, history, evaluations = run_random_search(
            data=data, seed=run_seed, candidates=budget
        )
        elapsed = time.perf_counter() - started
        cost = soft_cost(schedule, data)
        convergence[random_name].append(history)
        rows.append(
            {
                "configuration": random_name,
                "run": run_index + 1,
                "seed": run_seed,
                "cost": cost,
                "feasible": 1,
                "violations": 0,
                "evaluations": evaluations,
                "seconds": elapsed,
            }
        )
        if random_name not in best_schedules or cost < best_schedules[random_name][0]:
            best_schedules[random_name] = (cost, schedule)

    summaries = {
        name: _summary([row for row in rows if row["configuration"] == name])
        for name in convergence
    }
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (ROOT / "best_schedules.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["configuration", "shift", "employee"])
        for name, (_, schedule) in best_schedules.items():
            for shift, employee in zip(data["shifts"], schedule):
                writer.writerow([name, shift["label"], data["employees"][employee]])

    metrics = {
        "variant": 10,
        "student_number": 8,
        "group": "517",
        "year": 2026,
        "seed": seed,
        "runs": runs,
        "population_size": population_size,
        "generations": generations,
        "evaluations_per_run": budget,
        "events": len(data["shifts"]),
        "hard_conflict_pairs": len(data["hard_conflicts"]),
        "summaries": summaries,
        "best_solutions": {
            name: {
                "cost": cost,
                "schedule": [
                    {"shift": shift["label"], "employee": data["employees"][employee]}
                    for shift, employee in zip(data["shifts"], schedule)
                ],
            }
            for name, (cost, schedule) in best_schedules.items()
        },
    }
    (ROOT / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _write_convergence_svg(ROOT / "convergence.svg", convergence)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print(f"Данные: {len(data['shifts'])} событий, {len(data['hard_conflicts'])} пар конфликтов")
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    args = parser.parse_args()
    if args.runs < 1 or args.population < 30 or args.generations < 1:
        parser.error("runs >= 1, population >= 30 и generations >= 1 обязательны")
    run_experiments(
        runs=args.runs,
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
    )


if __name__ == "__main__":
    main()
