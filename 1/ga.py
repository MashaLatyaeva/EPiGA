"""A reproducible real-coded genetic algorithm for the Michalewicz function."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import random
import statistics
import time
from concurrent.futures import Executor, ProcessPoolExecutor
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
LOWER = 0.0
UPPER = math.pi
M = 10
DEFAULT_POPULATION = 100
DEFAULT_GENERATIONS = 250
DEFAULT_RUNS = 20


def michalewicz(point: Sequence[float], m: int = M) -> float:
    """Return the standard minimization-form Michalewicz objective."""
    return -sum(
        math.sin(value)
        * math.sin((index + 1) * value * value / math.pi) ** (2 * m)
        for index, value in enumerate(point)
    )


def _fitness(point: Sequence[float]) -> float:
    return michalewicz(point)


def evaluate_population(
    population: Sequence[Sequence[float]], executor: Executor | None
) -> list[float]:
    if executor is None:
        return [_fitness(point) for point in population]
    return list(executor.map(_fitness, population, chunksize=max(1, len(population) // 16)))


def _tournament(
    population: Sequence[Sequence[float]],
    scores: Sequence[float],
    rng: random.Random,
    size: int = 3,
) -> list[float]:
    contestants = rng.sample(range(len(population)), size)
    winner = min(contestants, key=scores.__getitem__)
    return list(population[winner])


def _blend(
    first: Sequence[float], second: Sequence[float], rng: random.Random, alpha: float = 0.5
) -> list[float]:
    child: list[float] = []
    for left, right in zip(first, second):
        low, high = sorted((left, right))
        spread = high - low
        child.append(rng.uniform(low - alpha * spread, high + alpha * spread))
    return child


def _mutate(
    point: list[float], rng: random.Random, probability: float, sigma: float
) -> None:
    for index, value in enumerate(point):
        if rng.random() < probability:
            point[index] = min(UPPER, max(LOWER, value + rng.gauss(0.0, sigma)))


def run_ga(
    *,
    seed: int,
    dimension: int,
    population_size: int,
    generations: int,
    mutation_probability: float,
    executor: Executor | None = None,
) -> tuple[list[float], list[dict[str, float]], list[list[float]], int]:
    rng = random.Random(seed)
    population = [
        [rng.uniform(LOWER, UPPER) for _ in range(dimension)]
        for _ in range(population_size)
    ]
    scores = evaluate_population(population, executor)
    evaluations = len(population)
    history: list[dict[str, float]] = []
    best_points: list[list[float]] = []
    base_sigma = 0.12 * (UPPER - LOWER)

    for generation in range(generations + 1):
        best_index = min(range(population_size), key=scores.__getitem__)
        history.append(
            {
                "generation": float(generation),
                "best": scores[best_index],
                "mean": statistics.fmean(scores),
                "worst": max(scores),
            }
        )
        best_points.append(list(population[best_index]))
        if generation == generations:
            break

        next_population = [list(population[best_index])]
        if population_size > 1:
            second_best = min(
                (index for index in range(population_size) if index != best_index),
                key=scores.__getitem__,
            )
            next_population.append(list(population[second_best]))

        while len(next_population) < population_size:
            parent_a = _tournament(population, scores, rng)
            parent_b = _tournament(population, scores, rng)
            child = _blend(parent_a, parent_b, rng)
            sigma = base_sigma * max(
                0.05, 1.0 - generation / max(generations, 1)
            )
            _mutate(child, rng, mutation_probability, sigma)
            next_population.append(child)

        population = next_population
        scores = evaluate_population(population, executor)
        evaluations += len(population)

    best_index = min(range(population_size), key=scores.__getitem__)
    return list(population[best_index]), history, best_points, evaluations


def run_random_search(
    *, seed: int, dimension: int, population_size: int, generations: int, executor: Executor | None
) -> tuple[list[float], list[dict[str, float]], int]:
    rng = random.Random(seed)
    history: list[dict[str, float]] = []
    best_point: list[float] | None = None
    best_score = math.inf
    evaluations = 0
    for generation in range(generations + 1):
        samples = [
            [rng.uniform(LOWER, UPPER) for _ in range(dimension)]
            for _ in range(population_size)
        ]
        scores = evaluate_population(samples, executor)
        evaluations += len(samples)
        local_best = min(range(len(samples)), key=scores.__getitem__)
        if scores[local_best] < best_score:
            best_score = scores[local_best]
            best_point = list(samples[local_best])
        history.append(
            {
                "generation": float(generation),
                "best": best_score,
                "mean": statistics.fmean(scores),
                "worst": max(scores),
            }
        )
    assert best_point is not None
    return best_point, history, evaluations


def _summary(values: Sequence[float]) -> dict[str, float]:
    return {
        "best": min(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "worst": max(values),
    }


def _write_convergence_svg(
    output: Path, series: dict[str, list[list[float]]], title: str
) -> None:
    width, height = 900, 520
    left, top, plot_width, plot_height = 78, 55, 775, 370
    all_values = [value for runs in series.values() for run in runs for value in run]
    low, high = min(all_values), max(all_values)
    padding = max((high - low) * 0.08, 0.05)
    low -= padding
    high += padding
    colors = ["#165DFF", "#F59E0B", "#4B5563"]

    def xy(index: int, value: float, length: int) -> tuple[float, float]:
        x = left + index / max(length - 1, 1) * plot_width
        y = top + (high - value) / (high - low) * plot_height
        return x, y

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-family="sans-serif" font-size="18">{html.escape(title)}</text>',
    ]
    for tick in range(6):
        value = low + (high - low) * tick / 5
        y = top + plot_height * tick / 5
        parts.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left+plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{left-10}" y="{y+4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11">{value:.3f}</text>')
    for key, runs in series.items():
        length = len(runs[0])
        lower = [min(run[index] for run in runs) for index in range(length)]
        upper = [max(run[index] for run in runs) for index in range(length)]
        mean = [statistics.fmean(run[index] for run in runs) for index in range(length)]
        color = colors.pop(0)
        polygon = [xy(i, lower[i], length) for i in range(length)]
        polygon += [xy(i, upper[i], length) for i in reversed(range(length))]
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in polygon)
        line = " ".join(
            f"{xy(i, value, length)[0]:.1f},{xy(i, value, length)[1]:.1f}"
            for i, value in enumerate(mean)
        )
        parts.append(f'<polygon points="{points}" fill="{color}" opacity="0.12"/>')
        parts.append(f'<polyline points="{line}" fill="none" stroke="{color}" stroke-width="2.5"/>')
    parts.extend(
        [
            f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="#111827"/>',
            f'<text x="{left+plot_width/2}" y="{height-18}" text-anchor="middle" font-family="sans-serif" font-size="13">Поколение</text>',
            f'<text x="20" y="{top+plot_height/2}" transform="rotate(-90 20 {top+plot_height/2})" text-anchor="middle" font-family="sans-serif" font-size="13">Лучшее значение f (меньше — лучше)</text>',
        ]
    )
    legend_x = left + 8
    for index, key in enumerate(series):
        color = ["#165DFF", "#F59E0B", "#4B5563"][index]
        parts.append(f'<line x1="{legend_x}" y1="42" x2="{legend_x+24}" y2="42" stroke="{color}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+30}" y="46" font-family="sans-serif" font-size="12">{html.escape(key)}</text>')
        legend_x += 190
    parts.append("</svg>")
    output.write_text("\n".join(parts), encoding="utf-8")


def _write_demo_html(output: Path, best_points: Sequence[Sequence[float]]) -> None:
    points = json.dumps(best_points)
    document = """<!doctype html>
<html lang="ru"><meta charset="utf-8"><title>Михалевич, d=2: сходимость ГА</title>
<style>body{font:16px system-ui;margin:24px;color:#172033}canvas{border:1px solid #9ca3af;max-width:100%}input{width:min(720px,90vw)}.row{margin:12px 0}</style>
<h1>Сходимость генетического алгоритма, d=2</h1>
<p>Тепловая карта функции Михалевича на [0, π]²; красная точка — лучшее решение данного поколения.</p>
<canvas id="plot" width="720" height="520"></canvas>
<div class="row"><input id="generation" type="range" min="0" max="0" value="0">
<span id="label"></span></div>
<script>
const points=__POINTS__, canvas=document.getElementById('plot'), ctx=canvas.getContext('2d');
const slider=document.getElementById('generation'), label=document.getElementById('label'), pi=Math.PI;
slider.max=points.length-1;
function f(x,y){return -(Math.sin(x)*Math.sin(x*x/pi)**20+Math.sin(y)*Math.sin(2*y*y/pi)**20);}
function draw(){
 const pad=42,w=canvas.width-2*pad,h=canvas.height-2*pad;
 let vals=[]; for(let j=0;j<90;j++)for(let i=0;i<120;i++)vals.push(f(pi*i/119,pi*j/89));
 const lo=Math.min(...vals),hi=Math.max(...vals); let k=0;
 for(let j=0;j<90;j++)for(let i=0;i<120;i++){
  const t=(vals[k++]-lo)/(hi-lo), hue=240-240*t;
  ctx.fillStyle=`hsl(${hue},75%,${92-42*t}%)`;
  ctx.fillRect(pad+i*w/120,pad+j*h/90,w/120+1,h/90+1);
 }
 ctx.strokeStyle='#111827';ctx.lineWidth=1;ctx.strokeRect(pad,pad,w,h);
 const p=points[Number(slider.value)],x=pad+p[0]/pi*w,y=pad+(1-p[1]/pi)*h;
 ctx.fillStyle='#dc2626';ctx.beginPath();ctx.arc(x,y,7,0,2*pi);ctx.fill();
 ctx.fillStyle='#111827';ctx.font='14px system-ui';ctx.fillText('x₁',pad+w/2,canvas.height-8);
 ctx.save();ctx.translate(14,pad+h/2);ctx.rotate(-Math.PI/2);ctx.fillText('x₂',0,0);ctx.restore();
 label.textContent=`Поколение ${slider.value} из ${points.length-1}; x=(${p[0].toFixed(4)}, ${p[1].toFixed(4)}), f=${f(...p).toFixed(8)}`;
}
slider.addEventListener('input',draw);draw();
</script></html>"""
    output.write_text(document.replace("__POINTS__", points), encoding="utf-8")


def run_experiments(
    *, runs: int, seed: int, population_size: int, generations: int, workers: int
) -> dict[str, dict[str, float]]:
    dimension = 5 + (10 % 6)
    settings = {
        "ГА, p_mut=0.10": 0.10,
        "ГА, p_mut=0.25": 0.25,
    }
    results: list[dict[str, object]] = []
    histories: dict[str, list[list[float]]] = {name: [] for name in settings}
    histories["Случайный поиск"] = []
    best_records: dict[str, tuple[float, list[float]]] = {}
    executor_context = ProcessPoolExecutor(max_workers=workers) if workers > 1 else None
    try:
        executor = executor_context
        for name, mutation_probability in settings.items():
            for run_index in range(runs):
                run_seed = seed + run_index
                started = time.perf_counter()
                point, history, _, evaluations = run_ga(
                    seed=run_seed,
                    dimension=dimension,
                    population_size=population_size,
                    generations=generations,
                    mutation_probability=mutation_probability,
                    executor=executor,
                )
                elapsed = time.perf_counter() - started
                value = michalewicz(point)
                histories[name].append([row["best"] for row in history])
                results.append(
                    {
                        "configuration": name,
                        "run": run_index + 1,
                        "seed": run_seed,
                        "best_f": value,
                        "evaluations": evaluations,
                        "seconds": elapsed,
                    }
                )
                if name not in best_records or value < best_records[name][0]:
                    best_records[name] = (value, point)

        for run_index in range(runs):
            run_seed = seed + run_index
            started = time.perf_counter()
            point, history, evaluations = run_random_search(
                seed=run_seed + 1_000_000,
                dimension=dimension,
                population_size=population_size,
                generations=generations,
                executor=executor,
            )
            elapsed = time.perf_counter() - started
            value = michalewicz(point)
            name = "Случайный поиск"
            histories[name].append([row["best"] for row in history])
            results.append(
                {
                    "configuration": name,
                    "run": run_index + 1,
                    "seed": run_seed + 1_000_000,
                    "best_f": value,
                    "evaluations": evaluations,
                    "seconds": elapsed,
                }
            )
            if name not in best_records or value < best_records[name][0]:
                best_records[name] = (value, point)
    finally:
        if executor_context is not None:
            executor_context.shutdown()

    summaries = {
        name: _summary(
            [float(row["best_f"]) for row in results if row["configuration"] == name]
        )
        for name in [*settings, "Случайный поиск"]
    }
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    (ROOT / "metrics.json").write_text(
        json.dumps(
            {
                "variant": 10,
                "dimension": dimension,
                "seed": seed,
                "runs": runs,
                "population_size": population_size,
                "generations": generations,
                "evaluations_per_run": population_size * (generations + 1),
                "summaries": summaries,
                "best_solutions": {
                    name: {"f": value, "x": point}
                    for name, (value, point) in best_records.items()
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    generation_count = generations + 1
    with (ROOT / "convergence.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.writer(file)
        writer.writerow(["generation", *[f"{name}_{stat}" for name in histories for stat in ("min", "mean", "max")]])
        for index in range(generation_count):
            row: list[float | int] = [index]
            for runs_history in histories.values():
                values = [run[index] for run in runs_history]
                row.extend([min(values), statistics.fmean(values), max(values)])
            writer.writerow(row)
    _write_convergence_svg(ROOT / "convergence.svg", histories, "Функция Михалевича, вариант 10")

    demo_point, _, demo_points, _ = run_ga(
        seed=seed,
        dimension=2,
        population_size=40,
        generations=80,
        mutation_probability=0.15,
    )
    _write_demo_html(ROOT / "demo_2d.html", demo_points)
    print(json.dumps(summaries, ensure_ascii=False, indent=2))
    print("Лучшее решение для d=2:", demo_point, "f=", michalewicz(demo_point))
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--population", type=int, default=DEFAULT_POPULATION)
    parser.add_argument("--generations", type=int, default=DEFAULT_GENERATIONS)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.runs < 1 or args.population < 30 or args.generations < 1 or args.workers < 1:
        parser.error("runs >= 1, population >= 30, generations >= 1 и workers >= 1 обязательны")
    run_experiments(
        runs=args.runs,
        seed=args.seed,
        population_size=args.population,
        generations=args.generations,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
