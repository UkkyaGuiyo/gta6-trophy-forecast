"""Independent synthetic targets, reference baselines, and executable checks."""

import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("entry", ROOT / "main.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
SCORES = (-1.645, -0.674, 0, 0.674, 1.645)
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
REGIMES = ("fast", "slow", "high_noise", "low_noise", "tracking_surge", "recount")


def timestamp(hour):
    return (EPOCH + timedelta(hours=hour)).isoformat()


def percent(value):
    return 100 / (1 + math.exp(-value))


def scoring_logit(rate):
    p = max(0.01, min(99.99, rate)) / 100
    return math.log(p / (1 - p))


def reference_quantiles(center, sigma):
    return [math.floor(percent(center + z * sigma) * 10000 + 0.5) / 10000
            for z in SCORES]


def reference_baselines(case):
    """Implement the prose rules, without importing any kit implementation."""
    target = (datetime.fromisoformat(case["target"]) - EPOCH).total_seconds() / 3600
    checkpoint = (datetime.fromisoformat(case["at"]) - EPOCH).total_seconds() / 3600
    drift = 0.5 * math.sqrt((target - checkpoint) / 24)
    persistence, curve = [], []
    for index in range(len(case["trophies"])):
        observations = [((datetime.fromisoformat(r["at"]) - EPOCH).total_seconds() / 3600,
                         scoring_logit(r["rates"][index])) for r in case["history"]
                        if r["rates"][index] is not None]
        last = observations[-1][1]
        persistence.append(reference_quantiles(last, drift))
        recent = [(math.log1p(h), y) for h, y in observations if h >= checkpoint - 24]
        if len(recent) < 3:
            curve.append(persistence[-1])
            continue
        mean_x = statistics.mean(x for x, _ in recent)
        mean_y = statistics.mean(y for _, y in recent)
        denominator = sum((x - mean_x) ** 2 for x, _ in recent)
        slope = sum((x - mean_x) * (y - mean_y) for x, y in recent) / denominator
        errors = [y - mean_y - slope * (x - mean_x) for x, y in recent]
        residual = math.sqrt(statistics.mean(e * e for e in errors))
        center = mean_y + slope * (math.log1p(target) - mean_x)
        curve.append(reference_quantiles(center, math.hypot(residual, drift)))
    return persistence, curve


def pinball(quantiles, truth):
    total = 0.0
    for qs, actual in zip(quantiles, truth):
        for tau, q in zip(LEVELS, qs):
            error = scoring_logit(actual) - scoring_logit(q)
            total += 2 * max(tau * error, (tau - 1) * error)
    return total / (len(truth) * len(LEVELS))


def synthetic_case(regime, seed):
    """A curved latent signal, Brownian drift, page noise, and optional events."""
    rng = random.Random(seed)
    checkpoint = rng.choice((36, 48, 60))
    target = checkpoint + rng.choice((24, 72, 108.5, 144))
    noise = 0.45 if regime == "high_noise" else 0.04 if regime == "low_noise" else 0.12
    speed = 1.6 if regime == "fast" else 0.35 if regime == "slow" else 1.0
    drift_scale = rng.uniform(0.25, 0.55)
    event_hour = checkpoint * rng.uniform(0.4, 0.7)
    recount = rng.choice((-1, 1)) * rng.uniform(0.4, 0.9)
    surge = rng.uniform(2, 4)
    trophies = []
    parameters = []
    tiers = ("platinum", "gold", "silver", "bronze")
    for index in range(24):
        group = index % 4
        trophies.append({"id": index + 1, "tier": tiers[group]})
        parameters.append(((-8.5, -6.2, -4.8, -3.2)[group] + rng.uniform(-1, 1),
                           (0.85, 1.1, 1.35, 1.5)[group] * speed * rng.uniform(0.7, 1.3),
                           rng.uniform(0.12, 0.45)))

    def observed_rate(hour, index, drift, read_noise):
        intercept, slope, curvature = parameters[index]
        x = math.log1p(hour)
        latent = intercept + slope * (-math.expm1(-curvature * x)) / curvature + drift
        if regime == "recount" and hour >= event_hour:
            latent += recount
        rate = percent(latent + read_noise)
        if regime == "tracking_surge" and hour >= event_hour:
            # New players catch up gradually after a denominator jump.
            rate /= 1 + (surge - 1) * math.exp(-(hour - event_hour) / 72)
        return math.floor(rate * 100 + 0.5) / 100

    history = []
    walks = [0.0] * len(trophies)
    for step in range(checkpoint * 2):
        hour = step / 2
        common_noise = rng.gauss(0, noise / 2)
        rates = []
        for index in range(len(trophies)):
            if step:
                walks[index] += rng.gauss(0, drift_scale * math.sqrt(0.5 / 24))
            rate = observed_rate(hour, index, walks[index], common_noise + rng.gauss(0, noise))
            rates.append(None if index % 7 == 0 and hour < 6 else rate)
        history.append({"at": timestamp(hour), "rates": rates,
                        "players_tracked": round(1000 * (surge if regime == "tracking_surge"
                                                        and hour >= event_hour else 1))})
    # Independent future Brownian increments and a new observation; never reuse history noise.
    common_noise = rng.gauss(0, noise / 2)
    truth = [observed_rate(target, index,
                           walk + rng.gauss(0, drift_scale * math.sqrt((target - hour) / 24)),
                           common_noise + rng.gauss(0, noise))
             for index, walk in enumerate(walks)]
    case = {"id": "synthetic", "at": timestamp(checkpoint), "target": timestamp(target),
            "horizon_minutes": (target - checkpoint) * 60, "levels": list(LEVELS),
            "trophies": trophies, "history": history}
    return case, truth


def evaluate(cases):
    totals = {regime: [[], [], []] for regime in REGIMES}
    for regime, case, truth, baselines in cases:
        predictions = [row["q"] for row in entry.forecast_case(case)["trophies"]]
        for bucket, forecast in zip(totals[regime], (predictions, *baselines)):
            bucket.append(pinball(forecast, truth))
    results = {regime: [statistics.mean(values) for values in buckets]
               for regime, buckets in totals.items()}
    results["aggregate"] = [statistics.mean(row[i] for row in results.values()) for i in range(3)]
    return results


def make_cases(start_seed, count):
    result = []
    for regime in REGIMES:
        for seed in range(start_seed, start_seed + count):
            case, truth = synthetic_case(regime, seed)
            result.append((regime, case, truth, reference_baselines(case)))
    return result


def assert_valid(case, output):
    assert set(output) == {"trophies"}
    rows = output["trophies"]
    assert len(rows) == len(case["trophies"])
    assert sorted(row["id"] for row in rows) == sorted(t["id"] for t in case["trophies"])
    for row in rows:
        assert set(row) == {"id", "q"}
        assert len(row["q"]) == 5
        assert all(isinstance(q, (int, float)) and not isinstance(q, bool)
                   and math.isfinite(q) and 0 <= q <= 100 for q in row["q"])
        assert row["q"] == sorted(row["q"])


def check_edge_cases():
    case = {"at": timestamp(12), "target": timestamp(13.5),
            "trophies": [{"id": i, "tier": "bronze"} for i in range(6)],
            "history": [{"at": timestamp(h), "rates": rates} for h, rates in
                        [(-0.15, [None, 0, 100, None, 1, 0.01]),
                         (1, [None, 0, 100, None, 1, 0.02]),
                         (6, [None, 0, 100, 0.5, 1, 0.03]),
                         (11, [None, 0, 100, 0.6, 2, 0.04])]]}
    output = entry.forecast_case(case)
    assert_valid(case, output)
    assert [row["q"][2] for row in output["trophies"][:4]] == [0.05, 0.01, 99.99, 0.6]
    flat_q = output["trophies"][0]["q"]
    assert scoring_logit(flat_q[-1]) - scoring_logit(flat_q[2]) > 1.2
    assert entry.rounded_quantiles(entry.rate_logit(1.23456), 0)[2] == 1.2346
    assert entry.rounded_quantiles(entry.rate_logit(1.23454), 0)[2] == 1.2345
    assert entry.rounded_quantiles(-1000, 1) == [0.01] * 5
    assert entry.rounded_quantiles(1000, 1) == [99.99] * 5
    case["history"] = []
    assert_valid(case, entry.forecast_case(case))
    assert all(row["q"][2] == 0.05 for row in entry.forecast_case(case)["trophies"])
    print("Edge checks: missing/zero/100/sparse/flat/rounding/extreme/empty passed")


def check_practice(kit):
    inputs = sorted((kit / "input").glob("*.in"))
    assert inputs, "No practice inputs"
    base = ROOT / ".checks" / f"practice-{time.time_ns()}"
    base.mkdir()
    for source in inputs:
        single_input = base / source.stem
        single_input.mkdir()
        (single_input / source.name).write_bytes(source.read_bytes())
        durations = []
        contents = []
        for repeat in range(2):
            destination = base / (source.stem + "-" + str(repeat))
            env = dict(os.environ, MAKEFUN_INPUT_DIR=str(single_input),
                       MAKEFUN_OUTPUT_DIR=str(destination), PYTHONHASHSEED="0")
            start = time.perf_counter()
            subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")],
                           env=env, cwd=base, check=True, timeout=8, capture_output=True)
            durations.append((time.perf_counter() - start) * 1000)
            raw = (destination / (source.stem + ".out")).read_bytes()
            assert len(raw) <= 65536
            assert_valid(json.loads(source.read_text(encoding="utf-8")), json.loads(raw))
            contents.append(raw)
        assert contents[0] == contents[1]
        print(f"{source.stem}: valid, deterministic, {len(contents[0])} bytes, "
              f"wall_ms={[round(ms, 3) for ms in durations]}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--tune", action="store_true")
    arguments = parser.parse_args()
    (ROOT / ".checks").mkdir(exist_ok=True)
    print("Python", sys.version.split()[0])
    check_edge_cases()
    if arguments.kit:
        check_practice(arguments.kit)
    if arguments.tune:
        for name, value in (("RECENCY_HOURS", 24.0), ("POOL_READS", 8.0),
                            ("DAMPING_SCALE", 2.0), ("DRIFT_PER_DAY", 0.45)):
            setattr(entry, name, value)
        cases = make_cases(100, 8)
        print("Initial selection", json.dumps(evaluate(cases)), flush=True)
        # Coordinate sweep on selection seeds only; held-out seeds are untouched.
        for name, options in (("RECENCY_HOURS", (6.0, 12.0, 24.0)),
                              ("POOL_READS", (4.0, 8.0, 16.0)),
                              ("DAMPING_SCALE", (2.0, 4.0, 8.0)),
                              ("DRIFT_PER_DAY", (0.35, 0.4, 0.45, 0.5))):
            scores = []
            for value in options:
                setattr(entry, name, value)
                loss = evaluate(cases)["aggregate"][0]
                scores.append((loss, value))
                print(f"selection {name}={value}: {loss:.6f}", flush=True)
            best = min(scores)[1]
            setattr(entry, name, best)
        print("Selected", {key: getattr(entry, key) for key in
                           ("RECENCY_HOURS", "POOL_READS", "DAMPING_SCALE", "DRIFT_PER_DAY")})
        print(json.dumps(evaluate(cases), indent=2))
    else:
        results = evaluate(make_cases(1000, 32))
        print(json.dumps(results, indent=2))
        assert results["aggregate"][0] < min(results["aggregate"][1:]), results


if __name__ == "__main__":
    main()
