"""Rolling-origin calibration, independent simulations, and executable checks."""

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
NORMAL_SCORES = (-1.645, -0.674, 0, 0.674, 1.645)
EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc)
REGIMES = ("normal_growth", "saturation", "dilution", "common_recount",
           "individual_recount", "stall_then_surge", "rate_decline", "duplicate_reads")
SEED_SETS = {"DEV": (21000, 12), "TUNING": (31000, 12), "HOLDOUT": (51000, 40)}
CALIBRATION_DEFAULTS = {"SATURATION_TAU": 1.0, "PROCESS_VARIANCE": 0.16,
                        "MODEL_SIGMA": 0.25, "LOWER_25": 0.7, "LOWER_05": 1.7,
                        "UPPER_25": 0.7, "UPPER_05": 1.7, "DILUTION_LOWER": 0.5}
DESIGN_DEFAULTS = {"RECENCY_DAYS": 1.0, "COHORT_RATIO_CAP": 4.0,
                   "POPULATION_DECAY": 1.0, "POPULATION_CAP": 3.0,
                   "USE_COHORT_MIXTURE": True}


def timestamp(day):
    return (EPOCH + timedelta(days=day)).isoformat()


def scoring_logit(rate):
    p = max(0.01, min(99.99, rate)) / 100
    return math.log(p / (1 - p))


def percent(value):
    return 100 / (1 + math.exp(-value)) if value >= 0 else 100 * math.exp(value) / (1 + math.exp(value))


def quantile_loss(qs, truth):
    total = 0.0
    for row, actual in zip(qs, truth):
        for tau, q in zip(LEVELS, row):
            error = scoring_logit(actual) - scoring_logit(q)
            total += 2 * max(tau * error, (tau - 1) * error)
    return total / (len(truth) * 5)


def normal_quantiles(center, sigma):
    return [math.floor(percent(center + z * sigma) * 10000 + 0.5) / 10000 for z in NORMAL_SCORES]


def reference_baselines(case):
    """House prose curve remains on log HOURS; the revised entry uses DAYS."""
    target = (datetime.fromisoformat(case["target"]) - EPOCH).total_seconds() / 3600
    checkpoint = (datetime.fromisoformat(case["at"]) - EPOCH).total_seconds() / 3600
    drift = 0.5 * math.sqrt((target - checkpoint) / 24)
    persistence, curve = [], []
    for i in range(len(case["trophies"])):
        observations = [((datetime.fromisoformat(r["at"]) - EPOCH).total_seconds() / 3600,
                         scoring_logit(r["rates"][i])) for r in case["history"] if r["rates"][i] is not None]
        persistence.append(normal_quantiles(observations[-1][1], drift))
        recent = [(math.log1p(h), y) for h, y in observations if h >= checkpoint - 24]
        if len(recent) < 3:
            curve.append(persistence[-1])
            continue
        mx, my = statistics.mean(x for x, _ in recent), statistics.mean(y for _, y in recent)
        denominator = sum((x - mx) ** 2 for x, _ in recent)
        slope = sum((x - mx) * (y - my) for x, y in recent) / denominator if denominator else 0.0
        residual = math.sqrt(statistics.mean((y - my - slope * (x - mx)) ** 2 for x, y in recent))
        curve.append(normal_quantiles(my + slope * (math.log1p(target) - mx), math.hypot(residual, drift)))
    return persistence, curve


def synthetic_scenario(regime, seed):
    """One population path, three rolling origins, independent target read noise."""
    rng = random.Random(seed)
    checkpoint = rng.choice((1.5, 2.0, 2.5))
    origins = [checkpoint * fraction for fraction in (0.65, 0.85, 1.0)]
    horizon = rng.choice((1.0, 3.0, 4.520833333333333, 6.0))
    targets = [origin + horizon for origin in origins]
    event = checkpoint * rng.uniform(0.5, 0.95)
    surge_factor, pop_speed = rng.uniform(1.8, 3.5), rng.uniform(0.01, 0.05)
    post_surge_speed, catchup = rng.uniform(0.12, 0.35), rng.uniform(2.0, 5.0)
    recount = rng.choice((-1, 1)) * rng.uniform(0.5, 1.2)
    diffusion, noise = rng.uniform(0.12, 0.30), rng.uniform(0.025, 0.14)
    trophies, parameters = [], []
    for i in range(24):
        group = i % 4
        trophies.append({"id": i + 1, "tier": ("platinum", "gold", "silver", "bronze")[group]})
        curvature = rng.uniform(0.7, 1.6) if regime == "saturation" else rng.uniform(0.05, 0.2)
        slope = (1.0, 1.3, 1.6, 1.8)[group] * rng.uniform(0.8, 1.2)
        if regime == "rate_decline":
            slope = -rng.uniform(0.4, 1.0)
        parameters.append(((-7.2, -5.5, -4.0, -2.5)[group] + rng.uniform(-0.6, 0.6), slope,
                           curvature, rng.uniform(0.1, 0.6), rng.choice((-1, 1)) * rng.uniform(0.5, 1.5)))
    times = sorted(set([step / 24 for step in range(math.ceil(max(targets) * 24) + 1)] + targets + origins))
    snapshots, cohorts = {}, []
    old_population, walks, common_walk, previous_day = 1_000_000, [0.0] * 24, 0.0, 0.0
    last_refresh_bucket, page = None, None
    for day in times:
        elapsed = day - previous_day
        common_walk += rng.gauss(0, 0.08 * math.sqrt(elapsed))
        for i in range(24):
            walks[i] += rng.gauss(0, diffusion * math.sqrt(elapsed))
        multiplier = math.exp(pop_speed * day)
        if regime == "dilution":
            ramp = 1 / (1 + math.exp(-(day - event) / 0.08))
            multiplier *= (1 + (surge_factor - 1) * ramp) * math.exp(post_surge_speed * max(0, day - event))
        population = max(1_000_000, 10000 * round(100 * multiplier))
        if population > old_population:
            cohorts.append((day, population - old_population))
        old_population = population
        bucket = math.floor(day * 4 + 1e-8) if regime == "duplicate_reads" else day
        if bucket != last_refresh_bucket:
            page_noise = rng.gauss(0, noise / 2)
            rates = []
            for i, (intercept, slope, curvature, cohort_start, individual_jump) in enumerate(parameters):
                x = math.log1p(day)
                if regime == "stall_then_surge":
                    x, slope = max(0.0, x - math.log1p(event)), slope * 3
                signal = intercept + slope * (-math.expm1(-curvature * x)) / curvature + walks[i] + common_walk
                if day >= event:
                    if regime == "common_recount":
                        signal += recount
                    elif regime == "individual_recount" and i % 4 == 0:
                        signal += individual_jump
                incumbent = percent(signal + page_noise + rng.gauss(0, noise))
                cohort_mass = sum(size * (1 - (1 - cohort_start) * math.exp(-(day - joined) / catchup))
                                  for joined, size in cohorts)
                printed = math.floor(incumbent * (1_000_000 + cohort_mass) / population * 100 + 0.5) / 100
                rates.append(None if i % 7 == 0 and day < 0.25 else printed)
            page = {"players_tracked": population, "rates": rates}
            last_refresh_bucket = bucket
        snapshots[day] = {**page, "at": timestamp(day)}
        previous_day = day
    cases = []
    for origin, target in zip(origins, targets):
        history = [snapshots[day] for day in times if day < origin and abs(day * 24 - round(day * 24)) < 1e-8]
        case = {"id": "synthetic", "at": timestamp(origin), "target": timestamp(target),
                "horizon_minutes": horizon * 1440, "levels": list(LEVELS), "trophies": trophies, "history": history}
        cases.append({"regime": regime, "case": case, "truth": snapshots[target]["rates"],
                      "baselines": reference_baselines(case)})
    return cases


def make_cases(name):
    start, count = SEED_SETS[name]
    return [case for regime in REGIMES for seed in range(start, start + count)
            for case in synthetic_scenario(regime, seed)]


def prepare_records(cases):
    return [{**record, "components": entry.forecast_components(record["case"])} for record in cases]


def entry_loss(records):
    return statistics.mean(quantile_loss([entry.component_quantiles(c) for c in record["components"]], record["truth"])
                           for record in records)


def evaluate(records):
    losses = {regime: [[], [], [], []] for regime in REGIMES}
    coverage, counts = {regime: [0] * 5 for regime in REGIMES}, {regime: 0 for regime in REGIMES}
    for record in records:
        regime = record["regime"]
        predictions = [entry.component_quantiles(c) for c in record["components"]]
        symmetric = []
        for c in record["components"]:
            sigma = math.sqrt(c["level_variance"] + c["dx"] ** 2 * c["slope_variance"] +
                              entry.PROCESS_VARIANCE * c["horizon"] + entry.MODEL_SIGMA ** 2)
            symmetric.append(normal_quantiles(c["center"], max(entry.FLAT_SIGMA, sigma) if c["flat"] else sigma))
        for bucket, qs in zip(losses[regime], (predictions, *record["baselines"], symmetric)):
            bucket.append(quantile_loss(qs, record["truth"]))
        for qs, truth in zip(predictions, record["truth"]):
            counts[regime] += 1
            for i, q in enumerate(qs):
                coverage[regime][i] += truth <= q
    result = {r: [statistics.mean(v) for v in buckets] for r, buckets in losses.items() if buckets[0]}
    result["aggregate"] = [statistics.mean(row[i] for row in result.values()) for i in range(4)]
    covered = {r: [n / counts[r] for n in values] for r, values in coverage.items() if counts[r]}
    covered["aggregate"] = [sum(v[i] for v in coverage.values()) / sum(counts.values()) for i in range(5)]
    return {"loss": result, "coverage": covered, "cases": len(records), "outcomes": sum(counts.values())}


def choose_widths(records):
    rows = []
    for record in records:
        for c, y in zip(record["components"], record["truth"]):
            sigma = math.sqrt(c["level_variance"] + c["dx"] ** 2 * c["slope_variance"] +
                              entry.PROCESS_VARIANCE * c["horizon"] + entry.MODEL_SIGMA ** 2)
            rows.append((c, scoring_logit(y), max(entry.FLAT_SIGMA, sigma) if c["flat"] else sigma))
    best = None
    lower_bound, upper_bound = scoring_logit(0.01), scoring_logit(99.99)
    for risk in (0.25, 0.5, 1.0, 2.0):
        chosen, total = {}, 0.0
        for name, tau, sign, grid in (("LOWER_05", 0.05, -1, (0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3)),
                                      ("LOWER_25", 0.25, -1, (0.25, 0.5, 0.75, 1, 1.25)),
                                      ("UPPER_25", 0.75, 1, (0.25, 0.5, 0.75, 1, 1.25)),
                                      ("UPPER_05", 0.95, 1, (0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3))):
            options = []
            for width in grid:
                if name == "LOWER_25" and width > chosen["LOWER_05"]:
                    continue
                if name == "UPPER_05" and width < chosen["UPPER_25"]:
                    continue
                value = 0.0
                for c, actual, sigma in rows:
                    factor = 1 + risk * min(1.0, c["growth"]) if sign < 0 else 1.0
                    predicted = min(upper_bound, max(lower_bound, c["center"] + sign * width * sigma * factor))
                    error = actual - predicted
                    value += 2 * max(tau * error, (tau - 1) * error)
                options.append((value, width))
            value, chosen[name] = min(options)
            total += value
        if best is None or (total, risk) < best[:2]:
            best = (total, risk, chosen)
    entry.DILUTION_LOWER = best[1]
    for name, value in best[2].items():
        setattr(entry, name, value)


def calibrate_dev(cases):
    tau_scores = []
    for tau in (0.3, 0.6, 1.2, 2.4):
        entry.SATURATION_TAU = tau
        records = prepare_records(cases)
        loss = statistics.mean(abs(scoring_logit(y) - min(scoring_logit(99.99), max(scoring_logit(0.01), c["center"])))
                               for r in records for c, y in zip(r["components"], r["truth"]))
        tau_scores.append((loss, tau))
    entry.SATURATION_TAU = min(tau_scores)[1]
    records = prepare_records(cases)
    for _ in range(2):
        scores = []
        for q in (0.0, 0.04, 0.09, 0.16, 0.25):
            for sigma in (0.12, 0.25, 0.4, 0.6):
                entry.PROCESS_VARIANCE, entry.MODEL_SIGMA = q, sigma
                scores.append((entry_loss(records), q, sigma))
        _, entry.PROCESS_VARIANCE, entry.MODEL_SIGMA = min(scores)
        choose_widths(records)
    print("DEV tau median-loss grid", tau_scores, flush=True)
    return evaluate(records)


def tune_design(cases):
    for name, values in (("RECENCY_DAYS", (0.5, 1.0, 2.0)), ("COHORT_RATIO_CAP", (1.0, 2.0, 4.0)),
                          ("POPULATION_DECAY", (0.5, 1.0, 2.0)), ("POPULATION_CAP", (1.5, 2.0, 3.0))):
        scores = []
        for value in values:
            setattr(entry, name, value)
            scores.append((entry_loss(prepare_records(cases)), value))
        setattr(entry, name, min(scores)[1])
        print("TUNING", name, scores, flush=True)
    return evaluate(prepare_records(cases))


def assert_valid(case, output):
    assert set(output) == {"trophies"}
    rows = output["trophies"]
    assert len(rows) == len(case["trophies"])
    assert sorted(row["id"] for row in rows) == sorted(t["id"] for t in case["trophies"])
    for row in rows:
        assert set(row) == {"id", "q"} and len(row["q"]) == 5
        assert all(isinstance(q, (int, float)) and not isinstance(q, bool)
                   and math.isfinite(q) and 0 <= q <= 100 for q in row["q"])
        assert row["q"] == sorted(row["q"])


def check_edge_cases():
    case = {"at": timestamp(0.5), "target": timestamp(0.5625),
            "trophies": [{"id": i, "tier": "bronze"} for i in range(6)],
            "history": [{"at": timestamp(h / 24), "players_tracked": 10000, "rates": rates}
                        for h, rates in [(-0.15, [None, 0, 100, None, 1, 0.01]),
                                         (1, [None, 0, 100, None, 1, 0.02]),
                                         (6, [None, 0, 100, 0.5, 1, 0.03]),
                                         (11, [None, 0, 100, 0.6, 2, 0.04])]]}
    assert entry.days_since_launch(timestamp(1)) == 1.0
    output = entry.forecast_case(case)
    assert_valid(case, output)
    assert [row["q"][2] for row in output["trophies"][:4]] == [0.05, 0.01, 99.99, 0.6]
    assert entry.rounded_quantiles(entry.rate_logit(1.23456), 0)[2] == 1.2346
    assert entry.rounded_quantiles(entry.rate_logit(1.23454), 0)[2] == 1.2345
    assert entry.rounded_quantiles(-1000, 1) == [0.01] * 5
    assert entry.rounded_quantiles(1000, 1) == [99.99] * 5
    frozen = [{"at": timestamp(h / 24), "players_tracked": 10000, "rates": [1.0]} for h in range(5)]
    states = entry.deduplicate_history(frozen)
    assert len(states) == 1 and states[0]["at"] == frozen[-1]["at"]
    flat = {**case, "trophies": [{"id": 1, "tier": "gold"}], "history": frozen}
    assert entry.forecast_components(flat)[0]["flat"]
    assert entry.forecast_case(flat) == entry.forecast_case({**flat, "history": [frozen[-1]]})
    assert scoring_logit(entry.forecast_case(flat)["trophies"][0]["q"][4]) - scoring_logit(1) >= 0.5
    fit = {"slope": 1.0, "slope_variance": 0.01}
    slopes = entry.hierarchical_slopes([fit, fit, {"slope": 5, "slope_variance": 100}], [{"tier": "bronze"}] * 3)
    assert abs(slopes[2] - 1) < 1
    dilution_states = [{"at": timestamp(day), "players_tracked": n, "rates": [rate]}
                       for day, n, rate in ((0, 10000, 10), (0.5, 20000, 5), (1, 40000, 2.5))]
    trophy = [{"id": 1, "tier": "gold"}]
    counts = [[state["rates"][0] * state["players_tracked"] / 100] for state in dilution_states]
    assert entry.estimate_cohort_ratios(dilution_states, trophy, counts) == [0.0]
    dilution_case = {"at": timestamp(1.01), "target": timestamp(2.5),
                     "trophies": trophy, "history": dilution_states}
    saved_mixture = entry.USE_COHORT_MIXTURE
    try:
        entry.USE_COHORT_MIXTURE = False
        incumbent = entry.forecast_case(dilution_case)["trophies"][0]["q"][2]
        entry.USE_COHORT_MIXTURE = True
        mixture = entry.forecast_case(dilution_case)["trophies"][0]["q"][2]
        assert abs(incumbent - 2.5) < 0.001 and mixture < incumbent
    finally:
        entry.USE_COHORT_MIXTURE = saved_mixture
    shock_states = [{"at": timestamp(day), "players_tracked": 10000,
                     "rates": [percent(-4 - i / 5 + 0.5 * math.log1p(day) + (1 if day >= 1 else 0))
                               for i in range(8)]} for day in (0, 0.5, 1, 1.5)]
    clean = entry.clean_observations(shock_states, [{"tier": "bronze"}] * 8, [1.0] * 8)
    assert all(abs(entry.fit_trophy_series(series)["slope"] - 0.5) < 1e-8 for series in clean)
    decline = synthetic_scenario("rate_decline", 991)[-1]
    medians = [c["center"] for c in entry.forecast_components(decline["case"])]
    latest = [scoring_logit(r) for r in decline["case"]["history"][-1]["rates"]]
    assert statistics.mean(medians) < statistics.mean(latest)
    case["history"] = []
    assert_valid(case, entry.forecast_case(case))
    assert all(row["q"][2] == 0.05 for row in entry.forecast_case(case)["trophies"])
    print("Edge checks: days, missing/zero/100/sparse/empty, dedup/flat, rounding, precision, counts/mixture, shock separation, decline passed", flush=True)


def check_practice(kit):
    base = ROOT / ".checks" / f"practice-{time.time_ns()}"
    base.mkdir()
    results = {}
    for source in sorted((kit / "input").glob("*.in")):
        single = base / source.stem
        single.mkdir()
        (single / source.name).write_bytes(source.read_bytes())
        case = json.loads(source.read_text(encoding="utf-8"))
        assert len(entry.deduplicate_history(case["history"])) == 1
        durations, contents = [], []
        for repeat in range(2):
            destination = base / (source.stem + "-" + str(repeat))
            env = dict(os.environ, MAKEFUN_INPUT_DIR=str(single), MAKEFUN_OUTPUT_DIR=str(destination), PYTHONHASHSEED="0")
            start = time.perf_counter()
            subprocess.run([sys.executable, "-P", "-s", str(ROOT / "main.py")], env=env,
                           cwd=base, check=True, timeout=8, capture_output=True)
            durations.append((time.perf_counter() - start) * 1000)
            raw = (destination / (source.stem + ".out")).read_bytes()
            assert len(raw) <= 65536
            assert_valid(case, json.loads(raw))
            contents.append(raw)
        assert contents[0] == contents[1]
        results[source.stem] = {"bytes": len(contents[0]), "wall_ms": durations, "valid": True, "deterministic": True}
    assert results
    print("Practice", json.dumps(results), flush=True)
    return results


def run_mixture_gate(cases):
    scores = {}
    dilution = [r for r in cases if r["regime"] == "dilution"]
    for enabled in (False, True):
        entry.USE_COHORT_MIXTURE = enabled
        scores[str(enabled)] = entry_loss(prepare_records(dilution))
    # The requested HOLDOUT gate is the sole reported selection exception.
    entry.USE_COHORT_MIXTURE = scores["True"] < scores["False"]
    return {"dilution_loss": scores, "ship_mixture": entry.USE_COHORT_MIXTURE}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kit", type=Path)
    parser.add_argument("--tune", action="store_true")
    args = parser.parse_args()
    (ROOT / ".checks").mkdir(exist_ok=True)
    print("Python", sys.version.split()[0], flush=True)
    check_edge_cases()
    results = {}
    dev, tuning = make_cases("DEV"), make_cases("TUNING")
    if args.tune:
        for name, value in {**CALIBRATION_DEFAULTS, **DESIGN_DEFAULTS}.items():
            setattr(entry, name, value)
        results["initial_dev"] = calibrate_dev(dev)
        results["design_selection"] = tune_design(tuning)
        results["DEV"] = calibrate_dev(dev)
    else:
        results["DEV"] = evaluate(prepare_records(dev))
    results["TUNING"] = evaluate(prepare_records(tuning))
    holdout = make_cases("HOLDOUT")
    results["mixture_gate"] = run_mixture_gate(holdout)
    results["HOLDOUT"] = evaluate(prepare_records(holdout))
    results["constants"] = {name: getattr(entry, name) for name in {**CALIBRATION_DEFAULTS, **DESIGN_DEFAULTS}}
    if args.kit:
        results["practice"] = check_practice(args.kit)
    (ROOT / ".checks" / "fix-round-results.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2), flush=True)
    loss = results["HOLDOUT"]["loss"]["dilution"]
    print("Dilution vs persistence:", "PASS" if loss[0] <= loss[1] else "SHORTFALL", flush=True)


if __name__ == "__main__":
    main()
