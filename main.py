"""Forecast trophy earn-rate quantiles using pooled, damped logit trends."""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import median


LAUNCH = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
NORMAL_SCORES = (-1.645, -0.674, 0.0, 0.674, 1.645)
RECENCY_HOURS = 24.0
POOL_READS = 16.0
DAMPING_SCALE = 2.0
DRIFT_PER_DAY = 0.50
RESIDUAL_FLOOR = 0.12
FLAT_SPREAD = 0.75


def hours_since_launch(timestamp):
    return (datetime.fromisoformat(timestamp) - LAUNCH).total_seconds() / 3600


def rate_logit(rate):
    bounded = min(99.99, max(0.01, rate)) / 100
    return math.log(bounded / (1 - bounded))


def logit_rate(value):
    if value >= 0:
        return 100 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return 100 * exponential / (1 + exponential)


def rounded_quantiles(center, spread):
    result = []
    for score in NORMAL_SCORES:
        rate = logit_rate(center + score * spread)
        rounded = math.floor(rate * 10000 + 0.5) / 10000
        result.append(max(result[-1] if result else 0.01,
                          min(99.99, max(0.01, rounded))))
    return result


def fit_trophy_series(series):
    """Return weighted slope, residual, effective count and recent anchor."""
    last_hour = series[-1][0]
    weights = [math.exp((hour - last_hour) / RECENCY_HOURS)
               for hour, _ in series]
    xs = [math.log1p(hour) for hour, _ in series]
    ys = [rate_logit(rate) for _, rate in series]
    total = sum(weights)
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    variance = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs))
    slope = (sum(w * (x - mean_x) * (y - mean_y)
                 for w, x, y in zip(weights, xs, ys)) / variance
             if variance > 0 else 0.0)
    residual = math.sqrt(sum(w * (y - mean_y - slope * (x - mean_x)) ** 2
                             for w, x, y in zip(weights, xs, ys)) / total)
    # shortcut: repeated refreshes count as reads; model refresh blocks if live calibration warrants it.
    effective_count = total * total / sum(w * w for w in weights)
    recent_weights = weights[-3:]
    anchor = sum(w * y for w, y in zip(recent_weights, ys[-3:])) / sum(recent_weights)
    return slope, residual, effective_count, anchor


def forecast_case(case):
    observations = [(hours_since_launch(read["at"]), read["rates"])
                    for read in case["history"]]
    series_list = [[(hour, rates[index]) for hour, rates in observations
                    if rates[index] is not None]
                   for index in range(len(case["trophies"]))]
    fits = [fit_trophy_series(series) if series else None for series in series_list]
    informative = [i for i, series in enumerate(series_list)
                   if len(series) >= 3 and len({rate for _, rate in series}) > 1]
    global_slope = median([fits[i][0] for i in informative]) if informative else 0.0
    global_residual = max(RESIDUAL_FLOOR,
                          median([fits[i][1] for i in informative]) if informative else 0.0)
    tier_slopes = {}
    for tier in {trophy["tier"] for trophy in case["trophies"]}:
        slopes = [fits[i][0] for i in informative if case["trophies"][i]["tier"] == tier]
        tier_slopes[tier] = median(slopes) if slopes else global_slope

    target_hour = hours_since_launch(case["target"])
    checkpoint_hour = hours_since_launch(case["at"])
    horizon_days = max(0.0, target_hour - checkpoint_hour) / 24
    rows = []
    for index, trophy in enumerate(case["trophies"]):
        series = series_list[index]
        if index not in informative:
            center = rate_logit(series[-1][1] if series else 0.05)
            spread = max(FLAT_SPREAD, DRIFT_PER_DAY * math.sqrt(horizon_days))
        else:
            slope, residual, effective_count, anchor = fits[index]
            reliability = effective_count / (effective_count + POOL_READS *
                                             (1 + (residual / global_residual) ** 2))
            pooled_slope = 0.75 * tier_slopes[trophy["tier"]] + 0.25 * global_slope
            slope = reliability * slope + (1 - reliability) * pooled_slope
            distance = max(0.0, math.log1p(target_hour) - math.log1p(series[-1][0]))
            damping = 1 / (1 + distance / DAMPING_SCALE)
            center = anchor + slope * distance * damping
            residual_variance = (reliability * residual ** 2 +
                                 (1 - reliability) * global_residual ** 2)
            spread = math.sqrt(max(RESIDUAL_FLOOR ** 2, residual_variance) +
                               DRIFT_PER_DAY ** 2 * horizon_days)
        rows.append({"id": trophy["id"], "q": rounded_quantiles(center, spread)})
    return {"trophies": rows}


def main():
    input_directory = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
    output_directory = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))
    output_directory.mkdir(parents=True, exist_ok=True)
    for case_file in sorted(input_directory.glob("*.in")):
        case = json.loads(case_file.read_text(encoding="utf-8"))
        output = json.dumps(forecast_case(case), separators=(",", ":"), allow_nan=False)
        (output_directory / (case_file.stem + ".out")).write_text(output, encoding="utf-8")


if __name__ == "__main__":
    main()
