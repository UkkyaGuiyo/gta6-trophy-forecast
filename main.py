"""Count-aware, shock-separated trophy forecasts with calibrated quantiles."""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from statistics import mean, median, variance


LAUNCH = datetime.fromisoformat("2026-11-19T00:00:00+00:00")
RECENCY_DAYS = 2.0
SATURATION_TAU = 1.2
PROCESS_VARIANCE = 0.25
MODEL_SIGMA = 0.25
LOWER_25 = 0.25
LOWER_05 = 1.0
UPPER_25 = 0.75
UPPER_05 = 3.0
DILUTION_LOWER = 0.25
FLAT_SIGMA = 0.75
FIT_NOISE_FLOOR = 0.02
COHORT_PRIOR_MASS = 0.25
COHORT_RATIO_CAP = 4.0
POPULATION_LOOKBACK = 1.0
POPULATION_DECAY = 2.0
POPULATION_CAP = 1.5
USE_COHORT_MIXTURE = False


def days_since_launch(timestamp):
    return (datetime.fromisoformat(timestamp) - LAUNCH).total_seconds() / 86400


def rate_logit(rate):
    bounded = min(99.99, max(0.01, rate)) / 100
    return math.log(bounded / (1 - bounded))


def logit_rate(value):
    if value >= 0:
        return 100 / (1 + math.exp(-value))
    exponential = math.exp(value)
    return 100 * exponential / (1 + exponential)


def rounded_quantiles(center, spread, growth=0.0):
    lower_factor = 1 + DILUTION_LOWER * min(1.0, growth)
    offsets = (-LOWER_05 * lower_factor, -LOWER_25 * lower_factor,
               0.0, UPPER_25, UPPER_05)
    result = []
    for score in offsets:
        rate = logit_rate(center + score * spread)
        rounded = math.floor(rate * 10000 + 0.5) / 10000
        result.append(max(result[-1] if result else 0.01,
                          min(99.99, max(0.01, rounded))))
    return result


def deduplicate_history(history):
    states = []
    for read in history:
        if (states and read["rates"] == states[-1]["rates"] and
                read["players_tracked"] == states[-1]["players_tracked"]):
            states[-1] = read
        else:
            states.append(read)
    return states


def estimate_cohort_ratios(states, trophies, counts):
    estimates = [[] for _ in trophies]
    for previous, current, old_counts, new_counts in zip(states, states[1:], counts, counts[1:]):
        incoming = current["players_tracked"] - previous["players_tracked"]
        if incoming <= 0:
            continue
        for i, rate in enumerate(current["rates"]):
            if rate is None or old_counts[i] is None:
                continue
            cohort_rate = min(100.0, max(0.0, 100 * (new_counts[i] - old_counts[i]) / incoming))
            ratio = min(COHORT_RATIO_CAP, cohort_rate / max(0.01, rate))
            estimates[i].append((ratio, incoming))
    raw = [sum(r * n for r, n in values) / sum(n for _, n in values)
           if values else None for values in estimates]
    global_ratio = median([r for r in raw if r is not None]) if any(r is not None for r in raw) else 1.0
    result = []
    for i, trophy in enumerate(trophies):
        peers = [raw[j] for j, other in enumerate(trophies)
                 if other["tier"] == trophy["tier"] and raw[j] is not None]
        tier_ratio = median(peers) if peers else global_ratio
        tier_weight = len(peers) / (len(peers) + 4)
        prior = tier_weight * tier_ratio + (1 - tier_weight) * global_ratio
        mass = sum(n for _, n in estimates[i]) / states[-1]["players_tracked"] if states else 0.0
        weight = mass / (mass + COHORT_PRIOR_MASS)
        result.append(weight * (raw[i] if raw[i] is not None else prior) + (1 - weight) * prior)
    return result


def robust_series_slope(series):
    if len(series) < 3:
        return 0.0
    lag = max(1, len(series) // 4)
    slopes = [(b[1] - a[1]) / (b[0] - a[0]) for a, b in zip(series, series[lag:])
              if b[0] > a[0]]
    return median(slopes) if slopes else 0.0


def clean_observations(states, trophies, ratios):
    days = [days_since_launch(state["at"]) for state in states]
    xs = [math.log1p(day) for day in days]
    dilution = [[0.0] * len(trophies) for _ in states]
    for t in range(1, len(states)):
        previous, current = states[t - 1], states[t]
        incoming = max(0, current["players_tracked"] - previous["players_tracked"])
        for i, rate in enumerate(previous["rates"]):
            step = 0.0
            if incoming and rate is not None:
                mixed = (rate * previous["players_tracked"] +
                         min(100.0, rate * ratios[i]) * incoming) / current["players_tracked"]
                step = rate_logit(mixed) - rate_logit(rate)
            dilution[t][i] = dilution[t - 1][i] + step
    base = [[None if rate is None else rate_logit(rate) - dilution[t][i]
             for i, rate in enumerate(state["rates"])] for t, state in enumerate(states)]
    preliminary = [robust_series_slope([(xs[t], row[i]) for t, row in enumerate(base)
                                        if row[i] is not None]) for i in range(len(trophies))]
    common = [0.0] * len(states)
    for t in range(1, len(states)):
        available = [i for i in range(len(trophies)) if base[t][i] is not None
                     and base[t - 1][i] is not None]
        shock = 0.0
        if available:
            observed_step = median([base[t][i] - base[t - 1][i] for i in available])
            trend_step = median([preliminary[i] * (xs[t] - xs[t - 1]) for i in available])
            shock = observed_step - trend_step
        common[t] = common[t - 1] + shock
    series = [[(days[t], xs[t], base[t][i] - common[t], dilution[t][i] + common[t])
               for t in range(len(states)) if base[t][i] is not None]
              for i in range(len(trophies))]
    return series


def fit_trophy_series(series):
    weights = [math.exp((row[0] - series[-1][0]) / RECENCY_DAYS) for row in series]
    xs = [row[1] for row in series]
    ys = [row[2] for row in series]
    total = sum(weights)
    mean_x = sum(w * x for w, x in zip(weights, xs)) / total
    mean_y = sum(w * y for w, y in zip(weights, ys)) / total
    x_variance = sum(w * (x - mean_x) ** 2 for w, x in zip(weights, xs)) / total
    slope = (sum(w * (x - mean_x) * (y - mean_y)
                 for w, x, y in zip(weights, xs, ys)) / (total * x_variance)
             if x_variance > 1e-12 else 0.0)
    residual_variance = max(FIT_NOISE_FLOOR ** 2,
                            sum(w * (y - mean_y - slope * (x - mean_x)) ** 2
                                for w, x, y in zip(weights, xs, ys)) / total)
    effective_count = total * total / sum(w * w for w in weights)
    slope_variance = residual_variance / (max(1.0, effective_count - 2) * max(1e-12, x_variance))
    recent_weights = weights[-3:]
    recent_total = sum(recent_weights)
    anchor = sum(w * y for w, y in zip(recent_weights, ys[-3:])) / recent_total
    anchor_distance = xs[-1] - sum(w * x for w, x in zip(recent_weights, xs[-3:])) / recent_total
    anchor_count = recent_total ** 2 / sum(w * w for w in recent_weights)
    level_variance = residual_variance / anchor_count + slope_variance * anchor_distance ** 2
    return {"slope": slope, "slope_variance": slope_variance, "anchor": anchor,
            "anchor_distance": anchor_distance, "level_variance": level_variance,
            "effective_count": effective_count}


def hierarchical_slopes(fits, trophies):
    usable = [i for i, fit in enumerate(fits) if fit is not None]
    if not usable:
        return [0.0] * len(trophies)
    slopes = [fits[i]["slope"] for i in usable]
    errors = [fits[i]["slope_variance"] for i in usable]
    tau_squared = max(0.0, variance(slopes) - mean(errors)) if len(slopes) > 1 else 0.0
    precision = [1 / (tau_squared + se) for se in errors]
    global_slope = sum(w * b for w, b in zip(precision, slopes)) / sum(precision)
    groups = {}
    for tier in sorted({trophy["tier"] for trophy in trophies}):
        members = [i for i in usable if trophies[i]["tier"] == tier]
        if members:
            weights = [1 / fits[i]["slope_variance"] for i in members]
            groups[tier] = (sum(w * fits[i]["slope"] for w, i in zip(weights, members)) / sum(weights),
                            1 / sum(weights))
    tier_dispersion = (max(0.0, variance([b for b, _ in groups.values()]) -
                           mean(se for _, se in groups.values())) if len(groups) > 1 else 0.0)
    shrunk_tiers = {}
    for tier, (slope, error) in groups.items():
        weight = tier_dispersion / (tier_dispersion + error)
        shrunk_tiers[tier] = weight * slope + (1 - weight) * global_slope
    result = [global_slope] * len(trophies)
    for i in usable:
        weight = tau_squared / (tau_squared + fits[i]["slope_variance"])
        result[i] = weight * fits[i]["slope"] + (1 - weight) * shrunk_tiers[trophies[i]["tier"]]
    return result


def project_population(states, horizon):
    if len(states) < 2:
        return 1.0, 0.0
    last_day = days_since_launch(states[-1]["at"])
    recent = [state for state in states if days_since_launch(state["at"]) >= last_day - POPULATION_LOOKBACK]
    if len(recent) < 2:
        return 1.0, 0.0
    elapsed = last_day - days_since_launch(recent[0]["at"])
    if elapsed <= 0:
        return 1.0, 0.0
    delta_log_n = math.log(recent[-1]["players_tracked"] / recent[0]["players_tracked"])
    trend = max(0.0, delta_log_n / elapsed)
    projected = min(math.log(POPULATION_CAP),
                    trend * POPULATION_DECAY * (-math.expm1(-horizon / POPULATION_DECAY)))
    return math.exp(projected), max(0.0, math.expm1(delta_log_n))


def forecast_components(case):
    states = deduplicate_history(case["history"])
    counts = [[None if rate is None else rate * state["players_tracked"] / 100
               for rate in state["rates"]] for state in states]
    ratios = estimate_cohort_ratios(states, case["trophies"], counts)
    series_list = clean_observations(states, case["trophies"], ratios)
    informative = [len(series) >= 3 and len({state["rates"][i] for state in states
                                           if state["rates"][i] is not None}) > 1
                   for i, series in enumerate(series_list)]
    fits = [fit_trophy_series(series) if informative[i] else None
            for i, series in enumerate(series_list)]
    slopes = hierarchical_slopes(fits, case["trophies"])
    target_day = days_since_launch(case["target"])
    horizon = max(0.0, target_day - days_since_launch(case["at"]))
    population_factor, growth = project_population(states, horizon)
    components = []
    for index, trophy in enumerate(case["trophies"]):
        series = series_list[index]
        distance = max(0.0, math.log1p(target_day) - series[-1][1]) if series else 0.0
        if fits[index] is None:
            center = series[-1][2] + series[-1][3] if series else rate_logit(0.05)
            level_variance, slope_variance = 0.0, 0.0
        else:
            fit = fits[index]
            increment = slopes[index] * SATURATION_TAU * (-math.expm1(-distance / SATURATION_TAU))
            center = fit["anchor"] + slopes[index] * fit["anchor_distance"] + series[-1][3] + increment
            if USE_COHORT_MIXTURE:
                incumbent = logit_rate(center)
                incoming = min(100.0, incumbent * ratios[index])
                center = rate_logit((incumbent + (population_factor - 1) * incoming) / population_factor)
            level_variance = fit["level_variance"]
            slope_variance = fit["slope_variance"]
        components.append({"id": trophy["id"], "center": center, "level_variance": level_variance,
                           "slope_variance": slope_variance, "dx": distance, "horizon": horizon,
                           "growth": growth, "flat": fits[index] is None})
    return components


def component_quantiles(component):
    spread = math.sqrt(component["level_variance"] + component["dx"] ** 2 * component["slope_variance"] +
                       PROCESS_VARIANCE * component["horizon"] + MODEL_SIGMA ** 2)
    if component["flat"]:
        spread = max(FLAT_SIGMA, spread)
    return rounded_quantiles(component["center"], spread, component["growth"])


def forecast_case(case):
    return {"trophies": [{"id": component["id"], "q": component_quantiles(component)}
                         for component in forecast_components(case)]}


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
