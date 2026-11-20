# Build report

## Result and scope

`main.py` is an original, single-file Python standard-library forecaster. It
reads the supplied history only and writes five percent quantiles per trophy.
No starter code, downloaded code, network data, or third-party packages were
used. The local kit rules and practice input were read; `starter/` was not read.

The first working draft was written and executed in this session. A later
assistant message edited `main.py`, changing pooling strength from 8 to 16 and
daily drift from 0.45 to 0.50 using the selection backtest below.

**Commit requirement BLOCKED:** the repository's `.git` directory is read-only
in this task's sandbox. Shell `git add` and `git commit` failed with permission
denied creating `.git/index.lock`. Thus the requested draft-commit/revision-commit
sequence could not be completed. Files are saved locally, without commits or
push. The observed branch is `main`, with no HEAD. Final repository readback
shows origin `https://github.com/UkkyaGuiyo/gta6-trophy-forecast.git` (the
initial readback had no remote). A second staging attempt also failed.

## Model

For each non-null trophy observation, let `h` be hours from
2026-11-19T00:00:00Z, `x = ln(1+h)`, and
`y = ln(p/(1-p))`, where `p = clamp(rate, 0.01, 99.99)/100`.
The launch-night observations just before hour zero remain negative hours;
the supplied earliest timestamp is within the domain of `ln(1+h)`.

For a trophy whose last usable observation is at `h_last`, fit an intercept
and slope `b` by weighted least squares, with
`w = exp((h-h_last)/24)`. Weighted residual RMS is `r`, and effective count is
`n_eff = (sum(w))^2 / sum(w^2)`. This counts independent-looking observations,
not literal trophy earners.

Compute the median informative slope `B` over all trophies and the median
informative slope `B_tier` within each tier. An informative trophy has at least
three usable observations and at least two distinct printed rates. A tier
without such a trophy uses `B`. The global residual scale is
`R = max(0.12, median(r))`. If no trophies are informative, both slope pools
are zero and `R = 0.12`.

Shrink by

```text
alpha = n_eff / (n_eff + 16 * (1 + (r/R)^2))
pooled = 0.75 * B_tier + 0.25 * B
b_used = alpha * b + (1-alpha) * pooled
```

More observations raise `alpha`; worse residual fit relative to the global
scale lowers it. The recent anchor `A` is the weighted average of the last
three usable logits, with the same recency weights. The extrapolating line is
shifted to equal `A` at `x_last`, preventing an old intercept from overriding
the recent level.

Using the case's actual `target`, let `D = max(0, x_target-x_last)` and
`d = max(0, target-checkpoint)` in days. Then

```text
damping = 1 / (1 + D/2)
mu = A + b_used * D * damping
sigma_fit^2 = max(0.12^2, alpha*r^2 + (1-alpha)*R^2)
sigma_total = sqrt(sigma_fit^2 + 0.50^2*d)
```

For any positive extrapolation, damping is below one and decreases as the
target moves farther away. It damps either sign of slope; a recent dilution
can yield a negative slope. No positivity assumption or handcrafted
trophy-name adjustments are used.

Quantiles are `100*logistic(mu + z*sigma_total)`, with the prescribed
`z = [-1.645, -0.674, 0, 0.674, 1.645]`. The inverse logit branches by sign to
avoid exponential overflow. For nonnegative percent values, rounding is
`floor(value*10000 + 0.5)/10000`, implementing the positive-number rule of
JavaScript `Math.round`, rather than Python's ties-to-even `round`. Final
values are clamped to `[0.01,99.99]` and made non-decreasing.

With fewer than three usable reads, or an entirely unchanged printed series,
the center is the last rate. A trophy with no usable reads gets 0.05 percent.
These cases use `sigma = max(0.75, 0.50*sqrt(d))`, including a short horizon.
Zero and 100 percent observations therefore remain finite after clamping.

## Constants and decisions

| Constant | Reason / selection evidence |
| --- | --- |
| Launch epoch; hours/day = 24 | Specified origin and unit conversion. |
| Clamp 0.01 to 99.99 | Scoring convention and required output bounds. |
| Five normal scores above | Required quantile construction. |
| Recency e-folding time 24 hours | Lowest selection loss among 6, 12, 24 hours; retains more signal than aggressive recency in noisy series. Its half-life is about 16.64 hours. |
| Pool strength 16 | Lowest selection loss among 4, 8, 16 pseudo-observations in the quality penalty. Stronger pooling reduced the damage from noisy or disrupted slopes. |
| Tier/global mix 0.75/0.25 | Fixed design choice: tier gets most prior weight while global pooling remains present even for a one-trophy tier. Not separately optimized. |
| Three recent observations | Fixed, small noise-reducing anchor; at the final read cadence its lag is short relative to the forecast horizon. Not separately optimized. |
| Residual ratio squared | Variance-based penalty: relative residual variance increases the prior's influence. The added 1 supplies pooling even for an exact fitted line. |
| Damping scale 2 | Lowest selection loss among 2, 4, 8; at log-age distance 1 it retains two-thirds of the pooled slope. It has no target-specific hardcoded horizon. |
| Drift 0.50 per sqrt(day) | Lowest selection loss among 0.35, 0.40, 0.45, 0.50, within the requested region; absorbs extrapolation and event uncertainty in addition to fitted residuals. |
| Residual floor 0.12 logit | Fixed uncertainty allowance, roughly a 13% odds factor per standard deviation, avoiding zero fit uncertainty on a nearly exact line. Not separately optimized. |
| Degenerate minimum sigma 0.75 | Fixed conservative floor above the required 0.5, independent of horizon. It allows odds to vary by about a factor 3.43 across either half of the 90% interval. The six non-flat simulations do not calibrate unseen short-horizon refreshes. |
| Missing-history median 0.05% | Required fallback, not inferred from simulation. |
| Minimum count 3 | Required degeneracy boundary; allows estimating a line with residual information. |
| 10,000 and 0.5 in rounding | Four decimal places and the specified half-up rounding rule for positive values. |

These are a small coordinate sweep and explicit fixed design choices, not a
claim of a global optimum. No final-case outcomes were available for tuning.

## Synthetic backtest

`tests/check_entry.py` generates data and independently implements the two
code baselines from their prose descriptions. It never imports a kit baseline.
It scores **twice** the usual pinball loss, matching the rules:

```text
error = logit(truth) - logit(quantile)
loss = 2 * max(tau*error, (tau-1)*error)
```

All trophies and all five levels receive equal weight. Regimes have equal
case and trophy counts, so their means receive equal aggregate weight.
All forecasts, including baselines, are evaluated after four-decimal percent
rounding and scoring-time clamping. The curve reference uses the checkpoint's
last 24 hours, ordinary least squares with intercept on log age, and residual
RMS combined in quadrature with `0.5*sqrt(days)`. Fewer than three observations
fall back to persistence. The brief does not specify a residual degrees-of-
freedom correction; this implementation uses RMS (denominator n).

Each synthetic case has 24 trophies, six of each tier, with a checkpoint at
36, 48, or 60 hours and observations every half hour. Forecast horizons are
24, 72, 108.5, or 144 hours. Latent logits follow

```text
a + b * (1-exp(-k*ln(1+h))) / k + Brownian(h)
```

Tier intercepts are -8.5/-6.2/-4.8/-3.2 plus uniform +/-1; base slopes are
0.85/1.10/1.35/1.50 with independent uniform 0.7-to-1.3 multipliers.
Curvature `k` varies independently from 0.12 to 0.45. Thus the latent slope
with respect to log age decreases continuously, unlike either straight-line
baseline. Brownian diffusion varies per case from 0.25 to 0.55 per sqrt(day).
Read noise has independent trophy and common-page Gaussian components; the
common component's standard deviation is half the trophy noise scale.
Printed observations and truth are rounded to two decimal percent places.
Every seventh trophy is missing before hour six.

The six regimes are:

- Fast growth: base slopes multiplied by 1.6; noise scale 0.12.
- Slow growth: slopes multiplied by 0.35; noise 0.12.
- High observation noise: ordinary slopes; noise 0.45.
- Low observation noise: ordinary slopes; noise 0.04.
- Tracking surge: ordinary slopes/noise; player count multiplies by a uniform
  2-to-4 factor at a time between 40% and 70% of the checkpoint age. Rates
  divide by `1+(surge-1)*exp(-(h-event)/72)` after that event, representing
  immediate dilution followed by newcomers catching up.
- Recount: ordinary slopes/noise; a signed uniform 0.4-to-0.9 logit step at
  an event between 40% and 70% of the checkpoint age, persisting to the target.

The target comes from the **same latent curve, event, and diffusion process**.
Its future Brownian increment and both observation-noise components are drawn
fresh after generating the checkpoint history. There is no reuse of a history
read's noise, no target in the input, and no target chosen to match a forecast.
The test script alone uses seeded randomness; `main.py` uses none.

### Selection

Selection seeds 100 through 107: 48 cases, 1,152 trophy outcomes, 5,760
pinball terms. Coordinate sweep order: recency, pooling, damping, drift.
Each sweep holds the preceding selections fixed. `--tune` explicitly resets
the original draft's four tunable constants before reproducing the sweep.

| Stage | Entry aggregate | Persistence | Curve |
| --- | ---: | ---: | ---: |
| First draft: 24 / 8 / 2 / 0.45 | 0.547402 | 0.589667 | 0.861474 |
| Selected: 24 / 16 / 2 / 0.50 | 0.545376 | 0.589667 | 0.861474 |

For recency 6/12/24 the losses were 0.572542/0.565116/0.547402.
For pooling 4/8/16: 0.548815/0.547402/0.545607.
For damping 2/4/8: 0.545607/0.553494/0.563886.
For drift 0.35/0.40/0.45/0.50:
0.557564/0.549505/0.545607/0.545376.

### Held-out results

Seeds 1000 through 1031, not used in selection: 192 cases, 4,608 trophy
outcomes, 23,040 pinball terms. These are the final revised model's measured
mean losses, rounded here to six decimals:

| Regime | Entry | Persistence | Curve |
| --- | ---: | ---: | ---: |
| Fast | 0.415388 | 0.560003 | 0.532021 |
| Slow | 0.401873 | 0.408448 | 0.513057 |
| High noise | 0.502850 | 0.606261 | 0.607773 |
| Low noise | 0.401385 | 0.458205 | 0.520553 |
| Tracking surge | 0.663775 | 0.632081 | 1.411129 |
| Recount | 0.422334 | 0.469547 | 0.709416 |
| **Aggregate** | **0.467934** | **0.522424** | **0.715658** |

The aggregate beats both code baselines. Tracking surge remains worse than
persistence. There is no test or claim of beating the house AI forecast.

## Executed checks

Local interpreter: **Python 3.14.6 on Windows**, not the grader's Python
3.13-slim. Child runs used `-P -s`, `PYTHONHASHSEED=0`, and an unrelated
working directory. Each run had one practice file in a fresh input directory
and its own previously nonexistent output directory. Timings include process
startup, imports, input parsing, forecasting and output writing.

Final-revision practice run:

| Case | Trophies | Output bytes | First wall ms | Second wall ms | Valid | Byte-identical |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| p00 | 55 | 2,647 | 87.355 | 71.133 | yes | yes |
| p01 | 55 | 2,647 | 82.901 | 71.451 | yes | yes |

1. **Validity:** both parse as JSON, have exactly the required keys, all
   trophy ids exactly once, five finite ordered numeric quantiles per row,
   bounds `[0,100]`, and file size below 65,536 bytes.
2. **Determinism:** both pairs of files are byte-identical across fresh runs.
3. **Timing:** all four measured practice process runs are below 88 ms on this
   host, against the 8,000 ms budget. This is not a container timing guarantee.
4. **Synthetic:** the held-out aggregate superiority assertion passes using
   the results above.

Additional assertions pass for missing early trophies, completely missing
trophies/history, exactly zero and 100%, sparse and flat histories, dates
before launch, short-horizon spread, rounding above/below a half increment,
and extreme logits. Both source files also parse using Python's 3.13 grammar
mode; that checks syntax rather than execution on that interpreter version.

Reproduce (substitute the kit directory):

```powershell
python -P -s tests/check_entry.py --tune --kit 'path/to/kit'
python -P -s tests/check_entry.py --kit 'path/to/kit'
```

Fresh practice artifacts remain under ignored `.checks/` directories. The
test script is never imported by the entrypoint.

## Limitations

- Synthetic superiority is conditional on the generator. Real earn rates,
  player selection, refresh lag, and recounts can differ substantially.
- A surge can contaminate both trophy slopes and tier priors; no explicit
  denominator correction is applied. This is visible in the per-regime loss.
- Repeated identical page reads are correlated. Effective count discounts
  old observations but does not estimate the page's actual refresh schedule.
- Sparse/flat forecasts deliberately keep their last median; a symmetric
  wide band cannot correct a systematic unseen refresh direction.
- Tier membership is an imperfect proxy for completion dynamics. Trophy
  descriptions and points are not modeled.
- Uncertainty is marginal per trophy. The file format does not express joint
  correlation or cross-trophy logical constraints.
- Inputs are assumed to follow the case schema and launch-time domain. Data
  years before launch or malformed files are outside this entry's contract.
- Python 3.13-slim runtime, its CPU/memory limits, final-case accuracy, public
  practice score, transcript gates, submission, and leaderboard results were
  not verified. Local files alone are not a competition submission.
- The required two commits remain blocked by read-only Git metadata. No push
  was attempted.
