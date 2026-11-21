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

## Fix round: independent statistical audit (2026-11-21)

This section records the audit-required revision and supersedes the model and
validation descriptions above for the current working tree. Earlier sections
are retained as the historical draft report. The relay reports that it committed
that draft as `c0f99f3`. This round used no Git commands, network, or kit starter;
the relay is responsible for the next commit. Edits are confined to `main.py`,
`tests/check_entry.py`, and this appended section.

The independent statistical audit returned **REVISE**: repeated unchanged reads
overstated information, denominator changes polluted slopes, the shrinkage and
uncertainty lacked a precision interpretation, and symmetric intervals concealed
different downside/upside risks. The implementation changes address those issues;
the measurements below also expose remaining calibration failures.

### Measurement qualification

The relay supplied an anchor asserting 825 exact integer reconstructed earner
counts. A direct check on the kit available in this workspace instead found
**15/825 exact within 1e-8**, with maximum distance from the nearest integer
**0.025000000000000355 earner**. This is consistent with rounding percentages
to two decimals; it does not independently confirm 825 exact integer counts.
The implementation uses the requested `E = rate*N/100` as a fractional count
estimate, without rounding to a fabricated integer. The shared-denominator
interpretation is retained, with this measurement qualification.

Both practice cases have exactly **one** distinct consecutive `(rates,N)`
state: p00 collapses 7 reads to 1 and p01 collapses 8 reads to 1. The final
timestamp of each unchanged block is retained. All entry fits, cohort updates,
effective counts, and standard errors use these deduplicated states. The house
reference baselines deliberately retain their original raw-read behavior.

### Revised central forecast

Time is now in **days**, with `x = ln(1+days_since_launch)`. This is a functional
change, not a relabeling of the former log-hours variable.

1. **Cohort estimates.** For each positive population change, compute
   `p_new = clamp(100*(E_now-E_previous)/(N_now-N_previous),0,100)` in percent.
   Divide by the current incumbent-proxy rate, floored at 0.01%, to estimate a
   cohort/incumbent ratio. Ratios are capped at 4 before pooling. Per-trophy
   ratios average these observations weighted by incoming population. Shrink
   toward a tier median and global median; tier weight is `m/(m+4)` for `m`
   usable peer trophies. Individual weight is `mass/(mass+0.25)`, where mass
   is cumulative incoming population divided by the last observed N. Cohort
   estimates are approximate: incumbent progress also contributes to delta E.
2. **Past dilution adjustment.** For an N increase, predict the rate mixture
   of previous incumbents and incoming players at the pooled cohort ratio.
   Its logit difference from the previous rate is the per-trophy dilution
   increment. Accumulate these increments and remove them before fitting.
3. **Common-shock separation.** On those dilution-adjusted series, obtain a
   preliminary slope from the median of logit differences separated by a lag
   of one quarter of the deduplicated observations (minimum one). For each
   consecutive state pair with common non-null trophies, compute
   `g_t = median(adjusted_logit_step) - median(preliminary_beta*dx_step)`.
   Remove cumulative g from each series. Missing trophies are omitted from
   the relevant step median, not treated as zero. This separates unexplained
   shared steps from the assumed smooth cross-trophy trend.
4. **WLS and hierarchy.** Fit the cleaned series with
   `w = exp((day-day_last)/2)`. The residual variance has a 0.02-logit standard
   deviation floor. Let `Vx = sum(w*(x-xbar)^2)/sum(w)` and
   `n_eff = sum(w)^2/sum(w^2)`. The slope error estimate is
   `SE_i^2 = residual_variance/(max(1,n_eff-2)*max(1e-12,Vx))`.
   Estimate cross-trophy slope variance as
   `tau_beta^2 = max(0, sample_variance(beta)-mean(SE_i^2))`.
   Individual shrinkage is exactly
   `lambda_i = tau_beta^2/(tau_beta^2+SE_i^2)` toward its tier estimate.
   Raw tier slopes use inverse-SE-variance means, with variance equal to the
   inverse summed precision. Tier-to-global shrinkage uses the analogous
   variance-of-tier-means minus mean tier-SE-variance estimate. The global
   slope uses weights `1/(tau_beta^2+SE_i^2)`. No ad-hoc read-count reliability
   multiplier remains.
5. **Anchor and bounded extrapolation.** Average the last three cleaned
   logits with recency weights, project that anchor to the last x using the
   shrunk slope, and restore the last cumulative historical adjustments.
   Historical shocks thereby remain in the current level; their past steps
   are not repeated in the future slope. For target distance dx, extrapolate
   by `dz = beta_used*1.2*(1-exp(-dx/1.2))`. This bounds either sign of future
   change. No temporal non-decreasing-median constraint is applied.

The optional future mixture projects recent delta-ln-N per day using the
endpoints of the last day of deduplicated population observations. This is
the time-weighted mean delta-ln-N trend. Its projected population factor is

```text
factor = exp(min(ln(1.5), max(0,trend)*2*(1-exp(-H_days/2))))
mixture_rate = (incumbent_rate + (factor-1)*cohort_rate)/factor
cohort_rate = clamp(pooled_ratio*incumbent_rate,0,100)
```

The population projection therefore has a 2-day decay and a cap of 50% total
new population. **The future mixture is disabled in the shipped constants**
by the mandatory gate below. Past dilution correction and cohort pooling
remain active; the comparison path stays executable in the test script.

### Revised uncertainty and asymmetric quantiles

For the recent anchor, let `n_anchor` be its effective count and `anchor_dx`
the distance from its weighted mean x to the last x. Use

```text
sigma_level^2 = residual_variance/n_anchor + SE_i^2*anchor_dx^2
sigma_T^2 = sigma_level^2 + dx^2*SE_i^2 + 0.25*H_days + 0.25^2
```

Here sigma_beta is the WLS slope standard error. The original residual-plus-
random-walk spread has been replaced. The learned multiplier vector is

```text
[-1.00*L, -0.25*L, 0, +0.75, +3.00]
L = 1 + 0.25*min(1, max(0, N_last/N_start_of_recent_window - 1))
```

Thus strong recent population growth widens both lower offsets by up to 25%.
The strictly positive widening constraint follows the audit; the initial
prototype grid that allowed zero was corrected before final validation.
The entry contains no symmetric normal-score construction; that construction
is retained only as a test comparison using the same revised centers/sigmas.

Fewer than three deduplicated usable reads, entirely equal printed rates, or
no usable history keep their required fallback median (last rate, or 0.05%).
Their sigma is at least **0.75**, independent of a short horizon, with the new
asymmetric offsets applied. Flat detection inspects printed rates directly,
avoiding roundoff in the shock-cleaned/reconstructed levels. Four-decimal
positive half-up rounding, bounds `[0.01,99.99]`, ordered quantiles, sorted file
iteration, and stable inverse logits remain as before.

### Calibration and validation protocol

All eight requested regimes appear in each disjoint seed set:

| Set | Seeds | Independent scenario paths | Rolling-origin cases | Trophy outcomes |
| --- | --- | ---: | ---: | ---: |
| DEV | 21000-21011 | 96 | 288 | 6,912 |
| TUNING | 31000-31011 | 96 | 288 | 6,912 |
| HOLDOUT | 51000-51039 | 320 | 960 | 23,040 |

Each scenario has 24 trophies, six per tier. Checkpoint age is 1.5, 2, or
2.5 days; rolling origins are 65%, 85%, and 100% of that age. Target horizon
is independently chosen from 1, 3, 4.520833333333333, and 6 days. Within a
scenario the three origins share the same simulated latent path; they are
correlated evaluations, not three independent trials.

The generative incumbent logit is
`a + b*(1-exp(-k*ln(1+day)))/k + individual_Brownian + common_Brownian`.
Tier intercepts are -7.2/-5.5/-4/-2.5 with uniform +/-0.6 variation; slopes
are 1/1.3/1.6/1.8 with uniform 0.8-to-1.2 multipliers. Ordinary curvature is
uniform 0.05-to-0.2. Individual diffusion is uniform 0.12-to-0.30 per sqrt(day)
and common diffusion is 0.08. Observation noise is uniform 0.025-to-0.14,
plus a page component with half that standard deviation.

Population starts at one million and grows at a uniform 0.01-to-0.05 log/day;
it is rounded to multiples of 10,000 so printed hundredth-percent rates
reconstruct integral synthetic counts. Incoming cohorts initially have a
uniform 0.1-to-0.6 fraction of incumbent earn rates, catching up exponentially
over a uniform 2-to-5-day scale. Counts/rates come from the population-weighted
mixture at each observation, rather than applying an unrelated rate shock to
an arbitrary N. Every seventh trophy is unavailable before day 0.25.

Regime-specific modifications:

- **normal_growth:** ordinary parameters above.
- **saturation:** curvature uniform 0.7-to-1.6 for stronger deceleration.
- **dilution:** a logistic population surge with factor 1.8-to-3.5 and
  transition scale 0.08 day, followed by an additional 0.12-to-0.35 log/day
  inflow. Event time is 50%-95% of checkpoint age. Cohorts earn and catch up
  by the same mixture process before and after the checkpoint.
- **common_recount:** every trophy receives the same signed 0.5-to-1.2
  persistent logit step at the event.
- **individual_recount:** every fourth trophy receives its own signed
  0.5-to-1.5 persistent logit step at the event.
- **stall_then_surge:** progress is stalled until the event; subsequent
  log-age growth starts there at three times the ordinary slope.
- **rate_decline:** slopes are negative, uniform -1.0-to--0.4.
- **duplicate_reads:** pages refresh every six hours while hourly recorder
  reads repeat their rates and N exactly between refreshes.

Ground truth is the future observation from the **same** curve, cohort,
diffusion, recount and refresh process. All future diffusion increments and
new observation noise are sampled in chronological order, independently of
the checkpoint read noise. A frozen-page target uses its last future refresh,
not a reused checkpoint refresh. No target is present in the case history.

DEV chooses tau by the median pinball term (equivalently absolute logit error),
then calibrates q/model sigma and the four widths/dilution factor by two
predetermined coordinate passes. q/model-sigma candidates use actual rounded
forecast loss. The width grid minimizes clamped-logit pinball before final
four-decimal percent rounding; final reported scores use rounded outputs.
There is no calibration stopping criterion that depends on HOLDOUT.
TUNING chooses the remaining design constants by mean pinball. After design
selection, DEV is recalibrated. All statistical constants are then frozen.

| Constant | Grid or fixed reason | Selected |
| --- | --- | ---: |
| SATURATION_TAU | DEV: 0.3, 0.6, 1.2, 2.4 log-age units | 1.2 |
| PROCESS_VARIANCE q | DEV: 0, 0.04, 0.09, 0.16, 0.25 logit variance/day | 0.25 |
| MODEL_SIGMA | DEV: 0.12, 0.25, 0.4, 0.6 logit | 0.25 |
| LOWER_05, UPPER_05 | DEV separately: 0.75, 1, 1.25, 1.5, 1.75, 2, 2.5, 3 | 1, 3 |
| LOWER_25, UPPER_25 | DEV separately: 0.25, 0.5, 0.75, 1, 1.25 | 0.25, 0.75 |
| DILUTION_LOWER | DEV: 0.25, 0.5, 1, 2; positivity is required | 0.25 |
| RECENCY_DAYS | TUNING: 0.5, 1, 2 days | 2 |
| COHORT_RATIO_CAP | TUNING: 1, 2, 4 | 4 |
| POPULATION_DECAY | TUNING: 0.5, 1, 2 days | 2 |
| POPULATION_CAP | TUNING: 1.5, 2, 3 times latest N | 1.5 |
| POPULATION_LOOKBACK | Fixed one-day recent population window | 1 day |
| COHORT_PRIOR_MASS | Fixed prior mass relative to last N, regularizes small inflows | 0.25 |
| Tier cohort pseudocount | Fixed small-group regularization in m/(m+4) | 4 trophies |
| FIT_NOISE_FLOOR | Fixed floor for finite standard errors under printed/quantized rates | 0.02 logit |
| FLAT_SIGMA | Retained conservative floor above required 0.5 | 0.75 logit |
| Anchor size | Retained small recent-level average | 3 states |
| Preliminary slope lag | Robust long differences; minimum 1, otherwise count//4 | quarter of states |
| 1e-12; max(1,n_eff-2) | Numeric variance protection and two fitted line coefficients | fixed |

For the final design, DEV median-error losses for tau 0.3/0.6/1.2/2.4 were
0.831487/0.772217/0.755796/0.769411. TUNING coordinate losses were:

- Recency 0.5/1/2: 0.518688/0.517565/0.517240.
- Cohort cap 1/2/4: 0.518810/0.517746/0.517240.
- Population decay 0.5/1/2: 0.517851/0.517240/0.516794.
- Population cap 1.5/2/3: 0.516706/0.516766/0.516794.

These TUNING comparisons use the pre-gate mixture candidate and preceding
coordinate selections. They are not the final non-mixture scores below.
Widths satisfy lower-05 >= lower-25 and upper-05 >= upper-25. Several selected
values are grid boundaries, so no claim of a global optimum is made.

### Mandatory mixture gate and holdout discipline

The audit instructions simultaneously prohibit HOLDOUT selection and require
choosing the mixture using HOLDOUT dilution outcomes. This round treats the
explicit binary gate as the **sole exception**. It is therefore not accurate
to call that binary choice entirely out of sample. No other parameter,
calibration, scenario weight, or design was changed based on final HOLDOUT.

An earlier prototype inspected seeds 41000-41039 before the positive-widening
constraint and printed-rate flat detection were corrected. Those results were
discarded from final validation. The correction was requirement-driven; final
HOLDOUT was replaced with untouched seeds 51000-51039 after the design and DEV
calibration were fixed. Subsequent reruns reproduce this same frozen result.

On final HOLDOUT dilution cases, with identical frozen uncertainty constants:

| Variant | Loss |
| --- | ---: |
| No future mixture | **0.382763** |
| Future cohort mixture | 0.382877 |
| Persistence | 0.539793 |

The mixture does not beat the non-mixture forecast, so
`USE_COHORT_MIXTURE = False`. No further fit was performed after this gate.
The shipped dilution model beats persistence; no dilution shortfall remains
on these simulations. The small mixture difference is not a statistical
significance claim.

### Final loss measurements

Mean doubled pinball on the scoring logit scale, all trophies/quantiles
equally weighted. Each regime has the same number of cases. The house curve
reference remains ordinary least squares on log **hours**, last 24 hours,
RMS residual plus the specified 0.5-per-sqrt-day spread; it is not silently
upgraded to the entry's log-days basis or deduplication.

| Set / regime | Entry | Persistence | Curve |
| --- | ---: | ---: | ---: |
| DEV normal growth | 0.289568 | 0.513388 | 0.324897 |
| DEV saturation | 0.344769 | 0.279967 | 0.294811 |
| DEV dilution | 0.366428 | 0.475959 | 0.475803 |
| DEV common recount | 0.403924 | 0.538797 | 0.651955 |
| DEV individual recount | 0.345529 | 0.535090 | 0.427520 |
| DEV stall then surge | 1.668334 | 2.198273 | 1.962038 |
| DEV rate decline | 0.307330 | 0.365204 | 0.305277 |
| DEV duplicate reads | 0.300543 | 0.554991 | 0.352933 |
| **DEV aggregate** | **0.503303** | **0.682709** | **0.599404** |
| TUNING normal growth | 0.296520 | 0.538013 | 0.329998 |
| TUNING saturation | 0.319944 | 0.283549 | 0.292041 |
| TUNING dilution | 0.344250 | 0.491808 | 0.509456 |
| TUNING common recount | 0.423287 | 0.611517 | 0.642076 |
| TUNING individual recount | 0.347326 | 0.582342 | 0.429622 |
| TUNING stall then surge | 1.828643 | 2.275039 | 2.184548 |
| TUNING rate decline | 0.287979 | 0.344984 | 0.291110 |
| TUNING duplicate reads | 0.303880 | 0.542657 | 0.345375 |
| **TUNING aggregate** | **0.518979** | **0.708739** | **0.628028** |
| HOLDOUT normal growth | 0.308043 | 0.602188 | 0.355980 |
| HOLDOUT saturation | 0.338488 | 0.304293 | 0.307109 |
| HOLDOUT dilution | 0.382763 | 0.539793 | 0.513291 |
| HOLDOUT common recount | 0.419557 | 0.663162 | 0.698741 |
| HOLDOUT individual recount | 0.362668 | 0.630655 | 0.457086 |
| HOLDOUT stall then surge | 1.887726 | 2.464959 | 2.166059 |
| HOLDOUT rate decline | 0.313593 | 0.361084 | 0.305222 |
| HOLDOUT duplicate reads | 0.309594 | 0.619907 | 0.365807 |
| **HOLDOUT aggregate** | **0.540304** | **0.773255** | **0.646162** |

The symmetric-normal comparison using the revised centers and sigmas scores
0.540996 on DEV, 0.566447 on TUNING, and 0.592909 on HOLDOUT. Asymmetry helps
aggregate loss, but is not superior in every individual regime.

### HOLDOUT quantile coverage

Coverage here is the empirical CDF `fraction(actual_percent <= quantile)`
using the rounded output quantiles; it is not 90% interval inclusion or
absolute error. Desired fractions are 0.05/0.25/0.50/0.75/0.95. Values below
are measured fractions, not percentages. Each regime has 2,880 outcomes.

| Regime | q05 | q25 | q50 | q75 | q95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Normal growth | 0.010764 | 0.168403 | 0.312847 | 0.838542 | 1.000000 |
| Saturation | 0.052083 | 0.425694 | 0.642361 | 0.967361 | 1.000000 |
| Dilution | 0.034722 | 0.181944 | 0.297917 | 0.733333 | 1.000000 |
| Common recount | 0.086806 | 0.287500 | 0.405208 | 0.770486 | 0.999306 |
| Individual recount | 0.037500 | 0.210417 | 0.338889 | 0.797569 | 0.999653 |
| Stall then surge | 0.000000 | 0.000347 | 0.001042 | 0.014583 | 0.497222 |
| Rate decline | 0.091319 | 0.373264 | 0.555903 | 0.934028 | 1.000000 |
| Duplicate reads | 0.010069 | 0.203472 | 0.359375 | 0.866319 | 1.000000 |
| **Aggregate** | **0.040408** | **0.231380** | **0.364193** | **0.740278** | **0.937023** |

Median coverage is materially below 0.5. The stalled-growth cases have severe
upside misses, while the same broad upper tail overcovers most other regimes.
The near-nominal aggregate tails do **not** establish conditional calibration.

### Re-executed checks and remaining limits

Latest practice execution: Python **3.14.6**, Windows, `-P -s`, one case per
subprocess, `PYTHONHASHSEED=0`, unrelated cwd, two fresh output directories.

| Case | Rows | Bytes | First wall ms | Second wall ms | Valid / identical |
| --- | ---: | ---: | ---: | ---: | --- |
| p00 | 55 | 2,650 | 88.168 | 73.244 | yes / yes |
| p01 | 55 | 2,650 | 80.836 | 75.176 | yes / yes |

JSON keys, ids exactly once, five finite ordered quantiles, percent bounds,
and the 65,536-byte limit pass. Timings include subprocess startup and file
I/O and are below the 8,000 ms limit on this host. No Python 3.13 container
execution or container resource-limit measurement was performed.

A separate final-sized runtime case had **55 trophies and 360 distinct
10-minute states**, a checkpoint at day 2.5, and target day 7. A fresh `-P -s`
subprocess produced a valid **2,822-byte** output in **179.827 ms**. Its input
and output are under ignored `.checks/`. Both Python source files also pass
`ast.parse(..., feature_version=(3,13))`; this checks grammar, not execution
on Python 3.13.

Focused assertions also pass for days conversion, missing/empty histories,
0/100%, sparse/flat histories, last-timestamp deduplication, byte equivalence
of a frozen block and its deduplicated state, rounding, extreme logits,
precision shrinkage, zero-earner incoming cohorts, mixture-induced decline,
recovery of a known slope after a shared logit jump, and decreasing medians.

Reproduce calibration and the frozen model checks respectively:

```powershell
python -P -s tests/check_entry.py --tune
python -P -s tests/check_entry.py --kit 'path/to/competition-kit'
```

`--tune` resets the documented initial candidate, calibrates only DEV and
TUNING, and then executes the explicitly requested HOLDOUT mixture gate. It
prints proposed constants; it does not edit the entry. The default run uses
the constants saved in `main.py` and reproduces the final scores above.
Generated checks remain under ignored `.checks/`; only `main.py` is imported
by the grader, and it does not import the test script or read those artifacts.

Remaining limitations:

- Shared smooth progress, shared shocks, and selection effects are not fully
  identifiable from aggregate rates. The robust trend/shock decomposition is
  an assumption; it is not observation of individual player histories.
- Delta-E/delta-N also includes incumbent progress and count rounding. Tier
  pooling reduces variance but cannot identify true incoming-cohort ability.
- The future population/cohort mixture failed its gate and is disabled.
  Past dilution adjustment is still estimated, rather than known exactly.
- Deduplication removes exact page repeats, not all serial dependence.
  WLS standard errors use an approximate independent-error interpretation.
- DEV and TUNING each contain only 12 independent seeds per regime; rolling
  origins add evaluations but not independent paths. Grid-boundary choices
  and the pre-rounding width objective limit calibration precision.
- The gate uses final dilution outcomes for a binary selection, as explicitly
  requested; the resulting variant selection is not untouched holdout proof.
- Saturation loses to persistence on HOLDOUT, and the curve baseline wins
  the rate-decline regime. The aggregate improvement does not mean every
  individual trophy or regime improves.
- The stall-then-surge regime remains poorly calibrated. The shared widths
  trade its large upside errors against overcoverage elsewhere. Flat unseen
  refreshes are protected by a floor, not comprehensively calibrated here.
- Final-case accuracy, beating the house AI, transcript acceptance, and prize
  eligibility remain unverified. No competition submission was made here.
