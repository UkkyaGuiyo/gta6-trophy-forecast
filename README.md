# GTA VI trophy rarity forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/UkkyaGuiyo/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/UkkyaGuiyo/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts Exophase PlayStation 5 trophy earn rates at each input case's target
time. The entry is one Python standard-library file, written for Python 3.13.

The model fits recent-weighted logit trends against log age, pools uncertain
slopes by trophy tier and across trophies, anchors at recent observations,
and damps future growth. Residual and time-dependent uncertainty produce five
quantiles. Flat or sparse histories receive a wider minimum spread.

Run locally in PowerShell:

```powershell
$env:MAKEFUN_INPUT_DIR = 'path/to/input'
$env:MAKEFUN_OUTPUT_DIR = 'path/to/output'
python -P -s main.py
```

With neither variable set, the program uses `/input` and `/output`. Input files
are processed in sorted order. It needs no network, packages, or random state.

Run validity, determinism, timing, edge checks, and the synthetic comparison:

```powershell
python -P -s tests/check_entry.py --kit 'path/to/competition-kit'
```

Use `--tune` to reproduce the small parameter sweep on separate selection
seeds. Test outputs go into ignored `.checks/` directories. The test script
is not loaded by `main.py`.

On held-out synthetic cases the entry's mean logit pinball loss was 0.540304,
versus 0.773255 for persistence and 0.646162 for the prose curve baseline.
These simulations do not establish final-case performance. See
[BUILD_REPORT.md](BUILD_REPORT.md) for formulas, constants, measured practice
checks, and limitations. Local execution used Python 3.14.6; Python 3.13.14 execution was
verified separately with byte-identical outputs (the grader's python:3.13-slim
container itself remains unverified). MIT licensed; copyright TOTE.
