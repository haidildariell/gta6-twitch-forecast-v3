# GTA VI launch-day Twitch viewers forecaster v3

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/haidildariell/gta6-twitch-forecast-v3/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/haidildariell/gta6-twitch-forecast-v3/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Entry for [make.fun's "Forecast GTA VI's launch-day Twitch viewers hour by hour" bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01): a program that reads one case and forecasts, for each of hours 2 to 24 of GTA VI's release day (2026-11-19 UTC), five quantiles of the concurrent viewers in Twitch's "Grand Theft Auto VI" category, given everything recorded up to the end of hour 1.

## How it works

Past big launches trace a characteristic release-day shape in log space, and that shape depends strongly on how big hour 1 already is — a launch whose first hour is tiny grows many-fold as regions unlock through the day, one whose first hour is already huge stays flat or fades. The program predicts the log-ratio curve `r_h = ln(viewers_h / viewers_1)` as an ensemble of four members, all fit on the case's own comparables (past launches with complete release days):

- **PMK1** — platform-matched kernel mean: a Gaussian-kernel weighted mean of the comparables' curves in log(hour-1 viewers), with a x8 weight bonus for comparables whose console-only status matches the target's. Console-only launches unlock rolling-midnight and trace a much tighter shape than mixed PC/console launches.
- **LL** — kernel-weighted local-linear fit of `r_h` on log(hour-1 viewers).
- **QV** — ridge regression of the log level `ln(viewers_h)` on a quadratic in log(hour-1 viewers).
- **QR** — ridge regression of `r_h` on log p1, log(p1/pre1) and log channels, where pre1 is the category's viewers 24 h before the release day: how much hour 1 already jumped over the pre-launch buzz.

The members are averaged in log space (weights 0.55 / 0.25 / 0.10 / 0.10). Quantiles are `p1 * exp(mu_h + z_tau * sd_h)`, with the spread from the honest leave-one-out residual dispersion of the training launches around their own kernel predictions, widened when the target's hour-1 size sits above every comparable's. With fewer than six usable comparables it falls back to persistence. Standard library only, deterministic, well under a second per case.

## What changed vs v2

- **Notes-based platform detection (the big one).** v2 read the target's console-only status from a comparable sharing the target's `day_start`. Real cases carry no such comparable, so on the actual practice cases (and the GTA VI race case) v2 always answered "not console-only" and handed its platform bonus to the PC launches. v3 reads it from the case notes instead ("released ... on PS5 only ...; real unlock: ..."). Measured on leave-one-out with the bug simulated, v2's real-case behaviour scores 0.203 console-only / 0.202 core7; v3 scores 0.152 / 0.141.
- **Platform-split tuning.** Console-only targets get a wider kernel (1.2 vs 0.8 sd — they are few, a narrow kernel overfits their sparsity), a x8 platform bonus (was x5), and a narrower spread scale (0.5 vs 0.8 — console shapes are tighter). PC/multi targets keep the v2 bandwidth with the wider spread.
- **Kernel-heavy weights (0.55 / 0.25 / 0.10 / 0.10).** The kernel mean is the member that degrades least when the target sits outside the training range, which GTA VI's hour 1 almost certainly will; the extrapolation spread widening was also raised (0.5 per log unit above the training max).

Ideas tried and dropped after LOO said no: tier filtering (comparable-only training), empirical-quantile spread, a 2D kernel member (log p1 × surge), a global-linear member, a rolling-midnight bonus, excluding single-global-unlock comparables, and a delayed-unlock branch for single-global-unlock targets (helps Wukong, wrecks Silksong — the jump size is unpredictable from hour 1).

Leave-one-out validation over the kit's 37 comparable launches (`python3 validate_loo.py`, scored through the same notes-based pipeline the real cases use): loss **0.363** overall, **0.152** console-only, **0.154** on the four rolling-midnight console launches closest to GTA VI, **0.141** console-only comparable-tier — against 0.370 / 0.172 / 0.161 / 0.168 for v2 on the same sets, and 0.930 / 0.323 / 0.301 for the house persistence baseline.

## Run

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Reads every `*.in` case in the input directory and writes one `<id>.out` forecast per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, ...]}` for hours 2–24.

## License

MIT — see [LICENSE](LICENSE).
