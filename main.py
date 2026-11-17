#!/usr/bin/env python3
"""GTA VI launch-day Twitch viewers forecaster (make.fun entry, v3).

Reads one case file (/input/<id>.in) and writes /output/<id>.out:
for each of hours 2..24 of the release day, five quantiles (5/25/50/75/95%)
of the concurrent viewers in the game's Twitch category, given everything
recorded up to the end of hour 1.

Method (standard library only, deterministic, no network):
  Past big launches trace a characteristic release-day shape in log space,
  and that shape depends strongly on how big hour 1 already is: a launch
  whose first hour is tiny grows many-fold as regions unlock through the
  day, one whose first hour is already huge stays flat or fades. The
  program predicts the log-ratio curve r_h = ln(viewers_h / viewers_1)
  as an ensemble of four members, all fit on the case's own comparables
  (past launches with complete release days):
    PMK1  platform-matched kernel mean: Gaussian-kernel weighted mean of
          the comparables' curves in log(hour-1 viewers), with a x8 weight
          bonus for comparables whose console-only status matches the
          target's. Console-only launches unlock rolling-midnight and
          trace a much tighter shape than mixed PC/console launches, so
          they also get a wider kernel (they are few, and a narrow kernel
          overfits their sparsity) and a narrower forecast spread.
    LL    kernel-weighted local-linear fit of r_h on log(hour-1 viewers).
    QV    ridge regression of the log LEVEL ln(viewers_h) on a quadratic
          in log(hour-1 viewers) (standardised inputs).
    QR    ridge regression of r_h on [log p1, log(p1/pre1), log channels]
          (standardised inputs), where pre1 is the category's viewers
          24 h before the release day: how much hour 1 already jumped
          over the pre-launch buzz.
  The four members are averaged in log space with weights
  0.55 / 0.25 / 0.10 / 0.10 (the kernel mean carries the most weight: it
  is the member that degrades least when the target sits outside the
  training range, which GTA VI's hour 1 almost certainly will).
  Quantiles are p1 * exp(mu_h + z_tau * sd_h), with z the standard
  normal quantiles and sd_h the honest leave-one-out residual spread of
  the training launches around their own kernel predictions
  (kernel-weighted to the target), scaled by 0.5 for console-only
  targets and 0.8 otherwise, and widened when the target's hour-1 size
  sits above every comparable's. With fewer than six usable comparables
  it falls back to persistence (the house baseline shape).

  The target's console-only status is read from the case notes
  ("released ... on PS5 only ...; real unlock: ..."), NOT from a
  same-day comparable entry: real cases carry no comparable whose
  day_start equals the target's, so reading it from the comparables
  always answered "not console-only" and handed the platform bonus to
  the PC launches. The notes parse is verified 37/37 on the kit's
  comparables and 4/4 on the practice cases.

Leave-one-out validation over the kit's 37 comparable launches with
complete release days (see validate_loo.py), scored through the same
notes-based pipeline the real cases use: loss 0.363 overall,
0.152 console-only, 0.154 on the four rolling-midnight console launches
closest to GTA VI, 0.141 on console-only comparable-tier launches --
against 0.370 / 0.172 / 0.161 / 0.168 for v2 on the same sets, and
0.930 / 0.323 / 0.301 for the house persistence baseline.
"""
import json
import math
import os
import sys
from datetime import datetime, timedelta

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)

# Tuned on leave-one-out over the kit's comparables (see validate_loo.py).
# Bandwidths and spread scales are split by platform: console-only
# launches are few and tightly shaped, PC/multi launches are many and wild.
BW_CONSOLE = 1.2     # kernel bandwidth (sd of log p1) for console targets
BW_PC = 0.8          # kernel bandwidth for PC/multi targets
PLATFORM_BONUS = 8.0  # weight bonus for platform-matched comparables
RIDGE_LAMBDA = 3.0    # ridge penalty on standardised inputs
WEIGHTS = (0.55, 0.25, 0.10, 0.10)  # PMK1, LL, QV, QR
GAMMA_CONSOLE = 0.5   # calibration multiplier on the residual spread
GAMMA_PC = 0.8        # (console shapes are tighter, so a narrower band)
SPREAD_FLOOR = 0.10   # minimum log-spread per hour
EXTRAP_BETA = 0.5     # extra spread per log unit above the training range
MIN_TRAIN = 6         # below this, fall back to persistence


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def solve_ridge(xs, ys, lam):
    """Least squares with ridge penalty, via the normal equations."""
    n = len(xs[0])
    a = [[sum(row[i] * row[j] for row in xs) + (lam if i == j else 0.0)
          for j in range(n)] for i in range(n)]
    b = [sum(row[i] * y for row, y in zip(xs, ys)) for i in range(n)]
    for i in range(n):
        piv = max(range(i, n), key=lambda r: abs(a[r][i]))
        a[i], a[piv] = a[piv], a[i]
        b[i], b[piv] = b[piv], b[i]
        if abs(a[i][i]) < 1e-12:
            continue
        for r in range(i + 1, n):
            f = a[r][i] / a[i][i]
            for c in range(i, n):
                a[r][c] -= f * a[i][c]
            b[r] -= f * b[i]
    sol = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = b[i] - sum(a[i][j] * sol[j] for j in range(i + 1, n))
        sol[i] = s / a[i][i] if abs(a[i][i]) > 1e-12 else 0.0
    return sol


def standardise(values):
    m = sum(values) / len(values)
    v = (sum((x - m) ** 2 for x in values) / len(values)) ** 0.5 or 1.0
    return m, v


def nearest_viewers(hourly, target, tol_hours=6):
    """Viewers of the point nearest to `target`, if within tolerance."""
    best = None
    for p in hourly:
        d = abs((parse_time(p["from"]) - target).total_seconds())
        if best is None or d < best[0]:
            best = (d, p)
    if best and best[0] <= tol_hours * 3600:
        return max(best[1]["viewers"], 1)
    return None


def parse_notes(notes):
    """Console-only status of the TARGET, from the case notes.

    Notes read like: '... released 2026-09-15 on PS5 only (...); real
    unlock: Rolling midnight local ...'. The platform clause sits between
    'released' and '; real unlock:'. A launch is console-only unless that
    clause names PC as a launch platform; 'console only' and 'no PC'
    (e.g. 'no PC until 2027') both override a bare PC mention, which can
    mean a later PC port rather than a launch platform.
    """
    notes = notes or ""
    seg = notes.split("released ", 1)[1] if "released " in notes else notes
    if "; real unlock:" in seg:
        plat = seg.split("; real unlock:")[0]
    else:
        plat = seg.split(".")[0]
    low = plat.lower()
    if "console only" in low or "no pc" in low:
        return True
    if "pc" in low:
        return False
    return True


def extract_launches(case):
    """Comparables with a complete 24-hour release day.

    Returns a list of dicts: name, p1, c1, pre1, vals (24 hourly viewer
    counts, hours 1..24), console_only, tier, plus the case's own
    target-history entry (if any comparable shares the target's
    day_start) for the target's features.
    """
    day_start = parse_time(case["day_start"])
    launches = []
    target_history = None
    for comp in sorted(case.get("comparables", []), key=lambda c: c["name"]):
        hourly = comp.get("hourly", [])
        comp_day = parse_time(comp["day_start"])
        day = [p for p in hourly
               if comp_day <= parse_time(p["from"]) < comp_day + timedelta(hours=24)]
        if comp_day == day_start:
            target_history = comp
        if len(day) < 24:
            continue
        vals = [max(p["viewers"], 1) for p in day]
        chans = [p.get("channels") or 0 for p in day]
        pre1 = nearest_viewers(hourly, comp_day - timedelta(hours=24))
        tier = ""
        unlock = comp.get("unlock", "")
        if "Tier:" in unlock:
            tier = unlock.split("Tier:")[1].split(".")[0].strip()
        launches.append({
            "name": comp["name"],
            "p1": vals[0],
            "c1": max(chans[0], 1),
            "pre1": pre1,
            "vals": vals,
            "tier": tier,
            "console_only": "PC" not in comp.get("platforms", ""),
        })
    return launches, target_history


def target_features(case, launches, target_history):
    """Feature dict for the target launch: log size, surge, channels."""
    known = case["known"][0]
    p1 = max(known["viewers"], 1)
    c1 = max(known.get("channels") or 1, 1)
    day_start = parse_time(case["day_start"])
    pre1 = None
    for pts in (case.get("history_hourly") or [],
                (target_history or {}).get("hourly", [])):
        pre1 = nearest_viewers(pts, day_start - timedelta(hours=24))
        if pre1:
            break
    if not pre1:
        have = sorted(l["pre1"] for l in launches if l["pre1"])
        pre1 = have[len(have) // 2] if have else p1
    # Notes first: real cases carry no same-day comparable, so the
    # comparables route would always answer "not console-only".
    if case.get("notes"):
        console_only = parse_notes(case["notes"])
    elif target_history:
        console_only = "PC" not in target_history.get("platforms", "")
    else:
        console_only = False
    return {"s": math.log(p1), "q": math.log(p1 / max(pre1, 1)),
            "k": math.log(c1), "p1": p1, "c1": c1,
            "console_only": console_only}


def match_weight(comp, console_only):
    """Platform bonus: x8 when the comparable's platform class matches."""
    return PLATFORM_BONUS if comp["console_only"] == console_only else 1.0


def kernel_weights(train, s_target, bw):
    """Gaussian kernel weights in log(hour-1 viewers) (no platform bonus)."""
    s_vals = [math.log(t["p1"]) for t in train]
    _m, sd = standardise(s_vals)
    out = []
    for t in train:
        d = abs(math.log(t["p1"]) - s_target) / sd
        out.append(math.exp(-0.5 * (d / bw) ** 2))
    return out, sd


def member_curves(train, feats, bw):
    """The four members' predicted log-ratio curves, hours 2..24."""
    s0 = feats["s"]
    w, _sd = kernel_weights(train, s0, bw)
    wpm = [wi * match_weight(t, feats["console_only"]) for t, wi in zip(train, w)]
    sw = sum(w)
    swpm = sum(wpm)
    # standardised features for the ridge members
    keys = ("s", "q", "k")
    have_pre = sorted(t["pre1"] for t in train if t["pre1"])
    med_pre = have_pre[len(have_pre) // 2] if have_pre else 1
    fvals = {}
    for key in keys:
        raw = [math.log(t["p1"]) if key == "s" else
               math.log(t["p1"] / max(t["pre1"] or med_pre, 1)) if key == "q" else
               math.log(t["c1"]) for t in train]
        fvals[key] = (raw, standardise(raw))
    tz = {key: (feats[key] - fvals[key][1][0]) / fvals[key][1][1] for key in keys}
    curves = {"pmk1": {}, "ll": {}, "qv": {}, "qr": {}}
    for h in range(2, 25):
        r = [math.log(t["vals"][h - 1] / t["vals"][0]) for t in train]
        lv = [math.log(t["vals"][h - 1]) for t in train]
        curves["pmk1"][h] = sum(wi * ri for wi, ri in zip(wpm, r)) / swpm
        # local-linear in log p1, centred on the target
        xs = [math.log(t["p1"]) - s0 for t in train]
        wxx = sum(wi * x * x for wi, x in zip(w, xs))
        wx = sum(wi * x for wi, x in zip(w, xs))
        wy = sum(wi * ri for wi, ri in zip(w, r))
        wxy = sum(wi * x * ri for wi, x, ri in zip(w, xs, r))
        den = sw * wxx - wx * wx
        if abs(den) > 1e-12:
            slope = (sw * wxy - wx * wy) / den
            curves["ll"][h] = (wy - slope * wx) / sw
        else:
            curves["ll"][h] = curves["pmk1"][h]
        # quadratic ridge on the log level
        xq = [[1.0,
               (math.log(t["p1"]) - fvals["s"][1][0]) / fvals["s"][1][1],
               ((math.log(t["p1"]) - fvals["s"][1][0]) / fvals["s"][1][1]) ** 2]
              for t in train]
        bq = solve_ridge(xq, lv, RIDGE_LAMBDA)
        curves["qv"][h] = (bq[0] + bq[1] * tz["s"] + bq[2] * tz["s"] ** 2) - s0
        # ridge on [s, q, k]
        xr = [[1.0,
               (math.log(t["p1"]) - fvals["s"][1][0]) / fvals["s"][1][1],
               (math.log(t["p1"] / max(t["pre1"] or med_pre, 1)) - fvals["q"][1][0]) / fvals["q"][1][1],
               (math.log(t["c1"]) - fvals["k"][1][0]) / fvals["k"][1][1]]
              for t in train]
        br = solve_ridge(xr, r, RIDGE_LAMBDA)
        curves["qr"][h] = (br[0] + br[1] * tz["s"] + br[2] * tz["q"]
                           + br[3] * tz["k"])
    return curves


def spread(train, feats, bw, gamma):
    """Per-hour log-spread from honest leave-one-out kernel residuals."""
    s0 = feats["s"]
    _w, sd_s = kernel_weights(train, s0, bw)
    w_tgt = []
    for t in train:
        d = abs(math.log(t["p1"]) - s0) / sd_s
        w_tgt.append(math.exp(-0.5 * (d / bw) ** 2) * match_weight(t, feats["console_only"]))
    out = {}
    for h in range(2, 25):
        num = den = 0.0
        for j, t in enumerate(train):
            # kernel prediction for launch j from the other launches
            wj = []
            for jj, tt in enumerate(train):
                if jj == j:
                    wj.append(0.0)
                else:
                    d = abs(math.log(tt["p1"]) - math.log(t["p1"])) / sd_s
                    wj.append(math.exp(-0.5 * (d / bw) ** 2) *
                              match_weight(tt, t["console_only"]))
            swj = sum(wj)
            r = [math.log(tt["vals"][h - 1] / tt["vals"][0]) for tt in train]
            pred = sum(a * b for a, b in zip(wj, r)) / swj if swj > 0 else sum(r) / len(r)
            e = math.log(t["vals"][h - 1] / t["vals"][0]) - pred
            num += w_tgt[j] * e * e
            den += w_tgt[j]
        var = num / den if den > 0 else 0.25
        over = max(0.0, s0 - max(math.log(t["p1"]) for t in train))
        out[h] = max(SPREAD_FLOOR, math.sqrt(var) * gamma * (1.0 + EXTRAP_BETA * over))
    return out


def persistence_forecast(p1):
    hours = []
    for h in range(2, 25):
        q = [max(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 1.0) for z in Z]
        hours.append({"hour": h, "q": sorted(q)})
    return {"hours": hours}


def forecast(case):
    launches, target_history = extract_launches(case)
    feats = target_features(case, launches, target_history)
    p1 = feats["p1"]
    if len(launches) < MIN_TRAIN:
        return persistence_forecast(p1)
    bw = BW_CONSOLE if feats["console_only"] else BW_PC
    gamma = GAMMA_CONSOLE if feats["console_only"] else GAMMA_PC
    curves = member_curves(launches, feats, bw)
    sds = spread(launches, feats, bw, gamma)
    hours = []
    for h in range(2, 25):
        mu = (WEIGHTS[0] * curves["pmk1"][h] + WEIGHTS[1] * curves["ll"][h] +
              WEIGHTS[2] * curves["qv"][h] + WEIGHTS[3] * curves["qr"][h])
        q = [max(p1 * math.exp(mu + z * sds[h]), 1.0) for z in Z]
        q = sorted(q)
        q = [min(v, 1e9) for v in q]
        hours.append({"hour": h, "q": q})
    return {"hours": hours}


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    for name in sorted(os.listdir(in_dir)):
        if not name.endswith(".in"):
            continue
        with open(os.path.join(in_dir, name), encoding="utf-8") as f:
            case = json.load(f)
        result = forecast(case)
        out_path = os.path.join(out_dir, name[:-3] + ".out")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, separators=(",", ":"))
            f.write("\n")


if __name__ == "__main__":
    main()
