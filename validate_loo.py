#!/usr/bin/env python3
"""Leave-one-out validation for main.py, over the kit's comparables.

For each comparable launch with a complete 24-hour release day, build a
synthetic case where that launch is the target (hour-1 print known, the
other launches as comparables) and score main.forecast() against the
target's actual hours 2..24 with the contest's pinball loss on the log
scale.

The synthetic cases deliberately mirror the REAL cases: they carry NO
same-day comparable entry for the target (real cases have none), and the
target's platform/unlock facts arrive only through the notes field, in
the same wording the kit uses ("released ... on ...; real unlock: ...").
That is the point of this validator: v2's LOO fed the target's platform
through a same-day comparable entry, which real cases never have, so its
0.172 console-only number was measured on a pipeline the real cases do
not run.

Usage: python3 validate_loo.py [path/to/p00.in]
"""
import json
import math
import sys
from datetime import timedelta

sys.path.insert(0, __file__.rsplit("/", 1)[0])
from main import forecast, parse_time, Z, LEVELS  # noqa: E402

CORE4 = {"Ghost of Yōtei", "The Legend of Zelda: Tears of the Kingdom",
         "FINAL FANTASY VII REBIRTH", "The Last of Us Part II"}


def pinball(y, qs):
    ly = math.log(max(y, 1))
    tot = 0.0
    for tau, q in zip(LEVELS, qs):
        d = ly - math.log(max(q, 1e-9))
        tot += 2 * max(tau * d, (tau - 1) * d)
    return tot / 5


def baseline_pinball(p1, y, h):
    q = [max(p1 * math.exp(z * 0.15 * math.sqrt(h - 1)), 1.0) for z in Z]
    return pinball(y, sorted(q))


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else (
        "/home/hatch/workspace/makefun/gta6fc-kit-fresh/"
        "gta6-twitch-ccu-forecast-01-kit/input/p00.in")
    case0 = json.load(open(path))
    comps = case0["comparables"]

    def day_vals(comp):
        ds = parse_time(comp["day_start"])
        day = [p for p in comp["hourly"]
               if ds <= parse_time(p["from"]) < ds + timedelta(hours=24)]
        return ds, day

    targets = []
    for comp in comps:
        ds, day = day_vals(comp)
        if len(day) < 24:
            continue
        targets.append((comp, ds, day))

    per, per_base = {}, {}
    for comp, ds, day in targets:
        vals = [max(p["viewers"], 1) for p in day]
        chans = [p.get("channels") or 0 for p in day]
        pre_hourly = [p for p in comp["hourly"] if parse_time(p["from"]) < ds]
        others = [c for c in comps if c is not comp]
        synth = {
            "id": "loo",
            "game": {"name": comp["name"], "twitch_id": "0"},
            "day_start": comp["day_start"],
            "checkpoint": (ds + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
            "levels": list(LEVELS),
            "known": [{"hour": 1, "from": comp["day_start"],
                       "until": (ds + timedelta(hours=1)).isoformat().replace("+00:00", "Z"),
                       "viewers": vals[0], "channels": chans[0], "reads": 1}],
            "forecast": [{"hour": h} for h in range(2, 25)],
            "history": [], "history_hourly": pre_hourly,
            "streams": None, "categories_hourly": None,
            "comparables": others,
            # Same wording the real cases use; no same-day entry, so the
            # platform facts can only come from here.
            "notes": (f"{comp['name']}, released {comp['day_start'][:10]} on "
                      f"{comp.get('platforms', '')}; real unlock: "
                      f"{comp.get('unlock', '')}"),
        }
        out = forecast(synth)
        qmap = {row["hour"]: row["q"] for row in out["hours"]}
        losses, losses_base = [], []
        for h in range(2, 25):
            losses.append(pinball(vals[h - 1], qmap[h]))
            losses_base.append(baseline_pinball(vals[0], vals[h - 1], h))
        per[comp["name"]] = sum(losses) / len(losses)
        per_base[comp["name"]] = sum(losses_base) / len(losses_base)

    def split(perdict, pred):
        xs = [perdict[c["name"]] for c, _ds, _d in targets if pred(c)]
        return sum(xs) / len(xs), len(xs)

    def is_console(c):
        return "PC" not in c.get("platforms", "")

    def tier(c):
        u = c.get("unlock", "")
        return u.split("Tier:")[1].split(".")[0].strip() if "Tier:" in u else ""

    print(f"launches: {len(targets)}")
    for label, pred in (("all", lambda c: True),
                        ("console-only", is_console),
                        ("core4 (rolling-midnight console)", lambda c: c["name"] in CORE4),
                        ("core7 (console-only + comparable)",
                         lambda c: is_console(c) and tier(c) == "comparable")):
        m, n = split(per, pred)
        mb, _ = split(per_base, pred)
        print(f"  {label}: model={m:.3f} (n={n})  baseline={mb:.3f}")
    print("per launch (model / baseline):")
    for c, _ds, _d in sorted(targets, key=lambda t: -per[t[0]["name"]]):
        print(f"  {c['name'][:30]:30} {per[c['name']]:.3f} / {per_base[c['name']]:.3f}")


if __name__ == "__main__":
    main()
