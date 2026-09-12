#!/usr/bin/env python3
"""
hike_forecast.py
================

Forecast temperature, wind and fog along a GPX track for a hike that starts
at a given time, using the free Open-Meteo API.

The track is walked with a Naismith/Langmuir time model (or a fixed total
duration), sampled every few hundred metres, and every sample point is
queried at its own coordinates and elevation. Hourly model values are
interpolated to the estimated time of arrival at each point.

Usage
-----
    python hike_forecast.py routes/quellenweg.gpx --start 2026-09-12T10:00
    python hike_forecast.py routes/quellenweg.gpx --start 10:00          # today
    python hike_forecast.py routes/quellenweg.gpx --start 10:00 --duration 100
    python hike_forecast.py routes/quellenweg.gpx --start 10:00 --reverse
    python hike_forecast.py routes/quellenweg.gpx --start 10:00 --json

Only the Python standard library is required. Reuses the fog heuristics from
alpine_inversion.py, which must sit next to this script.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import date, datetime, timedelta
from typing import Any

from alpine_inversion import API_URL, TIMEZONE, fog_score, lcl_height_above_ground

WEATHER_CODES = {
    0: "clear sky", 1: "mainly clear", 2: "partly cloudy", 3: "overcast",
    45: "fog", 48: "rime fog", 51: "light drizzle", 53: "drizzle", 55: "dense drizzle",
    56: "freezing drizzle", 57: "freezing drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    66: "freezing rain", 67: "freezing rain", 71: "light snow", 73: "snow", 75: "heavy snow",
    77: "snow grains", 80: "rain showers", 81: "rain showers", 82: "violent showers",
    85: "snow showers", 86: "snow showers", 95: "thunderstorm", 96: "thunderstorm, hail", 99: "thunderstorm, hail",
}
COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


# --------------------------------------------------------------------------- #
# GPX
# --------------------------------------------------------------------------- #

def read_gpx(path: str) -> tuple[str, list[dict[str, float]]]:
    ns = {"g": "http://www.topografix.com/GPX/1/1"}
    root = ET.parse(path).getroot()
    pts = root.findall(".//g:trkpt", ns) or root.findall(".//g:rtept", ns) or root.findall(".//trkpt")
    name_el = root.find(".//g:trk/g:name", ns) or root.find(".//g:metadata/g:name", ns)
    name = name_el.text.strip() if name_el is not None and name_el.text else path
    track = []
    for p in pts:
        ele = p.find("g:ele", ns)
        if ele is None:
            ele = p.find("ele")
        track.append({"lat": float(p.get("lat")), "lon": float(p.get("lon")),
                      "ele": float(ele.text) if ele is not None and ele.text else float("nan")})
    if len(track) < 2:
        raise SystemExit(f"No track points found in {path}")
    return name, track


def haversine(a: dict[str, float], b: dict[str, float]) -> float:
    r = 6371000.0
    la1, lo1, la2, lo2 = map(math.radians, (a["lat"], a["lon"], b["lat"], b["lon"]))
    h = math.sin((la2 - la1) / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin((lo2 - lo1) / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def leg_minutes(dist_m: float, dz: float, flat_kmh: float) -> float:
    """Naismith's rule with Langmuir's descent corrections."""
    t = dist_m / (flat_kmh * 1000.0) * 60.0
    if dz > 0:
        t += dz / 600.0 * 60.0                      # +1 h per 600 m ascent
    elif dist_m > 0:
        grade = -dz / dist_m
        if 0.087 < grade <= 0.213:                  # 5 to 12 degrees: faster
            t -= (-dz) / 300.0 * 10.0
        elif grade > 0.213:                          # steeper than 12 degrees: slower
            t += (-dz) / 300.0 * 10.0
    return max(t, 0.0)


def walk_track(track: list[dict[str, float]], flat_kmh: float,
               total_minutes: float | None) -> list[dict[str, float]]:
    """Attach cumulative distance, ascent, descent and minutes to every point."""
    out = [dict(track[0], dist=0.0, up=0.0, down=0.0, minutes=0.0)]
    for a, b in zip(track, track[1:]):
        d = haversine(a, b)
        dz = b["ele"] - a["ele"] if not (math.isnan(a["ele"]) or math.isnan(b["ele"])) else 0.0
        prev = out[-1]
        out.append(dict(b, dist=prev["dist"] + d, up=prev["up"] + max(dz, 0.0),
                        down=prev["down"] + max(-dz, 0.0),
                        minutes=prev["minutes"] + leg_minutes(d, dz, flat_kmh)))
    if total_minutes and out[-1]["minutes"] > 0:
        k = total_minutes / out[-1]["minutes"]
        for p in out:
            p["minutes"] *= k
    return out


def sample(points: list[dict[str, float]], step_m: float) -> list[dict[str, float]]:
    """Every `step_m` along the track plus the start, the end and the elevation extremes."""
    want = {0, len(points) - 1}
    want.add(max(range(len(points)), key=lambda i: points[i]["ele"]))
    want.add(min(range(len(points)), key=lambda i: points[i]["ele"]))
    nxt = step_m
    for i, p in enumerate(points):
        if p["dist"] >= nxt:
            want.add(i)
            nxt += step_m
    chosen: list[dict[str, float]] = []
    for i in sorted(want):
        if chosen and i != len(points) - 1 and points[i]["dist"] - chosen[-1]["dist"] < step_m * 0.3:
            continue                                   # too close to the previous sample
        chosen.append(points[i])
    return chosen


# --------------------------------------------------------------------------- #
# Weather
# --------------------------------------------------------------------------- #

HOURLY = [
    "temperature_2m", "apparent_temperature", "relative_humidity_2m", "dew_point_2m",
    "wind_speed_10m", "wind_gusts_10m", "wind_direction_10m", "cloud_cover", "cloud_cover_low",
    "visibility", "precipitation", "precipitation_probability", "weather_code",
]


def _fetch_many(samples, day, model):
    """Fetch hourly data for every sample point in one Open-Meteo request."""
    import urllib.request
    params = {
        "latitude": ",".join(f"{p['lat']:.5f}" for p in samples),
        "longitude": ",".join(f"{p['lon']:.5f}" for p in samples),
        "elevation": ",".join(f"{p['ele']:.0f}" for p in samples),
        "timezone": TIMEZONE, "start_date": day.isoformat(),
        "end_date": (day + timedelta(days=1)).isoformat(),
        "hourly": ",".join(HOURLY), "models": model,
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers={"User-Agent": "hike-forecast/1.0"})
    last = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except Exception as exc:  # noqa: BLE001
            last = exc
            import time; time.sleep(2 * (attempt + 1))
    else:
        raise SystemExit(f"Could not reach Open-Meteo: {last}")
    if isinstance(data, dict):
        if "error" in data:
            raise SystemExit(f"Open-Meteo error: {data.get('reason')}")
        data = [data]
    return data


def interp(hourly: dict[str, Any], key: str, when: datetime) -> float | None:
    times = hourly["time"]
    vals = hourly.get(key)
    if not vals:
        return None
    t0 = datetime.fromisoformat(times[0])
    pos = (when - t0).total_seconds() / 3600.0
    i = int(math.floor(pos))
    if i < 0 or i + 1 >= len(vals):
        i = max(0, min(len(vals) - 1, i))
        return vals[i]
    a, b = vals[i], vals[i + 1]
    if a is None or b is None:
        return a if b is None else b
    if key == "wind_direction_10m":            # circular interpolation
        diff = ((b - a + 180) % 360) - 180
        return (a + diff * (pos - i)) % 360
    if key == "weather_code":
        return a if pos - i < 0.5 else b
    return a + (b - a) * (pos - i)


def compass(deg: float | None) -> str:
    return "–" if deg is None else COMPASS[int((deg + 11.25) // 22.5) % 16]


def beaufort(kmh: float | None) -> str:
    if kmh is None:
        return "unknown"
    for limit, name in [(1, "calm"), (5, "light air"), (11, "light breeze"), (19, "gentle breeze"),
                        (28, "moderate breeze"), (38, "fresh breeze"), (49, "strong breeze"),
                        (61, "near gale"), (74, "gale"), (88, "strong gale"), (102, "storm")]:
        if kmh < limit:
            return name
    return "violent storm"


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def parse_start(text: str) -> datetime:
    if "T" in text:
        return datetime.fromisoformat(text)
    h, m = text.split(":")
    return datetime.combine(date.today(), datetime.min.time()).replace(hour=int(h), minute=int(m))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Forecast temperature, wind and fog along a GPX track.")
    ap.add_argument("gpx")
    ap.add_argument("--start", required=True, help="start time, HH:MM (today) or YYYY-MM-DDTHH:MM")
    ap.add_argument("--speed", type=float, default=4.0, help="flat walking speed in km/h (default 4)")
    ap.add_argument("--duration", type=float, help="force total duration in minutes")
    ap.add_argument("--step", type=float, default=700.0, help="sample spacing in metres (default 700)")
    ap.add_argument("--reverse", action="store_true", help="walk the track backwards")
    ap.add_argument("--model", default="best_match")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    name, track = read_gpx(args.gpx)
    if args.reverse:
        track = track[::-1]
    points = walk_track(track, args.speed, args.duration)
    samples = sample(points, args.step)
    start = parse_start(args.start)

    data = _fetch_many(samples, start.date(), args.model)
    if len(data) != len(samples):
        raise SystemExit(f"Expected {len(samples)} locations from Open-Meteo, got {len(data)}")

    rows = []
    for p, d in zip(samples, data):
        eta = start + timedelta(minutes=p["minutes"])
        h = d["hourly"]
        g = lambda k: interp(h, k, eta)  # noqa: E731
        t, td, rh = g("temperature_2m"), g("dew_point_2m"), g("relative_humidity_2m")
        vis, cl, wind, gust, wdir = g("visibility"), g("cloud_cover_low"), g("wind_speed_10m"), g("wind_gusts_10m"), g("wind_direction_10m")
        score, cat, reasons = fog_score(t, td, rh, vis, cl, wind)
        code = g("weather_code")
        rows.append({
            "dist_km": p["dist"] / 1000.0, "elevation_m": p["ele"], "lat": p["lat"], "lon": p["lon"],
            "model_elevation_m": d.get("elevation"), "minutes": p["minutes"], "eta": eta.strftime("%H:%M"),
            "temperature_c": t, "apparent_c": g("apparent_temperature"), "dew_point_c": td,
            "relative_humidity": rh, "wind_kmh": wind, "gust_kmh": gust, "wind_dir": wdir, "wind_from": compass(wdir),
            "beaufort": beaufort(wind), "cloud_cover_low": cl, "cloud_cover": g("cloud_cover"),
            "visibility_m": vis, "precip_mm": g("precipitation"), "precip_prob": g("precipitation_probability"),
            "weather": WEATHER_CODES.get(int(code) if code is not None else -1, "?"),
            "fog_score": score, "fog_category": cat, "fog_reasons": reasons,
            "cloud_base_msl_m": (p["ele"] + lcl_height_above_ground(t, td)) if t is not None and td is not None else None,
        })

    end = points[-1]
    summary = {
        "track": name, "start": start.isoformat(timespec="minutes"),
        "end": (start + timedelta(minutes=end["minutes"])).strftime("%H:%M"),
        "length_km": end["dist"] / 1000.0, "ascent_m": end["up"], "descent_m": end["down"],
        "duration_min": end["minutes"], "model": args.model, "points": rows,
    }

    if args.json:
        json.dump(summary, sys.stdout, indent=2, default=float)
        print()
        return 0

    f = lambda v, s=".1f": "  –" if v is None else format(v, s)  # noqa: E731
    print("=" * 96)
    print(f"{name}")
    print(f"start {start.strftime('%a %d %b %H:%M')}   finish ~{summary['end']}   "
          f"{summary['length_km']:.1f} km   +{end['up']:.0f} / -{end['down']:.0f} m   "
          f"{end['minutes']:.0f} min walking (flat {args.speed:g} km/h)   model {args.model}")
    print("=" * 96)
    print("  km   alt   ETA    T     feels  RH   wind  gust  from   low-cld  vis    weather          fog")
    for r in rows:
        vis = r["visibility_m"]
        print(f"{r['dist_km']:4.1f} {r['elevation_m']:5.0f}  {r['eta']}  {f(r['temperature_c']):>5} {f(r['apparent_c']):>6}  "
              f"{f(r['relative_humidity'], '.0f'):>3}  {f(r['wind_kmh'], '.0f'):>4}  {f(r['gust_kmh'], '.0f'):>4}  {r['wind_from']:<4}  "
              f"{f(r['cloud_cover_low'], '.0f'):>4} %  {f(vis / 1000 if vis is not None else None, '.0f'):>3} km  "
              f"{r['weather']:<16} {r['fog_category']} ({r['fog_score']})")
    temps = [r["temperature_c"] for r in rows if r["temperature_c"] is not None]
    feels = [r["apparent_c"] for r in rows if r["apparent_c"] is not None]
    winds = [r["wind_kmh"] for r in rows if r["wind_kmh"] is not None]
    gusts = [r["gust_kmh"] for r in rows if r["gust_kmh"] is not None]
    foggy = [r for r in rows if r["fog_score"] >= 3]
    wet = [r for r in rows if (r["precip_mm"] or 0) > 0.05]
    print()
    print(f"temperature {min(temps):.1f} to {max(temps):.1f} C (feels like {min(feels):.1f} to {max(feels):.1f} C)")
    hi = max(rows, key=lambda r: r["wind_kmh"] or 0)
    print(f"wind up to {max(winds):.0f} km/h, gusts to {max(gusts):.0f} km/h ({beaufort(max(winds))}), "
          f"strongest at km {hi['dist_km']:.1f} / {hi['elevation_m']:.0f} m from {hi['wind_from']}")
    if foggy:
        print("fog / mist possible at: " + ", ".join(f"km {r['dist_km']:.1f} ({r['elevation_m']:.0f} m, {r['eta']})" for r in foggy))
    else:
        print("fog: no point on the route reaches the fog threshold")
    if wet:
        print("precipitation at: " + ", ".join(f"km {r['dist_km']:.1f} {r['precip_mm']:.1f} mm" for r in wet))
    else:
        print("precipitation: none expected during the hike")
    print("\nModel values interpolated to the ETA at each point; 2 m values are downscaled to the track elevation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
