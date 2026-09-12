#!/usr/bin/env python3
"""
alpine_inversion.py
===================

Model the current temperature inversion and fog situation between two
mountain sites using the free Open-Meteo forecast API (no API key needed).

Default sites (Serfaus, Tyrol, Austria):

  * Serfaus Dorf              47.0403 N  10.6031 E   ~1427 m  (valley terrace)
  * Murmliwasser (Komperdell) 47.0399 N  10.5628 E   ~1980 m  (mid-mountain)

What the model does
-------------------
1. Pulls 2 m temperature, dew point, humidity, visibility, low cloud and wind
   for both sites, plus the free-air profile of the model column
   (950 ... 700 hPa: temperature, humidity, geopotential height).
2. Site-to-site lapse rate  dT/dz  between the lower and the upper site.
   Standard atmosphere is about -0.65 C / 100 m; a *positive* value means the
   upper site is warmer than the valley -> temperature inversion.
3. Potential-temperature gradient dTheta/dz as a stability measure.
4. Free-air inversion layers (levels where temperature rises with height).
5. Per-site fog diagnostics: dew-point spread, RH, visibility, low cloud, wind
   -> a heuristic fog score and category.
6. Lifting condensation level (cloud base) above the lower site and the top
   of the moist layer from the RH profile -> tells whether the upper site is
   above a "Nebelmeer" (sea of fog) sitting in the valley.
7. A time line for the next N hours showing when the inversion forms or breaks
   and how the fog risk evolves at both sites.

Usage
-----
    python alpine_inversion.py                 # text report, next 24 h
    python alpine_inversion.py --hours 36      # longer time line
    python alpine_inversion.py --model icon_d2 # force the 2 km ICON-D2 model
    python alpine_inversion.py --json          # machine readable output
    python alpine_inversion.py \
        --lower "Serfaus Dorf:47.0403:10.6031:1427" \
        --upper "Murmliwasser:47.0399:10.5628:1980"

Only the Python standard library is required.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any

API_URL = "https://api.open-meteo.com/v1/forecast"
TIMEZONE = "Europe/Vienna"
PRESSURE_LEVELS = [950, 925, 900, 850, 800, 700]

STANDARD_LAPSE = -0.65        # C per 100 m, ICAO standard atmosphere
DRY_ADIABATIC_LAPSE = -0.98   # C per 100 m
KAPPA = 0.2857                # R/cp for dry air


@dataclass
class Site:
    name: str
    lat: float
    lon: float
    elevation: float  # metres above sea level

    @classmethod
    def parse(cls, text: str) -> "Site":
        try:
            name, lat, lon, elev = text.split(":")
            return cls(name, float(lat), float(lon), float(elev))
        except ValueError as exc:  # pragma: no cover - CLI validation
            raise argparse.ArgumentTypeError(
                f"expected NAME:LAT:LON:ELEVATION, got {text!r}"
            ) from exc


DEFAULT_LOWER = Site("Serfaus Dorf", 47.0403, 10.6031, 1427)
DEFAULT_UPPER = Site("Murmliwasser (Komperdell)", 47.0399, 10.5628, 1980)


# --------------------------------------------------------------------------- #
# Data access
# --------------------------------------------------------------------------- #

def build_query(lower: Site, upper: Site, hours: int, model: str) -> str:
    hourly = [
        "temperature_2m", "dew_point_2m", "relative_humidity_2m",
        "visibility", "cloud_cover_low", "cloud_cover", "wind_speed_10m",
        "wind_gusts_10m", "wind_direction_10m", "surface_pressure", "weather_code",
    ]
    for p in PRESSURE_LEVELS:
        hourly += [
            f"temperature_{p}hPa",
            f"relative_humidity_{p}hPa",
            f"geopotential_height_{p}hPa",
        ]
    current = [
        "temperature_2m", "dew_point_2m", "relative_humidity_2m",
        "cloud_cover", "cloud_cover_low", "wind_speed_10m",
        "wind_direction_10m", "surface_pressure", "weather_code", "is_day",
    ]
    params = {
        "latitude": f"{lower.lat},{upper.lat}",
        "longitude": f"{lower.lon},{upper.lon}",
        "elevation": f"{lower.elevation},{upper.elevation}",
        "timezone": TIMEZONE,
        "past_hours": 6,
        "forecast_hours": hours,
        "current": ",".join(current),
        "hourly": ",".join(hourly),
        "daily": "sunrise,sunset",
        "models": model,
    }
    return API_URL + "?" + urllib.parse.urlencode(params)


def fetch(url: str, timeout: float = 30.0, retries: int = 3) -> list[dict[str, Any]]:
    req = urllib.request.Request(url, headers={"User-Agent": "alpine-inversion/1.0"})
    data = None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            break
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")
            raise SystemExit(f"Open-Meteo returned HTTP {exc.code}: {body[:300]}") from exc
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if attempt == retries:
                raise SystemExit(f"Could not reach Open-Meteo after {retries} attempts: {exc}") from exc
            print(f"warning: attempt {attempt} failed ({exc}), retrying...", file=sys.stderr)
            time.sleep(2 * attempt)
    if isinstance(data, dict):
        if "error" in data:
            raise SystemExit(f"Open-Meteo error: {data.get('reason')}")
        data = [data]
    if len(data) != 2:
        raise SystemExit(f"Expected two locations in the response, got {len(data)}")
    return data


# --------------------------------------------------------------------------- #
# Physics helpers
# --------------------------------------------------------------------------- #

def potential_temperature(t_c: float, p_hpa: float) -> float:
    """Potential temperature in Kelvin referenced to 1000 hPa."""
    return (t_c + 273.15) * (1000.0 / p_hpa) ** KAPPA


def lapse_per_100m(t_low: float, z_low: float, t_up: float, z_up: float) -> float:
    return (t_up - t_low) / (z_up - z_low) * 100.0


def lcl_height_above_ground(t_c: float, td_c: float) -> float:
    """Espy approximation of the lifting condensation level in metres."""
    return max(0.0, 125.0 * (t_c - td_c))


def classify_lapse(lapse: float | None) -> str:
    if lapse is None:
        return "unknown"
    if lapse > 0.3:
        return "strong inversion"
    if lapse > 0.0:
        return "inversion"
    if lapse > -0.3:
        return "isothermal / very stable"
    if lapse > -0.6:
        return "stable"
    if lapse > -0.9:
        return "near standard"
    return "unstable (near dry-adiabatic)"


def classify_theta_gradient(grad: float | None) -> str:
    if grad is None:
        return "unknown"
    if grad > 0.5:
        return "strongly stable"
    if grad > 0.1:
        return "stable"
    if grad > -0.1:
        return "neutral"
    return "unstable"


def fog_score(t: float | None, td: float | None, rh: float | None,
              vis: float | None, cloud_low: float | None,
              wind: float | None) -> tuple[int, str, list[str]]:
    """
    Heuristic fog score 0..10 with a category and the reasons that fired.
    Radiation / valley fog needs saturation, little wind and reduced visibility.
    """
    score = 0
    reasons: list[str] = []
    if t is not None and td is not None:
        spread = t - td
        if spread <= 0.5:
            score += 3; reasons.append(f"dew-point spread {spread:.1f} C (saturated)")
        elif spread <= 1.5:
            score += 2; reasons.append(f"dew-point spread {spread:.1f} C (near saturation)")
        elif spread <= 3.0:
            score += 1; reasons.append(f"dew-point spread {spread:.1f} C")
    if rh is not None:
        if rh >= 98:
            score += 3; reasons.append(f"RH {rh:.0f} %")
        elif rh >= 95:
            score += 2; reasons.append(f"RH {rh:.0f} %")
        elif rh >= 90:
            score += 1; reasons.append(f"RH {rh:.0f} %")
    if vis is not None:
        if vis < 1000:
            score += 3; reasons.append(f"visibility {vis:.0f} m (fog)")
        elif vis < 4000:
            score += 2; reasons.append(f"visibility {vis/1000:.1f} km (mist)")
        elif vis < 10000:
            score += 1; reasons.append(f"visibility {vis/1000:.1f} km (haze)")
    if cloud_low is not None and cloud_low >= 80:
        score += 1; reasons.append(f"low cloud {cloud_low:.0f} %")
    if wind is not None and wind > 12:
        score -= 1; reasons.append(f"wind {wind:.0f} km/h disperses fog")
    score = max(0, min(10, score))
    if score >= 6:
        category = "fog likely"
    elif score >= 3:
        category = "mist / patchy fog possible"
    else:
        category = "no fog"
    return score, category, reasons


# --------------------------------------------------------------------------- #
# Profile analysis
# --------------------------------------------------------------------------- #

def free_air_profile(hourly: dict[str, Any], i: int, min_height: float) -> list[dict[str, float]]:
    """Free-air levels of the model column at hour index i, above min_height."""
    levels = []
    for p in PRESSURE_LEVELS:
        z = hourly.get(f"geopotential_height_{p}hPa", [None])[i]
        t = hourly.get(f"temperature_{p}hPa", [None])[i]
        rh = hourly.get(f"relative_humidity_{p}hPa", [None])[i]
        if z is None or t is None:
            continue
        if z < min_height - 150:       # level is inside the terrain -> extrapolated
            continue
        levels.append({
            "pressure_hpa": p, "height_m": z, "temperature_c": t,
            "relative_humidity": rh, "theta_k": potential_temperature(t, p),
        })
    levels.sort(key=lambda d: d["height_m"])
    return levels


def find_inversion_layers(levels: list[dict[str, float]]) -> list[dict[str, float]]:
    """Layers between consecutive levels where temperature increases with height."""
    layers = []
    for a, b in zip(levels, levels[1:]):
        dz = b["height_m"] - a["height_m"]
        if dz <= 0:
            continue
        lapse = (b["temperature_c"] - a["temperature_c"]) / dz * 100.0
        if lapse > 0:
            layers.append({
                "base_m": a["height_m"], "top_m": b["height_m"],
                "delta_t_c": b["temperature_c"] - a["temperature_c"],
                "lapse_c_per_100m": lapse,
            })
    return layers


def moist_layer(levels: list[dict[str, float]], surface_z: float,
                surface_rh: float | None, threshold: float = 85.0) -> dict[str, float] | None:
    """
    Lowest layer of the column where RH >= `threshold` (stratus / fog layer).
    Returns base and top in metres MSL, interpolated linearly between the
    available points, or None if no point is moist enough. A layer that starts
    at the surface is ground fog; one that starts higher is a cloud deck.
    """
    points = []
    if surface_rh is not None:
        points.append((surface_z, surface_rh))
    points += [(l["height_m"], l["relative_humidity"]) for l in levels
               if l["relative_humidity"] is not None]
    points.sort()
    first = next((k for k, (_, rh) in enumerate(points) if rh >= threshold), None)
    if first is None:
        return None

    def crossing(z0: float, rh0: float, z1: float, rh1: float) -> float:
        return z0 + (rh0 - threshold) / (rh0 - rh1) * (z1 - z0)

    if first == 0:
        base = points[0][0]
    else:
        (z0, rh0), (z1, rh1) = points[first - 1], points[first]
        base = crossing(z1, rh1, z0, rh0)
    top = None
    for (z0, rh0), (z1, rh1) in zip(points[first:], points[first + 1:]):
        if rh1 < threshold:
            top = crossing(z0, rh0, z1, rh1)
            break
    return {"base_msl_m": base, "top_msl_m": top, "touches_ground": first == 0}


# --------------------------------------------------------------------------- #
# Assemble the picture for one hour
# --------------------------------------------------------------------------- #

def _val(hourly: dict[str, Any], key: str, i: int) -> float | None:
    seq = hourly.get(key)
    if not seq or i >= len(seq):
        return None
    return seq[i]


def site_state(site: Site, hourly: dict[str, Any], i: int) -> dict[str, Any]:
    t = _val(hourly, "temperature_2m", i)
    td = _val(hourly, "dew_point_2m", i)
    rh = _val(hourly, "relative_humidity_2m", i)
    vis = _val(hourly, "visibility", i)
    cl = _val(hourly, "cloud_cover_low", i)
    wind = _val(hourly, "wind_speed_10m", i)
    gust = _val(hourly, "wind_gusts_10m", i)
    wdir = _val(hourly, "wind_direction_10m", i)
    psfc = _val(hourly, "surface_pressure", i)
    score, category, reasons = fog_score(t, td, rh, vis, cl, wind)
    theta = potential_temperature(t, psfc) if (t is not None and psfc) else None
    lcl = lcl_height_above_ground(t, td) if (t is not None and td is not None) else None
    return {
        "site": site.name,
        "elevation_m": site.elevation,
        "temperature_c": t,
        "dew_point_c": td,
        "spread_c": (t - td) if (t is not None and td is not None) else None,
        "relative_humidity": rh,
        "visibility_m": vis,
        "cloud_cover_low": cl,
        "wind_speed_kmh": wind,
        "wind_gusts_kmh": gust,
        "wind_direction": wdir,
        "surface_pressure_hpa": psfc,
        "theta_k": theta,
        "lcl_above_ground_m": lcl,
        "cloud_base_msl_m": (site.elevation + lcl) if lcl is not None else None,
        "fog_score": score,
        "fog_category": category,
        "fog_reasons": reasons,
    }


def analyse_hour(lower: Site, upper: Site, low_h: dict[str, Any],
                 up_h: dict[str, Any], i: int) -> dict[str, Any]:
    lo = site_state(lower, low_h, i)
    up = site_state(upper, up_h, i)

    lapse = theta_grad = None
    if lo["temperature_c"] is not None and up["temperature_c"] is not None:
        lapse = lapse_per_100m(lo["temperature_c"], lower.elevation,
                               up["temperature_c"], upper.elevation)
    if lo["theta_k"] is not None and up["theta_k"] is not None:
        theta_grad = (up["theta_k"] - lo["theta_k"]) / (upper.elevation - lower.elevation) * 100.0

    levels = free_air_profile(low_h, i, lower.elevation)
    inv_layers = find_inversion_layers(levels)
    moist = moist_layer(levels, lower.elevation, lo["relative_humidity"])

    # Scenario logic --------------------------------------------------------
    inversion = lapse is not None and lapse > 0
    lower_fog = lo["fog_score"] >= 3
    upper_fog = up["fog_score"] >= 3
    lower_deck = (lo["cloud_cover_low"] or 0) >= 60
    upper_deck = (up["cloud_cover_low"] or 0) >= 60
    cloud_base = lo["cloud_base_msl_m"]
    deck_top = moist["top_msl_m"] if moist else None
    deck_below_upper = (
        (deck_top is not None and deck_top < upper.elevation)
        or (deck_top is None and moist is None and cloud_base is not None
            and cloud_base < upper.elevation)
    )

    if inversion and lower_fog and not upper_fog:
        scenario = "Inversion with valley fog: sea of fog below, upper site in the sun"
    elif inversion and lower_fog:
        scenario = "Inversion, both sites inside the fog layer"
    elif inversion:
        scenario = "Dry inversion: valley colder than the mountain, no fog"
    elif lower_fog and not upper_fog:
        scenario = "Valley fog without inversion between the sites, upper site clear"
    elif lower_fog:
        scenario = "Fog at both sites"
    elif lower_deck and not upper_deck and deck_below_upper:
        scenario = "Low cloud deck over the valley, upper site above the cloud top"
    elif lower_deck and not upper_deck:
        scenario = "Low cloud over the valley, upper site clear (deck top uncertain)"
    elif lower_deck and upper_deck:
        scenario = "Both sites under or inside low cloud"
    elif upper_deck or upper_fog:
        scenario = "Upper site in cloud, valley clear (cloud base above the village)"
    else:
        scenario = "No inversion, no fog: well mixed air"

    return {
        "time": low_h["time"][i],
        "lower": lo,
        "upper": up,
        "lapse_c_per_100m": lapse,
        "lapse_class": classify_lapse(lapse),
        "theta_gradient_k_per_100m": theta_grad,
        "stability_class": classify_theta_gradient(theta_grad),
        "inversion_between_sites": inversion,
        "free_air_levels": levels,
        "free_air_inversion_layers": inv_layers,
        "moist_layer": moist,
        "scenario": scenario,
    }


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #

def fmt(v: float | None, spec: str = ".1f", unit: str = "") -> str:
    return "  n/a" if v is None else f"{v:{spec}}{unit}"


def print_site(state: dict[str, Any]) -> None:
    print(f"  {state['site']}  ({state['elevation_m']:.0f} m)")
    print(f"    T {fmt(state['temperature_c'])} C   Td {fmt(state['dew_point_c'])} C   "
          f"spread {fmt(state['spread_c'])} C   RH {fmt(state['relative_humidity'], '.0f')} %")
    vis = state["visibility_m"]
    print(f"    visibility {fmt(vis / 1000 if vis is not None else None, '.1f', ' km')}   "
          f"low cloud {fmt(state['cloud_cover_low'], '.0f', ' %')}   "
          f"wind {fmt(state['wind_speed_kmh'], '.0f', ' km/h')} gusts "
          f"{fmt(state['wind_gusts_kmh'], '.0f', ' km/h')} from {fmt(state['wind_direction'], '.0f', ' deg')}")
    print(f"    theta {fmt(state['theta_k'], '.1f', ' K')}   "
          f"cloud base (LCL) {fmt(state['cloud_base_msl_m'], '.0f', ' m')} MSL")
    print(f"    fog: {state['fog_category']} (score {state['fog_score']}/10)")
    for r in state["fog_reasons"]:
        print(f"      - {r}")


def print_report(now_idx: int, hours: list[dict[str, Any]], lower: Site, upper: Site,
                 model: str, current: list[dict[str, Any]]) -> None:
    now = hours[now_idx]
    print("=" * 78)
    print(f"Alpine inversion & fog model   {now['time']}   model: {model}")
    print("=" * 78)
    print()
    print("Live 15-minute values from Open-Meteo:")
    for site, cur in zip((lower, upper), current):
        c = cur["current"]
        print(f"  {site.name:<28} T {c['temperature_2m']:5.1f} C  Td {c['dew_point_2m']:5.1f} C  "
              f"RH {c['relative_humidity_2m']:3.0f} %  low cloud {c['cloud_cover_low']:3.0f} %  "
              f"wind {c['wind_speed_10m']:4.1f} km/h")
    print()
    print(f"Hourly analysis for {now['time']}:")
    print()
    print_site(now["lower"])
    print()
    print_site(now["upper"])
    print()

    dz = upper.elevation - lower.elevation
    print(f"Between the sites (dz = {dz:.0f} m):")
    print(f"  lapse rate       {fmt(now['lapse_c_per_100m'], '+.2f')} C/100 m   "
          f"-> {now['lapse_class']}   (standard {STANDARD_LAPSE:+.2f}, "
          f"dry adiabatic {DRY_ADIABATIC_LAPSE:+.2f})")
    print(f"  theta gradient   {fmt(now['theta_gradient_k_per_100m'], '+.2f')} K/100 m   "
          f"-> {now['stability_class']}")
    print(f"  inversion        {'YES' if now['inversion_between_sites'] else 'no'}")
    print()

    print("Free-air profile of the model column (levels above the valley floor):")
    print("     p [hPa]   z [m]    T [C]   RH [%]   theta [K]")
    for lv in now["free_air_levels"]:
        print(f"     {lv['pressure_hpa']:>5}   {lv['height_m']:6.0f}   {lv['temperature_c']:5.1f}   "
              f"{fmt(lv['relative_humidity'], '4.0f'):>6}   {lv['theta_k']:7.1f}")
    if now["free_air_inversion_layers"]:
        for lay in now["free_air_inversion_layers"]:
            print(f"  free-air inversion  {lay['base_m']:.0f} - {lay['top_m']:.0f} m, "
                  f"+{lay['delta_t_c']:.1f} C ({lay['lapse_c_per_100m']:+.2f} C/100 m)")
    else:
        print("  no free-air inversion layer above the valley floor")
    ml = now["moist_layer"]
    if ml is None:
        print("  moist layer         none (no level with RH >= 85 %, no stratus expected)")
    else:
        kind = "ground fog / stratus on the ground" if ml["touches_ground"] else "cloud deck"
        top = "above the top of the column" if ml["top_msl_m"] is None else f"~{ml['top_msl_m']:.0f} m"
        rel = ""
        if ml["top_msl_m"] is not None:
            rel = "  (upper site " + ("above" if ml["top_msl_m"] < upper.elevation else "inside/below") + " it)"
        print(f"  moist layer         {kind}: base ~{ml['base_msl_m']:.0f} m, top {top} MSL{rel}")
    print()
    print(f"Scenario: {now['scenario']}")
    print()

    print("Time line (lapse rate between the sites; fog score lower / upper):")
    print("  time              lapse   inv  fog_lo fog_up  Tlo   Tup   RHlo RHup  moist base-top")
    for k, h in enumerate(hours):
        marker = "<- now" if k == now_idx else ""
        lo, up = h["lower"], h["upper"]
        ml = h["moist_layer"]
        if ml is None:
            span = "     -"
        else:
            span = f"{ml['base_msl_m']:.0f}-{ml['top_msl_m']:.0f}" if ml["top_msl_m"] is not None else f"{ml['base_msl_m']:.0f}-top"
        print(f"  {h['time']:<16}  {fmt(h['lapse_c_per_100m'], '+.2f'):>6}   "
              f"{'YES' if h['inversion_between_sites'] else ' - '}   "
              f"{lo['fog_score']:>3}    {up['fog_score']:>3}   "
              f"{fmt(lo['temperature_c'], '4.1f')} {fmt(up['temperature_c'], '5.1f')}  "
              f"{fmt(lo['relative_humidity'], '3.0f'):>4} {fmt(up['relative_humidity'], '3.0f'):>4}  "
              f"{span:>10} {marker}")
    print()
    print("Notes: 2 m values are model data downscaled to the site elevation, not")
    print("station observations. Fog score is a heuristic (0-10). Inversion means")
    print("the upper site is warmer than the lower one.")


TEMPLATE_NAME = "inversion_chart.template.html"


def write_html(path: str, payload: dict[str, Any]) -> None:
    """Fill the chart template next to this script with the analysis payload."""
    template = os.path.join(os.path.dirname(os.path.abspath(__file__)), TEMPLATE_NAME)
    try:
        with open(template, encoding="utf-8") as fh:
            html = fh.read()
    except OSError as exc:
        raise SystemExit(f"Chart template not found: {template} ({exc})") from exc
    blob = json.dumps(payload).replace("</", "<\\/")
    html = html.replace("__INVERSION_DATA__", blob)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(html)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lower", type=Site.parse, default=DEFAULT_LOWER,
                    help="lower site as NAME:LAT:LON:ELEVATION")
    ap.add_argument("--upper", type=Site.parse, default=DEFAULT_UPPER,
                    help="upper site as NAME:LAT:LON:ELEVATION")
    ap.add_argument("--hours", type=int, default=24, help="forecast hours (default 24)")
    ap.add_argument("--model", default="best_match",
                    help="Open-Meteo model, e.g. best_match, icon_d2, icon_eu, ecmwf_ifs025")
    ap.add_argument("--json", action="store_true", help="print JSON instead of text")
    ap.add_argument("--html", metavar="FILE", help="also write an interactive chart page to FILE")
    ap.add_argument("--dump-url", action="store_true", help="print the API URL and exit")
    args = ap.parse_args(argv)

    lower, upper = args.lower, args.upper
    if upper.elevation <= lower.elevation:
        ap.error("--upper must be higher than --lower")

    url = build_query(lower, upper, args.hours, args.model)
    if args.dump_url:
        print(url)
        return 0

    data = fetch(url)
    low_h, up_h = data[0]["hourly"], data[1]["hourly"]
    times = low_h["time"]

    cur_time = data[0]["current"]["time"]              # e.g. 2026-09-12T07:45
    now_key = cur_time[:13] + ":00"
    now_idx = times.index(now_key) if now_key in times else 0

    hours = [analyse_hour(lower, upper, low_h, up_h, i) for i in range(len(times))]
    daily = data[0].get("daily", {})
    payload = {
        "generated": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "lower": asdict(lower),
        "upper": asdict(upper),
        "now_index": now_idx,
        "sun": {"sunrise": daily.get("sunrise", []), "sunset": daily.get("sunset", [])},
        "current": [d["current"] for d in data],
        "hours": hours,
    }

    if args.html:
        write_html(args.html, payload)
        print(f"chart written to {args.html}", file=sys.stderr)
    if args.json:
        json.dump(payload, sys.stdout, indent=2)
        print()
    elif not args.html:
        print_report(now_idx, hours, lower, upper, args.model, data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
