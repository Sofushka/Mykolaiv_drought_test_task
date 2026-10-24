"""
Step 7: Validation of the whole pipeline (Steps 2-6).

Every check prints PASS / WARN / FAIL / SKIP / ERROR and a short reason.
  PASS  = looks right
  WARN  = unusual, look at it but it may be fine
  FAIL  = something is wrong with the data or the code
  SKIP  = could not be run (e.g. an optional data source did not answer)
  ERROR = the check itself crashed

Checks:
  A. Settings and input coverage (every year has all months / enough satellite scenes)
  B. Missing / invalid values in the yearly indicators, plausible ranges
  C. Independent rain check against CHIRPS (a different rain dataset)
  D. Farmland mask: size and same area in every year
  E. Drought table integrity (no negatives, no missing years, valid classes)
  F. Standardization works (index averages ~0; share of Moderate+ is as expected)
  G. Drought table agrees with the raw indicators (dry + not green => big drought area)
  H. Districts add up to the oblast
  I. Known years (2007, 2012 drought; 2004, 2021 not)
  J. Baseline stability (results do not change much if 2022-2025 are left out)
  K. Output files exist and are not empty
A text report is saved to outputs/step7_validation_report.txt
"""

import math
import os
import sys

import ee
from step2_boundaries import get_mykolaiv_boundaries
from step3_farmland_mask import get_cropland_250m
from step4_indicators import (
    get_water_balance, get_ndvi, yearly_means, season_dates, YEARS, MONTHS)
from step5_drought_detection import get_drought_collection, zonal_drought_stats

PROJECT_ID = ""

# ------------------------------------------------ settings you may change
RUN_CHIRPS = True       # independent rain check (needs the CHIRPS dataset)
RUN_DISTRICTS = True    # districts vs oblast check (about 1 minute)
RUN_BASELINE = True     # baseline stability check (about 30 seconds)

MAJOR_SHARE = 20.0      # same rule as Step 6: drought year if > 20% farmland Moderate+
EXPECTED_DISTRICTS = {"Bashtanskyi", "Mykolaivskyi", "Pervomaiskyi", "Voznesenskyi"}
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")

RESULTS = []   # (status, name, detail)
C = {}         # shared data between checks


# ---------------------------------------------------------------- helpers
def record(name, status, detail=""):
    RESULTS.append((status, name, detail))
    print(f"  [{status}] {name}: {detail}")


def run(label, fn):
    print(f"\n{label}")
    try:
        fn()
    except Exception as error:
        record(label, "ERROR", str(error)[:300])


def mod_plus_pct(stat):
    total = sum(stat.get(c, 0.0) for c in range(5))
    if total <= 0:
        return 0.0
    return 100.0 * sum(stat.get(c, 0.0) for c in (2, 3, 4)) / total


def finite(x):
    return x is not None and isinstance(x, (int, float)) and math.isfinite(x)


def ranks(values):
    """Ranks 1..n, ties get the average rank."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            out[order[k]] = avg
        i = j + 1
    return out


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    if sa == 0 or sb == 0:
        return float("nan")
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb)


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def grade(value, pass_at, warn_at):
    """PASS if value >= pass_at, WARN if >= warn_at, else FAIL."""
    if not finite(value):
        return "FAIL"
    return "PASS" if value >= pass_at else ("WARN" if value >= warn_at else "FAIL")


def season_series(region, image_for_year, scale):
    """{year: oblast mean of a one-band image}, one Earth Engine call."""
    feats = []
    for y in YEARS:
        value = ee.Dictionary(image_for_year(y).reduceRegion(
            reducer=ee.Reducer.mean(), geometry=region, scale=scale,
            maxPixels=1e13, bestEffort=True)).values().get(0)
        feats.append(ee.Feature(None, {"year": y, "v": value}))
    rows = ee.FeatureCollection(feats).getInfo()["features"]
    return {r["properties"]["year"]: r["properties"].get("v") for r in rows}


# ----------------------------------------------------------------- A
def t_config():
    ok_years = YEARS == list(range(2000, 2026))
    ok_months = MONTHS == list(range(4, 10))
    record("Years are 2000-2025, no gaps or repeats", "PASS" if ok_years else "FAIL",
           f"{len(YEARS)} years")
    record("Season is April-September", "PASS" if ok_months else "FAIL",
           f"months {MONTHS[0]}-{MONTHS[-1]}")


def t_input_coverage():
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
    modis = ee.ImageCollection("MODIS/061/MOD13Q1")
    feats = []
    for y in YEARS:
        s, e = season_dates(y, MONTHS)
        feats.append(ee.Feature(None, {
            "year": y,
            "era5": era5.filterDate(s, e).size(),
            "modis": modis.filterDate(s, e).size()}))
    rows = [r["properties"] for r in ee.FeatureCollection(feats).getInfo()["features"]]
    bad_era = [r["year"] for r in rows if r["era5"] != len(MONTHS)]
    bad_mod = [r["year"] for r in rows if r["modis"] < 10]
    modis_counts = [r["modis"] for r in rows]
    record("ERA5-Land has all 6 monthly images in every season",
           "PASS" if not bad_era else "FAIL",
           "all years complete" if not bad_era else f"incomplete years: {bad_era}")
    record("MODIS has at least 10 scenes in every season",
           "PASS" if not bad_mod else "FAIL",
           f"min {min(modis_counts)}, max {max(modis_counts)} scenes"
           + ("" if not bad_mod else f"; too few in: {bad_mod}"))


# ----------------------------------------------------------------- B
def t_indicator_values():
    wb, nd = C["wb_means"], C["ndvi_means"]
    miss_wb = [y for y in YEARS if not finite(wb.get(y))]
    miss_nd = [y for y in YEARS if not finite(nd.get(y))]
    record("No missing water-balance values", "PASS" if not miss_wb else "FAIL",
           "26/26 present" if not miss_wb else f"missing: {miss_wb}")
    record("No missing NDVI values", "PASS" if not miss_nd else "FAIL",
           "26/26 present" if not miss_nd else f"missing: {miss_nd}")
    wbv = [wb[y] for y in YEARS if finite(wb.get(y))]
    ndv = [nd[y] for y in YEARS if finite(nd.get(y))]
    bad_wb = [y for y in YEARS if finite(wb.get(y)) and not (-2500 <= wb[y] <= 500)]
    bad_nd = [y for y in YEARS if finite(nd.get(y)) and not (0.2 <= nd[y] <= 0.8)]
    record("Water balance in plausible range (-2500..500 mm)",
           "PASS" if not bad_wb else "FAIL",
           f"range {min(wbv):.0f}..{max(wbv):.0f} mm" if wbv else "no data")
    record("NDVI in plausible range (0.2..0.8)", "PASS" if not bad_nd else "FAIL",
           f"range {min(ndv):.3f}..{max(ndv):.3f}" if ndv else "no data")
    flat_wb = len(set(round(v, 3) for v in wbv)) < 5
    flat_nd = len(set(round(v, 4) for v in ndv)) < 5
    record("Indicators vary between years (not constant or copied)",
           "FAIL" if (flat_wb or flat_nd) else "PASS",
           f"distinct values: WB {len(set(round(v, 3) for v in wbv))}, "
           f"NDVI {len(set(round(v, 4) for v in ndv))}")


def t_rain_and_pet():
    region = C["region"]
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

    def rain(y):
        s, e = season_dates(y, MONTHS)
        return era5.filterDate(s, e).select("total_precipitation_sum").sum().multiply(1000)

    def pet(y):
        s, e = season_dates(y, MONTHS)
        return (era5.filterDate(s, e).select("potential_evaporation_sum")
                .sum().abs().multiply(1000))

    C["era_rain"] = season_series(region, rain, 11132)
    era_pet = season_series(region, pet, 11132)
    r = [v for v in C["era_rain"].values() if finite(v)]
    p = [v for v in era_pet.values() if finite(v)]
    ok_r = len(r) == len(YEARS) and all(100 <= v <= 700 for v in r)
    ok_p = len(p) == len(YEARS) and all(600 <= v <= 2000 for v in p)
    record("Season rain (Apr-Sep) in plausible range (100-700 mm)",
           "PASS" if ok_r else "WARN",
           f"{min(r):.0f}..{max(r):.0f} mm" if r else "no data")
    record("Season potential evaporation in plausible range (600-2000 mm)",
           "PASS" if ok_p else "WARN",
           f"{min(p):.0f}..{max(p):.0f} mm (ERA5-Land values are known to be high)"
           if p else "no data")


# ----------------------------------------------------------------- C
def t_chirps():
    if not RUN_CHIRPS:
        record("Independent rain check (CHIRPS)", "SKIP", "RUN_CHIRPS = False")
        return
    region = C["region"]
    chirps = ee.ImageCollection("UCSB-CHG/CHIRPS/PENTAD")

    def rain(y):
        s, e = season_dates(y, MONTHS)
        return chirps.filterDate(s, e).select("precipitation").sum()

    try:
        ch = season_series(region, rain, 5566)
    except Exception as error:
        record("Independent rain check (CHIRPS)", "SKIP",
               f"CHIRPS could not be read: {str(error)[:150]}")
        return
    era = C.get("era_rain", {})
    years = [y for y in YEARS if finite(ch.get(y)) and finite(era.get(y))]
    if len(years) < 20:
        record("Independent rain check (CHIRPS)", "SKIP",
               f"only {len(years)} comparable years")
        return
    a = [era[y] for y in years]
    b = [ch[y] for y in years]
    corr = pearson(a, b)
    ratio = (sum(a) / len(a)) / (sum(b) / len(b))
    record("ERA5 rain agrees with CHIRPS rain year by year (correlation)",
           grade(corr, 0.8, 0.6), f"r = {corr:.2f} over {len(years)} years")
    record("ERA5 and CHIRPS give a similar average amount of rain",
           "PASS" if 0.7 <= ratio <= 1.4 else "WARN",
           f"ERA5 / CHIRPS = {ratio:.2f}")


# ----------------------------------------------------------------- D
def t_mask():
    st = C["stats"]
    totals = [sum(st[y].values()) for y in YEARS if y in st]
    spread = max(totals) - min(totals)
    record("Farmland area is identical in every year", "PASS" if spread < 1.0 else "FAIL",
           f"{totals[0]:.0f} km2, difference between years {spread:.2f} km2")
    share = totals[0] / C["region_km2"]
    record("Farmland is 55-90% of the oblast", "PASS" if 0.55 <= share <= 0.90 else "FAIL",
           f"{100 * share:.0f}% ({totals[0]:.0f} of {C['region_km2']:.0f} km2)")


# ----------------------------------------------------------------- E
def t_stats_integrity():
    st = C["stats"]
    missing = [y for y in YEARS if y not in st]
    record("Drought table has a row for every year", "PASS" if not missing else "FAIL",
           "26/26 years" if not missing else f"missing: {missing}")
    bad_keys = sorted({k for y in st for k in st[y] if k not in range(5)})
    record("Only drought classes 0-4 appear", "PASS" if not bad_keys else "FAIL",
           "classes OK" if not bad_keys else f"unexpected classes: {bad_keys}")
    neg = [y for y in st for v in st[y].values() if not finite(v) or v < 0]
    record("No negative or invalid areas", "PASS" if not neg else "FAIL",
           "all areas valid" if not neg else f"years with bad values: {sorted(set(neg))}")
    pcts = [mod_plus_pct(st[y]) for y in YEARS if y in st]
    record("Moderate+ share is between 0 and 100% in every year",
           "PASS" if all(0 <= p <= 100 for p in pcts) else "FAIL",
           f"{min(pcts):.1f}%..{max(pcts):.1f}%")


# ----------------------------------------------------------------- F
def t_standardisation():
    z = C["z_means"]
    vals = [z[y] for y in YEARS if finite(z.get(y))]
    if len(vals) < len(YEARS):
        record("Combined index available for every year", "FAIL",
               f"{len(vals)}/{len(YEARS)} years")
        return
    mean_all = sum(vals) / len(vals)
    record("Combined index averages about 0 over the 26 years",
           "PASS" if abs(mean_all) < 0.05 else ("WARN" if abs(mean_all) < 0.15 else "FAIL"),
           f"mean of yearly oblast averages = {mean_all:+.3f} (expected ~0)")
    st = C["stats"]
    n_years = len(st)
    total = sum(sum(st[y].values()) for y in st)
    mild_plus = sum(sum(v for c, v in st[y].items() if c >= 1) for y in st) / total
    mod_plus = sum(sum(v for c, v in st[y].items() if c >= 2) for y in st) / total
    record("Share of all farmland-years that are Mild or worse is 20-45%",
           "PASS" if 0.20 <= mild_plus <= 0.45 else "WARN",
           f"{100 * mild_plus:.1f}% (about 31% expected from the method)")
    record("Share of all farmland-years that are Moderate or worse is 10-25%",
           "PASS" if 0.10 <= mod_plus <= 0.25 else "WARN",
           f"{100 * mod_plus:.1f}% (about 16% expected from the method), {n_years} years")


# ----------------------------------------------------------------- G
def t_agreement():
    wb, nd, st = C["wb_means"], C["ndvi_means"], C["stats"]
    years = [y for y in YEARS if finite(wb.get(y)) and finite(nd.get(y)) and y in st]
    stress = [a + b for a, b in zip(ranks([-wb[y] for y in years]),
                                    ranks([-nd[y] for y in years]))]
    share = [mod_plus_pct(st[y]) for y in years]
    rho = spearman(stress, share)
    record("Years that are dry AND not green have the largest drought area (rank correlation)",
           grade(rho, 0.7, 0.5), f"rho = {rho:.2f} over {len(years)} years")
    corr_wb = spearman([-wb[y] for y in years], share)
    corr_nd = spearman([-nd[y] for y in years], share)
    record("Drought area follows water balance on its own",
           grade(corr_wb, 0.6, 0.4), f"rho = {corr_wb:.2f}")
    record("Drought area follows NDVI on its own",
           grade(corr_nd, 0.6, 0.4), f"rho = {corr_nd:.2f}")


# ----------------------------------------------------------------- H
def t_districts():
    if not RUN_DISTRICTS:
        record("Districts add up to the oblast", "SKIP", "RUN_DISTRICTS = False")
        return
    districts = C["districts"]
    lst = districts.toList(districts.size())
    n = districts.size().getInfo()
    ds = {}
    for i in range(n):
        d = ee.Feature(lst.get(i))
        name = d.get("GAUL2_NAME").getInfo()
        ds[name] = zonal_drought_stats(C["drought_col"], d.geometry(), C["modis_proj"])
    record("District names are the expected four",
           "PASS" if set(ds) == EXPECTED_DISTRICTS else "WARN", f"{sorted(ds)}")
    st = C["stats"]
    worst_total = worst_mod = 0.0
    for y in YEARS:
        total = sum(st[y].values())
        parts = sum(sum(d[y].values()) for d in ds.values())
        mod = sum(st[y].get(c, 0.0) for c in (2, 3, 4))
        mod_parts = sum(sum(d[y].get(c, 0.0) for c in (2, 3, 4)) for d in ds.values())
        worst_total = max(worst_total, abs(total - parts))
        worst_mod = max(worst_mod, abs(mod - mod_parts))
    record("District farmland totals add up to the oblast total",
           "PASS" if worst_total < 1.0 else ("WARN" if worst_total < 10 else "FAIL"),
           f"largest difference {worst_total:.2f} km2")
    record("District Moderate+ areas add up to the oblast Moderate+ area",
           "PASS" if worst_mod < 1.0 else ("WARN" if worst_mod < 10 else "FAIL"),
           f"largest difference {worst_mod:.2f} km2")


# ----------------------------------------------------------------- I
def t_known_years():
    st = C["stats"]
    p = {y: mod_plus_pct(st.get(y, {})) for y in YEARS}
    for y in (2007, 2012):
        record(f"{y} (known severe drought) is a major drought year",
               "PASS" if p[y] > 40 else "WARN", f"{p[y]:.1f}% Moderate+")
    for y in (2004, 2021):
        record(f"{y} (wet / good-harvest year) is not a drought year",
               "PASS" if p[y] < 5 else "WARN", f"{p[y]:.1f}% Moderate+")
    major = [y for y in YEARS if p[y] > MAJOR_SHARE]
    print(f"      drought years with the {MAJOR_SHARE:.0f}% rule: {len(major)} -> {major}")
    for cut in (10, 20, 50):
        n = sum(1 for y in YEARS if p[y] > cut)
        print(f"      years above {cut}%: {n}")


# ----------------------------------------------------------------- J
def t_baseline():
    if not RUN_BASELINE:
        record("Baseline stability", "SKIP", "RUN_BASELINE = False")
        return
    years_short = [y for y in YEARS if y <= 2021]
    col = get_drought_collection(years_short, MONTHS, C["mask250"])
    short = zonal_drought_stats(col, C["region"], C["modis_proj"])
    full = C["stats"]
    a = [mod_plus_pct(full[y]) for y in years_short]
    b = [mod_plus_pct(short[y]) for y in years_short]
    rho = spearman(a, b)
    record("Year ranking is stable if 2022-2025 are left out of the baseline",
           grade(rho, 0.9, 0.8), f"rank correlation = {rho:.2f} over {len(years_short)} years")
    major_full = {y for y in years_short if mod_plus_pct(full[y]) > MAJOR_SHARE}
    major_short = {y for y in years_short if mod_plus_pct(short[y]) > MAJOR_SHARE}
    record("Same drought years found with the shorter baseline (2000-2021)",
           "PASS" if major_full == major_short else "WARN",
           f"full: {sorted(major_full)}; short: {sorted(major_short)}")


# ----------------------------------------------------------------- K
def t_outputs():
    def info(name):
        path = os.path.join(OUT_DIR, name)
        return os.path.exists(path) and os.path.getsize(path) > 0
    for name in ("step6_report.html", "step6_timeseries.png"):
        record(f"Output file {name} exists and is not empty",
               "PASS" if info(name) else "FAIL", "")
    maps = [f for f in os.listdir(OUT_DIR)
            if f.startswith("step6_drought_") and f.endswith(".png")] if os.path.isdir(OUT_DIR) else []
    bad = []
    for f in maps:
        with open(os.path.join(OUT_DIR, f), "rb") as fh:
            if not fh.read(8).startswith(b"\x89PNG"):
                bad.append(f)
    record("Drought map pictures exist", "PASS" if maps else "WARN",
           f"{len(maps)} maps found")
    if bad:
        record("Map files are valid PNG pictures", "FAIL", f"broken: {bad}")
    report = os.path.join(OUT_DIR, "step6_report.html")
    if os.path.exists(report):
        with open(report, encoding="utf-8") as fh:
            html = fh.read()
        missing_pics = html.count("was not created")
        record("Report has no missing pictures", "PASS" if missing_pics == 0 else "WARN",
               f"{missing_pics} picture(s) missing")
        record("Report contains the data tables", "PASS" if html.count("<table") >= 1 else "FAIL",
               f"{html.count('<table')} table(s)")


# ------------------------------------------------------------------- main
def write_report():
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "step7_validation_report.txt")
    counts = {}
    for status, _, _ in RESULTS:
        counts[status] = counts.get(status, 0) + 1
    with open(path, "w", encoding="ascii", errors="replace") as f:
        f.write("STEP 7 VALIDATION REPORT\n")
        f.write(", ".join(f"{k}: {v}" for k, v in sorted(counts.items())) + "\n\n")
        for status, name, detail in RESULTS:
            f.write(f"[{status}] {name}: {detail}\n")
    return path, counts


if __name__ == "__main__":
    print("=" * 60)
    print("STEP 7: Validation")
    print("=" * 60)

    run("A1. Settings", t_config)

    ee.Initialize(project=PROJECT_ID)
    oblast, districts = get_mykolaiv_boundaries()
    region = oblast.geometry()
    share250, mask250, modis_proj = get_cropland_250m()

    print("\nBuilding the data used by the checks...")
    wb_col = get_water_balance(YEARS, MONTHS, mask250)
    ndvi_col = get_ndvi(YEARS, MONTHS, mask250)
    drought_col = get_drought_collection(YEARS, MONTHS, mask250)
    C.update(region=region, districts=districts, mask250=mask250,
             modis_proj=modis_proj, drought_col=drought_col,
             region_km2=region.area(maxError=100).divide(1e6).getInfo())
    C["stats"] = zonal_drought_stats(drought_col, region, modis_proj)
    C["wb_means"] = yearly_means(wb_col, "water_balance", region, modis_proj)
    C["ndvi_means"] = yearly_means(ndvi_col, "ndvi", region, modis_proj)
    C["z_means"] = yearly_means(drought_col.select("combined_z"), "combined_z",
                                region, modis_proj)

    run("A2. Input coverage (ERA5-Land months, MODIS scenes)", t_input_coverage)
    run("B1. Missing / invalid values in the indicators", t_indicator_values)
    run("B2. Rain and evaporation plausibility", t_rain_and_pet)
    run("C.  Independent rain check (CHIRPS)", t_chirps)
    run("D.  Farmland mask", t_mask)
    run("E.  Drought table integrity", t_stats_integrity)
    run("F.  Standardisation", t_standardisation)
    run("G.  Drought table vs raw indicators", t_agreement)
    run("H.  Districts vs oblast", t_districts)
    run("I.  Known years", t_known_years)
    run("J.  Baseline stability", t_baseline)
    run("K.  Output files", t_outputs)

    path, counts = write_report()
    print("\n" + "=" * 60)
    print("SUMMARY: " + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
    for status, name, detail in RESULTS:
        if status in ("FAIL", "ERROR"):
            print(f"  [{status}] {name}: {detail}")
    print(f"Report saved to: {path}")
    sys.exit(1 if (counts.get("FAIL") or counts.get("ERROR")) else 0)