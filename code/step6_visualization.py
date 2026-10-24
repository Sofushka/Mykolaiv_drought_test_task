"""
Step 6: Maps, chart and report for the drought results.

Everything is made on YOUR computer. Nothing is sent to Google Drive.
  1. Rank the years by the share of farmland in Moderate-or-worse drought.
  2. Maps: Earth Engine averages the drought index over a grid of ~5 km cells
     (the same kind of calculation as Steps 4-5, on the MODIS 250 m grid).
     We download that small TABLE (no picture, no raster) and matplotlib draws it.
  3. Draw the time-series chart.
  4. Build the per-district table.
  5. Write ONE html report with all pictures embedded inside the file.
"""

import base64
import math
import os

import ee
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import PathPatch, Patch, Polygon
from matplotlib.path import Path

from step2_boundaries import get_mykolaiv_boundaries
from step3_farmland_mask import get_cropland_250m
from step4_indicators import get_water_balance, get_ndvi, yearly_means, YEARS, MONTHS
from step5_drought_detection import (
    get_drought_collection, zonal_drought_stats, CLASS_NAMES, WAR_YEARS)

PROJECT_ID = ""

# ------------------------------------------------ settings you may change
MAX_MAP_YEARS = 8        # draw maps for at most this many drought years
SKIP_DISTRICTS = False   # True = skip the slower district table
CELL_KM = 5.0            # size of the map cells (km); smaller = more detail, slower
MAJOR_SHARE = 20.0       # a year is a "drought year" if more than this % is Moderate+

OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

DROUGHT_PALETTE = ["#d9f0d3", "#ffff00", "#ffaa00", "#ff0000", "#8b0000"]  # None -> Extreme
NOFARM_COLOR = "#bdbdbd"                  # inside the oblast but no farmland
THRESHOLDS = [0.5, 1.0, 1.5, 2.0]         # same class limits as Step 5


def out_path(name):
    return os.path.join(OUT_DIR, name)


def mod_plus_pct(stat):
    """% of farmland in Moderate or worse (classes 2-4), from a {class: km2} dict."""
    total = sum(stat.get(c, 0.0) for c in range(5))
    if total <= 0:
        return 0.0
    return 100.0 * sum(stat.get(c, 0.0) for c in (2, 3, 4)) / total


def class_of(z):
    """Drought class 0-4 from a combined index value (None stays None)."""
    if z is None:
        return None
    return sum(1 for t in THRESHOLDS if z >= t)


# ------------------------------------------------------------------ maps
def make_grid(region, cell_km=CELL_KM):
    """Grid parameters: cells of about cell_km x cell_km in lon/lat."""
    ring = region.bounds().getInfo()["coordinates"][0]
    lons = [p[0] for p in ring]
    lats = [p[1] for p in ring]
    x0, x1, y0, y1 = min(lons), max(lons), min(lats), max(lats)
    dlat = cell_km / 111.2
    dlon = cell_km / (111.32 * math.cos(math.radians((y0 + y1) / 2)))
    nx = int(math.ceil((x1 - x0) / dlon))
    ny = int(math.ceil((y1 - y0) / dlat))
    return {"x0": x0, "y0": y0, "dlon": dlon, "dlat": dlat, "nx": nx, "ny": ny}


def fetch_cell_values(drought_col, region, modis_proj, years, grid):
    """
    For each year: mean combined index per grid cell, using the same call
    pattern as zonal_drought_stats in Step 5 (grouped reduceRegion on the
    MODIS grid). Returns [{"ring": [...], "z": {"y2007": value, ...}}, ...].
    """
    ll = ee.Image.pixelLonLat()
    ix = ll.select("longitude").subtract(grid["x0"]).divide(grid["dlon"]).floor()
    iy = ll.select("latitude").subtract(grid["y0"]).divide(grid["dlat"]).floor()
    cell = iy.multiply(grid["nx"]).add(ix).toInt().rename("cell")

    items = drought_col.toList(len(YEARS))
    values = {}
    for year in years:
        img = ee.Image(items.get(YEARS.index(year)))
        z = img.select("combined_z").clip(region)
        # value first, group field (cell number) LAST, as in Step 5
        res = z.addBands(cell).reduceRegion(
            reducer=ee.Reducer.mean().group(groupField=1, groupName="cell"),
            geometry=region,
            crs=modis_proj,
            scale=modis_proj.nominalScale(),
            maxPixels=1e13,
            tileScale=6,
        )
        groups = res.get("groups").getInfo()
        if year == years[0] and groups:
            print("    example result row:", groups[0])   # shows the real key names
        for g in groups:
            val_keys = [k for k in g if k != "cell"]
            if not val_keys:
                continue
            values.setdefault(int(g["cell"]), {})[f"y{year}"] = g[val_keys[0]]
        print(f"    {year}: done")

    cells = []
    for cid, z in values.items():
        j, i = divmod(cid, grid["nx"])
        xa = grid["x0"] + i * grid["dlon"]
        ya = grid["y0"] + j * grid["dlat"]
        xb, yb = xa + grid["dlon"], ya + grid["dlat"]
        cells.append({"ring": [[xa, ya], [xb, ya], [xb, yb], [xa, yb], [xa, ya]],
                      "z": z})
    return cells


def polygon_rings(geojson):
    """Outer rings of a Polygon / MultiPolygon / GeometryCollection."""
    kind = geojson.get("type")
    if kind == "Polygon":
        return [geojson["coordinates"][0]]
    if kind == "MultiPolygon":
        return [poly[0] for poly in geojson["coordinates"]]
    if kind == "GeometryCollection":
        rings = []
        for g in geojson["geometries"]:
            rings += polygon_rings(g)
        return rings
    return []


def fetch_outlines(region, districts):
    """Simplified oblast outline and district outlines, as lon/lat rings."""
    oblast_rings = polygon_rings(region.simplify(500).getInfo())
    district_lines = []
    lst = districts.toList(districts.size())
    for i in range(districts.size().getInfo()):
        d = ee.Feature(lst.get(i))
        name = d.get("GAUL2_NAME").getInfo()
        rings = polygon_rings(d.geometry().simplify(500).getInfo())
        district_lines.append((name, rings))
    return oblast_rings, district_lines


def draw_cell_map(cells, year, oblast_rings, district_lines, share, filename):
    verts, codes = [], []
    for ring in oblast_rings:
        pts = [(p[0], p[1]) for p in ring]
        if len(pts) < 3:
            continue
        verts += pts
        codes += [Path.MOVETO] + [Path.LINETO] * (len(pts) - 2) + [Path.CLOSEPOLY]
    shape = Path(verts, codes)

    fig, ax = plt.subplots(figsize=(10, 7.5))
    base = PathPatch(shape, facecolor=NOFARM_COLOR, edgecolor="none")
    ax.add_patch(base)

    patches, colors = [], []
    for c in cells:
        cls = class_of(c["z"].get(f"y{year}"))
        if cls is None:
            continue                      # no farmland in this cell -> stays grey
        patches.append(Polygon([(p[0], p[1]) for p in c["ring"]], closed=True))
        colors.append(DROUGHT_PALETTE[cls])
    coll = PatchCollection(patches, facecolor=colors, edgecolor="none", linewidth=0)
    ax.add_collection(coll)
    coll.set_clip_path(base)              # cut the cells at the oblast border

    for ring in oblast_rings:
        ax.plot([p[0] for p in ring], [p[1] for p in ring], color="#333333", lw=1.3)
    for name, rings in district_lines:
        for ring in rings:
            ax.plot([p[0] for p in ring], [p[1] for p in ring],
                    color="#555555", lw=0.6, ls="--")
        if rings:
            biggest = max(rings, key=len)
            ax.text(sum(p[0] for p in biggest) / len(biggest),
                    sum(p[1] for p in biggest) / len(biggest),
                    name, fontsize=8, ha="center", color="#222222")

    xs = [p[0] for r in oblast_rings for p in r]
    ys = [p[1] for r in oblast_rings for p in r]
    ax.set_xlim(min(xs) - 0.05, max(xs) + 0.05)
    ax.set_ylim(min(ys) - 0.05, max(ys) + 0.05)
    ax.set_aspect(1.0 / math.cos(math.radians((min(ys) + max(ys)) / 2)))
    ax.axis("off")
    war = " (war year)" if year in WAR_YEARS else ""
    ax.set_title(f"Drought severity {year}{war}: {share:.0f}% of farmland Moderate or worse\n"
                 f"(average of the index in {CELL_KM:.0f} km cells)", fontsize=12)

    handles = [Patch(facecolor=DROUGHT_PALETTE[c], edgecolor="#666666", label=CLASS_NAMES[c])
               for c in range(5)]
    handles.append(Patch(facecolor=NOFARM_COLOR, edgecolor="#666666", label="No farmland"))
    ax.legend(handles=handles, loc="lower left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    fig.savefig(filename, dpi=130)
    plt.close(fig)


def make_drought_maps(drought_col, region, modis_proj, districts, map_years, stats):
    """Returns the list of years whose map was created."""
    if not map_years:
        return []
    try:
        grid = make_grid(region)
        print(f"  grid: {grid['nx']} x {grid['ny']} cells of about {CELL_KM:.0f} km")
        print("  averaging the index in each cell (one Earth Engine call per year) ...")
        cells = fetch_cell_values(drought_col, region, modis_proj, map_years, grid)
        print(f"  cells with farmland: {len(cells)}")
        oblast_rings, district_lines = fetch_outlines(region, districts)
    except Exception as error:
        print(f"  [FAILED] could not get the map data: {str(error)[:400]}")
        return []

    made = []
    for year in map_years:
        try:
            draw_cell_map(cells, year, oblast_rings, district_lines,
                          mod_plus_pct(stats.get(year, {})),
                          out_path(f"step6_drought_{year}.png"))
            print(f"  [OK] saved step6_drought_{year}.png")
            made.append(year)
        except Exception as error:
            print(f"  [FAILED] map {year}: {str(error)[:300]}")
    return made

# ---------------------------------------------------------------- chart
def create_time_series_chart(stats, wb_means, ndvi_means):
    nan = float("nan")
    wb = [wb_means.get(y) if wb_means.get(y) is not None else nan for y in YEARS]
    nd = [ndvi_means.get(y) if ndvi_means.get(y) is not None else nan for y in YEARS]

    fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True)

    bottom = [0.0] * len(YEARS)
    for c in range(1, 5):
        vals = [stats.get(y, {}).get(c, 0.0) for y in YEARS]
        axes[0].bar(YEARS, vals, bottom=bottom, color=DROUGHT_PALETTE[c],
                    edgecolor="#666666", linewidth=0.3, label=CLASS_NAMES[c])
        bottom = [b + v for b, v in zip(bottom, vals)]
    axes[0].set_ylabel("Farmland area (km2)")
    axes[0].set_title("Drought area by severity (Mild to Extreme)")
    axes[0].legend(loc="upper left", ncol=4, fontsize=9)

    axes[1].plot(YEARS, wb, marker="o", color="#2166ac")
    axes[1].set_ylabel("Water balance (mm)")
    axes[1].set_title("Season rain minus potential evaporation (oblast mean)")

    axes[2].plot(YEARS, nd, marker="o", color="#1a9850")
    axes[2].set_ylabel("NDVI")
    axes[2].set_title("Season mean NDVI (oblast mean)")

    for ax in axes:
        ax.axvspan(min(WAR_YEARS) - 0.5, max(WAR_YEARS) + 0.5, color="#999999", alpha=0.15)
        ax.grid(alpha=0.3)
    axes[0].text(min(WAR_YEARS) - 0.4, axes[0].get_ylim()[1] * 0.92, "war years",
                 fontsize=9, color="#555555")
    axes[2].set_xticks(YEARS)
    axes[2].set_xticklabels([str(y) for y in YEARS], rotation=90, fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path("step6_timeseries.png"), dpi=130)
    plt.close(fig)
    print("  [OK] saved step6_timeseries.png")


# ------------------------------------------------------------- districts
def create_district_breakdown(drought_col, districts, modis_proj):
    district_list = districts.toList(districts.size())
    n = district_list.size().getInfo()
    results = {}
    for i in range(n):
        d = ee.Feature(district_list.get(i))
        name = d.get("GAUL2_NAME").getInfo()
        results[name] = zonal_drought_stats(drought_col, d.geometry(), modis_proj)
        print(f"  {name}: done")
    return results


def check_district_totals(stats, district_stats):
    """The districts together should cover the same farmland as the oblast."""
    worst = 0.0
    for y in YEARS:
        total = sum(stats.get(y, {}).values())
        parts = sum(sum(d.get(y, {}).values()) for d in district_stats.values())
        worst = max(worst, abs(total - parts))
    print(f"  check: districts vs oblast farmland total, largest difference "
          f"{worst:.1f} km2 (should be small)")


# ------------------------------------------------------------------ HTML
CSS = """
body {font-family: Arial, sans-serif; max-width: 960px; margin: 2rem auto; padding: 1rem; line-height: 1.6;}
h1 {color: #2c3e50;} h2 {color: #34495e; border-bottom: 1px solid #ddd; padding-bottom: 0.3rem;}
table {border-collapse: collapse; width: 100%; margin: 1rem 0;}
th, td {border: 1px solid #ddd; padding: 0.4rem 0.6rem; text-align: right;}
th {background: #f0f0f0;}
td.h1 {background: #fff3cd;} td.h2 {background: #ffd8a8;} td.h3 {background: #f5a3a3;}
tr.war td:first-child {font-style: italic; color: #6c757d;}
img {max-width: 100%; height: auto; margin: 0.5rem 0; border: 1px solid #ddd;}
.legend {display: flex; gap: 1rem; flex-wrap: wrap; margin: 1rem 0;}
.legend span {display: inline-flex; align-items: center; gap: 0.3rem;}
.sw {width: 1rem; height: 1rem; display: inline-block; border: 1px solid #999;}
.note {font-size: 0.9rem; color: #555;}
"""


def heat(pct):
    """CSS class for a Moderate+ percentage cell."""
    if pct >= 50:
        return "h3"
    if pct >= 20:
        return "h2"
    if pct >= 10:
        return "h1"
    return ""


def img_tag(path, alt):
    """Put the picture INSIDE the html file, so it can never show as a broken link."""
    if os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("ascii")
        return f'<img src="data:image/png;base64,{b64}" alt="{alt}">'
    return f'<p class="note">({alt}: picture was not created)</p>'


def generate_html_report(stats, district_stats, map_years):
    legend = [("None (&lt;0.5&sigma;)", DROUGHT_PALETTE[0]),
              ("Mild (0.5-1&sigma;)", DROUGHT_PALETTE[1]),
              ("Moderate (1-1.5&sigma;)", DROUGHT_PALETTE[2]),
              ("Severe (1.5-2&sigma;)", DROUGHT_PALETTE[3]),
              ("Extreme (&ge;2&sigma;)", DROUGHT_PALETTE[4]),
              ("No farmland", NOFARM_COLOR)]

    p = []
    p.append("<!DOCTYPE html>\n<html><head><meta charset=\"utf-8\">"
             "<title>Mykolaiv Drought Assessment 2000-2025</title>"
             f"<style>{CSS}</style></head><body>")
    p.append("<h1>Mykolaiv Oblast Drought Assessment (2000-2025)</h1>")
    p.append("<p><strong>Season:</strong> April-September | <strong>Grid:</strong> MODIS 250 m | "
             "<strong>Indicators:</strong> water balance (ERA5-Land rain minus potential "
             "evaporation) and MODIS NDVI, on farmland only.</p>")
    p.append("<p class=\"note\">Classes describe how unusual each year is compared with the "
             "same place in 2000-2025 (a standardised combined anomaly). Because every place "
             "is compared with its own history, a place is Moderate-or-worse in roughly one "
             "year in six by construction; look for years when a large share is affected "
             "at once. Years 2022-2025 (italic) are war years: NDVI there may also reflect "
             "abandoned or mined fields.</p>")

    p.append("<h2>Severity classes</h2><div class=\"legend\">")
    for label, color in legend:
        p.append(f'<span><i class="sw" style="background:{color}"></i> {label}</span>')
    p.append("</div>")

    p.append("<h2>Time series</h2>" + img_tag(out_path("step6_timeseries.png"), "Time series"))

    p.append("<h2>Oblast-level drought area (km&sup2;) by year</h2>"
             "<table><tr><th>Year</th><th>None</th><th>Mild</th><th>Moderate</th>"
             "<th>Severe</th><th>Extreme</th><th>Total</th><th>Moderate+ %</th></tr>")
    for y in YEARS:
        s = stats.get(y, {})
        areas = [s.get(c, 0.0) for c in range(5)]
        pct = mod_plus_pct(s)
        row = ' class="war"' if y in WAR_YEARS else ""
        cells = "".join(f"<td>{a:.0f}</td>" for a in areas)
        p.append(f"<tr{row}><td>{y}</td>{cells}<td>{sum(areas):.0f}</td>"
                 f'<td class="{heat(pct)}">{pct:.1f}%</td></tr>')
    p.append("</table>")

    if district_stats:
        names = list(district_stats.keys())
        head = "".join(f"<th>{n}</th>" for n in names)
        p.append("<h2>Districts: % of farmland in Moderate+ drought</h2>"
                 f"<table><tr><th>Year</th>{head}</tr>")
        for y in YEARS:
            row = ' class="war"' if y in WAR_YEARS else ""
            cells = ""
            for n in names:
                pct = mod_plus_pct(district_stats[n].get(y, {}))
                cells += f'<td class="{heat(pct)}">{pct:.1f}%</td>'
            p.append(f"<tr{row}><td>{y}</td>{cells}</tr>")
        p.append("</table>")

    p.append(f"<h2>Drought maps (years with more than {MAJOR_SHARE:.0f}% of farmland "
             "in Moderate+)</h2>")
    if not map_years:
        p.append('<p class="note">No map was created this time.</p>')
    for y in map_years:
        p.append(f"<h3>{y}</h3>" + img_tag(out_path(f"step6_drought_{y}.png"), f"Drought {y}"))

    p.append("</body></html>")
    with open(out_path("step6_report.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(p))
    print("  [OK] saved step6_report.html")


# ------------------------------------------------------------------ main
if __name__ == "__main__":
    print("=" * 60)
    print("STEP 6: Maps, chart and report")
    print("=" * 60)

    ee.Initialize(project=PROJECT_ID)
    print("Output folder:", OUT_DIR)

    oblast, districts = get_mykolaiv_boundaries()
    region = oblast.geometry()
    share250, mask250, modis_proj = get_cropland_250m()

    print("\nRebuilding the collections...")
    wb_col = get_water_balance(YEARS, MONTHS, mask250)
    ndvi_col = get_ndvi(YEARS, MONTHS, mask250)
    drought_col = get_drought_collection(YEARS, MONTHS, mask250)

    print("Computing oblast-level statistics...")
    stats = zonal_drought_stats(drought_col, region, modis_proj)

    ranked = sorted(YEARS, key=lambda y: mod_plus_pct(stats.get(y, {})), reverse=True)
    print("\nYears ranked by % of farmland in Moderate-or-worse drought (top 10):")
    for y in ranked[:10]:
        flag = "  (war year)" if y in WAR_YEARS else ""
        print(f"  {y}: {mod_plus_pct(stats.get(y, {})):.1f}%{flag}")
    drought_years = [y for y in ranked if mod_plus_pct(stats.get(y, {})) > MAJOR_SHARE]
    map_years = sorted(drought_years[:MAX_MAP_YEARS])

    print(f"\nDrawing maps for {map_years} ...")
    maps_made = make_drought_maps(drought_col, region, modis_proj, districts,
                                  map_years, stats)

    print("\nComputing yearly water balance / NDVI means for the chart...")
    wb_means = yearly_means(wb_col, "water_balance", region, modis_proj)
    ndvi_means = yearly_means(ndvi_col, "ndvi", region, modis_proj)
    create_time_series_chart(stats, wb_means, ndvi_means)

    district_stats = {}
    if SKIP_DISTRICTS:
        print("\nSkipping the district table (SKIP_DISTRICTS = True).")
    else:
        print("\nComputing the district table (repeats the calculation per district)...")
        district_stats = create_district_breakdown(drought_col, districts, modis_proj)
        check_district_totals(stats, district_stats)

    print("\nBuilding the report...")
    generate_html_report(stats, district_stats, maps_made)

    print("\nFiles in the output folder:")
    for name in sorted(os.listdir(OUT_DIR)):
        print("  ", name)
    print(f"\nOpen this file in your browser: {out_path('step6_report.html')}")