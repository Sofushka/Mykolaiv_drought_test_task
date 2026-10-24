"""
Step 4: Two indicators per year, on farmland only:
  1. Water balance (mm) = rain minus POTENTIAL evaporation (ERA5-Land)
  2. Greenness (NDVI) from MODIS, with cloudy/snowy data removed
Nothing is saved here. Step 5 imports these functions and rebuilds the collections.
"""

import ee
from step2_boundaries import get_mykolaiv_boundaries
from step3_farmland_mask import get_cropland_250m

PROJECT_ID = ""
YEARS = list(range(2000, 2026))   # 2000 ... 2025 inclusive
MONTHS = list(range(4, 10))       # 4..9 = April ... September (6 months)


def season_dates(year, months):
    """First day of the season (included) and the day after it ends (excluded)."""
    start = ee.Date.fromYMD(year, months[0], 1)
    end = ee.Date.fromYMD(year, months[-1], 1).advance(1, "month")
    return start, end


def get_water_balance(years, months, crop_mask):
    """One image per year: season rain minus potential evaporation, in mm."""
    era5 = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")

    def one_year(year):
        start, end = season_dates(year, months)
        season = era5.filterDate(start, end)
        rain = season.select("total_precipitation_sum").sum()
        # Evaporation is stored as a negative number (we saw it in the test),
        # so abs() turns it into a positive amount of water.
        pet = season.select("potential_evaporation_sum").sum().abs()
        wb = rain.subtract(pet).multiply(1000).rename("water_balance")  # m -> mm
        # NOTE: no .reproject() here. It made Earth Engine redraw the whole world.
        return wb.updateMask(crop_mask).set("year", year)

    return ee.ImageCollection([one_year(y) for y in years])


def get_ndvi(years, months, crop_mask):
    """One image per year: mean NDVI of the season (good-quality data only)."""
    modis = ee.ImageCollection("MODIS/061/MOD13Q1")

    def clean(img):
        good = img.select("SummaryQA").lte(1)      # 0 = good, 1 = usable
        return img.select("NDVI").updateMask(good)

    def one_year(year):
        start, end = season_dates(year, months)
        ndvi = (modis.filterDate(start, end)
                .map(clean)
                .mean()
                .multiply(0.0001)                  # stored as 0-10000, we want 0-1
                .rename("ndvi"))
        return ndvi.updateMask(crop_mask).set("year", year)

    return ee.ImageCollection([one_year(y) for y in years])


def yearly_means(collection, band, region, modis_proj):
    """Average value over the oblast's farmland, per year -> {year: value}."""
    def one(img):
        # Clip to region bounds first to avoid "reprojection output too large"
        clipped = img.clip(region)
        mean = clipped.reduceRegion(
            reducer=ee.Reducer.mean(),
            crs=modis_proj,                  # count on the MODIS 250 m grid
            scale=modis_proj.nominalScale(),
            maxPixels=1e13,
            tileScale=6,
            bestEffort=True,
        ).get(band)
        return ee.Feature(None, {"year": img.get("year"), band: mean})

    features = ee.FeatureCollection(collection.map(one)).getInfo()["features"]
    return {f["properties"]["year"]: f["properties"].get(band) for f in features}


def fmt(value, digits):
    return "n/a" if value is None else f"{value:.{digits}f}"


if __name__ == "__main__":
    ee.Initialize(project=PROJECT_ID)

    oblast, districts = get_mykolaiv_boundaries()
    region = oblast.geometry()
    share250, mask250, modis_proj = get_cropland_250m()

    print("\nBuilding indicators (this can take a few minutes)...")
    wb_col = get_water_balance(YEARS, MONTHS, mask250)
    ndvi_col = get_ndvi(YEARS, MONTHS, mask250)

    wb_means = yearly_means(wb_col, "water_balance", region, modis_proj)
    ndvi_means = yearly_means(ndvi_col, "ndvi", region, modis_proj)

    print(f"\nSeason: months {MONTHS[0]}-{MONTHS[-1]}")
    print(f"{'year':<6}{'water balance (mm)':>20}{'NDVI':>10}")
    for y in YEARS:
        print(f"{y:<6}{fmt(wb_means.get(y), 0):>20}{fmt(ndvi_means.get(y), 3):>10}")

    print("\nStep 4 complete.")