"""
Step 2: Mykolaiv Oblast and its districts (FAO GAUL 2025).
Stops with a clear message instead of continuing with empty boundaries.
"""

import ee

PROJECT_ID = ""

# Pieces of the name that should match however GAUL spells it.
NAME_FRAGMENTS = ["ykola", "ikola"]


def get_mykolaiv_boundaries():
    """Return (oblast_collection, districts_collection)."""
    level1 = ee.FeatureCollection("FAO/GAUL/2025/level1")  # oblasts
    level2 = ee.FeatureCollection("FAO/GAUL/2025/level2")  # districts

    # --- Oblast: Ukraine only, name contains one of the fragments ---
    ukraine = level1.filter(ee.Filter.eq("ISO3_CODE", "UKR"))
    name_match = ee.Filter.Or(
        *[ee.Filter.stringContains("GAUL1_NAME", f) for f in NAME_FRAGMENTS]
    )
    oblast = ukraine.filter(name_match)

    count = oblast.size().getInfo()
    print("Oblast matches:", count, oblast.aggregate_array("GAUL1_NAME").getInfo())

    if count != 1:
        print("All Ukrainian oblast names in GAUL 2025:",
              ukraine.aggregate_array("GAUL1_NAME").getInfo())
        print("Column names in level 1:", level1.first().propertyNames().getInfo())
        raise ValueError(f"Expected exactly 1 oblast, found {count}. "
                         "Check the names printed above.")

    # --- Districts: all level-2 units whose parent has the oblast's code ---
    oblast_code = oblast.first().get("GAUL1_CODE").getInfo()
    print("Oblast GAUL1_CODE:", oblast_code)

    districts = level2.filter(ee.Filter.eq("GAUL1_CODE", oblast_code))
    n_districts = districts.size().getInfo()
    print("Number of districts:", n_districts)

    if n_districts == 0:
        print("Column names in level 2:", level2.first().propertyNames().getInfo())
        raise ValueError("No districts found. Check the column names printed above.")

    print("District names:", districts.aggregate_array("GAUL2_NAME").getInfo())
    return oblast, districts


def check_area(oblast):
    """Sanity check: the oblast should be roughly 24,600 square km."""
    area_km2 = oblast.geometry().area(maxError=100).divide(1e6).getInfo()
    print(f"Oblast area: {area_km2:,.0f} km2 (expected roughly 24,600)")
    if not 22000 < area_km2 < 27000:
        print("WARNING: area looks wrong, check that the right shape was selected.")


if __name__ == "__main__":
    ee.Initialize(project=PROJECT_ID)
    oblast, districts = get_mykolaiv_boundaries()
    check_area(oblast)
    print("Step 2 complete.")