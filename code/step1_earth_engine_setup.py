"""
Step 1: Connect to Google Earth Engine and run a test that uses
the datasets this project will actually need (MODIS NDVI and ERA5-Land).
"""

import ee

# >>> CHANGE THIS to your own Google Cloud project ID <<<
PROJECT_ID = " "


def connect_to_earth_engine():
    """Initialize Earth Engine; sign in through the browser if needed."""
    if PROJECT_ID == "your-project-id-here":
        print("Please set PROJECT_ID at the top of this file to your own project ID.")
        return False

    # Attempt 1: use credentials saved from a previous login.
    try:
        ee.Initialize(project=PROJECT_ID)
        print("OK: connected using saved credentials.")
        return True
    except Exception as first_error:
        print(f"Could not connect yet ({first_error}). Trying to sign in...")

    # Attempt 2: sign in (opens a browser), then connect again.
    try:
        ee.Authenticate()
        ee.Initialize(project=PROJECT_ID)
        print("OK: signed in and connected.")
        return True
    except Exception as second_error:
        print(f"FAILED: {second_error}")
        print("\nMost common causes:")
        print("- PROJECT_ID is wrong (use the project ID, not its display name)")
        print("- The project is not registered/enabled for Earth Engine")
        print("- Your Google account has no permission to use that project")
        return False


def test_datasets():
    """Check that the two datasets we need respond."""
    try:
        # MODIS NDVI: when does the data end?
        modis = ee.ImageCollection("MODIS/061/MOD13Q1")
        last = ee.Date(modis.aggregate_max("system:time_start"))
        print("Latest MODIS NDVI image:", last.format("YYYY-MM-dd").getInfo())

        # ERA5-Land: how many monthly images exist?
        era = ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
        print("ERA5-Land monthly images:", era.size().getInfo())
        return True
    except Exception as error:
        print(f"Dataset test FAILED: {error}")
        return False


if __name__ == "__main__":
    if connect_to_earth_engine():
        test_datasets()