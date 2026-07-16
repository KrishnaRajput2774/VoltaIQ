"""
VoltIQ – tests/test_charging_proximity.py
==========================================
Unit tests for the Charging Proximity Index (CPI) implementation.

Phase 1 mandatory deliverable tests covering:
    TestHaversine                — Haversine distance formula correctness
    TestProximityIndexFormula    — CPI conversion formula edge cases
    TestCPIComputation           — compute_charging_proximity_index() full coverage
    TestRelationshipMapperCPI    — RelationshipMapper.compute_route_charging_proximity()

Data constraint tested:
    fleet_route_data.csv has NO latitude/longitude (city names only).
    The implementation uses city centroids derived from charging station data
    as a dataset-constrained approximation (not a true geospatial join).
"""

import math
import unittest

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------

def _routes(*cities) -> pd.DataFrame:
    """Build a minimal routes DataFrame with the given Start_Location values."""
    return pd.DataFrame({
        "Vehicle_ID":     [f"V{i}" for i in range(len(cities))],
        "Route_ID":       [f"RT-{i}" for i in range(len(cities))],
        "Start_Location": list(cities),
        "End_Location":   ["Dest"] * len(cities),
        "Distance_km":    [100.0] * len(cities),
    })


def _stations(city: str, lat: float, lon: float, n: int = 3) -> pd.DataFrame:
    """Build a minimal stations DataFrame with n stations near (lat, lon)."""
    return pd.DataFrame({
        "Station_ID": [f"S{i}" for i in range(n)],
        "City":       [city] * n,
        "Latitude":   [lat + 0.01 * i for i in range(n)],
        "Longitude":  [lon + 0.01 * i for i in range(n)],
        "Power_kW":   [50.0] * n,
    })


# ---------------------------------------------------------------------------
# Haversine distance tests
# ---------------------------------------------------------------------------

class TestHaversine(unittest.TestCase):
    """Tests for haversine_distance_km() formula implementation."""

    def test_same_point_is_zero(self):
        """Distance from a point to itself is 0 km."""
        from app.utils.relationships import haversine_distance_km
        d = haversine_distance_km(
            np.array([37.77]), np.array([-122.41]),
            np.array([37.77]), np.array([-122.41]),
        )
        self.assertAlmostEqual(float(d[0]), 0.0, places=6)

    def test_london_to_paris_approx_340km(self):
        """London to Paris is approximately 340 km by Haversine."""
        from app.utils.relationships import haversine_distance_km
        d = haversine_distance_km(
            np.array([51.5074]), np.array([-0.1278]),
            np.array([48.8566]), np.array([2.3522]),
        )
        self.assertAlmostEqual(float(d[0]), 340.0, delta=15.0)

    def test_vectorised_ordering(self):
        """Larger arc angle produces greater distance."""
        from app.utils.relationships import haversine_distance_km
        d = haversine_distance_km(
            np.array([0.0, 0.0]), np.array([0.0, 0.0]),
            np.array([1.0, 2.0]), np.array([0.0, 0.0]),
        )
        self.assertEqual(len(d), 2)
        self.assertLess(d[0], d[1])

    def test_symmetric(self):
        """d(A→B) == d(B→A)."""
        from app.utils.relationships import haversine_distance_km
        d_ab = haversine_distance_km(
            np.array([51.5]), np.array([-0.12]),
            np.array([48.85]), np.array([2.35]),
        )
        d_ba = haversine_distance_km(
            np.array([48.85]), np.array([2.35]),
            np.array([51.5]), np.array([-0.12]),
        )
        self.assertAlmostEqual(float(d_ab[0]), float(d_ba[0]), places=6)

    def test_returns_numpy_array(self):
        """Return type is a numpy ndarray."""
        from app.utils.relationships import haversine_distance_km
        result = haversine_distance_km(
            np.array([0.0]), np.array([0.0]),
            np.array([1.0]), np.array([0.0]),
        )
        self.assertIsInstance(result, np.ndarray)


# ---------------------------------------------------------------------------
# Proximity index formula tests
# ---------------------------------------------------------------------------

class TestProximityIndexFormula(unittest.TestCase):
    """Tests for proximity_index_from_distance()."""

    def test_zero_distance_gives_one(self):
        """Station at exact origin gives CPI = 1.0."""
        from app.utils.relationships import proximity_index_from_distance
        self.assertAlmostEqual(proximity_index_from_distance(0.0), 1.0, places=6)

    def test_at_half_life_gives_exp_minus_one(self):
        """CPI = exp(-1) at distance equal to half_life_km."""
        from app.utils.relationships import proximity_index_from_distance
        result = proximity_index_from_distance(20.0, half_life_km=20.0)
        self.assertAlmostEqual(result, math.exp(-1), places=6)

    def test_nan_returns_zero(self):
        """NaN distance gives CPI = 0.0."""
        from app.utils.relationships import proximity_index_from_distance
        self.assertEqual(proximity_index_from_distance(float("nan")), 0.0)

    def test_negative_returns_zero(self):
        """Negative distance gives CPI = 0.0."""
        from app.utils.relationships import proximity_index_from_distance
        self.assertEqual(proximity_index_from_distance(-1.0), 0.0)

    def test_infinite_returns_zero(self):
        """Infinite distance gives CPI = 0.0."""
        from app.utils.relationships import proximity_index_from_distance
        self.assertEqual(proximity_index_from_distance(float("inf")), 0.0)

    def test_zero_half_life_raises(self):
        """half_life_km=0 raises ValueError."""
        from app.utils.relationships import proximity_index_from_distance
        with self.assertRaises(ValueError):
            proximity_index_from_distance(10.0, half_life_km=0)

    def test_negative_half_life_raises(self):
        """Negative half_life_km raises ValueError."""
        from app.utils.relationships import proximity_index_from_distance
        with self.assertRaises(ValueError):
            proximity_index_from_distance(10.0, half_life_km=-5.0)

    def test_strictly_decreasing(self):
        """CPI decreases monotonically as distance increases."""
        from app.utils.relationships import proximity_index_from_distance
        cpis = [proximity_index_from_distance(float(d)) for d in [0, 5, 20, 50, 100]]
        for i in range(len(cpis) - 1):
            self.assertGreater(cpis[i], cpis[i + 1])

    def test_result_always_in_zero_one(self):
        """CPI is always within [0.0, 1.0]."""
        from app.utils.relationships import proximity_index_from_distance
        for d in [0.0, 0.001, 1.0, 10.0, 100.0, 1000.0]:
            cpi = proximity_index_from_distance(d)
            self.assertGreaterEqual(cpi, 0.0)
            self.assertLessEqual(cpi, 1.0)


# ---------------------------------------------------------------------------
# compute_charging_proximity_index — core function tests
# ---------------------------------------------------------------------------

class TestCPIComputation(unittest.TestCase):
    """Tests for compute_charging_proximity_index() function."""

    def test_matched_city_nonzero_cpi(self):
        """Routes whose city appears in station data receive CPI > 0."""
        from app.utils.relationships import compute_charging_proximity_index
        result = compute_charging_proximity_index(
            _routes("London", "London"),
            _stations("London", 51.5, -0.12, n=5),
        )
        self.assertTrue((result["Charging_Proximity_Index"] > 0).all())

    def test_unmatched_city_zero_cpi(self):
        """Routes whose city has no station data receive CPI = 0.0."""
        from app.utils.relationships import (
            compute_charging_proximity_index, CPI_METHOD_NO_COVERAGE,
        )
        result = compute_charging_proximity_index(
            _routes("Atlantis", "Narnia"),
            _stations("London", 51.5, -0.12, n=3),
        )
        self.assertTrue((result["Charging_Proximity_Index"] == 0.0).all())
        self.assertTrue((result["CPI_Method"] == CPI_METHOD_NO_COVERAGE).all())

    def test_mixed_matched_and_unmatched(self):
        """Mixed routes: matched rows get CPI > 0, unmatched get CPI = 0."""
        from app.utils.relationships import (
            compute_charging_proximity_index,
            CPI_METHOD_APPROX, CPI_METHOD_NO_COVERAGE,
        )
        result = compute_charging_proximity_index(
            _routes("London", "Atlantis", "London", "Narnia"),
            _stations("London", 51.5, -0.12, n=5),
        )
        matched   = result[result["CPI_Method"] == CPI_METHOD_APPROX]
        unmatched = result[result["CPI_Method"] == CPI_METHOD_NO_COVERAGE]
        self.assertEqual(len(matched),   2)
        self.assertEqual(len(unmatched), 2)
        self.assertTrue((matched["Charging_Proximity_Index"] > 0).all())
        self.assertTrue((unmatched["Charging_Proximity_Index"] == 0.0).all())

    def test_all_six_output_columns_present(self):
        """All six CPI output columns are added."""
        from app.utils.relationships import compute_charging_proximity_index
        result = compute_charging_proximity_index(
            _routes("London"), _stations("London", 51.5, -0.12)
        )
        for col in ("Centroid_Lat", "Centroid_Lon",
                    "Nearest_Station_Distance_km",
                    "Charging_Proximity_Index",
                    "CPI_Coverage_Category",
                    "CPI_Method"):
            self.assertIn(col, result.columns, f"Missing: {col}")

    def test_cpi_always_in_zero_one(self):
        """CPI values are bounded to [0, 1] for all route types."""
        from app.utils.relationships import compute_charging_proximity_index
        stations = pd.concat([
            _stations("London", 51.5, -0.12, n=3),
            _stations("Paris",  48.85, 2.35, n=3),
        ], ignore_index=True)
        result = compute_charging_proximity_index(
            _routes("London", "Paris", "Atlantis"), stations
        )
        cpi = result["Charging_Proximity_Index"]
        self.assertTrue((cpi >= 0.0).all(), "CPI must be >= 0")
        self.assertTrue((cpi <= 1.0).all(), "CPI must be <= 1")

    def test_invalid_station_coordinates_excluded(self):
        """Stations with out-of-range coordinates are excluded; valid ones used."""
        from app.utils.relationships import compute_charging_proximity_index
        stations = pd.DataFrame({
            "Station_ID": ["S1", "S2"],
            "City":       ["London", "London"],
            "Latitude":   [51.5, 999.0],
            "Longitude":  [-0.12, 50.0],
            "Power_kW":   [50.0, 50.0],
        })
        result = compute_charging_proximity_index(_routes("London"), stations)
        self.assertGreater(result["Charging_Proximity_Index"].iloc[0], 0.0)

    def test_null_station_coordinates_excluded(self):
        """Stations with NaN lat/lon are dropped; remaining valid rows used."""
        from app.utils.relationships import compute_charging_proximity_index
        stations = pd.DataFrame({
            "Station_ID": ["S1", "S2"],
            "City":       ["London", "London"],
            "Latitude":   [51.5, float("nan")],
            "Longitude":  [-0.12, float("nan")],
            "Power_kW":   [50.0, 50.0],
        })
        result = compute_charging_proximity_index(_routes("London"), stations)
        self.assertGreater(result["Charging_Proximity_Index"].iloc[0], 0.0)

    def test_empty_station_dataset_all_zero(self):
        """Empty stations dataset forces CPI = 0.0 for all routes."""
        from app.utils.relationships import compute_charging_proximity_index
        stations = pd.DataFrame(
            columns=["Station_ID", "City", "Latitude", "Longitude"]
        )
        result = compute_charging_proximity_index(_routes("London", "Paris"), stations)
        self.assertTrue((result["Charging_Proximity_Index"] == 0.0).all())

    def test_missing_location_column_raises_valueerror(self):
        """Missing Start_Location raises ValueError with descriptive message."""
        from app.utils.relationships import compute_charging_proximity_index
        routes = _routes("London").drop(columns=["Start_Location"])
        with self.assertRaises(ValueError):
            compute_charging_proximity_index(routes, _stations("London", 51.5, -0.12))

    def test_missing_lat_column_raises_valueerror(self):
        """Missing Latitude in stations raises ValueError."""
        from app.utils.relationships import compute_charging_proximity_index
        with self.assertRaises(ValueError):
            compute_charging_proximity_index(
                _routes("London"),
                _stations("London", 51.5, -0.12).drop(columns=["Latitude"]),
            )

    def test_missing_lon_column_raises_valueerror(self):
        """Missing Longitude in stations raises ValueError."""
        from app.utils.relationships import compute_charging_proximity_index
        with self.assertRaises(ValueError):
            compute_charging_proximity_index(
                _routes("London"),
                _stations("London", 51.5, -0.12).drop(columns=["Longitude"]),
            )

    def test_source_dataframes_not_modified(self):
        """Input DataFrames are never mutated by the computation."""
        from app.utils.relationships import compute_charging_proximity_index
        routes   = _routes("London", "Paris")
        stations = _stations("London", 51.5, -0.12, n=3)
        r_cols   = list(routes.columns)
        s_cols   = list(stations.columns)
        r_len    = len(routes)
        s_len    = len(stations)
        _ = compute_charging_proximity_index(routes, stations)
        self.assertEqual(list(routes.columns),   r_cols,  "routes columns changed")
        self.assertEqual(list(stations.columns), s_cols,  "stations columns changed")
        self.assertEqual(len(routes),            r_len,   "routes row count changed")
        self.assertEqual(len(stations),          s_len,   "stations row count changed")

    def test_no_coverage_category_label(self):
        """Unmatched routes get 'No Coverage' in CPI_Coverage_Category."""
        from app.utils.relationships import compute_charging_proximity_index
        result = compute_charging_proximity_index(
            _routes("Atlantis"), _stations("London", 51.5, -0.12)
        )
        self.assertIn("No Coverage", result["CPI_Coverage_Category"].iloc[0])


# ---------------------------------------------------------------------------
# RelationshipMapper CPI tests
# ---------------------------------------------------------------------------

class TestRelationshipMapperCPI(unittest.TestCase):
    """Tests for RelationshipMapper.compute_route_charging_proximity()."""

    def test_end_to_end_real_data(self):
        """
        Full pipeline on real datasets:
        8,000 routes, CPI in [0,1], >70% coverage, required columns present.
        """
        from app.utils.data_loader import data_loader
        from app.utils.relationships import (
            RelationshipMapper, CPI_METHOD_APPROX, CPI_METHOD_NO_COVERAGE,
        )
        dfs    = data_loader.load_all()
        mapper = RelationshipMapper(dfs)
        result = mapper.compute_route_charging_proximity()

        # Not None and correct type
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pd.DataFrame)

        # Row count matches fleet_routes
        self.assertEqual(len(result), 8000)

        # All output columns present
        for col in ("Centroid_Lat", "Centroid_Lon",
                    "Nearest_Station_Distance_km",
                    "Charging_Proximity_Index",
                    "CPI_Coverage_Category",
                    "CPI_Method"):
            self.assertIn(col, result.columns, f"Missing column: {col}")

        # CPI values in [0, 1]
        cpi = result["Charging_Proximity_Index"]
        self.assertTrue((cpi >= 0.0).all(), "CPI must be >= 0")
        self.assertTrue((cpi <= 1.0).all(), "CPI must be <= 1")

        # Coverage: known from data audit
        scored  = (result["CPI_Method"] == CPI_METHOD_APPROX).sum()
        no_cov  = (result["CPI_Method"] == CPI_METHOD_NO_COVERAGE).sum()
        self.assertEqual(scored + no_cov, 8000)
        self.assertGreater(
            scored / 8000 * 100, 70.0,
            f"Expected >70% coverage, got {scored/8000*100:.1f}%",
        )

        # All scored routes have CPI > 0
        scored_rows = result[result["CPI_Method"] == CPI_METHOD_APPROX]
        self.assertTrue(
            (scored_rows["Charging_Proximity_Index"] > 0.0).all(),
            "Scored routes must have CPI > 0",
        )

    def test_returns_none_if_routes_missing(self):
        """Returns None when fleet_routes is not loaded."""
        from app.utils.data_loader import data_loader
        from app.utils.relationships import RelationshipMapper
        dfs     = data_loader.load_all()
        partial = {k: v for k, v in dfs.items() if k != "fleet_routes"}
        result  = RelationshipMapper(partial).compute_route_charging_proximity()
        self.assertIsNone(result)

    def test_returns_none_if_stations_missing(self):
        """Returns None when charging_stations is not loaded."""
        from app.utils.data_loader import data_loader
        from app.utils.relationships import RelationshipMapper
        dfs     = data_loader.load_all()
        partial = {k: v for k, v in dfs.items() if k != "charging_stations"}
        result  = RelationshipMapper(partial).compute_route_charging_proximity()
        self.assertIsNone(result)

    def test_custom_half_life_changes_cpi(self):
        """A shorter half_life_km produces lower average CPI for same routes."""
        from app.utils.data_loader import data_loader
        from app.utils.relationships import RelationshipMapper
        dfs    = data_loader.load_all()
        mapper = RelationshipMapper(dfs)

        result_20  = mapper.compute_route_charging_proximity(half_life_km=20.0)
        result_5   = mapper.compute_route_charging_proximity(half_life_km=5.0)

        mean_20 = result_20["Charging_Proximity_Index"].mean()
        mean_5  = result_5["Charging_Proximity_Index"].mean()
        self.assertGreater(
            mean_20, mean_5,
            "Larger half_life should produce higher mean CPI",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
