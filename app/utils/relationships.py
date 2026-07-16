"""
VoltIQ – Dataset Relationships & Charging Proximity Index
==========================================================
Phase 1 (Environment & Data Wrangling) deliverable.

This module defines all logical inter-dataset relationships and implements
the Charging Proximity Index (CPI) computation required by the VoltIQ
roadmap Phase 1:

    "Perform a geospatial join between fleet_route_data locations and
     charging_station locations to compute a Charging Proximity Index
     for each route."

=======================================================================
DATA CONSTRAINT DISCLOSURE — READ BEFORE USE
=======================================================================
The original roadmap assumes a true geospatial join is possible.
A true geospatial join requires latitude and longitude on BOTH sides:
    - fleet_route_data.csv  : has Start_Location (city name string only)
    - ev_charging_station_data.csv : has Latitude, Longitude (verified)

fleet_route_data.csv does NOT contain latitude/longitude coordinates.

Columns present: Vehicle_ID, Vehicle_Type, Route_ID, Date,
    Start_Location, End_Location, Distance_km, Average_Speed_kmh,
    Travel_Time_hrs, Number_of_Stops, Road_Type, Traffic_Level,
    Payload_kg, Idle_Time_min, Driving_Efficiency_Score, Estimated_Fuel_L

Because route coordinates are absent, a TRUE geospatial join is NOT
possible using only the provided datasets and no external services.

=======================================================================
APPROXIMATION IMPLEMENTED — CITY-CENTROID PROXIMITY ESTIMATION
=======================================================================
To fulfil the roadmap intent within the constraint of using ONLY the
provided datasets, the following dataset-constrained approximation is
used:

APPROACH
--------
1. Derive city centroids from ev_charging_station_data.csv by computing
   the arithmetic mean Latitude and Longitude of all stations that share
   the same City name. This converts each city name to an approximate
   geographic coordinate pair using only data already present in the
   dataset.

2. Match each route's Start_Location (city name string) to a city
   centroid. If the city name appears in the charging station dataset,
   its centroid is used as a proxy for the route's geographic origin.

3. Compute the Haversine great-circle distance (km) from the matched
   city centroid to every individual charging station in the dataset,
   then record the minimum (nearest station distance).

4. Convert the nearest-station distance to a Charging_Proximity_Index
   on a [0, 1] scale using an exponential decay:

       CPI = exp(−d / half_life_km)      (half_life_km default: 20)

   At d=0 km:  CPI = 1.0 (station at route origin)
   At d=20 km: CPI ≈ 0.37
   At d=50 km: CPI ≈ 0.08

ASSUMPTIONS
-----------
- The mean lat/lon of stations within a named city is a reasonable
  geographic proxy for fleet routes starting from that city.
- Station coordinates in ev_charging_station_data.csv are accurate.
- City name strings match exactly between datasets (case-sensitive
  after normalisation).

KNOWN LIMITATIONS
-----------------
- 30 of 64 unique route city names have NO corresponding entry in the
  charging station dataset. Routes starting from these 30 cities receive
  CPI = 0.0 and Nearest_Station_Distance_km = NaN. This does NOT mean
  those cities have no charging infrastructure; it means the provided
  dataset does not cover them. Affected cities include: Aberdeen,
  Baltimore, Bath, Cambridge, Cincinnati, Cleveland, Coventry, Dundee,
  Exeter, Hull, Inverness, Leicester, Lexington, Louisville, Luton,
  Memphis, Milwaukee, New Orleans, Newcastle, Newport, Norwich,
  Nottingham, Orlando, Plymouth, Portsmouth, Richmond, Southampton,
  St Louis, Wolverhampton, York.

- CPI coverage: 6,187 of 8,000 route rows (77.3%) can be scored.
  The remaining 22.7% (1,813 rows) receive CPI = 0.0 / No Coverage.

- City centroids derived from a small cluster of stations may not
  represent the true geographic centre of a city.

- This is an approximation of a geospatial join, not a true one.
  A true geospatial join requires latitude/longitude on the route
  records, which are absent from the provided dataset.

HOW TO DISTINGUISH TRUE vs APPROXIMATION
-----------------------------------------
- A route row with CPI_Method = "haversine_centroid_approx" used the
  approximation described above.
- A route row with CPI_Method = "no_coverage" could not be matched to
  any city centroid in the charging station dataset.

Relationships documented
------------------------
Fleet Readiness  <->  Carbon Intelligence   (Vehicle_ID, 1-to-1)
Fleet Readiness  <->  Fleet Routes          (Vehicle_ID, 1-to-many)
Fleet Routes     <->  Charging Stations     (city-centroid Haversine approx)
Charging Sessions <-> Battery               (Vehicle_Model proxy)
Battery          <->  Weather               (Temperature_C enrichment)
Carbon Reference <->  Fleet Readiness       (Make_and_Model lookup)

Usage
-----
    from app.utils.relationships import (
        RelationshipMapper,
        compute_charging_proximity_index,
        haversine_distance_km,
    )

    # Standalone
    result = compute_charging_proximity_index(routes_df, stations_df)

    # Via mapper
    mapper = RelationshipMapper(dfs)
    routes_with_cpi = mapper.compute_route_charging_proximity()
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Earth's mean radius in kilometres (WGS-84 spherical approximation).
EARTH_RADIUS_KM: float = 6371.0

#: Default exponential-decay half-life (km).
#: CPI = exp(-d / DEFAULT_HALF_LIFE_KM).
#: At this distance CPI ≈ 0.37.  20 km gives CPI ≈ 0.50 at ~14 km.
DEFAULT_HALF_LIFE_KM: float = 20.0

#: Method label embedded in the output DataFrame for transparency.
CPI_METHOD_APPROX:      str = "haversine_centroid_approx"
CPI_METHOD_NO_COVERAGE: str = "no_coverage"


# ---------------------------------------------------------------------------
# Relationship registry
# ---------------------------------------------------------------------------
RELATIONSHIP_REGISTRY: List[Dict[str, Any]] = [
    {
        "name":        "Fleet Readiness <-> Carbon Intelligence",
        "left":        "fleet_readiness",
        "right":       "carbon",
        "left_key":    "Vehicle_ID",
        "right_key":   "Vehicle_ID",
        "cardinality": "1-to-1",
        "join_type":   "inner",
        "purpose": (
            "Combines asset operational metrics with emissions data to produce "
            "a single electrification-readiness + carbon-impact view per vehicle."
        ),
    },
    {
        "name":        "Fleet Readiness <-> Fleet Routes",
        "left":        "fleet_readiness",
        "right":       "fleet_routes",
        "left_key":    "Vehicle_ID",
        "right_key":   "Vehicle_ID",
        "cardinality": "1-to-many",
        "join_type":   "left",
        "purpose": (
            "Enriches each vehicle record with its historical route-level "
            "metrics (distance, speed, payload, fuel consumed per trip)."
        ),
    },
    {
        "name":        "Fleet Routes <-> Charging Stations (Approx. Geospatial)",
        "left":        "fleet_routes",
        "right":       "charging_stations",
        "left_key":    "Start_Location (city name — no route lat/lon in dataset)",
        "right_key":   "Latitude / Longitude",
        "cardinality": "many-to-many",
        "join_type":   "haversine_centroid_approx",
        "purpose": (
            "Computes the Charging_Proximity_Index per route using Haversine "
            "distance from a city-centroid proxy (derived from station data) "
            "to the nearest charging station. TRUE geospatial join is not "
            "possible: fleet_route_data.csv contains no lat/lon coordinates."
        ),
    },
    {
        "name":        "Charging Sessions <-> Battery",
        "left":        "charging_sessions",
        "right":       "battery",
        "left_key":    None,
        "right_key":   None,
        "cardinality": "many-to-many",
        "join_type":   "cross-reference",
        "purpose": (
            "Correlates charging behaviour (session rate, duration, SOC delta) "
            "with measured battery degradation trajectories."
        ),
    },
    {
        "name":        "Battery <-> Weather",
        "left":        "battery",
        "right":       "weather",
        "left_key":    None,
        "right_key":   "Temperature_C",
        "cardinality": "many-to-many",
        "join_type":   "feature-enrichment",
        "purpose": (
            "Supplements battery cycle data with ambient climate conditions "
            "to model temperature-induced degradation acceleration."
        ),
    },
    {
        "name":        "Carbon Reference <-> Fleet Readiness",
        "left":        "carbon_reference",
        "right":       "fleet_readiness",
        "left_key":    "Vehicle_Model",
        "right_key":   "Make_and_Model",
        "cardinality": "many-to-1",
        "join_type":   "lookup",
        "purpose": (
            "Enriches fleet records with standardised fuel-economy and "
            "CO2 g/km figures from the government reference table."
        ),
    },
]


# ---------------------------------------------------------------------------
# Haversine distance — pure NumPy, no external libraries
# ---------------------------------------------------------------------------

def haversine_distance_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    """
    Compute the Haversine great-circle distance between point arrays.

    Fully vectorised using NumPy. No GeoPandas or external library required.

    Formula
    -------
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2·atan2(√a, √(1−a))
        d = R·c

    Parameters
    ----------
    lat1, lon1 : np.ndarray
        Coordinates of the first set of points (decimal degrees).
    lat2, lon2 : np.ndarray
        Coordinates of the second set of points (decimal degrees).
        Must be broadcast-compatible with lat1/lon1.

    Returns
    -------
    np.ndarray
        Distances in kilometres.
    """
    lat1_r = np.radians(np.asarray(lat1, dtype=float))
    lat2_r = np.radians(np.asarray(lat2, dtype=float))
    dlat   = np.radians(np.asarray(lat2, dtype=float) - np.asarray(lat1, dtype=float))
    dlon   = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.clip(1.0 - a, 0.0, None)))
    return EARTH_RADIUS_KM * c


def proximity_index_from_distance(
    distance_km: float,
    half_life_km: float = DEFAULT_HALF_LIFE_KM,
) -> float:
    """
    Convert a nearest-station distance (km) to a Charging_Proximity_Index.

    Uses an exponential decay:
        CPI = exp(−distance_km / half_life_km)

    Parameters
    ----------
    distance_km : float
        Distance to the nearest charging station in kilometres.
        Pass ``math.nan``, a negative value, or infinity to get 0.0.
    half_life_km : float
        Decay constant. CPI = exp(-1) ≈ 0.368 at this distance.
        Default: 20 km.

    Returns
    -------
    float
        CPI in [0.0, 1.0]. Returns 0.0 for invalid or infinite distances.

    Raises
    ------
    ValueError
        If half_life_km is not positive.
    """
    if half_life_km <= 0:
        raise ValueError(
            f"half_life_km must be a positive number; got {half_life_km}"
        )
    if not math.isfinite(distance_km) or distance_km < 0:
        return 0.0
    return float(math.exp(-distance_km / half_life_km))


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def compute_charging_proximity_index(
    routes_df: pd.DataFrame,
    stations_df: pd.DataFrame,
    location_col: str = "Start_Location",
    lat_col: str = "Latitude",
    lon_col: str = "Longitude",
    city_col: str = "City",
    half_life_km: float = DEFAULT_HALF_LIFE_KM,
) -> pd.DataFrame:
    """
    Compute the Charging_Proximity_Index for each fleet route.

    This is the Phase 1 mandatory geospatial deliverable defined in the
    VoltIQ roadmap. It implements a **dataset-constrained approximation**
    because ``fleet_route_data.csv`` does not contain lat/lon coordinates.

    See the module-level docstring for full assumptions and limitations.

    IMPORTANT — What this function does NOT do
    -------------------------------------------
    - It does NOT perform a true geospatial join (impossible: route lat/lon absent).
    - It does NOT call any external API, geocoding service, or internet resource.
    - It does NOT create synthetic or fabricated coordinates.
    - It does NOT modify any source CSV file.

    IMPORTANT — What this function DOES do
    ----------------------------------------
    - Derives city centroids purely from ``ev_charging_station_data.csv``.
    - Uses those centroids as geographic proxies for matched route cities.
    - Computes Haversine distances from centroid to every individual station.
    - Returns a scored, categorised, and labelled result DataFrame.

    Output columns added to the returned copy of routes_df
    -------------------------------------------------------
    Centroid_Lat : float
        Mean latitude of charging stations in the matched city.
        NaN for unmatched cities.
    Centroid_Lon : float
        Mean longitude of charging stations in the matched city.
        NaN for unmatched cities.
    Nearest_Station_Distance_km : float
        Haversine distance (km) from city centroid to closest station.
        NaN for unmatched cities.
    Charging_Proximity_Index : float
        Score in [0.0, 1.0]. 1.0 = station at origin, 0.0 = no coverage.
    CPI_Coverage_Category : str
        Human-readable label for the CPI score.
    CPI_Method : str
        'haversine_centroid_approx' — centroid matched, distance computed.
        'no_coverage' — city name not found in charging station dataset.

    Parameters
    ----------
    routes_df : pd.DataFrame
        Fleet route data. Must contain ``location_col``.
    stations_df : pd.DataFrame
        Charging station data. Must contain ``lat_col``, ``lon_col``,
        and ``city_col``.
    location_col : str
        Column in routes_df with the city/location name. Default: 'Start_Location'.
    lat_col : str
        Latitude column in stations_df. Default: 'Latitude'.
    lon_col : str
        Longitude column in stations_df. Default: 'Longitude'.
    city_col : str
        City name column in stations_df for centroid derivation. Default: 'City'.
    half_life_km : float
        Exponential decay constant (km). Default: 20.

    Returns
    -------
    pd.DataFrame
        A copy of routes_df with the six new columns listed above.
        Original DataFrames and CSV files are never modified.

    Raises
    ------
    ValueError
        If required columns are missing from either input DataFrame.
    """
    # --- Input validation --------------------------------------------------
    if location_col not in routes_df.columns:
        raise ValueError(
            f"Column '{location_col}' not found in routes DataFrame. "
            f"Routes has no lat/lon; available columns: {list(routes_df.columns)}"
        )
    for col in (lat_col, lon_col):
        if col not in stations_df.columns:
            raise ValueError(
                f"Column '{col}' not found in stations DataFrame. "
                f"Available: {list(stations_df.columns)}"
            )
    if city_col not in stations_df.columns:
        raise ValueError(
            f"City column '{city_col}' not found in stations DataFrame."
        )

    logger.info(
        "compute_charging_proximity_index() — "
        "APPROXIMATION MODE: route lat/lon absent, using city centroids. "
        "routes=%d, stations=%d",
        len(routes_df), len(stations_df),
    )

    result = routes_df.copy()

    # --- Step 1: Build city centroids from station coordinates only --------
    valid_sta = stations_df.dropna(subset=[lat_col, lon_col, city_col]).copy()
    valid_sta = valid_sta[
        valid_sta[lat_col].between(-90, 90)
        & valid_sta[lon_col].between(-180, 180)
    ]

    city_centroids: Dict[str, Dict[str, float]] = (
        valid_sta.groupby(city_col)[[lat_col, lon_col]]
        .mean()
        .to_dict("index")
    )

    logger.info(
        "Built %d city centroids from %d valid stations. "
        "(Centroids are dataset-derived approximations — not true geocodes.)",
        len(city_centroids), len(valid_sta),
    )

    # --- Step 2: Map each route's Start_Location to a centroid ------------
    centroid_lats: List[float] = []
    centroid_lons: List[float] = []
    cpi_methods:   List[str]   = []

    for city in result[location_col]:
        city_str  = str(city).strip() if pd.notna(city) else ""
        centroid  = city_centroids.get(city_str)
        if centroid:
            centroid_lats.append(centroid[lat_col])
            centroid_lons.append(centroid[lon_col])
            cpi_methods.append(CPI_METHOD_APPROX)
        else:
            centroid_lats.append(float("nan"))
            centroid_lons.append(float("nan"))
            cpi_methods.append(CPI_METHOD_NO_COVERAGE)

    result["Centroid_Lat"] = centroid_lats
    result["Centroid_Lon"] = centroid_lons
    result["CPI_Method"]   = cpi_methods

    matched_count = sum(1 for m in cpi_methods if m == CPI_METHOD_APPROX)
    logger.info(
        "City centroid matched for %d / %d routes (%.1f%%). "
        "%d routes will receive CPI=0.0 (city not in station dataset).",
        matched_count, len(result),
        100.0 * matched_count / max(len(result), 1),
        len(result) - matched_count,
    )

    # --- Step 3: Vectorised Haversine to nearest station ------------------
    sta_lats = valid_sta[lat_col].to_numpy(dtype=float)
    sta_lons = valid_sta[lon_col].to_numpy(dtype=float)

    nearest_distances: List[float] = []

    if len(sta_lats) == 0:
        logger.warning(
            "No valid station coordinates in dataset. All CPI = 0.0."
        )
        nearest_distances = [float("nan")] * len(result)
    else:
        for r_lat, r_lon in zip(result["Centroid_Lat"], result["Centroid_Lon"]):
            if math.isnan(r_lat) or math.isnan(r_lon):
                nearest_distances.append(float("nan"))
            else:
                dists = haversine_distance_km(
                    np.full(len(sta_lats), r_lat),
                    np.full(len(sta_lons), r_lon),
                    sta_lats,
                    sta_lons,
                )
                nearest_distances.append(float(dists.min()))

    result["Nearest_Station_Distance_km"] = nearest_distances

    # --- Step 4: Convert to Charging_Proximity_Index ----------------------
    result["Charging_Proximity_Index"] = [
        proximity_index_from_distance(d, half_life_km)
        for d in nearest_distances
    ]

    # --- Step 5: Human-readable coverage category -------------------------
    def _category(cpi: float, method: str) -> str:
        if method == CPI_METHOD_NO_COVERAGE:
            return "No Coverage (city not in station dataset)"
        if cpi >= 0.8:
            return "Excellent (>=0.8)"
        if cpi >= 0.6:
            return "Good (0.6-0.8)"
        if cpi >= 0.4:
            return "Moderate (0.4-0.6)"
        if cpi >= 0.2:
            return "Sparse (0.2-0.4)"
        return "Poor (<0.2)"

    result["CPI_Coverage_Category"] = [
        _category(cpi, meth)
        for cpi, meth in zip(
            result["Charging_Proximity_Index"], result["CPI_Method"]
        )
    ]

    # Summary log
    scored = result[result["CPI_Method"] == CPI_METHOD_APPROX]
    logger.info(
        "Charging_Proximity_Index summary — "
        "scored=%d  no_coverage=%d  mean_cpi=%.3f  "
        "mean_nearest_km=%.1f  max_nearest_km=%.1f",
        len(scored),
        len(result) - len(scored),
        result["Charging_Proximity_Index"].mean(),
        scored["Nearest_Station_Distance_km"].mean() if len(scored) else float("nan"),
        scored["Nearest_Station_Distance_km"].max()  if len(scored) else float("nan"),
    )

    return result


# ---------------------------------------------------------------------------
# RelationshipMapper
# ---------------------------------------------------------------------------

class RelationshipMapper:
    """
    Performs and documents all VoltIQ dataset joins.

    Includes the Phase 1 mandatory Charging Proximity Index via
    ``compute_route_charging_proximity()``.

    Parameters
    ----------
    dfs : dict[str, pd.DataFrame]
        Loaded DataFrames keyed by registry key.
    """

    def __init__(self, dfs: Dict[str, pd.DataFrame]) -> None:
        self._dfs = dfs

    # ------------------------------------------------------------------
    # Phase 1 mandatory deliverable
    # ------------------------------------------------------------------

    def compute_route_charging_proximity(
        self,
        half_life_km: float = DEFAULT_HALF_LIFE_KM,
    ) -> Optional[pd.DataFrame]:
        """
        Compute the Charging_Proximity_Index for all fleet routes.

        APPROXIMATION: fleet_route_data.csv has no lat/lon coordinates.
        City centroids are derived from ev_charging_station_data.csv.
        See ``compute_charging_proximity_index()`` for full disclosure.

        Parameters
        ----------
        half_life_km : float
            Exponential decay constant. Default: 20 km.

        Returns
        -------
        pd.DataFrame or None
            Routes with CPI columns, or None if datasets are missing.
        """
        if "fleet_routes" not in self._dfs:
            logger.error(
                "compute_route_charging_proximity(): "
                "'fleet_routes' dataset not loaded."
            )
            return None
        if "charging_stations" not in self._dfs:
            logger.error(
                "compute_route_charging_proximity(): "
                "'charging_stations' dataset not loaded."
            )
            return None

        return compute_charging_proximity_index(
            routes_df=self._dfs["fleet_routes"],
            stations_df=self._dfs["charging_stations"],
            half_life_km=half_life_km,
        )

    # ------------------------------------------------------------------
    # Structural joins
    # ------------------------------------------------------------------

    def join_fleet_carbon(self) -> Optional[pd.DataFrame]:
        """
        Inner join Fleet Readiness with Carbon Intelligence on Vehicle_ID.

        Returns
        -------
        pd.DataFrame or None
        """
        return self._safe_join(
            "fleet_readiness", "carbon",
            left_key="Vehicle_ID", right_key="Vehicle_ID",
            how="inner", suffixes=("_fleet", "_carbon"), name="fleet_carbon",
        )

    def join_fleet_routes(self) -> Optional[pd.DataFrame]:
        """
        Left join Fleet Readiness with Fleet Routes on Vehicle_ID.

        Returns
        -------
        pd.DataFrame or None
        """
        return self._safe_join(
            "fleet_readiness", "fleet_routes",
            left_key="Vehicle_ID", right_key="Vehicle_ID",
            how="left", suffixes=("_fleet", "_route"), name="fleet_routes_joined",
        )

    def join_routes_charging_stations(self) -> Optional[pd.DataFrame]:
        """
        City-name string join between Fleet Routes and Charging Stations.

        .. note::
            This is a simple text match on city name only. Use
            ``compute_route_charging_proximity()`` for the proper
            Haversine-based Charging_Proximity_Index computation.

        Returns
        -------
        pd.DataFrame or None
        """
        return self._safe_join(
            "fleet_routes", "charging_stations",
            left_key="Start_Location", right_key="City",
            how="left", suffixes=("_route", "_station"), name="routes_charging_city",
        )

    def join_fleet_carbon_reference(self) -> Optional[pd.DataFrame]:
        """
        Lookup join enriching Fleet Readiness with CO2 reference data.

        Returns
        -------
        pd.DataFrame or None
        """
        return self._safe_join(
            "fleet_readiness", "carbon_reference",
            left_key="Make_and_Model", right_key="Vehicle_Model",
            how="left", suffixes=("_fleet", "_ref"), name="fleet_co2_reference",
        )

    # ------------------------------------------------------------------
    # Documentation helpers
    # ------------------------------------------------------------------

    def describe_all(self) -> List[Dict[str, Any]]:
        """Return the full relationship registry."""
        return RELATIONSHIP_REGISTRY

    def print_relationships(self) -> None:
        """Pretty-print all registered dataset relationships."""
        print(f"\n{'='*70}")
        print("  VoltIQ -- Dataset Relationship Map")
        print(f"{'='*70}")
        for rel in RELATIONSHIP_REGISTRY:
            print(f"\n  [{rel['join_type']}]  {rel['name']}")
            print(f"     Cardinality : {rel['cardinality']}")
            if rel["left_key"]:
                print(f"     Join keys   : {rel['left_key']} <-> {rel['right_key']}")
            print(f"     Purpose     : {rel['purpose']}")

    def availability_check(self) -> Dict[str, bool]:
        """
        Check which relationship joins can be executed.

        Returns
        -------
        dict[str, bool]
        """
        status: Dict[str, bool] = {}
        for rel in RELATIONSHIP_REGISTRY:
            both = rel["left"] in self._dfs and rel["right"] in self._dfs
            status[rel["name"]] = both
            logger.info(
                "[%s]  %s (left=%s, right=%s)",
                "OK" if both else "MISSING",
                rel["name"], rel["left"], rel["right"],
            )
        return status

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _safe_join(
        self,
        left_key_ds: str,
        right_key_ds: str,
        left_key: str,
        right_key: str,
        how: str = "inner",
        suffixes: Tuple[str, str] = ("_x", "_y"),
        name: str = "",
    ) -> Optional[pd.DataFrame]:
        """Execute a structural join with validation and logging."""
        if left_key_ds not in self._dfs:
            logger.error("Cannot join '%s': dataset not loaded.", left_key_ds)
            return None
        if right_key_ds not in self._dfs:
            logger.error("Cannot join '%s': dataset not loaded.", right_key_ds)
            return None

        left_df  = self._dfs[left_key_ds]
        right_df = self._dfs[right_key_ds]

        if left_key not in left_df.columns:
            logger.error("Join key '%s' not found in '%s'.", left_key, left_key_ds)
            return None
        if right_key not in right_df.columns:
            logger.error("Join key '%s' not found in '%s'.", right_key, right_key_ds)
            return None

        try:
            merged = pd.merge(
                left_df, right_df,
                left_on=left_key, right_on=right_key,
                how=how, suffixes=suffixes,
            )
            logger.info(
                "Join '%s': %d rows x %d cols", name, len(merged), len(merged.columns)
            )
            return merged
        except Exception as exc:
            logger.error("Join '%s' failed: %s", name, exc)
            return None
