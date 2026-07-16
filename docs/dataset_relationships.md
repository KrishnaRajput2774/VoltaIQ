# VoltIQ – Dataset Relationship Map

**Module**: `app/utils/relationships.py`  
**Phase**: Phase 1 — Environment & Data Wrangling  
**Generated**: 2026-07-14

---

## Data Constraint Disclosure

> [!IMPORTANT]
> The VoltIQ roadmap Phase 1 requires a geospatial join between `fleet_route_data.csv`
> and `ev_charging_station_data.csv` to compute a **Charging Proximity Index** per route.
>
> **A true geospatial join is NOT possible using only the provided datasets.**
>
> `fleet_route_data.csv` contains **city name strings only** (`Start_Location`,
> `End_Location`) — it has **no latitude or longitude columns**.
> This was verified by direct inspection of the dataset.
>
> Fleet route columns confirmed: `Vehicle_ID`, `Vehicle_Type`, `Route_ID`, `Date`,
> `Start_Location`, `End_Location`, `Distance_km`, `Average_Speed_kmh`,
> `Travel_Time_hrs`, `Number_of_Stops`, `Road_Type`, `Traffic_Level`,
> `Payload_kg`, `Idle_Time_min`, `Driving_Efficiency_Score`, `Estimated_Fuel_L`
>
> `ev_charging_station_data.csv` **does** contain `Latitude` and `Longitude` per station.

> [!NOTE]
> The implementation uses a **dataset-constrained approximation**: city centroids are
> derived from the charging station dataset and used as proxy coordinates for route
> origins. No external APIs, geocoding services, synthetic coordinates, or fabricated
> data are used.

---

## True vs Approximation — Distinction

| | True Geospatial Join | This Implementation |
|---|---|---|
| **Route coordinates** | Exact lat/lon per route record | City centroid derived from station data |
| **Station coordinates** | Exact lat/lon per station | Exact lat/lon per station ✅ |
| **Distance calculation** | Haversine or great-circle ✅ | Haversine ✅ |
| **Requires external data** | Yes (geocoding API) | No ✅ |
| **Uses only provided datasets** | Not always | Yes ✅ |
| **Coverage** | 100% of routes | 77.3% of routes (34/64 cities matched) |

---

## Charging Proximity Index — Calculation Method

### Step 1 — City Centroid Derivation
From `ev_charging_station_data.csv`, compute the arithmetic mean `Latitude` and
`Longitude` of all stations sharing the same `City` name. Stations with invalid or
out-of-range coordinates are excluded before averaging.

```
centroid(city) = mean(Latitude), mean(Longitude)
               for all stations WHERE City = city
               AND Latitude BETWEEN -90 AND 90
               AND Longitude BETWEEN -180 AND 180
```

**Result**: 40 unique city centroids derived from 8,206 charging stations.

### Step 2 — Route City Matching
Match each route's `Start_Location` (city name string) to its centroid.

- **Matched** (34 cities, 6,187 route rows — 77.3%): centroid coordinates assigned.
- **Unmatched** (30 cities, 1,813 route rows — 22.7%): no centroid available.

### Step 3 — Haversine Distance to Nearest Station
For each matched route, compute the great-circle distance from its city centroid
to **every** individual charging station using the Haversine formula:

```
a = sin²(Δlat/2) + cos(lat₁)·cos(lat₂)·sin²(Δlon/2)
c = 2·atan2(√a, √(1−a))
d = R·c      (R = 6371 km)
```

Select the **minimum** distance: `Nearest_Station_Distance_km`.

### Step 4 — Proximity Index
Convert the nearest-station distance to a `Charging_Proximity_Index` score:

```
CPI = exp(−d / 20)
```

where 20 km is the default half-life distance.

| Distance | CPI |
|---|---|
| 0 km (station at origin) | 1.000 |
| 5 km | 0.779 |
| 14 km | ≈ 0.500 |
| 20 km | ≈ 0.368 |
| 50 km | ≈ 0.082 |
| >100 km | < 0.007 |

### Step 5 — Coverage Category

| CPI Range | Label |
|---|---|
| ≥ 0.8 | Excellent (>=0.8) |
| 0.6 – 0.8 | Good (0.6-0.8) |
| 0.4 – 0.6 | Moderate (0.4-0.6) |
| 0.2 – 0.4 | Sparse (0.2-0.4) |
| > 0 and < 0.2 | Poor (<0.2) |
| = 0 (no match) | No Coverage (city not in station dataset) |

---

## Coverage Audit

### Cities with Station Data (34 of 64 — CPI Computed)
Atlanta, Birmingham, Boston, Bristol, Cardiff, Charlotte, Chicago, Columbus,
Dallas, Denver, Detroit, Edinburgh, Glasgow, Houston, Indianapolis, Kansas City,
Las Vegas, Leeds, Liverpool, London, Los Angeles, Manchester, Miami, Minneapolis,
Nashville, New York, Philadelphia, Phoenix, Pittsburgh, Portland, San Francisco,
Seattle, Sheffield, Washington DC

### Cities WITHOUT Station Data (30 of 64 — CPI = 0.0)
Aberdeen, Baltimore, Bath, Cambridge, Cincinnati, Cleveland, Coventry, Dundee,
Exeter, Hull, Inverness, Leicester, Lexington, Louisville, Luton, Memphis,
Milwaukee, New Orleans, Newcastle, Newport, Norwich, Nottingham, Orlando,
Plymouth, Portsmouth, Richmond, Southampton, St Louis, Wolverhampton, York

> [!WARNING]
> CPI = 0.0 for these 30 cities does NOT mean those locations have no charging
> infrastructure. It means the provided `ev_charging_station_data.csv` does not
> include stations for those cities. The data limitation affects 1,813 of 8,000
> route rows (22.7%).

---

## Output Columns Added to Routes DataFrame

| Column | Type | Description |
|---|---|---|
| `Centroid_Lat` | float | Mean latitude of matched city's stations. NaN if unmatched. |
| `Centroid_Lon` | float | Mean longitude of matched city's stations. NaN if unmatched. |
| `Nearest_Station_Distance_km` | float | Haversine distance to closest station. NaN if unmatched. |
| `Charging_Proximity_Index` | float | Score in [0, 1]. Higher = better access. 0 = no coverage. |
| `CPI_Coverage_Category` | str | Human-readable coverage label. |
| `CPI_Method` | str | `haversine_centroid_approx` or `no_coverage`. |

---

## Dataset Relationship Registry

```
Fleet Readiness  <->  Carbon Intelligence   [1-to-1,    inner join,    Vehicle_ID]
Fleet Readiness  <->  Fleet Routes          [1-to-many, left join,     Vehicle_ID]
Fleet Routes     <->  Charging Stations     [many-many, haversine approx, Start_Location -> Lat/Lon]
Charging Sessions <-> Battery               [many-many, cross-reference, proxy join]
Battery          <->  Weather               [many-many, feature-enrichment, Temperature_C]
Carbon Reference <->  Fleet Readiness       [many-to-1, lookup join,   Make_and_Model]
```

### Entity Relationship Diagram (ASCII)

```
fleet_electrification_readiness
        |          |
   Vehicle_ID  Make_and_Model ---------> carbon_emissions_reference
        |
   Vehicle_ID
        |
fleet_route_data
   Start_Location (city name)
        |
        | [city centroid approximation — no route lat/lon available]
        |
        v
ev_charging_station_data (Latitude, Longitude)
        |
ev_charging_sessions --[proxy]--> ev_battery_degradation
                                         |
                                    Temperature_C
                                         |
                                    weather_data
        |
fleet_carbon_intelligence <------- fleet_electrification_readiness
```

---

## Implementation Notes

- **No source CSV is ever modified.** All operations return new in-memory DataFrames.
- **No external API or geocoding service is used.** All coordinates come from the provided datasets only.
- **No synthetic or fabricated coordinates are created.** Routes with no matching city receive `CPI = 0.0` and `CPI_Method = no_coverage`.
- The `CPI_Method` column in every output row distinguishes between:
  - `haversine_centroid_approx` — city centroid matched, Haversine distance computed.
  - `no_coverage` — city not found in station dataset; CPI forced to 0.0.

---

## Usage

```python
from app.utils.data_loader import data_loader
from app.utils.relationships import RelationshipMapper, compute_charging_proximity_index

# Load the datasets
dfs = data_loader.load_all()

# Via RelationshipMapper (recommended)
mapper = RelationshipMapper(dfs)
routes_with_cpi = mapper.compute_route_charging_proximity()

# Standalone function
routes_with_cpi = compute_charging_proximity_index(
    routes_df=dfs["fleet_routes"],
    stations_df=dfs["charging_stations"],
    half_life_km=20.0,   # default: 20 km
)

# Inspect results
print(routes_with_cpi[["Route_ID", "Start_Location",
                         "Nearest_Station_Distance_km",
                         "Charging_Proximity_Index",
                         "CPI_Coverage_Category",
                         "CPI_Method"]].head(10))
```
