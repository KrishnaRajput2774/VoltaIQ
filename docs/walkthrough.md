# VoltIQ Phase 3, 4, & 5 Walkthrough

This walkthrough summarizes the implementation details and verification results for Phase 3 (Backend API), Phase 4 (Streamlit Visualization), and Phase 5 (Conversational AI Advisor).

---

## 1. Backend API (Phase 3)

### Core Schema Updates
- **[`app/schemas/payloads.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/schemas/payloads.py)**: Added full input/output Pydantic schemas for versioned predict routes, RUL queries, Comparative Carbon analysis, System diagnostics, and registry entries. Completely compliant with Pydantic v2.
- **[`app/schemas/__init__.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/schemas/__init__.py)**: Exported all new schemas for clean system imports.

### Business & Model Services
- **[`app/services/models_service.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/services/models_service.py)**: Exposes real inference functions using loaded serialised model Pipelines (StandardScaler + estimators) for Battery SOH, Battery RUL, and Fleet Readiness. Exposes fallback mathematical models in case pickles are missing. Exposes registry metadata retrieval.
- **[`app/services/carbon_service.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/services/carbon_service.py)**: Implements dynamic calculation of cumulative metrics (CO2 baseline, EV scenario savings, and net-zero progress) from fleet carbon datasets, and comparative vehicle carbon emissions analysis.
- **[`app/api/dependencies.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/api/dependencies.py)**: Standardizes ModelService and CarbonService instances via FastAPI dependency injection.

### Routing & Middleware Setup
- **[`app/middleware.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/middleware.py)**: Implements `RequestIDMiddleware` (generates and propagates unique UUID header log keys) and `RequestLoggingMiddleware` (records response status, HTTP route method, and execution time).
- **[`app/main.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/main.py)**: Updated to use lifespan events manager for loading models at startup, mounts timing/logging middlewares, adds global exception handlers (validation 422 errors and server 500 exceptions), and configures aggregated versioned `/api/v1` and unversioned `/api` routers.
- **[`app/api/router.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/api/router.py)**: Exposes versioned `/api/v1` routers including models registry and system diagnostics.
- **[`app/api/system.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/api/system.py)**: Implements health checks, uptime tracking, model load diagnostics, and dataset availability.
- **[`app/api/models_router.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/api/models_router.py)**: Exposes details from model registries.

### API Contract & Testing
- **[`docs/openapi.json`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/docs/openapi.json)**: OpenAPI contract generated and saved.
- **[`tests/test_api.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/tests/test_api.py)**: Integrated 20 comprehensive API test cases.
- **[`tests/test_models.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/tests/test_models.py)**: Updated test mocks to support fallback baseline math testing correctly when pickles exist on disk.

---

## 2. Streamlit Dashboard Frontend (Phase 4)

The frontend is fully connected to the FastAPI backend and uses Plotly and Vanilla CSS for premium visuals.

### API Client Service Layer
- **[`frontend/services/api_client.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/frontend/services/api_client.py)**: Implements reusable API methods with timeout management, incremental backoff retries (up to 3 times), and error status capturing. Exports standard UI widgets for empty states and service error cards.

### Landing Page
- **[`frontend/dashboard.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/frontend/dashboard.py)**: Features real-time KPI metrics fetched dynamically from backend endpoints (Total Vehicles, EV Readiness %, Cumulative CO2 Avoided, Server Health Status). Features Outfits and Jakarta Sans typography with responsive columns.

### Fleet Electrification Page
- **[`frontend/pages/Fleet.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/frontend/pages/Fleet.py)**: 
  - Displays transition segment donut charts.
  - Lists OEM replacement recommendation lists.
  - Renders a vehicle ID lookup form.
  - Hosts a custom operational parameters estimator form sending inputs to the trained LinearRegression pipeline to return readiness indexes.

### Battery APM Page
- **[`frontend/pages/Battery.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/frontend/pages/Battery.py)**:
  - Battery ID search box connected to the prediction lookup API.
  - Real-time Plotly charts plotting capacity degradation curves directly from loaded local CSV telemetry logs.
  - Form to enter custom battery readings and return predicted SOH, RUL, and color-coded warning alert zones.

### Carbon Tracker Page
- **[`frontend/pages/Carbon.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/frontend/pages/Carbon.py)**:
  - Displays baseline vs EV scenario CO2 emissions comparisons using Plotly bar charts.
  - Plots progress toward target net-zero parameters using a radial gauge.
  - Single-vehicle offsets comparative calculator returning emissions saved.

---

## 3. AI Fleet Advisor (Phase 5)

### LangChain Conversational Agent
- **[`app/agent/fleet_advisor.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/agent/fleet_advisor.py)**:
  - Implements 7 structured tools using the `@tool` decorator to compute overall metrics, perform searches, predict cell telemetry, and calculate carbon footprints.
  - Resolves tool actions dynamically using a tool-execution loop with OpenAI.
  - Supports conversational memory via context history messages.
  - Fallback: Rules-based pandas matcher that queries local datasets to answer standard electrification, battery degradation, and carbon offset questions when offline.
- **[`app/api/chat.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/app/api/chat.py)**: Bridges API payloads to the agent query engine.
- **[`frontend/pages/AI_Advisor.py`](file:///c:/Users/pati7/Desktop/AI%20for%20Industrial%20EV%20Supply%20Chain%20&%20Asset%20Intelligence/VoltIQ/frontend/pages/AI_Advisor.py)**: Features a conversational window supporting memory threads, scrollable logs, and referenced source badges.

---

## 4. Verification & Execution

- Syntax compiled and checked:
  ```bash
  python -m py_compile app/**/*.py frontend/**/*.py
  ```
  Result: **0 errors**.
- Full test execution:
  ```bash
  python -m pytest tests/ -v
  ```
  Result: **107 passed, 0 failed**.

---

_VoltIQ walkthrough Document_
