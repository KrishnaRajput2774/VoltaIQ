# VoltIQ – AI-Powered Fleet Intelligence Platform

VoltIQ is an enterprise-grade AI-powered platform designed to assist industrial organizations in transitioning from conventional internal combustion engine (ICE) fleets to electric vehicle (EV) fleets. It provides data-driven decision tools for fleet procurement, battery health analysis, carbon offset metrics, and natural language interactive querying.

---

## 🚀 Key Modules
1. **Fleet Electrification Readiness**: Scans ICE fleet mileage, routes, and payload configurations to generate an EV Readiness Index and map conventional vehicles to ideal commercial EV replacements.
2. **Battery Asset Performance Management (APM)**: Monitors battery telemetry (voltage, temperature, capacity) to predict State of Health (SOH), calculate Remaining Useful Life (RUL) in cycles, and issue predictive maintenance alerts.
3. **Carbon Intelligence Tracker**: Calculates current Scope 1 baseline greenhouse gas emissions, models potential carbon savings for EV scenarios, and monitors progress toward sustainability Net-Zero commitments.
4. **AI Fleet Advisor**: A conversational AI interface leveraging LangChain and Large Language Models (LLMs) to allow fleet operators to ask natural language questions about their operations.

---

## 🛠️ Technology Stack
*   **Backend API**: Python, FastAPI, Uvicorn, Pydantic, SQLAlchemy
*   **Machine Learning**: Scikit-Learn, Joblib, NumPy, Pandas
*   **Geospatial Processing**: GeoPandas, Folium, Streamlit-Folium
*   **Frontend Dashboard**: Streamlit, Plotly
*   **AI Agent**: LangChain, OpenAI GPT APIs
*   **Database**: SQLite

---

## 📂 Project Structure

```
VoltIQ/
├── .vscode/               # VS Code IDE configurations
│   ├── settings.json
│   ├── launch.json
│   └── extensions.json
├── app/                   # FastAPI Backend
│   ├── api/               # API Route Handlers
│   │   ├── fleet.py       # Fleet Electrification endpoints
│   │   ├── battery.py     # Battery APM endpoints
│   │   ├── carbon.py      # Carbon Tracker endpoints
│   │   ├── chat.py        # Conversational Agent endpoints
│   │   └── router.py      # Master API router
│   ├── config/            # Settings loader & configurations
│   │   ├── __init__.py
│   │   └── config.py
│   ├── database/          # Database ORM session configuration
│   ├── services/          # Business logic services
│   ├── models/            # SQLAlchemy Database Models
│   ├── schemas/           # Pydantic schemas for request/response
│   ├── utils/             # Helper utilities (e.g. data loaders)
│   ├── agent/             # LangChain AI Advisor Agent
│   ├── static/            # Static files for endpoints
│   ├── templates/         # HTML template files
│   └── main.py            # API Entrypoint
├── datasets/              # Dataset reference directory
├── frontend/              # Streamlit Frontend
│   ├── pages/             # Multi-page dashboard pages
│   │   ├── Fleet.py       # Fleet Readiness Page
│   │   ├── Battery.py     # Battery APM Page
│   │   ├── Carbon.py      # Carbon Tracker Page
│   │   └── AI_Advisor.py  # Conversational Agent Page
│   ├── components/        # Reusable Streamlit components
│   ├── assets/            # Local images, icons, and logos
│   ├── styles/            # Custom CSS styles
│   └── dashboard.py       # Streamlit Landing Page
├── saved_models/          # Directory to store trained ML models
│   ├── battery/           # SOH and RUL model binaries (.pkl)
│   ├── fleet/
│   └── carbon/
├── reports/               # Output folder for generated audits
│   ├── exports/
│   ├── pdf/
│   └── csv/
├── logs/                  # System log files
│   ├── app.log
│   └── errors.log
├── tests/                 # Test suite
│   ├── test_api.py        # Backend endpoint tests
│   ├── test_models.py     # ML inference model tests
│   └── test_data.py       # Data validation tests
├── notebooks/             # Exploratory notebooks (EDA & training)
├── docs/                  # API documentation and schemas
├── .env                   # Environment variable secrets config
├── .gitignore             # Git exclusion rules
├── run.py                 # Multi-process execution manager
└── requirements.txt       # Dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository**:
   ```bash
   git clone <repository_url>
   cd VoltIQ
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup environment variables**:
   Configure the variables inside your `.env` file:
   ```env
   OPENAI_API_KEY=your-api-key-here
   DATABASE_URL=sqlite:///./voltiq.db
   SECRET_KEY=dev_secret_key
   DATASETS_DIR=datasets
   ```

---

## 🏃 Running the Application

To run the full stack (FastAPI backend on port `8000` and Streamlit frontend on port `8501`) concurrently:

```bash
python run.py
```

### Running Components Individually

*   **Start Backend API Only**:
    ```bash
    uvicorn app.main:app --port 8000 --reload
    ```
    API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

*   **Start Streamlit Frontend Only**:
    ```bash
    streamlit run frontend/dashboard.py
    ```
    The web interface will be available at [http://127.0.0.1:8501](http://127.0.0.1:8501).

---

## 🗺️ Roadmap
*   **Phase 1**: Initial project scaffolding and environment configurations (Completed).
*   **Phase 2**: Exploratory Data Analysis (EDA) on battery and route telemetry.
*   **Phase 3**: Train SOH / RUL regression models and write SQL schema representations.
*   **Phase 4**: Develop FastAPI prediction endpoints and database read/write queries.
*   **Phase 5**: Build interactive Plotly and Folium visualizations inside Streamlit.
*   **Phase 6**: Integrate LangChain SQL/Pandas conversational agent for AI Advisor.
