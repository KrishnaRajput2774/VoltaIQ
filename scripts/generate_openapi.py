"""
VoltIQ -- OpenAPI Spec Exporter
===============================
Saves the FastAPI auto-generated OpenAPI JSON specification to disk
for API contract validation, documentation portal hosting, or client SDK generation.

Run:
    python scripts/generate_openapi.py
"""
import json
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fastapi.openapi.utils import get_openapi
from app.main import app

def main():
    docs_dir = Path("docs")
    docs_dir.mkdir(exist_ok=True)
    
    # Generate OpenAPI schema dictionary
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    output_path = docs_dir / "openapi.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(openapi_schema, f, indent=2)
        
    print(f"[SUCCESS] OpenAPI Spec exported to {output_path.resolve()}")

if __name__ == "__main__":
    main()
