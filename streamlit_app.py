"""
Streamlit Community Cloud Deployment Entry Point
Redirects execution to the full dashboard application in dashboard/app.py
"""
import sys
import runpy
from pathlib import Path

# Ensure root directory is on PYTHONPATH
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Launch dashboard/app.py
dashboard_app = ROOT / "dashboard" / "app.py"
runpy.run_path(str(dashboard_app), run_name="__main__")
