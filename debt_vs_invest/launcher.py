# debt_vs_invest/launcher.py
from __future__ import annotations

import os
import sys

from streamlit.web import cli as stcli


def main() -> None:
    app_path = os.path.join(os.path.dirname(__file__), "app.py")
    sys.argv = ["streamlit", "run", app_path]
    raise SystemExit(stcli.main())
