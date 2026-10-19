"""Community Cloud entry point: generate trusted local artifacts once per process."""
from pathlib import Path
import runpy
import subprocess
import sys
import streamlit as st

ROOT = Path(__file__).resolve().parent

@st.cache_resource(show_spinner='Preparing dataset and model...')
def prepare_application():
    # Always build on a fresh server process to avoid stale serialized estimators.
    subprocess.run([sys.executable, str(ROOT/'run_pipeline.py')], cwd=ROOT,
                   check=True, capture_output=True, text=True)
    return True

try:
    prepare_application()
except (OSError, subprocess.CalledProcessError) as exc:
    print(getattr(exc, "stderr", str(exc)), file=sys.stderr)
    st.error('Application initialization failed. The administrator should check dataset availability and server dependency logs.')
    st.stop()
runpy.run_path(str(ROOT/'app.py'), run_name='__main__')
