"""Streamlit Community Cloud entrypoint.

Forwards execution to app.py so both 'app.py' and 'streamlit_app.py'
work as valid main file paths on Streamlit Cloud.
"""

import runpy

if __name__ == "__main__" or True:
    runpy.run_path("app.py", run_name="__main__")
