import subprocess
import os

os.environ['STREAMLIT_CONFIG_FILE'] = '.streamlit/config.toml'
os.environ['STREAMLIT_BROWSER_GATHER_USAGE_STATS'] = 'false'

subprocess.run([
    'D:\\kw\\vt\\.venv\\Scripts\\streamlit.exe',
    'run',
    'app.py',
    '--server.port', '8501'
])
