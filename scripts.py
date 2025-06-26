import subprocess
from flask import Response, redirect

def function_to_liveData():
    subprocess.Popen(['streamlit', 'run', 'quantviz_app.py'])
    return Response(status=204)

def function_to_cryptoData():
    subprocess.Popen(['streamlit', 'run', 'live_monitor.py'])
    subprocess.Popen(['streamlit', 'run', 'streamlit_viz.py'])
    return redirect("http://localhost:8502", code=302)

