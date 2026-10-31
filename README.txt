ISL Learning Assistant - Frontend starter
=========================================

This starter provides:
1. desktop_app.py - a light-themed Tkinter desktop app using your existing GRU model
2. web_app.py + templates/index.html - a browser interface that sends webcam frames to a Python backend
3. recognizer.py - shared model and MediaPipe logic

IMPORTANT
---------
Copy these three existing project files into this folder:
- isl_gru_model.h5
- actions.pickle
- hand_landmarker.task

Keep your original project untouched. Try this in a COPY/new project first.

INSTALL (PyCharm terminal, with the correct interpreter active)
--------------------------------------------------------------
python -m pip install -r requirements.txt

DESKTOP APP
-----------
python desktop_app.py

WEB APP (local test)
--------------------
python web_app.py
Then open http://127.0.0.1:5000 in your browser and allow camera access.

DEPLOYMENT NOTE
---------------
The website uses a server-side Python model. To deploy it, upload the project files
and the three model/assets files to a Python hosting service that supports TensorFlow,
MediaPipe and sufficient memory. The browser camera requires HTTPS on a deployed site.
Some free hosting plans sleep or have memory/build limits; test the hosting provider
before promising a live demo. For a local demo, localhost is suitable.

MODEL BEHAVIOUR
---------------
This starter preserves the model's basic 30-frame rolling-window approach, 0.70
confidence threshold, 3 stable predictions, 2-second result display, and reset after
15 frames with no detected hands. It is not a fix for continuous-sign boundary confusion.
The website backend currently uses one shared recognizer state; it is intended for a
single-user demo, not multiple visitors at once. For public multi-user deployment,
use per-session recognizer state and a production setup.

If TensorFlow / MediaPipe versions conflict, use the versions from your working
project's environment rather than blindly upgrading packages.
