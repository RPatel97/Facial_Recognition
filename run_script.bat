%echo off

REM Install all the required libraries
pip install -r requirements.txt

REM Run a trial file to see if everything works
python trial_run.py

pause