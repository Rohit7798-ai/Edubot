# medibot.py - redirected to edubot for backwards compatibility
"""
EduBot: Customized Educational Assistant
This file imports and executes edubot.py so existing run scripts remain compatible.
"""
import runpy
from pathlib import Path

edubot_script = Path(__file__).resolve().parent / "edubot.py"
runpy.run_path(str(edubot_script), run_name="__main__")