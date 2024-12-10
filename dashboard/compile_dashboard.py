# setup.py
import sys
from cx_Freeze import setup, Executable
import os

# Dependencies are automatically detected, but it might need fine-tuning.
build_exe_options = {
    "packages": ["dash", "flask", "pandas", "plotly", "ctypes"],
    "includes": ["ctypes"],
    "include_files": [
        # Include any additional files or directories here
        # For example, if your Dash app uses assets, include the 'assets' folder
        # ("path/to/asset", "asset"),
    ],
    "excludes": [],
    "include_msvcr": True,  # Include the Microsoft Visual C++ Redistributable
}

# Base setup
base = None
if sys.platform == "win32":
    base = "Console"  # Use "Win32GUI" for GUI applications without a console

setup(
    name="ResidualDashboard",
    version="1.0",
    description="A GNSS residual dashboard application.",
    options={"build_exe": build_exe_options},
    executables=[Executable("main.py", base=base, target_name="residual_dashboard.exe")],
)
