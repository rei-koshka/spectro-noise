import sys
import subprocess

def launch_editor(app_name: str, path: str) -> None:
    if sys.platform.startswith("win"):
        command = f"start {app_name} \"{path}\""
    elif sys.platform.startswith("darwin"):
        command = f"open -W -a {app_name} '{path}'"
    elif sys.platform.startswith("linux"):
        command = f"{app_name} '{path}'"
    else:
        raise Exception(f"Unsupported OS for launching `{app_name}`")

    subprocess.Popen(command, shell=True).wait()
