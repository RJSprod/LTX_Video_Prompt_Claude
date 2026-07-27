@echo off
rem Reinstall the Python dependencies from requirements.txt without touching the
rem downloaded model or runtime. Use after pulling a new version of this repo.
setlocal
cd /d "%~dp0"
call start_windows.bat --update --no-launch
endlocal
