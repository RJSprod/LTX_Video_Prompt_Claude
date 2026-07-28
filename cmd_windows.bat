@echo off
rem Open a command prompt with the application's environment active, for running
rem pytest or python directly against the installed dependencies.
setlocal
cd /d "%~dp0"
if not exist "installer_files\env\Scripts\activate.bat" (
    echo The environment has not been created yet. Run start_windows.bat first.
    pause
    exit /b 1
)
call "installer_files\env\Scripts\activate.bat"
echo Environment active. "python app.py" launches the application.
cmd /k
