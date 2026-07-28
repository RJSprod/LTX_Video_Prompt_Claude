@echo off
rem ============================================================================
rem  Prompt Master Standalone - one-click installer and launcher
rem
rem  Double-click this file, or run it from a command prompt. It finds a Python
rem  to run one_click.py with, downloading a hash-verified private one only if
rem  the machine has none, and then hands over. Everything else - the virtual
rem  environment, the setup questions, the model download - happens in Python.
rem
rem  After the first run you can launch the app directly with:  python app.py
rem ============================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MC_NAME=Miniconda3-py312_25.5.1-0-Windows-x86_64.exe"
set "MC_URL=https://repo.anaconda.com/miniconda/%MC_NAME%"
set "MC_SHA=82cf1382eaa2c92d4f0cbd826a8888fac25080fe8ff3370bae8d0732b80006fb"
set "MC_FILE=installer_files\downloads\%MC_NAME%"

rem --- paths with these characters break cmd, pip or the NSIS bootstrap ------
echo "%CD%" | findstr /C:"!" /C:"%%" /C:"^" /C:"&" /C:"=" >nul && (
    echo.
    echo ERROR: this folder's path contains a special character that breaks the install:
    echo   %CD%
    echo Move the folder somewhere without ! %% ^^ ^& or = and run this file again.
    goto :fail
)

set "PYTHON_CMD="

rem --- 1. the environment a previous run already built, if the Python it was
rem        built with is one the pinned dependencies still cover. A newer one
rem        is not good enough: PySide6 6.8.1 stops at 3.13, so an environment
rem        on 3.14 cannot install requirements.txt and has to be rebuilt. -----
if not exist "installer_files\env\Scripts\python.exe" goto :no_env
call :supported "installer_files\env\Scripts\python.exe"
if not errorlevel 1 (
    set "PYTHON_CMD=installer_files\env\Scripts\python.exe"
    goto :found
)
echo The environment in installer_files\env was built with a Python the pinned
echo dependencies do not support. It will be rebuilt automatically.
echo.
:no_env

rem --- 2. a private Python a previous run already bootstrapped ---------------
if not exist "installer_files\conda\python.exe" goto :no_conda
call :supported "installer_files\conda\python.exe"
if not errorlevel 1 (
    set "PYTHON_CMD=installer_files\conda\python.exe"
    goto :found
)
:no_conda

rem --- 3. the py launcher, which sees installs that are not on PATH ----------
for %%V in (3.13 3.12) do (
    py -%%V -c "import sys" >nul 2>&1
    if not errorlevel 1 (
        set "PYTHON_CMD=py -%%V"
        goto :found
    )
)

rem --- 4. whatever "python" resolves to, if it is new enough to run the
rem        installer. one_click.py needs only 3.8; it upgrades from there. -----
python -c "import sys; sys.exit(0 if sys.version_info>=(3,8) else 1)" >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :found
)

rem --- 5. nothing usable: bootstrap one ---------------------------------------
echo.
echo No suitable Python was found on this machine.
echo A private copy will be installed under installer_files\conda .
echo Your system PATH and registry are not modified.
echo.

if not exist "installer_files\downloads" mkdir "installer_files\downloads"

if not exist "%MC_FILE%" (
    echo Downloading %MC_NAME% ...
    curl.exe -L --fail -# -o "%MC_FILE%" "%MC_URL%"
    if errorlevel 1 (
        echo.
        echo ERROR: download failed. Check your internet connection and try again.
        goto :fail
    )
)

echo Verifying SHA-256 ...
set "ACTUAL="
for /f "skip=1 delims=" %%H in ('certutil -hashfile "%MC_FILE%" SHA256') do (
    if not defined ACTUAL set "ACTUAL=%%H"
)
set "ACTUAL=%ACTUAL: =%"
if /I not "%ACTUAL%"=="%MC_SHA%" (
    echo.
    echo ERROR: checksum mismatch. The download was corrupted or tampered with.
    echo   expected %MC_SHA%
    echo   actual   %ACTUAL%
    del "%MC_FILE%"
    goto :fail
)

echo Installing ...
start /wait "" "%MC_FILE%" /InstallationType=JustMe /NoRegistry=1 /NoShortcuts=1 /S /D=%CD%\installer_files\conda
if not exist "installer_files\conda\python.exe" (
    echo.
    echo ERROR: the private Python did not install. Try running this file again.
    goto :fail
)
set "PYTHON_CMD=installer_files\conda\python.exe"

:found
echo.
%PYTHON_CMD% one_click.py %*
if errorlevel 1 goto :fail

endlocal
exit /b 0

:fail
echo.
pause
endlocal
exit /b 1

rem --- returns 0 when %1 is a Python the pinned dependencies have wheels for.
rem     Kept out of the blocks above so its parentheses and "<" stay quoted.
rem     Keep the range in step with MIN_PYTHON/MAX_PYTHON in one_click.py. ----
:supported
"%~1" -c "import sys; sys.exit(0 if (3,12) <= sys.version_info[:2] < (3,14) else 1)" >nul 2>&1
exit /b %errorlevel%
