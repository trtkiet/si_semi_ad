@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "FEATURE_DIM=32"
set "N=1000"

for %%D in (2) do (
  set "MODEL_NAME=deepsad_delta_%%D_%%N%%_d%FEATURE_DIM%"

  echo Training model for delta=%%D
  uv run python "%SCRIPT_DIR%train.py" --name !MODEL_NAME! --delta %%D --d %FEATURE_DIM% --anomaly-rate 0.05 --known-label-rate 0.1
  if errorlevel 1 (
    echo Training failed on delta=%%D
    exit /b 1
  )

  echo Running run_si_experiment.py with delta=%%D
  uv run python "%SCRIPT_DIR%run_si_experiment.py" --delta %%D --d %FEATURE_DIM% --anomaly-rate 0.05 --known-label-rate 0.1 --model-name !MODEL_NAME! --n %N%
  if errorlevel 1 (
    echo Experiment failed on delta=%%D
    exit /b 1
  )
)

echo All runs completed.
endlocal
