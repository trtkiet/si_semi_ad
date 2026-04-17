@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "FEATURE_DIM=10"
set "N=150"
set "METHOD=normal"
set "DEBUG_SEED=%~1"

if not "%DEBUG_SEED%"=="" (
  echo Debug mode enabled: running only seed=%DEBUG_SEED%
)

for %%D in (2) do (
  set "MODEL_NAME=deepsad_delta_%%D_%N%_d%FEATURE_DIM%"

  if exist "%SCRIPT_DIR%models\!MODEL_NAME!_model.pth" (
    echo Model !MODEL_NAME! already exists, skipping training.
  ) else (
    echo Training model for delta=%%D
    uv run python "%SCRIPT_DIR%train.py" --name !MODEL_NAME! --delta %%D --d %FEATURE_DIM% --anomaly-rate 0.05 --h-dims 64,32
    if errorlevel 1 (
      echo Training failed on delta=%%D
      exit /b 1
    )
  )

  echo Running run_si_experiment.py with delta=%%D
  if "%DEBUG_SEED%"=="" (
    uv run python "%SCRIPT_DIR%run_si_experiment.py" --delta %%D --d %FEATURE_DIM% --anomaly-rate 0.00 --model-name !MODEL_NAME! --n %N% --n-seeds 1000 --device cuda --h-dims 64,32 --method %METHOD%
  ) else (
    uv run python "%SCRIPT_DIR%run_si_experiment.py" --delta %%D --d %FEATURE_DIM% --anomaly-rate 0.00 --model-name !MODEL_NAME! --n %N% --seed %DEBUG_SEED% --device cuda --h-dims 64,32 --method %METHOD%
  )
  if errorlevel 1 (
    echo Experiment failed on delta=%%D
    exit /b 1
  )
)

echo All runs completed.
endlocal
