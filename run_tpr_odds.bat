@echo off
setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "DATA_ROOT=data"
set "METHOD=normal"
set "KNOWN_LABEL_RATE=0.1"
set "TARGET_P_VALUES=500"
set "H_DIMS=64,32"
set "REP_DIM=8"
set "DEBUG_SEED=%~1"

if not "!DEBUG_SEED!"=="" (
  echo Debug mode enabled: running a single seed=!DEBUG_SEED!
)

for %%D in ("arrhythmia" "cardio" "satellite" "satimage-2" "thyroid") do (
  set "DATASET=%%~D"
  set "MODEL_NAME=deepsad_odds_!DATASET!"

  echo.
  echo ====================================================
  echo [TPR] Dataset !DATASET!
  echo ====================================================

  if exist "%SCRIPT_DIR%models\!MODEL_NAME!_model.pth" (
    echo Model !MODEL_NAME! already exists, skipping training.
  ) else (
    echo Training model for !DATASET!
    uv run python "%SCRIPT_DIR%train.py" --name !MODEL_NAME! --dataset-name "!DATASET!" --data-root "%DATA_ROOT%"  --h-dims %H_DIMS% --rep-dim %REP_DIM% --seed 0
    if errorlevel 1 (
      echo Training failed on dataset=!DATASET!
      exit /b 1
    )
  )

  echo Running TPR SI experiment for !DATASET!
  if "!DEBUG_SEED!"=="" (
    uv run python "%SCRIPT_DIR%run_si_experiment.py" --dataset-name "!DATASET!" --data-root "%DATA_ROOT%"  --model-name !MODEL_NAME! --target-p-values %TARGET_P_VALUES% --device auto --h-dims %H_DIMS% --rep-dim %REP_DIM% --method %METHOD% --test-index-class anomaly
  ) else (
    uv run python "%SCRIPT_DIR%run_si_experiment.py" --dataset-name "!DATASET!" --data-root "%DATA_ROOT%"  --model-name !MODEL_NAME! --seed !DEBUG_SEED! --device auto --h-dims %H_DIMS% --rep-dim %REP_DIM% --method %METHOD% --test-index-class anomaly
  )
  if errorlevel 1 (
    echo TPR experiment failed on dataset=!DATASET!
    exit /b 1
  )
)

echo All TPR ODDS runs completed.
endlocal
