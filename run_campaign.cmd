@echo off
setlocal

:: This script runs the full pretraining campaign (SparK and MIM) in sequence.
:: No user input is required.

echo ======================================================================
echo [1/2] STARTING SPARK GRID SEARCH
echo ======================================================================
call ".\scripts\spark_grid_train.cmd" --training-name "spark-search_new" --config configs/spark_grid_search.yaml --verbose
if %ERRORLEVEL% neq 0 (
    echo [CRITICAL ERROR] SparK training failed. Aborting campaign.
    exit /b %ERRORLEVEL%
)

echo.
echo [GENERATING SPARK SUMMARY PLOTS]
call ".\scripts\plot_ssl_grid.cmd" --training-name "spark-search_new"

echo.
echo ======================================================================
echo [2/2] STARTING MIM GRID SEARCH
echo ======================================================================
call ".\scripts\train_ssl_grid.cmd" --training-name "mim-search_new" --config configs/mim_grid_config.yaml --verbose
if %ERRORLEVEL% neq 0 (
    echo [CRITICAL ERROR] MIM training failed. Aborting campaign.
    exit /b %ERRORLEVEL%
)

echo.
echo [GENERATING MIM SUMMARY PLOTS]
call ".\scripts\plot_ssl_grid.cmd" --training-name "mim-search_new"

echo.
echo ======================================================================
echo PRETRAINING CAMPAIGN COMPLETED SUCCESSFULLY
echo ======================================================================
pause
