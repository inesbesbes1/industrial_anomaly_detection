@echo off
REM Lance toute la partie A d'un coup (Windows).
REM Usage : scripts\run_all.bat [dossier_des_archives]
cd /d "%~dp0\.."
set ARCH=%1
if "%ARCH%"=="" set ARCH=.

echo === 1/4 Preparation des donnees ===
python -m detection.prepare_data --archives-dir "%ARCH%" || exit /b 1
echo === 2/4 Entrainement PatchCore ===
python -m detection.train || exit /b 1
echo === 3/4 Evaluation ===
python -m detection.evaluate || exit /b 1
echo === 4/4 Analyse des erreurs ===
python -m detection.error_analysis || exit /b 1

echo.
echo Termine. Resultats : results\results_table.md et results\^<categorie^>\
