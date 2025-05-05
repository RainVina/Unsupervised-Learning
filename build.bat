@echo off
echo === Building app.exe with PyInstaller ===

REM Activate virtual environment
call .venv\Scripts\activate

REM Build the executable
pyinstaller --noconsole --onefile app.py ^
--add-data ".venv\Lib\site-packages\face_recognition_models\models\shape_predictor_68_face_landmarks.dat;face_recognition_models/models" ^
--add-data ".venv\Lib\site-packages\face_recognition_models\models\shape_predictor_5_face_landmarks.dat;face_recognition_models/models" ^
--add-data ".venv\Lib\site-packages\face_recognition_models\models\mmod_human_face_detector.dat;face_recognition_models/models" ^
--add-data ".venv\Lib\site-packages\face_recognition_models\models\dlib_face_recognition_resnet_model_v1.dat;face_recognition_models/models"

echo.
echo === Build complete ===
echo Your EXE is in the "dist" folder: dist\app.exe
pause
