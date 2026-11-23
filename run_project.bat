@echo off
echo Installing dependencies...
pip install -r requirements.txt

echo.
echo Starting EcoGuard Geo-Intelligence System...
streamlit run app.py
pause
