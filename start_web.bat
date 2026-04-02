@echo off
call .venv\Scripts\activate
echo Starting Wound Segmentation Web App...
streamlit run web_app.py
pause
