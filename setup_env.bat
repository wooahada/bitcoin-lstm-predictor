@echo off

python -m venv venv
call venv\Scripts\activate

echo ✅ 가상환경 활성화됨!

pip install --upgrade pip
pip install streamlit pandas numpy requests scikit-learn tensorflow prophet xgboost matplotlib plotly

pip freeze > requirements.txt

echo ✅ 모든 라이브러리 설치 완료!
echo 📝 requirements.txt 생성됨.
