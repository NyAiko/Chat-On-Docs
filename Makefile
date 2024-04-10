install:
		pip install -r requirements.txt
		python3 setup_nltk.py

run:
		streamlit run app.py
