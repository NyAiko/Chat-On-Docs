install:
		pip install -r requirements.txt
		python setup_nltk.py

run:
		streamlit run app.py