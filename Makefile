setup:
	sudo apt-get update
	sudo apt-get install python3 python3-pip -y
	pip3 install -r requirements.txt

install:
	pip install -r requirements.txt
	python3 setup_nltk.py

run:
	streamlit run app.py
