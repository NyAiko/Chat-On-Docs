setup:
		sudo apt update -y
		sudo apt upgrade -y
		sudo apt install python3-pip -y

install:
		pip install -r requirements.txt
		python3 setup_nltk.py

run:
		streamlit run app.py



