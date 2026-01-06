# RAG-OSINT: Cyber Threat Intelligence Chatbot

## Overview
This project implements a **Retrieval-Augmented Generation (RAG)** based chatbot for **cyber threat intelligence** using **Open-Source Intelligence (OSINT)** from social media data.  
The system retrieves semantically relevant cybersecurity content and generates context-aware responses.

---

## Technologies Used
- Python  
- Pandas  
- HuggingFace Transformers  
- Sentence-Transformers (MiniLM)  
- ChromaDB  
- LangChain  
- Flask  

---

## Project Structure
RAG-OSINT/

├── app.py

├── README.md

├── requirements.txt

├── .gitignore


---

## Dataset
Due to size constraints, datasets are not included in this repository.

### Dataset Download Link (Google Drive)
https://drive.google.com/drive/folders/18dU6UK2wXOO4qwWixQV88jEzpVjdSTW7?usp=drive_link

After downloading, place the following files in the project root directory:

Compromised_IOCs_Cleaned.csv
Tweetfeed_cleaned.csv


---

## How to Run
```bash
pip install -r requirements.txt
python app.py

```
Open in browser:

UI: http://127.0.0.1:5000/ui
API: http://127.0.0.1:5000/


Author
Venkata Sai Thiruvakadu