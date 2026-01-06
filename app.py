# ================================
# 1. IMPORTS
# ================================

import os
import pandas as pd
from tqdm import tqdm

from flask import Flask, request, jsonify, render_template_string
from transformers import pipeline

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA


# ================================
# 2. CONFIGURATION
# ================================

IOC_FILE = "Compromised_IOCs_Cleaned.csv"
TWEET_FILE = "Tweetfeed_cleaned.csv"

VECTOR_DB_DIR = "./rag_db"
TOP_K = 3

os.environ["TOKENIZERS_PARALLELISM"] = "false"


# ================================
# 3. LOAD DATA
# ================================

print("Loading datasets...")

df_iocs = pd.read_csv(IOC_FILE, encoding="latin1")
df_tweets = pd.read_csv(TWEET_FILE, encoding="latin1")

# Normalize column names
df_iocs.columns = df_iocs.columns.str.lower()
df_tweets.columns = df_tweets.columns.str.lower()


# ---------- FIX FOR 'tweet' COLUMN ISSUE ----------
def find_text_column(df):
    candidates = [
        "tweet", "text", "full_text", "content",
        "message", "body"
    ]
    for col in candidates:
        if col in df.columns:
            return col

    # fallback: first string column
    for col in df.columns:
        if df[col].dtype == "object":
            return col

    raise ValueError(
        f"No suitable text column found. Columns: {df.columns.tolist()}"
    )


ioc_text_col = find_text_column(df_iocs)
tweet_text_col = find_text_column(df_tweets)

print("Using IOC text column:", ioc_text_col)
print("Using Tweetfeed text column:", tweet_text_col)

# Combine text from both datasets
texts = pd.concat(
    [df_iocs[ioc_text_col], df_tweets[tweet_text_col]],
    ignore_index=True
).drop_duplicates().dropna().astype(str).tolist()

print(f"Loaded {len(texts)} unique tweets")


# ================================
# 4. EMBEDDINGS & VECTOR STORE
# ================================

print("Initializing embeddings...")

embedding_function = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectordb = Chroma(
    persist_directory=VECTOR_DB_DIR,
    embedding_function=embedding_function
)

if vectordb._collection.count() == 0:
    print("Embedding tweets into ChromaDB...")

    ids = [str(i) for i in range(len(texts))]
    batch_size = 100

    for i in tqdm(range(0, len(texts), batch_size)):
        vectordb.add_texts(
            texts=texts[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )

    vectordb.persist()
    print("Vector database created and saved.")

else:
    print("Existing vector database loaded.")


# ================================
# 5. RAG PIPELINE
# ================================

print("Loading language model...")

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    max_new_tokens=200
)

llm = HuggingFacePipeline(pipeline=generator)

retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

print("RAG pipeline ready.")


# ================================
# 6. FLASK APP
# ================================

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "status": "running",
        "message": "RAG Cyber Threat Intelligence API",
        "endpoints": {
            "POST /chat": {
                "input": {"query": "string"},
                "output": {"answer": "string", "sources": "list"}
            },
            "GET /ui": "Browser-based interface"
        }
    })


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

    if not data or "query" not in data:
        return jsonify({"error": "Missing 'query' field"}), 400

    user_query = data["query"]

    prompt = (
        "Answer in 2–3 sentences based on the retrieved sources: "
        + user_query
    )

    result = qa_chain({"query": prompt})

    return jsonify({
        "query": user_query,
        "answer": result["result"],
        "sources": [
            doc.page_content for doc in result["source_documents"]
        ]
    })


@app.route("/ui", methods=["GET"])
def ui():
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>RAG Cyber Threat Intelligence</title>
        <style>
            body { font-family: Arial; margin: 40px; }
            textarea { width: 100%; height: 80px; }
            button { padding: 10px; margin-top: 10px; }
            .answer { margin-top: 20px; padding: 10px; border: 1px solid #ccc; }
        </style>
    </head>
    <body>
        <h2>RAG Cyber Threat Intelligence Chat</h2>

        <textarea id="query" placeholder="Enter your question here..."></textarea><br>
        <button onclick="sendQuery()">Ask</button>

        <div class="answer" id="result"></div>

        <script>
            function sendQuery() {
                const query = document.getElementById("query").value;

                fetch("/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ query: query })
                })
                .then(response => response.json())
                .then(data => {
                    let output = "<b>Answer:</b> " + data.answer + "<br><br>";
                    output += "<b>Sources:</b><ul>";
                    data.sources.forEach(src => {
                        output += "<li>" + src + "</li>";
                    });
                    output += "</ul>";
                    document.getElementById("result").innerHTML = output;
                });
            }
        </script>
    </body>
    </html>
    """
    return render_template_string(html)


# ================================
# 7. RUN SERVER
# ================================

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
