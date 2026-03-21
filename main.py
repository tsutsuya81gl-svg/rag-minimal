# main.py

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import os

# ===== ① テキスト =====
text = """
LangChainはLLMアプリ開発のためのフレームワークです。
RAGは外部データを使って回答精度を上げる仕組みです。
RAGでは検索した情報をもとに回答を生成します。
"""

# ===== ② 分割 =====
splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=20
)
texts = splitter.split_text(text)
docs = [Document(page_content=t) for t in texts]

# ===== ③ Embedding =====
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-base"
)

# ===== ④ DB =====
DB_PATH = "faiss_db"

if os.path.exists(DB_PATH):
    print("⇒ 既存DB読み込み")
    db = FAISS.load_local(DB_PATH, embeddings)
else:
    print("⇒ 新規DB作成")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(DB_PATH)

# ===== ⑤ 検索 =====
query = "RAGとは？"

retrieved_docs = db.similarity_search(query, k=3)

print("\n===== 取得データ =====")
for doc in retrieved_docs:
    print(doc.page_content)

# ===== ⑥ 抽出（修正版）=====

# ★ 文単位に分割
sentences = []
for doc in retrieved_docs:
    sentences.extend(doc.page_content.split("。"))

sentences = [s + "。" for s in sentences if s]

# ★ 抽出
selected = ""
for s in sentences:
    if "RAGは" in s:
        selected = s
        break

if selected == "":
    selected = sentences[0]

print("\n===== 抽出結果 =====")
print(selected)

# ===== ⑦ LLM（整形専用）=====
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

prompt = f"""
次の文章をそのまま出力してください。

{selected}

出力:
"""

inputs = tokenizer(prompt, return_tensors="pt")

outputs = model.generate(
    **inputs,
    max_new_tokens=50,
    do_sample=False
)

answer = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

if answer == "":
    answer = selected

# ===== ⑧ 出力 =====
print("\n===== 最終回答 =====")
print(answer)
