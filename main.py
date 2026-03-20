import os
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

# APIキー設定
os.environ["OPENAI_API_KEY"] = "YOUR_API_KEY"

# 1. テキスト読み込み
with open("data.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 2. 分割
splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
docs = splitter.split_text(text)

# 3. ベクトル化
embeddings = OpenAIEmbeddings()

# 4. DB作成
db = FAISS.from_texts(docs, embeddings)

# 5. 質問ループ
llm = ChatOpenAI(model_name="gpt-3.5-turbo")

while True:
    query = input("質問: ")
    if query == "exit":
        break

    # 6. 検索
    docs = db.similarity_search(query, k=3)

    context = "\n\n".join(docs)

    # 7. 回答生成
    prompt = f"""
あなたは正確に答えるアシスタントです。

【ルール】
- 必ず以下の情報だけを使う
- 情報にないことは「わかりません」と答える
- 推測は禁止
- できるだけ具体的に答える

【情報】
{context}

【質問】
{query}
"""

    response = llm.predict(prompt)
    print("回答:", response)
