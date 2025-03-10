import streamlit as st
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import pickle
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("api key를 명시해주세요.")

# FAISS 벡터 DB 로드
FAISS_INDEX_PATH = "faiss_index.pkl"

st.title("LangChain + FAISS 기반 RAG 검색 시스템")
st.subheader("💡 질문을 입력하면, 관련 문서를 검색하고 답변을 생성합니다.")

# FAISS 벡터 DB 로드
if os.path.exists(FAISS_INDEX_PATH):
    with open(FAISS_INDEX_PATH, "rb") as f:
        vectorstore = pickle.load(f)
    retriever = vectorstore.as_retriever()
else:
    st.error("❌ FAISS 벡터 DB가 없습니다. 먼저 `data_ingestion.py`를 실행하세요.")
    st.stop()

# LangChain RAG QA Chain 설정
llm = OpenAI(model_name="gpt-4")
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

# 질문 입력
query = st.text_input("🔍 질문을 입력하세요:")
if query:
    with st.spinner("검색 중..."):
        response = qa_chain.run(query)
        st.write("### 🔹 AI 응답:")
        st.write(response)
