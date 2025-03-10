import os
import pickle
import pdfplumber
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import ssl
import certifi

ssl._create_default_https_context = ssl._create_unverified_context

# .env 파일 로드
load_dotenv()

# PDF 파일 경로 (상위 폴더에 위치한 파일)
PDF_FILE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "sample.pdf"))

# PDF 문서가 존재하는지 확인
if not os.path.exists(PDF_FILE_PATH):
    raise FileNotFoundError(f"❌ PDF 파일을 찾을 수 없습니다: {PDF_FILE_PATH}")

print(f"📄 PDF 문서 {PDF_FILE_PATH} 로드 중...")

# PDF에서 텍스트 추출
documents = []
with pdfplumber.open(PDF_FILE_PATH) as pdf:
    for page in pdf.pages:
        text = page.extract_text()
        if text:
            documents.append(Document(page_content=text))

# 텍스트 분할
print("🔄 문서 분할 중...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Hugging Face 임베딩 생성 (무료 대체)
print("🔍 문서를 벡터화 중...")
embeddings = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# FAISS 인덱스 생성
vectorstore = FAISS.from_documents(texts, embeddings)

# FAISS 인덱스를 로컬 저장
FAISS_INDEX_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "faiss_index.pkl"))
with open(FAISS_INDEX_PATH, "wb") as f:
    pickle.dump(vectorstore, f)

print(f"✅ {len(texts)} 개의 문서를 FAISS에 저장 완료! (경로: {FAISS_INDEX_PATH})")
