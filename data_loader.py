from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
import json
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import MarkdownTextSplitter
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
import streamlit as st


def load_env():
    load_dotenv()

@st.cache_resource
def load_emotion_data():
    """
    지정된 경로에서 감성 대화 말뭉치 JSON 파일을 로드하여 Pandas DataFrame으로 반환합니다.
    """
    # 사용자 환경에 맞게 파일 경로를 수정해야 할 수 있습니다.
    file_path = "data/감성대화말뭉치(최종데이터)_Training.json"

    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"지정된 경로에 파일이 없습니다: {file_path}")

    df = pd.DataFrame(data)

    # 필요한 컬럼만 선택하여 반환
    return df[['감정_대분류', '사람문장1', '시스템문장1', '사람문장2', '시스템문장2']]

@st.cache_resource
def load_markdown_retriever():
    """
    지정된 Markdown 파일을 로드하여 RAG Retriever를 생성합니다.
    """
    file_path = "data/depression.md"

    try:
        loader = TextLoader(file_path, encoding='utf-8')
        documents = loader.load()
    except FileNotFoundError:
        raise FileNotFoundError(f"지정된 경로에 파일이 없습니다: {file_path}")

    # Markdown 문법 기준으로 텍스트 분할
    markdown_splitter = MarkdownTextSplitter(chunk_size=300, chunk_overlap=50)
    docs = markdown_splitter.split_documents(documents)

    # Upstage 임베딩과 Chroma DB를 사용하여 retriever 생성
    embeddings = UpstageEmbeddings(model="solar-embedding-1-large")
    vectorstore = Chroma.from_documents(docs, embeddings)

    return vectorstore.as_retriever()