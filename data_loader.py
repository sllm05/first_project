from dotenv import load_dotenv
from openai import OpenAI
import os
import pandas as pd
import json

def load_environment_and_client():
    """
    .env 파일에서 환경 변수를 로드하고 Upstage API 클라이언트를 초기화합니다.
    """
    load_dotenv()
    
    api_key = os.getenv("SOLAR_API_KEY")
    
    if not api_key:
        raise ValueError("SOLAR_API_KEY 환경변수가 설정되지 않았습니다. .env 파일을 확인해주세요.")
        
    client = OpenAI(api_key=api_key, base_url="https://api.upstage.ai/v1")
    
    return client

def load_emotion_data():
    """
    지정된 경로에서 감성 대화 말뭉치 JSON 파일을 로드하여 Pandas DataFrame으로 반환합니다.
    """
    # 사용자 환경에 맞게 파일 경로를 수정해야 할 수 있습니다.
    file_path = r"C:\Users\DJ\Documents\pythonProject\project\data\감성대화말뭉치_최종데이터_Training.json"
    
    try:
        with open(file_path, "r", encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"지정된 경로에 파일이 없습니다: {file_path}")

    df = pd.DataFrame(data)
    
    # 필요한 컬럼만 선택하여 반환
    return df[['감정_대분류', '사람문장1', '시스템문장1', '사람문장2', '시스템문장2']]


# 미작성
