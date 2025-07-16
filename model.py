# 기능
import random
import os
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

class EmotionBasedPsychotherapy:
    def __init__(self, client):
        self.client = client
        self.score = 0
        self.question_index = 0

        self.all_questions = [
            "최근 2주 동안, 일상적인 일에 대한 흥미나 즐거움이 거의 없었다.",
            "기분이 가라앉거나 우울하거나 절망감을 느꼈다.",
            "잠들기 어렵거나, 자주 깼거나, 혹은 너무 많이 잠을 잤다.",
            "피곤하거나 기운이 없었다.",
            "식욕이 줄었거나 지나치게 먹었다.",
            "자신이 실패자라고 느끼거나 자신이나 가족을 실망시켰다고 느꼈다.",
            "신문을 읽거나 TV를 보는 것처럼 집중하기가 어려웠다.",
            "다른 사람들이 알아차릴 정도로 너무 느리게 움직였거나, 너무 안절부절 못하게 움직였다.",
            "자신을 해치거나 죽이고 싶다는 생각을 했다."
        ]
        self.screening_questions = random.sample(self.all_questions, 5)
        self.total_questions = len(self.screening_questions)

        self.emotion_levels = {
            '위험': ['불안', '분노', '슬픔'],
            '보통': ['당황', '상처'],
            '정상': ['기쁨']
        }
        self.emotion_scores = {'위험': 3, '보통': 1, '정상': 0}
        self.rag_chain = None

    def get_emotion_level(self, emotion):
        for level, emotions in self.emotion_levels.items():
            if emotion in emotions:
                return level
        return '보통'

    def _call_solar_for_emotion(self, text):
        emotion_categories = ['불안', '분노', '슬픔', '상처', '당황', '기쁨']
        prompt_messages = [
            {
                "role": "system",
                "content": f"너는 문장의 감정을 분석하는 전문가야. 다음 문장의 감정을 {emotion_categories} 중에서 하나만 골라. 다른 말은 하지마."
            },
            {"role": "user", "content": text}
        ]
        try:
            response = self.client.chat.completions.create(
                model="solar-mini", messages=prompt_messages, temperature=0.0, max_tokens=10
            )
            content = response.choices[0].message.content.strip()
            return content if content in emotion_categories else '상처'
        except Exception as e:
            print(f"API 호출 오류: {e}")
            return '상처'

    def generate_empathetic_response_and_ask_question(self, user_input):
        if self.is_test_finished():
            return None

        next_question = self.screening_questions[self.question_index]
        system_prompt = f"""너는 따뜻한 심리 상담사이다. 사용자의 이전 답변에 짧게 공감한 후, 자연스럽게 다음 질문으로 대화를 이끌어간다.

# 지시사항
1. 사용자의 말에 적극적으로 공감한다.
2. 그 다음, 아래 전달된 '오늘의 질문'을 이어서 물어본다.
3. 두 문장을 합쳐서 부드러운 하나의 문단으로 만든다. "오늘의 질문:" 같은 제목은 절대 출력하지 않는다.

# 오늘의 질문
{next_question}
"""
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ]
        
        # Streamlit에서는 스트리밍 대신 한번에 응답을 받아 반환
        response = self.client.chat.completions.create(
            model="solar-pro", messages=messages, temperature=0.7
        )
        return response.choices[0].message.content

    def process_and_score_answer(self, answer):
        emotion = self._call_solar_for_emotion(answer)
        level = self.get_emotion_level(emotion)
        points = self.emotion_scores.get(level, 0)
        self.score += points
        self.question_index += 1
        
        # 분석 결과를 문자열로 반환
        return f"감정: {emotion}({level}), {points}점 추가 (현재 총점: {self.score}점)"

    def is_test_finished(self):
        return self.question_index >= self.total_questions

    def display_final_result(self):
        result_text = f"모든 질문이 완료되었습니다.\n\n**총점: {self.score}점**\n\n"
        if self.score >= 10:
            result_text += "🚨 **진단 결과: 우울증 위험** 🚨\n높은 수준의 우울감이 의심됩니다. 전문가의 도움이 필요할 수 있습니다."
        elif self.score >= 5:
            result_text += "💛 **진단 결과: 보통** 💛\n일상적인 스트레스나 가벼운 우울감을 겪고 계신 것 같습니다."
        else:
            result_text += "😄 **진단 결과: 정상** 😄\n"
        return result_text
    
    # RAG 체인 설정 함수 추가
    def setup_rag_chain(self, file_path):
        loader = PyPDFLoader(file_path)
        pages = loader.load_and_split()
        
        vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))
        retriever = vectorstore.as_retriever(k=2)
        
        chat = ChatUpstage(model="solar-1-mini")

        contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(chat, retriever, contextualize_q_prompt)

        qa_system_prompt = """당신은 질문에 답변하는 유용한 어시스턴트입니다. 검색된 컨텍스트를 사용하여 질문에 답변하세요. 답을 모른다면 모른다고 말하세요. 답변은 세 문장 이내로 간결하게 유지하세요.
        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", qa_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        Youtube_chain = create_stuff_documents_chain(chat, qa_prompt)
        self.rag_chain = create_retrieval_chain(history_aware_retriever, Youtube_chain)
        
    # RAG 답변 생성 함수 추가
    def get_rag_answer(self, user_input, chat_history):
        if not self.rag_chain:
            return "먼저 PDF 파일을 업로드하여 RAG 체인을 설정해주세요.", None

        result = self.rag_chain.invoke({"input": user_input, "chat_history": chat_history})
        return result["answer"], result["context"]