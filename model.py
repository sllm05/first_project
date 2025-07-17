import random
import os
from langchain_upstage import UpstageEmbeddings, ChatUpstage
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from datetime import datetime

# PDF 생성을 위한 라이브러리 추가
# 설치가 필요합니다: pip install reportlab
from reportlab.pdfgen import canvas
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from reportlab.lib.pagesizes import letter

class EmotionBasedPsychotherapy:
    # --- __init__ (생성자) 수정 ---
    def __init__(self, client, emotion_df, md_retriever):
        self.client = client
        self.emotion_df = emotion_df  # 감성대화 데이터 추가
        self.md_retriever = md_retriever  # Markdown Retriever 추가
        self.score = 0
        self.question_index = 0
        self.chat_history = [] # 대화 기록을 저장할 리스트

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

        # --- 기존 RAG 체인 삭제, 정보 검색 체인으로 대체 ---
        qa_system_prompt = """당신은 우울증 전문가입니다. 검색된 컨텍스트 정보를 사용하여 질문에 답변하세요. 답변은 한국어로, 세 문장 이내로 간결하게 유지하세요.
        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{input}")])
        Youtube_chain = create_stuff_documents_chain(ChatUpstage(model="solar-1-mini"), qa_prompt)
        self.rag_chain = create_retrieval_chain(self.md_retriever, Youtube_chain)

    def get_emotion_level(self, emotion):
        for level, emotions in self.emotion_levels.items():
            if emotion in emotions:
                return level
        return '보통'

    def _call_solar_for_emotion(self, text):
        emotion_categories = ['불안', '분노', '슬픔', '상처', '당황', '기쁨']
        prompt_messages = [
            {"role": "system", "content": f"너는 문장의 감정을 분석하는 전문가야. 다음 문장의 감정을 {emotion_categories} 중에서 하나만 골라. 다른 말은 하지마."},
            {"role": "user", "content": text}
        ]
        try:
            response = self.client.chat.completions.create(model="solar-mini", messages=prompt_messages, temperature=0.0, max_tokens=10)
            content = response.choices[0].message.content.strip()
            return content if content in emotion_categories else '상처'
        except Exception as e:
            print(f"API 호출 오류: {e}")
            return '상처'

    # --- generate_empathetic_response_and_ask_question 수정 ---
    def generate_empathetic_response_and_ask_question(self, user_input):
        if self.is_test_finished():
            return None

        # 대화 기록 저장
        self.chat_history.append({"role": "user", "content": user_input})

        next_question = self.screening_questions[self.question_index]

        # 감성대화 말뭉치에서 예시 추출 (Few-shot Prompting)
        samples = self.emotion_df.sample(n=2)
        few_shot_examples = ""
        for index, row in samples.iterrows():
            few_shot_examples += f"\n#대화 예시 {index+1}\n- 사용자: {row['사람문장1']}\n- 상담사: {row['시스템문장1']}"

        system_prompt = f"""너는 따뜻한 심리 상담사이다. 아래 대화 예시를 참고하여 사용자의 말에 자연스럽게 공감한 후, 다음 질문으로 대화를 이끌어간다.

{few_shot_examples}

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

        response = self.client.chat.completions.create(model="solar-pro", messages=messages, temperature=0.7)
        bot_response = response.choices[0].message.content

        # 챗봇의 답변도 대화 기록에 저장
        self.chat_history.append({"role": "assistant", "content": bot_response})

        return bot_response

    def process_and_score_answer(self, answer):
        emotion = self._call_solar_for_emotion(answer)
        level = self.get_emotion_level(emotion)
        points = self.emotion_scores.get(level, 0)
        self.score += points
        self.question_index += 1
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

    # --- 기존 RAG 함수들을 대체할 새로운 함수들 ---

    def get_info_from_md(self, user_input):
        """depression.md 파일에서 정보를 검색하여 답변하는 함수"""
        if not self.rag_chain:
            return "정보 검색 기능이 준비되지 않았습니다."

        result = self.rag_chain.invoke({"input": user_input})
        return result["answer"]

    # --- summarize_for_report 수정 ---
    def summarize_for_report(self, uploaded_pdf_text=None):
        """대화 내용과 업로드된 PDF를 바탕으로 보고서 내용을 요약/분석하는 함수"""

        conversation_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history])

        # 참고 자료 섹션 구성
        reference_material = f"--- 대화 내용 ---\n{conversation_text}"
        if uploaded_pdf_text:
            reference_material += f"\n\n--- 사용자가 업로드한 참고 문서 내용 ---\n{uploaded_pdf_text}"

        prompt = f"""
        당신은 정신과 전문의입니다. 아래 참고 자료(대화 내용, 사용자 제출 문서)를 종합적으로 검토하여 '우울증 자가 진단서'의 각 항목을 채워주세요.
        결과는 각 항목에 대한 설명만 간결하게 작성하고, 다른 말은 덧붙이지 마세요.
        각 항목은 "항목명: 내용" 형식으로 출력해주세요.

        - 환자 정보: (예: 특정 정보 없음, 온라인 사용자)
        - 주된 증상: (예: 불면, 불안, 우울감 등 대화에서 나타난 핵심 증상 요약)
        - 진단명(추정): (예: 우울증 의심, 스트레스 반응 등)
        - 조치결과(권장사항): (예: 전문가 상담 권유, 스트레스 관리 필요 등)

        {reference_material}
        """

        response = self.client.chat.completions.create(
            model="solar-pro",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3
        )

        summary_text = response.choices[0].message.content
        report_data = {}
        for line in summary_text.split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                report_data[key.strip()] = value.strip()

        return report_data

    def create_report_pdf(self, report_data, output_path):
        """분석된 데이터를 바탕으로 새로운 PDF 보고서를 생성하는 함수"""
        try:
            # 윈도우 환경에 맞는 나눔고딕 폰트 경로. 다른 환경에서는 경로 수정 필요.
            # 폰트 파일이 없다면 별도 설치가 필요합니다.
            font_path = "font/NanumGothic.ttf"
            pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
        except Exception as e:
            print(f"폰트 파일을 찾을 수 없습니다: {e}. 기본 폰트로 생성됩니다.")
            # 폰트가 없을 경우를 대비한 예외 처리도 가능

        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        # 제목
        c.setFont('NanumGothic', 18)
        c.drawCentredString(width / 2.0, height - 50, "우울증 자가 진단 결과서")

        # 내용
        c.setFont('NanumGothic', 12)
        text_y = height - 100

        # 기본 정보 추가
        report_data['진단일자'] = datetime.now().strftime("%Y-%m-%d")
        report_data['총점'] = f"{self.score} 점"

        # 데이터 순서 정의
        display_order = ['진단일자', '환자 정보', '총점', '주된 증상', '진단명(추정)', '조치결과(권장사항)']

        for key in display_order:
            value = report_data.get(key, "내용 없음")
            c.drawString(100, text_y, f"■ {key}: {value}")
            text_y -= 30 # 줄 간격

        c.save()
        return True