import random
import os
import json
from langchain_core.messages import HumanMessage, SystemMessage
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
from reportlab.platypus import Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.colors import black

class EmotionBasedPsychotherapy:
    # --- __init__ (생성자) 수정 ---
    def __init__(self, emotion_df, md_retriever):
        self.emotion_df = emotion_df  # 감성대화 데이터 추가
        self.md_retriever = md_retriever  # Markdown Retriever 추가
        self.score = 0
        self.question_index = 0
        self.chat_history = [] # 대화 기록을 저장할 리스트

            # --- 모든 LLM 호출을 담당할 ChatUpstage 객체 생성 ---
        self.llm = ChatUpstage(model="solar-mini")
    # PHQ-9(Patient Health Questionnaire-9) 우울증 자가 진단 도구
        self.all_questions = [
            "일상적인 활동에 대한 흥미나 즐거움이 많이 줄어들었나요?",
            "기분이 가라앉거나 우울하고 절망적인 느낌이 들었나요?",
            "잠들기 어렵거나 자주 깨는 등 수면에 문제가 있었나요?",
            "평소보다 피곤하고 기운이 없는 느낌을 자주 받았나요?",
            "식욕이 크게 줄거나 반대로 너무 많이 먹지는 않았나요?",
            "스스로를 실패자라고 느끼거나 가족을 실망시켰다는 죄책감이 들었나요?",
            "신문이나 TV를 보는 것과 같은 일상적인 일에 집중하기 어려웠나요?",
            "다른 사람이 알아챌 정도로 행동이 굼떠지거나, 혹은 너무 안절부절못하지는 않았나요?",
            "차라리 죽는 게 낫겠다거나 스스로를 해치고 싶다는 생각을 한 적이 있나요?"
        ]
        self.screening_questions = random.sample(self.all_questions, 5)
        self.total_questions = len(self.screening_questions)

        qa_system_prompt = """당신은 우울증 전문가입니다. 검색된 컨텍스트 정보를 사용하여 질문에 답변하세요. 답변은 한국어로, 세 문장 이내로 간결하게 유지하세요.
        {context}"""
        qa_prompt = ChatPromptTemplate.from_messages([("system", qa_system_prompt), ("human", "{input}")])
        Youtube_chain = create_stuff_documents_chain(self.llm, qa_prompt)
        self.rag_chain = create_retrieval_chain(self.md_retriever, Youtube_chain)

    def generate_final_analysis(self, user_data):
        # --- 1. 점수 기반으로 진단명 결정 (수정된 부분) ---
        score = self.score
        if score >= 13:
            diagnosis_title = "중증 우울증"
        elif 9 <= score <= 12:
            diagnosis_title = "초기 우울증"
        elif 5 <= score <= 8:
            diagnosis_title = "가벼운 우울 증상"
        else:
            diagnosis_title = "우울감 없음"
        
        # --- 2. 진단명을 중앙에 표시하는 헤더 생성 (수정된 부분) ---
        report_header = f"""
        <div style="text-align: center; margin-bottom: 20px;">
            <h2>진단 결과</h2>
            <h1 style="color: #D20A0A;">{diagnosis_title}</h1>
        </div>

        ---
        """

        # 3. 종합적인 정보 요약
        user_summary = f"""
        - 사용자: {user_data.get('나이')}세 {user_data.get('성별')}, 이름: {user_data.get('이름')}
        - 과거 병력: {user_data.get('과거 병력', '없음')}
        - 주요 증상: {user_data.get('주요 증상')}
        - 5가지 질문 평가 점수: {user_data.get('질문 총점')} 점
        - 서술형 답변 점수: {user_data.get('서술형 점수')} 점
        - 최종 총점: {self.score} 점
        - 사용자의 서술: {user_data.get('서술형 답변')}
        """

        # 4. RAG를 위한 핵심 질문 생성
        main_query = f"{user_data.get('주요 증상')}과 {user_data.get('서술형 답변')} 내용을 겪는 {user_data.get('나이')}세 사용자를 위한 우울증 관리 방법, 원인, 치료법, 지원 체계를 알려줘."

        # 5. RAG로 depression.md에서 관련 정보 검색
        rag_context = self.rag_chain.invoke({"input": main_query})
        retrieved_info = "\n".join([doc.page_content for doc in rag_context['context']])

        # 6. 최종 답변 생성을 위한 프롬프트
        final_prompt = f"""
        당신은 매우 공감 능력이 뛰어난 심리 상담 전문가입니다.
        아래 사용자 정보와 전문가의 분석 노트를 바탕으로, 사용자에게 전달할 최종 답변을 아래 지시사항에 따라 작성해주세요.
        사용자의 진단명은 '{diagnosis_title}'입니다. 이 점을 고려하여 답변해주세요.

        ### 사용자 정보
        {user_summary}

        ### 전문가 분석 노트 (검색된 정보)
        {retrieved_info}

        ### 지시사항
        1. **(지원 체계)**: 분석 노트를 참고하여, 사용자에게 도움이 될 만한 기관이나 지원 프로그램을 구체적으로 제시해주세요.
        2. **(관리 방법)**: 사용자가 일상에서 시도해볼 수 있는 현실적인 스트레스 및 우울감 관리 방법을 2-3가지 제안해주세요.
        3. **(원인 및 치료법)**: 사용자의 증상과 관련된 원인을 간단히 언급하고, 일반적인 치료 방법에 대해 희망적으로 설명해주세요.
        4. **(마음의 메시지)**: 모든 내용을 종합하여, 사용자의 노력을 인정하고 희망을 주는 매우 따뜻하고 진심 어린 응원 메시지로 마무리해주세요.

        위 4가지 항목을 각각 소제목으로 구분하여 자연스러운 문단으로 작성해주세요. 진단명은 이미 맨 위에 표시되므로 본문에서는 언급하지 않아도 됩니다.
        """

        response = self.llm.invoke([HumanMessage(content=final_prompt)], model="solar-pro", temperature=0.7)

        # --- 7. 헤더와 생성된 답변을 합쳐서 최종 결과 반환 (수정된 부분) ---
        return (report_header, response.content)


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

        system_prompt = f"""
            당신은 따뜻하고 공감능력이 뛰어난 심리 상담사입니다. 사용자의 이전 답변에 대해 한두 문장으로 짧게 공감해주세요.
            그 다음, 다른 말을 추가하지 말고 아래에 주어진 '다음에 할 질문'을 그대로 이어서 물어보세요.
            '오늘의 질문' 같은 제목이나 다른 질문을 절대로 만들지 마세요.

            {few_shot_examples}



            다음 질문을 자연스럽게 이어서 해주세요: {next_question}
            """
        messages = [
            SystemMessage(content=system_prompt), 
            HumanMessage(content=user_input)
        ]

        response = self.llm.invoke(messages, temperature=0.7)
        bot_response = response.content

        # 챗봇의 답변도 대화 기록에 저장
        self.chat_history.append({"role": "assistant", "content": bot_response})

        return bot_response

    def process_and_score_answer(self, answer):
        """LLM을 사용해 사용자의 답변을 분석하고 점수를 매기는 새로운 함수"""
        
        # 현재 어떤 질문에 대한 답변인지 명시
        current_question = self.screening_questions[self.question_index]
        
        system_prompt = f"""
        당신은 숙련된 심리 분석가입니다. 사용자의 답변을 주어진 질문의 맥락에서 분석하고, 우울감의 심각도를 0점에서 3점 사이로 평가해주세요.
        - 0점: 우울감이나 부정적 정서가 전혀 드러나지 않음.
        - 1점: 약간의 스트레스나 가벼운 우울감이 암시됨.
        - 2점: 꽤 명확한 우울감, 무기력, 불안 등이 드러남.
        - 3점: '자해', '자살', '죽음', '끝내고 싶다', 심각한 수준의 우울감, 절망, 사고 등 심리적으로 매우 심각하고 위험한 단어나 맥락이 포함됨.

        반드시 아래와 같은 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.
        {{
        "score": <평가 점수 (0-3)>,
        "reason": "<왜 그렇게 평가했는지에 대한 간략한 한글 설명>"
        }}

        ---
        질문: "{current_question}"
        ---
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"사용자 답변: \"{answer}\"")
        ]

        try:
            response = self.llm.invoke(messages, model="solar-mini", temperature=0.1)
            # LLM의 응답이 JSON 형식이므로, 이를 파싱합니다.
            result = json.loads(response.content)
            points = result.get("score", 0)
            reason = result.get("reason", "분석 실패")
            
            self.score += points
            self.question_index += 1
            
            return f"분석 결과: {reason} ({points}점 추가, 현재 총점: {self.score}점)"

        except Exception as e:
            print(f"점수 분석 중 오류 발생: {e}")
            self.question_index += 1
            return f"점수 분석 중 오류가 발생했습니다. (0점 처리, 현재 총점: {self.score}점)"
        
    def score_narrative_answer(self, narrative_text):
        """서술형 답변을 분석하고 점수를 매기는 함수"""
        
        system_prompt = f"""
        당신은 숙련된 심리 분석가입니다. 사용자의 자유 서술형 답변을 분석하고, 우울감 및 위험도의 심각성을 0점에서 3점 사이로 평가해주세요.
        - 0점: 우울감이나 부정적 정서가 거의 드러나지 않음.
        - 1점: 약간의 스트레스나 가벼운 우울감이 암시됨.
        - 2점: 꽤 명확한 우울감, 무기력, 불안 등이 드러남.
        - 3점: '자해', '자살', '죽음', '끝내고 싶다', 심각한 수준의 우울감, 절망, 사고 등 심리적으로 매우 심각하고 위험한 단어나 맥락이 포함됨.

        반드시 아래와 같은 JSON 형식으로만 응답해야 합니다. 다른 설명은 절대 추가하지 마세요.
        {{
        "score": <평가 점수 (0-3)>,
        "reason": "<왜 그렇게 평가했는지에 대한 간략한 한글 설명>"
        }}
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"사용자 답변: \"{narrative_text}\"")
        ]

        try:
            response = self.llm.invoke(messages, model="solar-mini", temperature=0.1)
            result = json.loads(response.content)
            points = result.get("score", 0)
            reason = result.get("reason", "분석 실패")
            
            # 사용자 데이터에 점수와 총점을 기록하기 위해 딕셔너리로 반환
            return {"points": points, "reason": reason}

        except Exception as e:
            print(f"서술형 답변 점수 분석 중 오류 발생: {e}")
            return {"points": 0, "reason": "점수 분석 중 오류가 발생했습니다."}

    def is_test_finished(self):
        return self.question_index >= self.total_questions
    

    def summarize_for_report(self, user_data, final_score):
        """점수 기반 진단 로직과 RAG를 활용하여 전문적인 PDF 보고서 내용을 생성하는 함수"""

        # 1. 점수 기준으로 기본 진단명 결정
        if final_score >= 14:
            diagnosis_title = "중증 우울증"
        elif 10 <= final_score <= 13:
            diagnosis_title = "중등도 우울증"
        elif 6 <= final_score <= 9:
            diagnosis_title = "경도 우울 증상"
        else: 
            diagnosis_title = "우울감 없음"

        # 2. RAG를 통해 사용자의 증상과 관련된 전문 정보 검색
        symptoms = user_data.get('주요 증상', '')
        narrative = user_data.get('서술형 답변', '')
        main_query = f"'{diagnosis_title}' 상태로 진단되었고, 주요 증상이 '{symptoms}'이며 '{narrative}'와 같은 어려움을 겪는 환자에 대한 정보"
        
        try:
            retrieved_docs = self.md_retriever.invoke(main_query)
            retrieved_info = "\n".join([doc.page_content for doc in retrieved_docs])
        except Exception as e:
            print(f"RAG 검색 오류: {e}")
            retrieved_info = "관련 정보를 찾는 데 실패했습니다."

        # 3. 검색된 정보와 사용자 데이터를 바탕으로 최종 보고서 생성
        final_prompt = f"""
        당신은 정신건강의학과 전문의입니다. 아래 정보를 바탕으로 '우울증 자가 진단 결과서'의 [진단명(추정)]과 [조치결과(권장사항)] 항목을 작성해주세요.

        ### 환자 정보 ###
        - 이름: {user_data.get('이름')}
        - 나이/성별: {user_data.get('나이')}세 / {user_data.get('성별')}
        - 총점: {final_score} 점
        - 점수 기반 진단: {diagnosis_title}
        - 주요 증상(사용자 입력): {symptoms}
        - 서술 내용(사용자 입력): {narrative}
        
        ### 관련 의학 정보 (depression.md) ###
        {retrieved_info}

        ### 작성 지침 ###
        1. **[진단명(추정)]**: 먼저 위 '점수 기반 진단'인 '{diagnosis_title}'을 언급해주세요. 그 다음, 환자의 증상과 의학 정보를 종합하여 전문적인 소견을 1~2 문장으로 구체화해주세요. (예: "초기 우울증 수준으로, 특히 대인관계 스트레스와 관련된 불안 및 무기력감이 두드러집니다.")
        2. **[조치결과(권장사항)]**: 실제 의사가 환자에게 말하듯, 현실적이고 구체적인 조치 방안을 제안해주세요. 정신건강의학과 방문 권유, 상담치료, 생활 습관 개선 등 도움이 될 만한 정보를 따뜻하고 신뢰감 있는 어조로 작성해주세요.
        
        결과는 아래와 같이 각 항목의 내용만 작성하고, 다른 말은 덧붙이지 마세요.
        [진단명(추정)]: <내용>
        [조치결과(권장사항)]: <내용>
        """
        
        messages = [HumanMessage(content=final_prompt)]
        response = self.llm.invoke(messages, model="solar-mini", temperature=0.5)
        report_content = response.content
        report_data = {}
        try:
            diag_part = report_content.split("[진단명(추정)]:")[1].split("[조치결과(권장사항)]:")[0].strip()
            reco_part = report_content.split("[조치결과(권장사항)]:")[1].strip()
            report_data['진단명(추정)'] = diag_part
            report_data['조치결과(권장사항)'] = reco_part
        except IndexError:
            print("보고서 내용 파싱 실패")
            report_data['진단명(추정)'] = "내용을 생성하지 못했습니다."
            report_data['조치결과(권장사항)'] = "내용을 생성하지 못했습니다."

        report_data['환자 정보'] = f"{user_data.get('이름')} ({user_data.get('나이')}세, {user_data.get('성별')})"
        report_data['주된 증상'] = user_data.get('주요 증상')
        return report_data


    def create_report_pdf(self, report_data, output_path):
        """분석된 데이터를 바탕으로 자동 줄바꿈이 적용된 PDF 보고서를 생성하는 함수"""
        try:
            font_path = "font/NanumGothic.ttf"
            if not os.path.exists(font_path):
                raise FileNotFoundError("폰트 파일이 없습니다.")
            pdfmetrics.registerFont(TTFont('NanumGothic', font_path))
            font_name = 'NanumGothic'
        except Exception as e:
            print(f"나눔고딕 폰트를 찾을 수 없어 기본 폰트로 생성합니다: {e}")
            font_name = 'Helvetica'

        c = canvas.Canvas(output_path, pagesize=letter)
        width, height = letter

        # --- 자동 줄바꿈을 위한 스타일 정의 ---
        styles = getSampleStyleSheet()
        # 기본 Paragraph 스타일을 우리 폰트에 맞게 새로 정의합니다.
        body_style = ParagraphStyle(
            name='Body',
            parent=styles['Normal'],
            fontName=font_name,
            fontSize=12,
            leading=18,  # 줄 간격
            alignment=TA_LEFT,
            textColor=black,
        )

        # 제목
        c.setFont(font_name, 18)
        c.drawCentredString(width / 2.0, height - 60, "우울증 자가 진단 결과서")

        # --- 내용을 Paragraph 객체로 그려주기 ---
        text_y = height - 100
        
        # 보고서에 표시될 정보와 순서
        report_data['진단일자'] = datetime.now().strftime("%Y-%m-%d")
        report_data['총점'] = f"{self.score} 점"
        display_order = ['진단일자', '환자 정보', '총점', '주된 증상', '진단명(추정)', '조치결과(권장사항)']

        for key in display_order:
            value = report_data.get(key, "내용 없음")
            
            # key 부분은 굵게(<b>) 처리하고, value와 합쳐서 하나의 문단으로 만듭니다.
            text = f"<b>■ {key}:</b> {value}"
            
            # Paragraph 객체 생성
            p = Paragraph(text, style=body_style)
            
            # 문단이 그려질 폭과 높이를 계산합니다. (좌우 여백 100씩 총 200)
            p_width, p_height = p.wrapOn(c, width - 200, height)
            
            # 계산된 높이만큼 y 위치를 조정한 후 문단을 그립니다.
            text_y -= p_height
            p.drawOn(c, 100, text_y)
            
            # 항목 간의 간격을 줍니다.
            text_y -= 15

        c.save()
        return True