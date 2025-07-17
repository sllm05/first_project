import streamlit as st
import os
from model import EmotionBasedPsychotherapy
from data_loader import load_env, load_emotion_data, load_markdown_retriever # import 수정

st.set_page_config(page_title="우울하신가요?", page_icon="❤️", layout="wide")

# --- 세션 상태 초기화 수정 ---
if 'bot' not in st.session_state:
    try:
        load_env()
        emotion_df = load_emotion_data()
        md_retriever = load_markdown_retriever()
        st.session_state.bot = EmotionBasedPsychotherapy(emotion_df, md_retriever)

        st.session_state.messages = []
        st.session_state.phase = "user_info_gathering" # 현재 단계를 관리할 'phase' 추가
        st.session_state.user_data = {} # 사용자 정보를 저장할 딕셔너리
    except Exception as e:
        st.error(f"초기화 오류: {e}.")
        st.stop()

st.title("우울증 자가 진단 챗봇 🌿")
st.markdown("안녕하세요. 당신의 마음 상태를 이해하고 도움을 드리기 위해 몇 가지 질문을 시작하겠습니다.")

# --- 단계별 UI 분기 처리 ---

# 1단계: 사용자 정보 입력
if st.session_state.phase == "user_info_gathering":
    st.subheader("먼저 자신에 대해 조금만 알려주시겠어요?")
    with st.form("user_info_form"):
        name = st.text_input("이름")
        gender = st.radio("성별", ["남성", "여성"])
        age = st.number_input("나이", min_value=1, max_value=120, step=1, placeholder="나이를 입력하세요", value=None)
        symptoms = st.text_area("현재 겪고 있는 주요 증상을 알려주세요.")
        history = st.text_area("과거에 관련된 병력이 있다면 알려주세요. (없으면 비워두세요)")
        submitted = st.form_submit_button("입력 완료")

        if submitted:
            st.session_state.user_data = {
                "이름": name, "성별": gender, "나이": age, 
                "주요 증상": symptoms, "과거 병력": history
            }
            st.session_state.phase = "screening_questions"
            st.rerun()

# 채팅 기록 표시 (2단계부터 표시)
if st.session_state.phase != "user_info_gathering":
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# 2단계: 5가지 질문 평가
if st.session_state.phase == "screening_questions":
    # 첫 질문 시작
    if not st.session_state.messages:
        with st.chat_message("assistant"):
            # 어색한 안내 메시지 삭제 후 바로 첫 질문 표시
            first_question = st.session_state.bot.screening_questions[0]
            st.session_state.messages.append({"role": "assistant", "content": first_question})
            st.markdown(first_question)

    if prompt := st.chat_input("답변을 입력해주세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        analysis_result = st.session_state.bot.process_and_score_answer(prompt)
        with st.expander("답변 분석 결과 보기"):
            st.info(analysis_result)

        if st.session_state.bot.is_test_finished():
            st.session_state.phase = "narrative_input"
            with st.chat_message("assistant"):
                narrative_prompt = "마지막으로, 현재 심정에 대해 일기를 쓰듯 자유롭게 이야기해주세요. 어떤 내용이든 괜찮습니다."
                st.session_state.messages.append({"role": "assistant", "content": narrative_prompt})
                st.markdown(narrative_prompt)
            st.rerun()
        else:
            with st.chat_message("assistant"):
                with st.spinner("생각 중..."):
                    response = st.session_state.bot.generate_empathetic_response_and_ask_question(prompt)
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.markdown(response)

# 3단계: 서술형/일기 입력
if st.session_state.phase == "narrative_input":
    if prompt := st.chat_input("여기에 자유롭게 작성해주세요..."):
        st.session_state.user_data["서술형 답변"] = prompt
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        st.session_state.phase = "final_analysis"
        st.rerun()

# 4단계: 최종 분석 및 정보 제공
if st.session_state.phase == "final_analysis":
    with st.chat_message("assistant"):
        with st.spinner("모든 정보를 바탕으로 맞춤형 분석을 진행하고 있습니다..."):
            final_report = st.session_state.bot.generate_final_analysis(
                st.session_state.user_data
            )
            st.session_state.messages.append({"role": "assistant", "content": final_report})
            st.markdown(final_report)
    st.session_state.phase = "finished"

# 5단계: 종료
if st.session_state.phase == "finished":
    st.info("상담이 종료되었습니다. 이 내용이 마음에 조금이나마 도움이 되었기를 바랍니다.")

    # PDF 생성 및 다운로드 기능 추가
    if st.button("진단 결과서 PDF로 다운로드"):
        with st.spinner("PDF 보고서를 생성하고 있습니다..."):
            try:
                # 1. 보고서 데이터 요약
                report_data = st.session_state.bot.summarize_for_report(st.session_state.user_data, st.session_state.bot.score)
                

                # 2. PDF 파일 생성
                pdf_path = "우울증_자가_진단_결과서.pdf"
                success = st.session_state.bot.create_report_pdf(report_data, pdf_path)

                if success and os.path.exists(pdf_path):
                    # 3. 다운로드 버튼 제공
                    with open(pdf_path, "rb") as f:
                        pdf_bytes = f.read()

                    st.download_button(
                        label="여기를 클릭하여 PDF 다운로드",
                        data=pdf_bytes,
                        file_name=f"{st.session_state.user_data.get('이름', '사용자')}_우울증_진단결과서.pdf",
                        mime="application/pdf"
                    )
                else:
                    st.error("PDF 파일 생성에 실패했습니다.")
            except Exception as e:
                st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")