import streamlit as st
import os
import tempfile
from model import EmotionBasedPsychotherapy
from data_loader import load_environment_and_client

# 페이지 설정
st.set_page_config(
    page_title="정신감정",
    page_icon="❤️",
    layout="wide"
)

#  세션 상태 초기화 
if 'bot' not in st.session_state:
    try:
        client = load_environment_and_client()
        st.session_state.bot = EmotionBasedPsychotherapy(client)
        st.session_state.messages = []
        st.session_state.test_started = False
        st.session_state.rag_ready = False
    except ValueError as e:
        st.error(f"초기화 오류: {e}. .env 파일에 SOLAR_API_KEY가 올바르게 설정되었는지 확인하세요.")
        st.stop()

#  사이드바: PDF 업로드 
with st.sidebar:
    st.header("pdf파일 영역")
    uploaded_file = st.file_uploader("상담에 참고할 PDF 파일을 올려주세요.", type="pdf")

    if uploaded_file and not st.session_state.rag_ready:
        with st.spinner("PDF 파일을 분석하고 있어요..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            try:
                st.session_state.bot.setup_rag_chain(tmp_file_path)
                st.session_state.rag_ready = True
                st.success("PDF 분석 완료! 이제 문서에 대해 질문할 수 있습니다.")
            except Exception as e:
                st.error(f"PDF 처리 중 오류가 발생했습니다: {e}")
            finally:
                os.remove(tmp_file_path) # 임시 파일 삭제

#  메인 화면 
st.title("❤️ 무제")
st.markdown("안녕하세요! 힘든 일이 있었나요? 편하게 이야기해주세요. 간단한 대화를 통해 마음 상태를 점검해볼게요.")

#  채팅 기록 표시 
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

#  메인 로직 
# 1. 시작 전
if not st.session_state.test_started:
    if st.button("상담 시작하기"):
        st.session_state.test_started = True
        with st.chat_message("assistant"):
            welcome_message = "안녕하세요. 현재의 상태를 편하게 이야기해주세요."
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
            st.markdown(welcome_message)
        st.rerun()

# 2. 상담 진행 중
elif not st.session_state.bot.is_test_finished():
    if prompt := st.chat_input("답변을 입력해주세요..."):
        # 사용자 답변 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 답변 처리 및 점수 계산
        analysis_result = st.session_state.bot.process_and_score_answer(prompt)
        with st.expander("답변 분석 결과 보기"):
            st.info(analysis_result)

        # 다음 질문 또는 결과 표시
        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                if not st.session_state.bot.is_test_finished():
                    # 공감 및 다음 질문 생성
                    response = st.session_state.bot.generate_empathetic_response_and_ask_question(prompt)
                else:
                    # 최종 결과 표시
                    response = st.session_state.bot.display_final_result()
                    if st.session_state.rag_ready:
                         response += "\n\n이제 업로드하신 PDF 문서에 대해 자유롭게 질문하실 수 있습니다."
                    else:
                         response += "\n\n상담이 종료되었습니다. 필요하시면 언제든 다시 찾아주세요."


            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)

# 3. 상담 종료 후 (RAG 질의응답)
else:
    if prompt := st.chat_input("자유롭게 질문하거나, PDF에 대해 물어보세요..."):
        # 사용자 질문 표시
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 챗봇 응답 생성
        with st.chat_message("assistant"):
            with st.spinner("답변을 찾고 있어요..."):
                # RAG가 준비되었다면 RAG 답변, 아니면 일반 답변
                if st.session_state.rag_ready:
                    # RAG 체인에 필요한 대화 기록 포맷팅
                    chat_history = [
                        {"role": msg["role"], "content": msg["content"]}
                        for msg in st.session_state.messages[:-1] # 마지막 질문 제외
                    ]
                    answer, context = st.session_state.bot.get_rag_answer(prompt, chat_history)
                    response = answer
                    with st.expander("참고한 문서 내용"):
                        st.write(context)
                else:
                    # RAG가 없을 때의 기본 응답
                    response = "상담이 종료되었습니다. 추가적인 도움이 필요하시면 전문가와 상담하는 것을 권장합니다."
            
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.markdown(response)