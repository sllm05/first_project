import streamlit as st
import os
import tempfile # 임시 파일 처리를 위해 추가
from langchain_community.document_loaders import PyPDFLoader # PDF 로드를 위해 추가
from model import EmotionBasedPsychotherapy
from data_loader import load_environment_and_client, load_emotion_data, load_markdown_retriever

# 페이지 설정
st.set_page_config(page_title="정신감정", page_icon="❤️", layout="wide")

# 세션 상태 초기화
if 'bot' not in st.session_state:
    try:
        client = load_environment_and_client()
        emotion_df = load_emotion_data()
        md_retriever = load_markdown_retriever()
        st.session_state.bot = EmotionBasedPsychotherapy(client, emotion_df, md_retriever)

        st.session_state.messages = []
        st.session_state.test_started = False
        st.session_state.uploaded_pdf_text = None # 업로드된 PDF 텍스트 저장용
    except Exception as e:
        st.error(f"초기화 오류: {e}. .env 파일, 데이터 파일 경로, 라이브러리 설치를 확인하세요.")
        st.stop()

# --- 사이드바 수정: PDF 업로드 기능과 생성 기능 공존 ---
with st.sidebar:
    st.header("참고 자료")
    uploaded_file = st.file_uploader("상담에 참고할 PDF 파일을 올려주세요.", type="pdf")

    # 파일이 업로드되면 텍스트로 변환하여 세션에 저장
    if uploaded_file:
        with st.spinner("PDF 파일을 분석하고 있어요..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            try:
                loader = PyPDFLoader(tmp_file_path)
                pages = loader.load_and_split()
                # 모든 페이지의 텍스트를 하나로 합침
                st.session_state.uploaded_pdf_text = "\n".join(page.page_content for page in pages)
                st.success("PDF 분석 완료! 상담 시 참고됩니다.")
            except Exception as e:
                st.error(f"PDF 처리 중 오류: {e}")
            finally:
                os.remove(tmp_file_path)

    st.divider() # 구분선 추가

    st.header("진단 결과서")
    st.write("상담이 완료되면, 아래에서 결과서를 받을 수 있습니다.")

    # 테스트가 종료되었을 때만 버튼 활성화
    if st.session_state.get('test_finished', False):
        if st.button("PDF 결과서 생성하기"):
            with st.spinner("결과 보고서를 생성하고 있어요..."):
                try:
                    # 1. 대화 내용 및 업로드된 PDF 텍스트로 요약
                    report_data = st.session_state.bot.summarize_for_report(
                        uploaded_pdf_text=st.session_state.uploaded_pdf_text
                    )

                    # 2. PDF 파일 생성
                    output_filename = "depression_report.pdf"
                    st.session_state.bot.create_report_pdf(report_data, output_filename)

                    # 3. 다운로드 링크 제공
                    with open(output_filename, "rb") as pdf_file:
                        PDFbyte = pdf_file.read()

                    st.download_button(
                        label="결과서 다운로드",
                        data=PDFbyte,
                        file_name=output_filename,
                        mime='application/octet-stream'
                    )
                    st.success("PDF 파일이 성공적으로 생성되었습니다!")
                except Exception as e:
                    st.error(f"PDF 생성 중 오류가 발생했습니다: {e}")

# --- 이하 메인 화면 및 로직은 이전과 동일 ---
st.title("❤️ 마음 상담 챗봇")
st.markdown("안녕하세요! 힘든 일이 있었나요? 편하게 이야기해주세요. 간단한 대화를 통해 마음 상태를 점검해볼게요.")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not st.session_state.test_started:
    if st.button("상담 시작하기"):
        st.session_state.test_started = True
        with st.chat_message("assistant"):
            welcome_message = "안녕하세요. 현재의 상태를 편하게 이야기해주세요."
            st.session_state.messages.append({"role": "assistant", "content": welcome_message})
            st.markdown(welcome_message)
        st.rerun()

elif not st.session_state.bot.is_test_finished():
    if prompt := st.chat_input("답변을 입력해주세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        analysis_result = st.session_state.bot.process_and_score_answer(prompt)
        with st.expander("답변 분석 결과 보기"):
            st.info(analysis_result)

        with st.chat_message("assistant"):
            with st.spinner("생각 중..."):
                response = st.session_state.bot.generate_empathetic_response_and_ask_question(prompt)
                st.session_state.messages.append({"role": "assistant", "content": response})
                st.markdown(response)

        if st.session_state.bot.is_test_finished():
            st.session_state.test_finished = True
            with st.chat_message("assistant"):
                final_result = st.session_state.bot.display_final_result()
                final_result += "\n\n상담이 종료되었습니다. 이제 우울증에 대해 궁금한 점을 자유롭게 질문하시거나, 사이드바에서 결과서를 받아보세요."
                st.session_state.messages.append({"role": "assistant", "content": final_result})
                st.markdown(final_result)
            st.rerun()

else:
    if prompt := st.chat_input("우울증에 대해 궁금한 점을 물어보세요..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("답변을 찾고 있어요..."):
                answer = st.session_state.bot.get_info_from_md(prompt)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.markdown(answer)