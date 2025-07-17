===
               AI 기반 우울증 자가 진단 챗봇 🌿
===
📝 프로젝트 소개
----------------------------------------------------------------------
본 프로젝트는 Streamlit과 Upstage Solar LLM을 활용하여 개발된 AI 기반 우울증 자가 진단 챗봇입니다. 
사용자와의 다단계 대화를 통해 심리 상태를 분석하고, RAG(검색 증강 생성) 기술을 접목하여 전문적인 정보를 바탕으로 한 분석 리포트를 제공합니다. 
최종 결과는 PDF 파일로 다운로드할 수 있습니다.


✨ 주요 기능
----------------------------------------------------------------------
- **단계별 대화형 진단**: 사용자 정보 입력 → 5가지 선별 질문 → 서술형 심정 묘사 → 최종 분석의 흐름으로 진행됩니다.
- **AI 기반 답변 분석**: 사용자의 답변을 LLM이 분석하여 우울감 점수를 실시간으로 채점합니다.
- **공감형 대화 생성**: 감성대화 말뭉치를 Few-shot 예시로 활용하여 따뜻하고 공감적인 챗봇 응답을 생성합니다.
- **RAG 기반 정보 제공**: 우울증 관련 전문 지식을 바탕으로 신뢰도 높은 분석과 조언을 제공합니다.
- **개인 맞춤형 최종 보고서**: 사용자의 모든 입력 데이터를 종합하여 맞춤형 최종 분석 리포트를 생성합니다.
- **PDF 결과서 다운로드**: 최종 진단 결과서를 깔끔한 포맷의 PDF 파일로 다운로드하는 기능을 제공합니다.


⚙️ 기술 스택 및 아키텍처
----------------------------------------------------------------------
- **UI & Application Flow**: Streamlit
- **Core AI Model**: Upstage Solar LLM
- **LLM Orchestration**: LangChain
- **Data Handling**: Pandas
- **RAG Implementation**: UpstageEmbeddings, ChromaDB
- **PDF Generation**: ReportLab

### 아키텍처
- **app.py**: Streamlit을 사용하여 전체 UI를 구성하고, 각 진단 단계(phase)를 관리하는 메인 애플리케이션 파일입니다.
- **data_loader.py**: 필요한 데이터(감성대화 말뭉치, 우울증 지식 문서)를 로드하고, LangChain을 이용해 텍스트를 분할, 임베딩하여 RAG 검색기(Retriever)를 생성합니다.
- **model.py**: 챗봇의 핵심 로직이 담긴 파일입니다. 사용자의 답변을 분석/채점하고, 공감형 응답을 생성하며 RAG를 활용해 최종 분석 리포트를 작성하는 클래스를 포함합니다.


🚀 설치 및 실행 방법
----------------------------------------------------------------------

### 1. 프로젝트 복제 (Clone a project)
   git clone https://your-repository-url.git
   cd your-project-directory

### 2. 가상 환경 설정 및 패키지 설치 (Set up a virtual environment & Install packages)
   # 가상환경 활성화 (권장)
   python activate (가상환경이름)

   # 필요한 패키지 설치
   pip install -r requirements.txt


### 3. 환경 변수 설정 (Set up environment variables)
   프로젝트 루트 디렉토리에 `.env` 파일을 생성하고, 발급받은 Upstage API 키를 입력합니다.


### 4. 데이터 및 폰트 파일 준비 (Prepare data and font files)
   폰트 파일은 나눔고딕 등을 다운로드하여 사용하세요.

   이 모델에 사용된 폰트 : NanumGothic.ttf

### 5. 애플리케이션 실행 (Run the application)
   터미널에서 아래 명령어를 실행합니다.

   streamlit run app.py
