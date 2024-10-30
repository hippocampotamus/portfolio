

# !pip install -q sentence-transformers

# !pip install -q gradio

import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# import gradio as gr

# 임베딩 모델 로드
encoder = SentenceTransformer('jhgan/ko-sroberta-multitask')

# 식당 관련 질문과 답변 데이터
questions = [
    "회원가입이 필요한가요?",
    "데이터는 뭘 이용했나요?",
    "앱의 기능은 무엇인가요?",
    "나의 보고 내역을 확인할 수 있나요?",
    "원하는 기능이 존재하지 않아요.",
    "쓰레기 분류가 잘 되지 않아요.",
    "쓰레기는 어떠한 종류를 분류할 수 있나요?",
    "사진을 분류하는 건 어떻게 하는 건가요?",
    "담당 지자체를 변경하고 싶어요.",
        "바다환경지킴이를 변경하고 싶어요.",
        "분류되는 쓰레기의 종류가 부족해요.",
    "직접 수거할 수 없는 쓰레기가 나왔어요."
]

answers = [
    "회원가입은 필수입니다.",
    "AIhub의 데이터를 기초로 데이터를 구축하였으며, 추가데이터셋은 웹스크롤링을 통해 구축하였습니다.",
    "바다환경지킴이를 위한 기능은 일일 정화내역 보고와 개인 보고 내역 대쉬보드입니다. 바다환경지킴이 담당 공무원를 위한 기능은 담당 인원의 보고 내역 대쉬보드와 바다환경지킴이 연락처 리스트입니다.",
    "네, 확인 가능합니다. ~에서 ~로 들어가시면 됩니다.",
    "필요한 기능을 챗봇에 질문해주시면 추후 업데이트하도록 하겠습니다.",
    "불편을 끼쳐 죄송합니다. 당신의 분류 덕분에 분류 기능이 나아지고 있습니다.",
    "~, ~, ~ 총 ~종류를 분류할 수 있습니다.",
    "분류 ai는 ~를 통해 제작되었습니다.",
    "~에서 변경하실 수 있습니다.",
    "~에서 변경하실 수 있습니다.",
    "필요한 쓰레기 종류를 챗봇에 질문해주시면 추후 업데이트하도록 하겠습니다.",
    "사진을 찍은 후, ~에서 ~ 버튼을 누르시면 직접 수거할 수 없는 쓰레기가 담당 공무원에게 보고 됩니다."
]

# 대화 이력을 저장하기 위한 Streamlit 상태 설정
if 'history' not in st.session_state:
    st.session_state.history = []

# 챗봇 함수 정의
def get_response(user_input):
    # 사용자 입력 임베딩
    embedding = encoder.encode(user_input)
    
    # 유사도 계산하여 가장 유사한 응답 찾기
    df['distance'] = df['embedding'].map(lambda x: cosine_similarity([embedding], [x]).squeeze())
    answer = df.loc[df['distance'].idxmax()]

    # 대화 이력에 추가
    st.session_state.history.append({"user": user_input, "bot": answer['챗봇']})

# Streamlit 인터페이스
st.title("식당 챗봇")
st.write("클리닝웨이브 앱에 관한 질문을 입력해보세요. 예: 담당 지자체를 변경하고 싶어요.")

user_input = st.text_input("user", "")

if st.button("Submit"):
    if user_input:
        get_response(user_input)
        user_input = ""  # 입력 초기화

# 대화 이력 표시
for message in st.session_state.history:
    st.write(f"**사용자**: {message['user']}")
    st.write(f"**챗봇**: {message['bot']}")












