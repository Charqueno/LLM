import streamlit as st
from streamlit_mic_recorder import mic_recorder, speech_to_text
from streamlit_extras.bottom_container import bottom  # to position the widget on the bottom
from streamlit_chat_widget import chat_input_widget
from gtts import gTTS
import numpy as np
from streamlit_chat import message
from LLM import LLM_Model
import re

def Create_Audio():
    audio_file = open("speech.mp3", "rb")
    audio_bytes = audio_file.read()
    return st.audio(audio_bytes, format="audio/ogg")

def Text_to_Speech(text):
    if text:
        answer = text  # AQUI VA LA RESPUESTA DEL CHAT
        tts = gTTS(answer, lang='es')
        tts.save("speech.mp3")
        Create_Audio()

if 'bot_responses' not in st.session_state:
    st.session_state['bot_responses'] = ["Hola. ¿En que puedo ayudarte hoy?"]
if 'user_responses' not in st.session_state:
    st.session_state['user_responses'] = [{'input':'', 'ner':[]}]

input_container = st.container()
response_container = st.container()

user_input = speech_to_text(language="es", use_container_width=True, just_once=True, key='STT')

with response_container:
    if user_input:
        print(user_input)
        response, ner = LLM_Model(user_input) #'hola'
        response = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
        st.session_state.user_responses.append({'input':user_input, 'ner':ner})
        st.session_state.bot_responses.append(response) #Text_to_Speech(response)

    if st.session_state['bot_responses']:
        for i in range(len(st.session_state['bot_responses'])):
            message(st.session_state['user_responses'][i]['input'], is_user=True, key=str(i) + '_user', avatar_style="initials",
                    seed="Kavita")
            #message(Text_to_Speech(st.session_state['bot_responses'][i]), key=str(i), avatar_style="initials", seed="AI", )
            message(st.session_state['bot_responses'][i], key=str(i), avatar_style="initials",
                    seed="AI", )
        print('La sesion del usuario: ', st.session_state['user_responses'])
        print('La sesion del Bot: ', st.session_state['bot_responses'])

with input_container:
    display_input = user_input