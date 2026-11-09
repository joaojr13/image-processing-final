"""Ajustes visuais leves; os controles continuam sendo componentes do Streamlit."""
import streamlit as st


def aplicar_estilo():
    st.markdown('''<style>
    .block-container {max-width: 1160px; padding-top: 2.1rem; padding-bottom: 3rem;}
    h1 {letter-spacing: -0.045em; font-weight: 700;}
    h2,h3 {letter-spacing: -0.025em;}
    [data-testid="stHeader"] {background: transparent;}
    [data-testid="stToolbar"] {visibility: hidden;}
    [data-testid="stTabs"] button {font-size: 1rem;}
    [data-testid="stTabs"] [role="tablist"] {gap: 1.4rem; margin-bottom: 1.2rem;}
    [data-testid="stVerticalBlockBorderWrapper"] > div {border-radius: 16px;}
    [data-testid="stMetric"] {background: #fff; padding: 14px 18px; border: 1px solid #DEE8E4; border-radius: 14px;}
    .selo {display:inline-block; padding:6px 12px; border-radius:100px; background:#E4EFEB; color:#276153; font-size:12px; letter-spacing:.07em; font-weight:700;}
    .introducao {color:#5B716B; max-width:720px; margin: .2rem 0 1.6rem;}
    .etapas {display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:12px; margin:18px 0;}
    .etapa {background:white; border:1px solid #DEE8E4; border-radius:14px; padding:20px;}
    .etapa b {display:block; color:#20483E; margin-bottom:6px;}
    .etapa p {font-size:14px; color:#5B716B; margin:0; line-height:1.6;}
    .numero {font-size:12px; color:#16736B; font-weight:700; display:block; margin-bottom:12px;}
    @media(max-width:640px) {.block-container {padding-top:1rem;} .etapas {grid-template-columns:1fr;} [data-testid="stTabs"] [role="tablist"] {gap:.5rem;} }
    /* Tradução visual dos textos fixos do seletor de arquivos do Streamlit. */
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span {font-size:0;}
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::after {content:"Arraste uma foto aqui"; font-size:14px;}
    [data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-child {display:none;}
    [data-testid="stFileUploaderDropzone"] button {font-size:0;}
    [data-testid="stFileUploaderDropzone"] button::after {content:"Selecionar foto"; font-size:14px;}
    </style>''', unsafe_allow_html=True)
