from __future__ import annotations

import streamlit as st


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1450px;}
        [data-testid="stMetric"] {background: rgba(255,255,255,0.04); border: 1px solid rgba(128,128,128,0.18); padding: 0.8rem; border-radius: 0.8rem;}
        .hero {padding: 1.4rem 1.6rem; border-radius: 1rem; background: linear-gradient(120deg, rgba(20,110,95,.16), rgba(38,92,180,.12)); border: 1px solid rgba(100,150,160,.25); margin-bottom: 1rem;}
        .hero h1 {margin: 0; font-size: 2.1rem;}
        .hero p {margin: .45rem 0 0 0; opacity: .86;}
        .result-good {padding: 1rem; border-left: 5px solid #1f9d72; border-radius: .5rem; background: rgba(31,157,114,.10);}
        .result-watch {padding: 1rem; border-left: 5px solid #d99a23; border-radius: .5rem; background: rgba(217,154,35,.10);}
        .small-note {font-size: .88rem; opacity: .75;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def model_disclaimer() -> None:
    st.caption(
        "Decision-support model only. Forecasts are estimates derived from the supplied "
        "dataset and synthetic target rules; they are not financial advice or a guarantee of returns."
    )
