from __future__ import annotations
import streamlit as st
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    # We use st.secrets because os.getenv doesn't work on Streamlit Cloud
    alpha_vantage_api_key: str = st.secrets.get("ALPHA_VANTAGE_API_KEY", "")
    newsapi_key: str = st.secrets.get("NEWSAPI_KEY", "")

settings = Settings()



