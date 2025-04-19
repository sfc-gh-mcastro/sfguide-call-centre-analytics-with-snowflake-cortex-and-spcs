import streamlit as st
from src.Cortexanalyst import cortexanlyst
session = st.connection('snowflake').session()


st.title("text2sql")
cortexanlyst(session=session)