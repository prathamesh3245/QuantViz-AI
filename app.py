import time 
import random
import streamlit as st

while True:
    fraud_score = random.uniform(0.85, 0.99)
    st.metric("Current BTC Transaction", f"{fraud_score:.2%} Fraud Risk")
    time.sleep(3)