import streamlit as st
import sqlite3
import os

# --- DB Helpers ---
def check_credentials(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = cursor.fetchone()
    conn.close()
    return user

# --- UI ---
st.set_page_config(page_title="Login", layout="centered")
st.title("🔐 Login")

username = st.text_input("Username")
password = st.text_input("Password", type="password")

if st.button("Login"):
    if check_credentials(username, password):
        st.success("Login successful")
        # Set a session flag and redirect
        st.session_state["authenticated"] = True
        st.session_state["username"] = username
        st.switch_page("pages/App.py")  # ⬅️ requires Streamlit 1.10+
    else:
        st.error("Invalid username or password")

