import streamlit as st
import joblib
import os
from collections import Counter
import matplotlib.pyplot as plt

# Page config
st.set_page_config(page_title="BOT PAPA Xóc Đĩa AI", page_icon="🎲")

# Text dictionary
TEXT = {
    "title": "🎲 BOT PAPA xocdia AI",
    "history": "📜 Kết quả gần nhất",
    "input": "🎮 Nhập kết quả tiếp theo",
    "undo": "🔙 Quay lại",
    "reset": "🧹 Xóa tất cả",
    "no_result": "*Chưa có kết quả.*",
    "prediction": "🔮 Dự đoán AI",
    "suggestion": "🎯 Gợi ý ưu tiên",
    "group": "🧠 Nhóm cược (Chẵn / Lẻ)",
    "stats": "📊 Tần suất 10 ván gần nhất",
}

# Load model
MODEL_PATH = "xocdia_model.pkl"
model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None

# Session state
if "history" not in st.session_state:
    st.session_state.history = []

# Title
st.title(TEXT["title"])

# History display
st.markdown(f"### {TEXT['history']}")
if st.session_state.history:
    result_str = " | ".join(str(num) for num in st.session_state.history[-20:])
    st.markdown(f"<div style='font-size:24px; font-weight:bold;'>{result_str}</div>", unsafe_allow_html=True)
else:
    st.markdown(TEXT["no_result"])

# Input buttons
st.markdown(f"### {TEXT['input']}")
cols = st.columns(6)

def click_number(n):
    st.session_state.history.append(n)
    st.rerun()

def undo_last():
    if st.session_state.history:
        st.session_state.history.pop()
        st.rerun()

def reset_all():
    st.session_state.history = []
    st.rerun()

with cols[0]:
    if st.button("0️⃣"): click_number(0)
with cols[1]:
    if st.button("1️⃣"): click_number(1)
with cols[2]:
    if st.button("2️⃣"): click_number(2)
with cols[3]:
    if st.button("3️⃣"): click_number(3)
with cols[4]:
    if st.button("4️⃣"): click_number(4)
with cols[5]:
    if st.button(TEXT["reset"]): reset_all()

if st.button(TEXT["undo"]):
    undo_last()

# Prediction
if len(st.session_state.history) >= 3:
    input_seq = st.session_state.history[-3:]
    pred = model.predict([input_seq])[0] if model else None

    if pred is not None:
        st.markdown(f"### {TEXT['prediction']}")
        st.markdown(f"<h4><strong>{pred}</strong></h4>", unsafe_allow_html=True)

        group = "Chẵn" if pred in [0,2,4] else "Lẻ"
        st.markdown(f"### {TEXT['group']}")
        st.markdown(f"➡️ <strong>{group}</strong>", unsafe_allow_html=True)

        st.markdown(f"### {TEXT['suggestion']}")
        st.markdown(f"👉 Ưu tiên: <strong>{group} - {pred}</strong>", unsafe_allow_html=True)

# Statistics
if len(st.session_state.history) >= 10:
    recent = st.session_state.history[-10:]
    freq = Counter(recent)
    st.markdown(f"### {TEXT['stats']}")
    for n in range(5):
        st.markdown(f"- {n}: {freq.get(n,0)} lần")

    # Biểu đồ
    fig, ax = plt.subplots()
    nums = [0,1,2,3,4]
    counts = [freq.get(n,0) for n in nums]
    ax.bar(nums, counts, color="orange")
    ax.set_xlabel("Kết quả (số đỏ)")
    ax.set_ylabel("Tần suất")
    ax.set_title("10 ván gần nhất")
    st.pyplot(fig)