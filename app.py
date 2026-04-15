import streamlit as st
import pandas as pd
import urllib.parse
from utils import load_data, vectorize, search_papers, get_answer
from pdf_utils import extract_text_from_pdf
from styles import load_css
import time


def get_answer_from_pdf(query, pdf_text):
    sentences = pdf_text.split('.')

    for s in sentences:
        if query.lower() in s.lower():
            return "📄 From PDF:\n" + s.strip()

    return "No relevant answer found in PDF."

# Page config
st.set_page_config(page_title="AI Research Assistant", layout="wide")

# Sidebar Navigation
page = st.sidebar.radio("📌 Navigate", ["🏠 Home", "🔍 Search", "📊 Dashboard", "💬 Chatbot"])

# Load CSS
st.markdown(load_css(), unsafe_allow_html=True)

# Sidebar Info
st.sidebar.title("📌 About")
st.sidebar.info("AI-powered research assistant using IRTM")

# Load dataset
df = load_data()
vectorizer, tfidf_matrix = vectorize(df)

if page == "🏠 Home":

    st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🤖 AI Research Assistant</h1>", unsafe_allow_html=True)

    st.markdown("<h3 style='text-align: center;'>Welcome to your intelligent research companion</h3>", unsafe_allow_html=True)

    st.markdown("---")

    # Logo (optional)
    st.image("https://cdn-icons-png.flaticon.com/512/4712/4712109.png", width=150)

    st.markdown("""
    ### 🔍 What this app does:
    - Search research papers  
    - Upload PDF and analyze  
    - Get AI-generated answers  
    - View analytics dashboard  

    ---
    """)

    st.success("👉 Use the sidebar to navigate to Search or Dashboard")

    if st.button("🚀 Start Exploring"):
        st.info("👉 Go to Search from sidebar")

# ============================
# 🔍 SEARCH PAGE
# ============================
if page == "🔍 Search":

    st.markdown("<div class='title'>🤖 AI Research Assistant</div>", unsafe_allow_html=True)
    st.markdown("<div class='subtitle'>Smart Search • PDF Analysis • AI Insights</div>", unsafe_allow_html=True)
    st.markdown("---")

    # Upload PDF
    st.markdown("### 📂 Upload Research Paper")
    uploaded_file = st.file_uploader("", type="pdf")

    if uploaded_file:
        pdf_text = extract_text_from_pdf(uploaded_file)
        st.write("📄 Preview:", pdf_text[:500])

    # Search UI
    col1, col2, col3 = st.columns([1,2,1])

    with col2:
        query = st.text_input("🔍 Enter your research topic:")
        search_btn = st.button("🚀 Search")

    # Results
    if search_btn:
        if query:
            results = search_papers(query, df, vectorizer, tfidf_matrix)

            st.markdown("## 📄 Top Research Papers 🔍")
            st.info("Showing most relevant papers based on your query")

            for i in results[:5]:
                title = df.iloc[i[0]]['title']
                abstract = df.iloc[i[0]]['abstract']

                # Links
                google_url = "https://scholar.google.com/scholar?q=" + urllib.parse.quote(title)
                arxiv_url = "https://arxiv.org/search/?query=" + urllib.parse.quote(title) + "&searchtype=all"

                # 🔹 Card UI
                st.markdown(f"""
                <div style='background:#1e1e1e;padding:15px;border-radius:12px;margin-bottom:15px'>
                    <h3 style='color:#4CAF50'>{title}</h3>
                    <p><b>Score:</b> {round(i[1],2)}</p>
                    <p>{abstract}</p>

                    <div style="margin-top:10px;">
                        <a href="{google_url}" target="_blank" style="margin-right:15px;">🔗 Google Scholar</a>
                        <a href="{arxiv_url}" target="_blank">📄 arXiv</a>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 🔥 BUTTON MUST BE OUTSIDE
                st.link_button("🌐 Open in Browser", google_url)

            st.markdown("## 🤖 AI Answer")
            st.success(get_answer(query, df, vectorizer, tfidf_matrix))

        else:
            st.warning("Enter a topic")


# ============================
# 📊 DASHBOARD PAGE
# ============================
elif page == "📊 Dashboard":

    st.title("📊 Research Dashboard")

    # Stats
    st.subheader("📌 Dataset Info")
    st.write("Total Papers:", len(df))

    # Table
    st.subheader("📄 Sample Data")
    st.dataframe(df.head())

    # Graph
    from collections import Counter
    import matplotlib.pyplot as plt

    words = " ".join(df['cleaned']).split()
    common = Counter(words).most_common(10)

    labels = [i[0] for i in common]
    values = [i[1] for i in common]

    fig, ax = plt.subplots()
    ax.bar(labels, values)
    ax.set_title("Top Keywords")

    st.pyplot(fig)


elif page == "💬 Chatbot":

    st.title("💬 AI Research Chatbot")

    # ✅ Initialize FIRST
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Show previous messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # ✅ Input box MUST be inside block
    user_input = st.chat_input("Ask a question...")

    if user_input:
        # Store user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate answer
        answer = get_answer(user_input, df, vectorizer, tfidf_matrix)

        with st.chat_message("assistant"):
            st.markdown(answer)

        # Store bot response
        st.session_state.messages.append({"role": "assistant", "content": answer})