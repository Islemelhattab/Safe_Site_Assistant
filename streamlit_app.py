import streamlit as st
from rag_chain import SafeSiteRAG

st.set_page_config(page_title="SafeSite Assistant", layout="wide")
st.title(" SafeSite Assistant")

if "rag" not in st.session_state:
    st.session_state.rag = SafeSiteRAG(search_type="similarity")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg and msg["sources"]:
            with st.expander("Sources citées"):
                for s in msg["sources"]:
                    st.write(f"- **{s['file']} (p. {s['page']})**: {s['excerpt']}")

if prompt := st.chat_input("Ask your question about security..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching security documents..."):
            response = st.session_state.rag.query(prompt)
            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander("Sources cited"):
                    for s in response["sources"]:
                        st.write(f"- **{s['file']} (p. {s['page']})**: {s['excerpt']}")

    st.session_state.messages.append({
        "role": "assistant",
        "content": response["answer"],
        "sources": response["sources"]
    })