# === Python Module ===
import streamlit as st
import requests
import uuid


# === Main Streamlit UI Body ===
def main() -> None:
    """
    Streamlit Chat UI for Mini RAG Agent
    (Markdown-based simple chatbot layout)
    """

    # === Webpage Config ===
    st.set_page_config(
        page_title = "Mini RAG Agent",
        page_icon = "🧠",
        layout = "centered"
    )

    # === Header ===
    st.title("🧠 Mini RAG Agent")

    # === Sidebar ===
    with st.sidebar:
        st.divider()
        st.caption("Mini RAG Agent — Markdown Chat UI")

    # === Initialize Chat History ===
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # === Create or Retrieve Session UUID ===
    if "unique_id" not in st.session_state:
        st.session_state.unique_id = str(uuid.uuid4())

    # === Backend Endpoint URL ===
    FASTAPI_URL = "http://127.0.0.1:8000/generate"

    # === Chat Display Section ===
    st.markdown("### 💬 Conversation")

    # Render each message using markdown
    for chat in st.session_state.chat_history:
        if chat["role"] == "user":
            st.markdown(f"**🧑‍💻 You:** {chat['content']}")
        else:
            st.markdown(f"**🤖 Agent:** {chat['content']}")

    # === Input Area ===
    user_input = st.chat_input("Type your message here...")

    # === Handle Input ===
    if user_input:
        # Add user message
        st.session_state.chat_history.append({
            "role": "user",
            "content": user_input
        })

        try:
            # === Send Query to FastAPI Backend ===
            response = requests.post(
                FASTAPI_URL,
                json = {
                    "unique_id": st.session_state.unique_id,
                    "query": user_input
                },
                timeout = 60
            )

            if response.status_code == 200:
                data = response.json()
                bot_response = data.get("answer", "No response generated.")
            else:
                bot_response = f"⚠️ Backend Error: {response.status_code}"

        except Exception as e:
            bot_response = f"❌ Connection Error: {e}"

        # Add bot response
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": bot_response
        })

        # Trigger rerun to refresh conversation
        st.rerun()


# === Entry Point ===
if __name__ == "__main__":
    main()