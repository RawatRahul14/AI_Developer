# === Python Module ===
import streamlit as st


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
        st.header("⚙️ Settings")
        st.write("Configuration options will appear here later.")
        st.divider()
        st.caption("Mini RAG Agent — Markdown Chat UI")

    # === Initialize Chat History ===
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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

        # Add placeholder bot response
        bot_response = "_(Backend not connected yet - placeholder response)_"
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": bot_response
        })

        # Trigger rerun to refresh conversation
        st.rerun()


# === Entry Point ===
if __name__ == "__main__":
    main()
