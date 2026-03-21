"""
Streamlit Chat UI for the Orchestrator Agent.

Dashboard'a entegre edilebilir chat widget.
Kullanim: Dogrudan calistir veya sprint1_dashboard.py icinden import et.

    streamlit run src/agent/chat_ui.py
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

try:
    import streamlit as st
except ImportError:
    print("Streamlit gerekli: pip install streamlit")
    sys.exit(1)

from src.agent.orchestrator import OrchestratorAgent


def render_chat_page():
    """Render the orchestrator chat page (can be embedded in dashboard)."""
    st.header("EdgeAgent Orchestrator")
    st.caption(
        "Pipeline'i chat ile kontrol edin: veri analizi, augmentation, "
        "egitim, degerlendirme, deployment"
    )

    # Initialize agent in session state
    if "orchestrator" not in st.session_state:
        st.session_state.orchestrator = OrchestratorAgent()
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    agent = st.session_state.orchestrator

    # Display chat history
    for msg in st.session_state.chat_messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    # Chat input
    if prompt := st.chat_input("Komut yazin... (ornek: 'Benim ne kadar datam var?')"):
        # Show user message
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Get agent response
        with st.chat_message("assistant"):
            with st.spinner("Islem yapiliyor..."):
                response = agent.process_message(prompt)
            st.markdown(f"```\n{response}\n```")

        st.session_state.chat_messages.append({"role": "assistant", "content": f"```\n{response}\n```"})

    # Sidebar: Quick actions
    with st.sidebar:
        st.subheader("Hizli Komutlar")
        col1, col2 = st.columns(2)

        with col1:
            if st.button("Veri Analizi", use_container_width=True):
                _quick_action(agent, "Benim ne kadar datam var?")
            if st.button("Label Kontrol", use_container_width=True):
                _quick_action(agent, "Label'lar temiz mi?")
            if st.button("Kurallar", use_container_width=True):
                _quick_action(agent, "Hangi urun tipleri var?")

        with col2:
            if st.button("Geri Bildirim", use_container_width=True):
                _quick_action(agent, "Geri bildirimleri analiz et")
            if st.button("Accuracy", use_container_width=True):
                _quick_action(agent, "Accuracy nedir?")
            if st.button("Yardim", use_container_width=True):
                _quick_action(agent, "help")


def _quick_action(agent: OrchestratorAgent, command: str):
    """Execute a quick action and add to chat."""
    st.session_state.chat_messages.append({"role": "user", "content": command})
    response = agent.process_message(command)
    st.session_state.chat_messages.append({"role": "assistant", "content": f"```\n{response}\n```"})
    st.rerun()


def main():
    st.set_page_config(
        page_title="EdgeAgent Orchestrator",
        page_icon="🔧",
        layout="wide",
    )
    render_chat_page()


if __name__ == "__main__":
    main()
