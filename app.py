from pathlib import Path
import re

import streamlit as st

from model_utils import generate_response_stream, load_model_and_tokenizer

st.set_page_config(
    page_title="AI Chat Interface",
    page_icon="AI",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_data(show_spinner=False)
def load_css(file_name):
    css_path = Path(__file__).resolve().parent / file_name
    if not css_path.exists():
        return ""
    return css_path.read_text(encoding="utf-8")


def extract_thinking_and_answer(buffer_text):
    """
    Parse streamed text into:
    - visible thinking summary between <think>...</think>
    - final answer outside think tags
    """
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)
    think_blocks = [
        chunk.strip()
        for chunk in think_pattern.findall(buffer_text)
        if chunk and chunk.strip()
    ]
    completed_thinking = "\n\n".join(think_blocks).strip()

    latest_open_start = buffer_text.lower().rfind("<think>")
    latest_close = buffer_text.lower().rfind("</think>")
    partial_thinking = ""
    answer_candidate = buffer_text

    if latest_open_start != -1 and latest_close < latest_open_start:
        partial_thinking = buffer_text[latest_open_start + len("<think>") :].strip()
        answer_candidate = buffer_text[:latest_open_start]

    answer_text = think_pattern.sub("", answer_candidate)
    answer_text = answer_text.replace("<think>", "").replace("</think>", "").strip()

    if partial_thinking:
        if completed_thinking:
            completed_thinking = f"{completed_thinking}\n\n{partial_thinking}"
        else:
            completed_thinking = partial_thinking

    return completed_thinking.strip(), answer_text.strip()


css_content = load_css("style.css")
if css_content:
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


if "messages" not in st.session_state:
    st.session_state.messages = []
if "model_ready" not in st.session_state:
    st.session_state.model_ready = False
if "device_name" not in st.session_state:
    st.session_state.device_name = "Unknown"


with st.sidebar:
    st.title("Settings")
    st.markdown("Configure inference parameters.")

    st.markdown("### Generation Settings")
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=2.0,
        value=0.7,
        step=0.1,
        help="Higher values make output more random, lower values more focused.",
    )
    max_tokens = st.slider(
        "Max New Tokens",
        min_value=64,
        max_value=2048,
        value=512,
        step=64,
    )
    top_p = st.slider(
        "Top P",
        min_value=0.1,
        max_value=1.0,
        value=0.9,
        step=0.05,
        help="Nucleus sampling threshold.",
    )
    max_history_messages = st.slider(
        "History Window (messages)",
        min_value=4,
        max_value=30,
        value=12,
        step=2,
        help="Only the latest messages in this window are sent to the model.",
    )

    st.divider()

    if st.button("Clear Chat History", use_container_width=True, type="primary"):
        st.session_state.messages = []
        st.rerun()

    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.markdown(
        "<div style='text-align: center; color: #888; font-size: 0.8rem;'>Powered by Streamlit + llama.cpp</div>",
        unsafe_allow_html=True,
    )


st.markdown("<h1 class='title-anim'>Local LLM Chat</h1>", unsafe_allow_html=True)
st.markdown("Start typing below.")

try:
    with st.spinner("Initializing llama.cpp server..."):
        reasoning_mode = "off"
        client, _, device_name = load_model_and_tokenizer(
            "local-qwen",
            reasoning_mode=reasoning_mode,
            reasoning_format="deepseek-legacy",
            reasoning_budget=256,
        )
        if not st.session_state.model_ready:
            st.session_state.model_ready = True
            st.session_state.device_name = device_name
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

st.caption(f"Runtime: {st.session_state.device_name}")


for message in st.session_state.messages:
    role = message.get("role")
    content = message.get("content")
    if role in {"user", "assistant"} and content:
        with st.chat_message(role):
            st.markdown(content)


if prompt := st.chat_input("Message the AI..."):
    prompt = prompt.strip()
    if not prompt:
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                streamer = generate_response_stream(
                    client,
                    "qwen-2.5-0.5b",
                    st.session_state.messages,
                    max_new_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    max_history_messages=max_history_messages,
                )

                full_stream_text = ""
                final_answer_text = ""
                answer_placeholder = st.empty()
                
                import time
                with st.status("Analyzing request...", expanded=True) as status:
                    st.write("Processing context...")
                    time.sleep(1)
                    st.write("Formulating response...")
                    time.sleep(1)
                    status.update(label="Complete", state="complete", expanded=False)

                for chunk in streamer:
                    piece = str(chunk)
                    full_stream_text += piece
                    
                    # Always strip reasoning tags regardless of mode now to hide real thoughts if they sneak in
                    thinking_text, final_answer_text = extract_thinking_and_answer(
                        full_stream_text
                    )

                    answer_placeholder.markdown(final_answer_text or "...")

                response_text = final_answer_text.strip()
                if not response_text:
                    response_text = "No response was generated. Please try again."

                st.session_state.messages.append(
                    {"role": "assistant", "content": response_text}
                )

            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I hit an internal error while generating a response. Please retry.",
                    }
                )
