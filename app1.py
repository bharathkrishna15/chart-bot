import streamlit as st
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    AIMessagePromptTemplate,
    ChatPromptTemplate
)
from PIL import Image
import io
import base64

# Custom CSS for OmniThink theme
st.markdown("""
    <style>
        .main { background-color: #ffffff; color: #000000; }
        .sidebar .sidebar-content { background-color: #1a2b4e; }
        .stTextInput textarea { color:rgb(9, 9, 9) !important; }
        div[role="listbox"] div { background-color: #1a2b4e !important; color: white !important; }
        .stChatMessage { background-color:rgb(10, 10, 10); border-radius: 10px; padding: 10px; margin: 5px 0; }
    </style>
""", unsafe_allow_html=True)

# App Title
st.title("🐥 OmniThink")
st.caption("🤖 Your Friendly AI Assistant for Problem Solving and Innovation! 🚀")

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Features")
    
    # Model selection
    selected_model = st.selectbox(
        "Select AI Model",
        ["tinyllama", "llama2", "mistral", "phi"],
        index=0,
        help="Choose the AI model you want to use for generating responses."
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Lower values make responses more deterministic, while higher values increase creativity."
    )
    
    # Clear chat history button
    if st.button("🧹 Clear Chat History"):
        st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm OmniThink. How can I assist you today? 🚀"}]
        st.rerun()
    
    # Export chat logs
    if st.button("📥 Export Chat Logs"):
        chat_logs = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in st.session_state.message_log])
        st.download_button(
            label="Download Chat Logs",
            data=chat_logs,
            file_name="omnithink_chat_logs.txt",
            mime="text/plain"
        )
    
    # Default System Prompts
    system_prompt_options = {
        "Problem Solver": "You are an expert AI assistant. Provide concise, correct solutions with strategic insights for problem-solving.",
        "Creative Writer": "You are a creative writer. Generate engaging stories, poems, or scripts based on user input.",
        "Code Helper": "You are a coding assistant. Help users write, debug, and optimize code in any programming language."
    }
    custom_system_prompt = st.selectbox(
        "📝 Select a System Prompt",
        list(system_prompt_options.keys()),
        index=0,
        help="Choose how the AI should behave during the conversation."
    )
    custom_system_prompt = system_prompt_options[custom_system_prompt]
    st.divider()
    
    st.subheader("🖼️ Analyze Images")
    uploaded_image = st.file_uploader("Upload an image for analysis", type=["jpg", "jpeg", "png"], help="Upload an image, and the AI will analyze or describe it.")
    if uploaded_image:
        # Display the uploaded image
        image = Image.open(uploaded_image)
        st.image(image, caption="Uploaded Image", use_column_width=True)
# Ensure Ollama is running
try:
    llm_engine = ChatOllama(
        model=selected_model,  # Use the selected model
        base_url="http://localhost:11434",  # Ensure Ollama is running
        temperature=temperature  # Use the selected temperature
    )
except Exception as e:
    st.error(f"❌ Error connecting to Ollama: {str(e)}")
    st.stop()

# Initialize session state for chat history
if "message_log" not in st.session_state:
    st.session_state.message_log = [{"role": "ai", "content": "Hi! I'm OmniThink. How can I assist you today? 🚀"}]

# Display previous chat messages
for message in st.session_state.message_log:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# System prompt configuration
system_prompt = SystemMessagePromptTemplate.from_template(custom_system_prompt)

def build_prompt_chain():
    prompt_sequence = [system_prompt]
    for msg in st.session_state.message_log:
        if msg["role"] == "user":
            prompt_sequence.append(HumanMessagePromptTemplate.from_template(msg["content"]))
        elif msg["role"] == "ai":
            prompt_sequence.append(AIMessagePromptTemplate.from_template(msg["content"]))
    return ChatPromptTemplate.from_messages(prompt_sequence)

def generate_ai_response(prompt_chain):
    try:
        processing_pipeline = prompt_chain | llm_engine | StrOutputParser()
        return processing_pipeline.invoke({})
    except Exception as e:
        return f"❌ Error generating response: {str(e)}"


    # Convert image to base64 for processing
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    # Add image description task to chat
    user_query = f"Analyze this image: ![Image](data:image/png;base64,{img_base64})"
    st.session_state.message_log.append({"role": "user", "content": user_query})
    with st.spinner("🔍 Analyzing image... Please wait."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    st.rerun()

# Chat input handling
user_query = st.chat_input("Type your question or problem here...")
if user_query:
    st.session_state.message_log.append({"role": "user", "content": user_query})
    # Generate AI response
    with st.spinner("🔍 Analyzing... Please wait."):
        prompt_chain = build_prompt_chain()
        ai_response = generate_ai_response(prompt_chain)
    st.session_state.message_log.append({"role": "ai", "content": ai_response})
    # Refresh chat display using latest Streamlit method
    st.rerun()