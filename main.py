import box
import timeit
import yaml
import argparse
from dotenv import find_dotenv, load_dotenv
from src.utils import setup_dbqa
import streamlit as st
from vid_conv_db_build import run_db_build

# Load environment variables from .env file
load_dotenv(find_dotenv())

# Import config vars
with open('config/config.yml', 'r', encoding='utf8') as ymlfile:
    cfg = box.Box(yaml.safe_load(ymlfile))


if __name__ == "__main__":

    st.title("Chatbot using LLAMA2")

    # Sidebar for video upload
    st.sidebar.title("Upload a Video")
    uploaded_video = st.sidebar.file_uploader("Choose a video file", type=["mp4"])

    # Check if a new video has been uploaded
    if uploaded_video and "video_uploaded" not in st.session_state:
        # Run DB build and setup only when a new video is uploaded
        st.session_state["video_uploaded"] = True
        run_db_build(uploaded_video)
        
        # Initialize session state for storing chat history if not already initialized
        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        # Display video preview in the sidebar
        st.sidebar.video(uploaded_video)

    # Input field for user query
    dbqa = setup_dbqa()
    user_input = st.text_input("Enter your query:", "")

    # Process the query if user input is provided
    if user_input:
        # Make sure dbqa is only set after a video is uploaded
        if "video_uploaded" in st.session_state:
            start = timeit.default_timer()
            response = dbqa({"query": user_input})
            answer = response.get("result", "No answer found.")
            end = timeit.default_timer()

            # Append user input and bot response to chat history
            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "bot", "content": answer})
            print('='*50)
            print(f"Time to retrieve response: {end - start}")

    # Display chat history
    for message in st.session_state.get("messages", []):
        if message["role"] == "user":
            st.markdown(
                f"""
                <div style="background-color: #d9f7be; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                    <strong>You:</strong> {message['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div style="background-color: #e6f7ff; padding: 10px; border-radius: 10px; margin-bottom: 5px;">
                    <strong>Bot:</strong> {message['content']}
                </div>
                """,
                unsafe_allow_html=True,
            )
