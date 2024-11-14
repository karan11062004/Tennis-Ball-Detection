
import streamlit as st
import torch
import cv2
import tempfile
import numpy as np
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

model_path = "models/tplayer_best.pt"
model = torch.hub.load("ultralytics/yolov5", "custom", path=model_path)

st.title("Tennis Player Detection")

col1, col2 = st.columns(2)

with col1:
    st.write("## Video Preview")

with col2:
    st.write("## Actions")

    video_file = st.file_uploader("Upload a video", type=["mp4", "mov"])

    if 'processed' not in st.session_state:
        st.session_state['processed'] = False
    if 'output_path' not in st.session_state:
        st.session_state['output_path'] = None
    if 'upload_path' not in st.session_state:
        st.session_state['upload_path'] = None
    if 'processing_msg' not in st.session_state:
        st.session_state['processing_msg'] = None

    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            tmp_file.write(video_file.read())
            st.session_state['upload_path'] = tmp_file.name
        st.success("Video uploaded successfully!")
        st.session_state['processed'] = False  # Reset processing state
        st.session_state['output_path'] = None  # Clear previous output path
        st.session_state['processing_msg'] = None  # Clear any previous processing message

    # Button to preview the uploaded video
    if st.session_state['upload_path'] and st.button("Preview Uploaded Video"):
        with col1:
            st.video(st.session_state['upload_path'])

    # Button to run the model
    if st.button("Run Model") and video_file is not None and not st.session_state['processed']:
        st.session_state['processing_msg'] = st.info("Processing video, please wait...")

        # Open the uploaded video
        video = cv2.VideoCapture(st.session_state['upload_path'])

        # Setup output file for saving processed video
        output_tempfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')  # Create temporary file
        st.session_state['output_path'] = output_tempfile.name  # Store the path of the temporary file
        output_tempfile.close()  # Close the file (still valid for writing)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = video.get(cv2.CAP_PROP_FPS)
        width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out = cv2.VideoWriter(st.session_state['output_path'], fourcc, fps, (width, height))

        # Process frames
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            results = model(frame)
            annotated_frame = np.squeeze(results.render())
            out.write(annotated_frame)

        video.release()
        out.release()

        # Mark processing complete and clear message
        st.session_state['processed'] = True
        st.session_state['processing_msg'].empty()  # Remove "Processing" message
        st.success("Processing complete!")

    # Button to preview processed video after running the model
    if st.session_state['output_path'] and st.session_state['processed'] :
        with col1:
            try:
                if Path(st.session_state['output_path']).exists():
                    st.video(st.session_state['output_path'])
                else:
                    st.error("Output video file not found.")
            except Exception as e:
                st.error(f"Error loading video: {e}")


    # Provide download button for processed video
    if st.session_state['output_path'] and st.session_state['processed'] :
        with open(st.session_state['output_path'], "rb") as f:
            st.download_button(
                label="Download Processed Video",
                data=f,
                file_name="processed_video.mp4",
                mime="video/mp4"
            )
