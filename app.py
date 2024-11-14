import streamlit as st
import torch
import cv2
from pathlib import Path
import tempfile
import numpy as np

# Import YOLOv5 model and utilities
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_boxes
from utils.torch_utils import select_device

# Load the custom YOLOv5 model
model_path = 'models/tplayer_best.pt'
device = select_device('')  # Use CUDA if available
model = DetectMultiBackend(model_path, device=device, dnn=False)
img_size = 640  # Set input size to 640x640 for model

# Custom CSS styling for wider layout and colorful theme
st.markdown("""
    <style>
        .main {
            max-width: 900px;
            padding: 20px;
            background-color: #ffffff;
            border-radius: 15px;
            box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1);
        }
        
        .css-18e3th9 {
            background: linear-gradient(135deg, #f6d365, #fda085);
        }
        
        h1 {
            color: #283593;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            font-weight: bold;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
        }
        
        .stButton>button {
            background-color: #0288d1;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            border: none;
            cursor: pointer;
            font-weight: bold;
            font-size: 16px;
        }
        .stButton>button:hover {
            background-color: #0277bd;
        }
        
        .stProgress .st-bs {
            background-color: #283593 !important;
        }
        
        .stDownloadButton > button {
            background-color: #388e3c;
            color: white;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: bold;
            font-size: 16px;
            border: none;
            cursor: pointer;
        }
        .stDownloadButton > button:hover {
            background-color: #2e7d32;
        }
    </style>
""", unsafe_allow_html=True)

st.title("🎾 Tennis Game Tracker")
st.write("Detect players and balls in a tennis match using YOLOv5. Upload a video, run detection, and download the processed result.")

# Layout divided into two columns
col1, col2 = st.columns([1, 1])

# File uploader and controls in the right column
with col2:
    st.write("### Upload Video for Detection")
    uploaded_video = st.file_uploader("Choose a video file", type=["mp4", "mov", "avi", "mkv"])

    if uploaded_video:
        # Temporary storage for uploaded video
        temp_video_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_video_path.write(uploaded_video.read())
        temp_video_path.close()

        # Process button
        if st.button("Run Detection"):
            # Open video, set parameters
            cap = cv2.VideoCapture(temp_video_path.name)
            output_path = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            out = cv2.VideoWriter(output_path, fourcc, fps, (img_size, img_size))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            st.write("Processing video...")
            progress_bar = st.progress(0)

            # Process each frame
            for i in range(total_frames):
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to 640x640 for YOLO input and output
                frame_resized = cv2.resize(frame, (img_size, img_size))

                # Prepare the frame for model input
                img = torch.from_numpy(frame_resized).to(device)
                img = img.permute(2, 0, 1).float() / 255.0  # Normalize and permute
                img = img.unsqueeze(0)  # Add batch dimension

                # Inference
                pred = model(img, augment=False, visualize=False)
                pred = non_max_suppression(pred, 0.25, 0.45, classes=None, agnostic=False)

                # Process detections
                for det in pred:
                    if len(det):
                        det[:, :4] = scale_boxes((img_size, img_size), det[:, :4], (img_size, img_size)).round()
                        for *xyxy, conf, cls in reversed(det):
                            x1, y1, x2, y2 = map(int, xyxy)
                            label = f'{model.names[int(cls)]} {conf:.2f}'
                            color = (0, 255, 0) if model.names[int(cls)] in ['player1', 'player2', 'person1', 'person2'] else (255, 0, 0)
                            cv2.rectangle(frame_resized, (x1, y1), (x2, y2), color, 2)
                            if label:
                                t_size = cv2.getTextSize(label, 0, fontScale=0.5, thickness=1)[0]
                                cv2.rectangle(frame_resized, (x1, y1 - t_size[1] - 4), (x1 + t_size[0], y1), color, -1)
                                cv2.putText(frame_resized, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness=1)

                # Write processed frame to output
                out.write(frame_resized)
                progress_bar.progress(int((i + 1) / total_frames * 100))

            # Release resources
            cap.release()
            out.release()

            st.success("Detection completed!")

            # Download processed video
            with open(output_path, "rb") as file:
                st.download_button(
                    label="Download Processed Video",
                    data=file,
                    file_name="processed_video.mp4",
                    mime="video/mp4"
                )

# Video preview in the left column
if uploaded_video:
    with col1:
        st.write("### Video Preview")
        st.video(temp_video_path.name)
