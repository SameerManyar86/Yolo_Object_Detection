import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Set up the Streamlit page
st.title("YOLO Object Detection")
st.sidebar.title("Upload and Detect")
st.sidebar.info("Upload a video file and see YOLOv8 in action!")

# Upload video file
uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "avi", "mov"])

if uploaded_file:
    # Save uploaded video temporarily
    with tempfile.NamedTemporaryFile(delete=False) as temp_video:
        temp_video.write(uploaded_file.read())
        video_path = temp_video.name

    # Load YOLO model
    st.sidebar.text("Loading YOLO model...")
    model = YOLO("yolo-Weights/yolov8n.pt")

    # Display uploaded video
    st.video(video_path)

    # Create a placeholder for displaying results
    placeholder = st.empty()

    # Start processing
    st.sidebar.text("Processing video...")
    cap = cv2.VideoCapture(video_path)

    # Output video details
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define codec and create output video file
    output_path = "output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection
        results = model(frame)

        # Draw detections on the frame
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                cls = int(box.cls[0])
                class_name = model.names[cls]

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, f"{class_name} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)

        # Write the processed frame to the output file
        out.write(frame)

        # Show progress
        placeholder.image(frame, channels="BGR", use_column_width=True)

    # Release resources
    cap.release()
    out.release()

    st.sidebar.success("Processing complete!")
    st.video(output_path)

    # Provide download link for the processed video
    with open(output_path, "rb") as file:
        btn = st.download_button(label="Download Processed Video", data=file, file_name="output.mp4", mime="video/mp4")

else:
    st.sidebar.warning("Please upload a video file to start!")
