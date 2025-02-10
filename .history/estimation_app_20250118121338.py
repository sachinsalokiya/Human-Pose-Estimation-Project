import streamlit as st
from PIL import Image
import numpy as np
import cv2
import altair as alt  # Correct import statement

DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

def process_and_annotate(image):
    """Process the image and annotate it with pose landmarks."""
    frameWidth = image.shape[1]
    frameHeight = image.shape[0]
    net.setInput(cv2.dnn.blobFromImage(image, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]
    assert(len(BODY_PARTS) == out.shape[1])
    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > 0.2 else None)
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(image, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(image, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(image, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
    return image

def main():
    st.set_page_config(page_title="Pose Estimation Hub", page_icon="🤸", layout="wide")

    # Load model at app startup
    try:
        model_path = "graph_opt.pb"
        net = cv2.dnn.readNetFromTensorflow(model_path)
        st.session_state['model'] = net
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return

    st.title("Pose Estimation Hub")
    st.write("Analyze human poses with advanced AI-based techniques")

    st.sidebar.header("Upload Your Image")
    uploaded_file = st.sidebar.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = np.array(Image.open(uploaded_file))
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        annotated_image = process_and_annotate(image_bgr)
        annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

        st.write("### Results")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Original Image", use_column_width=True)
        with col2:
            st.image(annotated_image_rgb, caption="Pose Annotated Image", use_column_width=True)

        st.success("Pose analysis completed! View the annotated image on the right.")
    else:
        st.info("Upload an image from the sidebar to begin pose estimation.")

    st.markdown("""
    ---
    ### About Pose Estimation
    Pose estimation is a technique to detect human body keypoints, like shoulders, elbows, and knees, in images or videos. 
    It has applications in:
    - Fitness Tracking
    - Healthcare & Rehabilitation
    - Sports Analytics
    - Augmented Reality

    ---
    ### How It Works
    1. Upload an image in the sidebar.
    2. Our model processes it using [OpenPose](https://github.com/CMU-Perceptual-Computing-Lab/openpose).
    3. The annotated image is displayed alongside the original.
    
    Developed with ❤️ by Shivam.
    """)

if __name__ == "__main__":
    main()

