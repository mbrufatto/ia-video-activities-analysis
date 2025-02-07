################################################################################################
# This code is used to detect multiple poses in a video file
# The code uses YOLOv5 to detect person in the frame and then uses MediaPipe to detect the pose
# The code then saves the video file with the poses detected
################################################################################################

import cv2
import mediapipe as mp

# PyTorch Hub
import torch
from mediapipe.python.solutions import pose as mp_pose
from tqdm import tqdm

# Model
yolo_model = torch.hub.load("ultralytics/yolov5", "yolov5s")

# since we are only intrested in detecting person
yolo_model.classes = [0]

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

video_path = "input_full.mp4"

cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

out = cv2.VideoWriter(
    "output_multiple_poses.mp4",
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height),
)

for _ in tqdm(range(total_frames), desc="Processando vídeo"):
    ret, frame = cap.read()
    if not ret:
        break

    # Recolor Feed from RGB to BGR
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # making image writeable to false improves prediction
    image.flags.writeable = False

    result = yolo_model(image)

    # Recolor image back to BGR for rendering
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # This array will contain crops of images incase we need it
    img_list = []

    # we need some extra margin bounding box for human crops to be properly detected
    MARGIN = 10

    for xmin, ymin, xmax, ymax, confidence, clas in result.xyxy[0].tolist():
        with mp_pose.Pose(
            min_detection_confidence=0.3, min_tracking_confidence=0.3
        ) as pose:
            # Media pose prediction ,we are
            results = pose.process(
                image[
                    int(ymin) + MARGIN : int(ymax) + MARGIN,
                    int(xmin) + MARGIN : int(xmax) + MARGIN :,
                ]
            )

            # Draw landmarks on image, if this thing is confusing please consider going through numpy array slicing
            mp_drawing.draw_landmarks(
                image[
                    int(ymin) + MARGIN : int(ymax) + MARGIN,
                    int(xmin) + MARGIN : int(xmax) + MARGIN :,
                ],
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(
                    color=(245, 117, 66), thickness=2, circle_radius=2
                ),
                mp_drawing.DrawingSpec(
                    color=(245, 66, 230), thickness=2, circle_radius=2
                ),
            )
            img_list.append(image[int(ymin) : int(ymax), int(xmin) : int(xmax) :])

    out.write(image)

cap.release()
out.release()
cv2.destroyAllWindows()
