import cv2
import mediapipe as mp
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
from tqdm import tqdm


def detect_faces_video(video_path, output_path):

    # mp_pose = mp.solutions.pose
    # pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)
    # mp_drawing = mp.solutions.drawing_utils

    # Capturar o vídeo
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    # Obter informações do vídeo
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Definir o codec e criar o objeto VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # MEDIAPIPE POSE

        # frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # results = pose.process(frame_rgb)

        # if results.pose_landmarks:
        #     mp_drawing.draw_landmarks(
        #         frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        # )

        # RETINA FACE

        # Detectar rostos no frame
        faces = RetinaFace.detect_faces(frame)

        # Desenhar os retângulos nos rostos detectados
        for key in faces.keys():
            identity = faces[key]
            facial_area = identity[
                "facial_area"
            ]  # Coordenadas do rosto [x1, y1, x2, y2]
            x1, y1, x2, y2 = facial_area

            # Desenhar um retângulo ao redor do rosto
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Imprime o número de pessoas detectadas
        cv2.putText(
            frame,
            "Pessoas: " + str(len(faces)),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (10, 10, 250),
            2,
        )

        # DEEP FACE

        # Analisar o frame para detectar faces e emoções
        # result = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)

        # try:
        #     result = DeepFace.analyze(
        #         frame, actions=["emotion"], enforce_detection=True
        #     )
        # except:
        #     result = []

        # for face in result:
        #     x, y, w, h = (
        #         face["region"]["x"],
        #         face["region"]["y"],
        #         face["region"]["w"],
        #         face["region"]["h"],
        #     )
        #     emotion = face["dominant_emotion"]

        #     # Para não desenhar o retângulo na borda do frame
        #     # if (w != frame_height-1 and h != frame_height-1):
        #     #     cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 10, 255), 2)
        #     if emotion != "neutral":
        #         cv2.putText(
        #             frame,
        #             emotion,
        #             (x, y - 10),
        #             cv2.FONT_HERSHEY_SIMPLEX,
        #             0.9,
        #             (10, 10, 250),
        #             2,
        #         )

        # Escrever o frame processado no vídeo de saída
        out.write(frame)

    # Liberar recursos
    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Exemplo de uso
detect_faces_video(
    "tests/input_shorter.mp4", "resultado_retinaface_michael_shorter.mp4"
)
