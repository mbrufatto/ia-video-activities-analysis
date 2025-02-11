import cv2
from deepface import DeepFace
import os
import numpy as np
from tqdm import tqdm
from collections import defaultdict

def detect_emotions(video_path, output_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Dicionário para contar as emoções
    emotion_count = defaultdict(int)

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        try:
            # Usando um detector mais robusto
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=True, detector_backend='retinaface')

            for face in result:
                x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
                dominant_emotion = face['dominant_emotion']

                # Contar a emoção detectada
                emotion_count[dominant_emotion] += 1

                # Desenhar retângulo e emoção detectada
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)
        
        except Exception as e:
            print(f"Erro ao processar frame: {e}")

        out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    # Exibir contagem de emoções detectadas
    print("\nContagem de emoções detectadas no vídeo:")
    for emotion, count in sorted(emotion_count.items(), key=lambda x: x[1], reverse=True):
        print(f"{emotion}: {count}")

# Caminho para o arquivo de vídeo
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')  
output_video_path = os.path.join(script_dir, 'output_video_v2.mp4')

detect_emotions(input_video_path, output_video_path)
