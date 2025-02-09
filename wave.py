import cv2
import mediapipe as mp
import numpy as np
import os
from collections import deque
from tqdm import tqdm
import time

# Inicializar MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class LeftHandTracker:
    """Classe para rastrear a mão esquerda e detectar movimentos de aceno"""
    def __init__(self):
        self.left_hand_up = False
        self.hand_movement_queue = deque(maxlen=10)  # Guardar as últimas 10 posições do pulso esquerdo
        self.last_waving_time = None  # Para registrar o momento do último aceno
        self.wave_detected = False  # Para garantir que o aceno não seja repetido desnecessariamente

    def update(self, left_wrist, left_shoulder, timestamp):
        """Atualiza a posição do pulso e verifica se houve mudança no estado da mão"""
        event = None
        left_up = left_wrist.y < left_shoulder.y  # A mão está levantada?

        if not self.left_hand_up and left_up:
            event = "Mão esquerda levantada!"
        elif self.left_hand_up and not left_up:
            event = "Mão esquerda abaixada!"

        self.left_hand_up = left_up  # Atualizar estado

        # Registrar a posição X do pulso esquerdo
        if left_up:
            self.hand_movement_queue.append((left_wrist.x, left_wrist.y, timestamp))

        # Verificar se está abanando
        if self.detect_waving(timestamp):
            event = "Acenando com a mão esquerda!"

        return event

    def detect_waving(self, timestamp):
        """Verifica se a mão esquerda está se movendo rapidamente para os lados (abano)"""
        if len(self.hand_movement_queue) < 5:
            return False  # Não há dados suficientes para análise

        # Obter as últimas posições (x, y, timestamp)
        positions = np.array(self.hand_movement_queue)
        x_positions = positions[:, 0]  # Posições X
        y_positions = positions[:, 1]  # Posições Y
        timestamps = positions[:, 2]  # Timestamps

        # Calcular a distância de movimento e o tempo
        movement_range = max(x_positions) - min(x_positions)
        time_range = timestamps[-1] - timestamps[0]

        # Verificar se o movimento é suficientemente rápido e largo
        if time_range > 0 and movement_range > 0.0 and time_range < 1.0:  # Ajuste a distância e tempo conforme necessário
            if not self.wave_detected:
                self.wave_detected = True
                self.last_waving_time = timestamp
                return True
        else:
            # Resetar se o aceno não for detectado dentro de um intervalo de tempo adequado
            if self.wave_detected and (timestamp - self.last_waving_time > 1):
                self.wave_detected = False

        return False


def detect_pose(video_path, output_path, scale=0.5, confidence_threshold=0.5):
    pose = mp_pose.Pose(min_detection_confidence=confidence_threshold, min_tracking_confidence=confidence_threshold)
    hand_tracker = LeftHandTracker()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Erro ao abrir o vídeo.")
        return

    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps, total_frames = int(cap.get(cv2.CAP_PROP_FPS)), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Ajustar o tamanho do frame
    new_width, new_height = int(width * scale), int(height * scale)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))

    start_time = time.time()  # Marcar o tempo inicial

    for _ in tqdm(range(total_frames), desc="Processando vídeo"):
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensionar frame
        frame = cv2.resize(frame, (new_width, new_height))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb_frame)

        timestamp = time.time() - start_time  # Calcular o timestamp de cada frame

        if results.pose_landmarks:
            # Verificar a confiança de cada landmark para reduzir falsos positivos
            for landmark in results.pose_landmarks.landmark:
                if landmark.visibility < confidence_threshold or landmark.presence < confidence_threshold:
                    continue  # Ignorar landmarks com baixa confiança

            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            landmarks = results.pose_landmarks.landmark
            left_wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]

            # Verificar se a mão esquerda foi levantada e se está abanando
            event = hand_tracker.update(left_wrist, left_shoulder, timestamp)
            if event:
                cv2.putText(frame, event, (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                print(event)

        out.write(frame)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()


# Caminhos dos vídeos
script_dir = os.path.dirname(os.path.abspath(__file__))
input_video_path = os.path.join(script_dir, 'Unlocking Facial Recognition_ Diverse Activities Analysis.mp4')
output_video_path = os.path.join(script_dir, 'left_hand_waving.mp4')

# Chamar função reduzindo o tamanho do frame para 50% e ajustando o limite de confiança
detect_pose(input_video_path, output_video_path, scale=0.5, confidence_threshold=0.6)
