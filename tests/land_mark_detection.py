import cv2
import mediapipe as mp
import os

# Inicializa o módulo de pose do MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Carrega o vídeo
video_path = "input_full.mp4"  # Substitua pelo caminho correto do vídeo
cap = cv2.VideoCapture(video_path)

# Obtém informações do vídeo
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define o caminho de saída
output_path = os.path.splitext(video_path)[0] + "_pose.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Inicializa o detector de pose
with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Converte para RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Processa a imagem
        results = pose.process(image_rgb)
        
        # Desenha as landmarks se houver detecção
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing_styles.get_default_pose_landmarks_style()
            )
        
        # Escreve o frame no arquivo de saída
        out.write(frame)
        
        # Exibe o frame (opcional)
        cv2.imshow('Pose Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Libera os recursos
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Vídeo processado e salvo em: {output_path}")
