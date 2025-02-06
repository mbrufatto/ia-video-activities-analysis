import cv2
import mediapipe as mp
import os

os.environ["OMP_NUM_THREADS"] = "1"

# Inicializar o detector do MediaPipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
detector = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

# Carregar a imagem
image_path = "imagem.jpg"
image = cv2.imread(image_path)

# Converter para RGB (requerido pelo MediaPipe)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Processar a imagem para detectar rostos
results = detector.process(image_rgb)

# Se encontrar rostos, desenhar os retângulos
if results.detections:
    for detection in results.detections:
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir a imagem
cv2.imshow("Detecção com MediaPipe", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
