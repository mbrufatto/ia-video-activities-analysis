import cv2
import numpy as np
from retinaface import RetinaFace

def detect_faces(image_path, output_path):
    # Carregar a imagem
    image = cv2.imread(image_path)
    if image is None:
        print("Erro ao carregar a imagem.")
        return
    
    # Detectar rostos
    faces = RetinaFace.detect_faces(image_path)
    
    # Desenhar os retângulos nos rostos detectados
    for key in faces.keys():
        identity = faces[key]
        facial_area = identity["facial_area"]  # Coordenadas do rosto [x1, y1, x2, y2]
        x1, y1, x2, y2 = facial_area
        
        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    
    # Salvar e exibir a imagem resultante
    cv2.imwrite(output_path, image)
    cv2.imshow("Faces Detectadas", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Exemplo de uso
detect_faces("imagem.jpg", "resultado.jpg")

