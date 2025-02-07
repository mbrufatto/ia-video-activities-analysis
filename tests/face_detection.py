#####################################################################
# Tenta achar as  pessoas da imagem. Resultado não é bom!
#####################################################################

import cv2
import face_recognition
import numpy as np

# Nome da imagem que está na mesma pasta
image_path = "multiple_people.jpg"  # Altere para o nome correto do arquivo

# Carregar a imagem
image = face_recognition.load_image_file(image_path)

# Encontrar as localizações dos rostos na imagem
face_locations = face_recognition.face_locations(image)

# Converter a imagem para o formato RGB para exibição com OpenCV
image_rgb = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# Desenhar um retângulo ao redor de cada rosto detectado
for top, right, bottom, left in face_locations:
    cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)

# Exibir a imagem com os rostos detectados
cv2.imshow("Face Detection", image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
