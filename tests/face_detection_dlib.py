import dlib
import cv2

# Carregar a imagem
image_path = "imagem.jpg"
image = cv2.imread(image_path)

# Criar o detector de rostos do dlib
detector = dlib.get_frontal_face_detector()

# Converter para escala de cinza (melhora a detecção)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detectar rostos
faces = detector(gray)

# Desenhar retângulos ao redor dos rostos
for face in faces:
    x, y, w, h = face.left(), face.top(), face.width(), face.height()
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

# Exibir a imagem
cv2.imshow("Detecção com dlib", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
