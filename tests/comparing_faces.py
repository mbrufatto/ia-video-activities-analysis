#################################################################################################
# Compara rostos e verifica se são da mesma pessoa
#################################################################################################

import os

from deepface import DeepFace


def compare_faces(image1, image2):
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))

        image1_path = os.path.join(script_dir, image1)
        image2_path = os.path.join(script_dir, image2)

        # Realiza a verificação das imagens
        result = DeepFace.verify(
            image1_path,
            image2_path,
            model_name="ArcFace",
            detector_backend="retinaface",
        )

        # Obtém se são a mesma pessoa e a distância de verificação
        is_same_person = result["verified"]
        distance = result["distance"]

        print("\n" + image1 + " => " + image2)
        print("São a mesma pessoa?", is_same_person)
        print("Distância facial:", distance)

    except Exception as e:
        print("Erro ao comparar imagens:", str(e))


if __name__ == "__main__":
    image1 = "compare_person/person_1.jpg"
    image2 = "compare_person/person_2.jpg"
    image3 = "compare_person/person_3.jpg"
    image4 = "compare_person/person_4.jpg"
    image5 = "compare_person/person_5.jpg"
    image6 = "compare_person/person_6.jpg"
    image7 = "compare_person/person_7.jpg"
    image8 = "compare_person/person_8.jpg"
    image9 = "compare_person/person_9.jpg"

    compare_faces(image1, image2)
    compare_faces(image1, image3)
    compare_faces(image1, image4)
    compare_faces(image1, image5)
    compare_faces(image1, image6)
    compare_faces(image1, image7)
    compare_faces(image1, image8)
    compare_faces(image1, image9)

    compare_faces(image2, image3)
    compare_faces(image2, image4)
    compare_faces(image2, image5)
    compare_faces(image2, image5)
    compare_faces(image2, image6)
    compare_faces(image2, image7)
    compare_faces(image2, image8)

    compare_faces(image3, image4)
    compare_faces(image3, image5)
    compare_faces(image3, image6)
    compare_faces(image3, image7)
    compare_faces(image3, image8)
    compare_faces(image3, image9)

    compare_faces(image4, image5)
    compare_faces(image4, image6)
    compare_faces(image4, image7)
    compare_faces(image4, image8)
    compare_faces(image4, image9)

    compare_faces(image5, image6)
    compare_faces(image5, image7)
    compare_faces(image5, image8)
    compare_faces(image5, image9)

    compare_faces(image6, image7)
    compare_faces(image6, image8)
    compare_faces(image6, image9)

    compare_faces(image7, image8)
    compare_faces(image7, image9)

    compare_faces(image8, image9)
