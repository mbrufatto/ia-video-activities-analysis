from deepface import DeepFace
import os

def compare_faces(image1_path, image2_path):
    try:
        # Realiza a verificação das imagens
        result = DeepFace.verify(image1_path, image2_path)
        
        # Obtém se são a mesma pessoa e a distância de verificação
        is_same_person = result['verified']
        distance = result['distance']
        
        return {
            'same_person': is_same_person,
            'distance': distance,
            'details': result
        }
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    image1 = os.path.join(script_dir, 'michael1.jpg')
    image2 = os.path.join(script_dir, 'michael3.jpg')

    
    comparison_result = compare_faces(image1, image2)
    
    if 'error' in comparison_result:
        print("Erro ao comparar imagens:", comparison_result['error'])
    else:
        print("São a mesma pessoa?", comparison_result['same_person'])
        print("Distância facial:", comparison_result['distance'])
