from deepface import DeepFace
import os

def load_image_database(database_path):
    """Carrega todas as imagens da base de dados e associa ao nome da pessoa."""
    image_database = {}
    for file in os.listdir(database_path):
        if file.endswith(('.jpg', '.png', '.jpeg')):
            person_name = os.path.splitext(file)[0]  # Nome do arquivo sem extensão            
            image_database[person_name] = os.path.join(database_path, file)
    return image_database

def identify_person(image_path, image_database):
    """Compara a imagem de entrada com a base de dados para identificar a pessoa."""
    try:
        for name, db_image in image_database.items():
            print(f"Comparando {image_path} com {db_image}")
            result = DeepFace.verify(image_path, db_image)
            if result['verified']:
                return {'identified_person': name, 'distance': result['distance']}
        return {'identified_person': None, 'message': 'Nenhuma correspondência encontrada.'}
    except Exception as e:
        return {'error': str(e)}

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # image1 = os.path.join(script_dir, 'michael1.jpg')
    # image2 = os.path.join(script_dir, 'michael3.jpg')

    database_path = "/home/michael/code/fiap/fase4/tc/pessoas"
    image_to_identify = "/home/michael/code/fiap/fase4/tc/michael1.jpg"
    
    image_database = load_image_database(database_path)
    print(image_database)
    identification_result = identify_person(image_to_identify, image_database)
    
    if 'error' in identification_result:
        print("Erro ao identificar pessoa:", identification_result['error'])
    else:
        print("Pessoa identificada:", identification_result['identified_person'])
        print("Distância facial:", identification_result.get('distance', 'N/A'))
