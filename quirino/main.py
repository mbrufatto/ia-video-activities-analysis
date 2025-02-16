import cv2
import face_recognition
import numpy as np
from tqdm import tqdm
import mediapipe as mp
from deepface import DeepFace

# Mediapipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

#######################################
# PARÂMETROS E ESTRUTURAS DE DADOS
#######################################
N_FRAMES_HISTORY = 10  # Quantos frames de histórico para suavizar a detecção de atividade
ACTIVITY_THRESHOLD = 25.0  # Exemplo de limiar de movimento (dependerá de escala da imagem)
###############################################################
# PARÂMETROS DE DETECÇÃO DE MUDANÇA BRUSCA DE EMOÇÃO
###############################################################
MIN_STABLE_FRAMES = 25

# Estrutura para cada indivíduo identificado
# {
#   "id": int,
#   "face_encoding": np.ndarray,
#   "activity_history": deque(maxlen=N_FRAMES_HISTORY),
#   ...
# }

#######################################
# FUNÇÕES AUXILIARES
#######################################

def load_video(input_video_path):
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Não foi possível abrir o vídeo: {input_video_path}")
        return None
    return cap

def create_video_writer(cap, output_video_path):
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    return out

def get_or_create_individual_record(face_encoding, known_individuals, tolerance=0.8):
    if not known_individuals:
        return create_new_individual_record(face_encoding, known_individuals)
    
    face_encodings = [ind["face_encoding"] for ind in known_individuals]
    matches = face_recognition.compare_faces(face_encodings, face_encoding, tolerance=tolerance)
    distances = face_recognition.face_distance(face_encodings, face_encoding)
    best_match_idx = np.argmin(distances)

    if matches[best_match_idx]:
        return known_individuals[best_match_idx]
    else:
        return create_new_individual_record(face_encoding, known_individuals)

def create_new_individual_record(face_encoding, known_individuals):
    new_id = 1 if not known_individuals else max(ind["id"] for ind in known_individuals) + 1
    from collections import deque
    record = {
        "id": new_id,
        "face_encoding": face_encoding,
        "activity_history": deque(maxlen=N_FRAMES_HISTORY),
        "last_activity": "Desconhecida",
        "last_emotion": "desconhecido",
        "emotion_streak": 0,
        "abrupt_changes": 0        
    }
    known_individuals.append(record)
    return record

###############################################################
# FUNÇÕES PARA ANÁLISE DE EMOÇÃO E DETECÇÃO DE MUDANÇA BRUSCA
###############################################################

def analyze_emotion(face_roi):
    """
    Usa DeepFace para analisar a emoção no recorte (face_roi).
    Retorna o nome da emoção dominante como string.
    """
    analysis = DeepFace.analyze(
        img_path=face_roi,
        actions=['emotion'],
        enforce_detection=False
    )
    # Se o DeepFace retornar vários resultados (lista),
    # pegue o primeiro (caso de múltiplos rostos no ROI).
    if isinstance(analysis, list):
        analysis = analysis[0]
    
    return str(analysis.get('dominant_emotion', 'desconhecido'))

def detect_abrupt_emotion_change(individual_record, current_emotion):
    """
    Verifica se houve mudança brusca de emoção para um determinado indivíduo.
    Critério:
      - last_emotion: emoção anterior
      - emotion_streak: quantos frames consecutivos estava na mesma emoção
      - MIN_STABLE_FRAMES: limiar para dizer que a emoção estava "estável"

    Se a emoção anterior estava estável (emotion_streak >= MIN_STABLE_FRAMES)
    e a emoção atual for diferente, incrementamos abrupt_changes.
    """
    last_emotion = individual_record["last_emotion"]
    streak = individual_record["emotion_streak"]

    if current_emotion == last_emotion:
        # Continua na mesma emoção => incrementa streak
        individual_record["emotion_streak"] += 1
        return False
    else:
        # Houve troca de emoção
        was_abrupt = False
        # Se a anterior estava estável (streak >= MIN_STABLE_FRAMES), conta como mudança brusca
        if streak >= MIN_STABLE_FRAMES:
            individual_record["abrupt_changes"] += 1
            was_abrupt = True
        
        # Reinicia contagem para a nova emoção
        individual_record["last_emotion"] = current_emotion
        individual_record["emotion_streak"] = 1
        
        return was_abrupt

##############################
# Função de Classificação de Atividade
##############################
def classify_activity(landmarks):
    """
    Dada a lista de landmarks (33 pontos do corpo),
    retorna uma string representando a atividade estimada:
      - "Sentado"
      - "Gesticulando"
      - "Movimentando" (exemplo)
      - "Parado"
      - "Desconhecida"

    Esta função verifica a visibilidade dos punhos antes
    de considerar a pessoa como 'gesticulando'.
    """

    if not landmarks:
        return "Desconhecida"

    # Para facilitar a leitura do código:
    left_hip     = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee    = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_wrist   = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist  = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_shoulder= landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]

    # 1) Verifica se a pessoa está "sentada": (exemplo)
    #    Checa a diferença entre joelho e quadril
    #    Se joelho estiver muito próximo do quadril em Y, assume que está sentado
    if (left_knee.y - left_hip.y) < 0.05:
        return "Sentado"

    # 2) Detectar "gesticulando"
    #    - Precisamos que pelo menos um punho esteja visível
    #    - Se a posição vertical do punho difere muito do ombro, ou
    #      se há alguma heurística que indique grande movimentação dos braços
    #    - Exemplo simples: se a diferença de Y em relação ao ombro for > 0.2
    #      e a visibilidade do punho for > 0.5, assume "Gesticulando"

    # Visibilidade dos punhos
    left_wrist_visible  = (left_wrist.visibility  > 0.5)
    right_wrist_visible = (right_wrist.visibility > 0.5)

    if left_wrist_visible:
        hand_diff_left = abs(left_wrist.y - left_shoulder.y)
    else:
        hand_diff_left = 0.0

    if right_wrist_visible:
        hand_diff_right = abs(right_wrist.y - left_shoulder.y)
    else:
        hand_diff_right = 0.0

    # Se algum punho estiver visível e o desvio for grande, consideramos "gesticulando"
    if (left_wrist_visible or right_wrist_visible):
        if (hand_diff_left > 0.2) or (hand_diff_right > 0.2):
            return "Gesticulando"

    # 3) Caso queira detectar "movimentando" (exemplo):
    #    Poderíamos comparar a variação do centro de gravidade do corpo
    #    em frames consecutivos. Aqui, simplificado:
    return "Parado"

def find_closest_pose_to_face(faces_locations, poses_landmarks):
    """
    Exemplo de emparelhamento. Recebe:
      - faces_locations: lista de (top, right, bottom, left) para cada rosto
      - poses_landmarks: lista de listas de landmarks do mediapipe (um por pessoa detectada)
    
    Retorna lista de mesmo tamanho de faces_locations, onde cada item é
    (face_idx, pose_idx) => qual pose está mais próxima de qual rosto.
    
    Estratégia simples: distância entre o centro do bounding box do rosto e
    o "centro de gravidade" dos landmarks do pose.
    
    Essa abordagem é simplória, mas serve como exemplo.
    """
    if not faces_locations or not poses_landmarks:
        return []

    centers_faces = []
    for (top, right, bottom, left) in faces_locations:
        cx = (left + right) / 2
        cy = (top + bottom) / 2
        centers_faces.append((cx, cy))
    
    centers_poses = []
    for landmark_list in poses_landmarks:
        xs = [lm.x for lm in landmark_list]
        ys = [lm.y for lm in landmark_list]
        # Como mediapipe retorna valores normalizados (0 a 1),
        # assumimos que o frame tem dimensões 1x1. Precisaríamos
        # escalar para pixel se quisermos maior precisão.
        # Para comparar distâncias, usaremos as coordenadas normalizadas.
        cx = np.mean(xs)
        cy = np.mean(ys)
        centers_poses.append((cx, cy))
    
    # Agora faz a correspondência
    # Neste exemplo, vamos para cada face, achar a pose mais próxima
    # e supor que é aquela.
    pairs = []
    used_pose_idxs = set()
    for face_idx, face_center in enumerate(centers_faces):
        fx, fy = face_center
        min_dist = float('inf')
        best_pose_idx = None
        for pose_idx, pose_center in enumerate(centers_poses):
            if pose_idx in used_pose_idxs:
                continue  # cada pose só pode ser usada uma vez
            px, py = pose_center
            dist = np.sqrt((fx - px*1000)**2 + (fy - py*1000)**2)
            # Multipliquei pose_center por 1000 pra ter uma escala minimamente comparável.
            if dist < min_dist:
                min_dist = dist
                best_pose_idx = pose_idx
        if best_pose_idx is not None:
            pairs.append((face_idx, best_pose_idx))
            used_pose_idxs.add(best_pose_idx)
    
    return pairs

#######################################
# PROCESSAMENTO DE FRAME
#######################################
def process_frame(frame, known_individuals, pose_instance):
    """
    1. Localiza rostos no frame (face_recognition).
    2. Identifica o ID único de cada rosto.
    3. Usa DeepFace para detectar emoção dominante.
    4. Verifica se houve mudança brusca de emoção.
    5. Desenha as anotações no frame (bounding box e labels).
    """ 
    # 1) FACE RECOGNITION
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detecta rostos
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    # 2) MEDIAPIPE POSE
    # Convertemos frame para RGB novamente (mediapipe também usa RGB)
    # (Já temos rgb_frame, podemos usar o mesmo)
    results = pose_instance.process(rgb_frame)
    
    # Coletar "pose_landmarks" de todas as pessoas detectadas
    # O mediapipe.pose padrão detecta só 1 pessoa, a menos que esteja usando 
    # soluções avançadas (p. ex. "BlazePose GHUM full body" com multi-person). 
    # Então, por simplicidade, assumimos 1 pessoa.
    # Se fosse multi-person, results.pose_world_landmarks seria uma lista.
    
    # Exemplo: se pose_instance não suportar multi-person nativamente, 
    # o results.pose_landmarks é apenas 1 "landmark list".
    poses_landmarks = []
    if results.pose_landmarks:
        # Converte para lista
        landmarks = results.pose_landmarks.landmark
        poses_landmarks.append(landmarks)

    # Emparelhar face ↔ pose
    # (Aqui assumindo 1 a 1. Se houver multi-person de fato, esse método 
    #  poderia emparelhar várias. Exemplo didático.)
    pairs = find_closest_pose_to_face(face_locations, poses_landmarks)

    # 3) PARA CADA FACE DETECTADA, IDENTIFICAR INDIVÍDUO E ATIVIDADE
    for i, (face_loc, face_enc) in enumerate(zip(face_locations, face_encodings)):
        individual_record = get_or_create_individual_record(face_enc, known_individuals)

        # 4) ANÁLISE DE EMOÇÃO: LAST
        # Recorte do rosto para análise de emoção
        face_roi = rgb_frame[face_loc[0]:face_loc[2], face_loc[3]:face_loc[1]]  # (top, bottom, left, right)
        current_emotion = analyze_emotion(face_roi)
        
        # Verificar se houve mudança brusca
        was_abrupt = detect_abrupt_emotion_change(individual_record, current_emotion)
        
        # Se houver pose emparelhada
        pose_idx = None
        for face_idx, p_idx in pairs:
            if face_idx == i:
                pose_idx = p_idx
                break
        
        # Se encontramos a pose correspondente
        if pose_idx is not None and pose_idx < len(poses_landmarks):
            # Classificar atividade
            activity = classify_activity(poses_landmarks[pose_idx])
            # Armazenar histórico
            individual_record["activity_history"].append(activity)
            
            # Define a atividade "corrente" como a mais frequente no histórico
            from collections import Counter
            c = Counter(individual_record["activity_history"])
            current_activity = c.most_common(1)[0][0]
        else:
            current_activity = "Desconhecida"
        
        individual_record["last_activity"] = current_activity
        
        # Desenhar bounding box e texto
        (top, right, bottom, left) = face_loc
        cv2.rectangle(frame, (left, top), (right, bottom), (0,255,0), 2)
        label_text = f"Pessoa {individual_record['id']} | {current_activity} | {current_emotion}"
        cv2.putText(
            frame,
            label_text,
            (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0,255,0),
            2
        )
        
        # Se houve mudança brusca, exibimos alerta
        if was_abrupt:
            cv2.putText(
                frame, "Mudança Brusca!", 
                (left, bottom + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                (0, 0, 255), 2
            )
    
    # Opcional: desenhar a pose no frame (se tiver 1 pessoa)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    
    return frame

#######################################
# SUMÁRIO FINAL
#######################################
def show_summary(total_frames, known_individuals):
    print("\n===== RESUMO =====")
    print(f"Total de frames analisados: {total_frames}")
    print(f"Quantidade de indivíduos únicos detectados: {len(known_individuals)}\n")
    for ind in known_individuals:
        print(f" - Pessoa {ind['id']}")
        print(f"   Última atividade: {ind['last_activity']}")
        print(f"   Histórico de atividade: {list(ind['activity_history'])}")
        print(f"   Última emoção detectada: {ind['last_emotion']}")
        print(f"   Mudanças bruscas de emoção: {ind['abrupt_changes']}\n")        
        print("")

#######################################
# FUNÇÃO PRINCIPAL
#######################################
def identify_people_in_video(input_video_path, output_video_path):
    cap = load_video(input_video_path)
    if cap is None:
        return
    out = create_video_writer(cap, output_video_path)
    
    known_individuals = []
    total_frames_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    processed_frames = 0

    # Iniciar MediaPipe Pose (detecção de uma pessoa).
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    ) as pose_instance:

        for _ in tqdm(range(total_frames_count), desc="Processando frames"):
            ret, frame = cap.read()
            if not ret:
                break
            processed_frames += 1

            # Processa frame (identificação + pose + atividade)
            frame = process_frame(frame, known_individuals, pose_instance)
            
            out.write(frame)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    show_summary(processed_frames, known_individuals)

##########################
# EXECUÇÃO DE EXEMPLO
##########################
if __name__ == "__main__":
    # Exemplo de uso - altere os caminhos conforme necessário
    # Altere os caminhos conforme necessário
    input_video = "data/sample_individuals.mp4"
    output_video = "data/output/outputv2.mp4"
    
    identify_people_in_video(input_video, output_video)
