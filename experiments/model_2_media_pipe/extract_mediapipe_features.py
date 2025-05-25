import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
import joblib

# --- Configuração ---
AT2_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DATASET_BASE_PATH = AT2_ROOT_PATH / "data" / "processed_mpii_poses"
OUTPUT_FEATURES_PATH = AT2_ROOT_PATH / "results" / "model_2_mediapipe_features"
OUTPUT_FEATURES_PATH.mkdir(parents=True, exist_ok=True)

CLASSES = ['sitting', 'standing', 'walking']
SPLITS = ['train', 'val', 'test']
NUM_MEDIAPIPE_KEYPOINTS = 33
VISIBILITY_THRESHOLD = 0.5  # Novo: limiar de visibilidade
NORMALIZATION_MODE = 1

# Inicializar MediaPipe
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
    model_complexity=2  # Melhora precisão
)


def load_image_paths_and_labels(dataset_base_dir: Path, split_name: str, class_names: list):
    """Carrega caminhos e labels com verificação reforçada"""
    all_image_paths = []
    all_labels = []
    print(f"Loading image paths for {split_name} split...")

    for class_idx, class_name in enumerate(class_names):
        class_path = dataset_base_dir / class_name / split_name
        if not class_path.exists():
            print(f"Warning: {class_path} not found. Skipping.")
            continue

        image_files = sorted([
            p for p in class_path.glob("*.*")
            if p.suffix.lower() in ['.jpg', '.jpeg', '.png']
        ])

        for img_path in image_files:
            all_image_paths.append(str(img_path))
            all_labels.append(class_idx)

    return all_image_paths, np.array(all_labels)


def normalize_keypoints_mediapipe(keypoints, image_width, image_height, mode=1):
    """Normalização com tratamento de visibilidade"""
    if keypoints is None:
        return np.full((NUM_MEDIAPIPE_KEYPOINTS, 2), np.nan), False

    # Converter coordenadas com verificação de visibilidade
    kps_pixels = []
    for kp in keypoints.landmark:
        if kp.visibility < VISIBILITY_THRESHOLD:  # Novo filtro
            kps_pixels.append([np.nan, np.nan])
        else:
            kps_pixels.append([
                kp.x * image_width,
                kp.y * image_height
            ])

    kps_pixels = np.array(kps_pixels)

    if mode == 1:
        # Calcular bbox ignorando NaNs
        with np.errstate(invalid='ignore'):
            x_min, y_min = np.nanmin(kps_pixels, axis=0)
            x_max, y_max = np.nanmax(kps_pixels, axis=0)

        bbox_w = x_max - x_min
        bbox_h = y_max - y_min

        if bbox_w <= 0 or bbox_h <= 0 or np.isnan(bbox_w) or np.isnan(bbox_h):
            return np.full((NUM_MEDIAPIPE_KEYPOINTS, 2), np.nan), False

        # Normalização tolerante a NaNs
        normalized_kps = np.zeros_like(kps_pixels)
        normalized_kps[:, 0] = (kps_pixels[:, 0] - x_min) / bbox_w
        normalized_kps[:, 1] = (kps_pixels[:, 1] - y_min) / bbox_h

        return normalized_kps, True
    else:
        raise ValueError("Modo de normalização não suportado.")


def extract_features_for_split_mediapipe(image_paths, original_labels, split_name): # Renomeado labels para original_labels
    """Extração com coleta de todas as amostras (incluindo NaNs) e labels correspondentes"""
    all_features_list = []
    all_corresponding_labels_list = [] # Nova lista para manter a sincronia

    print(f"\nExtracting features for {split_name} split using MediaPipe...")
    for i, img_path_str in enumerate(tqdm(image_paths, desc=f"Processing {split_name}")):
        img_path = Path(img_path_str)
        current_label = original_labels[i] # Pega o label correspondente à imagem atual

        feature_vector = np.full(NUM_MEDIAPIPE_KEYPOINTS * 2, np.nan) # Padrão NaN
        # success_normalization = False # Não estritamente necessário aqui se sempre adicionamos

        if not img_path.exists():
            print(f"  Aviso: Imagem não encontrada {img_path_str}, pulando e adicionando NaNs.")
            all_features_list.append(feature_vector)
            all_corresponding_labels_list.append(current_label)
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Aviso: Não foi possível ler a imagem {img_path_str}, pulando e adicionando NaNs.")
            all_features_list.append(feature_vector)
            all_corresponding_labels_list.append(current_label)
            continue

        img_height, img_width, _ = img.shape
        try:
            results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"  Erro ao processar imagem {img_path_str} com MediaPipe: {e}")
            all_features_list.append(feature_vector)
            all_corresponding_labels_list.append(current_label)
            continue


        if results.pose_landmarks:
            normalized_kps, success_normalization = normalize_keypoints_mediapipe(
                results.pose_landmarks, img_width, img_height, NORMALIZATION_MODE
            )
            if success_normalization:
                feature_vector = normalized_kps.flatten()
            # Se success_normalization for False, feature_vector permanece como NaNs
        # Se não houver pose_landmarks, feature_vector permanece como NaNs

        all_features_list.append(feature_vector)
        all_corresponding_labels_list.append(current_label) # Adiciona o label correspondente

    # Garante que ambos os arrays retornados tenham o mesmo número de amostras
    final_features = np.array(all_features_list)
    final_labels = np.array(all_corresponding_labels_list)

    if final_features.shape[0] != final_labels.shape[0]:
        # Esta verificação é uma salvaguarda, não deveria acontecer com a lógica acima
        raise RuntimeError(f"Erro crítico de sincronização em extract_features_for_split_mediapipe para {split_name}: "
                           f"Features {final_features.shape[0]}, Labels {final_labels.shape[0]}")

    return final_features, final_labels


def train_and_apply_imputer(features_dict):
    """Treina imputer nos dados de treino e aplica a todos splits"""
    imputer = SimpleImputer(strategy='median')

    # Treinar apenas com dados de treino
    imputer.fit(features_dict['train'])

    # Aplicar a todos splits
    for split in SPLITS:
        features_dict[split] = imputer.transform(features_dict[split])

    # Salvar imputer para uso futuro
    joblib.dump(imputer, OUTPUT_FEATURES_PATH / "mediapipe_imputer.pkl")

    return features_dict


def main():  # em extract_mediapipe_features.py
    features_dict = {}
    labels_dict = {}  # Este dicionário agora armazenará os labels sincronizados

    for split in SPLITS:
        image_paths, original_labels_for_split = load_image_paths_and_labels(DATASET_BASE_PATH, split, CLASSES)
        if not image_paths:
            print(f"Skipping {split} (no images found)")
            features_dict[split] = np.array([])
            labels_dict[split] = np.array([])  # Salva array vazio para labels também
            continue

        # Passa original_labels_for_split para a função de extração
        features, synced_labels = extract_features_for_split_mediapipe(image_paths, original_labels_for_split, split)
        features_dict[split] = features
        labels_dict[split] = synced_labels  # Usa os labels sincronizados

    # Treinar e aplicar imputer
    # Certifique-se de que não há splits vazios antes de passar para o imputer
    train_features = features_dict['train']
    if train_features.ndim == 1 and train_features.size == 0:  # Verifica se é um array vazio
        print("Não há features de treino para treinar o imputer. Pulando a imputação e salvamento.")
    elif train_features.shape[0] == 0:  # Verifica se tem 0 amostras, mas múltiplas features
        print("Features de treino têm 0 amostras. Pulando a imputação e salvamento.")
    else:
        features_dict_imputed = train_and_apply_imputer(
            features_dict.copy())  # Usar .copy() se train_and_apply_imputer modifica in-place

        # Salvar features processadas
        for split in SPLITS:
            # Usa features_dict_imputed e labels_dict (que já está sincronizado)
            if features_dict_imputed[split].size == 0 or labels_dict[split].size == 0:
                print(f"Skipping saving for {split} due to empty features or labels after processing.")
                continue

            # Verificação final de consistência antes de salvar
            if features_dict_imputed[split].shape[0] != labels_dict[split].shape[0]:
                print(f"ALERTA: Inconsistência de tamanho antes de salvar {split}! "
                      f"Features: {features_dict_imputed[split].shape[0]}, Labels: {labels_dict[split].shape[0]}. "
                      "Isso NÃO deveria acontecer.")
                # Decida como lidar: pular o salvamento, lançar erro, etc.
                # Por agora, vamos pular para evitar corromper os dados de entrada do models.py
                continue

            np.save(
                OUTPUT_FEATURES_PATH / f"mediapipe_features_{split}_norm{NORMALIZATION_MODE}.npy",
                features_dict_imputed[split]
            )
            np.save(
                OUTPUT_FEATURES_PATH / f"mediapipe_labels_{split}_norm{NORMALIZATION_MODE}.npy",
                labels_dict[split]  # Salva os labels sincronizados
            )
            print(
                f"Saved {split} split with {features_dict_imputed[split].shape[0]} features and {labels_dict[split].shape[0]} labels")

if __name__ == "__main__":
    main()