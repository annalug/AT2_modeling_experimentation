import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer
import joblib

# --- Configuração ---
AT2_ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DATASET_BASE_PATH = AT2_ROOT_PATH / "data" / "processed_mpii_poses"
OUTPUT_FEATURES_PATH = AT2_ROOT_PATH / "results" / "model_1_features"
OUTPUT_FEATURES_PATH.mkdir(parents=True, exist_ok=True)

YOLO_MODEL_NAME = 'yolov8s-pose.pt'
CLASSES = ['sitting', 'standing', 'walking']
SPLITS = ['train', 'val', 'test']
NORMALIZATION_MODE = 1
CONFIDENCE_THRESHOLD = 0.5  # Novo: limiar de confiança
NUM_YOLO_KEYPOINTS = 17


def load_image_paths_and_labels(dataset_base_dir: Path, split_name: str, class_names: list):
    """Carrega caminhos com verificação robusta de arquivos"""
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


def normalize_keypoints(keypoints, bbox=None, center_point=None, mode=0):
    """Normalização com tratamento de confiança e NaNs"""
    if keypoints is None or keypoints.shape[0] == 0:
        return np.full((NUM_YOLO_KEYPOINTS, 2), np.nan), False

    kps_xy = keypoints[:, :2].astype(float)
    confidences = keypoints[:, 2].astype(float)

    # Marcar keypoints com baixa confiança como NaN
    kps_xy[confidences < CONFIDENCE_THRESHOLD] = np.nan

    if mode == 1:
        if bbox is None:
            return np.full((NUM_YOLO_KEYPOINTS, 2), np.nan), False

        x1, y1, x2, y2 = bbox
        bbox_w = x2 - x1
        bbox_h = y2 - y1

        if bbox_w <= 0 or bbox_h <= 0 or np.isnan(bbox_w) or np.isnan(bbox_h):
            return np.full((NUM_YOLO_KEYPOINTS, 2), np.nan), False

        # Normalização tolerante a NaNs
        normalized_kps = np.zeros_like(kps_xy)
        with np.errstate(invalid='ignore'):
            normalized_kps[:, 0] = (kps_xy[:, 0] - x1) / bbox_w
            normalized_kps[:, 1] = (kps_xy[:, 1] - y1) / bbox_h

        return normalized_kps, True
    # ... (outros modos mantidos)


def extract_features_for_split(model, image_paths, labels, split_name, normalization_mode):
    """Extrai features para todos os casos (incluindo falhas como NaN)"""
    all_features = []
    all_labels = []

    print(f"\nExtracting features for {split_name} split...")
    for i, img_path_str in enumerate(tqdm(image_paths, desc=f"Processing {split_name}")):
        img_path = Path(img_path_str)

        # Processar e obter keypoints
        results = model(str(img_path), verbose=False)
        feature_vector = np.full(NUM_YOLO_KEYPOINTS * 2, np.nan)  # Padrão: NaN
        success = False

        if results and results[0].keypoints and results[0].keypoints.data.shape[0] > 0:
            person_keypoints_data = results[0].keypoints.data[0].cpu().numpy()
            person_bbox = results[0].boxes.xyxy[0].cpu().numpy() if results[0].boxes else None

            # Normalização
            normalized_kps, success = normalize_keypoints(
                person_keypoints_data,
                bbox=person_bbox,
                mode=normalization_mode
            )

            if success:
                feature_vector = normalized_kps.flatten()

        all_features.append(feature_vector)
        all_labels.append(labels[i])

    return np.array(all_features), np.array(all_labels)


def train_and_apply_imputer(features_dict):
    """Treina e aplica imputer nos dados"""
    imputer = SimpleImputer(strategy='median')
    imputer.fit(features_dict['train'])

    for split in SPLITS:
        if features_dict[split].size > 0:
            features_dict[split] = imputer.transform(features_dict[split])

    joblib.dump(imputer, OUTPUT_FEATURES_PATH / "yolo_imputer.pkl")
    return features_dict


def main():
    print(f"Initializing YOLOv8 model: {YOLO_MODEL_NAME}")
    yolo_model = YOLO(YOLO_MODEL_NAME)

    features_dict = {}
    labels_dict = {}

    for split in SPLITS:
        image_paths, labels = load_image_paths_and_labels(DATASET_BASE_PATH, split, CLASSES)
        if not image_paths:
            print(f"Skipping {split} (no images)")
            features_dict[split] = np.array([])
            labels_dict[split] = np.array([])
            continue

        features, split_labels = extract_features_for_split(
            yolo_model, image_paths, labels, split, NORMALIZATION_MODE
        )
        features_dict[split] = features
        labels_dict[split] = split_labels

    # Aplicar imputer
    features_dict = train_and_apply_imputer(features_dict)

    # Salvar
    for split in SPLITS:
        if features_dict[split].size == 0:
            continue

        np.save(
            OUTPUT_FEATURES_PATH / f"yolo_features_{split}_norm{NORMALIZATION_MODE}.npy",
            features_dict[split]
        )
        np.save(
            OUTPUT_FEATURES_PATH / f"yolo_labels_{split}_norm{NORMALIZATION_MODE}.npy",
            labels_dict[split]
        )
        print(f"Saved {split} split with {features_dict[split].shape[0]} samples")


if __name__ == "__main__":
    main()