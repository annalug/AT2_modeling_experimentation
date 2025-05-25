AT2 - Modelagem e Experimentação
================================

Este repositório contém um pipeline de experimentação para classificação de posturas humanas (sentado, em pé, caminhando). O projeto compara abordagens baseadas em YOLO e MediaPipe para detecção de poses e seleciona os melhores classificadores.

--------------------------------

ESTRUTURA DO PROJETO
--------------------
```
# ESTRUTURA DO PROJETO

AT2_modeling_experimentation/
├── data/
│   └── processed_mpii_poses/          # Imagens pré-processadas
│       ├── sitting/                   # Imagens de pessoas sentadas
│       ├── standing/                  # Imagens de pessoas em pé
│       └── walking/                   # Imagens de pessoas caminhando
│
├── experiments/
│   ├── model_1_yolov8_pose/           # Extração de features com YOLO
│   └── model_2_media_pipe/            # Pipeline do MediaPipe
│
├── models/
│   ├── models.py                      # Script de seleção de modelos
│   ├── pose_classifier_yolo_best.pkl  # Classificador treinado (YOLO)
│   └── pose_classifier_mediapipe_best.pkl  # Classificador treinado (MediaPipe)
│
├── notebooks/
│   └── models.ipynb                   # Análise interativa de métricas
│
└── results/
    ├── model_1_features/              # Features extraídas pelo YOLO (.npy)
    ├── model_2_mediapipe_features/    # Features extraídas pelo MediaPipe (.npy)
    └── graphs/                      # Métricas salvas em JPEG
        ├── metricas_por_classe.jpg
        ├── consistencia_metricas.jpg
        └── metricas_gerais.jpg
```
--------------------------------

CONFIGURAÇÃO DO AMBIENTE
------------------------

1. Clonar repositório:
   > git clone git@github.com:annalug/AT2_modeling_experimentation.git
   > cd AT2_modeling_experimentation

2. Criar ambiente virtual (Python 3.8+):
   > python -m venv pose_env
   > source pose_env/bin/activate  # Linux/Mac

3. Instalar dependências:
   > pip install -r requirements.txt

--------------------------------

EXECUÇÃO DO PIPELINE
--------------------

1. Extrair features:
   - YOLO:
     > python experiments/model_1_yolov8_pose/extract_yolo_features.py
   - MediaPipe:
     > python experiments/model_2_media_pipe/extract_mediapipe_features.py

2. Treinar e selecionar modelos:
   > python models/models.py  # Gera .pkl e gráficos

3. Visualizar resultados:
   - Gráficos: results/gráficos/
   - Relatório PDF: results/relatorio_comparativo.pdf

--------------------------------

