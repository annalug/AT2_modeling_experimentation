# AT2_modeling_experimentation

├── AT2_modeling_experimentation/       # Entrega da Atividade 2 (Pasta Atual da Imagem)
│   ├── data/                           # Esta pasta pode ser um LINK SIMBÓLICO ou você pode copiar
│   │                                   # o 'processed_data_AT1' para cá.
│   │   └── processed_mpii_poses/       # Dataset organizado pela AT1
│   │       ├── sitting/
│   │       ├── standing/
│   │       └── walking/
│   ├── experiments/                    # Onde o código da AT2 (modelos) vai residir
│   │   ├── common/
│   │   │   └── utils.py
│   │   ├── model_1_yolov8_pose/
│   │   │   ├── train_yolov8_classifier.py
│   │   │   └── predict_yolov8.py
│   │   ├── model_2_mediapipe_cnn/
│   │   │   ├── train_cnn.py
│   │   │   └── predict_mediapipe_cnn.py
│   │   └── evaluate_models.py
│   ├── models/                         # Modelos treinados salvos
│   ├── notebooks/                      # Jupyter Notebooks para exploração (opcional)
│   ├── report/                         # Onde o relatório PDF da AT2 será salvo
│   │   └── AT2_Report.pdf
│   ├── results/                        # Resultados numéricos, métricas salvas, plots de avaliação
│   ├── .git/
│   ├── .gitignore
│   ├── README.md                       # README específico da AT2
│   └── requirements.txt                # Requisitos da AT2
    