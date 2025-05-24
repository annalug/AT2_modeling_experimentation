import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lazypredict.Supervised import LazyClassifier
from sklearn.utils import all_estimators

# functions 

# Função para processar classification reports
def process_report(report, model_name, dataset_type):
    metrics = []
    # Processar métricas por classe
    for class_name in CLASSES:
        class_metrics = report[class_name]
        metrics.append({
            'Modelo': model_name,
            'Conjunto': dataset_type,
            'Classe': class_name,
            'Precisão': class_metrics['precision'],
            'Recall': class_metrics['recall'],
            'F1-Score': class_metrics['f1-score'],
            'Suporte': class_metrics['support']
        })
    
    # Processar métricas agregadas
    metrics.append({
        'Modelo': model_name,
        'Conjunto': dataset_type,
        'Classe': 'Macro Avg',
        'Precisão': report['macro avg']['precision'],
        'Recall': report['macro avg']['recall'],
        'F1-Score': report['macro avg']['f1-score'],
        'Suporte': report['macro avg']['support']
    })
    
    
    return metrics


# 1. Métricas por Classe (Atualizada)
def plot_metricas_por_classe(df):
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    for i, metric in enumerate(['Precisão', 'Recall', 'F1-Score']):
        ax = axes[i]
        sns.barplot(data=df[df['Classe'] != 'Macro Avg'],
                    x='Classe', y=metric, hue='Modelo',
                    ci=None, ax=ax)
        ax.set_title(f'{metric} por Classe e Modelo')
        ax.set_ylim(0, 1.1)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig('../results/metricas_por_classe.jpg', dpi=300, bbox_inches='tight')  # Nova linha
    plt.close()  # Fechar a figura após salvar

# 2. Consistência entre Validação e Teste (Atualizada)
def plot_consistencia(df):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, metric in enumerate(['Precisão', 'Recall', 'F1-Score']):
        ax = axes[i]
        sns.lineplot(data=df, x='Conjunto', y=metric, hue='Modelo',
                     style='Classe', markers=True, dashes=False, ax=ax)
        ax.set_title(f'Consistência de {metric} entre Conjuntos')
        ax.set_ylim(0, 1.1)

    plt.tight_layout()
    plt.savefig('../results/consistencia_metricas.jpg', dpi=300, bbox_inches='tight')  # Nova linha
    plt.close()


# 3. Métricas Gerais Agrupadas (Atualizada)
def plot_metricas_gerais(df):
    grouped = df.groupby(['Modelo', 'Conjunto']).agg({
        'Precisão': 'mean',
        'Recall': 'mean',
        'F1-Score': 'mean'
    }).reset_index()

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    for i, metric in enumerate(['Precisão', 'Recall', 'F1-Score']):
        ax = axes[i]
        sns.barplot(data=grouped, x='Modelo', y=metric, hue='Conjunto', ax=ax)
        ax.set_title(f'Média Geral de {metric}')
        ax.set_ylim(0, 1)
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')

    plt.tight_layout()
    plt.savefig('../results/metricas_gerais.jpg', dpi=300, bbox_inches='tight')  # Nova linha
    plt.close()


def get_best_model(X_train, X_val, y_train, y_val, target_names):
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.metrics import f1_score

    classifiers = {
        'RandomForest': RandomForestClassifier(),
        'LogisticRegression': LogisticRegression(max_iter=1000),
        'SVC': SVC(probability=True),
        'GradientBoosting': GradientBoostingClassifier()
    }

    best_f1 = -1
    best_model = None
    results = []

    for name, clf in classifiers.items():
        try:
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_val)
            f1 = f1_score(y_val, y_pred, average='macro')
            results.append({'Modelo': name, 'Macro Avg F1': f1})

            if f1 > best_f1:
                best_f1 = f1
                best_model = clf

        except Exception as e:
            print(f"Erro em {name}: {str(e)}")
            continue

    # Retreinar o melhor modelo com todos os dados
    if best_model is not None:
        best_model.fit(np.vstack([X_train, X_val]), np.concatenate([y_train, y_val]))

    return best_model, pd.DataFrame(results)

# Configurações
CLASSES = ['sitting', 'standing', 'walking']
random_state = 42


# Coletar dados para comparação
all_metrics = []

# =================================================================
# Processar resultados do YOLO
# =================================================================
## for YOLO model
# Carregar dados
X_train_yolo = np.load("../results/model_1_features/yolo_features_train_norm1.npy")
y_train_yolo = np.load("../results/model_1_features/yolo_labels_train_norm1.npy")
X_val_yolo = np.load("../results/model_1_features/yolo_features_val_norm1.npy")
y_val_yolo = np.load("../results/model_1_features/yolo_labels_val_norm1.npy")
X_test_yolo = np.load("../results/model_1_features/yolo_features_test_norm1.npy")
y_test_yolo = np.load("../results/model_1_features/yolo_labels_test_norm1.npy")

# Encontrar melhor classificador para YOLO
best_clf_yolo, yolo_metrics = get_best_model(X_train_yolo, X_val_yolo, y_train_yolo, y_val_yolo, CLASSES)
best_clf_yolo.fit(X_train_yolo, y_train_yolo)

# Avaliar
y_pred_yolo_val = best_clf_yolo.predict(X_val_yolo)
y_pred_yolo_test = best_clf_yolo.predict(X_test_yolo)

# Salvar melhor modelo YOLO
joblib.dump(best_clf_yolo, "../models/pose_classifier_yolo_best.pkl")

# Processar relatórios
yolo_val_report = classification_report(y_val_yolo, y_pred_yolo_val, target_names=CLASSES, output_dict=True)
yolo_test_report = classification_report(y_test_yolo, y_pred_yolo_test, target_names=CLASSES, output_dict=True)

# Processar relatórios e adicionar às métricas
all_metrics += process_report(yolo_val_report, 'YOLO', 'Validação')
all_metrics += process_report(yolo_test_report, 'YOLO', 'Teste')

# =================================================================
# Processar resultados do MediaPipe
# =================================================================


## for MediaPipe model
# Carregar dados
X_train_mp = np.load("../results/model_2_mediapipe_features/mediapipe_features_train_norm1.npy")
y_train_mp = np.load("../results/model_2_mediapipe_features/mediapipe_labels_train_norm1.npy")
X_val_mp = np.load("../results/model_2_mediapipe_features/mediapipe_features_val_norm1.npy")
y_val_mp = np.load("../results/model_2_mediapipe_features/mediapipe_labels_val_norm1.npy")
X_test_mp = np.load("../results/model_2_mediapipe_features/mediapipe_features_test_norm1.npy")
y_test_mp= np.load("../results/model_2_mediapipe_features/mediapipe_labels_test_norm1.npy")

# Encontrar melhor classificador para MediaPipe
best_clf_mp, mp_metrics = get_best_model(X_train_mp, X_val_mp, y_train_mp, y_val_mp, CLASSES)
best_clf_mp.fit(X_train_mp, y_train_mp)

# Avaliar
y_pred_mp_val = best_clf_mp.predict(X_val_mp)
y_pred_mp_test = best_clf_mp.predict(X_test_mp)

# Salvar melhor modelo MediaPipe
joblib.dump(best_clf_mp, "../models/pose_classifier_mediapipe_best.pkl")

# Processar relatórios
mediapipe_val_report = classification_report(y_val_mp, y_pred_mp_val, target_names=CLASSES, output_dict=True)
mediapipe_test_report = classification_report(y_test_mp, y_pred_mp_test, target_names=CLASSES, output_dict=True)

# Processar relatórios e adicionar às métricas
all_metrics += process_report(mediapipe_val_report, 'MediaPipe', 'Validação')
all_metrics += process_report(mediapipe_test_report, 'MediaPipe', 'Teste')
# Criar DataFrame
df_comparacao = pd.DataFrame(all_metrics)

# # Executar todos os gráficos
# print("\n1. Métricas por Classe")
# plot_metricas_por_classe(df_comparacao)

# print("\n2. Consistência entre Validação e Teste")
# plot_consistencia(df_comparacao)

# print("\n3. Métricas Gerais Agrupadas")
# plot_metricas_gerais(df_comparacao)

# Exibir métricas dos modelos testados
print("Top 5 Modelos YOLO:")
print(yolo_metrics.sort_values('Macro Avg F1', ascending=False).head(5))

print("\nTop 5 Modelos MediaPipe:")
print(mp_metrics.sort_values('Macro Avg F1', ascending=False).head(5))

plot_metricas_por_classe(df_comparacao)
plot_consistencia(df_comparacao)
plot_metricas_gerais(df_comparacao)