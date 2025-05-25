import numpy as np
from sklearn.metrics import classification_report, accuracy_score  # Adicionado accuracy_score
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import f1_score


# functions

# Função para processar classification reports
def process_report(report, model_name, dataset_type, y_true_for_accuracy,
                   y_pred_for_accuracy):  # Adicionados args para acurácia
    metrics = []
    # Processar métricas por classe
    for class_name in CLASSES:
        if class_name in report:  # Adicionar verificação se a classe existe no report
            class_metrics = report[class_name]
            metrics.append({
                'Modelo': model_name,
                'Conjunto': dataset_type,
                'Classe': class_name,
                'Acurácia': np.nan,  # Acurácia não é por classe, mas para manter a estrutura
                'Precisão': class_metrics['precision'],
                'Recall': class_metrics['recall'],
                'F1-Score': class_metrics['f1-score'],
                'Suporte': class_metrics['support']
            })
        else:
            print(f"Aviso: Classe '{class_name}' não encontrada no report para {model_name} ({dataset_type}).")

    # Processar métricas agregadas e acurácia geral
    if 'macro avg' in report:
        accuracy_geral = accuracy_score(y_true_for_accuracy, y_pred_for_accuracy)  # Calcula acurácia geral
        metrics.append({
            'Modelo': model_name,
            'Conjunto': dataset_type,
            'Classe': 'Macro Avg',  # Usamos 'Macro Avg' para agrupar as métricas gerais
            'Acurácia': accuracy_geral,  # Adiciona acurácia geral aqui
            'Precisão': report['macro avg']['precision'],
            'Recall': report['macro avg']['recall'],
            'F1-Score': report['macro avg']['f1-score'],
            'Suporte': report['macro avg']['support']  # Suporte para macro avg é o total de amostras
        })
    else:
        print(f"Aviso: 'macro avg' não encontrada no report para {model_name} ({dataset_type}).")

    return metrics


# 1. Métricas por Classe (Atualizada) - Não precisa de alteração para acurácia, pois é por classe
def plot_metricas_por_classe(df):
    if df.empty or df[df['Classe'] != 'Macro Avg'].empty:
        print("DataFrame vazio ou sem dados de classe para plotar métricas por classe.")
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))  # Mantém 3 subplots para Precisão, Recall, F1
    fig.suptitle('Desempenho por Classe e Modelo (Precisão, Recall, F1-Score)', fontsize=16, y=1.02)

    metrics_to_plot = ['Precisão', 'Recall', 'F1-Score']
    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        plot_df = df[(df['Classe'] != 'Macro Avg') & (df[metric].notna())]
        if plot_df.empty:
            ax.set_title(f'{metric} por Classe e Modelo (Sem dados)')
            continue

        sns.barplot(data=plot_df,
                    x='Classe', y=metric, hue='Modelo', ax=ax)
        ax.set_title(f'{metric} por Classe e Modelo')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Modelo')
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('../results/graphs/metricas_por_classe.jpg', dpi=300, bbox_inches='tight')
    plt.close()


# 2. Consistência entre Validação e Teste (Atualizada) - Também focado em Precisão, Recall, F1
def plot_consistencia(df):
    if df.empty:
        print("DataFrame vazio para plotar consistência.")
        return
    # Filtrar para 'Macro Avg' para Acurácia, e todas as classes para as outras
    df_accuracy = df[(df['Classe'] == 'Macro Avg') & df['Acurácia'].notna()]
    df_others = df[df['Classe'] != 'Macro Avg']  # Para Precisão, Recall, F1 por classe

    # Plot para Acurácia (geral)
    if not df_accuracy.empty:
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=df_accuracy, x='Conjunto', y='Acurácia', hue='Modelo',
                     markers=True, dashes=False)
        plt.title('Consistência da Acurácia Geral entre Conjuntos')
        plt.ylim(0, 1.1)
        plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('../results/graphs/consistencia_acuracia.jpg', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Sem dados de Acurácia para plotar consistência.")

    # Plots para Precisão, Recall, F1-Score (por classe)
    if not df_others.empty:
        fig_others, axes_others = plt.subplots(1, 3, figsize=(22, 6))  # Aumentado para acomodar legendas
        fig_others.suptitle('Consistência das Métricas (Precisão, Recall, F1-Score) por Classe', fontsize=16, y=1.03)
        metrics_to_plot = ['Precisão', 'Recall', 'F1-Score']
        for i, metric in enumerate(metrics_to_plot):
            ax = axes_others[i]
            plot_df = df_others[df_others[metric].notna()]
            if plot_df.empty:
                ax.set_title(f'Consistência de {metric} (Sem dados)')
                continue
            sns.lineplot(data=plot_df, x='Conjunto', y=metric, hue='Modelo',
                         style='Classe', markers=True, dashes=False, ax=ax)
            ax.set_title(f'Consistência de {metric} entre Conjuntos')
            ax.set_ylim(0, 1.1)
            ax.legend(title='Modelo / Classe', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        plt.tight_layout(rect=[0, 0, 0.9, 0.97])  # Ajustar right para caber legenda
        plt.savefig('../results/graphs/consistencia_metricas_por_classe.jpg', dpi=300, bbox_inches='tight')
        plt.close()
    else:
        print("Sem dados de Precisão, Recall, F1 por classe para plotar consistência.")


# 3. Métricas Gerais Agrupadas (Atualizada)
def plot_metricas_gerais(df):
    if df.empty:
        print("DataFrame vazio para plotar métricas gerais.")
        return

    # Pegar apenas a linha 'Macro Avg' que agora contém a acurácia geral
    # e as médias macro para Precisão, Recall, F1
    geral_metrics_df = df[df['Classe'] == 'Macro Avg'].copy()
    if geral_metrics_df.empty:
        print("Sem dados 'Macro Avg' para plotar métricas gerais.")
        return

    # Métricas a serem plotadas
    metrics_to_plot = ['Acurácia', 'Precisão', 'Recall', 'F1-Score']
    num_metrics = len(metrics_to_plot)

    fig, axes = plt.subplots(1, num_metrics, figsize=(5 * num_metrics, 6))  # Ajustar figsize
    if num_metrics == 1:  # Caso só tenha uma métrica (improvável aqui)
        axes = [axes]
    fig.suptitle('Comparativo Geral das Métricas (Macro Avg / Acurácia Geral)', fontsize=16, y=1.02)

    for i, metric in enumerate(metrics_to_plot):
        ax = axes[i]
        # Verificar se a coluna da métrica existe e tem dados não-nulos
        if metric not in geral_metrics_df.columns or geral_metrics_df[geral_metrics_df[metric].notna()].empty:
            ax.set_title(f'{metric} (Sem dados)')
            continue
        sns.barplot(data=geral_metrics_df[geral_metrics_df[metric].notna()], x='Modelo', y=metric, hue='Conjunto',
                    ax=ax)
        ax.set_title(f'{metric}')
        ax.set_ylim(0, 1.1)
        ax.legend(title='Conjunto')
        for p in ax.patches:
            ax.annotate(f"{p.get_height():.2f}",
                        (p.get_x() + p.get_width() / 2., p.get_height()),
                        ha='center', va='center',
                        xytext=(0, 10),
                        textcoords='offset points')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig('../results/graphs/metricas_gerais_com_acuracia.jpg', dpi=300, bbox_inches='tight')
    plt.close()


# ... (get_best_model permanece o mesmo) ...
def get_best_model(X_train, X_val, y_train, y_val, target_names):
    # Verificar consistência nos dados de entrada
    if X_train.shape[0] != y_train.shape[0] or X_val.shape[0] != y_val.shape[0]:
        raise ValueError("Inconsistent number of samples between X and y!")

    classifiers = {
        'RandomForest': RandomForestClassifier(random_state=random_state),  # Adicionado random_state
        'LogisticRegression': LogisticRegression(max_iter=1000, random_state=random_state),  # Adicionado random_state
        'SVC': SVC(probability=True, random_state=random_state),  # Adicionado random_state
        'GradientBoosting': GradientBoostingClassifier(random_state=random_state)  # Adicionado random_state
    }

    best_f1 = -1
    best_model_instance = None  # Renomeado
    best_model_name = ""  # Para rastrear o nome do melhor modelo
    results = []

    print(f"\n--- Buscando melhor modelo ---")
    for name, clf in classifiers.items():
        try:
            print(f"Treinando {name}...")
            clf.fit(X_train, y_train)
            y_pred_val = clf.predict(X_val)  # Renomeado para y_pred_val
            f1 = f1_score(y_val, y_pred_val, average='macro')
            print(f"  {name} - Macro F1 (Val): {f1:.4f}")
            results.append({'Modelo': name, 'Macro Avg F1': f1})

            if f1 > best_f1:
                best_f1 = f1
                best_model_instance = clf
                best_model_name = name


        except Exception as e:
            print(f"Erro em {name}: {str(e)}")
            continue

    if best_model_instance is None:
        print("Nenhum modelo pôde ser treinado. Verifique os dados ou os classificadores!")
        return None, pd.DataFrame(results)  # Retorna None e os resultados obtidos até então

    print(f"Melhor modelo (Validação): {best_model_name} com Macro F1: {best_f1:.4f}")

    # Opcional: Retreinar o melhor modelo com todos os dados de treino + validação
    # X_combined = np.vstack([X_train, X_val])
    # y_combined = np.concatenate([y_train, y_val])
    # print(f"Retreinando {best_model_name} com Treino+Validação...")
    # best_model_instance.fit(X_combined, y_combined)

    return best_model_instance, pd.DataFrame(results)


# Configurações
CLASSES = ['sitting', 'standing', 'walking']
random_state = 42  # Definido para reprodutibilidade nos classificadores

# Coletar dados para comparação
all_metrics = []

# =================================================================
# Processar resultados do YOLO
# =================================================================
print("\n" + "=" * 50)
print("Processando resultados do YOLO")
print("=" * 50)
try:
    X_train_yolo = np.load("../results/model_1_features/yolo_features_train_norm1.npy")
    y_train_yolo = np.load("../results/model_1_features/yolo_labels_train_norm1.npy")
    X_val_yolo = np.load("../results/model_1_features/yolo_features_val_norm1.npy")
    y_val_yolo = np.load("../results/model_1_features/yolo_labels_val_norm1.npy")
    X_test_yolo = np.load("../results/model_1_features/yolo_features_test_norm1.npy")
    y_test_yolo = np.load("../results/model_1_features/yolo_labels_test_norm1.npy")

    print(f"YOLO Shapes - Treino X: {X_train_yolo.shape}, y: {y_train_yolo.shape}")
    print(f"YOLO Shapes - Val X: {X_val_yolo.shape}, y: {y_val_yolo.shape}")
    print(f"YOLO Shapes - Teste X: {X_test_yolo.shape}, y: {y_test_yolo.shape}")

    best_clf_yolo, yolo_model_metrics_df = get_best_model(X_train_yolo, X_val_yolo, y_train_yolo, y_val_yolo, CLASSES)

    if best_clf_yolo is not None:
        # O modelo retornado por get_best_model já foi treinado com X_train_yolo
        # Se o retreinamento com X_combined estivesse ativo lá, ele estaria treinado com X_train+X_val.
        # A linha best_clf_yolo.fit(X_train_yolo, y_train_yolo) aqui é para garantir que
        # o modelo usado para predições de val/test seja o treinado apenas no split de treino original,
        # caso o retreinamento em get_best_model tenha sido comentado ou não seja o desejado para esta etapa.
        print("Refazendo o fit do melhor modelo YOLO apenas com dados de treino para avaliação padronizada...")
        best_clf_yolo.fit(X_train_yolo, y_train_yolo)

        y_pred_yolo_val = best_clf_yolo.predict(X_val_yolo)
        y_pred_yolo_test = best_clf_yolo.predict(X_test_yolo)

        joblib.dump(best_clf_yolo, "../models/pose_classifier_yolo_best.pkl")
        print("Modelo YOLO salvo em ../models/pose_classifier_yolo_best.pkl")

        yolo_val_report_dict = classification_report(y_val_yolo, y_pred_yolo_val, target_names=CLASSES,
                                                     output_dict=True, zero_division=0)
        yolo_test_report_dict = classification_report(y_test_yolo, y_pred_yolo_test, target_names=CLASSES,
                                                      output_dict=True, zero_division=0)

        all_metrics += process_report(yolo_val_report_dict, 'YOLO', 'Validação', y_val_yolo,
                                      y_pred_yolo_val)  # Passa y_true, y_pred
        all_metrics += process_report(yolo_test_report_dict, 'YOLO', 'Teste', y_test_yolo,
                                      y_pred_yolo_test)  # Passa y_true, y_pred
    else:
        print("Não foi possível treinar ou selecionar um modelo para YOLO.")
        yolo_model_metrics_df = pd.DataFrame()
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos de dados para YOLO: {e}. Pulando processamento YOLO.")
    yolo_model_metrics_df = pd.DataFrame()
except Exception as e:
    print(f"Erro inesperado no processamento YOLO: {e}")
    yolo_model_metrics_df = pd.DataFrame()

# =================================================================
# Processar resultados do MediaPipe
# =================================================================
print("\n" + "=" * 50)
print("Processando resultados do MediaPipe")
print("=" * 50)
try:
    X_train_mp = np.load("../results/model_2_mediapipe_features/mediapipe_features_train_norm1.npy")
    y_train_mp = np.load("../results/model_2_mediapipe_features/mediapipe_labels_train_norm1.npy")
    X_val_mp = np.load("../results/model_2_mediapipe_features/mediapipe_features_val_norm1.npy")
    y_val_mp = np.load("../results/model_2_mediapipe_features/mediapipe_labels_val_norm1.npy")
    X_test_mp = np.load("../results/model_2_mediapipe_features/mediapipe_features_test_norm1.npy")
    y_test_mp = np.load("../results/model_2_mediapipe_features/mediapipe_labels_test_norm1.npy")

    print(f"MediaPipe Shapes - Treino X: {X_train_mp.shape}, y: {y_train_mp.shape}")
    print(f"MediaPipe Shapes - Val X: {X_val_mp.shape}, y: {y_val_mp.shape}")
    print(f"MediaPipe Shapes - Teste X: {X_test_mp.shape}, y: {y_test_mp.shape}")

    best_clf_mp, mp_model_metrics_df = get_best_model(X_train_mp, X_val_mp, y_train_mp, y_val_mp, CLASSES)

    if best_clf_mp is not None:
        print("Refazendo o fit do melhor modelo MediaPipe apenas com dados de treino para avaliação padronizada...")
        best_clf_mp.fit(X_train_mp, y_train_mp)

        y_pred_mp_val = best_clf_mp.predict(X_val_mp)
        y_pred_mp_test = best_clf_mp.predict(X_test_mp)

        joblib.dump(best_clf_mp, "../models/pose_classifier_mediapipe_best.pkl")
        print("Modelo MediaPipe salvo em ../models/pose_classifier_mediapipe_best.pkl")

        mediapipe_val_report_dict = classification_report(y_val_mp, y_pred_mp_val, target_names=CLASSES,
                                                          output_dict=True, zero_division=0)
        mediapipe_test_report_dict = classification_report(y_test_mp, y_pred_mp_test, target_names=CLASSES,
                                                           output_dict=True, zero_division=0)

        all_metrics += process_report(mediapipe_val_report_dict, 'MediaPipe', 'Validação', y_val_mp,
                                      y_pred_mp_val)  # Passa y_true, y_pred
        all_metrics += process_report(mediapipe_test_report_dict, 'MediaPipe', 'Teste', y_test_mp,
                                      y_pred_mp_test)  # Passa y_true, y_pred
    else:
        print("Não foi possível treinar ou selecionar um modelo para MediaPipe.")
        mp_model_metrics_df = pd.DataFrame()
except FileNotFoundError as e:
    print(f"Erro ao carregar arquivos de dados para MediaPipe: {e}. Pulando processamento MediaPipe.")
    mp_model_metrics_df = pd.DataFrame()
except Exception as e:
    print(f"Erro inesperado no processamento MediaPipe: {e}")
    mp_model_metrics_df = pd.DataFrame()

# Criar DataFrame de comparação final
if all_metrics:
    df_comparacao = pd.DataFrame(all_metrics)
    print("\n--- Dataset comparativo ---")
    print(df_comparacao.to_string())  # Usar to_string() para imprimir o DataFrame inteiro

    # Executar todos os gráficos
    print("\nGerando Gráficos...")
    plot_metricas_por_classe(df_comparacao)
    print("Gráfico de métricas por classe salvo.")
    plot_consistencia(df_comparacao)
    print("Gráfico de consistência salvo.")
    plot_metricas_gerais(df_comparacao)
    print("Gráfico de métricas gerais com acurácia salvo.")
    print("Plots gerados com sucesso.")
else:
    print("\nNenhuma métrica foi coletada. Os gráficos não serão gerados.")
    df_comparacao = pd.DataFrame()

# Exibir métricas dos modelos testados
print("\n--- Métricas de Seleção de Modelos (Validação) ---")
if not yolo_model_metrics_df.empty:
    print("\nModelos YOLO (Validação) - Macro Avg F1:")
    print(yolo_model_metrics_df.sort_values('Macro Avg F1', ascending=False))
else:
    print("\nNenhuma métrica de modelo YOLO (validação) para exibir.")

if not mp_model_metrics_df.empty:
    print("\nModelos MediaPipe (Validação) - Macro Avg F1:")
    print(mp_model_metrics_df.sort_values('Macro Avg F1', ascending=False))
else:
    print("\nNenhuma métrica de modelo MediaPipe (validação) para exibir.")

print("\nScript concluído.")