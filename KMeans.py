import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    accuracy_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

# ===============================================================
# Metodo del gomito + silhouette per determinare k ottimale
# ===============================================================
def elbow_method(X):
    inertia = []
    silhouette_scores = []
    k_range = range(2, 11)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', max_iter=5, n_init=10, random_state=0)
        kmeans.fit(X)
        inertia.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))

    # Metodo del gomito
    ax[0].plot(k_range, inertia, marker='o', color='b', label='Inertia')
    ax[0].set_title('Metodo del Gomito')
    ax[0].set_xlabel('Numero di Cluster (k)')
    ax[0].set_ylabel('Inertia')
    ax[0].grid(True)
    optimal_k_inertia = np.argmin(inertia) + 2
    ax[0].plot(optimal_k_inertia, inertia[optimal_k_inertia - 2], 'ro', label=f'k={optimal_k_inertia}')
    ax[0].legend()

    # Silhouette Score
    ax[1].plot(k_range, silhouette_scores, marker='o', color='g', label='Silhouette Score')
    ax[1].set_title('Silhouette Score')
    ax[1].set_xlabel('Numero di Cluster (k)')
    ax[1].set_ylabel('Silhouette')
    ax[1].grid(True)
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    ax[1].plot(optimal_k_silhouette, max(silhouette_scores), 'ro', label=f'k={optimal_k_silhouette}')
    ax[1].legend()

    plt.show()
    plt.close()

    print(f"\nNumero ottimale di cluster (k): {optimal_k_silhouette}")
    return optimal_k_silhouette

# ===============================================================
# Funzione principale KMeans con i grafici + PCA 2D Visualization
# ===============================================================
def KMEANS(X, y):
    # 1️⃣ Distribuzione etichette reali -> BARPLOT ORIZZONTALE
    class_counts = np.bincount(y)
    plt.figure(figsize=(8, 5))
    plt.barh([f'Classe {i}' for i in range(len(class_counts))], class_counts, color='skyblue')
    plt.title('Distribuzione delle Etichette Reali')
    plt.xlabel('Numero di Campioni')
    plt.ylabel('Classe')
    for i, v in enumerate(class_counts):
        plt.text(v + 1, i, str(v), va='center')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 2️⃣ Metodo del gomito
    optimal_k = elbow_method(X)

    # 3️⃣ Clustering K-Means
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=0)
    y_kmeans = kmeans.fit_predict(X)

    # 4️⃣ Distribuzione dei cluster -> BARPLOT VERTICALE
    cluster_counts = np.bincount(y_kmeans)
    plt.figure(figsize=(8, 5))
    plt.bar([f'Cluster {i}' for i in range(optimal_k)], cluster_counts, color='orange')
    plt.title(f'Distribuzione dei Cluster (k={optimal_k})')
    plt.xlabel('Cluster')
    plt.ylabel('Numero di Campioni')
    for i, v in enumerate(cluster_counts):
        plt.text(i, v + 2, str(v), ha='center')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 5️⃣ Visualizzazione 2D con PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_kmeans, cmap='tab10', s=50, alpha=0.7, edgecolors='k')
    plt.title(f'Visualizzazione 2D dei Cluster (PCA) - k={optimal_k}')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend(*scatter.legend_elements(), title='Cluster')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 6️⃣ Precision-Recall Scatter Plot
    average_precision = average_precision_score(y, y_kmeans)
    precision, recall, _ = precision_recall_curve(y, y_kmeans)

    plt.figure(figsize=(8, 6))
    plt.scatter(recall, precision, c=precision, cmap='viridis', s=60, edgecolors='k')
    plt.colorbar(label='Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Scatter Plot (AP={average_precision:.2f})')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.close()

    # 7️⃣ Metriche di valutazione
    accuracy = accuracy_score(y, y_kmeans)
    precision_val = precision_score(y, y_kmeans, average='weighted')
    recall_val = recall_score(y, y_kmeans, average='weighted')
    f1 = f1_score(y, y_kmeans, average='weighted')

    print("\n--- Performance Summary ---")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Precision: {precision_val:.3f}")
    print(f"Recall: {recall_val:.3f}")
    print(f"F1-Score: {f1:.3f}")
    print(f"Average Precision: {average_precision:.3f}")

    print("\n--- Classification Report ---")
    print(classification_report(y, y_kmeans))

    # 8️⃣ Confusion Matrix con heatmap migliorata
    confusion_Matrix = confusion_matrix(y, y_kmeans)
    df_cm = pd.DataFrame(confusion_Matrix, index=['Classe 0', 'Classe 1'], columns=['Pred 0', 'Pred 1'])

    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm, annot=True, fmt='d', cmap='coolwarm', cbar=True, annot_kws={'size': 14})
    plt.title('Matrice di Confusione (Heatmap)')
    plt.tight_layout()
    plt.show()
    plt.close()

    # 9️⃣ Radar Chart per metriche
    metrics = {
        'Accuracy': accuracy,
        'Precision': precision_val,
        'Recall': recall_val,
        'F1-Score': f1,
        'Avg Precision': average_precision
    }

    labels = list(metrics.keys())
    values = list(metrics.values())
    values += values[:1] 
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color='tab:blue', linewidth=2)
    ax.fill(angles, values, color='skyblue', alpha=0.4)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{x:.1f}" for x in np.linspace(0, 1, 6)])
    ax.set_title('Confronto delle Metriche (Radar Chart)')
    plt.tight_layout()
    plt.show()
    plt.close()
