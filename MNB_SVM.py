import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    precision_recall_curve,
    roc_curve,
    roc_auc_score,
    average_precision_score,
)

# ===============================================================
# Naive Bayes con visual rinnovate
# ===============================================================

def NaiveBayes(X1, y1):
    # Split
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.75, random_state=13, stratify=y1)

    # GridSearch per alpha
    param_grid = {'alpha': [0.01, 0.1, 0.5, 1, 2, 5, 10]}
    grid_search = GridSearchCV(MultinomialNB(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X1_train, y1_train)
    best_alpha = grid_search.best_params_['alpha']
    print(f"\nMiglior alpha per Naive Bayes: {best_alpha}")

    # Fit best model
    clf_nb = MultinomialNB(alpha=best_alpha)
    clf_nb.fit(X1_train, y1_train)

    # Predizioni
    prediction_nb = clf_nb.predict(X1_test)
    probs_nb = clf_nb.predict_proba(X1_test)[:, 1]  # score per curva PR/ROC

    accuracy_nb = accuracy_score(y1_test, prediction_nb)
    print("\n--- Performance Summary Naive Bayes ---")
    print(f"Accuracy: {accuracy_nb:.3f}")
    print("\n--- Classification Report ---")
    print(classification_report(y1_test, prediction_nb))

    # === ROC: area chart (riempita) ===
    fpr_nb, tpr_nb, _ = roc_curve(y1_test, probs_nb)
    auc_nb = roc_auc_score(y1_test, probs_nb)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.fill_between(fpr_nb, tpr_nb, step='pre', alpha=0.35, label=f'NB ROC AUC = {auc_nb:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Naive Bayes ROC (Filled Area)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(); plt.close()

    # === Confusion Matrix: normalizzata (percentuali) ===
    cm_nb = confusion_matrix(y1_test, prediction_nb, normalize='true')
    df_cm_nb = pd.DataFrame(cm_nb, index=['Classe 0', 'Classe 1'], columns=['Pred 0', 'Pred 1'])
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm_nb, annot=True, fmt='.2f', cmap='magma', cbar=True, annot_kws={'size': 14})
    plt.title('Naive Bayes Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.show(); plt.close()

    # === Precision-Recall: scatter con colormap ===
    precision_nb, recall_nb, _ = precision_recall_curve(y1_test, probs_nb)
    ap_nb = average_precision_score(y1_test, probs_nb)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(recall_nb, precision_nb, c=precision_nb, cmap='plasma', s=55, edgecolors='k')
    plt.colorbar(sc, label='Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Naive Bayes Precision-Recall (Scatter) | AP={ap_nb:.2f}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(); plt.close()


# ===============================================================
# SVM con visual rinnovate
# ===============================================================

def SVM(X1, y1):
    # Split
    X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, train_size=0.75, random_state=13, stratify=y1)

    # Scaling
    scaler = StandardScaler()
    X1_train_scaled = scaler.fit_transform(X1_train)
    X1_test_scaled = scaler.transform(X1_test)

    # GridSearch su C, gamma, kernel
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 'auto', 0.1, 1],
        'kernel': ['linear', 'rbf', 'poly']
    }
    grid_search = GridSearchCV(svm.SVC(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X1_train_scaled, y1_train)
    best_params = grid_search.best_params_
    print(f"\nMigliori parametri trovati per SVM: {best_params}")

    # Fit best model
    clf_svm = svm.SVC(C=best_params['C'], gamma=best_params['gamma'], kernel=best_params['kernel'])
    clf_svm.fit(X1_train_scaled, y1_train)

    # Predizioni e score
    prediction_svm = clf_svm.predict(X1_test_scaled)
    scores_svm = clf_svm.decision_function(X1_test_scaled)  # score per PR/ROC

    print("\n--- Report di Classificazione per SVM ---")
    print(classification_report(y1_test, prediction_svm))

    # === ROC: area chart (riempita) ===
    fpr_svm, tpr_svm, _ = roc_curve(y1_test, scores_svm)
    auc_svm = roc_auc_score(y1_test, scores_svm)
    plt.figure(figsize=(8, 6))
    plt.plot([0, 1], [0, 1], linestyle='--', label='Random Classifier')
    plt.fill_between(fpr_svm, tpr_svm, step='pre', alpha=0.35, label=f'SVM ROC AUC = {auc_svm:.2f}')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('SVM ROC (Filled Area)')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(); plt.close()

    # === Confusion Matrix: normalizzata (percentuali) ===
    cm_svm = confusion_matrix(y1_test, prediction_svm, normalize='true')
    df_cm_svm = pd.DataFrame(cm_svm, index=['Classe 0', 'Classe 1'], columns=['Pred 0', 'Pred 1'])
    plt.figure(figsize=(8, 6))
    sn.heatmap(df_cm_svm, annot=True, fmt='.2f', cmap='magma', cbar=True, annot_kws={'size': 14})
    plt.title('SVM Confusion Matrix (Normalized)')
    plt.tight_layout()
    plt.show(); plt.close()

    # === Precision-Recall: scatter con colormap ===
    precision_svm, recall_svm, _ = precision_recall_curve(y1_test, scores_svm)
    ap_svm = average_precision_score(y1_test, scores_svm)
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(recall_svm, precision_svm, c=precision_svm, cmap='viridis', s=55, edgecolors='k')
    plt.colorbar(sc, label='Precision')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'SVM Precision-Recall (Scatter) | AP={ap_svm:.2f}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(); plt.close()


# ===============================================================
# Curve di apprendimento: trasformate in area charts con bande
# ===============================================================

def plot_comparison_learning_curves(X, y):
    """
    Confronto curve di apprendimento NB vs SVM in forma di area chart (riempite)
    con bande di confidenza (std).
    """
    clf_nb = MultinomialNB()
    clf_svm = svm.SVC(kernel='rbf', C=1, gamma='scale')

    train_sizes = np.linspace(0.1, 1.0, 10)
    ts_nb, tr_nb, te_nb = learning_curve(clf_nb, X, y, cv=5, n_jobs=-1, train_sizes=train_sizes)
    ts_svm, tr_svm, te_svm = learning_curve(clf_svm, X, y, cv=5, n_jobs=-1, train_sizes=train_sizes)

    tr_mean_nb, tr_std_nb = tr_nb.mean(axis=1), tr_nb.std(axis=1)
    te_mean_nb, te_std_nb = te_nb.mean(axis=1), te_nb.std(axis=1)
    tr_mean_svm, tr_std_svm = tr_svm.mean(axis=1), tr_svm.std(axis=1)
    te_mean_svm, te_std_svm = te_svm.mean(axis=1), te_svm.std(axis=1)

    plt.figure(figsize=(10, 6))

    # NB area
    plt.fill_between(ts_nb, tr_mean_nb - tr_std_nb, tr_mean_nb + tr_std_nb, alpha=0.25, label='NB Training (±1σ)')
    plt.fill_between(ts_nb, te_mean_nb - te_std_nb, te_mean_nb + te_std_nb, alpha=0.25, label='NB CV (±1σ)')

    # SVM area
    plt.fill_between(ts_svm, tr_mean_svm - tr_std_svm, tr_mean_svm + tr_std_svm, alpha=0.25, label='SVM Training (±1σ)')
    plt.fill_between(ts_svm, te_mean_svm - te_std_svm, te_mean_svm + te_std_svm, alpha=0.25, label='SVM CV (±1σ)')

    plt.title('Learning Curves (Area Charts) — NB vs SVM')
    plt.xlabel('Numero di Campioni nel Training Set')
    plt.ylabel('Score')
    plt.legend(loc='best')
    plt.ylim(0, 1.05)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show(); plt.close()


# ===============================================================
# Confronto metriche: Radar chart (Precision, Recall, AP, Accuracy)
# ===============================================================

def plot_comparison_metrics(X, y):
    """
    Confronto delle metriche (Precision, Recall, Average Precision, Accuracy)
    tra Naive Bayes e SVM con un radar chart, usando il train size massimo.
    """
    # Train/test sul dataset completo per una fotografia finale
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=13, stratify=y)

    # NB
    nb = MultinomialNB().fit(X_train, y_train)
    y_pred_nb = nb.predict(X_test)
    scores_nb = nb.predict_proba(X_test)[:, 1]

    precision_nb = precision_score(y_test, y_pred_nb)
    recall_nb = recall_score(y_test, y_pred_nb)
    ap_nb = average_precision_score(y_test, scores_nb)
    acc_nb = accuracy_score(y_test, y_pred_nb)

    # SVM
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    svc = svm.SVC(kernel='rbf', C=1, gamma='scale').fit(X_train_s, y_train)
    y_pred_svm = svc.predict(X_test_s)
    scores_svm = svc.decision_function(X_test_s)

    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    ap_svm = average_precision_score(y_test, scores_svm)
    acc_svm = accuracy_score(y_test, y_pred_svm)

    labels = ['Precision', 'Recall', 'Avg Precision', 'Accuracy']
    nb_vals = [precision_nb, recall_nb, ap_nb, acc_nb]
    svm_vals = [precision_svm, recall_svm, ap_svm, acc_svm]



    nb_vals += nb_vals[:1]
    svm_vals += svm_vals[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7.5, 7.5), subplot_kw=dict(polar=True))
    ax.plot(angles, nb_vals, linewidth=2)
    ax.fill(angles, nb_vals, alpha=0.25, label='Naive Bayes')

    ax.plot(angles, svm_vals, linewidth=2)
    ax.fill(angles, svm_vals, alpha=0.25, label='SVM')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.1f}" for v in np.linspace(0, 1, 6)])
    ax.set_title('Confronto Metriche — Radar Chart')
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.05))
    plt.tight_layout()
    plt.show(); plt.close()


# ===============================================================
# Esempio d'uso su Breast Cancer dataset
# ===============================================================
if __name__ == '__main__':
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()

    # NB: MultinomialNB richiede feature non negative (tipico per bag-of-words). Se X contiene valori <=0,
    X = data.data
    y = data.target

    NaiveBayes(X, y)
    SVM(X, y)
    plot_comparison_learning_curves(X, y)
    plot_comparison_metrics(X, y)
