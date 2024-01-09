import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Karar Ağacı
decision_tree_results = {
    'Model': 'Decision Tree',
    'Accuracy': 0.7275,
    'Precision': 0.7271945100253966,
    'Recall': 0.7275,
    'F1 Score': 0.7271648242414643
}

# Lojistik Regresyon
logistic_regression_results = {
    'Model': 'Logistic Regression',
    'Accuracy': 0.9406,
    'Precision': 0.9406624254140683,
    'Recall': 0.9406,
    'F1 Score': 0.9405668433828321
}

# K-En Yakın Komşu
knn_results = {
    'Model': 'K-Nearest Neighbors',
    'Accuracy': 0.9132,
    'Precision': 0.9144972741969015,
    'Recall': 0.9132,
    'F1 Score': 0.912876835790885
}

# Random Forest
random_forest_results = {
    'Model': 'Random Forest',
    'Accuracy': 0.8814,
    'Precision': 0.8809430752063004,
    'Recall': 0.8814,
    'F1 Score': 0.8806978895144381
}

# SVM
svm_results = {
    'Model': 'SVM',
    'Accuracy': 0.9504,
    'Precision': 0.9505161632226136,
    'Recall': 0.9504,
    'F1 Score': 0.9503625571163635
}


# Sonuçları bir liste olarak birleştirme
results = [
    decision_tree_results,
    logistic_regression_results,
    knn_results,
    svm_results,
    random_forest_results
]

# Sonuçları bir veri çerçevesine dönüştürme
results_df = pd.DataFrame(results)

# Grafik çizimini ayarlama
plt.figure(figsize=(12, 8))

# Accuracy
plt.subplot(3, 1, 1)
sns.barplot(x='Model', y='Accuracy', data=results_df, palette='viridis')

plt.ylim(0, 1)  # Accuracy değerleri 0 ile 1 arasında olduğu için y ekseni sınırlarını belirleme
for i, v in enumerate(results_df['Accuracy']):
    plt.text(i, v + 0.01, str(round(v, 4)), ha='center', va='bottom')

# Precision
plt.subplot(3, 1, 2)
sns.barplot(x='Model', y='Precision', data=results_df, palette='viridis')

plt.ylim(0, 1)  # Precision değerleri 0 ile 1 arasında olduğu için y ekseni sınırlarını belirleme
for i, v in enumerate(results_df['Precision']):
    plt.text(i, v + 0.01, str(round(v, 4)), ha='center', va='bottom')

# F1 Score
plt.subplot(3, 1, 3)
sns.barplot(x='Model', y='F1 Score', data=results_df, palette='viridis')

plt.ylim(0, 1)  # F1 Score değerleri 0 ile 1 arasında olduğu için y ekseni sınırlarını belirleme
for i, v in enumerate(results_df['F1 Score']):
    plt.text(i, v + 0.01, str(round(v, 4)), ha='center', va='bottom')

plt.suptitle('Performance Metrics', fontsize=16)  # Grafiklerin en üstüne başlık ekleme
plt.subplots_adjust(top=0.92)  # Başlıkları biraz daha yukarı çekme
plt.tight_layout()  # Alt başlıkların üst üste binmesini önler
plt.show()
