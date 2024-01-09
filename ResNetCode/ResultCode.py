import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Karar Ağacı
decision_tree_results = {
    'Model': 'Decision Tree',
    'Accuracy': 0.8206,
    'Precision': 0.8209868520235578,
    'Recall': 0.8206,
    'F1 Score': 0.8602841451511907
}

# Lojistik Regresyon
logistic_regression_results = {
    'Model': 'Logistic Regression',
    'Accuracy': 0.9709,
    'Precision': 0.9748943563777175,
    'Recall': 0.9709,
    'F1 Score': 0.978812400936089
}

# K-En Yakın Komşu
knn_results = {
    'Model': 'K-Nearest Neighbors',
    'Accuracy': 0.9349,
    'Precision': 0.9351942651883611,
    'Recall': 0.9349,
    'F1 Score': 0.9347298321706606
}

# Vektör Makineleri
svm_results = {
    'Model': 'Support Vector Machines',
    'Accuracy': 0.9254,
    'Precision': 0.9155626829011379,
    'Recall': 0.9254,
    'F1 Score': 0.9053450859479441
}

# Rastgele Orman
random_forest_results = {
    'Model': 'Random Forest',
    'Accuracy': 0.9254,
    'Precision': 0.9255626829011379,
    'Recall': 0.9254,
    'F1 Score': 0.9253450859479441
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
