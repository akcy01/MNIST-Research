import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Yeni Sonuçlar
new_decision_tree_results = {
    'Model': 'Decision Tree',
    'Accuracy': 0.6115,
    'Precision': 0.6115507982425774,
    'Recall': 0.6115,
    'F1 Score': 0.61132194619468
}

new_logistic_regression_results = {
    'Model': 'Logistic Regression',
    'Accuracy': 0.6702,
    'Precision': 0.6714810545072111,
    'Recall': 0.6702,
    'F1 Score': 0.6672450817465332
}

new_knn_results = {
    'Model': 'K-Nearest Neighbors',
    'Accuracy': 0.6789,
    'Precision': 0.6814970397684801,
    'Recall': 0.6789,
    'F1 Score': 0.6777741981433818
}

new_svm_results = {
    'Model': 'Support Vector Machines',
    'Accuracy': 0.7242,
    'Precision': 0.7280839419753946,
    'Recall': 0.7242,
    'F1 Score': 0.7233653774660558
}

new_random_forest_results = {
    'Model': 'Random Forest',
    'Accuracy': 0.741,
    'Precision': 0.7388382424098011,
    'Recall': 0.741,
    'F1 Score': 0.738859051040355
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
