import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

# MNIST veri setini yükleme
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Verileri EfficientNetB0 için uygun formata getirme ve normalleştirme
X_train = X_train[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0

# Veriyi daha az bellek kullanarak yeniden boyutlandırma
X_train_resized = tf.image.resize(X_train, (32, 32))
X_test_resized = tf.image.resize(X_test, (32, 32))

# Verileri RGB formatına çevirme
X_train_rgb = tf.image.grayscale_to_rgb(X_train_resized)
X_test_rgb = tf.image.grayscale_to_rgb(X_test_resized)

# EfficientNetB0 modelini indirme ve özellik çıkarıcı olarak kullanma
efficientnet_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
efficientnet_model.trainable = False

# Eğitim ve test verilerini özellik çıkarıcıdan geçirme
X_train_features = efficientnet_model.predict(X_train_rgb)
X_test_features = efficientnet_model.predict(X_test_rgb)

# Özellik haritasını düzleştirme
X_train_flatten = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_flatten = X_test_features.reshape(X_test_features.shape[0], -1)

# Veriyi daha küçük bir alt kümeyle kullanma
X_train_small, _, y_train_small, _ = train_test_split(X_train_flatten, y_train, train_size=0.1, random_state=42)

# K-Nearest Neighbors modelini oluşturma ve eğitme
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_small, y_train_small)

# Test seti üzerinde modeli değerlendirme
predictions_knn = knn.predict(X_test_flatten)

# Accuracy
accuracy_knn = accuracy_score(y_test, predictions_knn)
print("EfficientNetB0 ile K-Nearest Neighbors Modeli Doğruluğu:", accuracy_knn)

# Precision
precision_knn = precision_score(y_test, predictions_knn, average='weighted')
print("EfficientNetB0 ile K-Nearest Neighbors Modeli Doğruluğu: Precision:", precision_knn)

# Recall
recall_knn = recall_score(y_test, predictions_knn, average='weighted')
print("EfficientNetB0 ile K-Nearest Neighbors Modeli Doğruluğu: Recall:", recall_knn)

# F1 Score
f1_knn = f1_score(y_test, predictions_knn, average='weighted')
print("EfficientNetB0 ile K-Nearest Neighbors Modeli Doğruluğu: F1 Score:", f1_knn)