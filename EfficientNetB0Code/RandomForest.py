import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from sklearn.ensemble import RandomForestClassifier
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

# Random Forest modelini oluşturma ve eğitme
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_small, y_train_small)

# Test seti üzerinde modeli değerlendirme
predictions_rf = random_forest.predict(X_test_flatten)

# Accuracy
accuracy_rf = accuracy_score(y_test, predictions_rf)
print("EfficientNetB0 ile Random Forest Modeli Doğruluğu:", accuracy_rf)

# Precision
precision_rf = precision_score(y_test, predictions_rf, average='weighted')
print("EfficientNetB0 ile Random Forest Modeli Doğruluğu: Precision:", precision_rf)

# Recall
recall_rf = recall_score(y_test, predictions_rf, average='weighted')
print("EfficientNetB0 ile Random Forest Modeli Doğruluğu: Recall:", recall_rf)

# F1 Score
f1_rf = f1_score(y_test, predictions_rf, average='weighted')
print("EfficientNetB0 ile Random Forest Modeli Doğruluğu: F1 Score:", f1_rf)