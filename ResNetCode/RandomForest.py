import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier

# MNIST veri setini yükleme
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Verileri ResNet50 için uygun formata getirme ve normalleştirme
X_train = X_train[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0

# Veriyi daha az bellek kullanarak yeniden boyutlandırma
X_train_resized = tf.image.resize(X_train, (32, 32))
X_test_resized = tf.image.resize(X_test, (32, 32))

# Verileri RGB formatına çevirme
X_train_rgb = tf.image.grayscale_to_rgb(X_train_resized)
X_test_rgb = tf.image.grayscale_to_rgb(X_test_resized)

# ResNet50 modelini indirme ve özellik çıkarıcı olarak kullanma
resnet_model = ResNet50(weights='imagenet', include_top=False, input_shape=(32, 32, 3))
resnet_model.trainable = False

# Eğitim ve test verilerini özellik çıkarıcıdan geçirme
X_train_features = resnet_model.predict(X_train_rgb)
X_test_features = resnet_model.predict(X_test_rgb)

# Random Forest modelini oluşturma ve eğitme
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)
random_forest.fit(X_train_features.reshape(X_train_features.shape[0], -1), y_train)

# Test seti üzerinde modeli değerlendirme
X_test_flatten = X_test_features.reshape(X_test_features.shape[0], -1)
predictions_rf = random_forest.predict(X_test_flatten)

# Accuracy
accuracy_rf = accuracy_score(y_test, predictions_rf)
print("Random Forest Modeli Doğruluğu:", accuracy_rf)

# Precision
precision_rf = precision_score(y_test, predictions_rf, average='weighted')
print("Random Forest Modeli Doğruluğu: Precision:", precision_rf)

# Recall
recall_rf = recall_score(y_test, predictions_rf, average='weighted')
print("Random Forest Modeli Doğruluğu: Recall:", recall_rf)

# F1 Score
f1_rf = f1_score(y_test, predictions_rf, average='weighted')
print("Random Forest Modeli Doğruluğu: F1 Score:", f1_rf)