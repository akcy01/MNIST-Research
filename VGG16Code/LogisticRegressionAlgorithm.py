import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# MNIST veri setini yükleme
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

# Verileri VGG16 için uygun formata getirme ve normalleştirme
X_train = X_train[..., tf.newaxis] / 255.0
X_test = X_test[..., tf.newaxis] / 255.0

# Veriyi daha az bellek kullanarak yeniden boyutlandırma
X_train_resized = tf.image.resize(X_train, (64, 64))
X_test_resized = tf.image.resize(X_test, (64, 64))

# Verileri RGB formatına çevirme
X_train_rgb = tf.image.grayscale_to_rgb(X_train_resized)
X_test_rgb = tf.image.grayscale_to_rgb(X_test_resized)

# VGG16 modelini indirme ve özellik çıkarıcı olarak kullanma
vgg16_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
vgg16_model.trainable = False

# Eğitim ve test verilerini özellik çıkarıcıdan geçirme
X_train_features = vgg16_model.predict(X_train_rgb)
X_test_features = vgg16_model.predict(X_test_rgb)

# Verileri düzenleme ve normalleştirme
X_train_flatten = X_train_features.reshape(X_train_features.shape[0], -1)
X_test_flatten = X_test_features.reshape(X_test_features.shape[0], -1)

# Lojistik Regresyon modelini oluşturma ve eğitme
logistic_regression = LogisticRegression(max_iter=500)
logistic_regression.fit(X_train_flatten, y_train)

# Test seti üzerinde modeli değerlendirme
predictions_logistic = logistic_regression.predict(X_test_flatten)

# Accuracy
accuracy_logistic = accuracy_score(y_test, predictions_logistic)
print("Lojistik Regresyon Modeli Doğruluğu:", accuracy_logistic)


# Precision
precision_svm = precision_score(y_test, predictions_logistic, average='weighted')
print("Karar Ağacı Modeli Doğruluğu: Precision:", precision_svm)

# Recall
recall_svm = recall_score(y_test, predictions_logistic, average='weighted')
print("Karar Ağacı Modeli Doğruluğu: Recall:", recall_svm)

# F1 Score
f1_svm = f1_score(y_test, predictions_logistic, average='weighted')
print("Karar Ağacı Modeli Doğruluğu: F1 Score:", f1_svm)
