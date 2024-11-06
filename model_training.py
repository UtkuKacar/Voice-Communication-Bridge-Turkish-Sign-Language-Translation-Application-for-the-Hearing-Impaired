import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping

# TensorBoard için log dizini
base_log_dir = 'Logs'
if not os.path.exists(base_log_dir):
    os.makedirs(base_log_dir)

# Checkpoint yolu ve dizini
checkpoint_path = "model_checkpoints/cp-{epoch:04d}.weights.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Checkpoint dizini yoksa oluştur
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

# Early stopping callback
early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Verileri yükle
X_train = np.load('X_train5.npy')
y_train = np.load('y_train5.npy')
X_test = np.load('X_test5.npy')
y_test = np.load('y_test5.npy')

# Verilerin normalizasyonu
X_train = X_train / 255.0
X_test = X_test / 255.0

# Sınıf ağırlıklarını hesapla
actions = np.load('actions6.npy')  # Actions dizisini yükle
class_weights = {i: len(y_train) / (len(actions) * np.sum(y_train[:, i])) for i in range(len(actions))}

# Model tanımlama fonksiyonu
def create_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, activation='relu', input_shape=(30, 1662)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Aşırı öğrenmeyi engellemek için dropout
    model.add(LSTM(256, return_sequences=True, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Aşırı öğrenmeyi engellemek için dropout
    model.add(LSTM(128, return_sequences=False, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))  # Aşırı öğrenmeyi engellemek için dropout
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(len(actions), activation='softmax'))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    return model

# 5-fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold_no = 1
results = []

# Her fold için eğitim ve değerlendirme
for train_index, val_index in kf.split(X_train):
    print(f'Training on fold {fold_no}...')
    
    # Verileri ayır
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Yeni bir model örneği oluştur
    model = create_model()
    
    # Bu fold için benzersiz bir TensorBoard log dizini oluşturun
    fold_log_dir = os.path.join(base_log_dir, f"fold_{fold_no}")
    if not os.path.exists(fold_log_dir):
        os.makedirs(fold_log_dir)
    
    tb_callback = TensorBoard(log_dir=fold_log_dir)
    
    # Bu fold için checkpoint callback
    fold_checkpoint_path = f"model_checkpoints/fold{fold_no}-cp.weights.h5"
    cp_callback = ModelCheckpoint(
        filepath=fold_checkpoint_path,
        verbose=1,
        save_weights_only=True,
        save_freq='epoch'
    )
    
    # Modeli eğit
    model.fit(X_train_fold, y_train_fold, epochs=200, batch_size=32, callbacks=[tb_callback, cp_callback, early_stopping_callback], validation_data=(X_val_fold, y_val_fold), class_weight=class_weights)
    
    # Modeli test verisi üzerinde değerlendir
    yhat = model.predict(X_test)
    ytrue = np.argmax(y_test, axis=1).tolist()
    yhat = np.argmax(yhat, axis=1).tolist()
    
    # Fold sonuçlarını kaydet
    fold_results = {
        'confusion_matrix': multilabel_confusion_matrix(ytrue, yhat),
        'accuracy_score': accuracy_score(ytrue, yhat)
    }
    results.append(fold_results)
    
    print(f"Fold {fold_no} - Accuracy: {fold_results['accuracy_score']}")
    
    fold_no += 1

# Tüm foldlar için ortalama sonuçları hesapla
avg_accuracy = np.mean([result['accuracy_score'] for result in results])
print(f'Average accuracy across all folds: {avg_accuracy}')

# Nihai modeli kaydet
model.save('final_action_model3.h5')
