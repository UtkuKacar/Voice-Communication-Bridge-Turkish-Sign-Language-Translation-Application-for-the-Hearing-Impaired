import numpy as np

# İlk veri setini yükle
X_train1 = np.load('X_train5.npy')
X_test1 = np.load('X_test5.npy')
y_train1 = np.load('y_train5.npy')
y_test1 = np.load('y_test5.npy')

# İkinci veri setini yükle
X_train2 = np.load('X_train6.npy')
X_test2 = np.load('X_test6.npy')
y_train2 = np.load('y_train6.npy')
y_test2 = np.load('y_test6.npy')

# Eğitim verilerini birleştir
X_train = np.concatenate((X_train1, X_train2), axis=0)
y_train = np.concatenate((y_train1, y_train2), axis=0)

# Test verilerini birleştir
X_test = np.concatenate((X_test1, X_test2), axis=0)
y_test = np.concatenate((y_test1, y_test2), axis=0)

# Birleştirilmiş verileri kaydet
np.save('X_train7.npy', X_train)
np.save('y_train7.npy', y_train)
np.save('X_test7.npy', X_test)
np.save('y_test7.npy', y_test)

print("Birleştirilmiş veriler başarıyla kaydedildi:")
print("X_train7 shape:", X_train.shape)
print("y_train7 shape:", y_train.shape)
print("X_test7 shape:", X_test.shape)
print("y_test7 shape:", y_test.shape)
