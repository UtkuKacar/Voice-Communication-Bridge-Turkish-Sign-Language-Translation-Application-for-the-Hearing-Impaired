import cv2
import numpy as np
import os
import mediapipe as mp
from mediapipe_utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Veri yolunu ve etiketleri tanımlama
DATA_PATH = os.path.join('MP_Data5')  # Veri dosyalarının saklanacağı yol
actions = np.array(['Evet', 'Güle Güle', 'Günaydın', 'Hayır', 'Merhaba', 'Özür Dilerim', 'Rica Ederim', 'Seni Seviyorum', 'Teşekkür Ederim'])  # Tanımlanan hareketler

# Video ve sekans uzunluklarını tanımlama
no_sequences = 30  # Her hareket için 30 video
sequence_length = 30  # Her video 30 frame uzunluğunda

# Veri seti için dizinleri oluşturma
for action in actions: 
    for sequence in range(no_sequences):
        try: 
            os.makedirs(os.path.join(DATA_PATH, action, str(sequence)))
        except FileExistsError:
            pass

cap = cv2.VideoCapture(0)  # Kameradan görüntü alma
# Mediapipe modelini ayarlama
with mp.solutions.holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    X_data = []  # Anahtar nokta verilerini saklamak için liste
    y_data = []  # Etiketleri saklamak için liste

    # Hareketleri döngü ile geçme
    for action in actions:
        # Videoları döngü ile geçme
        for sequence in range(no_sequences):
            # Video uzunluğunu döngü ile geçme
            for frame_num in range(sequence_length):

                # Kameradan görüntü okuma
                ret, frame = cap.read()

                # Medaipipe ile tespit işlemi
                image, results = mediapipe_detection(frame, holistic)

                # Landmarks çizdirme
                draw_styled_landmarks(image, results)
                
                # Veri toplama işlemi başlangıcında bekleme mantığı
                if frame_num == 0: 
                    cv2.putText(image, 'STARTING COLLECTION', (120,200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255, 0), 4, cv2.LINE_AA)
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Ekrana görüntü bastırma
                    cv2.imshow('OpenCV Feed', image)
                    cv2.waitKey(2000)  # 2 saniye bekleme
                else: 
                    cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(action, sequence), (15,12), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                    # Ekrana görüntü bastırma
                    cv2.imshow('OpenCV Feed', image)
                
                # Anahtar noktaları dışa aktarma
                keypoints = extract_keypoints(results)
                npy_path = os.path.join(DATA_PATH, action, str(sequence), str(frame_num) + '.npy')
                
                # Dizin mevcut değilse oluşturma
                if not os.path.exists(os.path.dirname(npy_path)):
                    try:
                        os.makedirs(os.path.dirname(npy_path))
                    except OSError as exc:
                        if exc.errno != errno.EEXIST:
                            raise

                np.save(npy_path, keypoints)  # Anahtar noktaları .npy dosyasına kaydetme
                X_data.append(keypoints)  # Anahtar noktaları X_data listesine ekleme
                y_data.append(action)  # Etiketleri y_data listesine ekleme

                # Döngüyü sonlandırma
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                    
cap.release()  # Kamerayı serbest bırakma
cv2.destroyAllWindows()  # Tüm OpenCV pencerelerini kapatma

# Etiketleri sayısal değerlere dönüştürme
le = LabelEncoder()
y_data = le.fit_transform(y_data)

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

# Veri setlerini kaydetme
np.save('X_train6.npy', X_train)
np.save('y_train6.npy', y_train)
np.save('X_test6.npy', X_test)
np.save('y_test6.npy', y_test)
np.save('actions7.npy', actions)
