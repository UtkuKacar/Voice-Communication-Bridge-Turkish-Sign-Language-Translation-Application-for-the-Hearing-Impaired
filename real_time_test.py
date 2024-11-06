import numpy as np
import cv2
from mediapipe_utils import mediapipe_detection, draw_styled_landmarks, extract_keypoints, mp_holistic
from tensorflow.keras.models import load_model
from scipy import stats

# Hareketlerin tanımlanması
actions = np.array(['Evet', 'Güle Güle', 'Günaydın', 'Hayır', 'Merhaba', 'Özür Dilerim', 'Rica Ederim', 'Seni Seviyorum', 'Teşekkür Ederim'])
model = load_model('final_action_model2.h5')

# Tüm hareketler için yeterli renklerin olduğundan emin olun
colors = [(245, 117, 16), (117, 245, 16), (16, 117, 245)]
if len(colors) < len(actions):
    additional_colors = [(255, 255, 255)] * (len(actions) - len(colors))
    colors.extend(additional_colors)

# Olasılık görselleştirme fonksiyonu
def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    for num, prob in enumerate(res):
        color = colors[num % len(colors)]  # Renkleri döngüsel olarak kullan
        cv2.rectangle(output_frame, (0, 60 + num * 40), (int(prob * 100), 90 + num * 40), color, -1)
        cv2.putText(output_frame, actions[num], (0, 85 + num * 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    return output_frame

cap = cv2.VideoCapture(0)
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    sequence = []
    sentence = []
    predictions = []
    threshold = 0.5

    while cap.isOpened():
        ret, frame = cap.read()
        image, results = mediapipe_detection(frame, holistic)
        draw_styled_landmarks(image, results)
        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-30:]  # Sadece son 30 frame'i tut

        if len(sequence) == 30:
            res = model.predict(np.expand_dims(sequence, axis=0))[0]
            print(actions[np.argmax(res)])
            predictions.append(np.argmax(res))
        
            # Son 10 tahminin çoğunluğunu kontrol et ve eşik değerini karşılaştır
            if predictions and np.unique(predictions[-10:])[0] == np.argmax(res):
                if res[np.argmax(res)] > threshold:
                    if len(sentence) > 0:
                        if actions[np.argmax(res)] != sentence[-1]:
                            sentence.append(actions[np.argmax(res)])
                    else:
                        sentence.append(actions[np.argmax(res)])

            if len(sentence) > 5:
                sentence = sentence[-5:]

            image = prob_viz(res, actions, image, colors)
        
        cv2.rectangle(image, (0, 0), (640, 40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow('OpenCV Feed', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
