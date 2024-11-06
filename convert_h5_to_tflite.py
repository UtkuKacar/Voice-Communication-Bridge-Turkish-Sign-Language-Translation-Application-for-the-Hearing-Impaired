import tensorflow as tf

# Keras modelini .h5 dosyasından yükleme
model = tf.keras.models.load_model('final_action_model.h5')

# Modeli TensorFlow Lite formatına dönüştürme
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Gelişmiş seçenekleri etkinleştirme
converter.experimental_enable_resource_variables = True
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS,  # TensorFlow Lite'ın yerleşik işlemleri
    tf.lite.OpsSet.SELECT_TF_OPS  # TensorFlow işlemlerinin seçimi
]
converter._experimental_lower_tensor_list_ops = False

# Modeli dönüştürme
tflite_model = converter.convert()

# TensorFlow Lite modelini dosya olarak kaydetme
tflite_model_path = 'model.tflite'
with open(tflite_model_path, 'wb') as f:
    f.write(tflite_model)

print(f'TensorFlow Lite modeliniz {tflite_model_path} yolunda kaydedildi.')
