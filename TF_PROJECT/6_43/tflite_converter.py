import tensorflow as tf

# Load your existing .h5 model
model = tf.keras.models.load_model('find_median_and_model_conversion.py')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Perform the conversion
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('Median_1.tflite', 'wb') as f:
    f.write(tflite_model)

