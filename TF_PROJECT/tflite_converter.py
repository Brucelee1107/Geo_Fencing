import tensorflow as tf

# Load your existing .h5 model
model = tf.keras.models.load_model('geo_fencing_model.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Perform the conversion
tflite_model = converter.convert()

# Save the converted model to a .tflite file
with open('geo_fencing_model.tflite', 'wb') as f:
    f.write(tflite_model)

