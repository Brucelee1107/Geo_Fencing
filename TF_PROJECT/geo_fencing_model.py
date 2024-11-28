import tensorflow as tf
import numpy as np

# Sample function for training a simple decision model for geofencing (binary classification)
def create_geofencing_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(2,)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Output: inside(1) or outside(0)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Training data for a simple geofence (latitude, longitude) -> inside (1) or outside (0)
def train_geofencing_model():
    latitudes = [9.881132, 34.0522, 40.7128, 37.7749]  # Example latitudes
    longitudes = [78.072591, -118.2437, -74.0060, -122.4194]  # Example longitudes
    labels = [1, 0, 0, 0]  # 1: inside, 0: outside
    
    data = np.array(list(zip(latitudes, longitudes)))
    labels = np.array(labels)
    
    model = create_geofencing_model()
    model.fit(data, labels, epochs=10)
    
    return model

# Example function to predict using the trained model
def predict_geofence(model, lat, lon):
    prediction = model.predict(np.array([[lat, lon]]))
    return prediction[0][0] > 0.5  # If prediction > 0.5, inside; else outside

# Example usage
def main():
    # Train the model
    model = train_geofencing_model()

    # Input coordinates for testing
    test_lat = 9.8800  # Test latitude
    test_lon = 78.0700  # Test longitude

    # Use the trained model to predict if the test point is inside the geofence
    result = predict_geofence(model, test_lat, test_lon)
    
    if result:
        print("Device location is inside the geofence.")
    else:
        print("Device location is outside the geofence.")
    
    # Convert model to TFLite
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open('geofencing_model.tflite', 'wb') as f:
        f.write(tflite_model)
    print("Model converted to TFLite!")


main()

