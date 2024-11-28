import numpy as np
import math
import tensorflow as tf

# Haversine formula
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Earth radius in meters 
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)
    
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c  # Distance in meters

# Function to read lat and long from .txt file
def read_lat_lon_from_file(input_file):
    latitudes = []
    longitudes = []
    
    try:
        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip() # Remove the extra space before and after the values
                if line:
                    lat, lon = map(float, line.split(","))  # Split the lat and long values with comma
                    latitudes.append(lat)
                    longitudes.append(lon)
        if len(latitudes) == 0 or len(longitudes) == 0:
            raise ValueError("***No valid latitude and longitude values in the text file****")
        return np.array(latitudes), np.array(longitudes)
    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

# Function to generate training data from file
# Training data is used to train the machine learning (ML) model to make predictions
def generate_data_from_file(center_lat, center_lon, radius, input_file, num_samples=1000):
    latitudes, longitudes = read_lat_lon_from_file(input_file)
    if latitudes is None or longitudes is None:
        return None, None, None
    
    labels = [] # Empty list initiate
    for lat, lon in zip(latitudes, longitudes):
        # Calculate the distance using Haversine formula
        distance = haversine(center_lat, center_lon, lat, lon)
        
        # Label as 1 (inside) or 0 (outside) based on distance
        label = 1 if distance <= radius else 0
        labels.append(label)
    
    return latitudes, longitudes, np.array(labels)

# Create the TensorFlow model
def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, input_dim=2, activation='relu'),  # Input layer
        tf.keras.layers.Dense(32, activation='relu'),              # Hidden layer
        tf.keras.layers.Dense(1, activation='sigmoid')             # Output layer (binary classification)
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Main function to generate data, train model, and predict
def main():
    # Center point lat and long
    center_lat = 9.881132  # original lat
    center_lon = 78.072591  # original long
    radius = 100  # Radius in meters
    
    # Path to the input file containing latitude and longitude data
    input_file = '/home/dinesh/project/Geo_Fencing/TF_PROJECT/6_43/lat_long.txt'  # Path to our text file

    # Generate training data from the file
    latitudes, longitudes, labels = generate_data_from_file(center_lat, center_lon, radius, input_file)

    if latitudes is None or longitudes is None:
        print("Error: Unable to generate data from file.")
        return
    
    # Redirect to the model creating function
    model = create_model()

    # Prepare the input data (lat and long) for the model
    X = np.column_stack((latitudes, longitudes))
    y = labels

    # Train the model
    model.fit(X, y, epochs=10, batch_size=32)
    
    # Round-off lat and long point for prediction
    new_lat = 9.882  
    new_lon = 78.073  

    # Predict whether the point is inside or outside the radius
    prediction = model.predict(np.array([[new_lat, new_lon]]))

    # Output the prediction
    status = "inside" if prediction[0][0] > 0.5 else "outside"
    print(f"The point ({new_lat}, {new_lon}) is {status} the {radius}-meter radius.")

if __name__ == "__main__":
    main()

