import numpy as np
import tensorflow as tf

# Function to read lat and long from .txt file
def read_lat_lon_from_file(input_file):
    latitudes = []
    longitudes = []
    
    try:
        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip()  # Remove extra spaces
                if line:
                    lat, lon = map(float, line.split(","))  # Separate lat and long values using comma
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

# Function for TensorFlow model to calculate the median
class MedianModel(tf.keras.Model):
    def call(self, inputs):
        # sort the lat and long 
        sorted_lat = tf.sort(inputs[:, 0])
        sorted_lon = tf.sort(inputs[:, 1])
        
        # Get the size of the data
        n = tf.shape(sorted_lat)[0]

        # Find the middle data index
        middle = n // 2

        # Handle even and odd data in the file
        median_lat = tf.cond(
            tf.equal(n % 2, 0),
            lambda: (sorted_lat[middle - 1] + sorted_lat[middle]) / 2,
            lambda: sorted_lat[middle]
        )
        median_lon = tf.cond(
            tf.equal(n % 2, 0),
            lambda: (sorted_lon[middle - 1] + sorted_lon[middle]) / 2,
            lambda: sorted_lon[middle]
        )
        return tf.stack([median_lat, median_lon])

# Main function to calculate the median and convert the model
def main():
    # Path to the input file containing latitude and longitude data
    input_file = '/home/dinesh/project/Geo_Fencing/TF_PROJECT/6_43/lat_long.txt'  # Path to the text file

    # Read latitudes and longitudes from the file
    latitudes, longitudes = read_lat_lon_from_file(input_file)

    if latitudes is None or longitudes is None:
        print("Error: Unable to read data from file.")
        return
    
    # Convert all lat and long data to single array
    input_data = np.column_stack((latitudes, longitudes))

    # Create the TensorFlow model
    model = MedianModel()

    # Convert input data to TensorFlow tensor
    input_tensor = tf.convert_to_tensor(input_data, dtype=tf.float32)

    # Calculate the median using the model
    median = model(input_tensor)

    print(f"Median Latitude (using TensorFlow): {median[0].numpy()}")
    print(f"Median Longitude (using TensorFlow): {median[1].numpy()}")

    # Save the model in TensorFlow SavedModel format 
    saved_model_dir = "model_median"
    tf.saved_model.save(model, saved_model_dir)
    print(f"Model saved to: {saved_model_dir}")

    # Convert the model to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    tflite_file = "median_model.tflite"
    with open(tflite_file, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_file}")



main()

