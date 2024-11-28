import numpy as np
import tensorflow as tf

# Function to calculate median using TensorFlow operations
def calculate_median(inputs):
    # Compute the median by sorting the tensor
    sorted_lat = tf.sort(inputs[:, 0])
    sorted_lon = tf.sort(inputs[:, 1])
    
    # Get the size of the data
    n = tf.shape(sorted_lat)[0]

    # Find the middle index
    middle = n // 2

    # Handle even and odd cases for the median
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
    return median_lat, median_lon

# Define a TensorFlow model with a standalone function
@tf.function(input_signature=[tf.TensorSpec(shape=[None, 2], dtype=tf.float32)])
def median_model(inputs):
    median_lat, median_lon = calculate_median(inputs)
    return {"median_lat": median_lat, "median_lon": median_lon}

# Function to convert the model to TFLite
def convert_to_tflite(saved_model_dir, tflite_file):
    # Convert the SavedModel to TFLite format
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    tflite_model = converter.convert()

    # Save the TFLite model to a file
    with open(tflite_file, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to: {tflite_file}")

# Main function
def main():
    # Path to save the TensorFlow model
    saved_model_dir = "saved_model_median"

    # Create a TensorFlow model and save it
    tf.saved_model.save(median_model, saved_model_dir)

    # Path for the TFLite file
    tflite_file = "median_model.tflite"

    # Convert the TensorFlow model to TFLite
    convert_to_tflite(saved_model_dir, tflite_file)

    # Test the TFLite model
    interpreter = tf.lite.Interpreter(model_path=tflite_file)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Test input
    test_data = np.array([[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32)

    # Set input tensor
    interpreter.set_tensor(input_details[0]['index'], test_data)

    # Run inference
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print("TFLite Model Output:", output_data)

if __name__ == "__main__":
    main()

