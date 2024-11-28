import h5py

# Read the Python script as a string
with open('geo_fencing_model.py', 'r') as file:
    script_content = file.read()

# Save the script as a string in an HDF5 file
with h5py.File('geo_fencing_model.h5', 'w') as f:
    f.create_dataset('python_script', data=script_content.encode('utf-8'))

import tensorflow as tf

# Example: Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Save the model to a custom HDF5 file name
model.save('geo_fencing_model.h5')


