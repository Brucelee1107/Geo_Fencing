import tensorflow as tf

def main():
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/dinesh/project/Geo_Fencing/lat_long_within_radius.py")
    tflite_model = converter.convert()
    with open('model.tflite', 'wb') as f: 
        f.write(tflite_model)

main()
