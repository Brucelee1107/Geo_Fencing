import tensorflow as tf
import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in meters
    R = 6371000  # Earth radius in meters

    # Convert degrees to radians using math.radians
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    # Haversine formula
    a = math.sin(delta_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Distance in meters
    return R * c

def is_within_radius(center_lat, center_lon, target_lat, target_lon, radius):
    # Calculate the distance using the haversine formula 
    distance = haversine(center_lat, center_lon, target_lat, target_lon)
    # Check if the distance is within the radius
    return distance <= radius

def calculate_median_lat_lon(latitudes, longitudes):
    # Sort lat and long using TensorFlow's tf.sort
    sorted_latitudes = tf.sort(latitudes)
    sorted_longitudes = tf.sort(longitudes)

    # Calculate the median value using TensorFlow
    lat_len = tf.shape(latitudes)[0]
    lon_len = tf.shape(longitudes)[0]

    if lat_len % 2 == 0:
        median_lat = (sorted_latitudes[lat_len // 2 - 1] + sorted_latitudes[lat_len // 2]) / 2
        median_lon = (sorted_longitudes[lon_len // 2 - 1] + sorted_longitudes[lon_len // 2]) / 2
    else:
        median_lat = sorted_latitudes[lat_len // 2]
        median_lon = sorted_longitudes[lon_len // 2]

    return median_lat, median_lon

def check_geofence(center_lat, center_lon, target_latitudes, target_longitudes, radius):
    # Calculate the median lat and long using the helper function
    median_lat, median_lon = calculate_median_lat_lon(target_latitudes, target_longitudes)

    # Check if the median point is within the radius using the helper function
    return is_within_radius(center_lat, center_lon, median_lat, median_lon, radius)

def main():
    # Sample inputs
    center_lat = 9.881132  # latitude
    center_lon = 78.072591  # longitude
    radius = 100  # Radius in meters

    # Input file containing latitudes and longitudes (example data)
    input_file = '/home/dinesh/project/Geo_Fencing/lat_long.txt'

    # Read the latitude and longitude values from the file
    latitudes = []
    longitudes = []
    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()
            if line:
                lat, lon = map(float, line.split(","))
                latitudes.append(lat)
                longitudes.append(lon)

    # Convert to TensorFlow tensors
    latitudes = tf.constant(latitudes, dtype=tf.float32)
    longitudes = tf.constant(longitudes, dtype=tf.float32)

    # to see if the median point is within the radius
    result = check_geofence(center_lat=center_lat, center_lon=center_lon, 
                            target_latitudes=latitudes, target_longitudes=longitudes, 
                            radius=radius)

    # Directly use the result (boolean value)
    if result:
        print(f"Device location is inside the {radius}-meter radius.")
    else:
        print(f"Device location is outside the {radius}-meter radius.")



main()

