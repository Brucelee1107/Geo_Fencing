import math

def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in meters
    R = 6371000
    # Convert degrees to radians
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
    distance = haversine(center_lat, center_lon, target_lat, target_lon)
    return distance <= radius

def calculate_median_lat_lon(input_file):
    try:
        latitudes = []
        longitudes = []

        with open(input_file, 'r') as file:
            for line in file:
                line = line.strip()
                if line:
                    lat, lon = map(float, line.split(","))  # Assuming lat and lon are comma-separated
                    latitudes.append(lat)
                    longitudes.append(lon)

        if len(latitudes) == 0 or len(longitudes) == 0:
            raise ValueError("No valid latitude and longitude values in the file.")

        # For even length lists, calculate the average of the two middle values
        sorted_latitudes = sorted(latitudes)
        sorted_longitudes = sorted(longitudes)
        if len(latitudes) % 2 == 0:
            # If the length is even, average the two middle values
            median_lat = (sorted_latitudes[len(latitudes) // 2 - 1] + sorted_latitudes[len(latitudes) // 2]) / 2
            median_lon = (sorted_longitudes[len(longitudes) // 2 - 1] + sorted_longitudes[len(longitudes) // 2]) / 2
        else:
            # If the length is odd, take the middle value
            median_lat = sorted_latitudes[len(latitudes) // 2]
            median_lon = sorted_longitudes[len(longitudes) // 2]
        return median_lat, median_lon

    except FileNotFoundError:
        print(f"Error: File '{input_file}' not found.")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None

if __name__ == "__main__":
    # Center point latitude and longitude
    center_lat = 9.881132  # latitude
    center_lon = 78.072591  # longitude
    radius = 100  # Radius in meters

    # Input file containing latitudes and longitudes
    input_file = '/home/dinesh/project/Geo_Fencing/lat_long.txt'

    # Calculate the median latitude and longitude from the file
    median_lat, median_lon = calculate_median_lat_lon(input_file)

    if median_lat is not None and median_lon is not None:
        # Check if the median point is within the radius
        within_radius = is_within_radius(center_lat, center_lon, median_lat, median_lon, radius)

        status = "inside" if within_radius else "outside"
        print(f"Median location ({median_lat}, {median_lon}) is {status} the {radius}-meter radius.")

