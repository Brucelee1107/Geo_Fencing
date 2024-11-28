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

if __name__ == "__main__":
    # Center point latitude and longitude
    center_lat = 9.881132
    center_lon = 78.072591
    radius = 100  # Radius in meters

    # List of points to check
    points = [
        (9.880693, 78.072962),
        (9.881047, 78.073264),
        (9.881494, 78.073929),
        (9.881596, 78.073992),
        (9.881530, 78.073157),
        (9.881758, 78.073135),
        (9.882394, 78.073076),
        (9.881908, 78.072223),
        (9.881724, 78.072204)
    ]

    # Check each point
    for i, (lat, lon) in enumerate(points, start=1):
        within = is_within_radius(center_lat, center_lon, lat, lon, radius)
        status = "inside" if within else "outside"
        print(f"Point {i}: Location ({lat}, {lon}) is {status} the {radius}-meter radius.")

