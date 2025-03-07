from geographiclib.geodesic import Geodesic

result = Geodesic.WGS84.Inverse(48.71, -74.00, 48.86, 2.35)
print(f"距离:{result['s12']/1000:.1f}km")
print(f"初始方位角:{result['azi1']:.1f}度")