from __future__ import annotations
import math
from typing import Tuple

R_EARTH_KM: float = 6371.0
_EPS = 1e-12

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat/2.0)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon/2.0)**2
    return 2.0 * R_EARTH_KM * math.asin(min(1.0, math.sqrt(a)))

def initial_bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Forward azimuth from (lat1,lon1) to (lat2,lon2) in degrees [0,360)."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlam = math.radians(lon2 - lon1)
    y = math.sin(dlam) * math.cos(phi2)
    x = math.cos(phi1)*math.cos(phi2) + math.sin(phi1)*math.sin(phi2)*math.cos(dlam)
    theta = math.atan2(y, max(_EPS, x))
    return (math.degrees(theta) + 360.0) % 360.0

def latlon_to_local_xy_km(lat0: float, lon0: float, lat1: float, lon1: float) -> Tuple[float, float]:
    """
    Local tangent-plane approximation (equirectangular) around (lat0,lon0).
    Accurate for small 6h displacements.
    Returns (dx_km east-positive, dy_km north-positive).
    """
    dx = math.radians(lon1 - lon0) * math.cos(math.radians(lat0)) * R_EARTH_KM
    dy = math.radians(lat1 - lat0) * R_EARTH_KM
    return dx, dy

def local_xy_km_to_latlon(lat0: float, lon0: float, dx_km: float, dy_km: float) -> Tuple[float, float]:
    """Inverse of latlon_to_local_xy_km."""
    lat1 = lat0 + math.degrees(dy_km / R_EARTH_KM)
    denom = R_EARTH_KM * max(_EPS, math.cos(math.radians(lat0)))
    lon1 = lon0 + math.degrees(dx_km / denom)
    return lat1, lon1
