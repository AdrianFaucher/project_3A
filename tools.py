import numpy as np

def cartesian_to_equatorial(x, y, z):
    r = np.sqrt(x**2 + y**2 + z**2)
    dec = np.arcsin(z / r) if r != 0 else 0  # Déclinaison (-pi/2 <= delta <= pi/2)
    ra = np.arctan2(y, x)  # Ascension droite (-pi <= alpha <= pi)
    if ra < 0:
        ra += 2 * np.pi  # Convertir en [0, 2pi] si nécessaire
    return np.array([r, ra, dec])

def equatorial_to_cartesian(r, alpha, delta):
    x = r * np.cos(delta) * np.cos(alpha)
    y = r * np.cos(delta) * np.sin(alpha)
    z = r * np.sin(delta)
    return np.array([x,y,z])