import numpy as np
import pandas as pd

def cartesian_to_equatorial(x: float, y: float, z: float)->np.ndarray:
    """convert cartesians coordinates to equatorial coordinates

    Args:
        x (float): Cartesian coordinates along the x axis
        y (float): Cartesian coordinates along the y axis
        z (float): Cartesian coordinates along the z axis

    Returns:
        np.ndarray[float]: [r : float ,right ascension:  radians, declinaison: radians]
    """
    r = np.sqrt(x**2 + y**2 + z**2)
    dec = np.arcsin(z / r) if r != 0 else 0  # Déclinaison (-pi/2 <= delta <= pi/2)
    ra = np.arctan2(y, x)  # Ascension droite (-pi <= alpha <= pi)
    if ra < 0:
        ra += 2 * np.pi  # convert in [0, 2pi] if necessary
    return np.array([r, ra, dec])

def equatorial_to_cartesian(r: float, alpha:float, delta:float)->np.ndarray[float]:
    """convert equatorial coordinates to carteseians coordinates

    Args:
        r (float): norm
        alpha (float): right ascension
        delta (float): declinaison

    Returns:
        np.ndarray[float]: [x,y,z] cartesians coordinates
    """
    x = r * np.cos(delta) * np.cos(alpha)
    y = r * np.cos(delta) * np.sin(alpha)
    z = r * np.sin(delta)
    return np.array([x,y,z])

def mean_square_peculiar_velocity(df, column):
    mask = ~df['Name'].str.startswith('CoM_')
    filtered = df[(df[column] >= -100) & (df[column] <= 500)]
    return (filtered[mask][column] ** 2).mean()
# 
# def mean_square_peculiar_velocity(df,column):
#     return (df[column] ** 2).mean()