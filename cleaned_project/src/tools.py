import numpy as np
import pandas as pd
import itertools

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

def identify_abnormal_galaxies_per_velocity(df, velocity, mass_ratios, partial_row_name, lower_bound=-100, upper_bound=500):
    """Identifie les galaxies qui ont une vitesse anormale pour n'importe quel ratio de masse, pour un type de vitesse donné"""
    abnormal_galaxies = set()
    
    for mass_ratio in mass_ratios:
        column = f"{velocity}_{partial_row_name}_{mass_ratio}"
        mask = ~df['Name'].str.startswith('CoM_')  # Exclut les centres de masse
        # Identifie les galaxies avec des vitesses anormales
        abnormal = df[mask & ((df[column] < lower_bound) | (df[column] > upper_bound))]['Name']
        abnormal_galaxies.update(abnormal)
    
    return abnormal_galaxies

def mean_square_peculiar_velocity_consistent(df, column, abnormal_galaxies):
    """Calcule la moyenne des carrés des vitesses en excluant les galaxies anormales"""
    mask = (~df['Name'].str.startswith('CoM_')) & (~df['Name'].isin(abnormal_galaxies))
    return (df[mask][column] ** 2).mean()



def min_max_grid(f:callable, x0:np.ndarray, dr:np.ndarray, n_steps:int =10)->(float,float):
    """Évalue f sur une grille dans l'hypercube x0 ± dr, et retourne le min et max.

    Args:
        f (_type_): fonction à évaluer, qui prend un vecteur en entrée
        x0 (np.ndarray): vecteur numpy, point central
        dr (np.ndarrray): vecteur numpy, variation maximale autour de x0 (même taille que x0)
        n_steps (int, optional): nombre de points à tester par dimension (>=2). Defaults to 100.
    
    Retrun:
        (val_min, val_max) (float,float) : valeurs minimale et maximale de f dans l'hypercube
    """
    x0 = np.array(x0)
    dr = np.array(dr)
    n = len(x0)
    
    axes = [np.linspace(x0[i] - dr[i], x0[i] + dr[i], n_steps) for i in range(n)]
    
    points = itertools.product(*axes)

    values = [f(np.array(p)) for p in points]
    
    return min(values), max(values)


    
