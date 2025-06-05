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

def identify_abnormal_galaxies_per_velocity(df, velocity, mass_ratios, partial_row_name, lower_bound, upper_bound, max_dix):
    """Identifie les galaxies qui ont une vitesse anormale pour n'importe quel ratio de masse, pour un type de vitesse donné"""
    abnormal_galaxies = set()
    
    for mass_ratio in mass_ratios:
        column_vel = f"{velocity}_{partial_row_name}_{mass_ratio}"
        column_dis = f"dis_center_{partial_row_name}_{mass_ratio}"
        mask = ~df['Name'].str.startswith('CoM_')  # Exclut les centres de masse
        # Identifie les galaxies avec des vitesses anormales
        abnormal = df[mask & ((df[column_vel] < lower_bound) | (df[column_vel] > upper_bound) | (df[column_dis] > max_dix) )]['Name']
        abnormal_galaxies.update(abnormal)
    
    return abnormal_galaxies

def mean_square_peculiar_velocity_consistent(df, column, abnormal_galaxies):
    """Calcule la moyenne des carrés des vitesses en excluant les galaxies anormales"""
    mask = (~df['Name'].str.startswith('CoM_')) & (~df['Name'].isin(abnormal_galaxies))
    return (df[mask][column] ** 2).mean()



def min_max_grid(f: callable, x0: np.ndarray, dr, n_steps: int = 10) -> (float, float):
    """Évalue f sur une grille dans l'hypercube x0 ± dr, et retourne le min et max.

    Args:
        f (callable): fonction à évaluer, qui prend un vecteur en entrée
        x0 (np.ndarray): vecteur numpy, point central
        dr: variation autour de x0. Peut être:
            - np.ndarray: erreur symétrique (même taille que x0)
            - tuple (dr_min, dr_max): erreurs asymétriques où dr_min et dr_max 
              sont des np.ndarray de même taille que x0
        n_steps (int, optional): nombre de points à tester par dimension (>=2). Defaults to 10.
    
    Return:
        (val_min, val_max) (float, float): valeurs minimale et maximale de f dans l'hypercube
    """
    x0 = np.array(x0)
    n = len(x0)
    
    # Gestion des erreurs symétriques vs asymétriques
    if isinstance(dr, tuple) and len(dr) == 2:
        # Cas asymétrique: dr = (dr_min, dr_max)
        dr_min, dr_max = dr
        dr_min = np.array(dr_min)
        dr_max = np.array(dr_max)
        
        if len(dr_min) != n or len(dr_max) != n:
            raise ValueError("dr_min et dr_max doivent avoir la même taille que x0")
            
        axes = [np.linspace(x0[i] - dr_min[i], x0[i] + dr_max[i], n_steps) for i in range(n)]
    else:
        # Cas symétrique: dr est un array
        dr = np.array(dr)
        if len(dr) != n:
            raise ValueError("dr doit avoir la même taille que x0")
            
        axes = [np.linspace(x0[i] - dr[i], x0[i] + dr[i], n_steps) for i in range(n)]
    
    points = itertools.product(*axes)
    values = [f(np.array(p)) for p in points]
    
    return min(values), max(values)


    
