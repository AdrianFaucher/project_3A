import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def display_hubble_MW(df:pd.DataFrame)->None:
    """Display the velocities of galaxies depending on the distance from the milky way
    and calculate H_0

    Args:
        df (pd.DataFrame): DataFrame containing the galaxies
    """
    x_series=df.loc[~df['Name'].str.startswith('CoM_'), 'Dis']
    y_series=df.loc[~df['Name'].str.startswith('CoM_'), 'VLG']
    
    plt.figure(figsize=(8, 5))
    plt.scatter(x_series, y_series, color='b')

    # Linear regression
    x = x_series.values
    y = y_series.values

    A = x.reshape(-1, 1)  # Mise en forme pour lstsq
    a, _, _, _ = np.linalg.lstsq(A, y, rcond=None)  # Résolution du système


    # Tracé de la droite de régression
    x_vals = np.linspace(x.min(), x.max(), 100)
    y_vals = a * x_vals  # Pas de b
    plt.plot(x_vals, y_vals, 'r--', label=f"$H_0 = {a[0]:.2f}\\, \\text{{km.s}}^{{-1}}\\text{{.Mpc}}^{{-1}}$")

    # Affichage
    plt.xlabel("Distance from milky way (Mpc)")
    plt.ylabel(r"Radial velocity (km/s)")
    plt.title("Radial velocity depending on distance from Milky way")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def display_velocities_distance(df,velocities:list[str],galaxy_center:str,border:bool=False,):
    n=len(velocities)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))  # ajusted size for the number of subplot

    # Si n == 1, axes n'est pas un tableau, donc on le met dans une liste pour la boucle
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(df['dis_center'], df['major_infall_velocity'], color='b', label=r'$v_{r,\text{maj}}$') # Tracé des points
        ax.set_xlabel("Distance from "+galaxy_center+" (Mpc)")
        ax.set_ylabel(r"$v_{r,\text{maj}}$ (km/s)")
        ax.set_title(r"$V_{r,\text{maj}}$ depending on distance from "+galaxy_center)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 5)  # Limites de l'axe x entre 1 et 5
        ax.set_ylim(-100, 500)  # Limites de l'axe y entre -25 et 20

    plt.tight_layout()
    plt.show()