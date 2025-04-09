import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from src.tools import mean_square_peculiar_velocity

def display_hubble_MW(df:pd.DataFrame)->None:
    """Display the velocities of galaxies depending on the distance from the milky way
    and calculate H_0

    Args:
        df (pd.DataFrame): DataFrame containing the galaxies
    """
    mask = ~df['Name'].str.startswith('CoM_'))
    # mask = ((df['f_Dis']== "h") & ~df['Name'].str.startswith('CoM_'))
    x_series = pd.to_numeric(df.loc[mask, 'Dis'], errors='coerce')
    y_series = pd.to_numeric(df.loc[mask, 'VLG'], errors='coerce')
    
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
    
def display_velocities_distance(df,velocities:list[str],row_name:str,border:bool=False,):
    mask = ~df['Name'].str.startswith('CoM_')
    n=len(velocities)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))  # ajusted size for the number of subplot

    # Si n == 1, axes n'est pas un tableau, donc on le met dans une liste pour la boucle
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.scatter(df[mask]['dis_center_'+row_name], df[mask][velocities[i]+"_"+row_name], color='b', label=r"$v_{r,\text{"+velocities[i]+r"}}$") # Tracé des points
        ax.set_xlabel("Distance from "+row_name+" (Mpc)")
        ax.set_ylabel(r"$v_{r,\text{"+velocities[i]+r"}}$ (km/s)")
        ax.set_title(r"$V_{r,\text{"+velocities[i]+r"}}$ depending on distance from "+row_name)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 5)  # Limites de l'axe x entre 1 et 5
        ax.set_ylim(-100, 500)  # Limites de l'axe y entre -25 et 20

    plt.tight_layout()
    plt.show()
    
def display_velocities_distance_color(df, velocities: list[str], row_name: str, border: bool = False):
    mask = ~df['Name'].str.startswith('CoM_')
    plt.figure(figsize=(12, 6))

    colors = plt.cm.get_cmap('tab10', len(velocities))  # Palette de couleurs

    for i, velocity in enumerate(velocities):
        plt.scatter(
            df[mask]['dis_center_' + row_name],
            df[mask][velocity + "_" + row_name],
            color=colors(i),
            label=r"$v_{r,\text{" + velocity + r"}}$"
        )

    plt.xlabel("Distance from " + row_name + " (Mpc)")
    plt.ylabel("Radial velocity (km/s)")
    plt.title("Radial velocities depending on distance from " + row_name)
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 5)
    plt.ylim(-100, 500)
    plt.tight_layout()
    plt.show()
    
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

def animated_velocities_distance(df, velocities: list[str], mass_ratios: np.ndarray,
                                  partial_row_name:str):
    mask = ~df['Name'].str.startswith('CoM_')
    n = len(velocities)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5 * n))

    # Si un seul subplot, le mettre dans une liste
    if n == 1:
        axes = [axes]

    # Créer un scatter par subplot
    scatters = []
    for i, ax in enumerate(axes):
        sc = ax.scatter([], [], label=rf"$v_{{r,\text{{{velocities[i]}}}}}$")
        scatters.append(sc)

        ax.set_xlim(0, 5)
        ax.set_ylim(-100, 500)
        ax.set_xlabel("Distance from " + partial_row_name + " (Mpc)")
        ax.set_ylabel(rf"$v_{{r,\text{{{velocities[i]}}}}}$ (km/s)")
        ax.set_title(rf"$V_{{r,\text{{{velocities[i]}}}}}$ depending on distance from " + partial_row_name)
        ax.grid(True)
        ax.legend()

    # Initialisation : vide les scatter plots
    def init():
        for sc in scatters:
            sc.set_offsets(np.empty((0, 2)))
        return scatters

    # Mise à jour à chaque frame
    def update(frame):
        mass_ratio = frame

        x = df[mask]['dis_center_' + partial_row_name+"_"+str(frame)].values

        for i, sc in enumerate(scatters):
            y = df[mask][velocities[i] + "_" + partial_row_name+"_"+str(frame)].values
            sc.set_offsets(np.column_stack((x, y)))

        fig.suptitle(f"mass_ratio = {mass_ratio:.2f}", fontsize=16)
        return scatters

    ani = FuncAnimation(fig, update, frames=mass_ratios, init_func=init, blit=False, interval=100)
    plt.tight_layout()
    return HTML(ani.to_jshtml())


def display_mean_squared_velocity(df:pd.DataFrame,velocities:np.ndarray[str],mass_ratios:np.ndarray,partial_row_name:str):
    n=len(velocities)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))  # ajusted size for the number of subplot

    # Si n == 1, axes n'est pas un tableau, donc on le met dans une liste pour la boucle
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mean_square_values=[0 for r in mass_ratios]
        for k in range(0,len(mass_ratios)):
            mean_square_values[k] = mean_square_peculiar_velocity(df,velocities[i]+"_"+partial_row_name+"_"+str(mass_ratios[k]))
        ax.plot(mass_ratios, mean_square_values, color='b', label=r"$v_{r,\text{"+velocities[i]+r"}}$")
        ax.set_xlabel("m1 barre")
        ax.set_ylabel(r"mean square of $v_{r,\text{"+velocities[i]+r"}}$ (km/s)")
        ax.set_title(r"Mean-square of $V_{r,\text{"+velocities[i]+r"}}$ depending on m1_barre")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
def display_velocities_distance_hubble_regression(df,velocities:list[str],row_name:str,border:bool=False,force_origin:bool=True):
    mask = ~df['Name'].str.startswith('CoM_')

    n=len(velocities)
    fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))  # ajusted size for the number of subplot

    # Si n == 1, axes n'est pas un tableau, donc on le met dans une liste pour la boucle
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        mask2 = (
        (df['dis_center_'+row_name] >= 1) &
        (df[velocities[i]+"_"+row_name] >= -100) & (df[velocities[i]+"_"+row_name] <= 350) &
        ~df['Name'].str.startswith('CoM_')
        )
        
        ax.scatter(df[mask]['dis_center_'+row_name], df[mask][velocities[i]+"_"+row_name], color='b', label=r"$v_{r,\text{"+velocities[i]+r"}}$") # Tracé des points
        
        # regression linéaire
        x = pd.to_numeric(df.loc[mask, 'dis_center_'+row_name], errors='coerce').values
        y = pd.to_numeric(df.loc[mask, velocities[i]+"_"+row_name], errors='coerce').values
        # trace de la regression
        if force_origin:
            A = x.reshape(-1, 1)
            a = np.linalg.lstsq(A, y, rcond=None)[0][0]
            b = 0
        else:
            A = np.vstack([x, np.ones(len(x))]).T
            a, b = np.linalg.lstsq(A, y, rcond=None)[0]
        x_vals = np.linspace(x.min(), x.max(), 100)
        y_vals = a * x_vals + b
        ax.plot(x_vals, y_vals, 'r--', label=f"$H_0 = {a:.2f}\\, \\text{{km.s}}^{{-1}}\\text{{.Mpc}}^{{-1}}$")

        
        ax.set_xlabel("Distance from "+row_name+" (Mpc)")
        ax.set_ylabel(r"$v_{r,\text{"+velocities[i]+r"}}$ (km/s)")
        ax.set_title(r"$V_{r,\text{"+velocities[i]+r"}}$ depending on distance from "+row_name)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 5)  # Limites de l'axe x entre 1 et 5
        ax.set_ylim(-100, 500)  # Limites de l'axe y entre -25 et 20

    plt.tight_layout()
    plt.show()
    
