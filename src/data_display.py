import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from src.tools import mean_square_peculiar_velocity, identify_abnormal_galaxies_per_velocity, mean_square_peculiar_velocity_consistent

def display_hubble_MW(df:pd.DataFrame)->None:
    """Display the velocities of galaxies depending on the distance from the milky way
    and calculate H_0

    Args:
        df (pd.DataFrame): DataFrame containing the galaxies
    """
    mask = ~df['Name'].str.startswith('CoM_')
    # mask = ((df['f_Dis']== "h") & ~df['Name'].str.startswith('CoM_'))
    x_series = pd.to_numeric(df.loc[mask, 'Dis'], errors='coerce')
    y_series = pd.to_numeric(df.loc[mask, 'VLG'], errors='coerce')
    
    plt.figure(figsize=(8, 5))
    # errors 
    e_x_series = pd.to_numeric(df.loc[mask, 'e_Dis'], errors='coerce')
    e_y_series = pd.to_numeric(df.loc[mask, 'e_VLG'], errors='coerce')
    plt.errorbar(x_series, y_series, xerr=e_x_series, yerr=e_y_series, fmt='none', ecolor='black', elinewidth=0.8, capsize=3, marker='x', markersize=8)

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
    plt.xlabel("Distance from LG CoM (Mpc)")
    plt.ylabel(r"Radial velocity (km/s)")
    plt.title("Radial velocity depending on distance from LG CoM")
    plt.legend()
    plt.grid(True)
    plt.show()
    
def display_velocities_distance(df,velocities:list[str],row_name:str,border:bool=False,):
    mask = ~df['Name'].str.startswith('CoM_')
    n=len(velocities)
    fig, axes = plt.subplots(n, 1, figsize=(12, 10*n))  # ajusted size for the number of subplot
    affichage=['o','v']
    # Si n == 1, axes n'est pas un tableau, donc on le met dans une liste pour la boucle
    if n == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        x = pd.to_numeric(df.loc[mask, 'dis_center_'+row_name], errors='coerce').values
        y = pd.to_numeric(df.loc[mask, velocities[i]+"_"+row_name], errors='coerce').values


        col_e_x_max = 'e_max_dis_center_' + row_name
        col_e_x_min = 'e_min_dis_center_' + row_name
        col_e_y_max = 'e_max_'+velocities[i]+"_"+ row_name
        col_e_y_min = 'e_min_'+velocities[i]+"_"+ row_name
        # col_e_y_max  = 'e_analytic_'+ velocities[i]+"_"+ row_name
        # col_e_y_min  = 'e_analytic_'+ velocities[i]+"_"+ row_name
        if col_e_y_max in df.columns and col_e_y_min in df.columns and col_e_x_min in df.columns and col_e_x_max in df.columns:
            e_x_min = pd.to_numeric(df.loc[mask, col_e_x_min], errors='coerce').values
            e_x_max = pd.to_numeric(df.loc[mask, col_e_x_max], errors='coerce').values
            e_y_min = pd.to_numeric(df.loc[mask, col_e_y_min], errors='coerce').values
            e_y_max = pd.to_numeric(df.loc[mask, col_e_y_max], errors='coerce').values
            ax.errorbar(x, y, xerr=[e_x_min,e_x_max],yerr=[e_y_min,e_y_max],color="black",markersize=5, fmt='o', capsize=5, ecolor='red')#, label=r"$v_{r,\text{"+velocities[i]+r"}}$ with exact error")

        else:
            ax.scatter(x, y, color='b', label=r"$v_{r,\text{"+velocities[i]+r"}}$") # Tracé des points
        # ax.set_xlabel("Distance from "+row_name+" (Mpc)")
        # ax.set_ylabel(r"$v_{r,\text{"+velocities[i]+r"}}$ (km/s)")
        ax.set_xlabel("Distance from CoM (Mpc)")
        ax.set_ylabel(r"$v_{r}$ (km/s)")
        # ax.set_title(r"$V_{r,\text{"+velocities[i]+r"}}$ depending on distance from "+row_name)
        ax.legend()
        ax.grid(True)
        ax.set_xlim(0, 5)  # Limites de l'axe x entre 1 et 5
        ax.set_ylim(-100, 300)  # Limites de l'axe y entre -25 et 20

    plt.tight_layout()
    plt.show()
    
def display_velocities_distance_color(df, velocities: list[str], row_name: str, border: bool = False):
    mask = ~df['Name'].str.startswith('CoM_')
    plt.figure(figsize=(12, 6))

    colors = plt.cm.get_cmap('tab10', len(velocities))  # Palette de couleurs
    markers=['o','v','s']
    for i, velocity in enumerate(velocities):
        plt.scatter(
            df[mask]['dis_center_' + row_name],
            df[mask][velocity + "_" + row_name],
            color=colors(i),
            label=r"$v_{r,\text{" + velocity + r"}}$",
            marker=markers[i%3]
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
            mean_square_values[k] = np.sqrt(mean_square_peculiar_velocity(df,velocities[i]+"_"+partial_row_name+"_"+str(mass_ratios[k])))
        ax.plot(mass_ratios, mean_square_values, color='b', label=r"$v_{r,\text{"+velocities[i]+r"}}$")
        ax.set_xlabel("m1 barre")
        ax.set_ylabel(r"mean square of $v_{r,\text{"+velocities[i]+r"}}$ (km/s)")
        ax.set_title(r"Mean-square of $V_{r,\text{"+velocities[i]+r"}}$ depending on m1_barre")
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    




def display_mean_squared_velocity_consistent(df, velocities, mass_ratios, partial_row_name, 
                                           lower_bound=-100, upper_bound=500, combined_plot=False):
    n = len(velocities)
    
    # Dictionnaire pour stocker les valeurs minimales
    min_values = {}
    
    # Couleurs pour les différentes vitesses
    colors = ['b', 'r', 'g', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    if combined_plot:
        # Un seul graphique pour toutes les vitesses
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        for i, velocity in enumerate(velocities):
            # Pour chaque type de vitesse, identifier les galaxies anormales
            abnormal_galaxies = identify_abnormal_galaxies_per_velocity(df, velocity, mass_ratios, partial_row_name, lower_bound, upper_bound)
            print(f"Nombre de galaxies exclues pour {velocity}: {len(abnormal_galaxies)}")
            
            mean_square_values = []
            for mass_ratio in mass_ratios:
                column = f"{velocity}_{partial_row_name}_{mass_ratio}"
                mean_square = mean_square_peculiar_velocity_consistent(df, column, abnormal_galaxies)
                mean_square_values.append(np.sqrt(mean_square))
            
            # Trouver le ratio de masse qui donne la valeur minimale
            min_index = np.argmin(mean_square_values)
            min_mass_ratio = mass_ratios[min_index]
            min_value = mean_square_values[min_index]
            
            # Stocker dans le dictionnaire
            min_values[velocity] = {
                "mass_ratio": min_mass_ratio,
                "mean_square_value": min_value
            }
            
            # Tracer la courbe et marquer le minimum
            color = colors[i % len(colors)]
            ax.plot(mass_ratios, mean_square_values, color=color, 
                   label=r"$v_{r,\text{"+velocity+r"}}$", linewidth=2)
            ax.plot(min_mass_ratio, min_value, 'o', color=color, markersize=8, 
                   markeredgecolor='black', markeredgewidth=1)
        
        ax.set_xlabel("m1 barre", fontsize=16)
        ax.set_ylabel("Mean square velocity (km/s)", fontsize=16)
        ax.set_title("Mean-square velocities depending on m1_barre", fontsize=18)
        ax.legend(fontsize=14)
        ax.grid(True, alpha=0.3)
        
    else:
        # Graphiques séparés (comportement original)
        fig, axes = plt.subplots(n, 1, figsize=(12, 5*n))
        
        # Si n == 1, axes n'est pas un tableau, donc on le met dans une liste pour la boucle
        if n == 1:
            axes = [axes]

        for i, ax in enumerate(axes):
            # Pour chaque type de vitesse, identifier les galaxies anormales
            abnormal_galaxies = identify_abnormal_galaxies_per_velocity(df, velocities[i], mass_ratios, partial_row_name, lower_bound, upper_bound)
            print(f"Nombre de galaxies exclues pour {velocities[i]}: {len(abnormal_galaxies)}")
            
            mean_square_values = []
            for mass_ratio in mass_ratios:
                column = f"{velocities[i]}_{partial_row_name}_{mass_ratio}"
                mean_square = mean_square_peculiar_velocity_consistent(df, column, abnormal_galaxies)
                mean_square_values.append(np.sqrt(mean_square))
            
            # Trouver le ratio de masse qui donne la valeur minimale
            min_index = np.argmin(mean_square_values)
            min_mass_ratio = mass_ratios[min_index]
            min_value = mean_square_values[min_index]
            
            # Stocker dans le dictionnaire
            min_values[velocities[i]] = {
                "mass_ratio": min_mass_ratio,
                "mean_square_value": min_value
            }
            
            # Marquer le minimum sur le graphique
            ax.plot(mass_ratios, mean_square_values, color='b', label=r"$v_{r,\text{"+velocities[i]+r"}}$")
            ax.plot(min_mass_ratio, min_value, 'ro', markersize=8, label=f"Min: {min_mass_ratio:.2f}")
            
            ax.set_xlabel("m1 barre")
            ax.set_ylabel(r"mean square of $v_{r,\text{"+velocities[i]+r"}}$ (km/s)",fontsize=20)
            ax.set_title(r"Mean-square of $V_{r,\text{"+velocities[i]+r"}}$ depending on m1_barre",fontsize=20)
            ax.legend(fontsize=20)
            ax.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Afficher un résumé des valeurs minimales
    print("\nRésumé des valeurs minimales pour chaque vitesse:")
    for velocity, info in min_values.items():
        print(f"Vitesse {velocity}: Ratio de masse optimal = {info['mass_ratio']:.4f}, Valeur minimale = {info['mean_square_value']:.2f} km/s")
    
    return min_values
    
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
        x = pd.to_numeric(df.loc[mask2, 'dis_center_'+row_name], errors='coerce').values
        y = pd.to_numeric(df.loc[mask2, velocities[i]+"_"+row_name], errors='coerce').values
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
    
