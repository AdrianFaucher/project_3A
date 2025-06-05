import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Load data
galaxy_df = pd.read_csv('../test.csv')

# Filter out center-of-mass entries
mask = ~galaxy_df['Name'].str.startswith('CoM_')
x = pd.to_numeric(galaxy_df.loc[mask, "dis_center_CoM_N5128,Cen A_N5236, M83_0.76"], errors='coerce').values
e_x = pd.to_numeric(galaxy_df.loc[mask, "e_dis_center_CoM_N5128,Cen A_N5236, M83_0.76"], errors='coerce').values
y = pd.to_numeric(galaxy_df.loc[mask, "minor_infall_velocity_CoM_N5128,Cen A_N5236, M83_0.76"], errors='coerce').values

# Create a figure with two subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# Préparer les barres d'erreur pour qu'elles ne dépassent pas en territoire négatif
e_x_lower = np.minimum(e_x, x)  # Assurez-vous que l'erreur ne dépasse pas la valeur de x
e_x_upper = e_x  # Erreur normale vers la droite

# Premier subplot: Graphique avec barres d'erreur asymétriques
axs[0].errorbar(x, y, xerr=[e_x_lower, e_x_upper], fmt='o', capsize=5, ecolor='red', label="Données avec erreurs limitées")
axs[0].scatter(x, y, zorder=3)  # zorder pour que les points soient au-dessus des barres d'erreur
axs[0].set_xlabel("Distance from CoM")
axs[0].set_ylabel("Minor infall velocity")
axs[0].legend()
axs[0].grid(True)
axs[0].set_xlim(left=0)  # Fixer la limite gauche à 0 pour ne pas montrer la partie négative

# Deuxième subplot: Graphique scatter simple
axs[1].scatter(x, y)
axs[1].set_xlabel("Distance from CoM")
axs[1].set_ylabel("Minor infall velocity")
axs[1].grid(True)
axs[1].set_xlim(left=0)  # Cohérence avec le premier graphique

# Afficher les graphiques
plt.tight_layout()
plt.show()
