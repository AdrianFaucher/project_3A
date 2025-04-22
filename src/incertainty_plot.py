import matplotlib.pyplot as plt
import pandas as pd

# Load data
galaxy_df = pd.read_csv('../test.csv')

# Filter out center-of-mass entries
mask = ~galaxy_df['Name'].str.startswith('CoM_')
x = pd.to_numeric(galaxy_df.loc[mask, "dis_center_CoM_N5128,Cen A_N5236, M83_0.76"], errors='coerce').values
e_x = pd.to_numeric(galaxy_df.loc[mask, "e_dis_center_CoM_N5128,Cen A_N5236, M83_0.76"], errors='coerce').values
y = pd.to_numeric(galaxy_df.loc[mask, "minor_infall_velocity_CoM_N5128,Cen A_N5236, M83_0.76"], errors='coerce').values

# Create a figure with two subplots (1 row, 2 columns)
fig, axs = plt.subplots(1, 2, figsize=(12, 5))

# First subplot: Errorbar plot
axs[0].errorbar(x, y, xerr=e_x, fmt='o', capsize=5, ecolor='red', label="Données avec erreurs")
axs[0].scatter(x, y)
axs[0].set_xlabel("distance from CoM")
axs[0].set_ylabel("Minor infall velocity")
axs[0].legend()
axs[0].grid(True)

# Second subplot: Scatter plot
axs[1].scatter(x, y)
axs[1].set_xlabel("distance from CoM")
axs[1].set_ylabel("Minor infall velocity")

axs[1].grid(True)

# Display the plots
plt.tight_layout()
plt.show()
