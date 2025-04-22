import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import equatorial_to_cartesian

# Load data
galaxy_df = pd.read_csv('../data/test.csv')

# Filter out center-of-mass entries
mask = ~galaxy_df['Name'].str.startswith('CoM_')

# Convert necessary columns to numeric
r = pd.to_numeric(galaxy_df.loc[mask, "Dis"], errors='coerce').values
ra = pd.to_numeric(galaxy_df.loc[mask, "RA_radians"], errors='coerce').values
dec = pd.to_numeric(galaxy_df.loc[mask, "Dec_radians"], errors='coerce').values
vlg = pd.to_numeric(galaxy_df.loc[mask, "VLG"], errors='coerce').values
names = galaxy_df.loc[mask, "Name"].values

# Convert to Cartesian coordinates
coord = equatorial_to_cartesian(r, ra, dec)
x, y, z = coord[0], coord[1], coord[2]

# Create 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

# Scatter plot with color mapping to VLG
sc = ax.scatter(x, y, z, c=vlg, cmap='inferno', s=7,alpha=0.5)  # 's' controls point size

# Highlight M83 and Cen A with larger triangles, using the same color scale
highlight_mask = np.array(["Cen A" in name or "M83" in name for name in names])
ax.scatter(x[highlight_mask], y[highlight_mask], z[highlight_mask], c=vlg[highlight_mask], cmap='inferno', marker='^', s=50, vmin=vlg.min(), vmax=vlg.max(),alpha=1)

# Mark the origin
ax.scatter(0, 0, 0, color='black', marker="s", s=20)
ax.text(0, 0, 0, "LG CoM", fontsize=7)

# Add labels for M83 and Cen A
for xi, yi, zi, name in zip(x, y, z, names):
    if "Cen A" in name or "M83" in name:
        ax.text(xi, yi, zi, name, fontsize=8, color='black')

# Axis labels
ax.set_xlabel('X (Mpc)')
ax.set_ylabel('Y (Mpc)')
ax.set_zlabel('Z (Mpc)')

# Add colorbar
cbar = plt.colorbar(sc, ax=ax, pad=0.1)
cbar.set_label('VLG (km/s)')  # adjust label if needed

plt.show()
