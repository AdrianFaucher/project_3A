import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tools import equatorial_to_cartesian

# Load data
galaxy_df = pd.read_csv('../data/new_data.csv')
galaxy_df['Name'] = galaxy_df['Name'].astype(str)
# Filter out center-of-mass entries
mask = ~galaxy_df['Name'].str.startswith('CoM_') & (galaxy_df['Name'])

# Convert necessary columns to numeric
r = pd.to_numeric(galaxy_df.loc[mask, "Dis"], errors='coerce').values
ra = pd.to_numeric(galaxy_df.loc[mask, "RA"], errors='coerce').values*(np.pi/180)
dec = pd.to_numeric(galaxy_df.loc[mask, "Dec"], errors='coerce').values*(np.pi/180)
V_h = pd.to_numeric(galaxy_df.loc[mask, "V_h"], errors='coerce').values
names = galaxy_df.loc[mask, "Name"].values

# Convert to Cartesian coordinates
coord = equatorial_to_cartesian(r, ra, dec)
x, y, z = coord[0], coord[1], coord[2]

# Create the highlight mask for M83 and Cen A
highlight_mask = np.array(["CenA" in name or "M83" in name for name in names])

# Find indices of Cen A and M83
cen_a_idx = next((i for i, name in enumerate(names) if "CenA" in name), None)
m83_idx = next((i for i, name in enumerate(names) if "M83" in name), None)

if cen_a_idx is not None and m83_idx is not None:
    # Points for the plane: origin (0,0,0), Cen A, and M83
    p1 = np.array([0, 0, 0])  # Origin (Local Group Center of Mass)
    p2 = np.array([x[cen_a_idx], y[cen_a_idx], z[cen_a_idx]])  # Cen A
    p3 = np.array([x[m83_idx], y[m83_idx], z[m83_idx]])  # M83
    
    # Calculate normal vector to the plane
    v1 = p2 - p1
    v2 = p3 - p1
    normal = np.cross(v1, v2)
    normal = normal / np.linalg.norm(normal)  # Normalize
    
    # Create basis vectors for the plane
    e1 = v1 / np.linalg.norm(v1)  # First basis vector
    e2 = np.cross(normal, e1)  # Second basis vector (orthogonal to normal and e1)
    e2 = e2 / np.linalg.norm(e2)  # Normalize
    
    # Project all points onto the plane
    proj_x = []
    proj_y = []
    
    for i in range(len(x)):
        point = np.array([x[i], y[i], z[i]])
        # Project the point onto the basis vectors of the plane
        px = np.dot(point - p1, e1)
        py = np.dot(point - p1, e2)
        proj_x.append(px)
        proj_y.append(py)
    
    proj_x = np.array(proj_x)
    proj_y = np.array(proj_y)
    
    # FIRST PLOT - 3D Plot
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(projection='3d')
    
    # Scatter plot with color mapping to V_h (3D)
    sc1 = ax1.scatter(x, y, z, c=V_h, cmap='inferno', s=7, alpha=0.5)
    
    # Highlight M83 and Cen A with larger triangles (3D)
    ax1.scatter(x[highlight_mask], y[highlight_mask], z[highlight_mask], 
                c=V_h[highlight_mask], cmap='inferno', marker='*', s=50, 
                vmin=V_h.min(), vmax=V_h.max(), alpha=1)
    
    # Mark the origin (3D)
    ax1.scatter(0, 0, 0, color='black', marker="s", s=20)
    ax1.text(0, 0, 0, "LG CoM", fontsize=7)
    
    # Add labels for M83 and Cen A (3D)
    for xi, yi, zi, name in zip(x, y, z, names):
        if  "M83" in name:
            ax1.text(xi, yi, zi, "M83", fontsize=8, color='black')
        elif "CenA" in name:
            ax1.text(xi, yi, zi, "CenA", fontsize=8, color='black')
    
    # Axis labels (3D)
    ax1.set_xlabel('X (Mpc)')
    ax1.set_ylabel('Y (Mpc)')
    ax1.set_zlabel('Z (Mpc)')
    ax1.set_title('3D Distribution of Galaxies')
    
    # Add colorbar for 3D plot
    cbar1 = plt.colorbar(sc1, ax=ax1, pad=0.1)
    cbar1.set_label('V_h (km/s)')
    
    plt.tight_layout()
    plt.show()
    
    # SECOND PLOT - 2D Projection Plot
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111)
    
    # Scatter plot with color mapping to V_h (2D)
    sc2 = ax2.scatter(proj_x, proj_y, c=V_h, cmap='inferno', s=1, alpha=1)
    
    # Highlight M83 and Cen A with larger triangles (2D)
    ax2.scatter(proj_x[highlight_mask], proj_y[highlight_mask], 
                c=V_h[highlight_mask], cmap='inferno', marker='*', s=100, 
                vmin=V_h.min(), vmax=V_h.max(), alpha=1)
    
    # Mark the origin (2D) - This is the projection of LG CoM which will be at (0,0)
    ax2.scatter(0, 0, color='black', marker="s", s=20)
    ax2.text(0, 0, "LG CoM", fontsize=7)
    
    # Add labels for M83 and Cen A (2D)
    for px, py, name in zip(proj_x, proj_y, names):
        if "M83" in name:
            ax2.text(px, py, "M83", fontsize=12, color='black')
        if "CenA" in name:
            ax2.text(px, py, "CenA", fontsize=12, color='black')    
            
    # Axis labels (2D)
    ax2.set_xlabel('Projection on first basis vector (Mpc)')
    ax2.set_ylabel('Projection on second basis vector (Mpc)')
    ax2.set_title('2D Projection onto Plane Defined by LG CoM, Cen A, and M83')
    ax2.grid(True, alpha=0.3)
    
    # Add colorbar for 2D plot
    cbar2 = plt.colorbar(sc2, ax=ax2, pad=0.1)
    cbar2.set_label('V_h (km/s)')
    
    plt.tight_layout()
    plt.show()
else:
    print("Error: Could not find Cen A or M83 in the dataset.")