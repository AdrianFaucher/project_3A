import numpy as np
import matplotlib.pyplot as plt


# Data for Cen A-M83 system including separate minor and major "This Work"
cen_a_m83_data = [
    {"x": 6.4 + 0.8, "y": 1, "xerr": 1.8 + 0.9, "label": "Karachentsev et al. (2006)", "method": "Virial Mass"},
    {"x": 2.1, "y": 2, "xerr": 0.5, "label": "Peirani et al. (2008)", "method": "Hubble Flow"},
    {"x": 2.81, "y": 3, "xerr": 0.5, "label": "Del Popolo et al. (2022)", "method": "Hubble Flow"},
    {"x": 8.9, "y": 4, "xerr": [[1.89], [1.94]], "label": "Muller (2024)", "method": "Virial Theorem"},
    {"x": 2.54, "y": 5, "xerr": [[1.15], [1.75]], "label": "This Work Minor", "method": "Virial Theorem"},
    {"x": 9.68, "y": 6, "xerr": [[4.63], [7.90]], "label": "This Work Major", "method": "Virial Theorem"},
    {"x": 1.73, "y": 7, "xerr": [[0.64], [0.80]], "label": "This Work Minor", "method": "Hubble flow"},
    {"x": 2.97, "y": 8, "xerr": [[1.20], [1.58]], "label": "This Work Major", "method": "Hubble flow"}
]

# Plot only Cen A-M83 data
fig, ax = plt.subplots(figsize=(10, 5))

ax.set_ylim([0, 9])
ax.set_xlim([0, 18])
ax.set_xlabel(r'$M \ [10^{12}\, M_{\odot}]$', fontsize=20)
ax.yaxis.set_visible(False)

for data in cen_a_m83_data:
    if 'Minor' in data['label'] and "Virial" in data['method']:
        color = 'blue'
        use_dashed = False
    elif 'Major' in data['label'] and "Virial" in data['method']:
        color = 'red'
        use_dashed = False
    elif 'Minor' in data['label'] and "Hubble" in data['method']:
        color = 'blue'
        use_dashed = True
    elif 'Major' in data['label'] and "Hubble" in data['method']:
        color = 'red'
        use_dashed = True
    else:
        color = 'black'
        use_dashed = False

    if use_dashed:
        (_, caps, bars)=ax.errorbar(data["x"], data["y"], xerr=data["xerr"], fmt='o', color=color, linewidth=2, capsize=6)
        for bar in bars:
            bar.set_linestyle('--')   # style pointillé
    else:
        ax.errorbar(data["x"], data["y"], xerr=data["xerr"], fmt='o', color=color, linewidth=2, capsize=6)
    
    ax.text(data["x"] + 0.3, data["y"], f"{data['label']}\n({data['method']})",
            ha='left', va='center', fontsize=14, wrap=True)

ax.text(15, 8, "CenA-M83", fontsize=20, weight='bold', ha='center')

plt.tight_layout()
plt.savefig('../plot/mass_plot.png', dpi=300, bbox_inches='tight')
plt.show()