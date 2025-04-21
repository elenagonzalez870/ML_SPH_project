import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import io
from matplotlib.lines import Line2D

# Read and process summary.txt
with open('summary.txt', 'r') as file:
    lines = file.readlines()
# Filter out lines that are just dashes
lines = [line for line in lines if not line.strip().startswith('-')]
# Convert the filtered lines to a single string and use genfromtxt
data = np.genfromtxt(io.StringIO(''.join(lines)), skip_header=52, names=True, dtype=None, encoding=None)
# Check the fields
print(data.dtype.names)
print(data[:5])  # Show the first 5 rows

R_sun = 7e8
M_sun = 2e30
G = 6.67e-11
vunit = (G*M_sun/R_sun)**0.5 * 1e-3
print('velocity unit=',vunit,'km/s')

coll_id = data['Coll_id']
mini1 = data['Mini1']
mini2 = data['Mini2']
vrel_inf = data['Vrel_inf'] * vunit
imp_param = data['ImpParam']
mfin1 = data['Mfin1']
mfin2 = data['Mfin2']
n_star = data['Nstar']
d_m = data['dM']
d_e = data['dE']
d_l = data['dL']
d_theta = data['dTheta']

# Look for places in data file where final star indices should be swapped
for i in range(len(coll_id)):
    if mfin1[i] > mfin2[i]:
        print(f"Will swap Mfin1 and Mfin2 for Coll_id: {coll_id[i]}, Mini1: {mini1[i]}, Mini2: {mini2[i]}, Mfin1: {mfin1[i]}, M2: {mfin2[i]}")
        mfin1[i], mfin2[i] = mfin2[i], mfin1[i]
        #d_theta=-d_theta

#################################################################
# Read and process stars.txt
with open('stars.txt', 'r') as file:
    lines = file.readlines()
# Filter out lines that are just dashes
lines = [line for line in lines if not line.strip().startswith('-')]
# Convert the filtered lines to a single string and use genfromtxt
data = np.genfromtxt(io.StringIO(''.join(lines)), skip_header=24, names=True, dtype=None, encoding=None)
# Check the fields
print(data.dtype.names)
print(data[:5])  # Show the first 5 rows

mass = data['Mass']
radius = data['Radius']

# Create a dictionary mapping each mass value to its radius
mass_to_radius = dict(zip(mass, radius))

# Get corresponding radius values for each mass
rini1 = np.array([mass_to_radius.get(m) for m in mini1])
rini2 = np.array([mass_to_radius.get(m) for m in mini2])

normalized_imp_param = imp_param/(rini1+rini2)

#################################################################
# Make some plots and print some things out

print('The number of cases to be plotted is',len(mini1))
# scatter plot
plt.scatter(vrel_inf[n_star == 2], normalized_imp_param[n_star == 2], alpha=0.5, c=d_m[n_star == 2],marker='s', vmin=0, vmax=1)
plt.scatter(vrel_inf[n_star == 1], normalized_imp_param[n_star == 1], alpha=0.5, c=d_m[n_star == 1],marker='^', vmin=0, vmax=1)
plt.scatter(vrel_inf[n_star == 0], normalized_imp_param[n_star == 0], alpha=0.5, c=d_m[n_star == 0],marker='o', vmin=0, vmax=1)
cbar = plt.colorbar()
cbar.set_label(r'$\Delta M/(M_1+M_2)$')
plt.xlabel(r'$v_\infty$ [km/s]')
plt.ylabel(r'$b/(R_1+R_2)$')
plt.title('Circle = 0 stars, Triangle = 1 star, Square = 2 stars')
plt.grid(True)
plt.xlim(-100, 7000)
plt.ylim(-0.02, 1.1)
plt.show()

## scatter plot with unique colors for n_star and showing all data
#cmap = mcolors.ListedColormap(['red', 'blue', 'green'])
#bounds = [-0.5, 0.5, 1.5, 2.5]  # Define boundaries for n_star categories
#norm = mcolors.BoundaryNorm(bounds, cmap.N)
#scatter = plt.scatter(vrel_inf, normalized_imp_param, c=n_star, cmap=cmap, norm=norm, alpha=0.5)
## Create a legend manually
#legend_labels = ['n_star = 0', 'n_star = 1', 'n_star = 2']
#legend_colors = [cmap(norm(i)) for i in [0, 1, 2]]
#legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8) for color in legend_colors]
#plt.legend(legend_handles, legend_labels, loc='best')
#plt.xlabel(r'$v_\infty$ [km/s]')
#plt.ylabel(r'$b/(R_1+R_2)$')
#plt.grid(True)
#plt.show()

# Printing minimum and maximum values
print(f"Minimum value in vrel_inf: {np.min(vrel_inf)} km/s")
print(f"Maximum value in vrel_inf: {np.max(vrel_inf)} km/s")

# Find indices of a few example curious cases with large impact parameters
indices = np.where((normalized_imp_param > 20) & (normalized_imp_param < 90))
for idx in indices[0]:
    print(f"Tidal capture? Case number: {coll_id[idx]}, Normalized Impact Parameter: {normalized_imp_param[idx]}, vrel_inf: {vrel_inf[idx]}, n_stars: {n_star[idx]}")

print(f"Minimum value of initial mass: {np.min(mini1)} Msun")
print(f"Maximum value of initial mass: {np.max(mini2)} Msun")
print(f"Minimum value of final mass: {np.min(mfin1)} Msun")
print(f"Maximum value of final mass: {np.max(mfin2)} Msun")

# 3D scatter plot:
cmap = mcolors.ListedColormap(['red', 'green', 'blue'])  # Red for 0, Green for 1, Blue for 2
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
# Apply the colormap to the scatter plot, using n_star as the color array
sc = ax.scatter(vrel_inf, normalized_imp_param, mini1, alpha=0.5, c=n_star, marker='o', cmap=cmap)
ax.set_xlabel(r'$v_\infty$ [km/s]')
ax.set_ylabel(r'$b/(R_1+R_2)$')
ax.set_zlabel(r'$M_{1,i}$ [$M_\odot$]')
legend_entries = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='red', label='0 stars'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='green', label='1 star'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', label='2 stars')
]
ax.legend(handles=legend_entries)
plt.show()

# scatter plot for mini1 == mini2 == 1
plt.scatter(vrel_inf[(mini1 == 1) & (mini2 == 1) & (n_star == 2)], normalized_imp_param[(mini1 == 1) & (mini2 == 1) & (n_star == 2)], alpha=0.5, c=mfin2[(mini1 == 1) & (mini2 == 1) & (n_star == 2)], marker='s', vmin=0, vmax=2)
plt.scatter(vrel_inf[(mini1 == 1) & (mini2 == 1) & (n_star == 1)], normalized_imp_param[(mini1 == 1) & (mini2 == 1) & (n_star == 1)], alpha=0.5, c=mfin2[(mini1 == 1) & (mini2 == 1) & (n_star == 1)], marker='^', vmin=0, vmax=2)
plt.scatter(vrel_inf[(mini1 == 1) & (mini2 == 1) & (n_star == 0)], normalized_imp_param[(mini1 == 1) & (mini2 == 1) & (n_star == 0)], alpha=0.5, c=mfin2[(mini1 == 1) & (mini2 == 1) & (n_star == 0)], marker='o', vmin=0, vmax=2)
cbar = plt.colorbar()
cbar.set_label(r'$M_{2,f}$ [$M_\odot$]')
plt.xlabel(r'$v_\infty$ [km/s]')
plt.ylabel(r'$b/(R_1+R_2)$')
plt.title(r'Circle = 0 stars, Triangle = 1 star, Square = 2 stars (for $M_1=M_2=1 M_\odot$)', fontsize=10)
plt.grid(True)
plt.xlim(-100, 7000)
plt.ylim(-0.02, 1.1)
plt.show()


color_data = d_theta[(mini1 == 1) & (mini2 == 1)]
# Find the min and max of the color data
vmin = color_data.min()
vmax = color_data.max()
print("vmin:", vmin)
print("vmax:", vmax)

## scatter plot #2 for mini1 == mini2 == 1
#plt.scatter(vrel_inf[(mini1 == 1) & (mini2 == 1) & (n_star == 2)], normalized_imp_param[(mini1 == 1) & (mini2 == 1) & (n_star == 2)], alpha=0.5, c=d_theta[(mini1 == 1) & (mini2 == 1) & (n_star == 2)], marker='s', vmin=vmin, vmax=vmax)
##plt.scatter(vrel_inf[(mini1 == 1) & (mini2 == 1) & (n_star == 1)], normalized_imp_param[(mini1 == 1) & (mini2 == 1) & (n_star == 1)], alpha=0.5, c=d_theta[(mini1 == 1) & (mini2 == 1) & (n_star == 1)], marker='^', vmin=vmin, vmax=vmax)
##plt.scatter(vrel_inf[(mini1 == 1) & (mini2 == 1) & (n_star == 0)], normalized_imp_param[(mini1 == 1) & (mini2 == 1) & (n_star == 0)], alpha=0.5, c=d_theta[(mini1 == 1) & (mini2 == 1) & (n_star == 0)], marker='o', vmin=vmin, vmax=vmax)
#cbar = plt.colorbar()
#cbar.set_label(r'$\Delta \theta$')
#plt.xlabel(r'$v_\infty$ [km/s]')
#plt.ylabel(r'$b/(R_1+R_2)$')
#plt.title(r'Circle = 0 stars, Triangle = 1 star, Square = 2 stars (for $M_1=M_2=1 M_\odot$)', fontsize=10)
#plt.grid(True)
#plt.xlim(-100, 7000)
#plt.ylim(-0.02, 1.1)
#plt.show()
