import MDAnalysis as mda
import numpy as np

# Load the LAMMPS trajectory with explicit format
universe = mda.Universe('NVT_1200.data', '1200_NVT.lammpstrj', format='LAMMPSDUMP')

radii = {1: 4.66, 2: 3.66} 
grid_spacing = 0.3 

def remap_positions(positions, box):
    remapped_pos = np.mod(positions, box)
    return remapped_pos

def calculate_free_space(universe, radii, grid_spacing):
    box = universe.dimensions[:3]
    grid_points = np.mgrid[0:box[0]:grid_spacing, 0:box[1]:grid_spacing, 0:box[2]:grid_spacing].reshape(3, -1).T
    total_volume = np.prod(box) 
    free_volumes = []
    
    for ts in universe.trajectory:
        occupied = np.zeros(len(grid_points), dtype=bool)
        for atom in universe.atoms:
            atom_type = atom.type
            if isinstance(atom_type, str):
                atom_type = int(atom_type)  
            atom_radius = radii[atom_type]
            atom_pos_remap = remap_positions(atom.position, box)
            distances = np.linalg.norm(grid_points - atom_pos_remap, axis=1)
            occupied[distances < atom_radius] = True
        
        free_volume = grid_spacing**3 * (len(grid_points) - np.sum(occupied))
        free_volumes.append(free_volume)
        percent_free_space = (np.mean(free_volumes) / total_volume) * 100

    return np.mean(free_volumes), np.std(free_volumes), free_volumes, percent_free_space

average_free_space, std_free_space, free_volumes, percent_free_space = calculate_free_space(universe, radii, grid_spacing)

with open("kcl_RIM_1200.out", "w") as file:
    file.write(f"Average estimated free space (A^3): {average_free_space}\n")
    file.write(f"Standard deviation (A^3): {std_free_space}\n")
    file.write(f"Percent of free space: {percent_free_space:.4f}%\n")
    file.write(f"Free volumes over all time steps: {free_volumes}\n")
