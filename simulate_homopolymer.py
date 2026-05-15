# Importing general packages
from os import path, makedirs
import json
import sys
from time import perf_counter

# Importing packages for computation
import numpy as np

# Importing simulation packages
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter


# To display durations in seconds in a more readable format
def readable_duration(duration):
    remaining = int(duration)
    if remaining >= 3600:
        hours = remaining // 3600
        remaining = remaining % 3600
    else:
        hours = 0
    if remaining >= 60:
        minutes = remaining // 60
        remaining = remaining % 60
    else:
        minutes = 0
    return f"{hours} hours, {minutes} minutes and {remaining} seconds"


############### Parameter handling ###############

args = sys.argv
params = json.load(open(args[1], mode="r"))
default_params = json.load(open(f"/lus/home/CT3/c1916693/gganesh/repositories/polychrom-sim-launchers/resources/simulation/parameters/default_params_homopolymer.json"))

try:
    polymer_length = params["polymer_length"]
except KeyError:
    raise KeyError("The length of the polymer `polymer_length` has to be defined")
    
try:
    trajectory_length = params["trajectory_length"]
except KeyError:
    raise KeyError("The length of the simulation trajectory `trajectory_length` has to be defined")

try:
    length_scale = params["length_scale"]
except KeyError:
    length_scale = default_params["length_scale"]

try:
    integrator = params["integrator"]
    assert integrator in ["brownian", "langevin", "verlet", "variableLangevin", "variableVerlet"]
except KeyError:
    integrator = default_params["integrator"]

if "variable" not in integrator:
    try:
        timestep = params["timestep"]
    except KeyError:
        timestep = default_params["timestep"]
else:
    timestep = False
    
try:
    friction_coefficient = params["friction_coefficient"]
except KeyError:
    friction_coefficient = default_params["friction_coefficient"]

try:
    moldyn_steps = params["moldyn_steps"]
except KeyError:
    moldyn_steps = default_params["moldyn_steps"]

try:
    harmonic_bond_length = params["harmonic_bond_length"]
except KeyError:
    harmonic_bond_length = default_params["harmonic_bond_length"]

try:
    harmonic_bond_wiggle_dist = params["harmonic_bond_wiggle_dist"]
except KeyError:
    harmonic_bond_wiggle_dist = default_params["harmonic_bond_wiggle_dist"] * harmonic_bond_length

try:
    polymer_stiffness = params["polymer_stiffness"]
except KeyError:
    polymer_stiffness = default_params["polymer_stiffness"]

try:
    repulsion_energy_radius = params["repulsion_energy_radius"]
except KeyError:
    repulsion_energy_radius = default_params["repulsion_energy_radius"] * harmonic_bond_length

try:
    repulsion_energy = params["repulsion_energy"]
except KeyError:
    repulsion_energy = default_params["repulsion_energy"]

try:
    PBC = params["PBC"]
except KeyError:
    PBC = default_params["PBC"]

if PBC:
    try:
        density = params["density"]
    except KeyError:
        density = default_params["density"]
    simbox_length = (polymer_length / density) ** 0.333
    PBC_box = [simbox_length, simbox_length, simbox_length]
else:
    PBC_box = False
    simbox_length = (polymer_length / 0.0001) ** 0.333


params_final = {"polymer_length": polymer_length,
                "trajectory_length": trajectory_length,
                "length_scale": length_scale,
                "integrator": integrator,
                "friction_coefficient": friction_coefficient,
                "timestep": timestep,
                "moldyn_steps": moldyn_steps,
                "PBC": PBC,
                "simbox_length": simbox_length,
                "harmonic_bond_length": harmonic_bond_length,
                "harmonic_bond_wiggle_dist": harmonic_bond_wiggle_dist,
                "polymer_stiffness": polymer_stiffness,
                "repulsion_energy": repulsion_energy,
                "repulsion_energy_radius": repulsion_energy_radius,}

if PBC:
    params_final["density"] = density

# Save final parameters to file
sim_dir = args[2]
makedirs(sim_dir, exist_ok=True)
param_file_path = path.join(sim_dir, "parameters.json")
json.dump(params_final, open(param_file_path, mode="w"))
print(f"General parameters for the simulation set successfully and saved in {param_file_path}")


############### Simulating the polymer ###############

save_every_blocks = 1
try:
    saved_block_size = int(args[3])
except IndexError:
    saved_block_size = 1000
print(f"Saving every {save_every_blocks * moldyn_steps} timesteps")

try:
    conf = np.load(f"/lus/home/CT3/c1916693/gganesh/repositories/polychrom-sim-launchers/resources/simulation/configurations/starting_conformation_N={polymer_length}.npy")
    print("Loaded a polymer conformation from a previous equilibriated simulation")
except FileNotFoundError:
    conf = grow_cubic(polymer_length, int(simbox_length), method="linear")
    print("Starting simulation with a linear chain")

reporter_default = HDF5Reporter(folder=sim_dir, max_data_length=saved_block_size, overwrite=True, blocks_only=False)
print("Reporters ready, starting simulation")

start_timer = perf_counter()
sim = Simulation(
        platform="OpenCL",
        integrator=integrator,
        timestep=timestep,
        error_tol=0.01,
        length_scale=length_scale,
        collision_rate=friction_coefficient, 
        N=len(conf),
        reporters=[reporter_default],
        PBCbox=PBC_box,
        precision="mixed",
        verbose=False,)
sim.set_data(conf)
sim.add_force(
    forcekits.polymer_chains(
        sim,
        chains=[(0, None, False)],

        bond_force_func=forces.harmonic_bonds,
        bond_force_kwargs={
            "bondLength": harmonic_bond_length,
            "bondWiggleDistance": harmonic_bond_wiggle_dist
        },

        angle_force_func=forces.angle_force,
        angle_force_kwargs={
            "k": polymer_stiffness
        },

        nonbonded_force_func=forces.polynomial_repulsive,
        nonbonded_force_kwargs={
            "trunc": repulsion_energy,
            "radiusMult": repulsion_energy_radius,
        },

        except_bonds=True,      # do not calculate non-bonded forces between monomers connected by a bond
    )
)

sim.local_energy_minimization()     # local energy minimization at first step
    
for t in range(trajectory_length):
    if t % save_every_blocks == (save_every_blocks - 1):
        sim.do_block(steps=moldyn_steps)
    else:
        sim.integrator.step(moldyn_steps)  # do steps without getting the positions from the GPU (faster)
reporter_default.dump_data()

del sim     # delete the simulation object
print(f"Simulation completed in {readable_duration(perf_counter() - start_timer)}")