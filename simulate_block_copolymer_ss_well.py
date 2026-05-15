# Importing general packages
from os import path, makedirs
import json
import sys
from time import perf_counter

# Importing packages for computation
import numpy as np
from numbers import Number

# Importing simulation packages
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter
from custom_reporter import coordinateReporter

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
params_file_path = args[1]
params = json.load(open(params_file_path, mode="r"))
default_params = json.load(open("./resources/simulation/default_params_block_copolymer.json"))

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
    base_repulsion_energy = params["base_repulsion_energy"]
except KeyError:
    base_repulsion_energy = default_params["base_repulsion_energy"]

try:
    selective_repulsion_energy = params["selective_repulsion_energy"]
except KeyError:
    selective_repulsion_energy = default_params["selective_repulsion_energy"]

try:
    attraction_energy_radius = params["attraction_energy_radius"]
except KeyError:
    attraction_energy_radius = default_params["attraction_energy_radius"] * harmonic_bond_length

try:
    base_attraction_energy = params["base_attraction_energy"]
except KeyError:
    base_attraction_energy = default_params["base_attraction_energy"]

try:
    selective_attraction_energy = params["selective_attraction_energy"]
except KeyError:
    selective_attraction_energy = default_params["selective_attraction_energy"]

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

try:
    interaction_matrix = params["interaction_matrix"]
except:
    interaction_matrix = default_params["interaction_matrix"]
if isinstance(interaction_matrix, Number):
    interaction_matrix = [[interaction_matrix, 0], [0, interaction_matrix]]

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
                "base_repulsion_energy": base_repulsion_energy,
                "repulsion_energy_radius": repulsion_energy_radius,
                "base_attraction_energy": base_attraction_energy,
                "attraction_energy_radius": attraction_energy_radius,
                "selective_repulsion_energy": selective_repulsion_energy,
                "selective_attraction_energy": selective_attraction_energy,
                "interaction_matrix": list(interaction_matrix),}

if PBC:
    params_final["density"] = density

interaction_matrix = np.asarray(interaction_matrix)

custom_reporter_flag = False

# Save final parameters to file
sim_dir = args[3]
makedirs(sim_dir, exist_ok=False)
json.dump(params_final, open(path.join(sim_dir, "parameters.json"), mode="w"))
print(f"General parameters for the simulation set successfully and saved in {path.join(sim_dir, "parameters.json")}")

##### Monomer types/classes info #####
monomer_classes_file_path = args[2]
if monomer_classes_file_path.endswith(".csv"):
    import pandas as pd
    from itertools import chain
    raw_evec_data = pd.read_csv(monomer_classes_file_path, sep="\t")
    compartments = (raw_evec_data["E1"].to_numpy() >= 0).astype(int)
    monomer_classes = list(chain.from_iterable([[compartments[i] for _ in range(16)] for i in range(len(compartments))]))
    # getting linear loci positions from the data
    MS2_pos = np.array([9969, 9985, 9969, 9985, 9969, 9985, 9985]) - (raw_evec_data["start"][0] // 1000)
    parS_pos = np.array([10027, 9903, 10057, 9836, 10159, 9390, 6657]) - (raw_evec_data["start"][0] // 1000)
    linear_interlocus_dists = abs(MS2_pos - parS_pos)
    loci_positions = np.concatenate((MS2_pos[...,np.newaxis], parS_pos[...,np.newaxis]), axis=1)
    custom_reporter_flag = True
elif monomer_classes_file_path.endswith(".npy"):
    monomer_classes = np.load(monomer_classes_file_path)
else:
    raise IOError("Input file extension unrecognised. Only CSV and Numpy files are read")
monomer_classes = np.asarray(monomer_classes, dtype=int)
assert len(monomer_classes) == polymer_length, "All monomers must be assigned a class; the length of monomer_classes must be equal to the polymer length"
assert len(interaction_matrix) == len(np.unique(monomer_classes)), "The interaction matrix must define interactions between all given monomer types"
np.save(path.join(sim_dir, "monomer_classes.npy"), monomer_classes)

####################


############### Simulating the polymer ###############

save_every_blocks = 1
saved_block_size = int(args[4])
print(f"Saving every {save_every_blocks * moldyn_steps} timesteps")

try:
    conf = np.load(f"./resources/simulation/starting_conformation_N={polymer_length}.npy")
    print("Loaded a polymer conformation from a previous equilibriated simulation")
except FileNotFoundError:
    conf = grow_cubic(polymer_length, int(simbox_length), method="linear")
    print("Starting simulation with a linear chain")

reporter_default = HDF5Reporter(folder=sim_dir, max_data_length=saved_block_size, overwrite=True, blocks_only=False)
reporters = [reporter_default]
if custom_reporter_flag:
    reporter_specific = coordinateReporter(folder=sim_dir, monomer_pos=loci_positions)
    reporters.append(reporter_specific)
print("Reporters ready, starting simulation")

start_timer = perf_counter()
sim = Simulation(
        platform="cuda",
        integrator=integrator,
        timestep=timestep,
        error_tol=0.01, 
        GPU = "0",
        length_scale=length_scale,
        collision_rate=friction_coefficient,
        max_Ek=10,
        N = len(conf),
        reporters=reporters,
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

        nonbonded_force_func=forces.heteropolymer_SSW,      # setting the heteropolymer smoothed square-well potential properties
        nonbonded_force_kwargs={
            "interactionMatrix": interaction_matrix,
            "monomerTypes": monomer_classes,
            "extraHardParticlesIdxs": [],
            "attractionEnergy": base_attraction_energy,
            "attractionRadius": attraction_energy_radius,
            "repulsionEnergy": base_repulsion_energy,   
            "repulsionRadius": repulsion_energy_radius,
            "selectiveRepulsionEnergy": selective_repulsion_energy,
            "selectiveAttractionEnergy": selective_attraction_energy,
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
if custom_reporter_flag:
    reporter_specific.dump_data()

del sim     # delete the simulation object
print(f"Simulation completed in {readable_duration(perf_counter() - start_timer)}")