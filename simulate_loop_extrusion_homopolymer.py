# Importing general packages
from os import path
import json
import sys
from time import perf_counter, sleep

# Importing packages for computation
import numpy as np

# Importing simulation packages
import smc
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, load_hdf5_file


# Custom parabolic repulsive function
def parabolic_repulsive(sim_object:Simulation, max_energy=3.0, radius=1.0, name="parabolic_repulsive"):
    """This is a simple parabolic repulsive potential. It has the value
    of `max_energy` at zero, and steadily decreases to zero
    together with its first derivative at r=`radius`.

    Parameters
    ----------

    max_energy : float
        the energy value around r=`radius`

    """
    import openmm

    radius = sim_object.conlen * radius
    nbCutOffDist = radius
    repul_energy = "rho * (1 - (r / rrep) ^ 2)"

    force = openmm.CustomNonbondedForce(repul_energy)
    force.name = name

    force.addGlobalParameter("rho", max_energy * sim_object.kT)
    force.addGlobalParameter("rrep", radius)

    for _ in range(sim_object.N):
        force.addParticle(())

    force.setCutoffDistance(nbCutOffDist)

    return force


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
default_params = json.load(open("./resources/simulation/parameters/default_params_loop_extrusion_homopolymer.json"))

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

LEF_bond_length = params["LEF_bond_length"]
LEF_bond_wiggle_distance = params["LEF_bond_wiggle_dist"]

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
                "repulsion_energy_radius": repulsion_energy_radius,
                "LEF_bond_length": LEF_bond_length,
                "LEF_bond_wiggle_dist": LEF_bond_wiggle_distance,}

if PBC:
    params_final["density"] = density

# Save final parameters to file
sim_dir = args[2]
param_file_path = path.join(sim_dir, "parameters.json")
json.dump(params_final, open(param_file_path, mode="w"))
print(f"General parameters for the simulation set successfully and saved in {param_file_path}")


############### Getting saved lattice-based loop extrusion trajectories ###############

raw_trajectory_data = load_hdf5_file(path.join(sim_dir, "LEF_positions_trajectory.h5"))
timesteps = list(raw_trajectory_data.keys())
timesteps.sort(key=lambda x: int(x))
LEF_pos = []
for t in timesteps:
    posdata = raw_trajectory_data[t]
    LEF_pos.append([(posdata[i, 0], posdata[i, 1]) for i in range(len(posdata))])
milker = smc.extruder(LEF_pos)        # an extruder object for introducing bonds between LEF-connected monomers throughout the simulation


############### Simulating the polymer ###############

save_every_blocks = 1
restart_simulation_every_blocks = 10000     # TODO change based on trajectory length
assert trajectory_length % restart_simulation_every_blocks == 0
total_sim_inits = trajectory_length // restart_simulation_every_blocks
saved_block_size = int(args[3])
print(f"Saving every {save_every_blocks * moldyn_steps} timesteps")

for iteration in range(total_sim_inits):
    if iteration == 0:
        try:
            conf = np.load(f"./resources/simulation/configurations/starting_conformation_N={polymer_length}.npy")
            print("Loaded a polymer conformation from a previous equilibriated simulation")
        except FileNotFoundError:
            conf = grow_cubic(polymer_length, int(simbox_length), method="linear")
            print("Starting simulation with a linear chain")

        custom_reporter_flag = False
        reporter_default = HDF5Reporter(folder=sim_dir,
                                        max_data_length=saved_block_size,
                                        overwrite=True,
                                        blocks_only=False)
        reporters = [reporter_default]      # can be used to add other reporters
        print("Reporters ready, starting simulation")

        start_timer = perf_counter()

    sim = Simulation(
            platform="cuda",
            integrator=integrator,
            timestep=timestep,
            error_tol=0.01,
            GPU = "1",      # GPU 1 on lifou, 0 on kalam
            length_scale=length_scale,
            collision_rate=friction_coefficient, 
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

            nonbonded_force_func=parabolic_repulsive,
            nonbonded_force_kwargs={
                "max_energy": repulsion_energy,
                "radius": repulsion_energy_radius,
            },

            except_bonds=True,      # do not calculate non-bonded forces between monomers connected by a bond
        )
    )

    # ------------ initializing milker; adding bonds ---------
    # copied from addBond
    kbond = sim.kbondScalingFactor / (LEF_bond_wiggle_distance ** 2)
    bondDist = LEF_bond_length * sim.length_scale
    activeParams = {"length": bondDist,"k": kbond}
    inactiveParams = {"length": bondDist, "k": 0}
    milker.setParams(activeParams, inactiveParams)
    print("Setting up LEF bonds... (simulation speed-limiting step)")
    milker.setup(bondForce=sim.force_dict['harmonic_bonds'], blocks=restart_simulation_every_blocks)
    print("Successfully setup LEF bonds throughout the trajectory")

    if iteration == 0:
        sim.local_energy_minimization()     # local energy minimization at first step
    else:
        sim._apply_forces()
    for t in range(restart_simulation_every_blocks):
        if t % save_every_blocks == (save_every_blocks - 1):
            sim.do_block(steps=moldyn_steps)
        else:
            sim.integrator.step(moldyn_steps)  # do steps without getting the positions from the GPU (faster)
        if t < (restart_simulation_every_blocks - 1):
            curBonds, pastBonds = milker.step(sim.context)

    conf = sim.get_data()
    del sim
    sleep(0.1)
    reporter_default.blocks_only = True  # write output hdf5-files only for blocks

reporter_default.dump_data()

print(f"Simulation completed in {readable_duration(perf_counter() - start_timer)}")