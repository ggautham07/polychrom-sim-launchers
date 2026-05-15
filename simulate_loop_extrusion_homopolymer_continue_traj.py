# Importing general packages
from os import path
import json
import sys
from time import perf_counter, sleep

# Importing packages for computation
import numpy as np
import pandas as pd

# Importing simulation packages
import smc
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
from polychrom.starting_conformations import grow_cubic
from polychrom.hdf5_format import HDF5Reporter, load_hdf5_file, list_URIs
# from custom_reporter import coordinateReporter


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
sim_dir = args[1]
params = json.load(open(path.join(sim_dir, "parameters.json"), mode="r"))
print(f"Loaded existing parameters from < {path.join(sim_dir, 'parameters.json')} >")

trajectory_length = int(args[2])

polymer_length = params["polymer_length"]
length_scale = params["length_scale"]
integrator = params["integrator"]
assert integrator in ["brownian", "langevin", "verlet", "variableLangevin", "variableVerlet"]
if "variable" not in integrator:
    try:
        timestep = params["timestep"]
    except KeyError:
        timestep = False
else:
    timestep = False
    
friction_coefficient = params["friction_coefficient"]
moldyn_steps = params["moldyn_steps"]
harmonic_bond_length = params["harmonic_bond_length"]
harmonic_bond_wiggle_dist = params["harmonic_bond_wiggle_dist"]
polymer_stiffness = params["polymer_stiffness"]
repulsion_energy_radius = params["repulsion_energy_radius"]
repulsion_energy = params["repulsion_energy"]
PBC = params["PBC"]
if PBC:
    density = params["density"]
    simbox_length = (polymer_length / density) ** 0.333
    PBC_box = [simbox_length, simbox_length, simbox_length]
else:
    PBC_box = False
    simbox_length = (polymer_length / 0.0001) ** 0.333
LEF_bond_length = params["LEF_bond_length"]
LEF_bond_wiggle_distance = params["LEF_bond_wiggle_dist"]


############### Getting saved lattice-based loop extrusion trajectories ###############

URIs = list_URIs(sim_dir, return_dict=True)
URI_idxs = np.asarray(list(URIs.keys()))
timestep_last = URI_idxs[-1]
raw_trajectory_data = load_hdf5_file(path.join(sim_dir, "LEF_positions_trajectory.h5"))
timesteps = list(raw_trajectory_data.keys())
timesteps.sort(key=lambda x: int(x))
timesteps = timesteps[timestep_last+1:]
LEF_pos = []
for t in timesteps:
    posdata = raw_trajectory_data[t]
    LEF_pos.append([(posdata[i, 0], posdata[i, 1]) for i in range(len(posdata))])
milker = smc.extruder(LEF_pos)        # an extruder object for introducing bonds between LEF-connected monomers throughout the simulation


############### Simulating the polymer ###############

params["trajectory_length"] = trajectory_length + int(timestep_last) + 1
json.dump(params, open(path.join(sim_dir, "parameters.json"), mode="w"), indent=4)
print("Saved polymer simulation parameters successfully")

save_every_blocks = 1
restart_simulation_every_blocks = 10000     # TODO change based on trajectory length
assert trajectory_length % restart_simulation_every_blocks == 0
total_sim_inits = trajectory_length // restart_simulation_every_blocks
saved_block_size = int(args[3])
print(f"Saving every {save_every_blocks * moldyn_steps} timesteps")

for iteration in range(total_sim_inits):
    if iteration == 0:
        reporter_default = HDF5Reporter(folder=sim_dir,
                                        max_data_length=saved_block_size,
                                        check_exists=False,
                                        overwrite=False,
                                        blocks_only=True)
        _, data = reporter_default.continue_trajectory()
        reporters = [reporter_default]
        print(f"Reporters ready, continuing existing simulation from {reporter_default.counter['data'] * moldyn_steps * save_every_blocks:.2E} already completed timesteps for another {trajectory_length * moldyn_steps:.2E} timesteps")
        conf = data["pos"]
        start_timer = perf_counter()

    sim = Simulation(
            platform="OpenCL",
            integrator=integrator,
            timestep=timestep,
            error_tol=0.01,
            length_scale=length_scale,
            collision_rate=friction_coefficient, 
            N=len(conf),
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
    print("Successfully setup LEF bonds for this simulation instance")

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