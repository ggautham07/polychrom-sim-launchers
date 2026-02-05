# Importing general packages
from os import path
import json
import sys
from time import perf_counter

# Importing packages for computation
import numpy as np

# Importing simulation packages
from polychrom import forces
from polychrom import forcekits
from polychrom.simulation import Simulation
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


# New potential for heteropolymers based on a parabolic well
def heteropolymer_parabolic_well(sim_object:Simulation,
                                interactionMatrix:np.ndarray,
                                monomerTypes,
                                repulsionEnergy=3.0,
                                repulsionRadius=1.0,
                                selectiveAttractionEnergy=1.0,
                                attractionRadius=1.8,
                                name="heteropolymer_parabolic_well"):
    
    """Heteropolymer with common parabolic repulsive and selective
    attractive potentials given an array of monomer types and an
    interaction matrix describing the level of attraction between each type"""

    import openmm
    # Check type info for consistency
    Ntypes = max(monomerTypes) + 1  # IDs should be zero based
    if any(np.less(interactionMatrix.shape, [Ntypes, Ntypes])):
        raise ValueError("Need interactions for {0:d} types!".format(Ntypes))
    if not np.allclose(interactionMatrix.T, interactionMatrix):
        raise ValueError("Interaction matrix should be symmetric!")

    indexpairs = []
    for i in range(0, Ntypes):
        for j in range(0, Ntypes):
            if not interactionMatrix[i, j] == 0:
                indexpairs.append((i, j))
    assert len(indexpairs) > 0

    energy = "Erep * step(rrep - r) + Eatt * step(r - rrep);"
    energy += ("Eatt = Eattsel * (delta(type1-{0:d}) * delta(type2-{1:d}) * INT_{0:d}_{1:d}").format(
        indexpairs[0][0], indexpairs[0][1]
    )
    for i, j in indexpairs[1:]:
        energy += " + delta(type1-{0:d}) * delta(type2-{1:d}) * INT_{0:d}_{1:d}".format(i, j)
    energy += ");Eattsel = ((4 * eps) / (rrep - ratt) ^ 2) * (r - (rrep + ratt) / 2) ^ 2 - eps;"
    energy += "Erep = rho * (1 - (r / rrep) ^ 2);"

    force = openmm.CustomNonbondedForce(energy)
    force.name = name
    force.setCutoffDistance(attractionRadius * sim_object.conlen)
    force.addGlobalParameter("rho", repulsionEnergy * sim_object.kT)
    force.addGlobalParameter("rrep", repulsionRadius * sim_object.conlen)
    force.addGlobalParameter("ratt", attractionRadius * sim_object.conlen)
    force.addGlobalParameter("eps", selectiveAttractionEnergy * sim_object.kT)

    for i, j in indexpairs:
        force.addGlobalParameter("INT_{0:d}_{1:d}".format(i, j), interactionMatrix[i, j])
    
    force.addPerParticleParameter("type")

    for i in range(sim_object.N):
        force.addParticle((float(monomerTypes[i]),))

    return force


############### Parameter handling ###############

args = sys.argv
sim_dir = args[1]
params = json.load(open(path.join(sim_dir, "parameters.json"), mode="r"))

trajectory_length = int(args[2])
polymer_length = params["polymer_length"]
length_scale = params["length_scale"]
integrator = params["integrator"]
assert integrator in ["brownian", "langevin", "verlet", "variableLangevin", "variableVerlet"]
if "variable" not in integrator:
    timestep = params["timestep"]
else:
    timestep = False
friction_coefficient = params["friction_coefficient"]
moldyn_steps = params["moldyn_steps"]
harmonic_bond_length = params["harmonic_bond_length"]
harmonic_bond_wiggle_dist = params["harmonic_bond_wiggle_dist"]
polymer_stiffness = params["polymer_stiffness"]
repulsion_energy_radius = params["repulsion_energy_radius"]
base_repulsion_energy = params["base_repulsion_energy"]
attraction_energy_radius = params["attraction_energy_radius"]
selective_attraction_energy = params["selective_attraction_energy"]
PBC = params["PBC"]
if PBC:
    density = params["density"]
    simbox_length = (polymer_length / density) ** 0.333
    PBC_box = [simbox_length, simbox_length, simbox_length]
else:
    PBC_box = False
    simbox_length = (polymer_length / 0.0001) ** 0.333
interaction_matrix = np.asarray(params["interaction_matrix"])

monomer_classes = np.asarray(np.load(path.join(sim_dir, "monomer_classes.npy")), dtype=int)

####################


############### Simulating the polymer ###############

save_every_blocks = 1
saved_block_size = int(args[3])
print(f"Saving every {save_every_blocks * moldyn_steps} timesteps")

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
        max_Ek=15,
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

        nonbonded_force_func=heteropolymer_parabolic_well,      # setting the heteropolymer smoothed square-well potential properties
        nonbonded_force_kwargs={
            "interactionMatrix": interaction_matrix,
            "monomerTypes": monomer_classes,
            "repulsionEnergy": base_repulsion_energy,
            "repulsionRadius": repulsion_energy_radius,
            "selectiveAttractionEnergy": selective_attraction_energy,
            "attractionRadius": attraction_energy_radius,
        },

        except_bonds=True,      # do not calculate non-bonded forces between monomers connected by a bond
    )
)

sim._apply_forces()     # apply forces since this is not the first step - no energy minimisation

for t in range(trajectory_length):
    if t % save_every_blocks == (save_every_blocks - 1):
        sim.do_block(steps=moldyn_steps)
    else:
        sim.integrator.step(moldyn_steps)  # do steps without getting the positions from the GPU (faster)
reporter_default.dump_data()

del sim     # delete the simulation object
print(f"Simulation completed in {readable_duration(perf_counter() - start_timer)}")