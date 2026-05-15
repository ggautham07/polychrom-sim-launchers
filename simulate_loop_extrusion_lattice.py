from time import perf_counter
import numpy as np
import sys
import smc
import json
from numbers import Number
from itertools import chain
from polychrom.hdf5_format import save_hdf5_file
from os import path, makedirs
import matplotlib.pyplot as plt

"""
Simulate loop extrusion by LEFs on a one-dimensional lattice given a set of simulation parameters and record their positions and states


Simulation parameters guide
----------
Define a dictionary with the following keys and set the required values for the simulation parameters
trajectory_length : length of the simulation trajectory, example: 10000
system_length : length of the system, example: 600
num_systems : number of systems to simulate in parallel, by default: 1, NOTE: recommended not to change until the code is tailored for simulating multiple systems
buffer_zone_length : length of the portion to add on either side of the system in the lattice, by default: 0, useful in both lattice-based on molecular dynamics simulations
inital_LEF_occupancy : initial LEF occupancy of the system per 100 sites, will change over the trajectory if constant_LEF_occupancy is False, by default: 1 LEF per 100 sites
constant_LEF_occupancy : LEF occupancy does not change over time if True, LEF_load_prob needs to be defined if False, by default: True
LEF_load_prob : probability of loading a LEF in the system per time step, by default: 0.1, NOTE: only considered if constant_LEF_occupancy is False
LEF_unload_prob : probability of unloading for every LEF per time step, by default: 0.01
LEF_stalled_unload_prob : probability of unloading of every stalled LEF per time step (differential unloading probability), by default: same as LEF_unload_prob
LEF_traversal_prob : probability of LEFs traversing one another per LEF pair per time step when they encounter one another, will stall or unload otherwise
CTCF_sites : sites occupied by CTCF or other blockers of loop extrusion, pass the indices of the lattice, example: [120, 360, 480] on a system of length 600
CTCF_orientations : orientations of the CTCFs or other blockers, "+" blocks LEFs approaching from the right, "-" from the left, and "=" from both directions, example: ["+", "-", "="]
CTCF_capture_probs : probability of LEF capture by a CTCF, can be a scalar or a vector of length CTCF_sites, example: [0.96, 0.84, 0.72]
CTCF_release_probs : probability of LEF release by a CTCF, can be a scalar or a vector of length CTCF_sites, example: 0.01
"""

args = sys.argv
params = json.load(open(sys.argv[1], mode="r"))

default_params = {"num_systems": 1,
                  "buffer_zone_length": 0,
                  "initial_LEF_occupancy": 1,
                  "constant_LEF_occupancy": True,
                  "LEF_step_prob": 1,
                  "LEF_load_prob": 0.01,
                  "LEF_unload_prob": 0.01,
                  "LEF_traversal_prob": 0.2,
                  "CTCF_sites": [],
                  "CTCF_orientations": [],
                  "CTCF_capture_probs": 0.9,
                  "CTCF_release_probs": 0.1,}

try:
    trajectory_length = params["trajectory_length"]
    assert type(trajectory_length) == int, "Trajectory length must be an integer"
    assert 0 < trajectory_length <= 100000000, "Trajectory length falls outside the accepted range"
except KeyError:
    raise KeyError("The length of the trajectory `trajectory_length` has to be defined")

try:
    system_length = params["system_length"]
    assert type(system_length) == int, "System length must be an integer"
    assert 10 < system_length <= 1000000, "System length falls outside the accepted range"
except KeyError:
    raise KeyError("The length of the system `system_length` has to be defined")

try:
    num_systems = params["num_systems"]
    assert type(num_systems) == int, "Number of systems must be an integer"
    assert 0 < num_systems <= 1000, "Number of systems falls outside the accepted range"
except KeyError:
    num_systems = default_params["num_systems"]

try:
    buffer_zone_length = params["buffer_zone_length"]
    assert type(buffer_zone_length) == int, "Buffer zone must be an integer"
    assert 0 <= buffer_zone_length <= system_length / 2, "Buffer zone falls outside the accepted range"
except KeyError:
    buffer_zone_length = default_params["buffer_zone_length"]

try:
    initial_LEF_occupancy = params["initial_LEF_occupancy"]
    assert isinstance(initial_LEF_occupancy, Number), "Initial LEF must be a number"
    assert 0 < initial_LEF_occupancy <= 50, "Initial LEF occupancy falls outside the accepted range"
except KeyError:
    initial_LEF_occupancy = default_params["initial_LEF_occupancy"]


try:
    constant_LEF_occupancy = params["constant_LEF_occupancy"]
    assert type(constant_LEF_occupancy) == bool, "Value must be boolean"
except KeyError:
    constant_LEF_occupancy = default_params["constant_LEF_occupancy"]

try:
    LEF_step_prob = params["LEF_step_prob"]
    assert isinstance(initial_LEF_occupancy, Number), "Initial LEF must be a number"
    assert 0 < LEF_step_prob <= 1
except:
    LEF_step_prob = default_params["LEF_step_prob"]

if not constant_LEF_occupancy:
    try:
        LEF_load_prob = params["LEF_load_prob"]
        assert 0 <= LEF_load_prob <= 1, "LEF load probability must be between 0 and 1"
    except KeyError:
        LEF_load_prob = default_params["LEF_load_prob"]

try:
    LEF_unload_prob = params["LEF_unload_prob"]
    assert 0 <= LEF_unload_prob <= 1, "LEF unload probability must be between 0 and 1"
except KeyError:
    LEF_unload_prob = default_params["LEF_unload_prob"]

try:
    LEF_stalled_unload_prob = params["LEF_stalled_unload_prob"]
    assert 0 <= LEF_stalled_unload_prob <= 1, "Stalled LEF unload probability must be between 0 and 1"
except KeyError:
    LEF_stalled_unload_prob = LEF_unload_prob

try:
    LEF_traversal_prob = params["LEF_traversal_prob"]
    assert 0 <= LEF_traversal_prob <= 1, "LEF traversal probability must be between 0 and 1"
except KeyError:
    LEF_traversal_prob = default_params["LEF_traversal_prob"]

try:
    CTCF_sites = params["CTCF_sites"]
    assert type(CTCF_sites) == list, "CTCF sites must be a list"
except KeyError:
    CTCF_sites = default_params["CTCF_sites"]

try:
    CTCF_orientations = params["CTCF_orientations"]
    assert type(CTCF_sites) == list, "CTCF orientations must be a list"
except KeyError:
    CTCF_orientations = ["="] * len(CTCF_sites) if len(CTCF_sites) > 0 else []
assert len(CTCF_orientations) == len(CTCF_sites)

try:
    CTCF_capture_probs = params["CTCF_capture_probs"]
    assert isinstance(CTCF_capture_probs, (Number, list, np.ndarray))
except KeyError:
    CTCF_capture_probs = default_params["CTCF_capture_probs"]

try:
    CTCF_release_probs = params["CTCF_release_probs"]
    assert isinstance(CTCF_capture_probs, (Number, list, np.ndarray))
except KeyError:
    CTCF_release_probs = default_params["CTCF_release_probs"]

# compute adjusted system sizes and lattice length based on the parameters
system_length_adj = (system_length + (2 * buffer_zone_length))
lattice_length = system_length_adj * num_systems

# compute the initial number of LEFs that will occupy the lattice
initial_num_LEFs = int(np.round(initial_LEF_occupancy * (lattice_length / 100) * num_systems))     # NOTE: must find a better way to do this because casting must be set to unsafe
if initial_num_LEFs == 0:
    raise ValueError("Initial LEF occupancy per 100 monomers is too low, choose a value such that at least one LEF can be loaded onto the system")

# prepare the dictionary of CTCF positions, orientations, and probabilities throughout the lattice
CTCF_left_capture = {}
CTCF_right_capture = {}
CTCF_left_release = {}
CTCF_right_release = {}

CTCF_sites_adj = [site + buffer_zone_length for site in CTCF_sites]

if isinstance(CTCF_capture_probs, Number):
    CTCF_capture_probs = [CTCF_capture_probs] * len(CTCF_sites_adj)

if isinstance(CTCF_release_probs, Number):
    CTCF_release_probs = [CTCF_release_probs] * len(CTCF_sites_adj)

for system_idx in range(num_systems):
    for site_idx in range(len(CTCF_sites_adj)):
        site = (system_idx * system_length_adj) + CTCF_sites_adj[site_idx]
        if CTCF_orientations[site_idx] == "+":
            CTCF_left_capture[site] = CTCF_capture_probs[site_idx]
            CTCF_left_release[site] = CTCF_release_probs[site_idx]
            CTCF_right_capture[site] = 0
            CTCF_right_release[site] = 1
        elif CTCF_orientations[site_idx] == "-":
            CTCF_left_capture[site] = 0
            CTCF_left_release[site] = 1
            CTCF_right_capture[site] = CTCF_capture_probs[site_idx]
            CTCF_right_release[site] = CTCF_release_probs[site_idx]
        elif CTCF_orientations[site_idx] == "=":
            CTCF_left_capture[site] = CTCF_capture_probs[site_idx]
            CTCF_left_release[site] = CTCF_release_probs[site_idx]
            CTCF_right_capture[site] = CTCF_capture_probs[site_idx]
            CTCF_right_release[site] = CTCF_release_probs[site_idx]
        elif CTCF_orientations[site_idx] == "*":
            CTCF_left_capture[site] = 0
            CTCF_left_release[site] = 0
            CTCF_right_capture[site] = 0
            CTCF_right_release[site] = 0
        else:
            raise KeyError(f"{CTCF_orientations[site_idx]} is an invalid value for CTCF orientation. Accepted values are +, -, = and *")

# add a long enough equilibration time to the trajectory length
eq_time = 50000
trajectory_length = trajectory_length + eq_time

# defining the arguments to be passed as parameters for the simulation
simargs = {"trajectory_length": trajectory_length,
            "system_length": system_length,
            "num_systems": num_systems,
            "buffer_zone_length": buffer_zone_length,
            "lattice_length": lattice_length,
            "initial_LEF_occupancy": initial_LEF_occupancy,
            "initial_num_LEFs": initial_num_LEFs,
            "constant_LEF_occupancy": constant_LEF_occupancy,
            "LEF_step_prob": LEF_step_prob,
            "LEF_unload_prob": LEF_unload_prob,
            "LEF_stalled_unload_prob": LEF_stalled_unload_prob,
            "LEF_traversal_prob": LEF_traversal_prob,
            "CTCF_capture": {-1: CTCF_left_capture, 1: CTCF_right_capture},
            "CTCF_release": {-1: CTCF_left_release, 1: CTCF_right_release},
            }
if not constant_LEF_occupancy:
    simargs["LEF_load_prob"] = LEF_load_prob
# NOTE: previously used for preferential unloading of LEFs, will be completely removed in later versions
if buffer_zone_length > 0:
    buffer_zone = []
    for i in range(num_systems):
        buffer_zone.extend(list(range(system_length_adj * i, (system_length_adj * i) + buffer_zone_length, 1)))
        buffer_zone.extend(list(range(system_length_adj * (i + 1), (system_length_adj * (i + 1)) - buffer_zone_length, -1)))
    buffer_zone.sort()
    simargs["buffer_zone"] = buffer_zone
else:
    simargs["buffer_zone"] = []

print("Finished preparing simulation arguments")

# simulating LEF dynamics
def simulate_lattice_LEF_dynamics(simargs):
    occupied = np.zeros(simargs["lattice_length"])
    occupied[0] = 1
    occupied[-1] = 1
    LEFs = []
    state_trajectory = np.zeros((simargs["trajectory_length"], simargs["lattice_length"]), dtype=int)
    for _ in range(simargs["initial_num_LEFs"]):
        smc.load(LEFs, occupied, simargs)
    positions = []
    for t in range(simargs["trajectory_length"]):
        if t % 1000 == 0:
            print(f"... completed up to t={t}...")
        smc.translocate(LEFs, occupied, simargs)
        if t >= eq_time:
            positions.append([(LEF.left.pos, LEF.right.pos) for LEF in LEFs])
            state_trajectory[t-eq_time,:] = smc.state(LEFs, simargs)
    return positions, state_trajectory, simargs

print("Starting stochastic simulation... Equilibration time will be 5e4 steps")
start = perf_counter()
LEF_positions_trajectory, LEF_state_trajectory, simargs_final = simulate_lattice_LEF_dynamics(simargs)
print(f"Completed simulation of {trajectory_length - eq_time} steps in {perf_counter() - start:.2f} seconds")

simargs_final["trajectory_length"] = simargs_final["trajectory_length"] - eq_time
del simargs_final["CTCF_capture"]
del simargs_final["CTCF_release"]
del simargs_final["buffer_zone"]
simargs_final["CTCF_sites"] = CTCF_sites
simargs_final["CTCF_orientations"] = CTCF_orientations
simargs_final["CTCF_capture_probs"] = CTCF_capture_probs
simargs_final["CTCF_release_probs"] = CTCF_release_probs

occupancies = [len(LEF_positions_trajectory[i]) * (100 / lattice_length) for i in range(len(LEF_positions_trajectory))]
print(f"Mean LEF occupancy was {np.mean(occupancies[:]):.2f}")
loops = np.asarray(list(chain.from_iterable(LEF_positions_trajectory)))
loop_sizes = loops[:, 1] - loops[:, 0]
mean_loop_size = np.mean(loop_sizes)
print(f"Mean loop size was {mean_loop_size:.2f} kb")

print("Saving trajectories...")
sim_dir = args[2]
if sim_dir.endswith("---") or sim_dir.endswith("---/"):
    sim_dir = sim_dir.removesuffix("/")
    sim_dir = sim_dir.replace("---", f"_unload={simargs_final['LEF_unload_prob']:.2f}")
    if not simargs_final["constant_LEF_occupancy"]:
        sim_dir = sim_dir + f"_load={simargs_final['LEF_load_prob']:.2f}"
makedirs(sim_dir, exist_ok=True)
json.dump(simargs, open(path.join(sim_dir, "parameters_loop_extrusion.json"), mode="w"), indent=4)
if constant_LEF_occupancy:
    save_hdf5_file(path.join(sim_dir, "LEF_positions_trajectory.h5"), {"LEF_positions": np.asarray(LEF_positions_trajectory)})
else:
    save_hdf5_file(path.join(sim_dir, "LEF_positions_trajectory.h5"), {str(i): np.asarray(LEF_positions_trajectory[i]) for i in range(len(LEF_positions_trajectory))})
save_hdf5_file(path.join(sim_dir, "LEF_state_trajectory.h5"), {"trajectory": LEF_state_trajectory})

print("Finished saving trajectories and parameters")

plt.figure(figsize=(6, 6))
plt.hist(loop_sizes, density=True, bins=30);
plt.xlabel("Loop sizes ($l$)")
plt.ylabel("Density")
plt.axvline(mean_loop_size, color="red", label=f"$\\overline{{l}} = {mean_loop_size:.2f}$")
plt.legend()
plt.savefig(path.join(sim_dir, "loop_sizes.svg"), format="svg")
plt.close()

## To be copied to the analysis code for 1D simulations, with other observables like loop size

# N_LEF_evol = smc._LEF_occupancy(LEF_positions, params)
# N_LEF_evol = [n / params["num_systems"] for n in N_LEF_evol]
# plt.figure(figsize=(11, 6))
# plt.plot(N_LEF_evol[10000:11600])
# plt.xlabel("Time")
# plt.ylabel("LEF occupancy per 100 sites")
# plt.xticks(ticks=range(0, 1601, 100), labels=range(10000, 11601, 100))
# plt.title(f"Evolution of LEF density over a sample of the tajectory\nMean LEF occupancy per 100 kb = {np.mean(N_LEF_evol):.3f}")
# plt.savefig(path.join(sim_dir, "LEF_density_evolution_sample.png"))
# plt.close()

# # Plotting LEF state trajectory
# hw_ratio = simargs_final["trajectory_length"] / simargs_final["lattice_length"]
# states = ["none", "free", "stalled", "captured"]
# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
# fig, ax = plt.subplots()
# fig.set_figwidth(7)
# fig.set_figheight(7 * hw_ratio)
# num_states = np.max(LEF_state_trajectory) + 1
# print(num_states)
# cmap = plt.get_cmap("inferno", num_states)
# im = ax.imshow(LEF_state_trajectory[eq_time:], cmap=cmap)
# ax.set_ylabel("Time")
# ax.set_xlabel("Lattice")
# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="3%", pad=0.2)
# cbar = fig.colorbar(im, label="LEF state", cax=cax, ticks=range(num_states))
# cbar.ax.set_yticklabels(states[:num_states])
# # plt.savefig(path.join(sim_dir, "LEF_state_trajectory.svg"), format="svg")
# plt.savefig(path.join(sim_dir, "LEF_state_trajectory.png"), format="png", dpi=600)
# plt.close()