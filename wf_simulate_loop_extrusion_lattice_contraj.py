from time import perf_counter
import numpy as np
import h5py
import sys
import smc
import json
from numbers import Number
from itertools import chain
from polychrom.hdf5_format import load_hdf5_file, save_hdf5_file
from os import path
import matplotlib.pyplot as plt
from smc import head, LEF


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
sim_dir = args[1]
assert path.isdir(sim_dir)
params = json.load(open(path.join(sim_dir, "parameters_loop_extrusion.json"), mode="r"))
print(f"Loaded existing parameters from < {path.join(sim_dir, 'parameters_loop_extrusion.json')} >")

try:
    trajectory_length = int(args[2])
    assert type(trajectory_length) == int, "Trajectory length must be an integer"
    assert 0 < trajectory_length <= 100000000, "Trajectory length falls outside the accepted range"
except KeyError:
    raise KeyError("The length of the trajectory `trajectory_length` has to be defined")
print(f"Trajectory will be continued for {trajectory_length} steps")

# compute adjusted system sizes and lattice length based on the parameters
system_length_adj = (params["system_length"] + (2 * params["buffer_zone_length"]))

# prepare the dictionary of CTCF positions, orientations, and probabilities throughout the lattice
CTCF_left_capture = {}
CTCF_right_capture = {}
CTCF_left_release = {}
CTCF_right_release = {}

CTCF_sites_adj = [site + params["buffer_zone_length"] for site in params["CTCF_sites"]]
CTCF_orientations = params["CTCF_orientations"]
CTCF_capture_probs = params["CTCF_capture_probs"]
CTCF_release_probs = params["CTCF_release_probs"]

if isinstance(CTCF_capture_probs, Number):
    CTCF_capture_probs = [CTCF_capture_probs] * len(CTCF_sites_adj)

if isinstance(CTCF_release_probs, Number):
    CTCF_release_probs = [CTCF_release_probs] * len(CTCF_sites_adj)

for system_idx in range(params["num_systems"]):
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

# defining the arguments to be passed as parameters for the simulation
simargs = {"trajectory_length": trajectory_length,
            "system_length": params["system_length"],
            "num_systems": params["num_systems"],
            "buffer_zone_length": params["buffer_zone_length"],
            "lattice_length": params["lattice_length"],
            # "initial_LEF_occupancy": initial_LEF_occupancy,
            # "initial_num_LEFs": initial_num_LEFs,
            "constant_LEF_occupancy": params["constant_LEF_occupancy"],
            "LEF_step_prob": params["LEF_step_prob"],
            "LEF_unload_prob": params["LEF_unload_prob"],
            "LEF_stalled_unload_prob": params["LEF_stalled_unload_prob"],
            "LEF_traversal_prob": params["LEF_traversal_prob"],
            "CTCF_capture": {-1: CTCF_left_capture, 1: CTCF_right_capture},
            "CTCF_release": {-1: CTCF_left_release, 1: CTCF_right_release},
            }

if not params["constant_LEF_occupancy"]:
    simargs["LEF_load_prob"] = params["LEF_load_prob"]
# NOTE: previously used for preferential unloading of LEFs, will be completely removed in later versions
if params["buffer_zone_length"] > 0:
    # compute adjusted system sizes and lattice length based on the parameters
    system_length_adj = (params["system_length"] + (2 * params["buffer_zone_length"]))
    lattice_length = system_length_adj * params["num_systems"]
    buffer_zone = []
    for i in range(params["num_systems"]):
        buffer_zone.extend(list(range(system_length_adj * i, (system_length_adj * i) + params["buffer_zone_length"], 1)))
        buffer_zone.extend(list(range(system_length_adj * (i + 1), (system_length_adj * (i + 1)) - params["buffer_zone_length"], -1)))
    buffer_zone.sort()
    simargs["buffer_zone"] = buffer_zone
else:
    simargs["buffer_zone"] = []

print("Finished preparing simulation arguments from existing parameters file")

eq_time = 0

LEF_state_trajectory = load_hdf5_file(path.join(sim_dir, "LEF_state_trajectory.h5"))["trajectory"][:]
LEF_final_state = LEF_state_trajectory[-1,:]
occupied_last = np.asarray(LEF_final_state > 0, dtype=int)
print("Successfully loaded existing LEF state trajectory and set `occupied` LEF matrix")

data = load_hdf5_file(path.join(sim_dir, "LEF_positions_trajectory.h5"))
if simargs["constant_LEF_occupancy"]:
    LEF_positions_trajectory = data["LEF_positions"]
    trajectory_length_init = LEF_positions_trajectory.shape[0]
    LEF_pos_last = LEF_positions_trajectory[-1,:]
else:
    time_steps = [int(k) for k in data.keys()]
    trajectory_length_init = max(time_steps) + 1
    LEF_pos_last = data[str(trajectory_length_init-1)]
print(f"Initial trajectory length was {trajectory_length_init} steps")
print("Successfully loaded existing LEF positions trajectory and set last `LEF_pos`")

# simulating LEF dynamics
def continue_lattice_LEF_dynamics(occupied_last, LEFs_last, simargs):
    occupied = np.copy(occupied_last)
    occupied[0] = 1
    occupied[-1] = 1
    state_trajectory = np.zeros((simargs["trajectory_length"], simargs["lattice_length"]), dtype=int)
    LEFs = [LEF(head(LEFs_last[n,0]), head(LEFs_last[n,1])) for n in range(LEFs_last.shape[0])]
    positions = []
    for t in range(simargs["trajectory_length"]):
        if t % 1000 == 0:
            print(f"... completed up to t={t}...")
        smc.translocate(LEFs, occupied, simargs)
        if t >= eq_time:
            positions.append([(LEF.left.pos, LEF.right.pos) for LEF in LEFs])
            state_trajectory[t-eq_time,:] = smc.state(LEFs, simargs)
    return positions, state_trajectory, simargs

print("Starting stochastic simulation...")
start = perf_counter()
LEF_pos_traj_new, LEF_state_traj_new, simargs_final = continue_lattice_LEF_dynamics(occupied_last=occupied_last,
                                                                                    LEFs_last=LEF_pos_last,
                                                                                    simargs=simargs)
# print(LEF_state_traj_new.shape)
print(f"Completed simulation of {trajectory_length} steps in {perf_counter() - start:.2f} seconds")

simargs_final["trajectory_length"] = trajectory_length_init + simargs_final["trajectory_length"] - eq_time
del simargs_final["CTCF_capture"]
del simargs_final["CTCF_release"]
del simargs_final["buffer_zone"]
simargs_final["CTCF_sites"] = params["CTCF_sites"]
simargs_final["CTCF_orientations"] = params["CTCF_orientations"]
simargs_final["CTCF_capture_probs"] = params["CTCF_capture_probs"]
simargs_final["CTCF_release_probs"] = params["CTCF_release_probs"]

occupancies = [len(LEF_pos_traj_new[i]) * (100 / simargs_final["lattice_length"]) \
               for i in range(len(LEF_pos_traj_new))]
print(f"Mean LEF occupancy was {np.mean(occupancies[:]):.2f}")
loops = np.asarray(list(chain.from_iterable(LEF_pos_traj_new)))
loop_sizes = loops[:, 1] - loops[:, 0]
mean_loop_size = np.mean(loop_sizes)
print(f"Mean loop size was {mean_loop_size:.2f} kb")

print("Appending new trajectories to existing files...")

json.dump(simargs_final, open(path.join(sim_dir, "parameters_loop_extrusion.json"), mode="w"), indent=4)

hdf5_write_opts = {"compression_opts": 9, "compression": "gzip"}

if params["constant_LEF_occupancy"]:
    LEF_positions_trajectory = np.concatenate((LEF_positions_trajectory, np.asarray(LEF_pos_traj_new)), axis=0)
    with h5py.File(path.join(sim_dir, "LEF_positions_trajectory.h5"), mode="a") as handler:
        del handler["LEF_positions"]
        handler.create_dataset(name="LEF_positions", data=LEF_positions_trajectory, **hdf5_write_opts)
else:
    save_hdf5_file(path.join(sim_dir, "LEF_positions_trajectory.h5"),
                   {str(t+trajectory_length_init): np.asarray(LEF_pos_traj_new[t]) \
                    for t in range(len(LEF_pos_traj_new))},
                   mode="a")

LEF_state_trajectory = np.concatenate((LEF_state_trajectory, LEF_state_traj_new), axis=0)
# print(LEF_state_trajectory.shape)
with h5py.File(path.join(sim_dir, "LEF_state_trajectory.h5"), mode="a") as handler:
    del handler["trajectory"]    
    handler.create_dataset(name="trajectory", data=LEF_state_trajectory, **hdf5_write_opts)

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

N_LEF_evol = smc._LEF_occupancy(LEF_pos_traj_new, simargs_final)
N_LEF_evol = [n / params["num_systems"] for n in N_LEF_evol]
plt.figure(figsize=(11, 6))
plt.plot(N_LEF_evol)
plt.xlabel("Time")
plt.ylabel("LEF occupancy per 100 sites")
# plt.xticks(ticks=range(0, 1601, 100), labels=range(10000, 11601, 100))
plt.title(f"Evolution of LEF density over a sample of the tajectory\nMean LEF occupancy per 100 kb = {np.mean(N_LEF_evol):.3f}")
plt.savefig(path.join(sim_dir, "LEF_density_evolution_sample.png"))
plt.close()

print("Finished saving plots")

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