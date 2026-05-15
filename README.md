# About

Scripts for running OpenMM `polychrom`-based (https://github.com/open2c/polychrom) coarse-grained polymer simulations of chromatin.

Parameters are generally passed as a JSON file, which is accepted as an argument to the script.

Also includes scripts to simulate the dynamics of loop extruding factors (LEFs) such as cohesins on a 1D lattice.


# Contents

## Resources

- **resources**
    houses default simulation parameters and starting polymer conformations, see sub-directories for more info

## Simulation of simple homopolymer chain

- **simulate_homopolymer.py**

## Simulation of heteropolymer chains

- **simulate_block_copolymer_parabolic_well.py**
    heteropolymer with a custom parabolic well potential for attraction between monomers (see Fig. 3A of ![this article](https://elifesciences.org/reviewed-preprints/108117)). Used primarily to simulate compartmental segregation

- **simulate_block_copolymer_parabolic_well_continue_traj.py**
    to continue an existing trajectory simulated using the previous script

- **simulate_block_copolymer_ss_well.py**
    simulate a heteropolymer using the default smooth-squared well potential in `polychrom`

## Simulations of loop extrusion

- **smc.py**
    Python class for LEFs, adapted from `polychrom` examples

- **simulate_loop_extrusion_lattice.py**
    1D simulation script of loop extrusion on a lattice. Needs `smc.py`!

- **simulate_loop_extrusion_lattice_continue_traj.py**
    continue an existing trajectory simulated using the previous script

- **simulate_loop_extrusion_homopolymer.py**
    connect the 1D simulation to a 3D heteropolymer simulation the traditional `polychrom` way, i.e., bonds between anchored sites that change based on the 1D trajectory

- **simulate_loop_extrusion_homopolymer_continue_traj.py**
    continue an existing trajectory...

- **simulate_loop_extrusion_heteropolymer.py**
    connect the 1D simulation to a 3D heteropolymer simulation - used primarily to combine loop extrusion and compartments