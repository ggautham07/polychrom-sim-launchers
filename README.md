# polychrom-sim-launchers
Scripts for running OpenMM polychrom-based coarse-grained polymer simulations

# To know for Adastra

- AMD GPUs only, so OpenCL platform must be used
- uses ROCM, so usage can be viewed with `rocm-smi`
- GPU device selection is managed by SLURM automatically
- One simulation per GPU since `polychrom` does not handle parallelisation