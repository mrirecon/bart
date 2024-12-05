# Singularity Container for BART

Running BART on a high performance computing (HPC) cluster can be challenging due to missing libraries.
One solution exploits containers. They encapsulating the software with all required dependencies.
Here, we provide some basic information about how to run BART in a container using [Singularity](https://sylabs.io/singularity/).

A blueprint to create a singularity container for BART can be found in the definition files
* [`bart_ubuntu.def`](bart_ubuntu.def): for an Ubuntu 22.04 operating system. 
* [`bart_debian.def`](bart_debian.def): for a Debian 12 (bookworm) operating system.

After installing singularity, a container `container.sif` can be created in the Singularity Image Format (SIF):
```code
sudo singularity build container.sif bart.def
```

Both containers download and compile BART with all libraries including GPU support using CUDA.
Make sure to select the CUDA version your local HPC host provides.

You can start an interactive shell with

```code
singularity shell --nv container.sif
```
and the `--nv` adds access to the installed Nvidia drivers on the host system.

A BASH script can be executed inside the container with

```code
singularity exec --nv container.sif bash 'recon.sh'
```

### Note
The definition files above also represent simple guides of how to install BART on the individual operating systems.
