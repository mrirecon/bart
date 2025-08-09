# Singularity Container for BART

Running BART on a high performance computing (HPC) cluster can be challenging due to missing libraries.
One solution exploits containers. They encapsulate software with all their dependencies.
Here, we provide some basic information about how to run BART in a container using [Singularity](https://sylabs.io/singularity/).
The definition files have been tested with Singularity-CE in version 3.8.0.

A blueprint to create a singularity container for BART can be found in the definition files
* [`bart_ubuntu.def`](bart_ubuntu.def): for an Ubuntu 22.04 operating system. 
* [`bart_debian.def`](bart_debian.def): for a Debian 12 (bookworm) operating system.

After installing singularity, a container `container.sif` can be created in the Singularity Image Format (SIF):
```code
sudo singularity build container.sif bart.def
```

Both containers download and compile BART with all libraries including GPU support using CUDA.
Make sure to select the CUDA version your local HPC host provides.
The definition files above also represent simple guides of how to install BART on the individual operating systems.

### Run Scripts in Singularity Containers

To run a container load the required `singularity` and potentially `cuda` module on your HPC:
```bash
module load singularity
module load cuda
```
After transferring the container to BigPurple a script `recon.sh` can be executed within the container using `bash`:
```bash
singularity exec --nv container.sif bash 'recon.sh'
```
`--nv` allows the container to access the Nvidia drivers on the host system and is mandatory for running BART with GPU support.

### Run BART from the CLI inside a Singularity Container
Using the singularity image file, an interactive session can be started to run BART's command line tools:
```bash
singularity shell --nv container.sif
```

## Mounting Directories
It is convenient to mount directories in the singularity container to access executables or files from within the container.
This can be done by adding `--bind`. To mount the directory `/tmp` from the host to the location `/tmp` within the container.
```bash
singularity shell --nv --bind /tmp:/tmp container.sif
```

## Example
In this short example, we use a container `container.sif` on our HPC and compile and run a local BART version.

##### 1. Download BART
On the host, we download the newest master branch version of BART into the `/tmp` directory:
```bash
cd /tmp
git clone https://github.com/mrirecon/bart.git
```
and enter it
```bash
cd bart
```

##### 2. Create local Makefile for GPU Compilation
To compile BART with GPU support, we have to set the environment variable `CUDA=1`.
While this could be set during compilation
```bash
CUDA=1 make
```
the recommended solution is to just create a local Makefile
```bash
touch Makefiles/Makefile.local
printf "PARALLEL=1\nCUDA=1\nCUDA_BASE=/usr/local/cuda\nCUDA_LIB=lib64\n" > Makefiles/Makefile.local
```
With `CUDA_BASE=/usr/local/cuda` and `CUDA_LIB=lib64` you can define the local CUDA paths and with `PARALLEL=1` the compilation is performed in parallel.

##### 3. Create Interactive Session Inside Container
Next, the `singularity` and `cuda` modules are loaded in the host session on the HPC
```bash
module load singularity
module load cuda
```
and an interactive session within the container can be started
```bash
singularity shell --nv --bind /tmp:/tmp <path-to-container>/container.sif
```
Per default the session is started in the directory from which the `singularity` call has been executed.
Please make sure that you are already in the `/tmp/bart` folder
```bash
pwd
```

##### 4. Compile BART
To compile BART execute
```bash
make
```

##### 5. Run BART with GPU Support
The singularity container contains a default BART version. To make sure the newly compiled version is used in your scripts save the path of the executable and update the `$PATH` variable
```bash
BART="$(pwd)"/bart
export PATH=$BART:$PATH
```

Finally, BART can be started in the interactive container session
```bash
bart
```
and a short test verifies that the GPU support is provided:
```bash
# Generate k-space-based phantom data
bart traj trajectory
bart phantom -k -t trajectory kspace

# Perform an nuFFT on the GPU
bart nufft -g trajectory kspace reconstruction
```

### Note
The definition files above also represent simple guides of how to install BART on the individual operating systems.
