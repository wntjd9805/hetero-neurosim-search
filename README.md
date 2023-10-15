# hetero_neurosim_search
Hetero_neurosim_search is an end-to-end design automation tool for neuromorphic architectures by incorporating heterogeneous tile/PE/SA sizes. For full details, please see our recent ICCAD 2023 paper.
If you use this tool in your research, please cite : 
## Pre-requisites

### System dependencies

We tested our code on Ubuntu 18.04 amd64 system. We recommend using the Nvidia docker image.

```jsx
docker pull nvidia/cuda:11.0.3-devel-ubuntu18.04
```

### Software dependencies

Software pre-requisites for installing from the source should be satisfied for the following repositories:

- [DNN_NeuroSim_V1.3](https://github.com/neurosim/DNN_NeuroSim_V1.3.git)
- [Booksim2](https://github.com/booksim/booksim2.git)
- [TVM](https://github.com/apache/tvm)

We extended NeuroSim 1.3 and BookSim2 to simulate the ReRAM-based analog CiM architecture with a mesh interconnect. Please the follow this documentation to install all dependencies

In Ubuntu 18.04 amd64 system, following commands install package dependencies:

```jsx
apt-get update
apt-get install wget gcc libtinfo-dev zlib1g-dev build-essential cmake libedit-dev libxml2-dev git python3-pip aria2 flex bison curl
```

Next, install Python (== 3.6) dependencies.

Note: you need to use specific version of the library. See requirements.txt

```jsx
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
conda create -n neurosim python=3.6.9
conda activate neurosim
pip install -r requirments.txt
```

Install CMake (>= 3.18):

```bash
aria2c -q -d /tmp -o cmake-3.21.0-linux-x86_64.tar.gz  https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz
tar -zxf /tmp/cmake-3.21.0-linux-x86_64.tar.gz --strip=1 -C /usr

```

Install Clang and LLVM

```jsx
wget -c https://releases.llvm.org/9.0.0/clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
tar -xvf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
cp -rl clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04/* /usr/local
rm -rf clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04 \
       clang+llvm-9.0.0-x86_64-linux-gnu-ubuntu-18.04.tar.xz
```

Install rust and pueue

```jsx
curl --proto '=https' --tlsv1.3 https://sh.rustup.rs -sSf | sh
cargo install pueue
```

## Setup

Install and build hetero_neurosim_search repositories from the source. We prepared the installation script (docker/install.sh):

```bash
cd "$HOME"
git clone --recursive https://github.com/wntjd9805/hetero-neurosim-search.git
git clone --recursive https://github.com/apache/tvm tvm
git clone https://github.com/Glitchfix/TOPSIS-Python.git
mkdir tvm/build
cp hetero-neurosim-search/config.cmake tvm/build
cd tvm/build
cmake ..
make -j 8
cd "$HOME"
cd hetero-neurosim-search/booksim2/src
make -j 8
cd "$HOME"
cd hetero-neurosim-search/Inference_pytorch/NeuroSIM
make -j 8
```

Now, the directory should look like this:

```
. ($HOME)
./hetero-neurosim-search
./hetero-neurosim-search/booksim2
```

Finally, you need to set the following environment variables and include them to .bashrc for later session.

```bash
export TVM_HOME=/root/tvm
export PYTHONPATH=/root/tvm/python
```

## Run

Define the search space in search_space.txt.

```jsx
sa_set = 64,128,256
pe_max = 16
tile_max = 32
```

Make weight file and execute Neurosim

```bash
sh run_neurosim.sh Mobilenet_V2
# model : **['ResNet50','EfficientB0','Mobilenet_V2']**
```

### Homogeneous architecture search

```bash
pueued -d #launch pueue to use multiple cores
pueue parallel 8 #NCPU
python search_homogeneous.py --model=Mobilenet_V2
#Wait for the pueue tasks to finish. You can get the status via the pueue command.
python topsis_homogeneous.py --model=Mobilenet_V2
```

### Heterogeneous architecture search

```bash
pueued -d #launch pueue to use multiple cores
pueue parallel 8 #NCPU
python search_heterogeneous.py --model=Mobilenet_V2 --beam_size_m=500 --beam_size_n=3 --weight_latency=1 --weight_power=1 --weight_area=1
#Wait for the pueue tasks to finish. You can get the status via the pueue command.

python topsis_heterogeneous.py --model=Mobilenet_V2 --beam_size_m=500 --beam_size_n=3 --weight_latency=1 --weight_power=1 --weight_area=1
# --constrain_latency, --constrain_power, --constrain_area: you can constraints on performance using this option
```

output:

```bash
best_similarity  0.08165670676847116
--------0--------
21499192.25358721 205.69140070358782 23174539.013919998
['SA_row:128_SA_col:128_PE:2_TL:4']
--------1--------
8659548.5383368 104.40471143460402 121531758.3005
['SA_row:128_SA_col:128_PE:2_TL:16']
```

