# 3DGaussianLaplacian

'''
To install, run 
conda env create --file environment.yml

# tetra-nerf for triangulation
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
pip install -e .
'''