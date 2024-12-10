# 3DGaussianLaplacian

To clone,
```
git clone https://github.com/Zero-4869/3DGaussianLaplacian.git --recursive
```

To install, run 
```
conda env create --file environment.yml
conda activate gaussian_laplacian

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
# tetra-nerf for triangulation
cd submodules/tetra-triangulation
conda install cmake
conda install conda-forge::gmp
conda install conda-forge::cgal
cmake .
# you can specify your own cuda path
# export CPATH=/usr/local/cuda-11.3/targets/x86_64-linux/include:$CPATH
make 
# Extensions
pip install -e .
cd ../../extensions
pip install .
```