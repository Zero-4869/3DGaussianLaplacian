# 3DGaussianLaplacian

To clone,
```
git clone https://github.com/Zero-4869/3DGaussianLaplacian.git --recursive
```

To install, run 
```
cd 3DGaussianLaplacian
conda env create --file environment.yml
conda activate gaussian_laplacian

pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 -f https://download.pytorch.org/whl/torch_stable.html

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
# tetra-nerf for triangulation
cd submodules/tetra-triangulation
 
# Extensions
cd extensions
pip install .
```