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

pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu121 (or other versions)

pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn/
# tetra-nerf for triangulation
# cd submodules/tetra-triangulation
 
# Extensions
cd extensions
pip install .
```

To run the demo
```
python demo.py --path <path to the .ply file>
```