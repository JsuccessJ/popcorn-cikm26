# the versions of torch and torchtext must be matched (https://pypi.org/project/torchtext)
# the CUDA version must be matched with torch-scatter (https://github.com/rusty1s/pytorch_scatter)
TORCH_VERSION=2.1.0
TORCHTEXT_VERSION=0.16.0
TORCH_SCATTER_VERSION=2.1.2
CUDA_VERSION=cu118

# Disable user site-packages to force installation in conda environment
export PYTHONNOUSERSITE=1

python -m pip install torch==${TORCH_VERSION} --extra-index-url https://download.pytorch.org/whl/${CUDA_VERSION}
python -m pip install torchtext==${TORCHTEXT_VERSION}
python -m pip install nltk
python -m pip install numpy
python -m pip install scikit-learn
python -m pip install torch-scatter==${TORCH_SCATTER_VERSION} --no-build-isolation -f https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_VERSION}.html
