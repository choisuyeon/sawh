# Install dependencies. Here's an example that worked for me.
# Start with a fresh conda environment
conda create -n sawh python=3.10
conda activate sawh
pip install hydra-core --upgrade
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
pip install tqdm imageio opencv-python pytorch-lightning matplotlib omegaconf scipy scikit-learn scikit-image


# install tiny-cuda-nn
sudo apt-get install build-essential git
export PATH="/usr/local/cuda-12.6/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH"
git clone --recursive https://github.com/nvlabs/tiny-cuda-nn
cd tiny-cuda-nn
cmake . -B build -DCMAKE_BUILD_TYPE=RelWithDebInfo
cmake --build build --config RelWithDebInfo -j
cd bindings/torch
python setup.py install