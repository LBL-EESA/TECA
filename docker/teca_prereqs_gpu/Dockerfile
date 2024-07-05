# Use TECA's prerequisites to build a Docker image for TECA's CPU version
FROM obrientaiu/teca_prereqs:cpu

ARG distro=ubuntu2404
ARG arch=x86_64
ARG keyring_file=cuda-keyring_1.1-1_all.deb

# install the NVIDIA GPG key
RUN cd /tmp/ && \
    wget https://developer.download.nvidia.com/compute/cuda/repos/$distro/$arch/$keyring_file && \
    dpkg -i $keyring_file && \
    rm $keyring_file

# Install cudatoolkit
RUN apt-get update -qq && apt-get install -qq -y cuda-toolkit && rm -rf /var/lib/apt/lists/*

# Setup virtual environment and install Python packages
# clean up pip's cache afterward
RUN pip3 install --break-system-packages \
    cupy-cuda12x \
    torch \
    torchvision \
    torchaudio && \
    rm -rf /root/.cache/pip

# Set environment variables to make CUDA Toolkit binaries available in the PATH
ENV PATH=/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}