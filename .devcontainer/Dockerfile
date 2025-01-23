# Base image
FROM nvidia/cuda:12.2.0-base-ubuntu22.04

# Set noninteractive environment for apt
ENV DEBIAN_FRONTEND=noninteractive
ENV LANG en_US.UTF-8

# Update and install system-related packages
RUN apt-get update && apt-get install -y \
    locales \
    cmake \
    make \
    nano \
    vim \
    curl \
    git \
    pip \
    software-properties-common \
    build-essential \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    libzmq3-dev \
    '^libxcb.*-dev' \
    libx11-dev \
    libx11-xcb-dev \
    libxi-dev \
    libxrender-dev \
    libxkbcommon-dev \
    libxkbcommon-x11-dev \
    libglu1-mesa-dev \
    python3-tk \
    libglfw3-dev \
    kbd \
    && rm -rf /var/lib/apt/lists/*

# Configure locale
RUN localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8

# Create a non-root user
ARG USERNAME=atari
ARG USER_UID=1000
ARG USER_GID=1000

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID --create-home --shell /bin/bash $USERNAME \
    && mkdir -p /home/$USERNAME/.config \
    && chown -R $USER_UID:$USER_GID /home/$USERNAME

# Install sudo and grant sudo permissions
RUN apt-get update \
    && apt-get install -y sudo \
    && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && rm -rf /var/lib/apt/lists/*

# Switch to the non-root user
USER $USERNAME

# User permissions
RUN sudo chown -R $USERNAME:$USERNAME /home/$USERNAME/

# Create workspace directory
RUN mkdir -p /home/$USERNAME/workspace/
WORKDIR /home/$USERNAME/workspace/

# Install Miniconda
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3.sh && \
    bash Miniconda3.sh -b -p /home/$USERNAME/miniconda && \
    rm Miniconda3.sh
ENV PATH="/home/$USERNAME/miniconda/bin:$PATH"

# Initialize conda for bash
RUN /home/$USERNAME/miniconda/bin/conda init bash

# Create and activate the Conda environment
COPY ./environment.yml .

# Add conda-forge channel
RUN /home/$USERNAME/miniconda/bin/conda config --add channels conda-forge
RUN /home/$USERNAME/miniconda/bin/conda install conda-build
RUN bash -c "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && \
    conda env create -n atari_mpc -f environment.yml python=3.12"
RUN rm environment.yml

# Ensure the environment is always activated
RUN echo "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && conda activate atari_mpc" >> ~/.bashrc

# Set the working directory
WORKDIR /home/$USERNAME/

# Clone and install acados with Python interface
RUN git clone https://github.com/acados/acados.git && \
    cd acados && \
    git submodule update --recursive --init && \
    mkdir -p build && cd build && \
    cmake -DACADOS_WITH_QPOASES=ON .. && \
    make install -j$(nproc) && \
    bash -c "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && conda activate atari_mpc && \
    conda develop ../interfaces/acados_template"
ENV LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/home/$USERNAME/acados/lib"
ENV ACADOS_SOURCE_DIR="/home/$USERNAME/acados"

# Clone and install mj_pin_utils
RUN git clone https://github.com/Atarilab/mj_pin_utils.git && \
    bash -c "source /home/$USERNAME/miniconda/etc/profile.d/conda.sh && conda activate atari_mpc && \
    conda develop ./mj_pin_utils"

# Set the final working directory
WORKDIR /home/$USERNAME/workspace/

CMD ["bash"]