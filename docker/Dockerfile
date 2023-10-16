# Copyright (c) 2023 Boston Dynamics AI Institute LLC. All rights reserved.

FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
      && apt-get install --no-install-recommends -qy curl apt-transport-https gnupg \
      sudo ca-certificates git software-properties-common \
      build-essential \
      libgl1-mesa-dev libjpeg-dev libglm-dev libgl1-mesa-glx \
      libegl1-mesa-dev mesa-utils xorg-dev freeglut3-dev \
      python3.10-dev python3-pip python-is-python3 \
      && rm -rf /var/lib/apt/lists/* \
      && apt-get clean all

# Install the required Python packages
RUN pip install torch>=1.10.1 \
    numpy>=1.22.4 \
    flask>=2.3.2 \
    seaborn>=0.12.2 \
    open3d>=0.17.0 \
    transformers==4.26.0 \
    salesforce-lavis>=1.0.2 \
    pre-commit>=3.1.1 \
    pytest>=7.2.1 \
    pytest-cov>=4.0.0 \
    habitat-baselines==0.2.420230405 \
    habitat-lab==0.2.420230405 \
    bosdyn-client>=3.3.2 \
    bosdyn-api>=3.3.2 \
    six>=1.16.0

RUN pip install git+https://github.com/naokiyokoyama/frontier_exploration.git \
                git+https://github.com/ChaoningZhang/MobileSAM.git \
                git+https://github.com/naokiyokoyama/depth_camera_filtering \
                git+https://github.com/facebookresearch/habitat-sim.git@v0.2.4 \
                git+https://github.com/naokiyokoyama/bd_spot_wrapper.git \
                git+https://github.com/IDEA-Research/GroundingDINO.git
