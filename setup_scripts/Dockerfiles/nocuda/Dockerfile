#Dockerfile for cnn_subgraph_visualizer
FROM nvidia/cuda:10.2-base-ubuntu18.04

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    nano \
 && rm -rf /var/lib/apt/lists/*

# Create a working directory
RUN mkdir /app
WORKDIR /app

# Create a non-root user and switch to it
RUN adduser --disabled-password --gecos '' --shell /bin/bash user \
 && chown -R user:user /app
RUN echo "user ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-user
USER user

# All users can use /home/user as their home directory
ENV HOME=/home/user
RUN chmod 777 /home/user

# Install Miniconda and Python 3.8
ENV CONDA_AUTO_UPDATE_CONDA=false
ENV PATH=/home/user/miniconda/bin:$PATH
RUN curl -sLo ~/miniconda.sh https://repo.continuum.io/miniconda/Miniconda3-py38_4.8.2-Linux-x86_64.sh \
 && chmod +x ~/miniconda.sh \
 && ~/miniconda.sh -b -p ~/miniconda \
 && rm ~/miniconda.sh \
 && conda install -y python==3.8.1 \
 && conda clean -ya

# No CUDA-specific steps
ENV NO_CUDA=1
RUN conda install -y -c pytorch \
    cpuonly \
    "pytorch=1.5.0=py3.8_cpu_0" \
    "torchvision=0.6.0=py38_cpu" \
 && conda clean -ya

#Dash and other app specific packages
RUN conda install -y -c anaconda \
    pandas=1.0.5 \ 
    pillow=7.1.2 \
    scikit-learn=0.23.1 \ 
  && conda install -y -c plotly \
     plotly=4.8.2 \
  && conda install -y -c conda-forge \
     dash=1.13.4 \
     jupyterlab=2.2.0 \
  && conda clean -ya

RUN pip install googledrivedownloader==0.4


#pull project files from github
RUN git clone https://github.com/chrishamblin7/cnn_subgraph_visualizer.git && cd cnn_subgraph_visualizer && git checkout 27dc118     #Revision make sure to keep this up to date

#download extra files from gdrive
RUN cd cnn_subgraph_visualizer/setup_scripts && python download_from_gdrive.py

