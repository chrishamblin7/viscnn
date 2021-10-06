#Dockerfile for cnn_subgraph_visualizer
FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-runtime

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



 #Dash and other app specific packages
RUN conda install -y -c anaconda \
    pandas=1.0.5 \ 
    pillow=8.1.2 \
    scikit-learn=0.23.1 \ 
    sympy=1.6.2 \
    ipykernel=5.3.4 \
  && conda install -y -c plotly \
     plotly=4.14.3 \
  && conda install -y -c conda-forge \
     flask-caching=1.10.1 \
     dash=1.20.0 \
     dash-daq=0.5.* \
     jupyterlab=3.0.16 \
     umap-learn=0.5.1 \
     hdbscan=0.8.27 \ 
  && conda clean -ya

RUN pip install torch-lucent
#RUN pip install googledrivedownloader==0.4


#pull project files from github
#RUN git clone https://github.com/chrishamblin7/cnn_subgraph_visualizer.git && cd cnn_subgraph_visualizer && git checkout d1d7acf     #Revision make sure to keep this up to date

#download extra files from gdrive
#RUN cd cnn_subgraph_visualizer/setup_scripts && python download_from_gdrive.py