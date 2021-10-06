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
  && conda install -y -c plotly \
     plotly=4.8.2 \
  && conda install -y -c conda-forge \
     dash=1.13.4 \
     flask-caching=1.7.1 \
     jupyterlab=2.2.0 \
     umap-learn \
  && conda clean -ya

RUN pip install torch-lucent
#RUN pip install googledrivedownloader==0.4


#pull project files from github
#RUN git clone https://github.com/chrishamblin7/cnn_subgraph_visualizer.git && cd cnn_subgraph_visualizer && git checkout d1d7acf     #Revision make sure to keep this up to date

#download extra files from gdrive
#RUN cd cnn_subgraph_visualizer/setup_scripts && python download_from_gdrive.py