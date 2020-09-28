# CNN Subgraph Visualizer

## Setup

First things first for all setup methods is to clone this repo:

`git clone https://github.com/chrishamblin7/cnn_subgraph_visualizer.git`


### Docker
The recommended way of setting up this cnn visualizer tools environment is to use docker. If you dont have docker installed on your computer, you can [download it here](https://docs.docker.com/get-docker/)

Once you have docker install you can use docker commandline tools, and get the environment for this project by running:

`docker pull chrishamblin7/cnn_subgraph_visualizer:latest`

That might take a while to download. Once its done you can launch the tool by running:

`docker run -t -v [full/path/to/cloned/repo]:/workspace -p 8050:8050 chrishamblin7/cnn_subgraph_visualizer`

