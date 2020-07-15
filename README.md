# CNN Subgraph Visualizer

## Setup

Once the repository is cloned, the easiest way to set things up is by running:

`python setup.py`

Then answer the prompts to set up your python environment and download supplementary files too large for github storage.

`setup.py` will setup a python environment using virtualenv, but if you prefer to  use conda, you can do so with:

`conda env create -f setup_scripts/subgraph_visualizer_environment.yml` 

If using conda, download supplementary files by running the following while in your conda environment:

`python setup_scripts/download_from_gdrive.py`

If setup.py fails to set up your environment for any reason, try setting it up manually using `subgraph_visualizer_pip_requirements.txt` as the requirements file.

## Launching the Visualizer Tool 
With your setup virtual environment activated, run:

`python launch_visualizer.py [NAME OF PREPPED MODEL]`

Where [NAME OF PREPPED MODEL] is a subfolder of `/prepped models`. For example to visualize a cifar10 model downloaded during setup, run:

`python launch_visualizer.py cifar10_prunned`

Once a bunch of data is loaded into the visualizer, you should see something like `* Running on http://127.0.0.1:8050/ (Press CTRL+C to quit)` appear in your stdout. Following the url will take you to the visualizer of your model.

## Prepping your own Pytorch Models for Visualization

Prepping your own model requires 3 things:
  * A loadable pytorch model, with a feed-forward cnn architecture
  * An 'input_images' folder, containing images you would like to be able to feed to the model during visualization
  * A 'rank_images' folder, containing subfolders for each subgraph category you'd like to visualize (such as the label categories for a classifier). Each category subfolder should have at least a few hundred examples.
  Look in `image_data/cifar10` for examples of each of these image folders.
  
Edit the `prep_model_parameters.py` file, reading it carefully as you go. Primarily you must load (however you see fit) your pytorch model into the `model` variable of this file. 
Additionally you must specify the 'input_img_path' and 'rank_img_path', pointing to the 'input_images' and 'rank_images' folders described above. These folders can be anywhere, but in keeping with the preloaded models its best if they go in a subfolder of the 'image_data' folder. If you used a `torchvision` transform to preprocess your images before feeding them to your model, assign the transform to the `preprocess` variable in this file. The output_folder variable specifies the name of the 'prepped_models' folder to be generated, and determines the argument that will be passed to `launch_visualizer.py` when visualizing. There are other parameters you can set in this parameter file as well.

Once you've edited `prep_model_parameters.py` run:

`python prep_model.py`

to create a new prepped model in the `prepped_models/` folder. The script will copy the version of `prep_model_parameters.py` used into the generated folder.
