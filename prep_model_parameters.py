### PARAMETER FILE ###
import torch
import os


###MODEL

#There are a lot of ways to load up a model in pytorch, just do whatever you need to do here such that there is a variable 'model' in this file pointing to a working feed-forward CNN
#from model_classes import AlexNet_format
#model = AlexNet_format()
#model.load_state_dict(torch.load('./models/alexnet_statedict.pt'))
from model_classes import MNIST_edges
model = MNIST_edges()
model.load_state_dict(torch.load('models/mnist_blurryedges_ep39_0.992_statedict.pt'))

###IMAGE PATHS

input_img_path =  './image_data/mnist_data_loading/input_images'   #Set this to the system path for the folder containing input images you would like to see network activation maps for.
rank_img_path = './image_data/mnist_data_loading/ranking_images'       #Set this to a path with subfolders, where each subfolder contains a set of images. Subgraph ranks will be based on these subfolders. 

label_file_path = './image_data/mnist_data_loading/labels.txt'      #line seperated file with names of label classes as they appear in image names
						  #set to None if there are no target classes for your model
						  #make sure the order of labels matches the order in desired target vectors
 

output_folder = 'mnist_grad_test'     #name of folder you want prep model to output to. Not a path here, just a name, it will appear as a folder under prepped_models/. 
									  #When you launch the visualization tool you will do so with respect to this folder name

###IMAGE PREPROCESSING

from torchvision import transforms

#preprocess = None     # Set this to None or False if you dont want your input images to be preprocessed. Or set this to a torchvision transform, as in the example below. If you use
					  # a transform to load your test data set when training your model, put that in below.

preprocess =  transforms.Compose([
                    transforms.Grayscale(num_output_channels=1), 
                    transforms.ToTensor()
                             	 ])


#GPU
cuda = True       #use GPU acceleration

#LOSS
criterion = torch.nn.CrossEntropyLoss()   #this should be set to whatever loss function was used to train your model

#MEMORY
save_activations=True    #If getting activations for the input images is causing memory problems, set this to False. Instead of
				      		#saving a file with all the model activations, the model will run in the background of the visualizer, fetching
					  		#activations by running the model forward on an 'as need' basis.

#AUX (these params arent super important but you might want to change them)
num_workers = 4     #num workers argument in dataloader
seed = 2            #manual seed
batch_size = 300    #batch size for feeding rank image set through model (input image set is sent through all at once)






#Clean up (YOU DONT NEED TO EDIT ANYTHING BELOW THIS LINE)

input_img_path =  os.path.abspath(input_img_path)
rank_img_path = os.path.abspath(rank_img_path)
label_file_path = os.path.abspath(label_file_path)