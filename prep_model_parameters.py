### PARAMETER FILE ###
import torch
import os
import sys


output_folder = 'letter_mixed_not_trained'     #name of folder you want prep model to output to. Not a path here, just a name, it will appear as a folder under prepped_models/. 
									  #When you launch the visualization tool you will do so with respect to this folder name

###MODEL

#There are a lot of ways to load up a model in pytorch, just do whatever you need to do here such that there is a variable 'model' in this file pointing to a working feed-forward CNN
#from model_classes import AlexNet_format
#model = AlexNet_format()
#model.load_state_dict(torch.load('./models/alexnet_statedict.pt'))

#sys.path.insert(0, os.path.abspath('./model_scripts/'))
#from model_classes import MNIST_edges
#model = MNIST_edges()
#model.load_state_dict(torch.load('models/mnist_statedict.pt'))
#model = torch.load('../cnn_letter_rsa/models/mixednet/alexnet_mixed_letters_normed_ep1_imnet0.466_letter0.955.pt')
from torchvision import models
import torch.nn as nn
model = models.alexnet(pretrained=True) #load a model from the models.py script, or switch this to torch.load(path_to_model) to load a model from a .pt file   

num_ftrs = model.classifier[6].in_features
letter_linear = nn.Linear(num_ftrs,26)
new_classifier_weights = torch.cat((model.classifier[6].weight, letter_linear.weight), 0)
new_classifier_bias = torch.cat((model.classifier[6].bias, letter_linear.bias), 0)
new_classifier = nn.Linear(num_ftrs,1026)
new_classifier.weight = torch.nn.Parameter(new_classifier_weights)
new_classifier.bias = torch.nn.Parameter(new_classifier_bias)
model.classifier[6] = new_classifier

###IMAGE PATHS

input_img_path =  './image_data/imagenet_200/input_images'   #Set this to the system path for the folder containing input images you would like to see network activation maps for.
rank_img_path = './image_data/mixed10/ranking_images'       #Set this to a path with subfolders, where each subfolder contains a set of images. Subgraph ranks will be based on these subfolders. 

label_file_path = './image_data/mixed10/labels.txt'      #line seperated file with names of label classes as they appear in image names
						  #set to None if there are no target classes for your model
						  #make sure the order of labels matches the order in desired target vectors
 

###IMAGE PREPROCESSING

from torchvision import transforms

#preprocess = None     # Set this to None or False if you dont want your input images to be preprocessed. Or set this to a torchvision transform, as in the example below. If you use
					  # a transform to load your test data set when training your model, put that in below.

preprocess =  transforms.Compose([
        						transforms.Resize((224,224)),
        						transforms.ToTensor(),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     	 			 std=[0.229, 0.224, 0.225])])


#GPU
cuda = True       #use GPU acceleration

#LOSS
criterion = torch.nn.CrossEntropyLoss()   #this should be set to whatever loss function was used to train your model

#MEMORY
save_activations=False    #If getting activations for the input images is causing memory problems, set this to False. Instead of
				      		#saving a file with all the model activations, the model will run in the background of the visualizer, fetching
					  		#activations by running the model forward on an 'as need' basis.

#AUX (these params arent super important but you might want to change them)
num_workers = 4     #num workers argument in dataloader
seed = 2            #manual seed
batch_size = 10    #batch size for feeding rank image set through model (input image set is sent through all at once)



#Clean up (YOU DONT NEED TO EDIT ANYTHING BELOW THIS LINE)

input_img_path =  os.path.abspath(input_img_path)
rank_img_path = os.path.abspath(rank_img_path)
label_file_path = os.path.abspath(label_file_path)