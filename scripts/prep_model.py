import os
from subprocess import call

#kernels
print('pulling model convolutional kernels . . .')
call(['python', 'get_kernels.py'])

#activation maps
print('getting node and edge activation maps for input images')
call(['python','get_activation_maps_for_input_images.py'])

#ranks
print('getting node and edge subgraph importance ranks')
call(['python','get_ranks_for_all_classes.py'])




