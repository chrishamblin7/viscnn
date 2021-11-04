from collections import OrderedDict


def show_model_layer_names(model, getLayerRepr=False):
    """
    If getLayerRepr is True, return a OrderedDict of layer names, layer representation string pair.
    If it's False, just return a list of layer names
    """
    
    layers = OrderedDict() if getLayerRepr else []
    conv_linear_layers = []
    # recursive function to get layers
    def get_layers(net, prefix=[]):
        
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                if getLayerRepr:
                    layers["_".join(prefix+[name])] = layer.__repr__()
                else:
                    layers.append("_".join(prefix + [name]))
                
                if isinstance(layer, nn.Conv2d):
                    conv_linear_layers.append(("_".join(prefix + [name]),'  conv'))
                elif isinstance(layer, nn.Linear):
                    conv_linear_layers.append(("_".join(prefix + [name]),'  linear'))
                    
                get_layers(layer, prefix=prefix+[name])
                
    get_layers(model)
    
    print('All Layers:\n')
    for layer in layers:
        print(layer)

    print('\nConvolutional and Linear layers:\n')
    for layer in conv_linear_layers:
        print(layer)




import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from math import log, exp, ceil





def ref_name_modules(net):
    
    # recursive function to get layers
    def name_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                
                layer.ref_name = "_".join(prefix + [name])
                
                name_layers(layer,prefix=prefix+[name])

    name_layers(net)

    
    
def get_last_layer_from_feature_targets(net, feature_targets):
    
    
    #get dict version of feature targets
    feature_targets = feature_targets_list_2_dict(feature_targets)
    target_layers =  feature_targets.keys()      
    
    
    def check_layers(net,last_layer=None):
        if hasattr(net, "_modules"):
            for name, layer in reversed(net._modules.items()):  #use reversed to start from the end of the network

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if layer.ref_name in target_layers:
                        last_layer = layer.ref_name
                        break
        
                        
                last_layer = check_layers(layer)
        
        return last_layer

           
    last_layer = check_layers(net)    

    if last_layer is None:
        print('No effective last layer found!')
    else:
        print('%s effective last layer'%last_layer)
        
    return last_layer
    

 
def feature_targets_list_2_dict(feature_targets,feature_targets_coefficients=None):
    if isinstance(feature_targets_coefficients,list):
        feature_targets_coefficients_ls = feature_targets_coefficients
        feature_targets_coefficients = {}
    
    #get dict version of feature targets
    if isinstance(feature_targets,list):
        feature_targets_ls = feature_targets
        feature_targets = {}
        for i,feature_conj in enumerate(feature_targets_ls):
            layer, feature = feature_conj.split(':')
            if layer.strip() in feature_targets:
                feature_targets[layer.strip()].append(int(feature.strip()))
            else:
                feature_targets[layer.strip()] = [int(feature.strip())]
            if feature_targets_coefficients is not None:
                if layer.strip() in feature_targets_coefficients:
                    feature_targets_coefficients[layer.strip()].append(feature_targets_coefficients_ls[i])
                else:
                    feature_targets_coefficients[layer.strip()] = [feature_targets_coefficients_ls[i]]
                
    assert isinstance(feature_targets,dict)
    
    if feature_targets_coefficients is None:
        return feature_targets
    else:
        return feature_targets, feature_targets_coefficients
    


def setup_net_for_circuit_prune(net, feature_targets=None,save_target_activations=False,rank_field = 'image'):
    
    assert rank_field in ('image','min','max')
    
    #name model modules
    ref_name_modules(net)
    
    
    last_layer = None
    #get dict version of feature targets
    if feature_targets is not None:
        feature_targets = feature_targets_list_2_dict(feature_targets)
    
        #get effective last_layer
        last_layer = get_last_layer_from_feature_targets(net, feature_targets)
        
    
    def setup_layers(net):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if isinstance(layer, nn.Conv2d):
                    layer.save_target_activations = save_target_activations
                    layer.target_activations = {}
                    
                    layer.last_layer = False
                    if layer.ref_name == last_layer:
                        layer.last_layer = True
                        
                    
                    layer.rank_field = rank_field
                    layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                    
                    layer.feature_targets_indices = None
                    if feature_targets is not None:
                        if layer.ref_name in feature_targets: #layer has feature targets in it
                            layer.feature_targets_indices = feature_targets[layer.ref_name]

                    #setup masked forward pass
                    layer.forward = types.MethodType(circuit_prune_forward_conv2d, layer)

                elif isinstance(layer, nn.Linear):
                    layer.save_target_activations = save_target_activations
                    layer.target_activations = {}
                    
                    layer.last_layer = False
                    if layer.ref_name == last_layer:
                        layer.last_layer = True
                    
                    layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
                    
                    layer.feature_targets_indices = None
                    if feature_targets is not None:
                        if layer.ref_name in feature_targets: #layer has feature targets in it
                            layer.feature_targets_indices = feature_targets[layer.ref_name]

                    #setup masked forward pass
                    layer.forward = types.MethodType(circuit_prune_forward_linear, layer)


                setup_layers(layer)

           
    setup_layers(net)
    

    
#Error classes for breaking forward pass of model
# define Python user-defined exceptions
class ModelBreak(Exception):
	"""Base class for other exceptions"""
	pass

class TargetReached(ModelBreak):
	"""Raised when the output target for a subgraph is reached, so the model doesnt neeed to be run forward any farther"""
	pass    
    

    
def circuit_prune_forward_conv2d(self, x):
        
    #pass input through conv and weight mask
    x = F.conv2d(x, self.weight * self.weight_mask, self.bias,
                    self.stride, self.padding, self.dilation, self.groups) 
#     x = F.conv2d(x, self.weight, self.bias,
#                     self.stride, self.padding, self.dilation, self.groups) 


    self.target_activations = {}
    
    #gather feature targets
    if self.feature_targets_indices is not None: #there are feature targets in the conv 
        self.feature_targets = {}

        for feature_idx in self.feature_targets_indices:
            if self.rank_field == 'image':
                avg_activations = x.mean(dim=(0, 2, 3))
                self.feature_targets[feature_idx] = avg_activations[feature_idx]
                
            elif self.rank_field == 'max':
                max_acts = x.view(x.size(0),x.size(1), x.size(2)*x.size(3)).max(dim=-1).values
                max_acts_target = max_acts[:,feature_idx]
                self.feature_targets[feature_idx] = max_acts_target.mean()
                
            elif self.rank_field == 'min':
                min_acts = x.view(x.size(0),x.size(1), x.size(2)*x.size(3)).min(dim=-1).values
                min_acts_target = min_acts[:,feature_idx]
                self.feature_targets[feature_idx] = min_acts_target.mean()
                
            elif isinstance(self.rank_field,list):
                raise Exception('List type rank field not yet implemented, use "min", "max",or "image" as the rank field')
                #target_acts = 
                #optim_target = target_acts.mean()
            #print(optim_target)

            if self.save_target_activations:
                self.target_activations[feature_idx] = x[:,feature_idx,:,:].data.to('cpu')
            
    if self.last_layer: #stop model forward pass if all targets reached
        raise TargetReached
    
    return x
    



def circuit_prune_forward_linear(self, x):
    
    #pass input through weights and weight mask
    x = F.linear(x, self.weight * self.weight_mask, self.bias)
#     x = F.linear(x, self.weight, self.bias)
    
    self.target_activations = {}
    
    #gather feature targets
    if self.feature_targets_indices is not None: #there are feature targets in the conv 
        self.feature_targets = {}

        for feature_idx in self.feature_targets_indices:
            avg_activations = x.mean(dim=(0))
            self.feature_targets[feature_idx] = avg_activations[feature_idx] 
            
            if self.save_target_activations:
                self.target_activations[feature_idx] = x[:,feature_idx,:].data.to('cpu')
    
    if self.last_layer: #stop model forward pass if all targets reached
        raise TargetReached
    
    return x
  
 
def get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=None):
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if feature_targets_coefficients is None:
        feature_targets_indices = feature_targets_list_2_dict(feature_targets)
    else:
        feature_targets_indices,feature_targets_coefficients = feature_targets_list_2_dict(feature_targets,feature_targets_coefficients=feature_targets_coefficients) 
    

    
    def fetch_targets_values(net,feature_targets_values = {}):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if layer.ref_name in feature_targets_indices.keys():
                    feature_targets_values[layer.ref_name] = []
                    for idx in layer.feature_targets:
                        feature_targets_values[layer.ref_name].append(layer.feature_targets[idx])
                        
                feature_targets_values = fetch_targets_values(layer, feature_targets_values = feature_targets_values)
        
        return feature_targets_values
                
    feature_targets_values = fetch_targets_values(net)
    
    target = None
    for layer in feature_targets_values:
        for idx in range(len(feature_targets_values[layer])):
            coeff = 1
            if feature_targets_coefficients is not None:
                coeff = feature_targets_coefficients[layer][idx] 
            
            if target is None:
                target = coeff*feature_targets_values[layer][idx]
            else:
                target += coeff*feature_targets_values[layer][idx]

    return target



def clear_feature_targets_from_net(net):

    
    def clear_layers(net):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    layer.feature_targets = None


                clear_layers(layer)

           
    clear_layers(net)
    

def reset_masks_in_net(net):


    def reset_layers(net):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))


                reset_layers(layer)


    reset_layers(net)
    
    
def save_target_activations_in_net(net,save=True):
    
    def reset_layers(net):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    layer.save_target_activations = save


                reset_layers(layer)


    reset_layers(net)
   


def get_saved_target_activations_from_net(net):

    def fetch_activations(net,target_activations = {}):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():

                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue

                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    if layer.save_target_activations:
                        if layer.target_activations != {}:
                            for idx in layer.target_activations:
                                target_activations[layer.ref_name+':'+str(idx)] = layer.target_activations[idx]

                        
                target_activations = fetch_activations(layer, target_activations = target_activations)
        
        return target_activations
                
    target_activations = fetch_activations(net)
    

    return target_activations


def apply_mask(net,mask):
    count = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if mask is not None and count < len(mask): #we have a mask for these weights 
                layer.weight_mask = nn.Parameter(mask[count])
            else:
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            #nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            count += 1


def make_net_mask_only(net):
    #function keeps net mask but gets rid of other bells and whistles
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            layer.save_target_activations = False
            layer.target_activations = {}
            layer.last_layer = False
            layer.feature_targets_indices = None

    
def circuit_SNIP(net, dataloader, feature_targets = None, feature_targets_coefficients = None, full_dataset = True, keep_ratio=.1, num_params_to_keep=None, device=None, structure='weights', mask=None, criterion= None, setup_net=True,rank_field='image'):
    '''
    if num_params_to_keep is specified, this argument overrides keep_ratio
    '''

    assert structure in ('weights','kernels','filters')    
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
  

    #set up cirterion
    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    #import pdb; pdb.set_trace()

    if setup_net:
        setup_net_for_circuit_prune(net, feature_targets=feature_targets, rank_field = rank_field)
        
    #import pdb; pdb.set_trace() 
    
    #apply current mask
    count = 0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if mask is not None and count < len(mask): #we have a mask for these weights 
                layer.weight_mask = nn.Parameter(mask[count])
            else:
                layer.weight_mask = nn.Parameter(torch.ones_like(layer.weight))
            #nn.init.xavier_normal_(layer.weight)
            layer.weight.requires_grad = False
            count += 1

    
    #import pdb; pdb.set_trace()
    
    #do we iterate through the whole dataset or not
    iter_dataloader = iter(dataloader)
    
    iters = 1
    if full_dataset:
        iters = len(iter_dataloader)
    
    
    grads_abs = [] #computed scores
    
    for it in range(iters):
        clear_feature_targets_from_net(net)
        
        # Grab a single batch from the training dataset
        inputs, targets = next(iter_dataloader)
        inputs = inputs.to(device)
        targets = targets.to(device)




        # Compute gradients (but don't apply them)
        net.zero_grad()
        
        #Run model forward until all targets reached
        try:
            outputs = net.forward(inputs)
        except TargetReached:
            pass
        
        #get proper loss
        if feature_targets is None:
            loss = criterion(outputs, targets)
        else:   #the real target is feature values in the network
            loss = get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=device)
        
        loss.backward()

        #get weight-wise scores
        if grads_abs == []:
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    grads_abs.append(torch.abs(layer.weight_mask.grad))
                    if layer.last_layer:
                        break
        else:
            count = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    grads_abs[count] += torch.abs(layer.weight_mask.grad)        
                    count += 1
                    if layer.last_layer:
                        break
      
    #import pdb; pdb.set_trace()
                
    #structure scoring by weights, kernels, or filters   
    
    if structure == 'weights':
        structure_grads_abs = grads_abs
    elif structure == 'kernels':
        structure_grads_abs = []
        for grad in grads_abs:
            if len(grad.shape) == 4: #conv2d layer
                structure_grads_abs.append(torch.mean(grad,dim = (2,3))) #average across height and width of each kernel
    else:
        structure_grads_abs = []
        for grad in grads_abs:
            if len(grad.shape) == 4: #conv2d layer
                structure_grads_abs.append(torch.mean(grad,dim = (1,2,3))) #average across channel height and width of each filter
        
    #import pdb; pdb.set_trace()
    
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in structure_grads_abs])
    norm_factor = torch.sum(all_scores)
    all_scores.div_(norm_factor)
    
    #get num params to keep
    if num_params_to_keep is None:
        num_params_to_keep = int(len(all_scores) * keep_ratio)
    threshold, _ = torch.topk(all_scores, num_params_to_keep, sorted=True)
    acceptable_score = threshold[-1]

    keep_masks = []

    for g in structure_grads_abs:
        keep_masks.append(((g / norm_factor) >= acceptable_score).float())

    #print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks])))
    #import pdb; pdb.set_trace()
    
    return(keep_masks)

    
    
# def apply_prune_mask(net, keep_masks):

#     # Before I can zip() layers and pruning masks I need to make sure they match
#     # one-to-one by removing all the irrelevant modules:
#     prunable_layers = filter(
#         lambda layer: isinstance(layer, nn.Conv2d) or isinstance(
#             layer, nn.Linear), net.modules())

#     for layer, keep_mask in zip(prunable_layers, keep_masks):
#         assert (layer.weight.shape == keep_mask.shape)

#         def hook_factory(keep_mask):
#             """
#             The hook function can't be defined directly here because of Python's
#             late binding which would result in all hooks getting the very last
#             mask! Getting it through another function forces early binding.
#             """

#             def hook(grads):
#                 return grads * keep_mask

#             return hook

#         # mask[i] == 0 --> Prune parameter
#         # mask[i] == 1 --> Keep parameter

#         # Step 1: Set the masked weights to zero (NB the biases are ignored)
#         # Step 2: Make sure their gradients remain zero
#         layer.weight.data[keep_mask == 0.] = 0.
#         layer.weight.register_hook(hook_factory(keep_mask))
        
        
def expand_structured_mask(mask,net):
    '''Structured mask might have shape (filter, channel) for kernel structured mask, but the weights have
        shape (filter,channel,height,width), so we make a new weight wise mask based on the structured mask'''

    weight_mask = []
    count=0
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if count < len(mask):
                weight_mask.append(mask[count])
                while len(weight_mask[-1].shape) < 4:
                    weight_mask[-1] = weight_mask[-1].unsqueeze(dim=-1)
                weight_mask[-1] = weight_mask[-1].expand(layer.weight.shape)
            count+= 1
    return weight_mask





def circuit_FORCE_pruning(net, dataloader, feature_targets = None,feature_targets_coefficients = None, T=10,full_dataset = True, keep_ratio=.1, num_params_to_keep=None, device=None, structure='weights', rank_field = 'image', mask=None, setup_net=True):    #progressive skeletonization

    
    assert structure in ('weights','kernels','filters')
    assert rank_field in ('image','max','min')
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = net.to(device)
    
    if setup_net:
        setup_net_for_circuit_prune(net, feature_targets=feature_targets, rank_field = rank_field)
    
    
    #get total params given feature target might exclude some of network
    total_params = 0

    for layer in net.modules():
        if structure == 'weights' and (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear)):
            if not layer.last_layer:  #all params potentially important
                total_params += len(layer.weight.flatten())
            else:    #only weights leading into feature targets are important
                total_params += len(layer.feature_targets_indices)*int(layer.weight.shape[1])
                break
        elif isinstance(layer, nn.Conv2d):
            if not layer.last_layer:  #all params potentially important
                if structure == 'kernels':
                    total_params += int(layer.weight.shape[0]*layer.weight.shape[1])
                else:
                    total_params += int(layer.weight.shape[0])
                    
            else: #only weights leading into feature targets are important
                if structure == 'kernels':
                    total_params += int(len(layer.feature_targets_indices)*layer.weight.shape[1])
                else:
                    total_params += len(layer.feature_targets_indices)
                
                break
    
    if num_params_to_keep is None:
        num_params_to_keep = ceil(keep_ratio*total_params)
    else:
        keep_ratio = num_params_to_keep/total_params       #num_params_to_keep arg overrides keep_ratio
    
    print('pruning %s'%structure)
    print('total parameters: %s'%str(total_params))
    print('parameters after pruning: %s'%str(num_params_to_keep))
    print('keep ratio: %s'%str(keep_ratio))
  
    if num_params_to_keep >= total_params:
        print('num params to keep > total params, no pruning to do')
        return

    print("Pruning with %s pruning steps"%str(T))
    for t in range(1,T+1):
        
        print('step %s'%str(t))
        
        k = ceil(exp(t/T*log(num_params_to_keep)+(1-t/T)*log(total_params))) #exponential schedulr
         
        print('%s params'%str(k))
        
        #SNIP
        struct_mask = circuit_SNIP(net, dataloader, num_params_to_keep=k, feature_targets = feature_targets, feature_targets_coefficients = feature_targets_coefficients, structure=structure, mask=mask, full_dataset = full_dataset, device=device,setup_net=False)
        if structure is not 'weights':
            mask = expand_structured_mask(struct_mask,net) #this weight mask will get applied to the network on the next iteration
        else:
            mask = struct_mask
    apply_mask(net,mask)

    return struct_mask
        

def snip_scores(net,dataloader, feature_targets = None, feature_targets_coefficients = None, full_dataset = True, device=None, structure='weights', criterion= None, setup_net=False):
    ###Net should be a preset up, masked model

    
    assert structure in ('weights','kernels','filters')    
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    ##Calculate Scores
    
    #do we iterate through the whole dataset or not
    iter_dataloader = iter(dataloader)
    
    iters = 1
    if full_dataset:
        iters = len(iter_dataloader)
    
    
    grads_abs = [] #computed scores
    
    for it in range(iters):
        clear_feature_targets_from_net(net)
        
        # Grab a single batch from the training dataset
        inputs, targets = next(iter_dataloader)
        inputs = inputs.to(device)
        targets = targets.to(device)




        # Compute gradients (but don't apply them)
        net.zero_grad()
        
        #Run model forward until all targets reached
        try:
            outputs = net.forward(inputs)
        except TargetReached:
            pass
        
        #get proper loss
        if feature_targets is None:
            loss = criterion(outputs, targets)
        else:   #the real target is feature values in the network
            loss = get_feature_target_from_net(net, feature_targets, feature_targets_coefficients,device=device)
        
        loss.backward()

        #get weight-wise scores
        if grads_abs == []:
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    grads_abs.append(torch.abs(layer.weight_mask.grad))
                    if layer.last_layer:
                        break
        else:
            count = 0
            for layer in net.modules():
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    grads_abs[count] += torch.abs(layer.weight_mask.grad)        
                    count += 1
                    if layer.last_layer:
                        break
                
                
    #structure scoring by weights, kernels, or filters   
    
    if structure == 'weights':
        structure_grads_abs = grads_abs
    elif structure == 'kernels':
        structure_grads_abs = []
        for grad in grads_abs:
            if len(grad.shape) == 4: #conv2d layer
                structure_grads_abs.append(torch.mean(grad,dim = (2,3))) #average across height and width of each kernel
    else:
        structure_grads_abs = []
        for grad in grads_abs:
            if len(grad.shape) == 4: #conv2d layer
                structure_grads_abs.append(torch.mean(grad,dim = (1,2,3))) #average across channel height and width of each filter
                
        
    return structure_grads_abs


