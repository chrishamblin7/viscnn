


import torch
from torchvision import models
import torch.nn as nn



class dissected_Conv2d(torch.nn.Module):       #2d conv Module class that has presum activation maps as intermediate output
    
    def gen_inout_permutation(self,in_channels,out_channels):
        '''
        When we flatten out all the output channels not to be grouped by 'output channel', we still want the outputs sorted
        such that they can be conveniently added based on input channel later
        '''
        unsorted = range(in_channels*out_channels)
        sorted_perm = []
        for i in range(in_channels):
            for j in range(out_channels):
                sorted_perm.append(i+j*in_channels)
        return torch.LongTensor(sorted_perm)


    def make_preadd_conv(self,from_conv):
        '''
        nn.Conv2d takes in 'in_channel' number of feature maps, and outputs 'out_channel' number of maps. 
        internally it has in_channel*out_channel number of 2d conv kernels. Normally, featuremaps associated 
        with a particular output channel resultant from these kernel convolution are all added together,
        this function changes a nn.Conv2d module into a module where this final addition doesnt happen. 
        The final addition can be performed seperately with permute_add_feature_maps.
        '''
        in_chan = from_conv.weight.shape[1]
        out_chan = from_conv.weight.shape[0]
        kernel_size = from_conv.kernel_size
        padding = from_conv.padding
        stride = from_conv.stride
        new_conv = nn.Conv2d(in_chan,in_chan*out_chan,kernel_size = kernel_size,
                             bias = False, padding=padding,stride=stride,groups= in_chan)
        perm = self.gen_inout_permutation(in_chan,out_chan)
        #print(perm)
        new_conv.weight = torch.nn.parameter.Parameter(
                from_conv.weight.view(in_chan*out_chan,1,kernel_size[0],kernel_size[1])[perm])
        return new_conv

        
    def permute_add_featuremaps(self,feature_map):
        '''
        Perform the sum within output channels step.  (THIS NEEDS TO BE SPEED OPTIMIZED)
        '''
        added_maps = []
        for o in self.add_indices:
            added_map = feature_map[:,self.add_indices[o][0],:,:]
            for i in range(1,len(self.add_indices[o])):
                added_map = added_map + feature_map[:,self.add_indices[o][i],:,:]
            added_map = torch.unsqueeze(added_map,dim =1)
            added_maps.append(added_map)
        return torch.cat(added_maps,1)
    
    
    def __init__(self, from_conv,store_activations=False, store_rank = False):      # from conv is normal nn.Conv2d object to pull weights and bias from
        super(dissected_Conv2d, self).__init__()
        self.from_conv = from_conv
        self.store_activations = store_activations
        self.activations = None
        self.preadd_conv = self.make_preadd_conv(from_conv)
        if not self.from_conv.bias == None:
            self.bias = from_conv.bias.unsqueeze(1).unsqueeze(1)
        #generate a dict that says which indices should be added together in for 'permute_add_featuremaps'
        self.add_indices = {}
        for o in range(self.from_conv.out_channels):
            self.add_indices[o] = []
            for i in range(self.from_conv.in_channels):
                self.add_indices[o].append(o+i*self.from_conv.out_channels)
        
        
    def forward(self, x):
        
        preadd_out = self.preadd_conv(x)

        if self.store_activations:
            #store preadd activations as [img,out_channel, in_channel,h,w]
            out_acts_list = []
            for out_chan in self.add_indices:
                in_acts_list = []
                for in_chan in self.add_indices[out_chan]:
                    in_acts_list.append(preadd_out[:,in_chan,:,:].unsqueeze(dim=1).unsqueeze(dim=1))
                out_acts_list.append(torch.cat(in_acts_list,dim=2))
            self.activations = torch.cat(out_acts_list,dim=1)

        added_out = self.permute_add_featuremaps(preadd_out)
        bias_out = added_out + self.bias
        return bias_out




# takes a full model and replaces all conv2d instances with dissected conv 2d instances
def dissect_model(model,store_activations=False):
 
    for name, module in reversed(model._modules.items()):
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = dissect_model(module, store_activations=store_activations)

        if isinstance(module, torch.nn.modules.conv.Conv2d):    # found a 2d conv module to transform
            new_module = dissected_Conv2d(module, store_activations=store_activations) 
            model._modules[name] = new_module


    return model