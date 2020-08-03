


import torch
from torchvision import models
import torch.nn as nn
import numpy as np


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
    
    
    def __init__(self, from_conv,store_activations=False, store_ranks = False, cuda=True):      # from conv is normal nn.Conv2d object to pull weights and bias from
        super(dissected_Conv2d, self).__init__()
        self.from_conv = from_conv
        self.cuda = cuda
        self.store_activations = store_activations
        self.store_ranks = store_ranks
        self.postbias_ranks = {'act':None,'grad':None,'actxgrad':None}
        self.preadd_ranks = {'act':None,'grad':None,'actxgrad':None}
        self.preadd_conv = self.make_preadd_conv(from_conv)
        self.bias = None
        if self.from_conv.bias is not None:
            self.bias = from_conv.bias.unsqueeze(1).unsqueeze(1)
        #generate a dict that says which indices should be added together in for 'permute_add_featuremaps'
        self.add_indices = {}
        for o in range(self.from_conv.out_channels):
            self.add_indices[o] = []
            for i in range(self.from_conv.in_channels):
                self.add_indices[o].append(o+i*self.from_conv.out_channels)


    def compute_edge_rank(self,grad):
        #print('compute edge rank called')
        activation = self.preadd_out
        taylor = activation * grad 
        rank_key  = {'act':activation,'grad':grad,'actxgrad':taylor}
        for key in rank_key:
            if self.preadd_ranks[key] is None: #initialize at 0
                self.preadd_ranks[key] = torch.FloatTensor(activation.size(1)).zero_()
                if self.cuda:
                    self.preadd_ranks[key] = self.preadd_ranks[key].cuda()
            mean = rank_key[key].mean(dim=(0, 2, 3)).data
            self.preadd_ranks[key] += mean



    def compute_node_rank(self,grad):
        activation = self.postbias_out
        taylor = activation * grad 
        rank_key  = {'act':activation,'grad':grad,'actxgrad':taylor}
        for key in rank_key:
            if self.postbias_ranks[key] is None: #initialize at 0
                self.postbias_ranks[key] = torch.FloatTensor(activation.size(1)).zero_()
                if self.cuda:
                    self.postbias_ranks[key] = self.postbias_ranks[key].cuda()
            mean = rank_key[key].mean(dim=(0, 2, 3)).data
            self.postbias_ranks[key] += mean



    def format_edges(self, data= 'activations'):
        #fetch preadd activations as [img,out_channel, in_channel,h,w]
        #fetch preadd ranks as [out_chan,in_chan]

        if not self.store_activations:
            print('activations arent stored, use "store_activations=True" on model init. returning None')
            return None
        out_acts_list = []
        if data == 'activations':
            for out_chan in self.add_indices:
                in_acts_list = []
                for in_chan in self.add_indices[out_chan]:
                    in_acts_list.append(self.preadd_out[:,in_chan,:,:].unsqueeze(dim=1).unsqueeze(dim=1))                    
                out_acts_list.append(torch.cat(in_acts_list,dim=2))
            return torch.cat(out_acts_list,dim=1).cpu().detach().numpy()

        else:
            output = {}
            for rank_type in ['act','grad','actxgrad']:
                out_acts_list = []
                for out_chan in self.add_indices:
                    in_acts_list = []
                    for in_chan in self.add_indices[out_chan]:
                        in_acts_list.append(self.preadd_ranks[rank_type][in_chan].unsqueeze(dim=0).unsqueeze(dim=0))                   
                    out_acts_list.append(torch.cat(in_acts_list,dim=1))
                output[rank_type] = torch.cat(out_acts_list,dim=0).cpu().detach().numpy()
            return output
                        

    def normalize_ranks(self):
        for rank_type in ['act','grad','actxgrad']:
            e = torch.abs(self.preadd_ranks[rank_type])
            n = torch.abs(self.postbias_ranks[rank_type])

            e = e.cpu()
            e = e / np.sqrt(torch.sum(e * e))
            n = n.cpu()
            n = n / np.sqrt(torch.sum(n * n))

            self.preadd_ranks[rank_type] = e
            self.postbias_ranks[rank_type] = n



    def forward(self, x):
        if self.store_activations:
            self.input = x

        preadd_out = self.preadd_conv(x)  #get output of convolutions

        #store values of intermediate outputs after convolution
        if self.store_activations:
            self.preadd_out = preadd_out
 
        #Set hooks for calculating rank on backward pass
        if self.store_ranks:
            self.preadd_out = preadd_out
            self.preadd_out.register_hook(self.compute_edge_rank)
            #if self.preadd_ranks is not None:
            #    print(self.preadd_ranks.shape)

        added_out = self.permute_add_featuremaps(preadd_out)    #add convolution outputs by output channel
        if self.bias is not None:  
            postbias_out = added_out + self.bias
        else:
            postbias_out = added_out

        #Store values of final module output
        if self.store_activations:
            self.postbias_out = postbias_out
 
        #Set hooks for calculating rank on backward pass
        if self.store_ranks:
            self.postbias_out = postbias_out
            self.postbias_out.register_hook(self.compute_node_rank)
            #if self.postbias_ranks is not None:
            #    print(self.postbias_ranks.shape)

        return postbias_out


# takes a full model and replaces all conv2d instances with dissected conv 2d instances
def dissect_model(model,store_activations=True,store_ranks=True,cuda=True):
 
    for name, module in reversed(model._modules.items()):
        
        if len(list(module.children())) > 0:
            # recurse
            model._modules[name] = dissect_model(module, store_activations=store_activations)

        if isinstance(module, torch.nn.modules.conv.Conv2d):    # found a 2d conv module to transform
            new_module = dissected_Conv2d(module, store_activations=store_activations,store_ranks=store_ranks,cuda=cuda) 
            model._modules[name] = new_module


    return model