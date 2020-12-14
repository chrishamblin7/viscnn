#arithmetic can be put into weighting category field in the spirit of an fMRI contrast 
#these functions deal with those inputs 
import os
from copy import deepcopy
import torch
import pandas as pd
import numpy as np
from PIL import Image
import plotly.graph_objs as go
import sys
sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from dissected_Conv2d import *
from data_loading_functions import *
from visualizer_helper_functions import *
from sympy import *

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def image_category_or_contrast(contrast_string,params):
    is_image,image_path = get_image_path(contrast_string.strip(),params)
    if is_image:
        return 'input_image'
    if contrast_string.strip() in params['categories']:
        return 'category'
    return 'contrast'
    


def parse_contrast(contrast_string,input_image_list,categories_list): 
    arith_syms = ['*','+','/','(',')',' ','-']
    valid_variables = input_image_list+categories_list
    variables_found = 0
    variable_dict = {}
    var = ''
    #var_pos_start = 0
    for i in range(len(contrast_string)):
        if contrast_string[i] in arith_syms: #weve hit something that not variable name so check our var
            if var == '':
                continue
            if is_number(var): 
                var = ''
                continue
            found = False
            for valid_name in valid_variables:        #check our var against valid names
                if var.strip() == valid_name:
                    if var.strip() not in variable_dict:
                        variable_dict[valid_name] = 'x'+str(variables_found)
                        variables_found+=1
                    found =True
            if not found:
                message = '"%s" is not a valid weight category, all weight categories must be from image names in the "input_image" folder, or folder names under the "rank_image" folder'%var
                raise ValueError(message)
            var = ''
        else:
            var += contrast_string[i]
    if var is not '' and not is_number(var):  #must check last variable
        found = False
        for valid_name in valid_variables:        #check our var against valid names
            if var.strip() == valid_name:
                if var.strip() not in variable_dict:
                    variable_dict[valid_name] = 'x'+str(variables_found)
                    variables_found+=1
                found =True
        if not found:
            message = '"%s" is not a valid weight category, all weight categories must be from image names in the "input_image" folder, or folder names under the "rank_image" folder'%var
            raise ValueError(message)

    
    len_order_variable_dict = {}
    for k in sorted(variable_dict, key=len, reverse=True):
        len_order_variable_dict[k] = variable_dict[k]
    sym_contrast_string = contrast_string
    for k in len_order_variable_dict:
       sym_contrast_string = sym_contrast_string.replace(k,len_order_variable_dict[k])

    simple_string = simplify(sym_contrast_string)
    expanded_string = expand(simple_string)

    return len_order_variable_dict, sym_contrast_string
                

class layer_rank_arithmetic_obj():
    def __init__(self,arith_exprs,array_dict):
        self.num_layers = len(array_dict['x0']['nodes']['act'])
        #print('num_layers: ');print(self.num_layers)
        self.array_dict = array_dict
        #print('array_dict: ');print(self.array_dict)
        self.arith_exprs = arith_exprs
        #print('arith express: ');print(self.arith_exprs)
        self.result_dict = {
                            'nodes':{'act':[], 'grad':[], 'actxgrad':[]},
                            'edges':{'act':[], 'grad':[], 'actxgrad':[]}
                            }

        for part in ['nodes','edges']:      #Can be Parallelized!
            #print('\n'+part+'\n')
            for rank_type in ['act','grad','actxgrad']:
                #print('\n'+rank_type+'\n')
                for layer in range(self.num_layers):
                    #print('\n'+str(layer)+'\n')
                    sym_replace_exprs = self.arith_exprs
                    #print('replace express');print(sym_replace_exprs )
                    for k in self.array_dict:
                        sym_replace_exprs = sym_replace_exprs.replace(k,'array_dict["%s"]["%s"]["%s"][%s][1]'%(k,part,rank_type,str(layer)))
                    #print('replace express');print(sym_replace_exprs )
                    exec_exprs = 'self.result_array = ' + sym_replace_exprs
                    #print('exec express'); print(exec_exprs)
                    exec(exec_exprs)   #perform arithmetic over arrays to get "new_array"
                    #print('result_array');print(self.result_array)
                    self.thresh_indices = self.result_array < 0 
                    self.result_array[self.thresh_indices] = 0 # Threshold negative values at 0
                    #print('thresh_array');print(self.result_array)
                    dictupdate_exprs = 'self.result_dict["%s"]["%s"].append([array_dict["x0"]["nodes"]["act"][%s][0],self.result_array])'%(part,rank_type,str(layer))
                    #print('dictupdate_expr');print(dictupdate_exprs)
                    exec(dictupdate_exprs)
        


    def get_result(self):
        #print(self.result_dict)
        return self.result_dict

def layer_rank_arithmetic(arith_exprs,array_dict):
    obj = layer_rank_arithmetic_obj(arith_exprs,array_dict)
    return obj.get_result()

def var_dict_2_array_dict(var_dict,target_node,model_dis,params):
    array_dict = {}
    for var in var_dict:
        is_image, image_path = get_image_path(var,params)
        if is_image:
            array_dict[var_dict[var]] = get_model_ranks_from_image(image_path,target_node, model_dis, params)
        else:
            category_dict = {'nodes':{},'edges':{}} 
            category_dict['nodes'] = torch.load(os.path.join(params['ranks_data_path'],'categories_nodes','%s_nodes_rank.pt'%var))
            category_dict['edges'] = torch.load(os.path.join(params['ranks_data_path'],'categories_edges','%s_edges_rank.pt'%var))
            array_dict[var_dict[var]] = category_dict
    return array_dict

'''
def add_norm_2_prenorm_dict(prenorm_dict):
    num_layers = len(prenorm_dict['nodes']['act']['prenorm'])
    for part in ['nodes','edges']:
        for rank_type in ['act','grad','actxgrad']:
            prenorm_dict[part][rank_type]['norm'] = []
            for layer in range(num_layers):
                #prenorm_dict[part][rank_type]['prenorm'][layer] = prenorm_dict[part][rank_type]['prenorm'][layer].cpu()
                #a = torch.abs(prenorm_dict[part][rank_type]['prenorm'][layer])
                a = prenorm_dict[part][rank_type]['prenorm'][layer]
                norm_a = a / np.sqrt(np.sum(a * a))
                norm_a = np.nan_to_num(norm_a)
                prenorm_dict[part][rank_type]['norm'].append(norm_a)

    return prenorm_dict
'''

def contrast_str_2_dfs(contrast_string,target_node,model_dis,params):
    all_input_images = params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/')
    var_dict, sym_contrast_string = parse_contrast(contrast_string,all_input_images,params['categories'])
    array_dict = var_dict_2_array_dict(var_dict,target_node,model_dis,params)
    contrast_dict = layer_rank_arithmetic(sym_contrast_string,array_dict)
    #contrast_dict = add_norm_2_prenorm_dict(prenorm_contrast_dict)
    nodes_df, edges_df = rank_dict_2_df(contrast_dict)
    return nodes_df, edges_df 



