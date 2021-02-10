#functions for using lucent to get visualizations
from lucent_edited.optvis import render
from lucent_edited.modelzoo import inceptionv1
from lucent_edited.modelzoo.util import get_model_layers
from lucent_edited.optvis import render, param, transform, objectives

import inspect
import time
import pandas as pd

from visualizer_helper_functions import *
import sys
sys.path.insert(0, os.path.abspath('../prep_model_scripts/'))
from dissected_Conv2d import *

def gen_visualization(model,image_name,objective,parametrizer,optimizer,transforms,image_size,params):
    full_image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
    if parametrizer is None:
        parametrizer = lambda: param.image(image_size)
    _ = render.render_vis(model, objective, parametrizer, optimizer, transforms=transforms,save_image=True,image_name=full_image_path, show_inline=True)

def gen_objective_str(targetid,model,params):
    print('generating feature_viz objective string for %s'%targetid)
    if '-' in targetid:
        within_id = edgename_2_singlenum(model,targetid,params)
        layer_name = nodeid_2_perlayerid(targetid.split('-')[1],params)[2]
        return layer_name+'_preadd_conv:'+str(within_id)
    else:
        layer, within_id, layer_name = nodeid_2_perlayerid(targetid,params)
        return layer_name+':'+str(within_id)


#possibility of neuron
def gen_objective(targetid,model,params):
    print('generating feature_viz objective string for %s'%targetid)
    if '-' in targetid:
        within_id = edgename_2_singlenum(model,targetid,params)
        layer_name = nodeid_2_perlayerid(targetid.split('-')[1],params)[2]
        return layer_name+'_preadd_conv:'+str(within_id)
    else:
        layer, within_id, layer_name = nodeid_2_perlayerid(targetid,params)
        return layer_name+':'+str(within_id)

def object_2_str(obj,prefix):
    if isinstance(obj,str):
        return obj
    if isinstance(obj,list):
        return obj
    if obj is None:
        return('none')
    else:
        s= inspect.getsource(obj).replace(' ','').replace(prefix,'').strip()
        return inspect.getsource(obj)
    

def fetch_deepviz_img(model,targetid,params):
    model = set_across_model(model,'target_node',None)
    objective = gen_objective_str(targetid,model,params)
    file_path = params['prepped_model_path']+'/visualizations/images.csv'
    parametrizer = params['deepviz_param']
    optimizer = params['deepviz_optim']
    transforms = params['deepviz_transforms']
    image_size = params['deepviz_image_size']

    param_str = object_2_str(parametrizer,"params['deepviz_param']=")
    optimizer_str = object_2_str(optimizer,"params['deepviz_optim']=")
    transforms_str = object_2_str(transforms,"params['deepviz_transforms']=")
    df = pd.read_csv(file_path,dtype=str)
    df_sel = df.loc[(df['targetid'] == str(targetid)) & (df['objective'] == objective) & (df['parametrizer'] == param_str) & (df['optimizer'] == optimizer_str) & (df['transforms'] == transforms_str)]
    if len(df_sel) == 0:
        print('deepviz image not found, generating . . .')
        #image_name = 'deepviz_'+str(targetid)+'_'+objective+'_'+str(time.time())+'.jpg'
        image_name = str(targetid)+'_'+objective+'_'+str(time.time())+'.jpg'
        gen_visualization(model,image_name,objective,parametrizer,optimizer,transforms,image_size,params) 
        with open(file_path, 'a') as csv:
            csv.write(','.join([image_name,str(targetid),objective,param_str,optimizer_str,transforms_str])+'\n')
    else:
        print('found pre-generated image')
        image_name = df_sel.iloc[0]['image_name']    
    return image_name

def fetch_deepviz_img_for_node_inputs(model,edgeid,params):
    model = set_across_model(model,'target_node',None)
    objective = gen_objective_str(edgeid,model,params)
    file_path = params['prepped_model_path']+'/visualizations/images.csv'
    parametrizer = params['deepviz_param']
    optimizer = params['deepviz_optim']
    transforms = params['deepviz_transforms']
    image_size = params['deepviz_image_size']

    param_str = object_2_str(parametrizer,"params['deepviz_param']=")
    optimizer_str = object_2_str(optimizer,"params['deepviz_optim']=")
    transforms_str = object_2_str(transforms,"params['deepviz_transforms']=")
    df = pd.read_csv(file_path,dtype=str)
    df_sel = df.loc[(df['targetid'] == str(edgeid)) & (df['objective'] == objective) & (df['parametrizer'] == param_str) & (df['optimizer'] == optimizer_str) & (df['transforms'] == transforms_str)]
    if len(df_sel) == 0:
        if 'r' in edgeid or 'g' in edgeid or 'b' in edgeid or 'gs' in edgeid:
            print('deepviz image not found for node input %s, generating . . .'%edgeid)
            #image_name = 'deepviz_'+str(targetid)+'_'+objective+'_'+str(time.time())+'.jpg'
            image_name = str(edgeid)+'_'+objective+'_'+str(time.time())+'.jpg'
            gen_visualization(model,image_name,objective,parametrizer,optimizer,transforms,image_size,params) 
            with open(file_path, 'a') as csv:
                csv.write(','.join([image_name,str(edgeid),objective,param_str,optimizer_str,transforms_str])+'\n')
        else:
            nodeid = edgeid.split('-')[0]
            print('no deepviz for edge %s for node inputs, looking for image for node %s'%(edgeid,nodeid))
            objective = gen_objective_str(nodeid,model,params)
            df_sel = df.loc[(df['targetid'] == str(nodeid)) & (df['objective'] == objective) & (df['parametrizer'] == param_str) & (df['optimizer'] == optimizer_str) & (df['transforms'] == transforms_str)]
            if len(df_sel) == 0:
                print('STILL no deepviz image found! generating now, but you should pregenerate these!')
                image_name = str(nodeid)+'_'+objective+'_'+str(time.time())+'.jpg'
                gen_visualization(model,image_name,objective,parametrizer,optimizer,transforms,image_size,params) 
            else:
                print('found pregenerated node image')
                image_name = df_sel.iloc[0]['image_name'] 
            
    else:
        print('found pre-generated edge image')
        image_name = df_sel.iloc[0]['image_name']   
    return image_name
