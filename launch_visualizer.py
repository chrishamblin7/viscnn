#fetch command line argument (prepped model)
import sys
import os

possible_models = os.listdir('prepped_models')

if len(sys.argv) < 2:
    print('the script must be called with an argument of the name of the prepped model you would like to visualize. Looking in the "prepped_models" folder, your current options are:')
    for name in possible_models:
        print(name)
    exit()
elif sys.argv[1] not in possible_models:
    print('%s is not a subfolder of "prepped_models". Your current options are:'%(sys.argv[1]))
    for name in possible_models:
        print(name)
    exit()
else:
    prepped_model_folder = sys.argv[1]    

sys.path.append('prepped_models/%s'%prepped_model_folder)
import prep_model_params_used as params

import pandas as pd
import numpy as np

#load nodes df
print('loading nodes data')

nodes_df = pd.read_csv('prepped_models/%s/node_ranks.csv'%prepped_model_folder)

#make wide version
nodes_wide_df = nodes_df.pivot(index = 'node_num',columns='class', values='rank_score')

def get_col(node_num, df = nodes_df, idx = 'node_num', col = 'layer'):
    return df.loc[(df[idx] == node_num) & (df['class'] == df['class'].unique()[0]), col].item()

nodes_wide_df.reset_index(inplace=True)
nodes_wide_df['layer'] = nodes_wide_df['node_num'].apply(get_col)
nodes_wide_df = nodes_wide_df.rename(columns = {'class':'index'})


#list of layer nodes
layer_nodes = {}
for row in nodes_df[nodes_df['class'] == 'overall'].itertuples(): 
    if row.layer not in layer_nodes:
        layer_nodes[row.layer] = []
    layer_nodes[row.layer].append(row.node_num)

num_layers = max(layer_nodes.keys()) + 1
num_nodes = len(nodes_wide_df.index)

#list of classes
classes = list(nodes_df['class'].unique())
classes.remove('overall')
classes.insert(0,'overall')



#misc formatting functions

def nodeid_2_perlayerid(nodeid):    #takes in node unique id outputs tuple of layer and within layer id
    if isinstance(nodeid,str):
        if not nodeid.isnumeric():
            layer = 'img'
            within_layer_id = imgnode_names.index(nodeid)
            return layer,within_layer_id
    nodeid = int(nodeid)
    layer = nodes_df[nodes_df['class']=='overall'][nodes_df['node_num'] == nodeid]['layer'].item()
    within_layer_id = nodes_df[nodes_df['class']=='overall'][nodes_df['node_num'] == nodeid]['node_num_by_layer'].item()
    return layer,within_layer_id

def layernum2name(layer,offset=1,title = 'layer'):
    return title+' '+str(layer+offset)


def get_nth_element_from_nested_list(l,n):    #this seems to come up with the nested layer lists
    flat_list = [item for sublist in l for item in sublist]
    return flat_list[n]
  



## adding images
print('loading input images')

import os

input_image_directory = params.input_img_path
list_of_input_images = os.listdir(input_image_directory)
list_of_input_images.sort()

from PIL import Image

def rgb2hex(r, g, b):
    return '#{:02x}{:02x}{:02x}'.format(r, g, b)


def image2heatmap(image_path,resize = False,size = (32,32)):          #displays image as a plotly heatmap object, with colors preserved 
    img = Image.open(image_path)
    if resize:
        img = img.resize(size,resample=Image.NEAREST)
    pixels = img.convert('RGBA').load() #rgba values
    width, height = img.size   #width and height of image
    num_pixels = width*height
    step = 1/num_pixels
    
    colorscale = []           
    z = []
    
    i = 0
    for y in range(height):
        z.append([])
        for x in range(width):
            z[-1].append(step*(i+.5))
            r, g, b, a = pixels[x, y]
            #colorscale.append([step*i,'rgba(%s,%s,%s,%s)'%(r,b,g,a)])
            #colorscale.append([step*(i+1),'rgba(%s,%s,%s,%s)'%(r,b,g,a)])
            colorscale.append([step*i,rgb2hex(r, g, b)])
            colorscale.append([step*(i+1),rgb2hex(r, g, b)])
            i+=1     
    heatmap = go.Heatmap(z=np.flip(np.array(z),0), 
                     colorscale = colorscale, 
                     showscale = False)
    fig = go.Figure(data=[heatmap])

    fig.update_layout(width=350, 
                      height=350,
                      uirevision = True,
                      margin=dict(
                        l=1,
                        r=1,
                        b=1,
                        t=1,
                        pad=1)
                    )
    return fig

def get_channelwise_image(image_name,channel,input_image_directory=input_image_directory):    
    #THIS NEEDS TO BE NORMALIZED AS PER THE MODELS DATALOADER
    im = Image.open(input_image_directory+'/'+image_name)
    np_full_im = np.array(im)
    return np_full_im[:,:,channel]




#load edges
print('loading edge data')

edges_df = pd.read_csv('prepped_models/%s/edge_ranks.csv'%prepped_model_folder)   

#make edges wide format df
edges_wide_df = edges_df.pivot(index = 'edge_num',columns='class', values='rank_score')
edges_wide_df.reset_index(inplace=True)
edges_wide_df['layer'] = edges_wide_df['edge_num'].apply(get_col, df=edges_df,idx='edge_num')
edges_wide_df['in_channel'] = edges_wide_df['edge_num'].apply(get_col, df=edges_df,idx='edge_num',col='in_channel')
edges_wide_df['out_channel'] = edges_wide_df['edge_num'].apply(get_col, df=edges_df,idx='edge_num',col='out_channel')

print(edges_df.head(4605))
edges_wide_df.head(165)
#edges_df.loc[(edges_df['rank_score'] > .05) & (edges_df['class'] == 'frog')]

num_edges = len(edges_wide_df.index) #number of total edges



#generate mds projections of nodes layerwise, as determined by their per class rank scores
print('generating mds projection of nodes')

import numpy as np
from sklearn import manifold
from sklearn.metrics import euclidean_distances

def add_norm_col(df,classes=classes[1:]):
    norms = []
    norm = 0
    for index, row in df.iterrows():
        for label in classes:
            norm += row[label]**2
        norm = np.sqrt(norm)
        norms.append(norm)
    norms = np.array(norms)
    df['class_norm'] = norms

add_norm_col(nodes_wide_df)   
    
layer_similarities = {}
for layer in layer_nodes:
    layer_df = nodes_wide_df[nodes_wide_df['layer'] == layer]
    for label in classes:
        layer_df[label] = layer_df.apply(lambda row : row[label]/row['class_norm'], axis = 1)   
    layer_similarities[layer] = euclidean_distances(layer_df.iloc[:,1:-2])



layer_mds = {}
for layer in layer_similarities:
	print('layer: %s'%str(layer))
	mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9, 
      random_state=2, dissimilarity="precomputed", n_jobs=1)
	pos = mds.fit(layer_similarities[layer]).embedding_
	layer_mds[layer] = pos

#print(layer_mds)



#generate node colors based on target class (nodes that aren't important should be faded)
print('generating node colors')

target_class = classes[1]

#Node Opacity
layer_colors = ['rgba(31,119,180,', 
                'rgba(255,127,14,',
                'rgba(44,160,44,', 
                'rgba(214,39,40,',
                'rgba(39, 208, 214,', 
                'rgba(242, 250, 17,',
                'rgba(196, 94, 255,',
                'rgba(193, 245, 5,',
                'rgba(245, 85, 5,',
                'rgba(5, 165, 245,',
                'rgba(245, 5, 105,',
                'rgba(218, 232, 23,',
                'rgba(148, 23, 232,',
                'rgba(23, 232, 166,',]


def node_color_scaling(x):
    return -(x-1)**4+1

def gen_node_colors(target_class):

    node_colors = []
    node_weights = []
    for layer in layer_nodes:
        node_colors.append([])
        node_weights.append([])
        for node in layer_nodes[layer]:
            node_weight = nodes_df[nodes_df['class']==target_class].iloc[node].rank_score
            node_weights[-1].append(node_weight)
            alpha = node_color_scaling(node_weight)
            node_colors[-1].append(layer_colors[layer%len(layer_colors)]+str(round(alpha,3))+')')
            
    return node_colors,node_weights

node_colors,node_weights = gen_node_colors(target_class)     #list of lists




#Node positions
#def gen_node_positions()
layer_distance = 1   # distance in X direction each layer is separated by
node_positions = []
layer_offset = 0
for layer in layer_mds:
    node_positions.append({})
    node_positions[-1]['X'] = [] 
    node_positions[-1]['Y'] = [] 
    node_positions[-1]['Z'] = []  
    for i in range(len(layer_mds[layer])): 
        node_positions[-1]['Y'].append(layer_mds[layer][i][0])
        node_positions[-1]['Z'].append(layer_mds[layer][i][1])
        node_positions[-1]['X'].append(layer_offset)
    layer_offset+=1*layer_distance

#print(node_positions[0])






#image nodes (one for each channel of input image)
print('generating image channel nodes')


num_img_chan = len(edges_df.loc[edges_df['layer'] == 0]['in_channel'].unique()) #number of channels in input image

def gen_imgnode_graphdata(num_chan = num_img_chan):     #returns positions, colors and names for imgnode graph points
    if num_chan == 1: #return a centered position, grey square, with 'gs' label
        return {'X':[-1*layer_distance],'Y':[0],'Z':[0]}, ['rgba(170,170,170,.7)'], ['gs']
    if num_chan == 3:
        colors = ['rgba(255,0,0,.7)','rgba(0,255,0,.7)','rgba(0,0,255,.7)']
        names = ['r','g','b']
    else:
        #colors
        other_colors = ['rgba(255,0,0,.7)','rgba(0,255,0,.7)','rgba(0,0,255,.7)',
                        'rgba(255,150,0,.7)','rgba(0,255,150,.7)','rgba(150,0,255,.7)',
                        'rgba(255,0,150,.7)','rgba(150,255,0,.7)','rgba(0,150,255,.7)']
        colors = []
        for i in num_chan:
            colors.append(i%len(other_colors)) 
        #names
        names = []
        for i in range(num_chan):
            names.append('img_'+str(i))   
            
    positions = {'X':[],'Y':[],'Z':[]}     #else return points evenly spaced around a unit circle
    a = 2*np.pi/num_chan          #angle to rotate each point
    for p in range(num_chan):
        positions['X'].append(-1*layer_distance)
        positions['Y'].append(round(np.sin(a*p)/5,2))
        positions['Z'].append(round(np.cos(a*p)/5,2)) 
    
    return positions, colors, names
    

imgnode_positions,imgnode_colors,imgnode_names = gen_imgnode_graphdata()




#Edge selection
print('edge selection')
def edge_width_scaling(x):
    return max(.4,(x*10)**1.7)

def edge_color_scaling(x):
    return max(.7,-(x-1)**4+1)


def get_thresholded_edges(threshold,df=edges_df,target_class=target_class):          #just get those edges that pass the threshold criteria for the target class
    if len(threshold) != 2:
        raise Exception('length of threshold needs to be two ([lower, higher])')
    return edges_df.loc[(edges_df['rank_score'] >= threshold[0]) & (edges_df['rank_score'] <= threshold[1]) & (edges_df['class'] == target_class)]
    #return edges_df.loc[(threshold[0] <= edges_df['rank_score'] <= threshold[1]) & (edges_df['class'] == target_class)]
    #return edges_df.loc[(edges_df['rank_score'] >= threshold[0]) & (edges_df['class'] == target_class)]


edge_threshold = [.1,1]
edges_select_df = get_thresholded_edges(edge_threshold)



def get_max_edge_widths(edge_widths):
    maxes = []
    for layer in range(len(edge_widths)):
        if len(edge_widths[layer]) >0:
            maxes.append(edge_widths[layer].index(max(edge_widths[layer])))
        else:
            maxes.append(None)
    return maxes

def gen_edge_graphdata(df = edges_select_df, node_positions = node_positions, num_hoverpoints=15,target_class=target_class):
    edge_positions = []
    colors = []
    widths = []
    weights = []
    names = []
    #max_weight = 0
    for row in df.itertuples():
        while row.layer > len(edge_positions): # we skipped a layer, its got no edges in threshold, so lets add empty lists
            edge_positions.append({'X':[],'Y':[],'Z':[]})
            colors.append([])
            widths.append([])
            weights.append([])
            names.append([])  
        if row.layer == len(edge_positions):
            edge_positions.append({'X':[],'Y':[],'Z':[]})
            colors.append([])
            widths.append([])
            weights.append([])
            names.append([])        
        #position
        for dim in ['X','Y','Z']:
            end_pos = node_positions[row.layer][dim][row.out_channel]
            if row.layer != 0:
                start_pos = node_positions[row.layer-1][dim][row.in_channel]
            else:
                start_pos = imgnode_positions[dim][row.in_channel]
            
            step = (end_pos-start_pos)/(num_hoverpoints+1)
            points = [start_pos]
            for i in range(1,num_hoverpoints+1):
                points.append(start_pos+i*step)
            points.append(end_pos)
            edge_positions[row.layer][dim].append(points)
        #color
        alpha = edge_color_scaling(row.rank_score)
        colors[row.layer].append(layer_colors[row.layer%len(layer_colors)]+str(round(alpha,3))+')')
        #width
        widths[row.layer].append(edge_width_scaling(row.rank_score))
        #weight
        weights[row.layer].append(row.rank_score)
        #max_weight = max(max_weight, row.rank_score)
        #names
        out_node = layer_nodes[row.layer][row.out_channel]
        if row.layer != 0:
            in_node = layer_nodes[row.layer-1][row.in_channel]
        else:
            in_node = imgnode_names[row.in_channel]
        names[row.layer].append(str(in_node)+'-'+str(out_node))
    max_width_indices = get_max_edge_widths(widths)
    while len(names) < num_layers:
        edge_positions.append({'X':[],'Y':[],'Z':[]})
        colors.append([])
        widths.append([])
        weights.append([])
        names.append([])  
    return edge_positions, colors,widths,weights,names, max_width_indices


def get_edge_from_curvenumber(curvenum,edge_names, num_layers= num_layers):
    edgenum = curvenum-(1+num_layers)
    curve=0
    for layer in range(len(edge_names)):
        for i in range(len(edge_names[layer])):
            if curve==edgenum:
                return layer, i, edge_names[layer][i]
            curve+=1
    return None,None,None
    


edge_positions, edge_colors, edge_widths, edge_weights, edge_names, max_edge_width_indices = gen_edge_graphdata()
#max_edge_width_indices = get_max_edge_widths(edge_widths)
max_edge_weight = edges_df.max().rank_score




#Format Node Feature Maps
print('loading activation maps')

import torch
activations = torch.load('prepped_models/%s/input_img_activations.pt'%prepped_model_folder)

print(activations['edges'][0].shape)
print(activations['nodes'][0].shape)



#Format Edge Kernels
print('loading convolutional kernels')

kernels = torch.load('prepped_models/%s/kernels.pt'%prepped_model_folder)

#print(kernels[0].shape)

#Function for taking a string of form 'node1-node2' and outputting edge info
def check_edge_validity(nodestring):
    from_node = nodestring.split('-')[0]
    to_node = nodestring.split('-')[1]
    try:
        from_layer,from_within_id = nodeid_2_perlayerid(from_node)
        to_layer,to_within_id = nodeid_2_perlayerid(to_node)
        #check for valid edge
        valid_edge = False
        if from_layer=='img':
            if to_layer== 0:
                valid_edge = True
        elif to_layer == from_layer+1:
            valid_edge = True
        if not valid_edge:
            print('invalid edge name')
            return [False, None, None, None, None]
        return True, from_layer,to_layer,from_within_id,to_within_id
    except:
        #print('exception')
        return [False, None, None, None, None] 
    
    
def edgename_2_edge_figures(edgename,kernels=kernels,activations=activations, imagename = list_of_input_images[0]):  #returns truth value of valid edge and kernel if valid
    #print('hello')
    valid,from_layer,to_layer,from_within_id,to_within_id  = check_edge_validity(edgename)
    if valid:
        kernel = kernels[to_layer][to_within_id][from_within_id]
        if from_layer == 'img':
            in_map = get_channelwise_image(imagename,from_within_id)
        else:
            ####!!!!!!!! This needs to be put through activation function (relu)
            in_map = activations['nodes'][from_layer][list_of_input_images.index(imagename)][from_within_id]
        out_map = activations['edges'][to_layer][list_of_input_images.index(imagename)][to_within_id][from_within_id]
        return np.flip(kernel,0),np.flip(in_map,0),np.flip(out_map,0)
        
    else:
        return None,None,None
    

print(edgename_2_edge_figures('b-0')[0])
print(np.flip(kernels[0][0][2],0))




print('building graph')
#hidden state, stores python values within the html itself
state = {'edge_positions':edge_positions,'edge_colors': edge_colors, 'edge_widths':edge_widths,'edge_names':edge_names,
         'edge_threshold':edge_threshold,'edge_weights':edge_weights,'max_edge_width_indices':max_edge_width_indices,
         'imgnode_positions':imgnode_positions,'imgnode_colors':imgnode_colors,'imgnode_names':imgnode_names,
         'node_positions':node_positions,'node_colors':node_colors,'node_weights':node_weights,'layer_distance':layer_distance,'target_class':target_class,
         'node_select_history':['0'],'edge_select_history':[edge_names[0][0]],'last_trigger':None}








#import chart_studio.plotly as py
import plotly.offline as py    #added
import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)   #added

from copy import deepcopy

# right now this is called everytime graph is updated, might be more efficient to store combined_traces somehow,
# and only update whats changed
def gen_networkgraph_traces(state):
    #add imgnodes
    colors = deepcopy(state['imgnode_colors'])
    if not str(state['node_select_history'][-1]).isnumeric():
        colors[state['imgnode_names'].index(state['node_select_history'][-1])] = 'rgba(0,0,0,1)'
    imgnode_trace=go.Scatter3d(x=state['imgnode_positions']['X'],
               y=state['imgnode_positions']['Y'],
               z=state['imgnode_positions']['Z'],
               mode='markers',
               name='image channels',
               marker=dict(symbol='square',
                             size=8,
                             opacity=.99,
                             color=colors,
                             #colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=.5)
                             ),
               text=state['imgnode_names'],
               hoverinfo='text'
               )

    imgnode_traces = [imgnode_trace]


    node_traces = []
    select_layer,select_position = None,None
    if str(state['node_select_history'][-1]).isnumeric():
        select_layer,select_position = nodeid_2_perlayerid(state['node_select_history'][-1])
    for layer in layer_nodes:
        #add nodes
        colors = deepcopy(state['node_colors'][layer])
        if layer == select_layer:
            colors[select_position] = 'rgba(0,0,0,1)'
        node_trace=go.Scatter3d(x=state['node_positions'][layer]['X'],
                   y=state['node_positions'][layer]['Y'],
                   z=state['node_positions'][layer]['Z'],
                   mode='markers',
                   name=layernum2name(layer,title = 'nodes'),
                   marker=dict(symbol='circle',
                                 size=6,
                                 opacity=.99,
                                 color=colors,
                                 #colorscale='Viridis',
                                 line=dict(color='rgb(50,50,50)', width=.5)
                                 ),
                   text=layer_nodes[layer],
                   hoverinfo='text'
                   )

        node_traces.append(node_trace)


    edge_traces = []
    for layer in range(len(state['edge_positions'])):  
        legendgroup = layernum2name(layer ,title = 'edges')
        for edge_num in range(len(state['edge_positions'][layer]['X'])):  
        #add edges      
            color = deepcopy(state['edge_colors'][layer][edge_num])
            if state['edge_names'][layer][edge_num] == state['edge_select_history'][-1]:
                color = 'rgba(0,0,0,1)'
            showlegend = False
            if state['max_edge_width_indices'][layer] == edge_num:
                showlegend = True
            edge_trace=go.Scatter3d(x=state['edge_positions'][layer]['X'][edge_num],
                                    y=state['edge_positions'][layer]['Y'][edge_num],
                                    z=state['edge_positions'][layer]['Z'][edge_num],
                                    legendgroup=legendgroup,
                                    showlegend=showlegend,
                                    name=layernum2name(layer ,title = 'edges'),
                                    mode='lines',
                                    #line=dict(color=edge_colors_dict[layer], width=1.5),
                                    line=dict(color=color, width=state['edge_widths'][layer][edge_num]),
                                    text = state['edge_names'][layer][edge_num],
                                    hoverinfo='text'
                                    )
            edge_traces.append(edge_trace)


    combined_traces = imgnode_traces+node_traces+edge_traces
    return combined_traces

combined_traces = gen_networkgraph_traces(state)


#layout
axis=dict(showbackground=False,
          showspikes=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          #range=[0,0],
          title=''
          )

network_graph_layout = go.Layout(
         #title="%s through Prunned Cifar10 CNN"%target_class,
         #title = target_class,
         #width=1000,
         clickmode = 'event+select',
         transition = {'duration': 500},
         height=600,
         #showlegend=False,
         margin = dict(l=20, r=20, t=20, b=20),
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
             aspectmode ="manual", 
             aspectratio = dict(x=1, y=0.5, z=0.5) #adjusting this stretches the network layer-to-layer
         ),
         uirevision =  True   
         #hovermode='closest',
   )


input_image_layout = go.Layout(width=350, 
                      height=350,
                      uirevision = True,
                      margin=dict(
                        l=1,
                        r=1,
                        b=1,
                        t=1,
                        pad=1))

node_actmap_layout = go.Layout(
    autosize=False,
    width=390,
    height=350,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ))


edge_inmap_layout = go.Layout(
    #title = 'edge input map',
    autosize=False,
    width=260,
    height=200,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=10,
        pad=1
    ))


edge_outmap_layout = go.Layout(
    #title = 'edge output map',
    autosize=False,
    width=260,
    height=200,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=10,
        pad=1
    ))


kernel_layout = go.Layout(
    #title='kernel'
    autosize=False,
    width=270,
    height=200,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ))

network_graph_fig=go.Figure(data=combined_traces, layout=network_graph_layout)

#state['combined_traces']=combined_traces


print('setting up dash app')

import dash
import dash_core_components as dcc
import dash_html_components as html
#import utils.dash_reusable_components as drc
import flask
import os

import json

from dash.dependencies import Input, Output, State



#external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(external_stylesheets = external_stylesheets)


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


app.layout = html.Div(
        [html.Div(         #Left side control panel
            children = [
             html.Label('Weighting Category'),
             dcc.Dropdown(
                id='weight-category',
                options=[{'label': i, 'value': i} for i in classes],
                value=target_class
                ),
             html.Br(),
             html.Label('Weighting Criterion'),
             dcc.Dropdown(
                id='weight-criterion',
                options=[
                    {'label': 'Activations*Grads', 'value': 'actgrads'},
                    {'label': 'Activations', 'value': 'acts'}
                ],
                value='actgrads'
                ),
             html.Br(),   
             html.Label('Layer Projection'),
             dcc.Dropdown(
                id = 'layer-projection',
                options=[
                    {'label': 'MDS', 'value': 'MDS'},
                    {'label': 'Grid', 'value': 'grid'},
                    #{'label': 'SOM', 'value': 'SOM'}
                ],
                value='MDS'
                ),

            html.Br(),
            html.Label('Edge Thresholds'),
                dcc.RangeSlider(
                    id='edge-thresh-slider',
                    min=0,
                    max=np.ceil(max_edge_weight*10)/10,
                    step=0.001,
                    marks={i/10: str(i/10) for i in range(0,int(np.ceil(max_edge_weight*10))+1,int(round(np.ceil(max_edge_weight*10)/5)))},
                    value=[.1,np.ceil(max_edge_weight*10)/10],
                ),
                
            ], className="two columns"
        ),

        html.Div(
            children = [
                
            html.Div([
                dcc.Graph(
                    id='network-graph',
                    figure=network_graph_fig
                )
            ], className= 'row'
            ),
                
            html.Div([
                html.Div([
                html.Label('Input Image'),
                dcc.Dropdown(
                    id='input-image-dropdown',
                    options=[{'label': i, 'value': i} for i in list_of_input_images],
                    value=list_of_input_images[0]
                ),
                html.Br(),
                dcc.Graph(
                    id='img-actmap-graph',
                    figure=image2heatmap(input_image_directory+'/'+list_of_input_images[0]),
                    config={
                            'displayModeBar': False
                            }
                )
                ], className = "three columns"),
                
                html.Div([
                html.Label('Node'),
                dcc.Dropdown(
                    id='node-actmap-dropdown',
                    options=[{'label': str(j), 'value': str(j)} for j in imgnode_names]+[{'label': str(i), 'value': str(i)} for i in range(num_nodes)],
                    value='0'
                ),
                html.Br(),
                dcc.Graph(
                    id='node-actmap-graph',
                    figure=go.Figure(data=go.Heatmap(
                                        z = np.flip(activations['nodes'][0][0][0],0)),
                                        layout=input_image_layout
                                    ),
                    config={
                            'displayModeBar': False
                            }
                )
                ], className = "three columns"),
                
                html.Div([
                html.Label('Edge'),    
                dcc.Input(
                    id='edge-actmaps-input',value=state['edge_names'][0][0], type='text'),
                #html.Button(id='edge-kernel-button',n_clicks=0, children='Submit'),
                html.Br(),
                html.Br(),
                dcc.Graph(
                    id='edge-kernel-graph',
                    figure=go.Figure(data=go.Heatmap(
                                        z = edgename_2_edge_figures(state['edge_names'][0][0])[0]),
                                     layout=kernel_layout
                                    ),
                    config={
                            'displayModeBar': False
                            }
                )
                ], className = "two columns"),
                
                
                html.Div([
                dcc.Graph(
                    id='edge-inmap-graph',
                    figure=go.Figure(data=go.Heatmap(
                                        z = edgename_2_edge_figures(state['edge_names'][0][0])[1]),
                                     layout=edge_inmap_layout
                                    ),
                    config={
                            'displayModeBar': False
                            }
                ),
                html.Br(),
                html.Br(),
                dcc.Graph(
                    id='edge-outmap-graph',
                    figure=go.Figure(data=go.Heatmap(
                                        z = edgename_2_edge_figures(state['edge_names'][0][0])[2]),
                                     layout=edge_outmap_layout
                                    ),
                    config={
                            'displayModeBar': False
                            }
                )
                ], className = "three columns")
                
                
             ], className= 'row'
             ),
                
                
            html.Div([
                html.Div([
                    dcc.Markdown("""
                        **Hover Data**

                        Mouse over values in the graph.
                    """),
                    html.Pre(id='hover-data', style=styles['pre'])
                ], className='two columns'),

                html.Div([
                    dcc.Markdown("""
                        **Click Data**

                        Click on points in the graph.
                    """),
                    html.Pre(id='click-data', style=styles['pre']),
                ], className='two columns'),

                html.Div([
                    dcc.Markdown("""
                        **Selection Data**

                        Choose the lasso or rectangle tool in the graph's menu
                        bar and then select points in the graph.

                        Note that if `layout.clickmode = 'event+select'`, selection data also 
                        accumulates (or un-accumulates) selected data if you hold down the shift
                        button while clicking.
                    """),
                    html.Pre(id='selected-data', style=styles['pre']),
                ], className='two columns'),

#                 html.Div([
#                     dcc.Markdown("""
#                         **Zoom and Relayout Data**

#                         Click and drag on the graph to zoom or click on the zoom
#                         buttons in the graph's menu bar.
#                         Clicking on legend items will also fire
#                         this event.
#                     """),
#                     html.Pre(id='relayout-data', style=styles['pre']),
#                 ], className='two columns')
                
                html.Div([
                    dcc.Markdown("""
                        **Figure Data**

                        Figure json info.
                    """),
                    html.Pre(id='figure-data', style=styles['pre']),
                ], className='four columns')
                
            ], className= 'row'
            )
        ], className="ten columns"
        ),
    #hidden divs for storing intermediate values     
    # The memory store reverts to the default on every page refresh
    dcc.Store(id='memory'),
    # The local store will take the initial data
    # only the first time the page is loaded
    # and keep it until it is cleared.
    dcc.Store(id='local', storage_type='local'),
    # Same as the local store but will lose the data
    # when the browser/tab closes.
    dcc.Store(id='session', storage_type='session',data=state)
    ]
)



####Call Back Functions


# @app.callback(
#     [Output('network-graph', 'figure'),
#      Output('previous-click','children')],
#     [Input('network-graph', 'clickData'),
#      Input('previous-click','children')])
# def highlight_on_click(clickData,state,previous_click =False):
#     #if clickData['points'][0]['curveNumber'] == None:
#     if not clickData:
#         raise Exception('no point clicked yet') 
#     #recolor previous click 
#     if previous_click:
#         if previous_click == 0:
#             state['combined_traces'][0]['marker']['color'] = state['imgnode_colors']
#         elif previous_click < num_layers+1:
#             state['combined_traces'][previous_click]['marker']['color'] = state['node_colors'][previous_click-1]
#         else:
#             layer,position = get_edge_layer_from_curvenumber(previous_click,state['edge_names'])
#             state['combined_traces'][previous_click]['line']['color'] == state['edge_colors'][layer][position]
#     #blackout click
#     trace_num = int(clickData['points'][0]['curveNumber'])
#     if trace_num == 0:
#         new_colors = state['imgnode_colors']
#         new_colors[clickData['points'][0]['pointNumber']] = 'rgba(0,0,0,1)'        
#     elif trace_num < num_layers +1:   #highlight point
#         new_colors = list(state['node_colors'][trace_num-1])
#         new_colors[clickData['points'][0]['pointNumber']] = 'rgba(0,0,0,1)'
#         state['combined_traces'][trace_num]['marker']['color'] = new_colors
#     else: #highlight edge
#         state['combined_traces'][trace_num]['line']['color'] = 'rgba(0,0,0,1)'
#     return 
  
#     layout = graph_layout
#     layout['uirevision'] = True
#     previous_click = {'curveNumber':clickData['points'][0]['curveNumber'],'pointNumber':clickData['points'][0]['pointNumber']}
#     return {'data': combined_traces,'layout': layout},json.dumps(previous_click)



@app.callback(
    Output('session', 'data'),
    [Input('weight-category', 'value'),
     Input('node-actmap-dropdown', 'value'),
     Input('edge-actmaps-input', 'value'),
     Input('edge-thresh-slider','value')],
    [State('session', 'data')])
def update_store(target_class,node_value,edge_value,edge_threshold,state):
    print('CALLED: update_store\n')
    ctx = dash.callback_context
    if not ctx.triggered:
        raise Exception('no figure updates yet')
    else:
        trigger = ctx.triggered[0]['prop_id']
    state['last_trigger'] = trigger  #store the last trigger of state change in state
    print('TRIGGER %s'%trigger)
    if trigger == 'weight-category.value':
        print('changing target class to %s'%target_class)
        state['node_colors'], state['node_weights'] = gen_node_colors(target_class)
        #state['max_edge_weight'] = get_max_edge_weight(target_class)
        edges_select_df = get_thresholded_edges(threshold=edge_threshold,target_class=target_class)
        state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(df = edges_select_df,node_positions = state['node_positions'],target_class=target_class)
    elif trigger == 'node-actmap-dropdown.value' or trigger == 'edge-actmaps-input.value':
        state['last_trigger'] = 'selection_change'
        print(edge_value)
        #update node if button value different than store value
        if state['node_select_history'][-1] != node_value:
            print('changing selected node to %s'%node_value)
            state['node_select_history'].append(node_value)
            if len(state['node_select_history']) > 10:
                del state['node_select_history'][0] 
        #update edge if button value different than store value
        if state['edge_select_history'][-1] != edge_value and check_edge_validity(edge_value.strip())[0]:
            print('changing selected edge to %s'%edge_value)
            state['edge_select_history'].append(edge_value)
            print(state['edge_select_history'])
            if len(state['edge_select_history']) > 10:
                del state['edge_select_history'][0]              

    elif trigger == 'edge-thresh-slider.value':
        print('changing edge thresholds to %s - %s'%(edge_threshold[0],edge_threshold[1]))
        state['edge_threshold'] == edge_threshold
        edges_select_df = get_thresholded_edges(threshold=edge_threshold,target_class=target_class)
        print('found %s edges'%len(edges_select_df))
        state['edge_positions'], state['edge_colors'], state['edge_widths'], state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(df = edges_select_df,node_positions = state['node_positions'],target_class=target_class)
    else:
        raise Exception('unknown trigger: %s'%trigger)    
    return state



@app.callback(
    Output('network-graph', 'figure'),
    [Input('session', 'data')],
    [State('network-graph','figure')])
def update_figure(state, fig):
    #network_graph_layout['uirevision'] = True
    print('CALLED: update_figure\n')
    print(state['edge_threshold'])
    if state['last_trigger'] == 'selection_change':   #minimal updates
        #hightlight edge
        print('updating edge highlight to %s'%state['edge_select_history'][-1])
        #if len(state['edge_select_history']) >1:
        #if state['edge_select_history'][-1] != state['edge_select_history'][-2]:  #didnt click same point
        flat_edge_names = [item for sublist in state['edge_names'] for item in sublist]
        flat_edge_colors = [item for sublist in state['edge_colors'] for item in sublist]
        try:  #update current edge if it exists to black
            print(flat_edge_names)
            fig['data'][flat_edge_names.index(state['edge_select_history'][-1])+num_layers+1]['line']['color'] = 'rgba(0,0,0,1)'
        except:
            print('select edge, %s,  not recolored as no longer shown'%state['edge_select_history'][-1])
        if len(state['edge_select_history']) > 1: #there is a previous edge to unselect
            try: #recolor previous edge if it exists from black
                fig['data'][flat_edge_names.index(state['edge_select_history'][-2])+num_layers+1]['line']['color'] = flat_edge_colors[flat_edge_names.index(state['edge_select_history'][-2])]
            except:
                print('previous edge, %s,  not recolored as no longer shown'%state['edge_select_history'][-2])
        #highlight node
        print('updating node highlight to %s'%state['node_select_history'][-1])
        #if len(state['node_select_history']) >1:
        #    if state['node_select_history'][-1] != state['node_select_history'][-2]: 
                #update current node color to black
        if str(state['node_select_history'][-1]).isnumeric():  #if normal node
            select_layer,select_position = nodeid_2_perlayerid(state['node_select_history'][-1])
            fig['data'][select_layer+1]['marker']['color'][select_position] = 'rgba(0,0,0,1)'
        else:   #imgnode
            fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-1])] = 'rgba(0,0,0,1)'
        #update previous node color to its usual color
        if len(state['node_select_history']) > 1: #there is a previous node to unselect
            if str(state['node_select_history'][-2]).isnumeric():  #if normal node
                prev_select_layer,prev_select_position = nodeid_2_perlayerid(state['node_select_history'][-2])
                fig['data'][prev_select_layer+1]['marker']['color'][prev_select_position] = state['node_colors'][prev_select_layer][prev_select_position]
            else:   #imgnode
                fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-2])] = state['imgnode_colors'][fig['data'][0]['text'].index(state['node_select_history'][-2])]
        return fig    
    else:   #regenerate full traces
        combined_traces = gen_networkgraph_traces(state)    
        layout = network_graph_layout
        #layout['uirevision'] = True
        return {'data': combined_traces,
            'layout': layout}



#CALLBACKS

@app.callback(
    Output('node-actmap-dropdown', 'value'),
    [Input('network-graph', 'clickData')],
    [State('node-actmap-dropdown', 'value')])
def switch_node_actmap_click(clickData,current_value):
    print('CALLED: switch_node_actmap_click')
    if clickData is None:
        return current_value 
        #raise Exception('no click data')
    if int(clickData['points'][0]['curveNumber']) > num_layers:
        return current_value
        #raise Exception('edge was clicked')
    return clickData['points'][0]['text']

@app.callback(
    Output('edge-actmaps-input', 'value'),
    [Input('network-graph', 'clickData')],
    [State('edge-actmaps-input', 'value'),
     State('session', 'data')])
def switch_edge_actmaps_click(clickData,current_value,state):
    print('CALLED: switch_edge_actmaps_click')
    if clickData is None:
        return current_value
        #raise Exception('no click data')
    if int(clickData['points'][0]['curveNumber']) <= num_layers:
        return current_value
        #raise Exception('node was clicked')
    return get_nth_element_from_nested_list(state['edge_names'],int(clickData['points'][0]['curveNumber'])-(num_layers+1))




@app.callback(
    Output('node-actmap-graph', 'figure'),
    [Input('node-actmap-dropdown', 'value'),
     Input('input-image-dropdown', 'value')])
def update_node_actmap(nodeid,image_name):       #EDIT: needs support for black and white images
    print('CALLED: update_node_actmap')
    layer, within_id = nodeid_2_perlayerid(nodeid)
    if layer == 'img': #code for returning color channel as activation map
        np_chan_im = get_channelwise_image(image_name,state['imgnode_names'].index(nodeid),input_image_directory=input_image_directory)
        return go.Figure(data=go.Heatmap( z = np.flip(np_chan_im,0)),
                        layout=node_actmap_layout) 
    
    return go.Figure(data=go.Heatmap( z = np.flip(activations['nodes'][layer][list_of_input_images.index(image_name)][within_id],0)),
                     layout=node_actmap_layout) 


@app.callback(
    Output('img-actmap-graph', 'figure'),
    [Input('input-image-dropdown', 'value')])
def update_inputimg_actmap(image_name): 
    print('CALLED: update_inputimg_actmap')
    return image2heatmap(input_image_directory+'/'+image_name)


@app.callback(
    Output('edge-kernel-graph', 'figure'),
    [Input('edge-actmaps-input','value')],
    [State('edge-kernel-graph','figure')])
def update_edge_kernelmap(edgename,figure):
    print('CALLED: update_edge_kernelmap')
    kernel,inmap,outmap = edgename_2_edge_figures(edgename)
    if kernel is not None:
        return go.Figure(data=go.Heatmap(z = kernel),
                         layout=kernel_layout)
    else:
        return figure
                

@app.callback(
    Output('edge-inmap-graph', 'figure'),
    [Input('edge-actmaps-input','value'),
     Input('input-image-dropdown', 'value')],
    [State('edge-inmap-graph','figure')])
def update_edge_inmap(edgename,imagename,figure):
    print('CALLED: update_edge_inmap')
    kernel,inmap,outmap = edgename_2_edge_figures(edgename,imagename=imagename)
    if inmap is not None:
        return go.Figure(data=go.Heatmap(z = inmap),
                         layout=edge_inmap_layout)
    else:
        print('edge inmap error')
        return figure
    
@app.callback(
    Output('edge-outmap-graph', 'figure'),
    [Input('edge-actmaps-input','value'),
     Input('input-image-dropdown', 'value')],
    [State('edge-outmap-graph','figure')])
def update_edge_outmap(edgename,imagename,figure):
    print('CALLED: update_edge_outmap')
    kernel,inmap,outmap = edgename_2_edge_figures(edgename,imagename=imagename)
    if outmap is not None:
        return go.Figure(data=go.Heatmap(z = outmap),
                         layout=edge_outmap_layout)
    else:
        print('edge outmap error')
        return figure
        
        



# #JSON INFO

@app.callback(
    Output('hover-data', 'children'),
    [Input('network-graph', 'hoverData')])
def display_hover_data(hoverData):
    return json.dumps(hoverData, indent=2)




@app.callback(
    Output('click-data', 'children'),
    [Input('network-graph', 'clickData')])
def display_click_data(clickData):
    return json.dumps(clickData, indent=2)


@app.callback(
    Output('selected-data', 'children'),
    [Input('network-graph', 'selectedData')])
def display_selected_data(selectedData):
    return json.dumps(selectedData, indent=2)






@app.callback(
    Output('figure-data', 'children'),
    [Input('weight-category', 'value'),
     Input('network-graph', 'clickData'),
     Input('edge-thresh-slider','value'),
     Input('session','data')])
def display_trigger(target_class,clickData,edge_thresh,state):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise Exception('no figure updates yet')
    else:
        trigger = ctx.triggered[0]['prop_id'].split('.')[0]
    ctx_msg = json.dumps({
        'states': ctx.states,
        'triggered': ctx.triggered,
        'inputs': ctx.inputs,
        'full_state':state
    }, indent=2)
    return ctx_msg
    
    

print('launching dash app')
app.run_server()



