#fetch command line argument (prepped model)
#%reset
import sys
import os
from copy import deepcopy
import pickle
import torch
import plotly.offline as py
import plotly.graph_objs as go


sys.path.insert(0, os.path.abspath('./prep_model_scripts/'))
sys.path.insert(0, os.path.abspath('./visualizer_scripts/'))
from visualizer_helper_functions import *
from contrast_helper_functions import *
from featureviz_helper_functions import *
from ablation_functions import *
from receptive_field import *
from dissected_Conv2d import *
from copy import deepcopy



prepped_model_folder = 'alexnet_sparse'    #set this to a subfolder of prunned_models

full_prepped_model_folder = os.path.abspath('prepped_models/%s'%prepped_model_folder)

possible_models = os.listdir('prepped_models')
print('possible models to visualizer are:')
print(possible_models)

print('\nYou\'ve chosen to visualize %s'%prepped_model_folder)


sys.path.insert(0,'prepped_models/%s'%prepped_model_folder)

import prep_model_params_used as prep_model_params

params = {}
params['prepped_model'] = prepped_model_folder
params['prepped_model_path'] = full_prepped_model_folder
params['device'] = 'cuda:1'

#Parameters

#Non-GUI parameters

#deepviz
params['deepviz_param'] = None
params['deepviz_optim'] = None
params['deepviz_transforms'] = None
params['deepviz_image_size'] = prep_model_params.deepviz_image_size

#backend
params['cuda'] = prep_model_params.cuda    #use gpu acceleration when running model forward
params['input_image_directory'] = prep_model_params.input_img_path+'/'   #path to directory of imput images you want fed through the network
params['preprocess'] = prep_model_params.preprocess     #torchvision transfrom to pass input images through
params['label_file_path'] = prep_model_params.label_file_path
params['criterion'] = prep_model_params.criterion
params['rank_img_path'] = prep_model_params.rank_img_path
params['num_workers'] = prep_model_params.num_workers
params['seed'] = prep_model_params.seed
params['batch_size'] = prep_model_params.batch_size
#params['dynamic_act_cache_num'] = 4  #max number of input image activations 'dynamic_activations' will have simultaneously

 
#aesthetic 

params['node_size'] = 12
params['edge_size'] = 1
params['max_node_inputs'] = 10    #there is a dropdown showing the top weighted edge inputs to nodes, how many maps in dropdown?
params['layer_colors'] = ['rgba(31,119,180,', 
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




#GUI parameters initialization (these parameters can be set in the GUI, but what values should they be initialized to?)
target_category = 'overall'     #category of images edges and nodes are weighted based on (which subgraph) 
rank_type = 'actxgrad'       #weighting criterion (actxgrad, act, grad, or weight)
projection = 'MDS smooth'           #how nodes within a layer are projected into the 2d plane (MDS or Grid)
edge_threshold = [.7,1]     #what range do edge ranks need to be in to be visualized
node_threshold = [.4,1]     #only relevant for hierarchical subgraph 

#### DONT EDIT BELOW initializations

figure_init = go.Figure()
figure_init.add_trace(go.Scatter(
            x=[],
            y=[]))
figure_init.update_layout(xaxis=dict(visible=False),
                  yaxis=dict(visible=False),
                  annotations = [dict(text="No Inputs",
                                      xref="paper",
                                      yref="paper",
                                      showarrow=False,
                                      font=dict(size=28))]
                 )

params['max_edge_weight'] = 1  #for the edge threshold slider, this dynamically adjusted its max value to max edge rank
                     #before there were multiple rank criterions, which made things confusing
                     #so well just fix it to 1 for now


#load Model

model_dis = dissect_model(deepcopy(prep_model_params.model),store_ranks=True,clear_ranks=True,cuda=params['cuda'],device=params['device']) #version of model with accessible preadd activations in Conv2d modules 
if params['cuda']:
    model_dis.cuda()
model_dis = model_dis.eval()    
model_dis.to(params['device'])

print('loaded model:')
print(prep_model_params.model)
        
#del prep_model_params.model
model = prep_model_params.model
if params['cuda']:
   model.cuda()
model = model.eval()
model.to(params['device'])


#load misc graph data
print('loading misc graph data')
misc_data = pickle.load(open('./prepped_models/%s/misc_graph_data.pkl'%prepped_model_folder,'rb'))
params['layer_nodes'] = misc_data['layer_nodes']
params['num_layers'] = misc_data['num_layers']
params['num_nodes'] = misc_data['num_nodes']
params['categories'] = misc_data['categories']
params['num_img_chan'] = misc_data['num_img_chan']
params['imgnode_positions'] = misc_data['imgnode_positions']
params['imgnode_colors'] = misc_data['imgnode_colors']
params['imgnode_names'] = misc_data['imgnode_names']
params['prepped_model_path'] = full_prepped_model_folder
params['ranks_data_path'] = full_prepped_model_folder+'/ranks/'


print('model has categories:')
print(params['categories'])

#load nodes df
print('loading nodes rank data')
target_node = 'loss'

categories_nodes_df = pd.read_csv('prepped_models/%s/ranks/categories_nodes_ranks.csv'%prepped_model_folder)
target_nodes_df = categories_nodes_df.loc[categories_nodes_df['category']==target_category]

target_nodes_df = minmax_normalize_ranks_df(target_nodes_df,params,weight=False)

weight_nodes_df = pd.read_csv('prepped_models/%s/ranks/weight_nodes_ranks.csv'%prepped_model_folder)

weight_nodes_df = minmax_normalize_ranks_df(weight_nodes_df,params,weight=True)

node_colors,node_weights = gen_node_colors(target_nodes_df,rank_type,params) 

#load node positions
print('loading node position data')
all_node_positions = pickle.load(open('./prepped_models/%s/node_positions.pkl'%prepped_model_folder,'rb'))

if projection == 'Grid':
    node_positions = all_node_positions[projection]
else:
    node_positions = all_node_positions[projection][rank_type]

#load edges
print('loading edge data')

categories_edges_df = None
if os.path.exists('prepped_models/%s/edge_ranks.csv'%prepped_model_folder):
    categories_edges_df = pd.read_csv('prepped_models/%s/ranks/categories_edges_ranks.csv'%prepped_model_folder)   #load edges

if categories_edges_df is not None:
    #overall_edges_df = categories_edges_df.loc[categories_edges_df['category']=='overall']
    target_edges_df = categories_edges_df.loc[categories_edges_df['category']==target_category]
else:
    #overall_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','overall_edges_rank.pt'))
    target_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','%s_edges_rank.pt'%target_category))

target_edges_df = minmax_normalize_ranks_df(target_edges_df,params,weight=False)

weight_edges_df = pd.read_csv('prepped_models/%s/ranks/weight_edges_ranks.csv'%prepped_model_folder)
  
weight_edges_df = minmax_normalize_ranks_df(weight_edges_df,params,weight=True)    
    
edges_thresholded_df = get_thresholded_ranksdf(edge_threshold,rank_type,target_edges_df)
 
    
num_edges = len(target_edges_df)
edges_df_columns = list(target_edges_df.columns)

edge_positions, edge_colors, edge_widths, edge_weights, edge_names, max_edge_width_indices = gen_edge_graphdata(edges_thresholded_df, node_positions, rank_type, target_category,params)


#Load Edge Kernels
print('loading convolutional kernels')
kernels = torch.load('prepped_models/%s/kernels.pt'%prepped_model_folder)



#Input Image names
params['input_image_directory'] = prep_model_params.input_img_path+'/'
params['input_image_list'] = os.listdir(params['input_image_directory'])
params['input_image_list'].sort()
input_image_name = params['input_image_list'][0]

receptive_fields = None
if os.path.exists('prepped_models/%s/receptive_fields.pkl'%prepped_model_folder):
    receptive_fields = pickle.load(open('prepped_models/%s/receptive_fields.pkl'%prepped_model_folder,'rb'))
    
input_image_size = 224   #got to figure out a way not to hard-code this
receptive_fields

#Format Node Feature Maps
print('loading activation maps')

all_activations = {'nodes':{},'edges_in':{},'edges_out':{}}
if os.path.exists('prepped_models/%s/input_img_activations.pt'%prepped_model_folder):
    all_activations = torch.load('prepped_models/%s/input_img_activations.pt'%prepped_model_folder)


#hidden state, stores python values within the html itself
state = {'projection':projection,'rank_type':rank_type,'edge_positions':edge_positions,'edge_colors': edge_colors, 'edge_widths':edge_widths,'edge_names':edge_names,
         'edge_threshold':edge_threshold,'edge_weights':edge_weights,'max_edge_width_indices':max_edge_width_indices,
         'node_positions':node_positions,'node_colors':node_colors,'node_weights':node_weights,'node_threshold':node_threshold,'target_category':target_category,'target_node':'loss',
         'node_select_history':['0'],'edge_select_history':[edge_names[0][0]],'last_trigger':None,'input_image_name':input_image_name,
         'imgnode_positions':params['imgnode_positions'],'imgnode_colors':params['imgnode_colors'],'imgnode_names':params['imgnode_names']}


#App Component Layouts
axis=dict(showbackground=False,
          showspikes=False,
          showline=False,
          zeroline=False,
          showgrid=False,
          showticklabels=False,
          #range=[0,0],
          title=''
          )

camera = dict(
    up=dict(x=0, y=0, z=1),
    center=dict(x=0, y=0, z=0),
    eye=dict(x=-1.00, y=-1.25, z=1.25)
)


network_graph_layout = go.Layout(
         #title="%s through Prunned Cifar10 CNN"%target_category,
         #title = target_category,
         #width=1000,
         clickmode = 'event+select',
         transition = {'duration': 20},
         height=500,
         #showlegend=False,
         margin = dict(l=20, r=20, t=20, b=20),
         scene=dict(
             xaxis=dict(axis),
             yaxis=dict(axis),
             zaxis=dict(axis),
             aspectmode ="manual", 
             aspectratio = dict(x=1, y=0.5, z=0.5) #adjusting this stretches the network layer-to-layer
         ),
         scene_camera = camera,
         uirevision =  True   
         #hovermode='closest',
   )


input_image_layout = go.Layout(#width=200, 
                      #height=200,
                      uirevision = True,
                      margin=dict(
                        l=12,
                        r=1,
                        b=12,
                        t=1,
                        pad=10
                        ),
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        xaxis=dict(range=(0,10),showline=False,showgrid=False,showticklabels=False),
                        yaxis=dict(range=(0,10),showline=False,showgrid=False,showticklabels=False))


node_actmap_layout = go.Layout(
    #autosize=False,
    #width=270,
    #height=200,
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
    #autosize=False,
    #width=270,
    #height=200,
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
    #autosize=False,
    #width=270,
    #height=200,
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
    #autosize=False,
    #width=180,
    #height=120,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ))


#Generate Network Graph
combined_traces = gen_networkgraph_traces(state,params)
network_graph_fig=go.Figure(data=combined_traces, layout=network_graph_layout)

#Dash App Setup
print('setting up dash app')

import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.exceptions import PreventUpdate
#import utils.dash_reusable_components as drc
import flask
import os

import json

from dash.dependencies import Input, Output, State

from plotly.subplots import make_subplots

from flask_caching import Cache

#external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(external_stylesheets = external_stylesheets)



if not os.path.exists(full_prepped_model_folder+'/cache/'):
    os.mkdir(full_prepped_model_folder+'/cache/')
CACHE_CONFIG = {
    # try 'filesystem' if you don't want to setup redis
    'CACHE_TYPE': 'filesystem',
    'CACHE_DIR': full_prepped_model_folder+'/cache/'}
cache = Cache()
cache.init_app(app.server, config=CACHE_CONFIG)
    


styles = {
    'pre': {
        'border': 'thin lightgrey solid',
        'overflowX': 'scroll'
    }
}


theme =  {
    'dark': True,
    'detail': '#007439',
    'primary': '#00EA64',
    'secondary': '#6E6E6E',
}


app.layout = html.Div([
        html.Div(
            children = [
                
            html.Div(
                #Left side control panel
                children = [
                 html.Label('Subgraph Controls', style={'fontSize': 18,'font-weight':'bold'}),
                 html.Br(),
                 html.Label('Input'),
                 #dcc.Dropdown(
                 #  id='weight-category',
                 #  options=[{'label': i, 'value': i} for i in params['categories']],
                 #   value=target_category
                 #   ),
                dcc.Input(id='input-category',value=state['target_category']),
                html.Br(),
                html.Br(),
                html.Label('Output'),
                 #dcc.Dropdown(
                 #  id='weight-category',
                 #  options=[{'label': i, 'value': i} for i in params['categories']],
                 #   value=target_category
                 #   ),
                dcc.Dropdown(
                    id='target-node',
                    options=[
                    {'label': i, 'value': i} for i in ['loss']+[str(node) for node in list(range(params['num_nodes']))]
                    ],
                    value=state['target_node']),
                 html.Br(),
                 html.Label('Subgraph Criterion'),
                 dcc.Dropdown(
                    id='subgraph-criterion',
                    options=[
                        {'label': 'Activations*Grads', 'value': 'actxgrad'},
                        {'label': 'Activations', 'value': 'act'},
                        {'label': 'Gradients', 'value': 'grad'},
                        {'label': 'Weights', 'value': 'weight'},
                        {'label': 'Hierarchical', 'value': 'hierarchical'}
                        
                    ],
                    value='actxgrad'
                    ),
                 html.Br(),   
                 html.Label('Layer Projection'),
                 dcc.Dropdown(
                    id = 'layer-projection',
                    options=[
                        {'label': 'MDS', 'value': 'MDS'},
                        {'label': 'MDS smooth', 'value': 'MDS smooth'},
                        {'label': 'Grid', 'value': 'Grid'},
                        #{'label': 'SOM', 'value': 'SOM'}
                    ],
                    value='MDS smooth'
                    ),

                html.Br(),
                html.Label('Edge Thresholds'),
                    dcc.RangeSlider(
                        id='edge-thresh-slider',
                        min=0,
                        max=np.ceil(params['max_edge_weight']*10)/10,
                        step=0.001,
                        marks={i/10: str(i/10) for i in range(0,int(np.ceil(params['max_edge_weight']*10))+1,int(round(np.ceil(params['max_edge_weight']*10)/5)))},
                        value=edge_threshold,
                    ),
                html.Label('Node Thresholds'),
                    dcc.RangeSlider(
                        id='node-thresh-slider',
                        min=0,
                        max=1,
                        step=0.001,
                        marks={i/10: str(i/10.0) for i in range(0,11)},
                        value=node_threshold,
                    ),

                ], className="two columns",
                ),
                
            html.Div([
                dcc.Graph(
                    id='network-graph',
                    figure=network_graph_fig
                )
                ], className= 'ten columns'
                ),
            ], className="row"
        ),


                
        html.Div([
            html.Div([
            html.Label('Input Image', style={'fontSize': 18,'font-weight':'bold'}),
            dcc.Dropdown(id="dynamic-input-image-dropdown",value=params['input_image_list'][0]),
            #dcc.Dropdown(
            #    id='input-image-dropdown',
            #    options=[{'label': i, 'value': i} for i in params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/')],
            #    value=input_image_name
            #),
            html.Br(),
            dcc.Graph(
                id='img-actmap-graph',
                style={
               'width': '14vw',
               'height':'14vw'
                },
                figure=image2heatmap(params['input_image_directory']+input_image_name,input_image_layout),
                config={
                        'displayModeBar': False
                        }
            )
            ], className = "two columns"),

            html.Div([
            html.Label('Node', style={'fontSize': 18,'font-weight':'bold'}),
            dcc.Dropdown(
                id='node-actmap-dropdown',
                options=[{'label': str(j), 'value': str(j)} for j in params['imgnode_names']]+[{'label': str(i), 'value': str(i)} for i in range(params['num_nodes'])],
                value='0'
            ),
            html.Br(),
            dcc.Graph(
                id='node-actmap-graph',
                style={
               'width': '18vw',
               'height':'14vw'
                },
                figure=figure_init,
                config={
                        'displayModeBar': False
                        }
            ),
            dcc.Checklist(
                id = 'relu-checkbox',
                options = [{'label':'relu','value':'relu'}],
                value = []
                
            ),
            html.Div(id='node-sum', style={'whiteSpace': 'pre-line'}),
            html.Br(),
            html.Br(),
            dcc.Graph(
                id='node-deepviz-image',
                style={
               'width': '14vw',
               'height':'14vw'
                },
                figure=figure_init,
                config={
                        'displayModeBar': False
                        }
            )
            ], className = "three columns"),
            
            html.Div([
            html.Label('Node Inputs', style={'fontSize': 18,'font-weight':'bold'}),
            html.Br(),
            html.Div(dcc.Graph(
                id='node-inputs-graph',
                figure=figure_init,
                config={
                        'displayModeBar': False
                        }
            ),style={'overflowY': 'scroll', 'height': 500})
            ], className = "three columns"),

            html.Div([
            html.Label('Edge', style={'fontSize': 18,'font-weight':'bold'}),    
            dcc.Input(
                id='edge-actmaps-input',value=state['edge_names'][0][0], type='text'),
            #html.Button(id='edge-kernel-button',n_clicks=0, children='Submit'),
            html.Br(),
            html.Br(),
            html.Label('Kernel'),
            dcc.Graph(
                id='edge-kernel-graph',
                style={
               'width': '14vw',
               'height':'10vw'
                },
                figure=go.Figure(data=go.Heatmap(
                                    z = edgename_2_edge_figures(state['edge_names'][0][0], input_image_name, kernels, None,params)[0]),
                                 layout=kernel_layout
                                ),
                config={
                        'displayModeBar': False
                        }
            ),
            html.Br(),
            html.Br(),
            dcc.Graph(
               id='edge-deepviz-image',
               style={
              'width': '14vw',
              'height':'14vw'
               },
               figure=figure_init,
               config={
                       'displayModeBar': False
                       }
            )
            ], className = "two columns"),


            html.Div([
            html.Label('Edge Input'),
            html.Br(),
            dcc.Graph(
                id='edge-inmap-graph',
                style={
               'width': '18vw',
               'height':'14vw'
                },
                figure=figure_init,
                config={
                        'displayModeBar': False
                        }
            ),
            html.Div(id='edgein-sum', style={'whiteSpace': 'pre-line'}),
            html.Br(),
            html.Br(),
            html.Label('Edge Output'),
            html.Br(),
            dcc.Graph(
                id='edge-outmap-graph',
                style={
               'width': '18vw',
               'height':'14vw'
                },
                figure=figure_init,
                config={
                        'displayModeBar': False
                        }
            ),
            html.Div(id='edgeout-sum', style={'whiteSpace': 'pre-line'}),

            ], className = "two columns")


         ], className= 'row'
         ),
    
    
    html.Div([
            html.Div([
            html.Label('Image Manipulations', style={'fontSize': 18,'font-weight':'bold'}),
            html.Br(),
            html.Label('rotation'),
            dcc.Slider(
                id='image-rotation-slider',
                min=0,
                max=350,
                step=10,
                marks={
                        0:   '0°',
                        20:  '20°',
                        40:  '40°',
                        60:  '60°',
                        80:  '80°',
                        100: '100°',
                        120: '120°',
                        140: '140°',
                        160: '160°',
                        180: '180°',
                        200: '200°',
                        220: '220°',
                        240: '240°',
                        260: '260°',
                        280: '280°',
                        300: '300°',
                        320: '320°',
                        340: '240°',
                        },
                included=False,
                value=0,
            ),
            html.Br(),
            html.Label('scaling'),
            dcc.Slider(
                id='image-scaling-slider',
                min=-10,
                max=10,
                step=1,
                marks={
                        -8:  '.33',
                        -6:  '.4',
                        -4: '.5',
                        -2: '.67',
                         0: '1',
                         2: '1.5',
                         4: '2',
                         6: '2.5',
                         8: '3',
                        },
                included=False,
                value=0,
            ),            
            html.Br(),
            html.Label('colors'),

                    html.Label('R',style={'fontSize': 10,'font-weight':'italic'}),
                    dcc.Slider(
                        id='image-r-slider',
                        min=-1,
                        max=1,
                        step=.05,
                        marks={
                                -1:'-1',
                                -.8:'-.8',
                                -.6:'-.6',
                                -.4:'-.4',
                                 -.2:'-.2',
                                 0:'0',
                                 .2:'.2',
                                 .4:'.4',
                                 .6:'.6',
                                 .8:'.8',
                                  1:'1',
                                },
                        included=False,
                        value=0,
                    ),
 

                    html.Label('G',style={'fontSize': 10,'font-weight':'italic'}),
                    dcc.Slider(
                        id='image-g-slider',
                        min=-1,
                        max=1,
                        step=.05,
                        marks={
                                -1:'-1',
                                -.8:'-.8',
                                -.6:'-.6',
                                -.4:'-.4',
                                 -.2:'-.2',
                                 0:'0',
                                 .2:'.2',
                                 .4:'.4',
                                 .6:'.6',
                                 .8:'.8',
                                  1:'1',
                                },
                        included=False,
                        value=0,
                    ),
    

                    html.Label('B',style={'fontSize': 10,'font-weight':'italic'}),
                    dcc.Slider(
                        id='image-b-slider',
                        min=-1,
                        max=1,
                        step=.05,
                        marks={
                                -1:'-1',
                                -.8:'-.8',
                                -.6:'-.6',
                                -.4:'-.4',
                                 -.2:'-.2',
                                 0:'0',
                                 .2:'.2',
                                 .4:'.4',
                                 .6:'.6',
                                 .8:'.8',
                                  1:'1',
                                },
                        included=False,
                        value=0,
                    )
           
            ], className = "three columns"),
                
                
            html.Div([
            html.Label('Feature Visualizations', style={'fontSize': 18,'font-weight':'bold'}),
            html.Br(),
            html.Div( style=dict(display='flex'),
                children = [     
                    daq.ToggleSwitch(
                        id='featviz-nodeedge-toggle',
                        label=['node','edge    '],
                        style={'float': 'right','margin': 'auto'}
                        #labelPosition='bottom'
                    ), 
                    html.Label(''),
                    daq.ToggleSwitch(
                        id='featviz-channelneuron-toggle',
                        label=['channel','neuron    '],
                        style={'float': 'right','margin': 'auto'}
                        #labelPosition='bottom'
                    ),
                    html.Label(''),
                    daq.ToggleSwitch(
                        id='featviz-positivenegative-toggle',
                        label=['positive','negative    '],
                        style={'float': 'right','margin': 'auto'}
                        #labelPosition='bottom'
                    )
                ]),
            html.Br(),
            dcc.Graph(
               id='featviz-image',
               style={
              'width': '14vw',
              'height':'14vw'
               },
               figure=figure_init,
               config={
                       'displayModeBar': False
                       }
            ),
            html.Button('Generate', id='featviz-button')
            #html.Button('Generate', id='gen-featviz-button')
            ], className= "five columns"),
        
        
        
            html.Div([
            html.Label('Model Ablations', style={'fontSize': 18,'font-weight':'bold'}),
            dcc.Textarea(
                id='ablations-textarea',
                value='',
                style={'width': '70%', 'height': 300}),
            html.Button('Ablate', id='ablate-model-button')
            ], className= "four columns"),
        
        ], className="row"
        ),
                
#         html.Div([
#             html.Div([
#                 dcc.Markdown("""
#                     **Hover Data**

#                     Mouse over values in the graph.
#                 """),
#                 html.Pre(id='hover-data', style=styles['pre'])
#             ], className='two columns'),

#             html.Div([
#                 dcc.Markdown("""
#                     **Click Data**

#                     Click on points in the graph.
#                 """),
#                 html.Pre(id='click-data', style=styles['pre']),
#             ], className='two columns'),

#             html.Div([
#                 dcc.Markdown("""
#                     **Selection Data**

#                     Choose the lasso or rectangle tool in the graph's menu
#                     bar and then select points in the graph.

#                     Note that if `layout.clickmode = 'event+select'`, selection data also 
#                     accumulates (or un-accumulates) selected data if you hold down the shift
#                     button while clicking.
#                 """),
#                 html.Pre(id='selected-data', style=styles['pre']),
#             ], className='two columns'),

# #                 html.Div([
# #                     dcc.Markdown("""
# #                         **Zoom and Relayout Data**

# #                         Click and drag on the graph to zoom or click on the zoom
# #                         buttons in the graph's menu bar.
# #                         Clicking on legend items will also fire
# #                         this event.
# #                     """),
# #                     html.Pre(id='relayout-data', style=styles['pre']),
# #                 ], className='two columns')
                
#             html.Div([
#                 dcc.Markdown("""
#                     **Figure Data**

#                     Figure json info.
#                 """),
#                 html.Pre(id='figure-data', style=styles['pre']),
#             ], className='four columns')

#         ], className= 'row'
#         ),

    #hidden divs for storing intermediate values     
    # The memory store reverts to the default on every page refresh
    dcc.Store(id='memory',data=state),
    # The local store will take the initial data
    # only the first time the page is loaded
    # and keep it until it is cleared.
    #dcc.Store(id='local', storage_type='local'),
    # Same as the local store but will lose the data
    # when the browser/tab closes.
    #dcc.Store(id='session', storage_type='session',data=state),
    

    # hidden signal value
    html.Div(id='input-image-signal',  style={'display': 'none'}),
    html.Div(id='target-signal', style={'display': 'none'},children = [state['target_category'],state['target_node']]),
    html.Div(id='ablations-signal',  style={'display': 'none'}, children = [])
])



# perform expensive computations in this "global store"
# these computations are cached in a globally available
# 'cached' folder in the prepped_models/[model] folder
@cache.memoize()
def activations_store(image_name,ablation_list):

    print('Updating cached activations with {}'.format(image_name))
    activations = get_model_activations_from_image(get_image_path(image_name,params)[1], model_dis, params)
    
    return activations

@app.callback(Output('input-image-signal', 'children'), 
              [Input('dynamic-input-image-dropdown', 'value'),
               Input('ablations-signal', 'children')])
def update_activations_store(image_name,ablation_list):
    # compute value and send a signal when done
    activations_store(image_name,ablation_list)
    return image_name


@cache.memoize()
def ranksdf_store(target_category, target_node,ablation_list,model_dis=model_dis):
    print('Updating cached rank dfs with {}'.format(target_category))
    model_dis = clear_ranks_across_model(model_dis)
    target_type = image_category_or_contrast(target_category,params)
    target_category_nodes_df = None
    target_category_edges_df = None
    if target_type == 'category' and target_node == 'loss' and ablation_list == []:
        #edges
        if categories_edges_df is not None:
            if len(categories_edges_df.loc[categories_edges_df['category']==target_category]) > 0:
                target_category_edges_df = categories_edges_df.loc[categories_edges_df['category']==target_category]
        if target_category_edges_df is None:
            target_category_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','%s_edges_rank.pt'%target_category))   
        #node
        if categories_nodes_df is not None:
            if len(categories_nodes_df.loc[categories_nodes_df['category']==target_category]) > 0:
                target_category_nodes_df = categories_nodes_df.loc[categories_nodes_df['category']==target_category]
        if target_category_nodes_df is None:
            target_category_nodes_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_nodes','%s_nodes_rank.pt'%target_category))
    elif target_type == 'category':
        target_category_nodes_df,target_category_edges_df = rank_dict_2_df(get_model_ranks_for_category(target_category, target_node, model_dis,params))
    elif target_type == 'input_image':
        target_category_nodes_df,target_category_edges_df = rank_dict_2_df(get_model_ranks_from_image(get_image_path(target_category,params)[1],target_node, model_dis, params))

    else:  #contrast
        target_category_nodes_df,target_category_edges_df = contrast_str_2_dfs(target_category,target_node,model_dis,params,ablation_list)
    print('FROM RANKS DF STORE')
    print(target_category_edges_df)
    return target_category_nodes_df,target_category_edges_df

@app.callback(Output('target-signal', 'children'), 
              [Input('input-category', 'value'),
               Input('target-node','value'),
               Input('ablations-signal', 'children')])
def update_ranksdf_store(target_category,target_node,ablation_list):
    # compute value and send a signal when done
    print('update ranksdf_store triggered')
    ranksdf_store(target_category,target_node,ablation_list)
    return [target_category,target_node]



####Call Back Functions

#Ablations
@app.callback(Output('ablations-signal', 'children'), 
              [Input('ablate-model-button', 'n_clicks')],
              [State('ablations-textarea','value')])
def update_ablations(n_clicks,text,model_dis=model_dis):
    # compute value and send a signal when done
    ablation_list = ablation_text_2_list(text, params)
    ablate_model_with_list(ablation_list,model_dis,params)
    return ablation_list


#Hidden State
@app.callback(
    Output('memory', 'data'),
    [Input('target-signal', 'children'),
     Input('node-actmap-dropdown', 'value'),
     Input('edge-actmaps-input', 'value'),
     Input('edge-thresh-slider','value'),
     Input('node-thresh-slider','value'),
     Input('layer-projection','value'),
     Input('subgraph-criterion','value')],
    [State('memory', 'data'),
     State('ablations-signal', 'children')])
def update_store(target,node_value,edge_value,edge_threshold,node_threshold,projection,rank_type,state,ablation_list):
    print('CALLED: update_store\n')
    ctx = dash.callback_context
    if not ctx.triggered:
        raise Exception('no figure updates yet')
    else:
        trigger = ctx.triggered[0]['prop_id']
    state['last_trigger'] = trigger  #store the last trigger of state change in state
    print('TRIGGER %s'%trigger)
    
    hierarchical = False
    if rank_type == 'hierarchical':
        hierarchical = True
        rank_type = 'actxgrad'
        
    target_category,target_node = target[0],target[1]
    #fetch select edges DF
    if trigger in ['target-signal.children','edge-thresh-slider.value','node-thresh-slider.value','layer-projection.value','subgraph-criterion.value']:
        if rank_type == 'weight':
            target_edges_df = weight_edges_df
            target_nodes_df = weight_nodes_df
            weight=True
        else:   
            target_nodes_df,target_edges_df = ranksdf_store(target_category,target_node,ablation_list)
            weight=False   
        target_edges_df = minmax_normalize_ranks_df(target_edges_df,params,weight=weight)
        target_nodes_df = minmax_normalize_ranks_df(target_nodes_df,params,weight=weight)

        if hierarchical:
            #nodes_thresholded_df = get_thresholded_ranksdf(node_threshold,rank_type, target_nodes_df)
            #filter_edges_df = filter_edges_by_nodes(target_edges_df,nodes_thresholded_df)
            #edges_thresholded_df = get_thresholded_ranksdf(edge_threshold,rank_type,filter_edges_df)
            #edges_thresholded_df = hierarchically_threshold_edges(edge_threshold,rank_type,target_edges_df,nodes_thresholded_df)
            print('finding hierarchical subgraph')
            start = time.time()
            nodes_thresholded_df,edges_thresholded_df = hierarchical_accum_threshold(node_threshold[0],edge_threshold[0],rank_type,target_edges_df,target_nodes_df,ascending=False)
            print('time: %s'%str(time.time() - start))
            print('found %s nodes and %s edges'%(str(len(nodes_thresholded_df)),str(len(edges_thresholded_df))))
            #node_minmax = node_threshold
            node_min = {}
            for layer in target_nodes_df['layer'].unique():
                if len(nodes_thresholded_df.loc[nodes_thresholded_df['layer']==layer]) > 1:
                    node_min[layer] = nodes_thresholded_df.loc[nodes_thresholded_df['layer']==layer][rank_type+'_rank'].min()
                else:
                    node_min[layer] = None

        else: 
            nodes_thresholded_df = None
            edges_thresholded_df = get_thresholded_ranksdf(edge_threshold,rank_type,target_edges_df)
            node_min = None

    if trigger == 'target-signal.children':
        print('changing target category to %s'%target_category)
        #print(target_nodes_df)
        state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params,node_min=node_min)
        #state['max_edge_weight'] = get_max_edge_weight(target_category)
        state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)

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
        if state['edge_select_history'][-1] != edge_value and check_edge_validity(edge_value.strip(),params)[0]:
            print('changing selected edge to %s'%edge_value)
            state['edge_select_history'].append(edge_value)
            print(state['edge_select_history'])
            if len(state['edge_select_history']) > 10:
                del state['edge_select_history'][0]              

    elif trigger == 'edge-thresh-slider.value':
        print('changing edge thresholds to %s - %s'%(edge_threshold[0],edge_threshold[1]))
        state['edge_threshold'] == edge_threshold
        print('found %s edges'%len(edges_thresholded_df))
        state['edge_positions'], state['edge_colors'], state['edge_widths'], state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)
    
    elif trigger == 'node-thresh-slider.value':
        print('changing node thresholds to %s - %s'%(node_threshold[0],node_threshold[1]))
        state['node_threshold'] == node_threshold
        print('found %s nodes'%len(nodes_thresholded_df))
        state['edge_positions'], state['edge_colors'], state['edge_widths'], state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)
        state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params,node_min=node_min)
        
        
    elif trigger == 'layer-projection.value':
        print('changing layer projection to %s\n'%projection)
        state['projection']=projection
        if projection == 'Grid':
            node_positions = all_node_positions[projection]
        else:
            node_positions = all_node_positions[projection][rank_type]
        state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)

    elif trigger == 'subgraph-criterion.value':
        print('changing weighting criterion to %s\n'%rank_type)
        state['rank_type']=rank_type
        state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params,node_min=node_min)
        #state['node_positions']=format_node_positions(projection=projection,rank_type=rank_type)
        state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)

    else:
        raise Exception('unknown trigger: %s'%trigger)    
    return state


#Network Graph Figure
@app.callback(
    Output('network-graph', 'figure'),
    [Input('memory', 'data')],
    [State('network-graph','figure')])
def update_figure(state, fig):
    #network_graph_layout['uirevision'] = True
    print('CALLED: update_figure\n')
    print(state['edge_threshold'])
    print(state['edge_select_history'])
    print(state['node_select_history'])
    if state['last_trigger'] == 'selection_change':   #minimal updates
        #hightlight edge
        print('updating edge highlight to %s'%state['edge_select_history'][-1])
        #if len(state['edge_select_history']) >1:
        #if state['edge_select_history'][-1] != state['edge_select_history'][-2]:  #didnt click same point
        flat_edge_names = [item for sublist in state['edge_names'] for item in sublist]
        flat_edge_colors = [item for sublist in state['edge_colors'] for item in sublist]
        try:  #update current edge if it exists to black
            #print(flat_edge_names)
            fig['data'][flat_edge_names.index(state['edge_select_history'][-1])+params['num_layers']+1]['line']['color'] = 'rgba(0,0,0,1)'
        except:
            print('select edge, %s,  not recolored as no longer shown'%state['edge_select_history'][-1])
        if len(state['edge_select_history']) > 1: #there is a previous edge to unselect
            try: #recolor previous edge if it exists from black
                fig['data'][flat_edge_names.index(state['edge_select_history'][-2])+params['num_layers']+1]['line']['color'] = flat_edge_colors[flat_edge_names.index(state['edge_select_history'][-2])]
            except:
                print('previous edge, %s,  not recolored as no longer shown'%state['edge_select_history'][-2])
        #highlight node
        print('updating node highlight to %s'%state['node_select_history'][-1])
        #if len(state['node_select_history']) >1:
        #    if state['node_select_history'][-1] != state['node_select_history'][-2]: 
                #update current node color to black
        if str(state['node_select_history'][-1]).isnumeric():  #if normal node
            select_layer,select_position,select_layer_name = nodeid_2_perlayerid(state['node_select_history'][-1],params)
            fig['data'][select_layer+1]['marker']['color'][select_position] = 'rgba(0,0,0,1)'
        else:   #imgnode
            fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-1])] = 'rgba(0,0,0,1)'
        #update previous node color to its usual color
        if len(state['node_select_history']) > 1: #there is a previous node to unselect
            if str(state['node_select_history'][-2]).isnumeric():  #if normal node
                prev_select_layer,prev_select_position,prev_select_layer_name = nodeid_2_perlayerid(state['node_select_history'][-2],params)
                print(prev_select_layer,prev_select_position,prev_select_layer_name)
                fig['data'][prev_select_layer+1]['marker']['color'][prev_select_position] = state['node_colors'][prev_select_layer][prev_select_position]
            else:   #imgnode
                fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-2])] = state['imgnode_colors'][fig['data'][0]['text'].index(state['node_select_history'][-2])]
        #fig['layout']['uirevision']=True   
        return fig    
    else:   #regenerate full traces
        combined_traces = gen_networkgraph_traces(state,params)
        fig['data'] = combined_traces
        #layout = network_graph_layout
        #layout['uirevision'] = True
        return fig

#Node Actmap Dropdown
@app.callback(
    Output('node-actmap-dropdown', 'value'),
    [Input('network-graph', 'clickData')],
    [State('node-actmap-dropdown', 'value')])
def switch_node_actmap_click(clickData,current_value):
    print('CALLED: switch_node_actmap_click')
    if clickData is None:
        return current_value 
        #raise Exception('no click data')
    if int(clickData['points'][0]['curveNumber']) > params['num_layers']:
        return current_value
        #raise Exception('edge was clicked')
    return clickData['points'][0]['text']

#Edge Actmaps Input
@app.callback(
    Output('edge-actmaps-input', 'value'),
    [Input('network-graph', 'clickData')],
    [State('edge-actmaps-input', 'value'),
     State('memory', 'data')])
def switch_edge_actmaps_click(clickData,current_value,state):
    print('CALLED: switch_edge_actmaps_click')
    if clickData is None:
        return current_value
        #raise Exception('no click data')
    if int(clickData['points'][0]['curveNumber']) <= params['num_layers']:
        return current_value
        #raise Exception('node was clicked')
    return get_nth_element_from_nested_list(state['edge_names'],int(clickData['points'][0]['curveNumber'])-(params['num_layers']+1))


#Node actmap graph
@app.callback(
    Output('node-actmap-graph', 'figure'),
    [Input('node-actmap-dropdown', 'value'),
     Input('relu-checkbox','value'),
     Input('input-image-signal', 'children')],
    [State('ablations-signal', 'children')])
def update_node_actmap(nodeid,relu_checked,image_name,ablation_list):       #EDIT: needs support for black and white images
    print('CALLED: update_node_actmap')
    layer, within_id,layer_name = nodeid_2_perlayerid(nodeid,params)
    #fetch activations
    if image_name in all_activations['nodes'] and ablation_list == []:
        activations = all_activations
    else:
        activations  = activations_store(image_name,ablation_list)
        
    if layer == 'img': #code for returning color channel as activation map
        #np_chan_im = get_channelwise_image(image_name,state['imgnode_names'].index(nodeid),params['input_image_directory']=params['input_image_directory'])
        np_chan_im = activations['edges_in'][image_name][0][within_id]
        return go.Figure(data=go.Heatmap( z = np.flip(np_chan_im,0), name = nodeid),
                        layout=node_actmap_layout) 
    act_map = activations['nodes'][image_name][layer][within_id]
    if relu_checked != []:
        act_map = relu(act_map)
    return go.Figure(data=go.Heatmap( z = np.flip(act_map,0),
                                      #zmin=-11,
                                      #zmax=14,
                                      colorbar = dict(thicknessmode = "fraction",thickness=.1)
                                    ),
                     layout=node_actmap_layout) 

@app.callback(
    Output('node-sum', 'children'),
    [Input('node-actmap-graph', 'figure')])
def update_node_sum(fig):
    mean = np.mean(fig['data'][0]['z'])
    return 'mean: %s'%str(mean)


#Node deepviz graph
@app.callback(
    Output('node-deepviz-image', 'figure'),
    [Input('node-actmap-dropdown', 'value')])
def update_node_deepviz(nodeid):       #EDIT: needs support for black and white images
    print('CALLED: update_node_deepviz')
    layer,within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)    
    if layer == 'img': 
        return figure_init
    image_name = fetch_deepviz_img(model,str(nodeid),params)
    image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
    return image2plot(image_path,input_image_layout)
    

#Edge deepviz graph
@app.callback(
    Output('edge-deepviz-image', 'figure'),
    [Input('edge-actmaps-input', 'value')])
def update_edge_deepviz(edgename):       #EDIT: needs support for black and white images
    print('CALLED: update_edge_deepviz')
    #layer,within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)    
    #if layer == 'img': 
    #    return figure_init
    image_name = fetch_deepviz_img(model_dis,edgename,params)
    image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
    return image2plot(image_path,input_image_layout)
     

#Node inputs actmap graph
@app.callback(
    Output('node-inputs-graph', 'figure'),
    [Input('node-actmap-dropdown', 'value'),
     Input('input-image-signal', 'children'),
     Input('target-signal', 'children'),
     Input('subgraph-criterion','value')],
     [State('ablations-signal', 'children')])
def update_node_inputs(nodeid,image_name,target,rank_type,ablation_list,model=model_dis,max_num = params['max_node_inputs']):       
    print('CALLED: update_node_inputs')
    
    hierarchical = False
    if rank_type == 'hierarchical':
        hierarchical = True
        rank_type = 'actxgrad'
    
    
    target_category,target_node = target[0],target[1]
    node_layer,node_within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)
    #fetch activations
    if image_name in all_activations['nodes'] and ablation_list == []:
        activations = all_activations
    else:
        activations = activations_store(image_name,ablation_list)
    #fetch edges df
    if rank_type == 'weight':
        target_edges_df = weight_edges_df
    else:
        target_edges_df = ranksdf_store(target_category,target_node,ablation_list)[1]
    #return no input if on input image node 
    if node_layer == 'img':
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=[],
            y=[]))
        fig.update_layout(xaxis=dict(visible=False),
                          yaxis=dict(visible=False),
                          annotations = [dict(text="No Inputs",
                                              xref="paper",
                                              yref="paper",
                                              showarrow=False,
                                              font=dict(size=28))]
                         )
        return fig

    all_node_edges_df = target_edges_df.loc[(target_edges_df['layer']==node_layer) & (target_edges_df['out_channel'] == node_within_layer_id)]
    #if sort_images:                      
    all_node_edges_df = all_node_edges_df.sort_values(by=[rank_type+'_rank'],ascending=False)
    top_node_edges_df = all_node_edges_df.head(max_num)
    fig = make_subplots(rows=len(top_node_edges_df)+1, cols=3)
    #print(top_node_edges_df)
    i=1
    for row in top_node_edges_df.itertuples():
        if node_layer == 0:
            edge_name = str(params['imgnode_names'][row.in_channel])+'-'+str(nodeid)
        else:
            edge_name = str(params['layer_nodes'][node_layer-1][1][row.in_channel])+'-'+str(nodeid)
        #add activation map
        fig.add_trace(
               go.Heatmap(z = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)[2],
                          #zmin = -1,
                          #zmax = 1,
                          name = edge_name,
                          coloraxis="coloraxis"
                          #showscale = False,
                          #colorbar = dict(lenmode='fraction',len=1/len(top_node_edges_df), 
                          #                y=(i)/len(top_node_edges_df)-.01,
                          #                thicknessmode = "fraction",thickness=.1,
                          #                ypad=1
                          #               )
                          ),
               row=i, col=2),
        #add kernel
        fig.add_trace(
               go.Heatmap(z = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)[0],
                          #zmin = -1,
                          #zmax = 1,
                          name = edge_name+'_kernel',
                          coloraxis="coloraxis2"
                          #showscale = False,
                          #colorbar = dict(lenmode='fraction',len=1/len(top_node_edges_df), 
                          #                y=(i)/len(top_node_edges_df)-.01,
                          #                thicknessmode = "fraction",thickness=.1,
                          #                ypad=1
                          #               )
                          ),
               row=i, col=3),

        #add visualization
        viz_img_name = fetch_deepviz_img_for_node_inputs(model,edge_name,params)
        viz_img_path = params['prepped_model_path']+'/visualizations/images/'+viz_img_name
        viz_img = Image.open(viz_img_path)
        #fig.add_trace(go.Image(z=viz_img,name=viz_img_name), row=i, col=1)
        fig.add_trace(go.Scatter(x=[],y=[]),row=i,col=1)
        fig.add_layout_image(
                            source=viz_img,
                            xref="x",
                            yref="y",
                            x=0,
                            y=10,
                            sizex=10,
                            sizey=10,
                            sizing="stretch",
                            opacity=1,
                            layer="below",
                            row=i, col=1
                            )
        fig.update_xaxes(visible=False,range=(0,10),showline=False,showgrid=False,showticklabels=False,row=i,col=1)
        fig.update_yaxes(visible=False,range=(0,10),showline=False,showgrid=False,showticklabels=False,row=i,col=1)
   


        i+=1
    
    fig.update_layout(height=200*len(top_node_edges_df), 
                      width=340,
                      #yaxis=dict(scaleanchor="x", scaleratio=1/len(top_node_edges_df)),
                      #title_text="Inputs to Node",
                      #xaxis=dict(visible=False),
                      #yaxis=dict(visible=False),
                      coloraxis_showscale=False,
                      coloraxis2 = dict(showscale=False,
                                        colorscale='inferno',
                                        colorbar = dict(
                                                        thicknessmode = "fraction",thickness=.05, 
                                                        lenmode='fraction',len=.7)),
                      margin=dict(
                                    l=0,
                                    r=0,
                                    b=0,
                                    t=0,
                                    pad=0)
                     )
    fig.update_coloraxes(colorscale='inferno',colorbar = dict(
                                                              thicknessmode = "fraction",thickness=.05, 
                                                              lenmode='fraction',len=.7)
                        )
#     fig.update_coloraxes2(colorscale='inferno',colorbar = dict(
#                                                               thicknessmode = "fraction",thickness=.05, 
#                                                               lenmode='fraction',len=.7)
#                        )
    return fig







#image graph
@app.callback(
    Output('img-actmap-graph', 'figure'),
    [Input('dynamic-input-image-dropdown', 'value'),
     Input('node-actmap-graph','clickData'),
     Input('node-actmap-graph','figure')],
    [State('img-actmap-graph', 'figure'),
     State('node-actmap-dropdown', 'value')])
def update_inputimg_actmap(image_name,click_data,node_actmap_fig,image_fig,nodeid): 
    print('CALLED: update_inputimg_actmap')
    #if os.path.exists(params['input_image_directory']+image_name):
    ctx = dash.callback_context
    if not ctx.triggered:
        raise Exception('no figure updates yet')
    else:
        trigger = ctx.triggered[0]['prop_id']
    if trigger == 'dynamic-input-image-dropdown.value':
        return image2plot(get_image_path(image_name,params)[1],input_image_layout)
    elif receptive_fields is None:
        return image2plot(get_image_path(image_name,params)[1],input_image_layout)
    elif click_data is None:
        return image2plot(get_image_path(image_name,params)[1],input_image_layout)
    else:
        #nodeid = node_actmap_fig['data'][0]['name']
        layer_name = nodeid_2_perlayerid(nodeid,params)[2]
        if layer_name == 'img':
            raise Exception('no receptive fields for input image actmap')
        heatmap_dim_y = len(node_actmap_fig['data'][0]['z'])
        heatmap_dim_x = len(node_actmap_fig['data'][0]['z'][0]) 
        x_click = click_data['points'][0]['x']
        y_click = heatmap_dim_y - click_data['points'][0]['y']-1
        print('x_click')
        print(x_click)
        print('y_click')
        print(y_click)
        recep_field = receptive_field_for_unit(receptive_fields, layer_name, (x_click,y_click))
        recep_field_normed = [[recep_field[0][0]*10/input_image_size,recep_field[0][1]*10/input_image_size],
                              [recep_field[1][0]*10/input_image_size,recep_field[1][1]*10/input_image_size]]
        print('normalized')
        print(recep_field_normed)
        x_points = [recep_field_normed[0][0],recep_field_normed[0][0],recep_field_normed[0][1],recep_field_normed[0][1],recep_field_normed[0][0]]
        y_points = [10 - recep_field_normed[1][0],10 - recep_field_normed[1][1],10 - recep_field_normed[1][1],10 - recep_field_normed[1][0],10 - recep_field_normed[1][0]]
        print('x points')
        print(x_points)
        print('y points')
        print(y_points)
        image_fig['data'] = [{'mode': 'lines', 'x': x_points, 'y': y_points, 'type': 'scatter'}]
        return image_fig
        



    #else:
        #return image2plot(params['prepped_model_path']+'/visualizations/'+image_name,input_image_layout)
    
# #image dropdown
# @app.callback(
#     Output('input-image-dropdown', 'options'),
#     [Input('node-deepviz-image', 'figure'),
#      Input('edge-deepviz-image', 'figure')])
# def update_inputimg_dropdown(node_fig,edge_fig): 
#     print('CALLED: update_inputimg_dropdown options')
#     return [{'label': i, 'value': i} for i in params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/')]

#dynamic dropdown
@app.callback(
    dash.dependencies.Output("dynamic-input-image-dropdown", "options"),
    [dash.dependencies.Input("dynamic-input-image-dropdown", "search_value")],
)
def update_options(search_value):
    if not search_value:
        raise PreventUpdate
    return [{'label': i, 'value': i} for i in params['input_image_list']+os.listdir(params['prepped_model_path']+'/visualizations/images/') if search_value in i]





#kernel
@app.callback(
    Output('edge-kernel-graph', 'figure'),
    [Input('edge-actmaps-input','value')],
    [State('edge-kernel-graph','figure')])
def update_edge_kernelmap(edge_name,figure):
    print('CALLED: update_edge_kernelmap')
    kernel,inmap,outmap = edgename_2_edge_figures(edge_name, None, kernels, None,params)
    if kernel is not None:
        return go.Figure(data=go.Heatmap(z = kernel,
                                         #zmin=-.5,
                                         #zmax=.5,
                                         colorbar = dict(thicknessmode = "fraction",thickness=.1)),
                         layout=kernel_layout)
    else:
        return figure
                

#edge in        
@app.callback(
    Output('edge-inmap-graph', 'figure'),
    [Input('edge-actmaps-input','value'),
     Input('input-image-signal', 'children')],
    [State('edge-inmap-graph','figure'),
     State('ablations-signal', 'children')])
def update_edge_inmap(edge_name,image_name,figure,ablation_list):
    print('CALLED: update_edge_inmap')
    #fetch activations
    if image_name in all_activations['nodes'] and ablation_list == []:
        activations = all_activations
    else:
        activations = activations_store(image_name, ablation_list)
        
    kernel,inmap,outmap = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)
    if inmap is not None:
        return go.Figure(data=go.Heatmap(z = inmap,
                                         #zmin=-2,zmax=2,
                                         colorbar = dict(thicknessmode = "fraction",thickness=.1)
                                        ),
                         layout=edge_inmap_layout)
    else:
        print('edge inmap error')
        return figure

@app.callback(
    Output('edgein-sum', 'children'),
    [Input('edge-inmap-graph', 'figure')])
def update_node_sum(fig):
    mean = np.mean(fig['data'][0]['z'])
    return 'mean: %s'%str(mean)    

#edge out
@app.callback(
    Output('edge-outmap-graph', 'figure'),
    [Input('edge-actmaps-input','value'),
     Input('input-image-signal', 'children')],
    [State('edge-outmap-graph','figure'),
     State('ablations-signal', 'children')])
def update_edge_outmap(edge_name,image_name,figure, ablation_list):
    print('CALLED: update_edge_outmap')
    #fetch activations
    if image_name in all_activations['nodes'] and ablation_list == []:
        activations = all_activations
    else:
        activations = activations_store(image_name,ablation_list)
        
    kernel,inmap,outmap = edgename_2_edge_figures(edge_name, image_name, kernels, activations,params)
    if outmap is not None:
        return go.Figure(data=go.Heatmap(z = outmap,
                                         #zmin=-11,
                                         #zmax=14,
                                         colorbar = dict(thicknessmode = "fraction",thickness=.1)
                                        ),
                         layout=edge_outmap_layout)
    else:
        print('edge outmap error')
        return figure
        
@app.callback(
    Output('edgeout-sum', 'children'),
    [Input('edge-outmap-graph', 'figure')])
def update_node_sum(fig):
    mean = np.mean(fig['data'][0]['z'])
    return 'mean: %s'%str(mean)


 
#feature viz graph
@app.callback(
    Output('featviz-image', 'figure'),
    [Input('featviz-button','n_clicks')],
    [State('featviz-nodeedge-toggle', 'value'),
     State('featviz-channelneuron-toggle', 'value'),
     State('node-actmap-dropdown','value'),
     State('edge-actmaps-input','value')],
    prevent_initial_call=True)
def update_featviz_image(n_clicks,edge,neuron,nodeid,edgeid):       #EDIT: needs support for black and white images
    print('CALLED: update_featviz')
    if edge:
        image_name = regen_visualization(model_dis,edgeid,neuron,params)
    else:
        layer,within_layer_id,layer_name = nodeid_2_perlayerid(nodeid,params)    
        if layer == 'img': 
            return figure_init
        image_name = regen_visualization(model_dis,nodeid,neuron,params)
        
    image_path = params['prepped_model_path']+'/visualizations/images/'+image_name
    return image2plot(image_path,input_image_layout)


# #JSON INFO

# @app.callback(
#     Output('hover-data', 'children'),
#     [Input('node-actmap-graph', 'hoverData')])
# def display_hover_data(hoverData):
#     return json.dumps(hoverData, indent=2)




# @app.callback(
#     Output('click-data', 'children'),
#     [Input('network-graph', 'clickData')])
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)


# @app.callback(
#     Output('selected-data', 'children'),
#     [Input('network-graph', 'selectedData')])
# def display_selected_data(selectedData):
#     return json.dumps(selectedData, indent=2)


# @app.callback(
#     Output('figure-data', 'children'),
#     [Input('input-category', 'value'),
#      Input('network-graph', 'clickData'),
#      Input('edge-thresh-slider','value'),
#      Input('memory','data')])
# def display_trigger(target_category,clickData,edge_thresh,state):
#     ctx = dash.callback_context
#     if not ctx.triggered:
#         raise Exception('no figure updates yet')
#     else:
#         trigger = ctx.triggered[0]['prop_id'].split('.')[0]
#     ctx_msg = json.dumps({
#         'states': ctx.states,
#         'triggered': ctx.triggered,
#         'inputs': ctx.inputs,
#         'full_state':state
#     }, indent=2)
#     return ctx_msg


def extract_subgraph(name,model,thresholded_nodes_df,thresholded_egdes_df,params, save=True):
    #this is currently hacky only works on models with all nn.sequential or .features module
    model.to('cpu')
    l = 0
    subgraph_model = nn.Sequential()
    
    #hack
    for layer in model.children():
        if not isinstance(layer, nn.Conv2d):
            model = model.features
            break
        break
     
    with torch.no_grad():
        for layer in model.children():
            if isinstance(layer, nn.Conv2d):
                name = 'conv_{}'.format(l)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(l)
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(l)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(l)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            if not isinstance(layer, nn.Conv2d):
                subgraph_model.add_module(name, layer)
            else:
                print('layer: %s'%str(l))
                old_conv = layer
                out_channels = list(thresholded_nodes_df.loc[thresholded_nodes_df['layer'] == l]['node_num_by_layer'].sort_values().unique())
                num_out_channels = len(out_channels)
                if l == 0:
                    num_in_channels = old_conv.in_channels
                    in_channels = list(range(num_in_channels))
                else:
                    in_channels = list(thresholded_nodes_df.loc[thresholded_nodes_df['layer'] == l-1]['node_num_by_layer'].sort_values().unique())
                    num_in_channels = len(in_channels)
                new_conv = \
                        torch.nn.Conv2d(in_channels = num_in_channels, \
                        out_channels = num_out_channels ,
                        kernel_size = old_conv.kernel_size, \
                        stride = old_conv.stride,
                        padding = old_conv.padding,
                        dilation = old_conv.dilation,
                        groups = old_conv.groups,
                        bias = (old_conv.bias is not None))                
            
                #full_weights = []
                #GENERATE WEIGHT MATRIX
                weights = new_conv.weight
                weights.fill_(0.)
                print(out_channels)
                print(len(out_channels))
                for o_i,o in enumerate(out_channels):
                    for i_i,row in enumerate(thresholded_egdes_df.loc[(thresholded_egdes_df['layer'] == l) & (thresholded_egdes_df['out_channel'] == o)].sort_values('in_channel').itertuples()):
                        weights[o_i,in_channels.index(row.in_channel),:,:] = old_conv.weight[o,row.in_channel,:,:]
                #for node in thresholded_nodes_df.loc[thresholded_nodes_df['layer'] == l].sort_values('node_num_by_layer')['node_num_by_layer']:
                    #print(node)
                    #ws.append(weights[node,:,:,:].unsqueeze(0))
                
                #GENERATE BIAS 
                if new_conv.bias is not None:
                    for o_i,o in enumerate(out_channels):
                        new_conv.bias[o_i] = old_conv.bias[o]
                
                
                subgraph_model.add_module(name, new_conv)
                #next layer
                l += 1

            if l not in thresholded_nodes_df['layer'].unique():
                break

        
            
    return subgraph_model


model_dis.to('cuda:1')

start = time.time()
for node in range(262,1153):
    n_df,e_df=ranksdf_store('small_SPAN', str(node), [],model_dis=model_dis)
    n_df = minmax_normalize_ranks_df(n_df,params)
    threshed_n_df,threshed_e_df = hierarchical_accum_threshold(.9,.9,rank_type,e_df,n_df)
    sub_model = extract_subgraph('',model,threshed_n_df,threshed_e_df,params)
    sub_model.to('cuda:0')
    name = None
    for name, module in sub_model.named_modules():
        pass
    param_f = lambda: param.image(224,batch=1)
    obj =  objectives.neuron(name, 0, batch=0) 
    _ = render.render_vis(sub_model, obj, param_f, verbose=True, show_inline=False,show_image=False, save_image=True,image_name = 'prepped_models/alexnet_sparse/subgraphs/visualizations/%s_.9.9_target.jpg'%str(node))
    torch.save(sub_model,'prepped_models/alexnet_sparse/subgraphs/models/%s_.9.9.pt'%str(node))
print(time.time() - start)




