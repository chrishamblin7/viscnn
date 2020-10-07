#This launches the visualizer tool on a given model. 
#run with python launch_visualizer.py [prepped model name]
#script accepts command line arguments, see what they are with python launch_visualizer.py --help
#other parameters are settable in this script, go to line 86

#fetch command line argument (prepped model)
import sys
import os
from subprocess import call
from copy import deepcopy
import pickle
import torch
import plotly.offline as py
import plotly.graph_objs as go

sys.path.insert(0, os.path.abspath('./setup_scripts/'))
sys.path.insert(0, os.path.abspath('./prep_model_scripts/'))
sys.path.insert(0, os.path.abspath('./visualizer_scripts/'))
from visualizer_helper_functions import *
from contrast_helper_functions import *
from download_from_gdrive import online_model_names
from dissected_Conv2d import *
from copy import deepcopy
import argparse


#assemble possible moodels to visualize
possible_models = os.listdir('prepped_models')
for m in possible_models:
	if not os.path.isdir('./prepped_models/'+m):
		possible_models.remove(m)

for m in online_model_names:
	if m not in possible_models:
		possible_models.append(m)

#command Line argument parsing
def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument("model", type = str, choices=possible_models,
						help="Name of model (the folder names within 'prepped_models'). Can also be model available online.")
	parser.add_argument('--cuda', action='store_true', help='Use GPU acceleration')
	parser.add_argument("--port", type = int, default = 8050,
						help="port visualizer tool will display to (default 8050). after running script access tool in browser at localhost:[port]")   
	parser.add_argument("--input-image-directory", type = str, default = None,
						help = "Path to directory of input images you want to visualizer through the model. (Defaults to path specified during prep_model.py)")
	parser.add_argument("--max-node-inputs", type = int, default = 20,
						help="the tool shows input activation maps to the highlighted node sorted by importance, how many should it display?")
	parser.add_argument("--dont-download-images", action='store_true', 
						help='If downloading the model from online, dont download the input_image folder associated with that model. \
							  You should use this in conjuction with the --input-image-directory directory flag to set your own.')
	args = parser.parse_args()
	return args

args = get_args()

print('called with arguments:')
print(args)
print('\n')

prepped_model_folder = args.model
full_prepped_model_folder = os.path.abspath('prepped_models/%s'%prepped_model_folder)

print('possible models to visualizer are:')
print(possible_models)
print('\nYou\'ve chosen to visualize %s'%prepped_model_folder+'\n')


#Download model if not local
if not os.path.isdir('./prepped_models/'+prepped_model_folder):
	print('\n %s not present locally. Downloading from online . . .'%prepped_model_folder)
	os.chdir('./setup_scripts')
	if args.dont_download_images:
		call('python download_from_gdrive.py %s --dont-download-images'%prepped_model_folder,shell=True)
	else:
		call('python download_from_gdrive.py %s'%prepped_model_folder,shell=True)
	os.chdir('../')


sys.path.insert(0,'prepped_models/%s'%prepped_model_folder)
import prep_model_params_used as prep_model_params

params = {}
params['cuda'] = args.cuda    #use gpu acceleration when running model forward
if args.input_image_directory is None:
	params['input_image_directory'] = prep_model_params.input_img_path+'/'   #path to directory of imput images you want fed through the network
else:
	params['input_image_directory'] = args.input_image_directory
params['max_node_inputs'] = args.max_node_inputs

print(params['input_image_directory'])


#####              PARAMETERS                  #####
#####          (you can set these)             #####

#Non-GUI parameters

#backend
params['preprocess'] = prep_model_params.preprocess     #torchvision transfrom to pass input images through
params['label_file_path'] = prep_model_params.label_file_path
params['criterion'] = prep_model_params.criterion


#aesthetic     
#params['node_size'] = 12
#params['edge_size'] = 1
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
projection = 'MDS'           #how nodes within a layer are projected into the 2d plane (MDS or Grid)
edge_threshold = [.1,1]     #what range do edge ranks need to be in to be visualized

#######           END OF PARAMETERS          ########
#######            DONT EDIT BELOW           ########



#other initializations
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


#MODEL LOADING

model_dis = dissect_model(deepcopy(prep_model_params.model),store_ranks=True,clear_ranks=True,cuda=params['cuda']) #version of model with accessible preadd activations in Conv2d modules 
if params['cuda']:
	model_dis.cuda()
model_dis.eval()

print('loaded model:')
print(prep_model_params.model)
print('\n')
		
del prep_model_params.model

#load misc graph data
print('loading misc graph data'+'\n')
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
print('\n')


#load nodes df
print('loading nodes rank data')
categories_nodes_df = pd.read_csv('prepped_models/%s/ranks/categories_nodes_ranks.csv'%prepped_model_folder)
target_nodes_df = categories_nodes_df.loc[categories_nodes_df['category']==target_category]

weight_nodes_df = pd.read_csv('prepped_models/%s/ranks/weight_nodes_ranks.csv'%prepped_model_folder)

node_colors,node_weights = gen_node_colors(target_nodes_df,rank_type,params) 

#load node positions
print('loading node position data')
all_node_positions = pickle.load(open('./prepped_models/%s/node_positions.pkl'%prepped_model_folder,'rb'))

if projection == 'MDS':
	node_positions = all_node_positions[projection][rank_type+'_norm']
else:
	node_positions = all_node_positions[projection]



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

	
weight_edges_df = pd.read_csv('prepped_models/%s/ranks/weight_edges_ranks.csv'%prepped_model_folder)
	
edges_thresholded_df = get_thresholded_edges(edge_threshold,rank_type,target_edges_df,target_category)
	
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

#Format Node Feature Maps
print('loading activation maps')

all_activations = {'nodes':{},'edges_in':{},'edges_out':{}}
if os.path.exists('prepped_models/%s/input_img_activations.pt'%prepped_model_folder):
	all_activations = torch.load('prepped_models/%s/input_img_activations.pt'%prepped_model_folder)


#hidden state, stores python values within the html itself
state = {'projection':projection,'rank_type':rank_type,'edge_positions':edge_positions,'edge_colors': edge_colors, 'edge_widths':edge_widths,'edge_names':edge_names,
		 'edge_threshold':edge_threshold,'edge_weights':edge_weights,'max_edge_width_indices':max_edge_width_indices,
		 'node_positions':node_positions,'node_colors':node_colors,'node_weights':node_weights,'target_category':target_category,
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
		 height=400,
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
combined_traces = gen_networkgraph_traces(state,params,categories_nodes_df)
network_graph_fig=go.Figure(data=combined_traces, layout=network_graph_layout)

#Dash App Setup
print('setting up dash app')

import dash
import dash_core_components as dcc
import dash_html_components as html
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
				 html.Label('Weighting Category'),
				 #dcc.Dropdown(
				 #  id='weight-category',
				 #  options=[{'label': i, 'value': i} for i in params['categories']],
				 #   value=target_category
				 #   ),
				dcc.Input(id='weight-category',value=state['target_category']),
				 html.Br(),
				 html.Label('Weighting Criterion'),
				 dcc.Dropdown(
					id='weight-criterion',
					options=[
						{'label': 'Activations*Grads', 'value': 'actxgrad'},
						{'label': 'Activations', 'value': 'act'},
						{'label': 'Gradients', 'value': 'grad'},
						{'label': 'Weights', 'value': 'weight'}
					],
					value='actxgrad'
					),
				 html.Br(),   
				 html.Label('Layer Projection'),
				 dcc.Dropdown(
					id = 'layer-projection',
					options=[
						{'label': 'MDS', 'value': 'MDS'},
						{'label': 'Grid', 'value': 'Grid'},
						#{'label': 'SOM', 'value': 'SOM'}
					],
					value='MDS'
					),

				html.Br(),
				html.Label('Edge Thresholds'),
				dcc.RangeSlider(
					id='edge-thresh-slider',
					min=0,
					max=np.ceil(params['max_edge_weight']*10)/10,
					step=0.001,
					marks={i/10: str(i/10) for i in range(0,int(np.ceil(params['max_edge_weight']*10))+1,int(round(np.ceil(params['max_edge_weight']*10)/5)))},
					value=[.1,np.ceil(params['max_edge_weight']*10)/10],
				),
				# dcc.Checklist(
				# 				id='normalize-box'
				# 				options=[{'label': 'Per-Layer Normalize', 'value': 'normalize'}],
				# 				value=['normalize']
							# )
  
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
			html.Label('Input Image'),
			dcc.Dropdown(
				id='input-image-dropdown',
				options=[{'label': i, 'value': i} for i in params['input_image_list']],
				value=input_image_name
			),
			html.Br(),
			dcc.Graph(
				id='img-actmap-graph',
				style={
			   'width': '14vw',
			   'height':'14vw'
				},
				figure=image2plot(params['input_image_directory']+input_image_name,input_image_layout),
				config={
						'displayModeBar': False
						}
			)
			], className = "two columns"),

			html.Div([
			html.Label('Node'),
			dcc.Dropdown(
				id='node-actmap-dropdown',
				options=[{'label': str(j), 'value': str(j)} for j in params['imgnode_names']]+[{'label': str(i), 'value': str(i)} for i in range(params['num_nodes'])],
				value='0'
			),
			html.Br(),
			dcc.Graph(
				id='node-actmap-graph',
				style={
			   'width': '19vw',
			   'height':'14vw'
				},
				figure=figure_init,
				config={
						'displayModeBar': False
						}
			)
			], className = "three columns"),
			
			html.Div([
			html.Label('Node Inputs'),
			html.Br(),
			html.Div(dcc.Graph(
				id='node-inputs-graph',
				figure=figure_init,
				config={
						'displayModeBar': False
						}
			),style={'overflowY': 'scroll', 'height': 500})
			], className = "two columns"),

			html.Div([
			html.Label('Edge'),    
			dcc.Input(
				id='edge-actmaps-input',value=state['edge_names'][0][0], type='text'),
			#html.Button(id='edge-kernel-button',n_clicks=0, children='Submit'),
			html.Br(),
			html.Br(),
			dcc.Graph(
				id='edge-kernel-graph',
				style={
			   'width': '14vw',
			   'height':'10vw'
				},
				figure=go.Figure(data=go.Heatmap(
									z = edgename_2_edge_figures(state['edge_names'][0][0], input_image_name, kernels, None,categories_nodes_df,params)[0]),
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
				style={
			   'width': '19vw',
			   'height':'14vw'
				},
				figure=figure_init,
				config={
						'displayModeBar': False
						}
			),
			html.Br(),
			html.Br(),
			dcc.Graph(
				id='edge-outmap-graph',
				style={
			   'width': '19vw',
			   'height':'14vw'
				},
				figure=figure_init,
				config={
						'displayModeBar': False
						}
			)
			], className = "three columns")


		 ], className= 'row'
		 ),
				
				
# 		html.Div([
# 			html.Div([
# 				dcc.Markdown("""
# 					**Hover Data**

# 					Mouse over values in the graph.
# 				"""),
# 				html.Pre(id='hover-data', style=styles['pre'])
# 			], className='two columns'),

# 			html.Div([
# 				dcc.Markdown("""
# 					**Click Data**

# 					Click on points in the graph.
# 				"""),
# 				html.Pre(id='click-data', style=styles['pre']),
# 			], className='two columns'),

# 			html.Div([
# 				dcc.Markdown("""
# 					**Selection Data**

# 					Choose the lasso or rectangle tool in the graph's menu
# 					bar and then select points in the graph.

# 					Note that if `layout.clickmode = 'event+select'`, selection data also 
# 					accumulates (or un-accumulates) selected data if you hold down the shift
# 					button while clicking.
# 				"""),
# 				html.Pre(id='selected-data', style=styles['pre']),
# 			], className='two columns'),

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
				
# 			html.Div([
# 				dcc.Markdown("""
# 					**Figure Data**

# 					Figure json info.
# 				"""),
# 				html.Pre(id='figure-data', style=styles['pre']),
# 			], className='four columns')

# 		], className= 'row'
# 		),

	#hidden divs for storing intermediate values     
	# The memory store reverts to the default on every page refresh
	dcc.Store(id='memory'),
	# The local store will take the initial data
	# only the first time the page is loaded
	# and keep it until it is cleared.
	dcc.Store(id='local', storage_type='local'),
	# Same as the local store but will lose the data
	# when the browser/tab closes.
	dcc.Store(id='session', storage_type='session',data=state),
	

	# hidden signal value
	html.Div(id='input-image-signal', style={'display': 'none'}),
	html.Div(id='target-category-signal', style={'display': 'none'})

])



# perform expensive computations in this "global store"
# these computations are cached in a globally available
# redis memory store which is available across processes
# and for all time.
@cache.memoize()
def activations_store(image_name):

	print('Updating cached activations with {}'.format(image_name))
	activations = get_model_activations_from_image(params['input_image_directory']+image_name, model_dis, params)
	
	return activations

@app.callback(Output('input-image-signal', 'children'), 
			  [Input('input-image-dropdown', 'value')])
def update_activations_store(image_name):
	# compute value and send a signal when done
	activations_store(image_name)
	return image_name


@cache.memoize()
def ranksdf_store(target_category):
	print('Updating cached rank dfs with {}'.format(target_category))
	target_type = image_category_or_contrast(target_category,params['input_image_list'],params['categories'])
	target_nodes_df = None
	target_edges_df = None
	if target_type == 'category':
		#edges
		if categories_edges_df is not None:
			if len(categories_edges_df.loc[categories_edges_df['category']==target_category]) > 0:
				target_edges_df = categories_edges_df.loc[categories_edges_df['category']==target_category]
		if target_edges_df is None:
			target_edges_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_edges','%s_edges_rank.pt'%target_category))   
		#node
		if categories_nodes_df is not None:
			if len(categories_nodes_df.loc[categories_nodes_df['category']==target_category]) > 0:
				target_nodes_df = categories_nodes_df.loc[categories_nodes_df['category']==target_category]
		if target_nodes_df is None:
			target_nodes_df = rank_file_2_df(os.path.join(params['ranks_data_path'],'categories_nodes','%s_nodes_rank.pt'%target_category))
	elif target_type == 'input_image':
		target_nodes_df,target_edges_df = rank_dict_2_df(get_model_ranks_from_image(params['input_image_directory']+'/'+target_category, model_dis, params))
	else:  #contrast
		target_nodes_df,target_edges_df = contrast_str_2_dfs(target_category,model_dis,params)
		
	return target_nodes_df,target_edges_df

@app.callback(Output('target-category-signal', 'children'), 
			  [Input('weight-category', 'value')])
def update_ranksdf_store(target_category):
	# compute value and send a signal when done
	ranksdf_store(target_category)
	return target_category



####Call Back Functions

#Hidden State
@app.callback(
	Output('session', 'data'),
	[Input('target-category-signal', 'children'),
	 Input('node-actmap-dropdown', 'value'),
	 Input('edge-actmaps-input', 'value'),
	 Input('edge-thresh-slider','value'),
	 Input('layer-projection','value'),
	 Input('weight-criterion','value')],
	[State('session', 'data')])
def update_store(target_category,node_value,edge_value,edge_threshold,projection,rank_type,state):
	print('CALLED: update_store\n')
	ctx = dash.callback_context
	if not ctx.triggered:
		raise Exception('no figure updates yet')
	else:
		trigger = ctx.triggered[0]['prop_id']
	state['last_trigger'] = trigger  #store the last trigger of state change in state
	print('TRIGGER %s'%trigger)
	
	#fetch select edges DF
	if trigger in ['target-category-signal.children','edge-thresh-slider.value','layer-projection.value','weight-criterion.value']:
		if rank_type == 'weight':
			target_edges_df = weight_edges_df
			target_nodes_df = weight_nodes_df
			edges_thresholded_df = get_thresholded_edges(edge_threshold,rank_type,target_edges_df,target_category)
		else:   
			target_nodes_df,target_edges_df = ranksdf_store(target_category)
			edges_thresholded_df = get_thresholded_edges(edge_threshold,rank_type,target_edges_df,target_category)
	
	if trigger == 'target-category-signal.children':
		print('changing target category to %s'%target_category)
		print(target_nodes_df)
		state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params)
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
		if state['edge_select_history'][-1] != edge_value and check_edge_validity(edge_value.strip(),categories_nodes_df,params)[0]:
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
		
	elif trigger == 'layer-projection.value':
		print('changing layer projection to %s\n'%projection)
		state['projection']=projection
		if projection == 'MDS':
			state['node_positions'] = all_node_positions[projection][rank_type+'_norm']
		else:
			state['node_positions'] = all_node_positions[projection]
		state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)
		
	elif trigger == 'weight-criterion.value':
		print('changing weighting criterion to %s\n'%rank_type)
		state['rank_type']=rank_type
		state['node_colors'], state['node_weights'] = gen_node_colors(target_nodes_df,rank_type,params)
		#state['node_positions']=format_node_positions(projection=projection,rank_type=rank_type)
		state['edge_positions'], state['edge_colors'], state['edge_widths'],state['edge_weights'], state['edge_names'], state['max_edge_width_indices'] = gen_edge_graphdata(edges_thresholded_df, state['node_positions'], rank_type, target_category,params)
		
	else:
		raise Exception('unknown trigger: %s'%trigger)    
	return state


#Network Graph Figure
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
			select_layer,select_position = nodeid_2_perlayerid(state['node_select_history'][-1],categories_nodes_df,params)
			fig['data'][select_layer+1]['marker']['color'][select_position] = 'rgba(0,0,0,1)'
		else:   #imgnode
			fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-1])] = 'rgba(0,0,0,1)'
		#update previous node color to its usual color
		if len(state['node_select_history']) > 1: #there is a previous node to unselect
			if str(state['node_select_history'][-2]).isnumeric():  #if normal node
				prev_select_layer,prev_select_position = nodeid_2_perlayerid(state['node_select_history'][-2],categories_nodes_df,params)
				fig['data'][prev_select_layer+1]['marker']['color'][prev_select_position] = state['node_colors'][prev_select_layer][prev_select_position]
			else:   #imgnode
				fig['data'][0]['marker']['color'][fig['data'][0]['text'].index(state['node_select_history'][-2])] = state['imgnode_colors'][fig['data'][0]['text'].index(state['node_select_history'][-2])]
		#fig['layout']['uirevision']=True   
		return fig    
	else:   #regenerate full traces
		combined_traces = gen_networkgraph_traces(state,params,categories_nodes_df)
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
	 State('session', 'data')])
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
	 Input('input-image-signal', 'children')])
def update_node_actmap(nodeid,image_name):       #EDIT: needs support for black and white images
	print('CALLED: update_node_actmap')
	layer, within_id = nodeid_2_perlayerid(nodeid,categories_nodes_df,params)
	#fetch activations
	if image_name in all_activations['nodes']:
		activations = all_activations
	else:
		activations = activations = activations_store(image_name)
		
	if layer == 'img': #code for returning color channel as activation map
		#np_chan_im = get_channelwise_image(image_name,state['imgnode_names'].index(nodeid),params['input_image_directory']=params['input_image_directory'])
		np_chan_im = activations['edges_in'][image_name][0][within_id]
		return go.Figure(data=go.Heatmap( z = np.flip(np_chan_im,0)),
						layout=node_actmap_layout) 
	
	return go.Figure(data=go.Heatmap( z = np.flip(activations['nodes'][image_name][layer][within_id],0),
									  zmin=-1,
									  zmax=1,
									  colorbar = dict(thicknessmode = "fraction",thickness=.1)
									),
					 layout=node_actmap_layout) 


#Node inputs actmap graph
@app.callback(
	Output('node-inputs-graph', 'figure'),
	[Input('node-actmap-dropdown', 'value'),
	 Input('input-image-signal', 'children'),
	 Input('target-category-signal', 'children'),
	 Input('weight-criterion','value')])
def update_node_inputs(nodeid,image_name,target_category,rank_type,max_num = params['max_node_inputs']):       
	print('CALLED: update_node_inputs')
	node_layer,node_within_layer_id = nodeid_2_perlayerid(nodeid,categories_nodes_df,params)
	#fetch activations
	if image_name in all_activations['nodes']:
		activations = all_activations
	else:
		activations = activations_store(image_name)
	#fetch edges df
	if rank_type == 'weight':
		target_edges_df = weight_edges_df
	else:
		target_edges_df = ranksdf_store(target_category)[1]
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
	all_node_edges_df = all_node_edges_df.sort_values(by=[rank_type+'_norm_rank'],ascending=False)
	top_node_edges_df = all_node_edges_df.head(max_num)
	fig = make_subplots(rows=len(top_node_edges_df)+1, cols=1)
	i=1
	for row in top_node_edges_df.itertuples():
		if node_layer == 0:
			edge_name = str(params['imgnode_names'][row.in_channel])+'-'+str(nodeid)
		else:
			edge_name = str(params['layer_nodes'][node_layer-1][row.in_channel])+'-'+str(nodeid)

		fig.add_trace(
			   go.Heatmap(z = edgename_2_edge_figures(edge_name, image_name, kernels, activations,categories_nodes_df,params)[2],
						  zmin = -1,
						  zmax = 1,
						  name = edge_name,
						  showscale = False,
						  colorbar = dict(lenmode='fraction',len=1/len(top_node_edges_df), 
										  y=(i)/len(top_node_edges_df)-.01,
										  thicknessmode = "fraction",thickness=.1,
										  ypad=1
										 )),
			   row=i, col=1)
		i+=1
	fig.update_layout(height=200*len(top_node_edges_df), 
					  width=170,
					  #yaxis=dict(scaleanchor="x", scaleratio=1/len(top_node_edges_df)),
					  #title_text="Inputs to Node",
					  margin=dict(
									l=0,
									r=0,
									b=0,
									t=0,
									pad=0)
					 )
	return fig

#image graph
@app.callback(
	Output('img-actmap-graph', 'figure'),
	[Input('input-image-dropdown', 'value')])
def update_inputimg_actmap(image_name): 
	print('CALLED: update_inputimg_actmap')
	return image2plot(params['input_image_directory']+image_name,input_image_layout)


#kernel
@app.callback(
	Output('edge-kernel-graph', 'figure'),
	[Input('edge-actmaps-input','value')],
	[State('edge-kernel-graph','figure')])
def update_edge_kernelmap(edge_name,figure):
	print('CALLED: update_edge_kernelmap')
	kernel,inmap,outmap = edgename_2_edge_figures(edge_name, None, kernels, None,categories_nodes_df,params)
	if kernel is not None:
		return go.Figure(data=go.Heatmap(z = kernel),
						#,zmin=-.5,zmax=.5),
						 layout=kernel_layout)
	else:
		return figure
				

#edge in        
@app.callback(
	Output('edge-inmap-graph', 'figure'),
	[Input('edge-actmaps-input','value'),
	 Input('input-image-signal', 'children')],
	[State('edge-inmap-graph','figure')])
def update_edge_inmap(edge_name,image_name,figure):
	print('CALLED: update_edge_inmap')
	#fetch activations
	if image_name in all_activations['nodes']:
		activations = all_activations
	else:
		activations = activations = activations_store(image_name)
		
	kernel,inmap,outmap = edgename_2_edge_figures(edge_name, image_name, kernels, activations,categories_nodes_df,params)
	if inmap is not None:
		return go.Figure(data=go.Heatmap(z = inmap,zmin=-1,zmax=1,
										 colorbar = dict(thicknessmode = "fraction",thickness=.1)
										),
						 layout=edge_inmap_layout)
	else:
		print('edge inmap error')
		return figure

#edge out
@app.callback(
	Output('edge-outmap-graph', 'figure'),
	[Input('edge-actmaps-input','value'),
	 Input('input-image-signal', 'children')],
	[State('edge-outmap-graph','figure')])
def update_edge_outmap(edge_name,image_name,figure):
	print('CALLED: update_edge_outmap')
	#fetch activations
	if image_name in all_activations['nodes']:
		activations = all_activations
	else:
		activations = activations = activations_store(image_name)
		
	kernel,inmap,outmap = edgename_2_edge_figures(edge_name, image_name, kernels, activations,categories_nodes_df,params)
	if outmap is not None:
		return go.Figure(data=go.Heatmap(z = outmap,zmin=-1,zmax=1,
										 colorbar = dict(thicknessmode = "fraction",thickness=.1)
										),
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
def display_trigger(target_category,clickData,edge_thresh,state):
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

#launch app,   you can now access the tool at localhost:8050  (or whatever port you set)
app.run_server(port=args.port,host='0.0.0.0')
