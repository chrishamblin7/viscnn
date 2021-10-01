import torch
from viscnn.utils import *
from viscnn.visualizers.layouts import *
from viscnn.visualizers.cnn_gui import *
from viscnn.utils import *
import os

import plotly.offline as py
import plotly.graph_objs as go
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_daq as daq
from dash.exceptions import PreventUpdate
#import utils.dash_reusable_components as drc
import flask
import json
from dash.dependencies import Input, Output, State
from plotly.subplots import make_subplots
from flask_caching import Cache
import base64

def circuit_edge_width_scaling(x):
	#return max(.4,(x*10)**1.7)
	return max(.5,np.exp(1.5*x))
	

def circuit_curve_2_id(curve_num,point_num,subgraph_dict,params):
	node_df = deepcopy(subgraph_dict['node_df'])
	node_df = node_df.sort_values(by=['node_num'])
	if curve_num == 0:
		imgnode_dict = {0:'r',1:'g',2:'b'}
		return imgnode_dict[point_num]
	elif curve_num <= len(node_df['layer'].unique()):
		layer = curve_num-1
		return str(node_df.loc[node_df['layer']==layer].iloc[point_num]['node_num'])
	else:
		edge_row_idx = curve_num - 1 - len(node_df['layer'].unique())
		row = subgraph_dict['edge_df'].iloc[edge_row_idx]
		if row['layer'] != 0:
			in_node = params['layer_nodes'][row['layer']-1][1][row['in_channel']]
		else:
			in_node = params['imgnode_names'][row['in_channel']]
		out_node = params['layer_nodes'][row['layer']][1][row['out_channel']]
		return str(in_node)+'-'+str(out_node)
	

def gen_kernel_img(edge_name,kernels,params):
	kernel,inmap,outmap = edgename_2_edge_figures(edge_name, None, kernels, None,params)
	if kernel is not None:
		fig =  go.Figure(data=go.Heatmap(z = kernel,
										 colorscale='RdBu',
										 reversescale=True,
										 zmid=0,
										 #zmin=-.5,
										 #zmax=.5,
										showscale=False),
						 layout=kernel_layout)
		fig.update(layout_showlegend=False)
		img_file_path = params['prepped_model_path']+'/visualizations/images/kernels/%s.jpg'%str(edge_name)
		if not os.path.exists(img_file_path):
			fig.write_image(img_file_path,format='jpg')

			
def gen_vizualizations_for_subgraph(path_2_subgraph_dict, params): #takes full path to subgraph dict
	subgraph_dict = torch.load(path_2_subgraph_dict)
	model = subgraph_dict['model']
	_ = model.to(params['device']).eval()
	subgraph_name = '.'.join(path_2_subgraph_dict.split('/')[-1].split('.')[:-1])
	viz_folder = '/'.join(path_2_subgraph_dict.split('/')[:-2])+'/visualizations/'+subgraph_name
	if not os.path.exists(viz_folder):
		os.mkdir(viz_folder)
		os.mkdir(viz_folder+'/channel')
		os.mkdir(viz_folder+'/neuron')
		with open(viz_folder+'/images.csv', 'a') as images_csv:
			images_csv.write('image_name,targetid,objective,parametrizer,optimizer,transforms,neuron\n')
		images_csv.close()
	layer = -1
	within_id = 0
	node_df = deepcopy(subgraph_dict['node_df'])
	node_df = node_df.sort_values(by=['node_num'])
	for row in node_df.itertuples():
		layer_name = 'conv_'+str(row.layer)
		if row.layer == layer:
			within_id+=1
		else:
			layer += 1
			within_id = 0
		fetch_deepviz_img_for_subgraph(model,layer_name,within_id,row.node_num,viz_folder,params)
		
	
def subgraph_2_2d_circuit(subgraph_dict_path, params=None, rank_type = 'actxgrad_rank', num_hoverpoints=4,min_w=4,max_w=10):

	subgraph_dict_path = os.path.abspath(subgraph_dict_path)
	subgraph_name = subgraph_dict_path.split('/')[-1]

	subgraph_dict = torch.load(subgraph_dict_path)

	split_path = subgraph_dict_path.split('/')
	prepped_model_path = '/'.join(split_path[:split_path.index('prepped_models')+2])

		#set up params
	if params is None:
		params = load_cnn_gui_params(prepped_model_path)
		
	kernel_colors = torch.load(prepped_model_path+'/kernels.pt')['kernel_colors']

	layer_offset = 5
	vert_offset = 1
	rank = 'actxgrad_rank'
	pos_dict_nodes = {}
	imgnode_positions = {'X':[-layer_offset,-layer_offset,-layer_offset],'Y':[2,0,-2]}
	#add img nodes
	imgnode_trace=go.Scatter(x=imgnode_positions['X'],
		   y=imgnode_positions['Y'],
		   mode='markers',
		   name='image channels',
		   marker=dict(symbol='square',
						 size=8,
						 opacity=.99,
						 color=params['imgnode_colors'],
						 #colorscale='Viridis',
						 line=dict(color='rgb(50,50,50)', width=.5)
						 ),
		   text=params['imgnode_names'],
		   hoverinfo='text'
		   )

	imgnode_traces = [imgnode_trace]
	
	node_df = deepcopy(subgraph_dict['node_df'])
	node_df = node_df.sort_values(by=['node_num'])
	node_traces = []
	
	for layer in list(node_df['layer'].unique()):
		#add nodes

		within_layer_ids = list(node_df.loc[node_df['layer']==layer]['node_num_by_layer'])
		scores = list(node_df.loc[node_df['layer']==layer][rank])
		ids = list(node_df.loc[node_df['layer']==layer]['node_num'])
		#print(np.dstack((ids,within_layer_ids,scores)).shape)
		#print(np.dstack((ids,within_layer_ids,scores)))
		# hovertext = ['<b>%{id}</b>' +
		# 			'<br><i>layerwise ID</i>: %{within_layer_id}'+
		# 			'<br><i>Score</i>: %{score}<br>'
		# 			 for id, within_layer_id, score in
		# 			 zip(ids, within_layer_ids, scores)]
		#print(hovertext) 
		x_positions = []
		y_positions = []
		y_adjustment = (len(within_layer_ids)-1)/2*vert_offset
		for i in range(len(within_layer_ids)):
			x_positions.append(layer*layer_offset)
			y_positions.append(i*vert_offset-y_adjustment)
		node_trace=go.Scatter(x=x_positions,
				   y=y_positions,
				   mode='markers',
				   name=list(node_df.loc[node_df['layer']==layer]['layer_name'].unique())[0],
				   marker=dict(symbol='circle',
								 size=6,
								 color='rgba(50,50,50,0)',
								 opacity=0,
								 #colorscale='Viridis',
								 line=dict(color='rgba(50,50,50,0)', width=.5)
								 ),
				   text=ids,
				   #customdata = np.dstack((ids,within_layer_ids,scores)),
				   customdata = np.stack((ids,within_layer_ids,scores),axis=-1),
				   hovertemplate =	'<b>%{customdata[0]}</b>' +
							'<br><i>layerwise ID</i>: %{customdata[1]}'+
							'<br><i>Score</i>: %{customdata[2]:.3f}<br>'
				   #hoverinfo='text'
				   )

		node_traces.append(node_trace)
		pos_dict_nodes[layer] = {'name':ids,'X':x_positions,'Y':y_positions}
		
	edge_df =  minmax_normalize_ranks_df(subgraph_dict['edge_df'],params)
	edge_traces = []
	pos_dict_edges = {}
	for layer in list(edge_df['layer'].unique()):  
		pos_dict_edges[layer] = {'name':[],'X':[],'Y':[]}
		legendgroup = layernum2name(layer ,title = 'edges')
		
		#edge_widths = []
		#names = []
		#colors = []
		for row in edge_df.loc[edge_df['layer']==layer].itertuples():
			showlegend = False
			if getattr(row, rank_type) > .999:
				showlegend = True
			#positions
			edge_positions = {'X':[],'Y':[]}
			for dim in ['X','Y']:
				end_pos = pos_dict_nodes[layer][dim][pos_dict_nodes[layer]['name'].index(params['layer_nodes'][layer][1][row.out_channel])]
				if layer != 0:
					start_pos = pos_dict_nodes[layer-1][dim][pos_dict_nodes[layer-1]['name'].index(params['layer_nodes'][layer-1][1][row.in_channel])]
				else:
					start_pos = imgnode_positions[dim][row.in_channel]

				step = (end_pos-start_pos)/(num_hoverpoints+1)
				points = [start_pos]
				for i in range(1,num_hoverpoints+1):
					points.append(start_pos+i*step)
				points.append(end_pos)
				edge_positions[dim]=points
			#widths
			edge_width = circuit_edge_width_scaling(getattr(row, rank_type))
			#edge_widths.append(edge_width_scaling(getattr(row, rank_type)))
			#names
			out_node = params['layer_nodes'][row.layer][1][row.out_channel]
			if row.layer != 0:
				in_node = params['layer_nodes'][row.layer-1][1][row.in_channel]
			else:
				in_node = params['imgnode_names'][row.in_channel]
			#names.append(str(in_node)+'-'+str(out_node))
			edge_name = str(in_node)+'-'+str(out_node)
			#color
			if kernel_colors is None:
				alpha = edge_color_scaling(getattr(row, rank_type))
				#colors.append(params['layer_colors'][layer%len(params['layer_colors'])]+str(round(alpha,3))+')')
				edge_color = params['layer_colors'][layer%len(params['layer_colors'])]+str(round(alpha,3))+')'
			else:
				#colors.append(color_vec_2_str(kernel_colors[int(layer)][int(row.out_channel)][int(row.in_channel)]))
				edge_color = color_vec_2_str(kernel_colors[int(layer)][int(row.out_channel)][int(row.in_channel)])
			edge_trace=go.Scatter(x=edge_positions['X'],
							y=edge_positions['Y'],
							legendgroup=legendgroup,
							showlegend=showlegend,
							name=params['layer_nodes'][layer][0],
							mode='lines',
							#line=dict(color=edge_colors_dict[layer], width=1.5),
							line=dict(color=edge_color, width=edge_width),
							text = edge_name,
							hoverinfo='text'
							)
			edge_traces.append(edge_trace)
			pos_dict_edges[layer]['name'].append(edge_name)
			pos_dict_edges[layer]['X'].append(edge_positions['X'])
			pos_dict_edges[layer]['Y'].append(edge_positions['Y'])
	#trace just for storing data
	misc_trace=go.Scatter(x=[-layer_offset-.5],
				y=[-2.5],
				showlegend=False,
				name='misc',
				mode='markers',
				marker=dict(symbol='circle',
				 size=6,
				 color='rgba(255,255,255,0)',
				 opacity=0,
				 #colorscale='Viridis',
				 line=dict(color='rgba(255,255,255,0)', width=.5)
				 ),

				text = 'full', #text info storing full or partial graph
				hoverinfo='skip'
				)
	combined_traces = imgnode_traces+node_traces+edge_traces+[misc_trace]
	return combined_traces, pos_dict_nodes, pos_dict_edges


def launch_circuit_gui(subgraph_dict_path,port=8050,params=None,viz_folder=None):
	subgraph_dict_path = os.path.abspath(subgraph_dict_path)
	subgraph_name = subgraph_dict_path.split('/')[-1]

	subgraph_dict = torch.load(subgraph_dict_path)

	split_path = subgraph_dict_path.split('/')
	prepped_model_path = '/'.join(split_path[:split_path.index('prepped_models')+2])
	print(prepped_model_path)
	#set up params
	if params is None:
		params = load_cnn_gui_params(prepped_model_path)
		
	kernels = torch.load(prepped_model_path+'/kernels.pt')['kernels']
	kernel_colors = torch.load(prepped_model_path+'/kernels.pt')['kernel_colors']

	#load Model
	model = subgraph_dict['model']
	_ = model.to(params['device']).eval()	

	#set up circuit visualizations
	if viz_folder is None:
		viz_folder = '/'.join(subgraph_dict_path.split('/')[:-2])+'/visualizations/'+subgraph_name
	if not os.path.exists(viz_folder):
		os.mkdir(viz_folder)
		os.mkdir(viz_folder+'/channel')
		os.mkdir(viz_folder+'/neuron')
		with open(viz_folder+'/images.csv', 'a') as images_csv:
			images_csv.write('image_name,targetid,objective,parametrizer,optimizer,transforms,neuron\n')
		images_csv.close()
	

	circuit_traces,pos_dict_nodes,pos_dict_edges = subgraph_2_2d_circuit(subgraph_dict_path)
	circuit_fig=go.Figure(data=circuit_traces, layout=circuit_layout)

	#add circuit images (should probably be its own functions)
	layer = -1
	within_id = 0
	node_df = deepcopy(subgraph_dict['node_df'])
	node_df = node_df.sort_values(by=['node_num'])
	for row in node_df.itertuples():
		layer_name = 'conv_'+str(row.layer)
		if row.layer == layer:
			within_id+=1
		else:
			layer += 1
			within_id = 0
		image_name = fetch_deepviz_img_for_subgraph(model,layer_name,within_id,row.node_num,viz_folder,params)
		img_file_path = viz_folder + '/'+image_name
		img = base64.b64encode(open(img_file_path, 'rb').read())
		
		circuit_fig.add_layout_image(
			dict(
				source='data:image/jpg;base64,{}'.format(img.decode()),
				#source="http://chrishamblin.xyz/images/viscnn_images/%s.jpg"%nodeid,
				x=pos_dict_nodes[layer]['X'][within_id],
				y=pos_dict_nodes[layer]['Y'][within_id],
				sizex=1,
				sizey=1,
				name = pos_dict_nodes[layer]['name'][within_id]
			))

	#fetch kernel images on edges
	kernel_positions = []
	for layer in pos_dict_edges:
		for i in range(len(pos_dict_edges[layer]['name'])):
			edgeid = pos_dict_edges[layer]['name'][i]
			gen_kernel_img(edgeid,kernels,params)
			img_file_path = prepped_model_path+'/visualizations/images/kernels/%s.jpg'%str(edgeid)
			img = base64.b64encode(open(img_file_path, 'rb').read())
			#getting best position
			best_dist=[0,0]
			for pos in [2,3,1,4]:
				far_enough_all=True
				smallest_dist = 1000000000
				for kernel_position in kernel_positions:
					dist, far_enough = min_distance(np.array([pos_dict_edges[layer]['X'][i][pos],pos_dict_edges[layer]['Y'][i][pos]]),np.array(kernel_position))
					if dist < smallest_dist:
						smallest_dist=dist
					if not far_enough:
						far_enough_all=False
				if far_enough_all:
					kernel_positions.append([pos_dict_edges[layer]['X'][i][pos],pos_dict_edges[layer]['Y'][i][pos]])
					break
				elif smallest_dist>best_dist[1]:
					best_dist = [pos,smallest_dist]
				if pos == 4:
					kernel_positions.append([pos_dict_edges[layer]['X'][i][best_dist[0]],pos_dict_edges[layer]['Y'][i][best_dist[0]]])
						
			circuit_fig.add_layout_image(
				dict(
					source='data:image/jpg;base64,{}'.format(img.decode()),
					#source="http://chrishamblin.xyz/images/viscnn_images/%s.jpg"%nodeid,
					#x=pos_dict_edges[layer]['X'][i][2],
					#y=pos_dict_edges[layer]['Y'][i][2],
					x=kernel_positions[-1][0],
					y=kernel_positions[-1][1],
					sizex=.5,
					sizey=.5,
					name=edgeid,
					visible=True
				))        
			

	circuit_fig.update_layout_images(dict(
			xref="x",
			yref="y",
			xanchor="center",
			yanchor="middle"
	))


	#external_stylesheets = ['https://codepen.io/amyoshino/pen/jzXypZ.css']
	external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

	app = dash.Dash(external_stylesheets = external_stylesheets)


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




	state = {'select':False}

	app.layout = html.Div([
			html.Div(
				children = [
					html.Label('Feature Size'),
					dcc.Slider(
							id='feat_size',
							min=.1,
							max=3,
							step=0.0005,
							value = 1
						),
					html.Label('Kernel Size'),
					dcc.Slider(
							id='kernel_size',
							min=.05,
							max=2,
							step=0.0005,
							value = .5
						)
					], className="three columns",
					),
					
				html.Div([
					dcc.Graph(
						id='fig',
						figure=circuit_fig
					)
					], className= 'nine columns'
					),
				dcc.Store(id='memory',data=state)
				], className="row")




	#Network Graph Figure
	@app.callback(
		Output('fig', 'figure'),
		[Input('feat_size', 'value'),
		Input('kernel_size','value'),
		Input('fig', 'clickData')],
		[State('fig','figure')])
	def update_figure(feat_size, kernel_size, clickData, fig):
		print(clickData)
		ctx = dash.callback_context
		if not ctx.triggered:
			raise Exception('no figure updates yet')
		else:
			trigger = ctx.triggered[0]['prop_id']
		print(trigger)
		
		if trigger in ['feat_size.value','kernel_size.value']:
			for i in range(len(fig['layout']['images'])):
				if '-' not in fig['layout']['images'][i]['name']:
					fig['layout']['images'][i]['sizex'] = feat_size
					fig['layout']['images'][i]['sizey'] = feat_size
				else:
					fig['layout']['images'][i]['sizex'] = kernel_size
					fig['layout']['images'][i]['sizey'] = kernel_size
					
		
		#highlight graph
		elif fig['data'][-1]['text'] == 'full':
			
			click_name = circuit_curve_2_id(clickData['points'][0]['curveNumber'],clickData['points'][0]['pointNumber'],subgraph_dict,params)
			highlight = {'nodes':[],'edges':[]}
			#edge clicked
			if '-' in click_name:
				highlight['edges'].append(click_name)
				highlight['nodes'].append(click_name.split('-')[0])
				highlight['nodes'].append(click_name.split('-')[1])
			#node clicked
			else:
				highlight['nodes'].append(click_name)
				layer,within_layer_id,layer_name = nodeid_2_perlayerid(click_name,params)
				for row in subgraph_dict['edge_df'].itertuples():
					if row.layer == layer and row.out_channel==within_layer_id:
						if row.layer != 0:
							in_node = params['layer_nodes'][row.layer-1][1][row.in_channel]
						else:
							in_node = params['imgnode_names'][row.in_channel]
						highlight['edges'].append(str(in_node)+'-'+str(click_name))
						highlight['nodes'].append(str(in_node))
					if row.layer == int(layer)+1 and row.in_channel==int(within_layer_id):
						out_node = params['layer_nodes'][row.layer][1][row.out_channel]
						highlight['edges'].append(str(click_name)+'-'+str(out_node))
						highlight['nodes'].append(str(out_node))
			#highlight from highlight dict
			#print(highlight)
			for i in range(len(fig['data'])-1):
				if i == 0:
					op = []
					for c in ['r','g','b']:
						if c in highlight['nodes']:
							op.append(.99)
						else:
							op.append(.1)
					fig['data'][i]['marker']['opacity'] = op
				#if i <= len(subgraph_dict['node_df']['layer'].unique()):
				#    op = []
				#    for n in fig['data'][i]['text']:
				#        if n in highlight['nodes']:
				#            op.append(.99)
				#        else:
				#            op.append(.1)
				elif i > len(subgraph_dict['node_df']['layer'].unique()):
					old_color = fig['data'][i]['line']['color']
					if fig['data'][i]['text'] in highlight['edges']:
						new_color = ','.join(old_color.split(',')[:-1])+',1)'  #makes opacity full
					else:
						new_color = ','.join(old_color.split(',')[:-1])+',.1)'
					fig['data'][i]['line']['color'] = new_color
			for i in range(len(fig['layout']['images'])):
				if fig['layout']['images'][i]['name'] in highlight['edges'] or fig['layout']['images'][i]['name'] in highlight['nodes']:
					fig['layout']['images'][i]['visible'] = True
				else:
					fig['layout']['images'][i]['visible'] = False
			fig['data'][-1]['text'] = 'partial'
		#reset opacity
		else:
			print('resetting graph opacity')
			for i in range(len(fig['data'])-1):
				if i == 0:
					fig['data'][i]['marker']['opacity'] = .99
				elif i > len(subgraph_dict['node_df']['layer'].unique()):
					old_color = fig['data'][i]['line']['color']
					new_color = ','.join(old_color.split(',')[:-1])+',1)'  #makes opacity full
					fig['data'][i]['line']['color'] = new_color
			for i in range(len(fig['layout']['images'])):
				fig['layout']['images'][i]['visible'] = True
			fig['data'][-1]['text'] = 'full'
		return fig

	app.run_server(port=port)