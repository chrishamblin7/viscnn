import pandas as pd

nodes_df = pd.read_csv('cifar_prunned_ranks.csv')
nodes_wide_df = nodes_df.pivot(index = 'filter_num',columns='class', values='prune_score')

classes = list(nodes_df['class'].unique())
classes.remove('overall')
classes.insert(0,'overall')

def get_layer(filter_num, df = nodes_df, col = 'layer'):
    return df.loc[(df['filter_num'] == filter_num) & (df['class'] == df['class'].unique()[0]), col].item()


nodes_wide_df.reset_index(inplace=True)

nodes_wide_df['layer'] = nodes_wide_df['filter_num'].apply(get_layer)



layers = {}
for index, row in nodes_df[nodes_df['class'] == 'overall'].iterrows(): 
    if row['layer'] not in layers:
        layers[row['layer']] = []
    layers[row['layer']].append(row['filter_num'])
nodes_df.tail(10)


num_layers = max(layers.keys()) + 1

def nodeid_2_perlayerid(nodeid):    #takes in node unique id outputs tuple of layer and within layer id
    layer = nodes_df[nodes_df['class']=='overall'][nodes_df['filter_num'] == nodeid]['layer'].item()
    within_layer_id = nodes_df[nodes_df['class']=='overall'][nodes_df['filter_num'] == nodeid]['filter_num_by_layer'].item()
    return layer,within_layer_id

#nodes_wide_df['filter_num_by_layer'] = nodes_wide_df.apply(lambda row: get_layer(row['filter_num'], col='filter_num_by_layer'), axis = 1)



import numpy as np
from sklearn import manifold
from sklearn.metrics import euclidean_distances

def add_norm_col(df,classes=classes):
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
for layer in layers:
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

def layernum2name(layer,offset=1,title = 'layer'):
    return title+' '+str(layer+offset)



import numpy as np
#import igraph as ig
import json
import urllib

N = len(nodes_wide_df.index)
sizes = list(nodes_wide_df['overall'])
format_sizes = []
for size in sizes:
    format_sizes.append(50*np.cbrt(size))
sizes = format_sizes

#N=len(data_dict['nodes'])


Edges_full = {}
for layer in layers:
    if layer+1 in layers:
        Edges_full[layer+1] = []
        for i in range(len(layers[layer])):
            for j in range(len(layers[layer+1])):
                Edges_full[layer+1].append((layers[layer][i],layers[layer+1][j],i,j))             
       
  
#print(Edges_full)



labels_list=(list(nodes_df[nodes_df['class'] == 'overall'].filter_num))
layers_list=(list(nodes_df[nodes_df['class'] == 'overall'].layer))
N = len(labels_list)



target_class = 'airplane'

#Node Opacity
layer_colors = {0:'rgba(31,119,180,', 
                1:'rgba(255,127,14,',
                2:'rgba(44,160,44,', 
                3:'rgba(214,39,40,',
                4:'rgba(148, 103, 189,', 
                5:'rgba(140, 86, 75,',
                6:'rgba(227, 119, 194,',
                7:'rgba(127, 127, 127,',
                8:'rgba(188, 189, 34,',
                9:'rgba(23, 190, 207,'}


def color_scaling(x):
    return -(x-1)**4+1


def gen_node_colors(target_class):

    node_colors_dict = {}
    for layer in layers:
        node_colors_dict[layer] = []
        for node in layers[layer]:
            alpha = color_scaling(nodes_df[nodes_df['class']==target_class].iloc[node].prune_score)
            node_colors_dict[layer].append(layer_colors[layer]+str(round(alpha,3))+')')
    return node_colors_dict

node_colors_dict = gen_node_colors(target_class)

#print(colors_dict)




#Node positions
#def gen_node_positions()
layer_distance = 1   # distance in X direction each layer is separated by
node_positions = {}
layer_offset = 0
for layer in layer_mds:
    node_positions[layer] = {}
    node_positions[layer]['X'] = [] 
    node_positions[layer]['Y'] = [] 
    node_positions[layer]['Z'] = []  
    for i in range(len(layer_mds[layer])): 
        node_positions[layer]['Y'].append(layer_mds[layer][i][0])
        node_positions[layer]['Z'].append(layer_mds[layer][i][1])
        node_positions[layer]['X'].append(layer_offset)
    layer_offset+=1*layer_distance

#print(node_positions[0])




#Edge selection

def gen_edge_subset_and_weights(target_class,edge_threshold=.1):
    edge_weights = {}
    #edge_threshold = .1
    Edges = {}
    for layer in Edges_full:
        Edges[layer] = []
        edge_weights[layer] = []
        for i in range(len(Edges_full[layer])):
            edge_weight = nodes_df[nodes_df['class']==target_class].iloc[Edges_full[layer][i][0]].prune_score*nodes_df[nodes_df['class']==target_class].iloc[Edges_full[layer][i][1]].prune_score
            if edge_weight > edge_threshold:
                Edges[layer].append(Edges_full[layer][i])
                edge_weights[layer].append(edge_weight)
    return Edges, edge_weights

Edges,edge_weights = gen_edge_subset_and_weights(target_class)
            
#print(Edges)

#Edge Positions
def gen_edge_positions(Edges):
    edge_positions = {}
    for layer in Edges:
        edge_positions[layer] = {}
        edge_positions[layer]['X'] = []
        edge_positions[layer]['Y'] = []
        edge_positions[layer]['Z'] = []
        for edge in Edges[layer]:
            edge_positions[layer]['X']+=([node_positions[layer-1]['X'][edge[2]],node_positions[layer]['X'][edge[3]], None])# x-coordinates of edge ends
            edge_positions[layer]['Y']+=([node_positions[layer-1]['Y'][edge[2]],node_positions[layer]['Y'][edge[3]], None])
            edge_positions[layer]['Z']+=([node_positions[layer-1]['Z'][edge[2]],node_positions[layer]['Z'][edge[3]], None])    
    return edge_positions

edge_positions = gen_edge_positions(Edges)

#print(edge_positions)


#Edge Colors
edge_colors_dict = {}
for layer in Edges:
    edge_colors_dict[layer] = []
    for weight in edge_weights[layer]:
        alpha = color_scaling(weight)
        edge_colors_dict[layer].append(layer_colors[layer]+str(round(alpha,3))+')')



#Format Node Feature Maps
import pickle

activations = pickle.load(open('activations/cifar_prunned_.816_activations.pkl','rb'))
  
    
node_ids = []
for layer in layers:
    for i in range(len(layers[layer])):
        node_ids.append(str(layer+1)+'_'+str(i))
    
print(activations['airplane']['0001.png'][0].shape)




#Format Edge Kernels

kernels = pickle.load(open('kernels/cifar_prunned_.816_kernels.pkl','rb'))
print(kernels[0].shape)


#Function for taking a string of form 'node1-node2' and outputting edge info
def nodestring_2_edge_info(nodestring):
    from_node = int(nodestring.split('-')[0])
    to_node = int(nodestring.split('-')[1])
    from_layer,from_within_id = nodeid_2_perlayerid(from_node)
    to_layer,to_within_id = nodeid_2_perlayerid(to_node)
    kernel = kernels[to_layer][to_within_id][from_within_id]
    return np.flip(kernel,0)


#print(nodestring_2_edge_info('0-47'))



## adding images
import glob
import os

input_image_directory = 'input_images_testing/'
list_of_input_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(input_image_directory))]

static_input_image_route = '/static_input_images/'


# edge_image_directory = '/Users/chrishamblin/Desktop/graph_viz/edge_images/'
# list_of_edge_images = [os.path.basename(x) for x in glob.glob('{}*.png'.format(edge_image_directory))]

# edge_static_image_route = '/static_edge/'
#list_of_input_images[0]


#import chart_studio.plotly as py
import plotly.offline as py    #added
import plotly.graph_objs as go
#py.init_notebook_mode(connected=True)   #added

node_data = []
for layer in layers:
    #add nodes
    node_trace=go.Scatter3d(x=node_positions[layer]['X'],
               y=node_positions[layer]['Y'],
               z=node_positions[layer]['Z'],
               mode='markers',
               name=layernum2name(layer,title = 'nodes'),
               marker=dict(symbol='circle',
                             size=6,
                             opacity=.99,
                             color=node_colors_dict[layer],
                             #colorscale='Viridis',
                             line=dict(color='rgb(50,50,50)', width=.5)
                             ),
               text=layers[layer],
               hoverinfo='text'
               )
        
    node_data.append(node_trace)
    
edge_data = []    
for layer in edge_positions:        
    #add edges      
    edge_trace=go.Scatter3d(x=edge_positions[layer]['X'],
                            y=edge_positions[layer]['Y'],
                            z=edge_positions[layer]['Z'],
                            name=layernum2name(layer ,title = 'edges'),
                            mode='lines',
                            #line=dict(color=edge_colors_dict[layer], width=1.5),
                            line=dict(color='rgb(100,100,100)', width=1.5),
                            text = list(range(len(Edges[layer])))
                            #hoverinfo='text'
                            )
    edge_data.append(edge_trace)

 
combined_data = node_data+edge_data


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

graph_layout = go.Layout(
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
         ),
         uirevision =  True   
         #hovermode='closest',
   )


fig=go.Figure(data=combined_data, layout=graph_layout)




import dash
import dash_core_components as dcc
import dash_html_components as html
#import utils.dash_reusable_components as drc
import flask
import os

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
            html.Label('Lower Edge Threshold'),
                dcc.Slider(
                    id='lower-thresh-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    marks={i/10: str(i/10) for i in range(0,12,2)},
                    value=.1,
                ),
                
            html.Br(),
            html.Label('Upper Edge Threshold'),
                dcc.Slider(
                    id='upper-thresh-slider',
                    min=0,
                    max=1,
                    step=0.01,
                    marks={i/10: str(i/10) for i in range(0,12,2)},
                    value=1,
                ),
                
            ], className="two columns"
        ),

        html.Div(
            children = [
                
            html.Div([
                dcc.Graph(
                    id='network-graph',
                    figure=fig
                )
            ], className= 'row'
            ),
                
            html.Div([
                html.Div([
                html.Label('Input Image'),
                dcc.Dropdown(
                    id='input-image-dropdown',
                    options=[{'label': i, 'value': i} for i in list_of_input_images],
                    value=list_of_input_images[6]
                ),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Br(),
                html.Img(id='input-image')#,style={'height':'200%', 'width':'200%'}
                ], className = "three columns"),
                
                html.Div([
                html.Label('Node'),
                dcc.Dropdown(
                    id='node-actmap-dropdown',
                    options=[{'label': str(i), 'value': i} for i in range(N)],
                    value=0
                ),
                dcc.Graph(
                    id='node-actmap-graph',
                    figure=go.Figure(data=go.Heatmap(
                                        z = np.flip(activations[list_of_input_images[0].split('_')[0]][list_of_input_images[0].split('_')[1]][0][0],0)),
                                        layout=dict(
                                            height=400,
                                            width=400)
                                    ),
                    config={
                            'displayModeBar': False
                            }
                )
                ], className = "four columns"),
                
                html.Div([
                html.Label('Edge'),    
                dcc.Input(
                    id='edge-kernel-input',value='0-%s'%str(layers[1][0]), type='text'),
                html.Button(id='edge-kernel-button',n_clicks=0, children='Submit'),
                dcc.Graph(
                    id='edge-kernel-graph',
                    figure=go.Figure(data=go.Heatmap(
                                        z = nodestring_2_edge_info('0-%s'%str(layers[1][0]))
                                        ),
                                        layout=dict(
                                            height=300,
                                            width=300)
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
                ], className='three columns')
                
            ], className= 'row'
            )
        ], className="ten columns"
        )
    ]
)




####Call Back Functions

# @app.callback(
#     Output('figure-data', 'children'),
#     [Input('network-graph', 'figure')])
# def display_figure_data(figure):
#     return json.dumps(figure, indent=2)



# @app.callback(
#     Output('network-graph', 'figure'),
#     [Input('network-graph', 'clickData')])
# def highlight_on_click(clickData):
#     if clickData['points'][0]['curveNumber'] == None:
#         raise Exception('no point clicked yet') 
#     trace_num = int(clickData['points'][0]['curveNumber'])
#     if trace_num < num_layers:   #highlight point
#         for layer in node_colors_dict:
#             if layer == trace_num:
#                 new_colors = list(node_colors_dict[trace_num])
#                 new_colors[clickData['points'][0]['pointNumber']] = 'rgba(0,0,0,1)'
#                 combined_data[trace_num]['marker']['color'] = new_colors
#             else:
#                 combined_data[layer]['marker']['color'] = node_colors_dict[layer]
#     else: #highlight edge
#         #raise Exception('lets skip edges for now') 
#         for layer in Edges:
#             new_colors = list(['rgb(125,125,125)' for i in range(len(Edges[layer]))])
#             #new_colors = edge_colors_dict[layer]
#             if layer == trace_num-num_layers+1:
#                 new_colors[clickData['points'][0]['text']] = 'rgba(150,0,0,1)'
#             combined_data[layer]['line']['color'] = new_colors
    
#     layout = graph_layout
#     layout['uirevision'] = True
#     return {'data': combined_data,
#             'layout': layout}



@app.callback(
    Output('node-actmap-dropdown', 'value'),
    [Input('network-graph', 'clickData')])
def switch_node_actmap_click(clickData):
    if clickData['points'][0]['curveNumber'] == None:
        raise Exception('no point clicked yet') 
    if int(clickData['points'][0]['curveNumber']) >= num_layers:
        raise Exception('Do nothing, they clicked an edge')
    return int(clickData['points'][0]['text'])


         
#cant currently click edges
# @app.callback(
#     Output('edge-image-dropdown', 'value'),
#     [Input('network-graph', 'clickData')])
# def switch_edge_image_click(clickData):
#     if int(clickData['points'][0]['curveNumber']) < num_layers:
#         raise Exception('Do nothing, they clicked a node')
#     return list_of_edge_images[int(clickData['points'][0]['pointNumber'])]



#Node activation map
@app.callback(
    Output('node-actmap-graph', 'figure'),
    [Input('node-actmap-dropdown', 'value'),
     Input('input-image-dropdown', 'value')])
def update_node_actmap(node_id,image_name):
    layer, within_id = nodeid_2_perlayerid(node_id)
    
    return go.Figure(data=go.Heatmap( z = np.flip(activations[image_name.split('_')[0]][image_name.split('_')[1]][layer][within_id],0)),
                     layout=dict(height=400,
                                 width=400,
                                 uirevision=True)) 
#     return {'data':go.Heatmap(
#                               z = activations[image_name.split('_')[0]][image_name.split('_')[1]][layer][within_id]),
#             'layout':dict(height=500,width=500)}
 
    
      

@app.callback(
    Output('edge-kernel-graph', 'figure'),
    [Input('edge-kernel-button','n_clicks')],
    [State('edge-kernel-input', 'value')])
def update_edge_kernelmap(n_clicks,nodestring):
    return go.Figure(data=go.Heatmap(z = nodestring_2_edge_info(nodestring)),
                     layout=dict(height=300,
                                 width=300,
                                 uirevision=True)) 
                

#Input Images
@app.callback(
    Output('input-image', 'src'),
    [Input('input-image-dropdown', 'value')])
def update_input_image_src(value):
    return static_input_image_route + value

@app.server.route('{}<image_path>.png'.format(static_input_image_route))
def serve_input_image(image_path):
    image_name = '{}.png'.format(image_path)
    if image_name not in list_of_input_images:
        raise Exception('"{}" is excluded from the allowed static files'.format(image_path))
    return flask.send_from_directory(input_image_directory, image_name)




#JSON INFO

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
    Output('network-graph', 'figure'),
    [Input('weight-category', 'value'),
     Input('network-graph', 'clickData'),
     Input('lower-thresh-slider','value')])
def update_figure(target_class,clickData,edge_thresh):
    node_colors_dict = gen_node_colors(target_class)
    Edges,edge_weights = gen_edge_subset_and_weights(target_class,edge_threshold=edge_thresh)
    edge_positions = gen_edge_positions(Edges)
    click_layer = int(clickData['points'][0]['curveNumber'])
    for layer in node_colors_dict:
        if layer == click_layer:
            new_colors = list(node_colors_dict[click_layer])
            new_colors[clickData['points'][0]['pointNumber']] = 'rgba(0,0,0,1)'
            combined_data[layer]['marker']['color'] = new_colors
        else:
            combined_data[layer]['marker']['color'] = node_colors_dict[layer]

    for layer in edge_positions:
        combined_data[layer-1+num_layers] = go.Scatter3d(x=edge_positions[layer]['X'],
                                y=edge_positions[layer]['Y'],
                                z=edge_positions[layer]['Z'],
                                name=layernum2name(layer ,title = 'edges'),
                                mode='lines',
                                #line=dict(color=edge_colors_dict[layer], width=1.5),
                                line=dict(color='rgb(100,100,100)', width=1.5),
                                text = list(range(len(Edges[layer])))
                                #hoverinfo='text'
                                )
   
    layout = graph_layout
    layout['uirevision'] = True
    return {'data': combined_data,
            'layout': layout}




app.run_server()












