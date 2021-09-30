#Loadable dash "layouts" for visualizer

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

