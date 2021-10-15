import plotly.graph_objs as go



'Layouts for cnn_gui.py'
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
        height=700,
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

double_image_layout = go.Layout(#width=400, 
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
    #width=50,
    #height=50,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ))

edge_outmap_layout = go.Layout(
    #title = 'edge output map',
    #autosize=False,
    #width=60,
    #height=60,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
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







'Layouts for circuit_gui.py'
circuit_layout = go.Layout(
         #title="%s through Prunned Cifar10 CNN"%target_category,
         #title = target_category,
         #width=1000,
         clickmode = 'event+select',
         transition = {'duration': 20},
         height=1300,
         #showlegend=False,
         margin = dict(l=20, r=20, t=20, b=20),  
         #hovermode='closest',
         paper_bgcolor='rgba(0,0,0,0)',
         plot_bgcolor='rgba(0,0,0,0)',
         xaxis=dict(showline=False,showgrid=False,showticklabels=False),
         yaxis=dict(showline=False,showgrid=False,showticklabels=False))


circuit_kernel_layout = go.Layout(
    #title='kernel'
    #autosize=False,
    width=50,
    height=50,
    uirevision = True,
    margin=dict(
        l=1,
        r=1,
        b=1,
        t=1,
        pad=1
    ),
    xaxis=dict(showline=False,showgrid=False,showticklabels=False),
    yaxis=dict(showline=False,showgrid=False,showticklabels=False))