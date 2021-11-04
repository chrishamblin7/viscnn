from subprocess import call

model = 'alexnet_sparse'
layers = ['features_6','features_8','features_10']
data_path = '/mnt/data/chris/dropbox/Research-Hamblin/Projects/cnn_subgraph_visualizer/image_data/imagenet_2/'


layers = ['features_6','features_8','features_10']
Ts = [1,2,8]
units = range(20)
ratios = [.5,.1,.05,.01]
device = 'cuda:2'

for layer in layers:
    for unit in units:
        feature_target = {layer:[unit]}
        for T in Ts:
            for ratio in ratios:
                call('python force_prune.py --T %s --ratio %s --unit %s --layer %s --model %s --data-path %s --device %s'%(str(T),str(ratio),str(unit),layer,model,data_path,device),shell=True)