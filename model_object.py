from torch.nn import CrossEntropyLoss
from torch.optim import *
from Net import *
from utils.api import *
from loss_function import joint_loss, loss_in_IMF
from utils.basic import get_scheduler
from Dataset import *

models = {
'ITCFN':{
        'Name': 'Triple_model_CoAttention_Fusion',
        'Model': Triple_model_CoAttention_Fusion,
        'dataset': MriPetCliDataset,
        'shape': (96, 128, 96),
        'Loss': joint_loss,
        'Optimizer': Adam,
        'Lr': 0.0001,
        'w1': 0.2,
        'w2': 0.01,
        'Scheduler': get_scheduler,
        'Run': run_main_1
    },
'ITFN':{
        'Name': 'Triple_model_Fusion',
        'Model': Triple_model_Fusion,
        'dataset': MriPetCliDataset,
        'shape': (96, 128, 96),
        'Loss': joint_loss,
        'Optimizer': Adam,
        'Lr': 0.0001,
        'w1': 0.2,
        'w2': 0.01,
        'Scheduler': get_scheduler,
        'Run': run_main_1
    },
'TFN':{
        'Name': 'Triple_model_Fusion_Incomplete',
        'Model': Triple_model_Fusion,
        'dataset': MriPetCliDataset,
        'shape': (96, 128, 96),
        'Loss': joint_loss,
        'Optimizer': Adam,
        'Lr': 0.0001,
        'w1': 0.2,
        'w2': 0.01,
        'Scheduler': get_scheduler,
        'Run': run_main_1
    },
'TCFN':{
        'Name': 'Triple_model_CoAttention_Fusion_Incomplete',
        'Model': Triple_model_CoAttention_Fusion,
        'dataset': MriPetCliDataset,
        'shape': (96, 128, 96),
        'Loss': joint_loss,
        'Optimizer': Adam,
        'Lr': 0.0001,
        'w1': 0.2,
        'w2': 0.01,
        'Scheduler': get_scheduler,
        'Run': run_main_1
    },
'HFBSurv': {
        'Name': 'HFBSurv',
        'Lr': 0.0001,
        'Model': HFBSurv,
        'dataset': MriPetCliDataset,
        'shape': (96, 128, 96),
        'Optimizer': Adam,
        'Loss': CrossEntropyLoss,
        'Scheduler': get_scheduler,
        'Run': run_main_for_hfbsurve,
    },
'IMF':{
        'Name': 'Interactive_Multimodal_Fusion_Model',
        'Model': Interactive_Multimodal_Fusion_Model,
        'dataset': MriPetCliDatasetWithTowLabel,
        'shape': (96, 128, 96),
        'Loss': loss_in_IMF,
        'Optimizer': Adam,
        'Lr': 0.0001,
        'w1': 0.2,
        'w2': 0.01,
        'Scheduler': get_scheduler,
        'Run': run_main_for_IMF,
    },
'AweSomeNet':{
        'Name': 'AweSomeNet',
        # generate_model(model_depth=18, in_planes=1, num_classes=2)
        'Model': AweSomeNet,
        'dataset': MriPetCliDataset,
        'shape': (96, 128, 96),
        'Loss': CrossEntropyLoss,
        'Optimizer': Adam,
        'Lr': 0.001,
        'Run': run_main_for_awesome_net,
        'Scheduler': get_scheduler,
},
'Resnet':{
        'Name': 'ResnetMriPet',
        # generate_model(model_depth=18, in_planes=1, num_classes=2)
        'Model': ResnetMriPet,
        'dataset': MriPetDataset,
        'shape': (96, 128, 96),
        'Loss': CrossEntropyLoss,
        'Optimizer': Adam,
        'Lr': 0.001,
        'Run': run_main_for_resnet,
        'Scheduler': get_scheduler,
},

}
