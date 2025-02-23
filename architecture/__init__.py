import torch
from .edsr import EDSR
from .HDNet import HDNet
from .hinet import HINet
from .hrnet import SGN
from .HSCNN_Plus import HSCNN_Plus
from .MIRNet import MIRNet
from .MPRNet import MPRNet
from .MST import MST
from .MST_Plus_Plus import MST_Plus_Plus
from .Restormer import Restormer
from .AWAN import AWAN

def model_generator(method, pretrained_model_path=None):
    if method == 'mirnet':
        model = MIRNet(n_RRG=3, n_MSRB=1, height=3, width=1)
    elif method == 'mst_plus_plus':
        model = MST_Plus_Plus()
    elif method == 'mst':
        model = MST(dim=31, stage=2, num_blocks=[4, 7, 5])
    elif method == 'hinet':
        model = HINet(depth=4)
    elif method == 'mprnet':
        model = MPRNet(num_cab=4)
    elif method == 'restormer':
        model = Restormer()
    elif method == 'edsr':
        model = EDSR()
    elif method == 'hdnet':
        model = HDNet()
    elif method == 'hrnet':
        model = SGN()
    elif method == 'hscnn_plus':
        model = HSCNN_Plus()
    elif method == 'awan':
        model = AWAN()
    else:
        print(f'Method {method} is not defined !!!!')
    if pretrained_model_path is not None:
        print(f'load model from {pretrained_model_path}')
        checkpoint = torch.load(pretrained_model_path)
        model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()},
                              strict=True)
    return model

if __name__ == "__main__":
    methods = ['mirnet', 'mst_plus_plus', 'mst', 'hinet', 'mprnet', 'restormer', 'edsr', 'hdnet', 'hrnet', 'hscnn_plus', 'awan']
    input_shape = (2, 3, 128, 128)
    output_shape = (2, 31, 128, 128)
    X = torch.rand(input_shape)

    for method in methods:
        model = model_generator(method)
        output_shape = model(X)
        print(f"Model Name: {method}, input shape: {input_shape}, output_shape:{output_shape.shape}")

