require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'


network = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_rgb/bactch60/adversarial_27.net'
savefile = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_rgb/bactch60/Style_GAN_nofcn.net'

tmp = torch.load(network)
model_D = tmp.D
model_G = tmp.G
model_FCN = tmp.FCN 

torch.save(savefile, {D = model_D:clearState(), G = model_G:clearState(), FCN = model_FCN:clearState()})


