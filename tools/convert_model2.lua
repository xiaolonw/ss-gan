require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'


network = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/joint_Style_GAN.net'
savefile = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/joint_Style_GAN.net'

tmp = torch.load(network)
model_D = tmp.D
model_G = tmp.G
model_FCN = tmp.FCN 

torch.save(savefile, {D = model_D:clearState(), G = model_G:clearState(), FCN = model_FCN:clearState()})


