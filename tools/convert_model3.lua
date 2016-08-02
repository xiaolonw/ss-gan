require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'


network = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_fcn_cls/fcn_80.net'
savefile = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_fcn_cls/FCN.net'

tmp = torch.load(network)
model_FCN = tmp.FCN


list = model_FCN:listModules()
model_FCN_new = nn.Sequential()

for i = 3, #list - 2 do 
	model_FCN_new:add(list[i])
end

model_FCN_new:add(nn.Transpose({2,3},{3,4}))
model_FCN_new:add(nn.View(-1, 40))
model_FCN_new:add(nn.LogSoftMax())

torch.save(savefile, {FCN = model_FCN_new:clearState() })


