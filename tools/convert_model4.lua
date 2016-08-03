require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'


network = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/adversarial_1.net'
savefile = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/joint_Style_GAN.net'

tmp = torch.load(network)
model_FCN = tmp.FCN
model_D = tmp.D 
model_G = tmp.G 


list = model_FCN:listModules()
model_FCN_new = nn.Sequential()

for i = 3, #list - 2 do 
	model_FCN_new:add(list[i])
end

model_FCN_new:add(nn.Transpose({2,3},{3,4}))
model_FCN_new:add(nn.View(-1, 40))
model_FCN_new:add(nn.LogSoftMax())

torch.save(savefile, {G = model_G:clearState(), D = model_D:clearState(), FCN = model_FCN_new:clearState() })


