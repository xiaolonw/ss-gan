require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'


network = '/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_72/save2/adversarial_12.net'
savefile = '/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_72/save2/Structure_GAN.net'

tmp = torch.load(network)
model_D = tmp.D
model_G = tmp.G


list = model_G:listModules()
model_G_new = nn.Sequential()


for i = 2, #list - 3 do 
	model_G_new:add(list[i])
end

model_G_new:add(nn.Transpose({2,3},{3,4}))
model_G_new:add(nn.View(-1, 3))
model_G_new:add(nn.Normalize(2))
model_G_new:add(nn.View(-1, 72, 72, 3))
model_G_new:add(nn.Transpose({4,3},{3,2}))

torch.save(savefile, {D = model_D:clearState(), G = model_G_new:clearState() })


