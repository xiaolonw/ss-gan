require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'nnx'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end



----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_jointall_bi_s4")      subdirectory to save logs
  --saveFreq         (default 3)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.000001)      learning rate
  -b,--batchSize     (default 128)         batch size
  -m,--momentum      (default 0.5)         momentum term of adam
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)     L2 penalty on the weights
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 128)          scale of images to train on
  --epochSize        (default 500)        number of samples per epoch
  --scratch          (default 0)
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --classnum         (default 31)          number of classnum
  --learningRate_FCN  (default 0)      learning rate
  --classnum         (default 40)    
  --classification   (default 1)
  --lamda0           (default 0.1)
  --lamda            (default 1)
  --trainD1          (default 1)
  --trainG1          (default 1)
  --trainD           (default 1)
  --optimG           (default 1)
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
opt.network1 = '/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_72/save2/Structure_GAN.net'
opt.network3 = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/joint_Style_GAN.net'
opt.pause = 0
print(opt)

opt.loadSize  = opt.scale 
opt.labelSize  = opt.scale 
opt.labelSize2 = 72
opt.flag = 1

-- fix seed
-- torch.manualSeed(1)
opt.manualSeed = torch.random(1,10000) -- torch.random(1,10000) -- fix seed
print("Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. torch.getnumthreads())

if opt.gpu then
  cutorch.setDevice(opt.gpu + 1)
  print('<gpu> using device ' .. opt.gpu)
  torch.setdefaulttensortype('torch.CudaTensor')
else
  torch.setdefaulttensortype('torch.FloatTensor')
end


classes = {'0','1'}
-- opt.noiseDim = {1, opt.scale / 4, opt.scale / 4}
opt.geometry = {3, opt.scale, opt.scale}
opt.condDim = {3, opt.scale, opt.scale}

function model_load(filename)
   local net = torch.load(filename)
   net:apply(function(m) if m.weight then 
      m.gradWeight = m.weight:clone():zero(); 
      m.gradBias = m.bias:clone():zero(); end end)
   return net
end


if opt.pause == 0 then 
  print('<trainer> reloading previously trained network: ' .. opt.network1)
  tmp = torch.load(opt.network1)
  -- print('<trainer> reloading previously trained network: ' .. opt.network2)
  -- tmp2 = model_load(opt.network2)
  print('<trainer> reloading previously trained network: ' .. opt.network3)
  tmp3 = torch.load(opt.network3)
  model_D1 = tmp.D
  model_G1 = tmp.G
  model_D =  tmp3.D
  model_G =  tmp3.G
else
  tmp = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_jointall_bi_s4/save_dcgan_normal_72/adversarial_G_1.net')
  model_G1 = tmp.G1
  model_G  = tmp.G
  tmp2 = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_jointall_bi_s4/save_dcgan_normal_72/adversarial_D_1.net')
  model_D1 = tmp2.D1
  model_D  = tmp2.D
  
end


model_upsample = nn.Sequential()
model_upsample:add(nn.SpatialReSampling({owidth=128,oheight=128}) )


model_upsample:add(nn.Transpose({2,3},{3,4}))
model_upsample:add(nn.View(-1, 3))
model_upsample:add(nn.Normalize(2))
model_upsample:add(nn.View(-1, 128, 128, 3))
model_upsample:add(nn.Transpose({4,3},{3,2}))


-- model_upsample:add(nn.ReArrange())
-- model_upsample:add(nn.Normalize(2))
-- model_upsample:add(nn.ReArrangeBack(128, 128))


model_upsample:float()

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()
parameters_D1,gradParameters_D1 = model_D1:getParameters()
parameters_G1,gradParameters_G1 = model_G1:getParameters()


-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

paths.dofile('data.lua')
adversarial = paths.dofile('joint_train_bi.lua')

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()  
  model_D1:cuda()
  model_G1:cuda()
  -- model_upsample:cuda()
  criterion:cuda()
end

-- Training parameters
adamState_D = {
  learningRate = opt.learningRate,
  beta1 = opt.momentum,
  optimize = true,
  numUpdates = 0
}

adamState_G = {
  learningRate = opt.learningRate,
  beta1 = opt.momentum,
  optimize = true,
  numUpdates = 0
}

adamState_D1 = {
  learningRate = opt.learningRate * 0.001,
  beta1 = opt.momentum,
  optimize = true,
  numUpdates = 0
}

adamState_G1 = {
  learningRate = opt.learningRate * 0.001,
  beta1 = opt.momentum,
  optimize = true,
  numUpdates = 0
}

local function train()
   print('\n<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. adamState_D.learningRate .. ', momentum = ' .. adamState_D.beta1 .. ']')
  
   confusion:zero()
   model_D:training()
   model_G:training()
   batchNumber = 0
   for i=1,opt.epochSize do
      donkeys:addjob(
         function()
            return makeData(trainLoader:sample(opt.batchSize))
         end,
         adversarial.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()
   print(confusion)
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

end


epoch = 1
-- training loop
while true do
  -- train/test
  train()

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('adversarial_D_%d.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D1 = model_D1:clearState(), D = model_D:clearState(), opt = opt})
    
    filename = paths.concat(opt.save, string.format('adversarial_G_%d.net',epoch))
    print('<trainer> saving network to '..filename)
    torch.save(filename, {G1 = model_G1:clearState(), G = model_G:clearState(), opt = opt})
  end

  epoch = epoch + 1

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    trainLogger:style{['% mean class accuracy (train set)'] = '-'}
    testLogger:style{['% mean class accuracy (test set)'] = '-'}
    trainLogger:plot()
    testLogger:plot()

    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
