require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


local sanitize = require('sanitize')


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_fcn_cls")      subdirectory to save logs
  --saveFreq         (default 5)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.01)      learning rate
  -b,--batchSize     (default 10)         batch size
  -m,--momentum      (default 0.9)         momentum term of adam
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0.0005)     L2 penalty on the weights
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 512)          scale of images to train on
  --epochSize        (default 2000)        number of samples per epoch
  --scratch          (default 0)
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --weightDecay      (default 0.0005)        weight decay
  --classnum         (default 40)    
  --classification   (default 1)
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

opt.flag = 1

opt.loadSize  = opt.scale 
opt.labelSize = opt.scale / 16

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
opt.outDim =  opt.classnum


local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.01)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ---------------------------------------------------------------------

  local nplanes = 64
  local inputsize = 128

  dx_I = nn.Identity()()
  dh1 = nn.SpatialConvolution(3, 96, 11, 11, 4, 4, 100 - 32, 100 - 32)(dx_I)  -- size / 4
  db1 = nn.SpatialBatchNormalization(96)(dh1)
  dr1 = nn.LeakyReLU(0.2, true)(db1)

  dp1 = nn.SpatialMaxPooling(3,3,2,2)(dr1)  -- size / 2

  dh2 = nn.SpatialConvolution(96, 256, 5, 5, 1, 1, 2, 2)(dp1)  -- same size
  db2 = nn.SpatialBatchNormalization(256)(dh2)
  dr2 = nn.ReLU(true)(db2)

  dp2 = nn.SpatialMaxPooling(3,3,2,2)(dr2)  -- size / 2

  dh3 = nn.SpatialConvolution(256, 384, 3, 3, 1, 1, 1, 1)(dp2) -- same size 
  db3 = nn.SpatialBatchNormalization(384)(dh3)
  dr3 = nn.ReLU(true)(db3)

  dh4 = nn.SpatialConvolution(384, 384, 3, 3, 1, 1, 1, 1)(dr3) -- same size
  db4 = nn.SpatialBatchNormalization(384)(dh4)
  dr4 = nn.ReLU(true)(db4)

  dh5 = nn.SpatialConvolution(384, 256, 3, 3, 1, 1, 1, 1)(dr4) -- same size
  db5 = nn.SpatialBatchNormalization(256)(dh5)
  dr5 = nn.ReLU(true)(db5)

  dp5 = nn.SpatialMaxPooling(3,3,2,2)(dr5)  -- size / 2

  dh6 = nn.SpatialConvolution(256, 1024, 6, 6, 1, 1, 1, 1)(dp5)
  db6 = nn.SpatialBatchNormalization(1024)(dh6)
  dr6 = nn.ReLU(true)(db6)
  -- dp6 = nn.Dropout()(dr6)

  dh7 = nn.SpatialFullConvolution(1024, 512, 4, 4, 2, 2, 1, 1)(dr6) 
  db7 = nn.SpatialBatchNormalization(512)(dh7)
  dr7 = nn.ReLU(true)(db7)
  -- dp7 = nn.Dropout()(dr7)

  dh8 = nn.SpatialConvolution(512, opt.classnum, 3, 3, 1, 1, 1, 1)(dr7) 
  -- dr8 = nn.LeakyReLU(0.2, true)(dh8)
  -- dout = nn.SpatialUpSamplingNearest(8)(dr8)
  -- dh9 = nn.SpatialFullConvolution(3, 3, 16, 16, 8, 8, 8, 8)(dr8) 

  rea = nn.ReArrange()(dh8)
  dout = nn.LogSoftMax()(rea)

  model_FCN = nn.gModule({dx_I}, {dout})

  model_FCN:apply(weights_init)


else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_FCN = tmp.FCN
end

-- loss function: negative log-likelihood
criterion = nn.ClassNLLCriterion()




-- retrieve parameters and gradients
parameters_FCN,gradParameters_FCN = model_FCN:getParameters()

-- print networks
print('fcn network:')
print(model_FCN)

paths.dofile('data.lua')
fcn = paths.dofile('fcn_train_cls.lua')


if opt.gpu then
  print('Copy model to gpu')
  model_FCN:cuda()
  criterion:cuda()

end


local optimState = {
    learningRate = opt.learningRate,
    learningRateDecay = 0.0,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}


local function train()
   print('\n<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ' lr = ' .. optimState.learningRate .. ', momentum = ' .. optimState.momentum .. ']')
  
   model_FCN:training()
   batchNumber = 0
   for i=1,opt.epochSize do
      donkeys:addjob(
         function()
            return makeData_cls(trainLoader:sample(opt.batchSize))
         end,
         fcn.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()

end

local filezero = paths.concat(opt.save, string.format('fcn_zero.net'))
torch.save(filezero, { FCN = sanitize(model_FCN), opt = opt})


epoch = 1
-- training loop
while true do
  -- train/test
  train()

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('fcn_%d.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, { FCN = sanitize(model_FCN), opt = opt})
  end

  epoch = epoch + 1

  -- plot errors
  if opt.plot  and epoch and epoch % 1 == 0 then
    torch.setdefaulttensortype('torch.FloatTensor')

    if opt.gpu then
      torch.setdefaulttensortype('torch.CudaTensor')
    else
      torch.setdefaulttensortype('torch.FloatTensor')
    end
  end
end
