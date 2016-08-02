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
  -s,--save          (default "/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_rgb")      subdirectory to save logs
  --saveFreq         (default 3)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.0002)      learning rate
  -b,--batchSize     (default 80)         batch size
  -m,--momentum      (default 0.5)         momentum term of adam
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0.00001)     L2 penalty on the weights
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 128)          scale of images to train on
  --epochSize        (default 2000)        number of samples per epoch
  --scratch          (default 0)
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --high             (default 1)           high resolution 
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
-- opt.network = '/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_rgb/save3/adversarial_6.net'

print(opt)

opt.loadSize  = opt.scale 

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




local function weights_init(m)
   local name = torch.type(m)
   if name:find('Convolution') then
      m.weight:normal(0.0, 0.02)
      m.bias:fill(0)
   elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:fill(0) end
   end
end

local input_sz = opt.geometry[1] * opt.geometry[2] * opt.geometry[3]

if opt.network == '' then
  ----------------------------------------------------------------------
  -- my D network 
  -- (64-128-256-256-128)
  -- (36-36-18-9-9)
  -- (5-5-3-3-3)

  local nplanes = 64
  local inputsize = 128
  dx_I = nn.Identity()()
  dx_C = nn.Identity()()
  dc1 = nn.JoinTable(2, 2)({dx_I, dx_C})
  dh1 = nn.SpatialConvolution(3 + 3, nplanes, 5, 5, 2, 2, 2, 2)(dc1)  -- size / 2 
  -- db1 = nn.SpatialBatchNormalization(nplanes)(dh1)
  dr1 = nn.LeakyReLU(0.2, true)(dh1)

  dh2 = nn.SpatialConvolution(nplanes, nplanes * 2, 5, 5, 2, 2, 2, 2)(dr1) -- size / 2 
  -- db2 = nn.SpatialBatchNormalization(nplanes * 2)(dh2)
  dr2 = nn.LeakyReLU(0.2, true)(dh2)

  dh3 = nn.SpatialConvolution(nplanes * 2, nplanes * 4, 3, 3, 2, 2, 1, 1)(dr2) -- size / 2
  -- db3 = nn.SpatialBatchNormalization(nplanes * 4)(dh3)
  dr3 = nn.LeakyReLU(0.2, true)(dh3)
  dp3 = nn.SpatialDropout(0.2)(dr3)

  dh4 = nn.SpatialConvolution(nplanes * 4, nplanes * 8, 3, 3, 2, 2, 1, 1)(dp3) -- size / 2
  -- db4 = nn.SpatialBatchNormalization(nplanes * 4)(dh4)
  dr4 = nn.LeakyReLU(0.2, true)(dh4)
  dp4 = nn.SpatialDropout(0.2)(dr4)

  dh5 = nn.SpatialConvolution(nplanes * 8, nplanes * 2, 3, 3, 1, 1, 1, 1)(dp4) -- same size
  -- db5 = nn.SpatialBatchNormalization(nplanes * 2)(dh5)
  dr5 = nn.LeakyReLU(0.2, true)(dh5)

  local outputsize3 =  nplanes * 2 * (inputsize / 16) * (inputsize / 16) 
  rshp = nn.Reshape(outputsize3)(dr5)
  dh6 = nn.Linear(outputsize3, 1)(nn.Dropout()(rshp))
  dout = nn.Sigmoid()(dh6)
  model_D = nn.gModule({dx_I, dx_C}, {dout})

  model_D:apply(weights_init)

  

  ----------------------------------------------------------------------
  -- define G network to train
  -- my G network
  local nplanes = 64
  x_I = nn.Identity()()  -- 1 channel noise
  x_C = nn.Identity()()  -- 3 channel surface normal 
  h1 = nn.SpatialConvolution(3, nplanes, 5, 5, 2, 2, 2, 2)(x_C)  -- size / 2
  b1 = nn.SpatialBatchNormalization(nplanes)(h1) 
  r1 = nn.ReLU(true)(b1) 

  h2 = nn.SpatialConvolution( nplanes, nplanes * 2, 5, 5, 2, 2, 2, 2)(r1) -- size / 2  32 * 32
  b2 = nn.SpatialBatchNormalization(nplanes * 2)(h2) 
  r2 = nn.ReLU(true)(b2)
  
  hi1 = nn.SpatialFullConvolution(opt.noiseDim, nplanes , 8, 8)(x_I) 
  bi1 = nn.SpatialBatchNormalization(nplanes)(hi1)
  ri1 = nn.ReLU(true)(bi1)


  hi2 = nn.SpatialFullConvolution(nplanes, nplanes , 4, 4, 2, 2, 1, 1)(ri1) -- size * 2 for noise 16 * 16 
  bi2 = nn.SpatialBatchNormalization(nplanes)(hi2)
  ri2 = nn.ReLU(true)(bi2)

  hi3 = nn.SpatialFullConvolution(nplanes , nplanes , 4, 4, 2, 2, 1, 1)(ri2) -- size * 2 for noise 32 * 32 
  bi3 = nn.SpatialBatchNormalization(nplanes)(hi3)
  ri3 = nn.ReLU(true)(bi3)


  c2 = nn.JoinTable(2, 2)({r2, ri3 })  -- merge 

  h3 = nn.SpatialConvolution(nplanes*2 + nplanes , nplanes * 4, 3, 3, 1, 1, 1, 1)(c2) -- the same size
  b3 = nn.SpatialBatchNormalization(nplanes * 4)(h3) 
  r3 = nn.ReLU(true)(b3)

  h4 = nn.SpatialConvolution(nplanes*4, nplanes * 8, 3, 3, 2, 2, 1, 1)(r3) -- size / 2 
  b4 = nn.SpatialBatchNormalization(nplanes * 8)(h4) 
  r4 = nn.ReLU(true)(b4)

  h5 = nn.SpatialConvolution(nplanes*8, nplanes * 8, 3, 3, 1, 1, 1, 1)(r4) -- same size
  b5 = nn.SpatialBatchNormalization(nplanes * 8)(h5) 
  r5 = nn.LeakyReLU(0.2, true)(b5)

  h6 = nn.SpatialFullConvolution(nplanes*8, nplanes * 4, 4, 4, 2, 2, 1, 1)(r5) -- size * 2
  b6 = nn.SpatialBatchNormalization(nplanes * 4)(h6) 
  r6 = nn.LeakyReLU(0.2, true)(b6)

  h7 = nn.SpatialFullConvolution(nplanes*4, nplanes * 2, 4, 4, 2, 2, 1, 1)(r6) -- size * 2
  b7 = nn.SpatialBatchNormalization(nplanes * 2)(h7) 
  r7 = nn.LeakyReLU(0.2, true)(b7)
  -- u7 = nn.SpatialUpSamplingNearest(2)(r7)  -- size * 2

  h8 = nn.SpatialFullConvolution(nplanes*2, nplanes , 4, 4, 2, 2, 1, 1)(r7)  -- size * 2
  b8 = nn.SpatialBatchNormalization(nplanes)(h8)
  r8 = nn.LeakyReLU(0.2, true)(b8)

  h9 = nn.SpatialConvolution(nplanes, 3 , 5, 5, 1, 1, 2, 2)(r8) -- same size

  gout = nn.Tanh()(h9)

  model_G = nn.gModule({x_I, x_C}, {gout})

  model_G:apply(weights_init)

else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()
-- 2nd loss function
criterion2 = nn.MSECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

paths.dofile('data.lua')
adversarial = paths.dofile('conditional_adversarial_rgb.lua')


-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

if opt.gpu then
  print('Copy model to gpu')
  model_D:cuda()
  model_G:cuda()
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
            return makeData(trainLoader:sample(opt.batchSize)),
                   makeData(trainLoader:sample(opt.batchSize))
         end,
         adversarial.train)
   end
   donkeys:synchronize()
   cutorch.synchronize()
   print(confusion)
   -- tr_acc0 = confusion.valids[1] * 100
   -- tr_acc1 = confusion.valids[2] * 100
   -- if tr_acc0 ~= tr_acc0 then tr_acc0 = 0 end
   -- if tr_acc1 ~= tr_acc1 then tr_acc1 = 0 end
   trainLogger:add{['% mean class accuracy (train set)'] = confusion.totalValid * 100}

end


epoch = 1
-- training loop
while true do
  -- train/test
  train()

  if epoch % opt.saveFreq == 0 then
    local filename = paths.concat(opt.save, string.format('adversarial_%d.net',epoch))
    os.execute('mkdir -p ' .. sys.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, {D = sanitize(model_D), G = sanitize(model_G), opt = opt})
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
