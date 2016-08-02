require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
image_utils = require 'image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end


----------------------------------------------------------------------
-- parse command-line options
opt = lapp[[
  -s,--save          (default "/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_72")      subdirectory to save logs
  --saveFreq         (default 3)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.0002)      learning rate
  -b,--batchSize     (default 64)         batch size
  -m,--momentum      (default 0.5)         momentum term of adam
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)     L2 penalty on the weights
  -t,--threads       (default 2)           number of threads
  -g,--gpu           (default 0)          gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 72)          scale of images to train on
  --epochSize        (default 2000)        number of samples per epoch
  --scratch          (default 0)
  --forceDonkeys     (default 0)
  --nDonkeys         (default 2)           number of data loading threads
  --high             (default 1)           high resolution 
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
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
-- opt.condDim = {3, opt.scale, opt.scale}


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

  local nplanes = 64
  local inputsize = 72
  dx_I = nn.Identity()()
  dh1 = nn.SpatialConvolution(3, nplanes, 5, 5, 2, 2, 2, 2)(dx_I)  -- size / 2 
  -- db1 = nn.SpatialBatchNormalization(nplanes)(dh1)
  dr1 = nn.LeakyReLU(0.2, true)(dh1)

  dh2 = nn.SpatialConvolution(nplanes, nplanes * 2, 5, 5, 1, 1, 2, 2)(dr1)
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

  local outputsize3 =  nplanes * 2 * (inputsize / 8) * (inputsize / 8) 
  rshp = nn.Reshape(outputsize3)(dr5)
  dh6 = nn.Linear(outputsize3, 1)(nn.Dropout()(rshp))
  dout = nn.Sigmoid()(dh6)
  model_D = nn.gModule({dx_I}, {dout})

  model_D:apply(weights_init)

  
  ----------------------------------------------------------------------
  -- define G network to train
  -- my G network
  local nplanes = 64
  x_I = nn.Identity()()  -- 1 channel noise

  hi1 = nn.SpatialFullConvolution(opt.noiseDim, nplanes * 2, 9, 9)(x_I) 
  bi1 = nn.SpatialBatchNormalization(nplanes * 2)(hi1)
  ri1 = nn.ReLU(true)(bi1)

  hi2 = nn.SpatialFullConvolution(nplanes * 2, nplanes * 2 , 4, 4, 2, 2, 1, 1)(ri1) -- size * 2 for noise 18 * 18 
  bi2 = nn.SpatialBatchNormalization(nplanes * 2)(hi2)
  ri2 = nn.ReLU(true)(bi2)

  hi3 = nn.SpatialConvolution(nplanes  * 2 , nplanes  * 2, 3, 3, 1, 1, 1, 1)(ri2) -- the same size for noise 36 * 36 
  bi3 = nn.SpatialBatchNormalization(nplanes * 2)(hi3)
  ri3 = nn.ReLU(true)(bi3)

  h3 = nn.SpatialConvolution(nplanes * 2, nplanes * 4, 3, 3, 1, 1, 1, 1)(ri3) -- the same size
  b3 = nn.SpatialBatchNormalization(nplanes * 4)(h3) 
  r3 = nn.ReLU(true)(b3)

  h4 = nn.SpatialConvolution(nplanes*4, nplanes * 8, 3, 3, 1, 1, 1, 1)(r3) -- the same size
  b4 = nn.SpatialBatchNormalization(nplanes * 8)(h4) 
  r4 = nn.ReLU(true)(b4)

  h5 = nn.SpatialConvolution(nplanes*8, nplanes * 8, 3, 3, 1, 1, 1, 1)(r4) -- same size
  b5 = nn.SpatialBatchNormalization(nplanes * 8)(h5) 
  r5 = nn.LeakyReLU(0.2, true)(b5)

  h6 = nn.SpatialFullConvolution(nplanes*8, nplanes * 4, 4, 4, 2, 2, 1, 1)(r5) -- size * 2
  b6 = nn.SpatialBatchNormalization(nplanes * 4)(h6) 
  r6 = nn.LeakyReLU(0.2, true)(b6)

  h7 = nn.SpatialConvolution(nplanes*4, nplanes * 2, 3, 3, 1, 1, 1, 1)(r6)  -- same size
  b7 = nn.SpatialBatchNormalization(nplanes * 2)(h7) 
  r7 = nn.LeakyReLU(0.2, true)(b7)


  h8 = nn.SpatialFullConvolution(nplanes*2, nplanes, 4, 4, 2, 2, 1, 1)(r7)  -- size * 2
  b8 = nn.SpatialBatchNormalization(nplanes)(h8)
  r8 = nn.LeakyReLU(0.2, true)(b8)

  h9 = nn.SpatialConvolution(nplanes, 3, 5, 5, 1, 1, 2, 2)(r8) -- same size


  tanhout = nn.Tanh()(h9)
  shff1 = nn.Transpose({2,3},{3,4})(tanhout)
  rshp  = nn.View(-1, 3)(shff1)
  normout = nn.Normalize(2)(rshp)
  rshp2 = nn.View(-1, 72, 72, 3)(normout) 
  gout = nn.Transpose({4,3},{3,2})(rshp2)

  -- old layer ReArrange implemented by wxl 
  -- rshp = nn.ReArrange()(tanhout)
  -- normout = nn.Normalize(2)(rshp)
  -- gout = nn.ReArrangeBack(72, 72)(normout) 

  model_G = nn.gModule({x_I} , {gout})

  model_G:apply(weights_init)


else
  print('<trainer> reloading previously trained network: ' .. opt.network)
  tmp = torch.load(opt.network)
  model_D = tmp.D
  model_G = tmp.G
end

-- loss function: negative log-likelihood
criterion = nn.BCECriterion()

-- retrieve parameters and gradients
parameters_D,gradParameters_D = model_D:getParameters()
parameters_G,gradParameters_G = model_G:getParameters()

-- print networks
print('Discriminator network:')
print(model_D)
print('Generator network:')
print(model_G)

paths.dofile('data.lua')
adversarial = paths.dofile('adversarial_ct.lua')


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
    torch.save(filename, {D = model_D:clearState(), G = model_G:clearState(), opt = opt})
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
