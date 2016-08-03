require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
-- require 'datasets.scaled_3d_test'
require 'pl'
require 'paths'

ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end

opt = lapp[[
  -s,--save          (default "logs")      subdirectory to save logs
  --saveFreq         (default 10)          save every saveFreq epochs
  -n,--network       (default "")          reload pretrained network
  -p,--plot                                plot while training
  -r,--learningRate  (default 0.02)        learning rate
  -b,--batchSize     (default 100)         batch size
  -m,--momentum      (default 0)           momentum, for SGD only
  --coefL1           (default 0)           L1 penalty on the weights
  --coefL2           (default 0)           L2 penalty on the weights
  -t,--threads       (default 4)           number of threads
  -g,--gpu           (default 0)           gpu to run on (default cpu)
  -d,--noiseDim      (default 100)         dimensionality of noise vector
  --K                (default 1)           number of iterations to optimize D for
  -w, --window       (default 3)           windsow id of sample image
  --scale            (default 128)          scale of images to train on
  --high             (default 1)           high resolution 
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

-- fix seed
torch.manualSeed(torch.random(1,10000))

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

model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/joint_Style_GAN.net') 
model_G = model.G
model_G = model_G:cuda()

opt.noiseDim = {100, 1, 1}
opt.geometry = {3, opt.scale, opt.scale}
opt.condDim = {3, opt.scale, opt.scale}
opt.div_num = 127.5
opt.loadSize = opt.scale 

paths.dofile('donkey.lua')



-- Get examples to plot
function getSamples(dataset, N, beg)

  local resultpath = '/nfs/hn38/users/xiaolonw/dcgan/test_style_gan2/' 
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local cond_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local gt_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])

  -- Generate samples
  noise_inputs:uniform(-1, 1)
  batch_data = makeData_joint(trainLoader:get(beg + 1, beg + N))

  gt_inputs:copy(batch_data[1])
  cond_inputs:copy(batch_data[3])

  local samples = model_G:forward({noise_inputs:cuda(), cond_inputs:cuda()})

  for i=1,N do

      output_name = paths.concat(resultpath, string.format('%04d_pred.jpg',i + beg))
      norm_name = paths.concat(resultpath, string.format('%04d_norm.jpg',i + beg))
      gt_name = paths.concat(resultpath, string.format('%04d_ori.jpg',i + beg))

      samples[i] = (samples[i] + 1 ) * opt.div_num
      cond_inputs[i] = (cond_inputs[i] + 1 ) * opt.div_num
      gt_inputs[i] = (gt_inputs[i] + 1 ) * opt.div_num

      output_img = samples[i]:clone()
      norm_img = cond_inputs[i]:clone()
      gt_img = gt_inputs[i]:clone()

      output_img = output_img:byte():clone()
      norm_img = norm_img:byte():clone()
      gt_img = gt_img:byte():clone()

      image.save(output_name, output_img )
      image.save(norm_name, norm_img)
      image.save(gt_name, gt_img)


  end



end



for i = 1,10 do 
  getSamples(trainData, 10, (i - 1) * 10 )
end






