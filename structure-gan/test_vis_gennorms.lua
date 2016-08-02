require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'datasets.scaled_3d_test'
require 'pl'
require 'paths'

image_utils = require 'utils.image'
ok, disp = pcall(require, 'display')
if not ok then print('display not found. unable to plot') end
adversarial = require 'train.conditional_adversarial_ct'
require 'layers.SpatialConvolutionUpsample'

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
  --hidden_G         (default 8000)        number of units in hidden layers of G
  --hidden_D         (default 1600)        number of units in hidden layers of D
  --scale            (default 72)          scale of images to train on
]]

if opt.gpu < 0 or opt.gpu > 8 then opt.gpu = false end
print(opt)

-- fix seed
torch.manualSeed(torch.random(1,10000))

-- torch.manualSeed(1)

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


-- model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal3/save/adversarial_15.net') 
-- model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal3_64/save/adversarial_18.net') 
-- model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_64_2/adversarial_9.net') 
model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_72/save2/Structure_GAN.net') 
-- model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal2/save/adversarial_25.net') 
model_G = model.G

opt.noiseDim = {100, 1, 1}
opt.geometry = {3, opt.scale, opt.scale}
opt.condDim = {3, opt.scale, opt.scale}

opt.div_num = 127.5
opt.finescale = opt.scale * 2

-- trainData = normal3d.loadTrainSet(1, 100)
-- image_utils.simplenorm(trainData.data, opt.div_num, -1)
-- image_utils.simplenorm(trainData.labels, opt.div_num, -1)

-- normal3d.setScale(opt.scale)
-- trainData:scaleData()

-- Get examples to plot
function getSamples(dataset, N, beg)

  local resultpath = '/nfs/hn38/users/xiaolonw/dcgan/test_structure_gan/' 
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local cond_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local gt_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])

  -- Generate samples
  noise_inputs:uniform(-1, 1)

  local samples = model_G:forward(noise_inputs)
  sample_norm = torch.norm(samples, 2, 2)
  sample_norm = torch.cat({sample_norm, sample_norm, sample_norm}, 2)
  samples = torch.cdiv(samples, sample_norm)

  -- local to_plot = torch.FloatTensor(3, opt.scale * 10,opt.scale * 30)
  for i=1,N do
      output_name = paths.concat(resultpath, string.format('img_%04d.jpg',i + beg))

      samples[i] = (samples[i] + 1 ) * opt.div_num

      output_img = samples[i]:clone()
      output_img = output_img:byte():clone()
      image.save(output_name, output_img )
      -- image.save(norm_name, norm_img)
      -- image.save(gt_name, gt_img)

  end



end



for i = 1,10 do 
  getSamples(trainData, 10, (i - 1) * 10 )
end



-- torch.save('to_plot.t7', to_plot)

--disp.image(to_plot, {win=opt.window, width=700, title=opt.save})







