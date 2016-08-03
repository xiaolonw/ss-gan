require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
require 'pl'
require 'paths'
require 'nnx'

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


-- model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_jointall_bi_s4/adversarial_G_9.net') 
-- model_G = model.G:cuda()
-- model_G1= model.G1:cuda()

model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_joint4/save/joint_Style_GAN.net')
model_G = model.G
model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/dcgan_normal_72/save2/Structure_GAN.net')
model_G1 = model.G



opt.noiseDim = {100, 1, 1}
opt.geometry = {3, opt.scale, opt.scale}
opt.condDim = {3, opt.scale, opt.scale}

opt.div_num = 127.5
opt.finescale = opt.scale * 2


model_upsample = nn.Sequential()
model_upsample:add(nn.SpatialReSampling({owidth=128,oheight=128}) )

model_upsample:add(nn.ReArrange())
model_upsample:add(nn.Normalize(2))
model_upsample:add(nn.ReArrangeBack(128, 128))


model_upsample:float()


-- Get examples to plot
function getSamples(dataset, N, beg)

  local resultpath = '/nfs/hn38/users/xiaolonw/dcgan/joint_all_results/' 
  local N = N or 8
  local noise_inputs = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local noise_inputs2 = torch.Tensor(N, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
  local cond_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local gt_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])

  -- Generate samples
  noise_inputs:uniform(-1, 1)
  noise_inputs2:uniform(-1, 1)

  local samples1 = model_G1:forward(noise_inputs)
  local samples = model_upsample:forward(samples1:float())
  local imgsamples = model_G:forward({noise_inputs2, samples:cuda()}) 


  -- local to_plot = torch.FloatTensor(3, opt.scale * 10,opt.scale * 30)
  for i=1,N do
      output_name = paths.concat(resultpath, string.format('%04d_norm.jpg',i + beg))
      output_imgname = paths.concat(resultpath, string.format('%04d_img.jpg',i + beg))

      samples[i] = (samples[i] + 1 ) * opt.div_num
      imgsamples[i] = (imgsamples[i] + 1 ) * opt.div_num

      output_norm = samples[i]:clone()
      output_norm = output_norm:byte():clone()
      image.save(output_name, output_norm )

      output_img = imgsamples[i]:clone()
      output_img = output_img:byte():clone()
      image.save(output_imgname, output_img )

  end



end



for i = 1,10 do 
  getSamples(trainData, 10, (i - 1) * 10 )
end






