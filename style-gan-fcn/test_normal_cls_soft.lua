require 'torch'
require 'nngraph'
require 'cunn'
require 'optim'
require 'image'
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
  --hidden_G         (default 8000)        number of units in hidden layers of G
  --hidden_D         (default 1600)        number of units in hidden layers of D
  --scale            (default 512)          scale of images to train on
  --classnum         (default 40)    
  --classification   (default 1)
  --high             (default 1)
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



model = torch.load('/nfs.yoda/xiaolonw/torch_projects/models/train_3dnormal_fcn_cls/FCN.net')
model_FCN = model.FCN
model_FCN = model_FCN:cuda()

model2 = nn.Sequential()
model2:add(nn.SoftMax())
model2:cuda()


opt.noiseDim = {100, 1, 1}
opt.geometry = {3, opt.scale, opt.scale}
opt.condDim = {3, opt.scale, opt.scale}

opt.div_num = 127.5
opt.finescale = opt.scale * 2
opt.datasize = 654


local list_file = '/nfs/hn46/xiaolonw/cnncode/viewer/testLabels.txt'
local path_dataset = '/nfs/hn46/xiaolonw/cnncode/viewer/croptest/'

local f = assert(io.open(list_file, "r"))

function loadImage(path)
   -- print(path) 
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.scale, opt.scale)
   -- find the smaller dimension, and resize it to loadSize[2] (while keeping aspect ratio)
   input = input * 255
   return input
end


local codebooktxt = 'codebook_40.txt' 
local codebook = torch.Tensor(40,3)
if type(opt.classification) == 'number' and opt.classification == 1 then 
  local fcode = torch.DiskFile(codebooktxt, 'r')
  for i = 1, 40 do 
    for j = 1, 3 do 
      codebook[{{i},{j}}] = fcode:readFloat()
    end
  end
  fcode:close()

end



-- Get examples to plot
function getSamples(dataset, N, beg)

  local resultpath = '/nfs/hn38/users/xiaolonw/dcgan/ssgan_fcn/'
  local N = N or 8
  local gt_inputs = torch.Tensor(N, opt.condDim[1], opt.condDim[2], opt.condDim[3])
  local namelist = {}

  for n = 1,N do
    if n + beg > opt.datasize then 
      break
    end
    filename = f:read("*line")
    table.insert(namelist, filename) 
    filename = path_dataset .. filename

    local sample = loadImage(filename)
    gt_inputs[n] = sample:clone()
  end   

  gt_inputs:div(opt.div_num)
  gt_inputs:add(-1)

  model_FCN:forward(gt_inputs)
  local modellist = model_FCN:listModules()
  local layernum = #modellist
  local lastid   = layernum - 3

  local networkoutput = modellist[lastid].output 
  local samples = model2:forward(networkoutput)

  print(samples:size())
  local norms = torch.Tensor((#samples)[1], 3, (#samples)[3], (#samples)[4])

  for i = 1, (#samples)[1] do 
    for h = 1, (#samples)[3] do
      for w = 1, (#samples)[4] do 
        local nowpixel = torch.reshape(samples[{{i}, {}, {h}, {w}}], opt.classnum)
        local tnorm = torch.Tensor(3):fill(0)
        for c = 1, opt.classnum do
           tnorm = tnorm + codebook[c] * nowpixel[c]
        end
        norms[{{i},{},{h},{w}}] = torch.reshape(tnorm:clone(), 3)

      end
    end
  end


  samples = norms:clone()
  sample_norm = torch.norm(samples, 2, 2)
  sample_norm = torch.cat({sample_norm, sample_norm, sample_norm}, 2)
  samples = torch.cdiv(samples, sample_norm)


  -- local to_plot = torch.FloatTensor(3, opt.scale * 10,opt.scale * 30)
  for i=1,N do

      if i + beg > opt.datasize then 
        break
      end
    
      output_name = paths.concat(resultpath, namelist[i])
      gt_name = paths.concat(resultpath, string.format('%04d.jpg',i + beg))
      txt_name = paths.concat(resultpath, namelist[i]..'.txt')
      file = torch.DiskFile(txt_name, "w")
      nowsample = samples[i]:clone()

      nownorm = torch.totable(samples[i]:float())
      for c = 1, (#nowsample)[1] do 
        for w = 1, (#nowsample)[2] do 
          for h = 1, (#nowsample)[3] do 
            file:writeFloat( nownorm[c][h][w]) 
          end
        end
      end
      file:close()

      -- change scale 

      samples[i] = (samples[i] + 1 ) * opt.div_num
      gt_inputs[i] = (gt_inputs[i] + 1 ) * opt.div_num

      output_img = samples[i]:clone()
      -- output_img = image.scale(output_img, 128, 128)
      gt_img = gt_inputs[i]:clone()
      output_img = output_img:byte():clone()
      gt_img = gt_img:byte():clone()

      image.save(output_name, output_img )
      image.save(gt_name, gt_img)

  end



end



for i = 1,66 do 
  print(i)
  getSamples(trainData, 10, (i - 1) * 10 )
end

f:close()






