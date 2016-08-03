--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

require 'image'
require 'struct'
require 'image'
require 'string'

paths.dofile('dataset.lua')

-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
-------- COMMON CACHES and PATHS
-- a cache file of the training metadata (if doesnt exist, will be created)
local cache = "../cache"
os.execute('mkdir -p '..cache)
local trainCache = paths.concat(cache, 'trainCache_high.t7')
local testCache = paths.concat(cache, 'testCache.t7')
local meanstdCache = paths.concat(cache, 'meanstdCache.t7')

-- Check for existence of opt.data
opt.data = os.getenv('DATA_ROOT') or '../logs'
--------------------------------------------------------------------------------------------
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

local loadSize   = {3, opt.loadSize}
local sampleSize = {3, opt.loadSize}

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize)
   input = input * 255
   return input
end


local function loadLabel_high(path)
   local input = image.load(path, 3, 'float')
   input = image.scale(input, opt.loadSize, opt.loadSize )
   input = input * 255
   return input
end


local function loadLabel(path)

  local f = assert(io.open(path, "rb"))
  local data = f:read("*all")

  local height = opt.loadSize
  local width = opt.loadSize
  local nowidx = 1
  local trdata = torch.FloatTensor(3, height,width)
  local num 
  for k = 1,3 do 
    for i = 1,height do
      for j = 1,width do
        num, nowidx = struct.unpack("f", data, nowidx)
        trdata[{k,i,j}] = num
      end
    end
  end

  f:close()

  return trdata

end


local savepath = '/nfs.yoda/xiaolonw/torch_projects/t_imgs/'
function saveData(img, imgname)
  img = (img + 1 ) * 127.5
  img = img:byte()
  image.save(imgname, img )
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



function makeData(fine, label, classes)


   local tlabel = torch.Tensor(opt.batchSize, 3, opt.labelSize, opt.labelSize)
   local tlabel2 = torch.Tensor(opt.batchSize, 3, opt.labelSize2, opt.labelSize2)
   for i = 1, opt.batchSize do
    tlabel[i] = image.scale(label[i], opt.labelSize, opt.labelSize)
    tlabel2[i] = image.scale(label[i], opt.labelSize2, opt.labelSize2)
   end

   label = tlabel:clone()   
   local label2 = tlabel2:clone()
   sample_norm2 = torch.norm(label2, 2, 2)
   sample_norm2 = torch.cat({sample_norm2, sample_norm2, sample_norm2}, 2)
   label2 = torch.cdiv(label2, sample_norm2)


   if opt.flag == 0 then 
    for i = 1, opt.batchSize do 
      local label_name = paths.concat(savepath, string.format('%04d_label.jpg',i ))
      local img_name = paths.concat(savepath, string.format('%04d_img.jpg',i ))
      saveData(label[i]:clone(), label_name)
      saveData(fine[i]:clone(), img_name)
    end
   end
   opt.flag = 1

   return {fine, label2 ,label}
end




-- channel-wise mean and std. Calculate or load them from disk later in the script.
local div_num, sub_num
div_num = 127.5
sub_num = -1
--------------------------------------------------------------------------------
-- Hooks that are used for each image that is loaded

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, imgpath, lblpath)
   collectgarbage()
   local img = loadImage(imgpath)
   local label = loadLabel_high(lblpath)
   img:div(div_num)
   img:add(sub_num)

   label:div(div_num)
   label:add(sub_num)

   return img, label

end

--------------------------------------
-- trainLoader
if paths.filep(trainCache) then
   print('Loading train metadata from cache')
   trainLoader = torch.load(trainCache)
   trainLoader.sampleHookTrain = trainHook
   trainLoader.loadSize = {3, opt.loadSize, opt.loadSize}
   trainLoader.sampleSize = {3, sampleSize[2], sampleSize[2]}
else
   print('Creating train metadata')
   trainLoader = dataLoader{
      paths = {paths.concat(opt.data, 'train')},
      loadSize = {3, loadSize[2], loadSize[2]},
      sampleSize = {3, sampleSize[2], sampleSize[2]},
      split = 100,
      verbose = true
   }
   torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
end
collectgarbage()


