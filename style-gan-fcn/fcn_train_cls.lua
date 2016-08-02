require 'torch'
require 'optim'
require 'pl'
require 'paths'

local fcn = {}

local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
-- local targets = torch.Tensor(opt.batchSize, opt.outDim[1], opt.outDim[2], opt.outDim[3])
local targets = torch.Tensor(opt.batchSize * opt.labelSize * opt.labelSize)


local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()


-- training function
function fcn.train(inputs_all)
  cutorch.synchronize()
  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize 

  local fevalFCN = function(x)
    collectgarbage()
    if x ~= parameters_FCN then -- get new parameters
      parameters_FCN:copy(x)
    end
    gradParameters_FCN:zero() -- reset gradients

    -- forward pass
    local outputs = model_FCN:forward(inputs)
    local f = criterion:forward(outputs, targets)
    print(string.format('FCN: %f',f))

    local df_samples = criterion:backward(outputs, targets)
    -- print(df_samples:type()) 
    model_FCN:backward(inputs, df_samples)
    return f,gradParameters_FCN
  end

  inputs:copy(inputs_all[1])
  targets:copy(inputs_all[3]) --:reshape(opt.batchSize, opt.outDim[1] * opt.outDim[2] * opt.outDim[3]))

  optim.sgd(fevalFCN, parameters_FCN, optimState)
  batchNumber = batchNumber + 1
  cutorch.synchronize(); collectgarbage();
  print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime))
  dataTimer:reset()

end


return fcn


