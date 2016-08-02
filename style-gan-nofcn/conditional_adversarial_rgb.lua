require 'torch'
require 'optim'
require 'pl'
require 'paths'

local adversarial = {}


function adam(opfunc, x, config, state)
    --print('ADAM')
    -- (0) get/update state
    local config = config or {}
    local state = state or config
    local lr = config.learningRate or 0.001

    local beta1 = config.beta1 or 0.9
    local beta2 = config.beta2 or 0.999
    local epsilon = config.epsilon or 1e-8

    -- (1) evaluate f(x) and df/dx
    local fx, dfdx = opfunc(x)
    if config.optimize == true then
      -- Initialization
      state.t = state.t or 0
      -- Exponential moving average of gradient values
      state.m = state.m or x.new(dfdx:size()):zero()
      -- Exponential moving average of squared gradient values
      state.v = state.v or x.new(dfdx:size()):zero()
      -- A tmp tensor to hold the sqrt(v) + epsilon
      state.denom = state.denom or x.new(dfdx:size()):zero()

      state.t = state.t + 1
      
      -- Decay the first and second moment running average coefficient
      state.m:mul(beta1):add(1-beta1, dfdx)
      state.v:mul(beta2):addcmul(1-beta2, dfdx, dfdx)

      state.denom:copy(state.v):sqrt():add(epsilon)

      local biasCorrection1 = 1 - beta1^state.t
      local biasCorrection2 = 1 - beta2^state.t
      
    local fac = 1
    if config.numUpdates < 10 then
        fac = 50.0
    elseif config.numUpdates < 30 then
        fac = 5.0
    else 
        fac = 1.0
    end
    io.write(" ", lr/fac, " ")
        local stepSize = (lr/fac) * math.sqrt(biasCorrection2)/biasCorrection1
      -- (2) update x
      x:addcdiv(-stepSize, state.m, state.denom)
    end
    config.numUpdates = config.numUpdates +1
    -- return x*, f(x) before optimization
    return x, {fx}
end


local inputs = torch.Tensor(opt.batchSize, opt.geometry[1], opt.geometry[2], opt.geometry[3])
local targets = torch.Tensor(opt.batchSize)
local noise_inputs 
if type(opt.noiseDim) == 'number' then
  noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim, 1, 1)
else
  noise_inputs = torch.Tensor(opt.batchSize, opt.noiseDim[1], opt.noiseDim[2], opt.noiseDim[3])
end
local cond_inputs 
if type(opt.condDim) == 'number' then
  cond_inputs = torch.Tensor(opt.batchSize, opt.condDim, 1, 1)
else
  cond_inputs = torch.Tensor(opt.batchSize, opt.condDim[1], opt.condDim[2], opt.condDim[3])
end

local sampleTimer = torch.Timer()
local dataTimer = torch.Timer()

-- training function
function adversarial.train(inputs_all, inputs_all2)
  cutorch.synchronize()
  epoch = epoch or 1
  local dataLoadingTime = dataTimer:time().real; sampleTimer:reset(); -- timers
  local dataBatchSize = opt.batchSize / 2


  -- do one epoch
  -- for t = 1,N,dataBatchSize*opt.K do 

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of discriminator
    local fevalD = function(x)
      collectgarbage()
      if x ~= parameters_D then -- get new parameters
        parameters_D:copy(x)
      end

      gradParameters_D:zero() -- reset gradients

      --  forward pass
      local outputs = model_D:forward({inputs, cond_inputs})
      err_R = criterion:forward(outputs:narrow(1, 1, opt.batchSize / 2), targets:narrow(1, 1, opt.batchSize / 2))
      err_F = criterion:forward(outputs:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2), targets:narrow(1, (opt.batchSize / 2) + 1, opt.batchSize / 2))

      local margin = 0.3
      adamState_D.optimize = true
      adamState_G.optimize = true      
      if err_F < margin or err_R < margin then
         adamState_D.optimize = false
      end
      if err_F > (1.0-margin) or err_R > (1.0-margin) then
         adamState_G.optimize = false
      end
      if adamState_G.optimize == false and adamState_D.optimize == false then
         adamState_G.optimize = true 
         adamState_D.optimize = true
      end

      -- io.write("v1_3d| R:", err_R,"  F:", err_F, "  ")
      

      local f = criterion:forward(outputs, targets)

      -- backward pass 
      local df_do = criterion:backward(outputs, targets)
      model_D:backward({inputs, cond_inputs}, df_do)

      print(string.format('D: %f',f))

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_D,1)
        f = f + opt.coefL2 * norm(parameters_D,2)^2/2
        -- Gradients:
        gradParameters_D:add( sign(parameters_D):mul(opt.coefL1) + parameters_D:clone():mul(opt.coefL2) )
      end
      -- update confusion (add 1 since classes are binary)
      for i = 1,opt.batchSize do
        local c
        if outputs[i][1] > 0.5 then c = 2 else c = 1 end
        confusion:add(c, targets[i]+1)
      end

      return f,gradParameters_D
    end

    ----------------------------------------------------------------------
    -- create closure to evaluate f(X) and df/dX of generator 
    local fevalG = function(x)
      collectgarbage()
      if x ~= parameters_G then -- get new parameters
        parameters_G:copy(x)
      end
      
      gradParameters_G:zero() -- reset gradients
--      debugger.enter()

      -- forward pass
      local samples = model_G:forward({noise_inputs, cond_inputs})
      local outputs = model_D:forward({samples, cond_inputs})
      local f = criterion:forward(outputs, targets)

      print(string.format('G: %f',f))

      --  backward pass
      local df_samples = criterion:backward(outputs, targets)
      -- model_D:backward({samples, cond_inputs}, df_samples)
      -- do not update D
      model_D:updateGradInput({samples, cond_inputs}, df_samples)
      local df_do = model_D.gradInput[1]
      model_G:backward({noise_inputs, cond_inputs}, df_do)

      -- penalties (L1 and L2):
      if opt.coefL1 ~= 0 or opt.coefL2 ~= 0 then
        local norm,sign= torch.norm,torch.sign
        -- Loss:
        f = f + opt.coefL1 * norm(parameters_G,1)
        f = f + opt.coefL2 * norm(parameters_G,2)^2/2
        -- Gradients:
        gradParameters_G:add( sign(parameters_G):mul(opt.coefL1) + parameters_G:clone():mul(opt.coefL2) )
      end

      return f,gradParameters_G
    end

    ----------------------------------------------------------------------
    -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    -- Get half a minibatch of real, half fake

    inputs:copy(inputs_all[1])
    cond_inputs:copy(inputs_all[3])

    for k=1,opt.K do
      -- (1.1) Real data 
      local k = dataBatchSize + 1
      targets[{{1,dataBatchSize}}]:fill(1)
      -- (1.2) Sampled data
      noise_inputs:uniform(-1, 1)

      local samples = model_G:forward({noise_inputs[{{dataBatchSize+1,opt.batchSize}}], cond_inputs[{{dataBatchSize+1,opt.batchSize}}]})
      for i = 1, dataBatchSize do
        inputs[k] = samples[i]:clone()
        k = k + 1
      end
      targets[{{dataBatchSize+1,opt.batchSize}}]:fill(0)

      adam(fevalD, parameters_D, adamState_D)
    end -- end for K

    ----------------------------------------------------------------------
    -- (2) Update G network: maximize log(D(G(z)))
    noise_inputs:uniform(-1, 1)
    cond_inputs:copy(inputs_all2[3])
    targets:fill(1)
    adam(fevalG, parameters_G, adamState_G)
    batchNumber = batchNumber + 1
    cutorch.synchronize(); collectgarbage();
    print(('Epoch: [%d][%d/%d]\tTime %.3f DataTime %.3f'):format(epoch, batchNumber, opt.epochSize, sampleTimer:time().real, dataLoadingTime))
    dataTimer:reset()

end


return adversarial
