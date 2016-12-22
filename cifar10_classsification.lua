--[[
Due to interest of time, please prepared the data before-hand into a 4D torch
ByteTensor of size 50000x3x32x32 (training) and 10000x3x32x32 (testing) 
]]

require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

print(trainData:size())

do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip2:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip2:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs):le(bs/2)
      for i=1,input:size(1) do
        if flip_mask[i] == 1 then image.hflip(input[i], input[i]) end
      end
    end
    self.output:set(input)
    return self.output
  end
end

local function horizontal_reflection(x)
    return image.hflip(x)
end
-------------------added data augmantation----------------------------------
do -- data augmentation module
  local BatchFlip,parent = torch.class('nn.BatchFlip', 'nn.Module')

  function BatchFlip:__init()
    parent.__init(self)
    self.train = true
  end

  function BatchFlip:updateOutput(input)
    if self.train then
      local bs = input:size(1)
      local flip_mask = torch.randperm(bs)--:le(bs/2)
      for i=1,input:size(1) do
       	if (flip_mask[i] % 2 == 1) then horizontal_reflection(input[i]) end
	--if flip_mask[i] % 2 == 1 then image.vflip(input[i]) end
	--if flip_mask[i] % 2 == 1 then image.crop(input[i],tl,32,32) end
	--if flip_mask[i] % 2 == 1 then image.rotate(input[i],1.57079633) end
	--if flip_mask[i] % 2 == 1 then image.minmax(input[i],) end
      end
    end
    self.output:set(input)
    return self.output
  end
end
-----------------------------------------------------------------------------------

--  ****************************************************************
--  Full Example - Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:

local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
print(#redChannel)

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    print('Channel ' .. i .. ', Mean: ' .. mean[i])
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    print('Channel ' .. i .. ', Standard Deviation: ' .. stdv[i])
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end


--  ****************************************************************
--  Define our neural network
--  ****************************************************************

local model = nn.Sequential()
model:add(nn.BatchFlip2():float())
--model:add(cudnn.SpatialConvolution(3, 32, 5, 5)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(cudnn.SpatialConvolution(3, 32, 5, 5, 1, 1, 2, 2)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(cudnn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(nn.LeakyReLU(true))                          -- ReLU activation function
model:add(cudnn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
--model:add(nn.SpatialMaxPooling(2,2,2,2))      -- A max-pooling operation that looks at 2x2 windows and finds the max.
--model:add(cudnn.ReLU(true))                          -- ReLU activation function
model:add(cudnn.SpatialConvolution(32, 32, 5, 5, 1, 1, 2, 2))
model:add(cudnn.SpatialBatchNormalization(32))    --Batch normalization will provide quicker convergence
model:add(nn.LeakyReLU(true))                          -- ReLU activation function
--model:add(cudnn.SpatialConvolution(32, 64, 3, 3))
model:add(cudnn.SpatialMaxPooling(2,2,2,2))
--model:add(nn.Dropout(0.2)) 
--model:add(cudnn.SpatialConvolution(16, 16, 5, 5, 1, 1, 2, 2))
--model:add(cudnn.SpatialMaxPooling(2,2,2,2))
--model:add(cudnn.ReLU(true))
--model:add(cudnn.SpatialBatchNormalization(16))
--model:add(nn.LeakyReLU(true))
--model:add(cudnn.SpatialConvolution(64, 32, 3, 3))
model:add(cudnn.SpatialConvolution(32, 16, 5, 5, 2, 2, 2, 2))
model:add(nn.View(16*4*4):setNumInputDims(3))  -- reshapes from a 3D tensor of 32x4x4 into 1D tensor of 32*4*4
model:add(nn.Linear(16*4*4, 32))             -- fully connected layer (matrix multiplication between input and weights)
--model:add(cudnn.ReLU(true))
model:add(nn.LeakyReLU(true))
model:add(nn.Dropout(0.4))                      --Dropout layer with p=0.2
model:add(nn.Linear(32, #classes))            -- 10 is the number of outputs of the network (in this case, 10 digits)
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 16
local optimState = {}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end

function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

---------------------------------------------------------------------

epochs = 60
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
    end
end

plotError(trainError, testError, 'Classification Error')


--  ****************************************************************
--  Network predictions
--  ****************************************************************


--model:evaluate()   --turn off dropout
--
--print(classes[testLabels[10]])
--print(testData[10]:size())
--saveTensorAsGrid(testData[10],'testImg10.jpg')
--local predicted = model:forward(testData[10]:view(1,3,32,32):cuda())
--print(predicted:exp()) -- the output of the network is Log-Probabilities. To convert them to probabilities, you have to take e^x 
--
---- assigned a probability to each classes
--for i=1,predicted:size(2) do
--    print(classes[i],predicted[1][i])
--end



--  ****************************************************************
--  Visualizing Network Weights+Activations
--  ****************************************************************


--local Weights_1st_Layer = model:get(1).weight
--local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_1st_Layer,padding=2}),200)
--saveTensorAsGrid(scaledWeights,'Weights_1st_Layer.jpg')


--print('Input Image')
--saveTensorAsGrid(testData[100],'testImg100.jpg')
--model:forward(testData[100]:view(1,3,32,32):cuda())
--for l=1,9 do
--  print('Layer ' ,l, tostring(model:get(l)))
--  local layer_output = model:get(l).output[1]
--  saveTensorAsGrid(layer_output,'Layer'..l..'-'..tostring(model:get(l))..'.jpg')
--  if ( l == 5 or l == 9 )then
--	local Weights_lst_Layer = model:get(l).weight
	--local scaledWeights = image.scale(image.toDisplayTensor({input=Weights_lst_Layer[1],padding=2}),200)
	--saveTensorAsGrid(scaledWeights,'Weights_'..l..'st_Layer.jpg')
  --end 
--end
