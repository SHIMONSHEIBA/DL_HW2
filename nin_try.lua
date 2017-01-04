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
      local flip_mask = torch.randperm(bs)
      for i=1, bs do
       	--if (flip_mask[i] % 4 == 0) then image.hflip(input[i],input[i]) end
	if (flip_mask[i] % 3 == 1) then image.vflip(input[i],input[i]) end
	--if (flip_mask[i] % 3 == 1) then image.vflip(input[i],input[i]) end
	--if (flip_mask[i] % 6 == 2) then image.RandomCrop(input[i],input[i],tl,32,32) end
	--if (flip_mask[i] % 4 == 2) then image.rotate(input[i],input[i],1.57079633) end
	--if (flip_mask[i] % 6 == 4) then image.minmax(input[i]) end
    end
    end
    self.output:set(input:cuda())
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

--local function Block(...)
  --local arg = {...}
  --model:add(cudnn.SpatialConvolution(...))
  --model:add(cudnn.SpatialBatchNormalization(arg[2]))--,1e-3))
  --model:add(nn.ReLU(true))
  --return model
--end
model:add(nn.BatchFlip():float())
model:add(cudnn.SpatialConvolution(3,32,5,5,1,1,2,2))
model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(32,32,1,1))---doesnt do anything to the dimensions
model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
model:add(nn.ReLU(true))
--model:add(cudnn.SpatialConvolution(32,32,1,1))---doesnt do anything to the dimensions
--model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
--model:add(nn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
--model:add(nn.Dropout())
model:add(cudnn.SpatialConvolution(32,32,5,5,1,1,2,2))
model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(32,32,1,1))---doesnt do anything to the dimensions
model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.2))
--model:add(cudnn.SpatialConvolution(32,32,1,1))---doesnt do anything to the dimensions
--model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
--model:add(nn.ReLU(true))
--model:add(cudnn.SpatialAveragePooling(3,3,2,2):ceil())
--model:add(nn.Dropout())
model:add(cudnn.SpatialConvolution(32,64,3,3,1,1,1,1))
model:add(cudnn.SpatialBatchNormalization(64))--,1e-3))
model:add(nn.ReLU(true))
--model:add(cudnn.SpatialConvolution(32,64,1,1))---doesnt do anything to the 2 dimensions
--model:add(cudnn.SpatialBatchNormalization(64))--,1e-3))
--model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(64,#classes,1,1))
model:add(cudnn.SpatialBatchNormalization(#classes))--,1e-3))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(#classes))

--for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do
  --v.weight:normal(0,0.05)
  --v.bias:zero()
--end

model:cuda()
--criterion = nn.ClassNLLCriterion():cuda()
--criterionName = CrossEntropyCriterion
criterion = nn.CrossEntropyCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

local f = assert(io.open('logFileVSGD.log', 'w'), 'Failed to open input file')
 --print('open the file')
   --f:write('The model is: ')
--print('start print to the log')
   --f:write(model)
   f:write('Number of parameters: ')
   f:write(w:nElement())
   f:write('\n The criterion is: CrossEntropyCriterion')
   --f:write(criterionName)
   f:write('\n optim function: ')
   f:write('sgd\n')



function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 32
f:write('batchSize: ')
f:write(batchSize)
f:write('\n')
f:close()
local optimState = {
 learningRate = 1,
 momentum =  0.9,
 weightDecay =  0.0005
}

function forwardNet(data,labels, train)
    --another helpful function of optim is ConfusionMatrix
	--print('check0')
    local confusion = optim.ConfusionMatrix(classes)
	--print('check1')
    local lossAcc = 0
	--print('check2')
    local numBatches = 0
	--print('check3')
    if train then
        --set network into training mode
        model:training()
    else
		--print('evaluate')
		--io.read()
        model:evaluate() -- turn of drop-out
    end
	--print('check4')
    for i = 1, data:size(1) - batchSize, batchSize do
	--print('check5')
        numBatches = numBatches + 1
	--print('check6')
        local x = data:narrow(1, i, batchSize)--:cuda()
	--print('check7')
        local yt = labels:narrow(1, i, batchSize)--:cuda()
	--print('check8')
        local y = model:forward(x)
	--print('check9')
	--print(yt:min())
	--print(yt:max())
	--print(y:min())
	--print(y:max())
	--print(x:size(1))
	--print(y:size(1))
	--print(yt:size(1))
	--print(x:size(2))
	--print(y:size(2))
        local err = criterion:forward(y, yt)
	--print('check10')
        lossAcc = lossAcc + err
	--print('check11')
	--print('y size: '.. y:size()..'.')
	--print('labels size: '.. yt:size()..'.')
        confusion:batchAdd(y,yt)
	--print('check12')
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
	--print('check13')
            optim.sgd(feval, w, optimState)
        end
    end
   -- print('check14')
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

epochs = 1000
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)

--reset net weights
model:apply(function(l) l:reset() end)

timer = torch.Timer()

for e = 1, epochs do
    print('start epoc ' .. e .. ':')
	
    if e % 25 == 0 then optimState.learningRate = optimState.learningRate/2 end
	
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
    
    if e % 5 == 0 then
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
        print(confusion)
   else
	 
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
   end
   
   if e == 1 then
      bestError = testError[e]
   end

local WritetrainError = trainError[e]
local WritetrainLoss = trainLoss[e] 
local WritetestError = testError[e]
local WritetestLoss = testLoss[e]
local f = assert(io.open('logFileVSGD.log', 'a+'), 'Failed to open input file')
   if e > 1 then
	print('test Error: ')
	print(testError[e])
	print('\nbest Error: ')
        print(bestError)
	if (testError[e] < bestError) then
	    bestError = testError[e]
	    print('save the model')
	    torch.save('ConvClassifierModelVSGD.t7', model)
	        --f = assert(io.open('logFile.log', 'r'), 'Failed to open input file')
	    f:write('Epoc ' .. e .. ': \n')
	    WritetrainError = trainError[e]
	    WritetrainLoss = trainLoss[e] 
	    WritetestError = testError[e]
	    WritetestLoss = testLoss[e]
	    f:write('Training error: ' .. WritetrainError ..  ' Training Loss: ' .. WritetrainLoss .. '\n')
	    f:write('Test error: ' .. WritetestError .. ' Test Loss: ' .. WritetestLoss ..'\n')
	end
    else
       print('save the model')
       torch.save('ConvClassifierVSGD.t7', model)
       f:write('Epoc ' .. e .. ': \n')
       WritetrainError = trainError[e]
       WritetrainLoss = trainLoss[e] 
       WritetestError = testError[e]
       WritetestLoss = testLoss[e]
       f:write('Training error: ' .. WritetrainError ..  ' Training Loss: ' .. WritetrainLoss .. '\n')
       f:write('Test error: ' .. WritetestError .. ' Test Loss: ' .. WritetestLoss ..'\n')
    end	
    f:close()
end



plotError(trainError, testError, 'Classification Error')

require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('lossVSGD.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('errorVSGDpng')
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()


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
