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

------------------------------------normlize to 0-255----------------------------------------------------------------------------------

for i=1,3 do -- over each image channel
    trainData[{ {}, {i}, {}, {}  }] = ((trainData[{ {}, {i}, {}, {}  }])/255)
end

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }] = ((testData[{ {}, {i}, {}, {}  }])/255)
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
      local flip_mask = torch.randperm(bs)
      for i=1, bs do
      	image.yuv2rgb(input[i],input[i])
				
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
       if (flip_mask[i] % 3 == 0) then image.hflip(input[i],input[i]) end
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

--local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}
--print(#redChannel)



--  ****************************************************************
--  Define our neural network
--  ****************************************************************

local model = nn.Sequential()

model:add(nn.BatchFlip():float())
model:add(cudnn.SpatialConvolution(3,32,5,5,1,1,2,2))
model:add(cudnn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(32,32,1,1))---doesnt do anything to the dimensions
model:add(cudnn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(cudnn.SpatialConvolution(32,32,5,5,1,1,2,2))
model:add(cudnn.SpatialBatchNormalization(32))--,1e-3))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(32,32,1,1))---doesnt do anything to the dimensions
model:add(cudnn.SpatialBatchNormalization(32))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.2))
model:add(cudnn.SpatialConvolution(32,64,3,3,1,1,1,1))
model:add(cudnn.SpatialBatchNormalization(64))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialConvolution(64,#classes,1,1))
model:add(cudnn.SpatialBatchNormalization(#classes))
model:add(nn.ReLU(true))
model:add(cudnn.SpatialAveragePooling(8,8,1,1):ceil())
model:add(nn.View(#classes))

model:cuda()
criterion = nn.CrossEntropyCriterion():cuda()


w, dE_dw = model:getParameters()
print('Number of parameters:', w:nElement())
print(model)

--Create a log file to save the results
local f = assert(io.open('logFileBestModel.log', 'w'), 'Failed to open input file')
   f:write('Number of parameters: ')
   f:write(w:nElement())
   f:write('\n The criterion is: CrossEntropyCriterion')
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
        local x = data:narrow(1, i, batchSize)
        local yt = labels:narrow(1, i, batchSize)
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
            optim.sgd(feval, w, optimState)
        end
    end
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
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

    --every 25 epochs decerase the learning rate
    if e % 25 == 0 then optimState.learningRate = optimState.learningRate/2 end
	
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(trainData, trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)    

	--print error and loss
        print('Epoch ' .. e .. ':')
        print('Training error: ' .. trainError[e], 'Training Loss: ' .. trainLoss[e])
        print('Test error: ' .. testError[e], 'Test Loss: ' .. testLoss[e])
    if e % 5 == 0 then
        print(confusion)
   end
   
   if e == 1 then
      bestError = testError[e]
   end

--write the error and the loss, and save the model when the test error decrease
local WritetrainError = trainError[e]
local WritetrainLoss = trainLoss[e] 
local WritetestError = testError[e]
local WritetestLoss = testLoss[e]
local f = assert(io.open('logFileBestModel.log', 'a+'), 'Failed to open input file')
   if e > 1 then
	--print('test Error: ')
	--print(testError[e])
	--print('\nbest Error: ')
        --print(bestError)
	if (testError[e] < bestError) then
	    bestError = testError[e]
	    print('Better error : save the model')
	    torch.save('ConvClassifierModeBestModel.t7', model)

	    f:write('Epoc ' .. e .. ': \n')
	    WritetrainError = trainError[e]
	    WritetrainLoss = trainLoss[e] 
	    WritetestError = testError[e]
	    WritetestLoss = testLoss[e]
	    f:write('Training error: ' .. WritetrainError ..  ' Training Loss: ' .. WritetrainLoss .. '\n')
	    f:write('Test error: ' .. WritetestError .. ' Test Loss: ' .. WritetestLoss ..'\n')
	end
    else
       print('Better error : save the model')
       torch.save('ConvClassifierBestModel.t7', model)
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


--  ****************************************************************
--  plots
--  ****************************************************************
function plotError(trainError, testError, title)
	require 'gnuplot'
	local range = torch.range(1, trainError:size(1))
	gnuplot.pngfigure('testVsTrainError.png')
	gnuplot.plot({'trainError',trainError},{'testError',testError})
	gnuplot.xlabel('epochs')
	gnuplot.ylabel('Error')
	gnuplot.plotflush()
end

plotError(trainError, testError, 'Classification Error')

require 'gnuplot'
local range = torch.range(1, epochs)
gnuplot.pngfigure('lossBestModel.png')
gnuplot.plot({'trainLoss',trainLoss},{'testLoss',testLoss})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Loss')
gnuplot.plotflush()

gnuplot.pngfigure('errorBestModel.png')
gnuplot.plot({'trainError',trainError},{'testError',testError})
gnuplot.xlabel('epochs')
gnuplot.ylabel('Error')
gnuplot.plotflush()
