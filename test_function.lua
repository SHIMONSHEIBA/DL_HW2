require 'optim'
require 'opt'
if opt.gpu > 0 then
  require 'cunn'
  require 'cudnn'
  require 'nn'
end
require 'torch'
require 'image'


model = torch.load('ConvClassifierModel8_2.t7')

trainset = torch.load('cifar.torch/cifar10-train.t7')
testset = torch.load('cifar.torch/cifar10-test.t7')

classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
trainLabels = trainset.label:float():add(1)
testData = testset.data:float()
testLabels = testset.label:float():add(1)

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
       	if (flip_mask[i] % 3 == 0) then image.hflip(input[i],input[i]) end
    end
    end
    self.output:set(input:cuda())
    return self.output
  end
end

--normalizing our data
local redChannel = trainData[{ {}, {1}, {}, {}  }] -- this picks {all images, 1st channel, all vertical pixels, all horizontal pixels}

local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future

for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
end

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

function TestModel()
	
	local confusion = optim.ConfusionMatrix(classes)
	local lossAcc = 0
	local numBatches = 0
	local batchSize = 32
	
	criterion = nn.CrossEntropyCriterion():cuda()
	
	model:evaluate()
	
    for i = 1, testData:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = testData:narrow(1, i, batchSize)
        local yt = testLabels:narrow(1, i, batchSize)
	print('start model')
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    return avgError
end


testError = TestModel()
print('The test error is: ' .. testError)
print('The accuracy of the model is: ' .. (1-testError)*100 .. '%')
