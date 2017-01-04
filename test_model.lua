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
       if (flip_mask[i] % 3 == 0) then image.hflip(input[i],input[i]) end
	--if (flip_mask[i] % 3 == 1) then image.vflip(input[i],input[i]) end
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




model = torch.load('ConvClassifierModel8_2.t7')

trainset = torch.load('cifar.torch/cifar10-train.t7')
testset = torch.load('cifar.torch/cifar10-test.t7')
