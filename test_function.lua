local mnist = require 'mnist';
require 'nn'
require 'cunn'
require 'optim'

model = torch.load('ClassifierModel.t7')

local trainData = mnist.traindataset().data:float();
testData = mnist.testdataset().data:float();
testLabels = mnist.testdataset().label:add(1);

--normalizing our data
local mean = trainData:mean()
local std = trainData:std()
trainData:add(-mean):div(std); 
testData:add(-mean):div(std);

function TestModel()
	
	local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())
	local lossAcc = 0
	local numBatches = 0
	local batchSize = 16
	
	criterion = nn.CrossEntropyCriterion():cuda()
	
	model:evaluate()
	
    for i = 1, testData:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = testData:narrow(1, i, batchSize):cuda()
        local yt = testLabels:narrow(1, i, batchSize):cuda()
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
