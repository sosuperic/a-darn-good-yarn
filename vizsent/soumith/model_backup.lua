--require 'nngraph'
require 'cunn'
require 'image'

model = nn.Sequential()

-- Conv Layer 1: 96 kernels, 11 x 11 x 3, stride = 4
model:add( nn.SpatialConvolution(3,96,11,11,4,4) )
model:add( nn.Tanh() )
model:add( nn.SpatialMaxPooling(2,2,2,2) )
model:add( nn.SpatialContrastiveNormalization(96, image.gaussian(3)) )

-- (W - F)/ S + 1
vol = (227 - 11) / 4 + 1 -- conv
vol = math.floor(vol / 2) -- pool

-- Conv Layer 2: 256 kernels, 5 x 5, stride = 2
model:add( nn.SpatialConvolution(96,256,5,5,2,2) )
model:add( nn.Tanh() )
model:add( nn.SpatialMaxPooling(2,2,2,2) )
model:add( nn.SpatialContrastiveNormalization(256, image.gaussian(3)) )

vol = (vol - 5) / 2 + 1 -- conv
vol = math.floor(vol / 2) -- pool

-- 4 Fully Connected Layers: 512, 512, 24, 2
model:add( nn.Reshape(256 * vol * vol) )
-- model:add(nn.View(256*vol*vol))                    -- reshapes from a 3D tensor of 16x5x5 into 1D tensor of 16*5*5

model:add( nn.Linear(256 * vol * vol, 512) )
model:add( nn.Tanh() )
model:add( nn.Linear(512, 512) )
model:add( nn.Tanh() )
model:add( nn.Linear(512, 24) )
model:add( nn.Tanh() )

-- model:add(nn.LogSoftMax()) 
model:add( nn.Linear(24, 2) )
model:add( nn.LogSoftMax() )

model = model:cuda()

criterion = nn.ClassNLLCriterion()
--criterion = nn.MSECriterion()
criterion = criterion:cuda()

collectgarbage()
