require 'nngraph'
require 'image'
ffi = require 'ffi'

ffi.cdef[[
typedef struct { uint8_t red, green, blue, alpha; } rgba_pixel;
]]
-- require 'itorch'
-- require 'iterate_files'

SENTIBANK_PATH = 'Sentibank/'

function str_split(inputstr, sep)
    if sep == nil then
            sep = "%s"
    end
    local t={} ; i=1
    for str in string.gmatch(inputstr, "([^"..sep.."]+)") do
            t[i] = str
            i = i + 1
    end
    return t
end

function get_phrase_sentiment_values()
    local sents = {}
    -- sent_file = io.open(SENTIBANK_PATH .. 'VSO/3244ANPs.txt', 'r')
    for line in io.lines(SENTIBANK_PATH .. 'VSO/3244ANPs.txt') do
        -- '  dark_night [sentiment: -1.41] [#imgs: 1,660,000]'
        if string.find(line, '_') then
            -- Get phrase
            local phrase = str_split(line, '%s+')[1]

            -- Get sentiment value
            local tmp = str_split(line, ':')
            tmp = str_split(tmp[2], ']')
            local value = string.gsub(tmp[1], "%s+", "")
            value = tonumber(value)

            sents[phrase] = value
        end
    end
    return sents -- key = phrase
end

function resize_center_crop(im, n)
    -- Resize and center crop to n x n
    im = image.scale(im, '^' .. n) -- Minimum h or w is n
    im = image.crop(im, 'c', n, n) -- Center crop
    return im
end

-- Get training, validation, test data
function read_data_split(path)
    local data = {}
    p = 0
    for line in io.lines(path) do
        table.insert(data, line)
        p = p + 1
    end
    -- print(data[1])
    return data
end

function prepare_split(split)
    local prepared = {}
    local d = 0
    -- print(d)
    -- local im = ffi.new("rgba_pixel[?]", )
    local tmp = ''; phrase = ''; file = '';
    for idx, datum in ipairs(split) do
        tmp = str_split(datum, '-')
        -- phrase = str_split(datum, '-')[1]
        -- file = str_split(datum, '-')[2]
        phrase = tmp[1]
        file = tmp[2]
        print(d)
        -- print(d .. ' ' .. phrase .. ' ' .. file)

        -- for phrase, filename in pairs(split) do
        -- if pcall(function() -- pcall returns true if no error. Some were PNG's resaved as jpg's, throwing error
            -- print(phrase ..'/' .. filename)
        -- end 

        -- If we have a sentiment value, and if dimensions are correct
        -- Dimensions could be incorrect if image passed to resize_center_crop were smaller than 227
        -- if (SENTS[phrase]) and (im:size()[2] == 227) and (im:size()[3] == 227) then
        if (SENTS[phrase]) then 
            local im = image.load(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase ..'/' .. file .. '.jpg')
            table.insert(prepared,
                {
                    resize_center_crop(im, 227),
                    torch.Tensor({SENTS[phrase]})
                })
        end
        -- end 
        d = d + 1
        if d > 50 then
            break
        end
        -- print(d)
        -- print('what2')
    end
    print(d)
    print(table.getn(prepared))
    return prepared
end


-------------------------------------------------------
-- Model
-------------------------------------------------------

SENTS = get_phrase_sentiment_values()
-- require 'pl.pretty'.dump(SENTS)

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
model:add( nn.Linear(24, 3) )
model:add( nn.LogSoftMax())

criterion = nn.ClassNLLCriterion()
-- criterion = nn.MSECriterion()

trainer = nn.StochasticGradient(model, criterion)
trainer.learningRate = 0.01
trainer.maxIteration = 5 -- just do 5 epochs of training.

dim = 227
require 'iterate_files'
tr = {
        {
            image.crop(images[1], 'c', dim, dim),
            torch.Tensor({1})
        },
        {
            image.crop(images[2], 'c', dim, dim),
            torch.Tensor({2})
        },
        {
            image.crop(images[3], 'c', dim, dim),
            torch.Tensor({3})
        }
}
print(trainset)
-- setmetatable(trainset, 
--     {__index = function(t, i) 
--                     return {t.data[i], t.label[i]} 
--                 end}
-- );
-- -- trainset.data = trainset.data:double() -- convert the data from a ByteTensor to a DoubleTensor.
-- tr = read_data_split('te.txt')
-- tr = prepare_split(tr)
print(table.getn(tr))
-- tr = prepare_split(tr)

-- for phrase, filename in pairs(read_data_split(SENTIBANK_PATH .. 'Flickr/te.txt')) do
--     im = image.load(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase ..'/' .. filename .. '.jpg')
--     im = resize_center_crop(im, 227)
--     -- if (SENTS[phrase]) then 
--     --     table.insert(prepared,
--     --         {
--     --             im,
--     --             torch.Tensor({SENTS[phrase])
--     --         })
--     -- end
--     print(im:size())
--     break
-- end
-- va = read_data_split(SENTIBANK_PATH + 'Flickr/va.txt')
-- te = read_data_split(SENTIBANK_PATH + 'Flickr/te.txt')
function tr:size() 
    return table.getn(self)
end

print('time to train')
trainer:train(tr)

predicted = model:forward(tr[1][1]); print(predicted:exp())
predicted = model:forward(tr[2][1]); print(predicted:exp())


-- torch.save('trained.bin', model)
-- Visualize
-- itorch.image(model:get(2).output)

