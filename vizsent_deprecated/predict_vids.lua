require 'image'
require 'cunn'

function resize_center_crop(im, n)
    -- Resize and center crop to n x n
    im = image.scale(im, '^' .. n) -- Minimum h or w is n
    im = image.crop(im, 'c', n, n) -- Center crop
    return im
end

-- Get training, validation, test data
function scandir(directory, ext)
    i, t, popen = 0, {}, io.popen
    for filename in popen('ls "'..directory..'"'):lines() do
    -- for filename in popen('ls -a "'..directory..'"'):lines() do
        if (string.find(filename, ext)) then 
        -- if (not string.find(filename, '%.')) and (string.find(filename, ext)) then
            i = i + 1
            t[i] = filename
        end
    end
    -- t = table.sort(t, function (a,b) return a < b end)
    return t
end

function write(path, data, sep)
    sep = sep or ','
    local file = assert(io.open(path, "w"))
    for i=1,#data do
    	for j=1,2 do
        -- for j=1,#data[i] do
            if j>1 then file:write(sep) end
            file:write(data[i][j])
        end
        file:write('\n')
    end
    file:close()
end


FRAMES_PATH = '/home/lsm/echu/vids/ForrestGump/frames'
frame_files = scandir(FRAMES_PATH, 'jpg')
table.sort(frame_files, function (a,b) return a < b end)

MODEL_PATH = '/home/lsm/echu/data_split_trained/alexnet12,cropSize=227,nClasses=2,nEpochs=20/,SatDec2604:38:212015/model_9.t7'
model = torch.load(MODEL_PATH)

frame_preds = {}
for idx, file in pairs(frame_files) do
	print(FRAMES_PATH)
	print(file)
	im = image.load(FRAMES_PATH .. file)
	im = resize_center_crop(im, 227)
	pred = model:forward(resize_center_crop(im, 227):cuda()):exp()
	frame_preds[idx] = pred
end

CSV_PATH = '/home/lsm/echu/vids/ForrestGump/'
write(CSV_PATH, frame_preds, ',')
