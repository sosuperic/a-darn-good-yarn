-- Some images were pngs saved as jpg
-- Detect by going through original bi_concepts and printing name

-- require 'pl'
require 'image'

SENTIBANK_PATH = 'Sentibank/'

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


function detect_bad()
    dirs = scandir(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/', '') -- directories
    for idx, phrase in pairs(dirs) do
        imgs = scandir(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase, 'jpg')
        for img_idx, img_file in pairs(imgs) do
            -- local tmp = image.load(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase .. '/' .. img_file)
            if pcall(function() 
                local tmp = image.load(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase .. '/' .. img_file, 3, 'float')
            end) then
                local tmp = 2
            else
                print('BADBADBAD' .. img_file)
            end
        end
    end
end

function detect_bad_lsm()
    dirs = scandir(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/', '') -- directories
    for idx, phrase in pairs(dirs) do
        imgs = scandir(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase, 'jpg')
        for img_idx, img_file in pairs(imgs) do
            -- local tmp = image.load(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase .. '/' .. img_file)
            if pcall(function() 
                local tmp = image.load(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase .. '/' .. img_file, 3, 'float')
            end) then
                local tmp = 2
            else
                print('BADBADBAD' .. img_file)
            end
        end
    end
end

detect_bad()

-- data = split_train_valid_test()
-- write_data_split(data.tr, 'tr.txt')
-- -- write_data_split(data.va, 'va.txt')
-- write_data_split(data.va, 'va.txt')