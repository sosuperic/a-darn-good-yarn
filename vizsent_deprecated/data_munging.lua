-- Create tr.txt, va.txt, te.txt. Files with pathnames
-- These are then used in mv_data_classify.lua, which creates the split
-- necessary for soumith

-- require 'pl'

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

function defaultdict(default)
    local tbl = {}
    local mtbl = {}
    mtbl.__index = function(tbl, key)
        local val = rawget(tbl, key)
        return val or default
    end
    setmetatable(tbl, mtbl)
    return tbl
end

function split_train_valid_test()
    tr, va, te = {}, {}, {} -- key = phrase, value = table of filenames
    dirs = scandir(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/', '') -- directories
    for idx, phrase in pairs(dirs) do
        imgs = scandir(SENTIBANK_PATH .. 'Flickr/bi_concepts1553/' .. phrase, 'jpg')
        for img_idx, img_file in pairs(imgs) do
            print(phrase .. '-' .. img_file)
            img_file = str_split(img_file, '%.')[1]
            r = math.random()
            if r < 0.9 then
                if not tr[phrase] then tr[phrase] = {} end
                tr[phrase][#tr[phrase] + 1] = img_file
                -- table.insert(tr[phrase], img_file)
            -- elseif r < 0.9 then
            --     if not va[phrase] then va[phrase] = {} end 
            --     va[phrase][#va[phrase] + 1] = img_file
            --     -- table.insert(va[phrase], img_file)
            else
                if not va[phrase] then va[phrase] = {} end 
                va[phrase][#va[phrase] + 1] = img_file
                -- table.insert(te[phrase], img_file)
            end
        end
        -- break
    end
    result = {tr=tr, va=va, te=te}
    -- require 'pl.pretty'.dump(te)
    return result
end
function write_data_split(split, path)
    out_file = io.open(path, 'w')
    for phrase, files in pairs(split) do
        for idx, file in pairs(files) do     
            print(phrase .. '@' .. file)   
            -- io.write(phrase .. '-' .. file)
            -- io.write('\n')
            out_file:write(phrase .. '-' .. file)
            out_file:write('\n')
        end
    end
    out_file.close()
end

data = split_train_valid_test()
write_data_split(data.tr, 'tr.txt')
-- write_data_split(data.va, 'va.txt')
write_data_split(data.va, 'va.txt')