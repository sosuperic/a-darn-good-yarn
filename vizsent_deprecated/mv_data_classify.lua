-- Used to copy imgs from original folders biconcepts
-- To structure required for soumith 

SENTIBANK_PATH = 'Sentibank/'



function file.copy(src, dest)
  local content = file.read(src)
  file.write(dest, content)
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

function move_data(data_path, src_path, dest_path)
    for line in io.lines(data_path) do
        local tmp = str_split(line, '-')
        local phrase = tmp[1]
        local file = tmp[2]
        if (SENTS[phrase]) then
            local sentiment = tonumber(SENTS[phrase])
            local label = ''
            if (sentiment > 0.5) then
                label = 'positive'
            elseif (sentiment < -0.5) then
                label = 'negative'
            else
                label = 'neutral'
            end
            os.execute(string.format('cp "%s" "%s"',
                src_path .. phrase .. '/' .. file .. '.jpg',
                dest_path .. label .. '/' .. line .. '.jpg'
                ))
        end
    end
end

SENTS = get_phrase_sentiment_values()
move_data('tr.txt', 'Sentibank/Flickr/bi_concepts1553/', 'Sentibank/Flickr/data_split/train/')
move_data('va.txt', 'Sentibank/Flickr/bi_concepts1553/', 'Sentibank/Flickr/data_split/valid/')