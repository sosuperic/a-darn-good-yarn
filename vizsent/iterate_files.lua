require 'nngraph'
require 'image'
-- require 'gfx.js'

SENTIBANK_PATH = '/Users/eric/projects/vizsent/Sentibank/'

files = {}

opt = {}
opt.ext = 'jpg'
opt.dir = '/Users/eric/projects/vizsent/Sentibank/Flickr/bi_concepts1553/abandoned_asylum'

-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(opt.dir) do
   -- We only load files that match the extension
   if file:find(opt.ext .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(opt.dir,file))
   end
end

-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ' .. opt.ext)
end

----------------------------------------------------------------------
-- 3. Sort file names

-- We sort files alphabetically, it's quite simple with table.sort()

table.sort(files, function (a,b) return a < b end)

print('Found files:')
print(files)

----------------------------------------------------------------------
-- 4. Finally we load images

-- Go over the file list:
images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
end

print('Loaded images:')
-- print(images)
-- 
-- Display a of few them
-- for i = 1,math.min(#files,2) do
--    image.display{image=images[i], legend=files[i]}
-- end

----------------------------------------------------------------------
-- Resize images to 256 x 256 (center crop if necessary)