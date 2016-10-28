---
--- HELPER FUNCTIONS
---

function MaybePutOnGPU(obj, params)
  if params.gpu >= 0 then
    if params.backend ~= 'clnn' then
      return obj:cuda()
    else
      return obj:cl()
    end
  end
  return obj
end

-- Preprocess an image before passing it to a Caffe model.
-- We need to rescale from [0, 1] to [0, 255], convert from RGB to BGR,
-- and subtract the mean pixel.
function preprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):mul(256.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img
end

-- Undo the above preprocessing.
function deprocess(img)
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img = img + mean_pixel
  local perm = torch.LongTensor{3, 2, 1}
  img = img:index(1, perm):div(256.0)
  return img
end

function save_image(img, fileName)
  local disp = deprocess(img:double())
  disp = image.minmax{tensor=disp, min=0, max=1}
  image.save(fileName, disp)
end

-- Checks whether a table contains a specific value
function tabl_contains(tabl, val)
   for i=1,#tabl do
      if tabl[i] == val then
         return true
      end
   end
   return false
end

-- Sums up all element in a given table
function tabl_sum(t)
  local sum = t[1]:clone()
  for i=2, #t do
    sum:add(t[i])
  end
  return sum
end

function str_split(str, delim, maxNb)
    -- Eliminate bad cases...
    if string.find(str, delim) == nil then
        return { str }
    end
    if maxNb == nil or maxNb < 1 then
        maxNb = 0    -- No limit
    end
    local result = {}
    local pat = "(.-)" .. delim .. "()"
    local nb = 1
    local lastPos
    for part, pos in string.gfind(str, pat) do
        result[nb] = part
        lastPos = pos
        nb = nb + 1
        if nb == maxNb then break end
    end
    -- Handle the last field
    result[nb] = string.sub(str, lastPos)
    return result
end

function fileExists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

function calcNumberOfContentImages(params)
  local frameIdx = 1
  while frameIdx < 100000 do
    local fileName = string.format(params.content_pattern, frameIdx + params.start_number)
    if not fileExists(fileName) then return frameIdx end
    frameIdx = frameIdx + 1
  end
  -- If there are too many content frames, something may be wrong.
  return 0
end

function build_OutFilename(params, image_number, iterationOrRun)
  local ext = paths.extname(params.output_image)
  local basename = paths.basename(params.output_image, ext)
  local fileNameBase = '%s%s-' .. params.number_format
  if iterationOrRun == -1 then
    return string.format(fileNameBase .. '.%s',
      params.output_folder, basename, image_number, ext)
  else
    return string.format(fileNameBase .. '_%d.%s',
      params.output_folder, basename, image_number, iterationOrRun, ext)
  end
end

function getFormatedFlowFileName(pattern, fromIndex, toIndex)
  local flowFileName = pattern
  flowFileName = string.gsub(flowFileName, '{(.-)}',
    function(a) return string.format(a, fromIndex) end )
  flowFileName = string.gsub(flowFileName, '%[(.-)%]',
    function(a) return string.format(a, toIndex) end )
  return flowFileName
end

function getContentImage(frameIdx, params)
  local fileName = string.format(params.content_pattern, frameIdx)
  if not fileExists(fileName) then return nil end
  local content_image = image.load(string.format(params.content_pattern, frameIdx), 3)
  content_image = preprocess(content_image):float()
  content_image = MaybePutOnGPU(content_image, params)
  return content_image
end

function getStyleImages(params)
  -- Needed to read content image size
  local firstContentImg = image.load(string.format(params.content_pattern, params.start_number), 3)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    -- Scale the style image so that it's area equals the area of the content image multiplied by the style scale.
    local img_scale = math.sqrt(firstContentImg:size(2) * firstContentImg:size(3) / (img:size(3) * img:size(2)))
        * params.style_scale
    img = image.scale(img, img:size(3) * img_scale, img:size(2) * img_scale, 'bilinear')
    print("Style image size: " .. img:size(3) .. " x " .. img:size(2))
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe)
  end

  for i = 1, #style_images_caffe do
     style_images_caffe[i] = MaybePutOnGPU(style_images_caffe[i], params)
  end

  return style_images_caffe
end
