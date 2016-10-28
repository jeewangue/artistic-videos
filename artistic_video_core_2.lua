-- Rebuild the network, insert style loss and return the indices for content and temporal loss
function buildNet(cnn, params, style_images_caffe)
   -- Handle style blending weights for multiple style inputs
  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_images_caffe do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',')
    assert(#style_blend_weights == #style_images_caffe,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum
  end

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
  -- Which layer to use for the temporal loss. By default, it uses a pixel based loss, masked by the certainty
  --(indicated by initWeighted).
  local temporal_layers = params.temporal_weight > 0 and {'initWeighted'} or {}

  local style_losses = {}
  local contentLike_layers_indices = {}
  local contentLike_layers_type = {}

  local next_content_i, next_style_i, next_temporal_i = 1, 1, 1
  local current_layer_index = 1
  local net = nn.Sequential()

  -- Set up pixel based loss.
  if temporal_layers[next_temporal_i] == 'init' or temporal_layers[next_temporal_i] == 'initWeighted'  then
    print("Setting up temporal consistency.")
    table.insert(contentLike_layers_indices, current_layer_index)
    table.insert(contentLike_layers_type,
      (temporal_layers[next_temporal_i] == 'initWeighted') and 'prevPlusFlowWeighted' or 'prevPlusFlow')
    next_temporal_i = next_temporal_i + 1
  end

  -- Set up other loss modules.
  -- For content loss, only remember the indices at which they are inserted, because the content changes for each frame.
  if params.tv_weight > 0 then
    local tv_mod = nn.TVLoss(params.tv_weight):float()
    tv_mod = MaybePutOnGPU(tv_mod, params)
    net:add(tv_mod)
    current_layer_index = current_layer_index + 1
  end
  for i = 1, #cnn do
    if next_content_i <= #content_layers or next_style_i <= #style_layers or next_temporal_i <= #temporal_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer = MaybePutOnGPU(avg_pool_layer, params)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)
      else
        net:add(layer)
      end
      current_layer_index = current_layer_index + 1
      if name == content_layers[next_content_i] then
        print("Setting up content layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'content')
        next_content_i = next_content_i + 1
      end
      if name == temporal_layers[next_temporal_i] then
        print("Setting up temporal layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'prevPlusFlow')
        next_temporal_i = next_temporal_i + 1
      end
      if name == style_layers[next_style_i] then
        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        gram = MaybePutOnGPU(gram, params)
        local target = nil
        for i = 1, #style_images_caffe do
          local target_features = net:forward(style_images_caffe[i]):clone()
          local target_i = gram:forward(target_features):clone()
          target_i:div(target_features:nElement())
          target_i:mul(style_blend_weights[i])
          if i == 1 then
            target = target_i
          else
            target:add(target_i)
          end
        end
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
        loss_module = MaybePutOnGPU(loss_module, params)
        net:add(loss_module)
        current_layer_index = current_layer_index + 1
        table.insert(style_losses, loss_module)
        next_style_i = next_style_i + 1
      end
    end
  end
  return net, style_losses, contentLike_layers_indices, contentLike_layers_type
end
