


require 'optim'

-- modified to include a threshold for relative changes in the loss function as stopping criterion
local lbfgs_mod = require 'lbfgs'

---
--- MAIN FUNCTIONS
---

function runOptimization(params, net, content_losses, style_losses, temporal_losses,
   spatial_losses, img, frameIdx, runIdx, max_iter)
  local isMultiPass = (runIdx ~= -1)

  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.     - 모든 gradients는 extra loss modules로부터오니까,
                                                             -- 일단 gradiet 사이즈 정하기위해 0 통과
  local y = net:forward(img)
  local dy = img.new(#y):zero()

  -- Declaring this here lets us access it in maybe_print
  local optim_state = nil
  if params.optimizer == 'lbfgs' then
    optim_state = {
      maxIter = max_iter,
      tolFunRelative = params.tol_loss_relative,
      tolFunRelativeInterval = params.tol_loss_relative_interval,
      verbose=true,
    }
  elseif params.optimizer == 'adam' then
    optim_state = {
      learningRate = params.learning_rate,
    }
			print('----------------jh', optim_state.learningRate)
  else
    error(string.format('Unrecognized optimizer "%s"', params.optimizer))
  end

  local function maybe_print(t, loss, alwaysPrint)
    local should_print = (params.print_iter > 0 and t % params.print_iter == 0) or alwaysPrint
    if should_print then
      print(string.format('Iteration %d / %d', t, max_iter))
      for i, loss_module in ipairs(content_losses) do
        print(string.format('  Content %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(temporal_losses) do
        print(string.format('  Temporal %d loss: %f', i, loss_module.loss))
      end
      for i, loss_module in ipairs(style_losses) do
        print(string.format('  Style %d loss: %f', i, loss_module.loss))
			end

			for i, loss_module in ipairs(spatial_losses) do
				print(string.format('  Spatial %d loss: %f', i, loss_module.loss))
			end

      print(string.format('  Total loss: %f', loss))
    end
  end

  local function print_end(t)      -- 마지막 프린트니까 loss도 다 계산함 원래 feval이 하던 놈
    --- calculate total loss
    local loss = 0
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(temporal_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
    end

		for _, mod in ipairs(spatial_losses) do
			loss = loss + mod.loss
		end

    -- print informations
    maybe_print(t, loss, true)
  end

  local function maybe_save(t, isEnd)
    local should_save_intermed = params.save_iter > 0 and t % params.save_iter == 0
    local should_save_end = t == max_iter or isEnd  -- t가 max거나 isEnd면 무조건 save
    if should_save_intermed or should_save_end then
      local filename = nil
      if isMultiPass then
        filename = build_OutFilename(params, frameIdx, runIdx)
      else
        filename = build_OutFilename(params, math.abs(frameIdx - params.start_number + 1), should_save_end and -1 or t)
      end
      save_image(img, filename)
    end
  end

  -- Function to evaluate loss and gradient. We run the net forward and
  -- backward to get the gradient, and sum up losses from the loss modules.
  -- optim.lbfgs internally handles iteration and calls this fucntion many
  -- times, so we manually count the number of iterations to handle printing
  -- and saving intermediate results.
  --Lbfgs는 이 feval 함수를(optim max_iter만큼) 내부적으로 계속 부르니까, function 자체적으로 몇번 불렸는지 세고있어서, 그때마다 print해줘야함
  -- 그리고, num_call은 global 변수!! 1! ! !!! 중요!
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:backward(x, dy)                            -- grad = net:backward(input, gradOutput) 꼴인 것!
    local loss = 0                                              -- backward가 알아서 updateGradInput부르고, 저장해줌.
    for _, mod in ipairs(content_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(temporal_losses) do
      loss = loss + mod.loss
    end
    for _, mod in ipairs(style_losses) do
      loss = loss + mod.loss
		end

		for _, mod in ipairs(spatial_losses) do
			loss = loss + mod.loss
		end
    maybe_print(num_calls, loss, (num_calls==100))
    -- Only need to print if single-pass algorithm is used.
    if not isMultiPass then
      maybe_save(num_calls, false)
    end

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())  --- gradients를 한줄로 만들어서 보내기. lbfgs는 한줄짜리를 필요로함
  end

  start_time = os.time()

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = lbfgs_mod.optimize(feval, img, optim_state) -- img는 내가 stylized하고있는 img를 말함. optim_state에 max_iter이미있음
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')   -- Adam은 max_iter만큼 돌리라고 우리가 지정해줘야함.
    for t = 1, max_iter do
      local x, losses = optim.adam(feval, img, optim_state)   --QQQQ x 어따쓰지? 안쓰나 그냥? -->830 해결! x는 새롭게 img자리에 들어갈 x*
    end                                                       --- x*, f(x) = adam(func(x), x,[config],[state])
  end

  end_time = os.time()
  elapsed_time = os.difftime(end_time-start_time)
  print("Running time: " .. elapsed_time .. "s")

  print_end(num_calls)
  maybe_save(num_calls, true)
end

-- Rebuild the network, insert style loss and return the indices for content and temporal loss --- 823_01_Buildnet분석
function buildNet(cnn, params, style_images_caffe)
   -- Handle style blending weights for multiple style inputs
  ---1단계. 스타일 블렌딩 설정.  여러 스타일 인풋들에 대해 style blending weights들 알맞게 대응-----

  local style_blend_weights = nil
  if params.style_blend_weights == 'nil' then     -----param의 style_blend_weights 가 nil이면, style_images_caffe 크기만큼,1로된 array만듬
    -- Style blending not specified, so use equal weighting
    style_blend_weights = {}
    for i = 1, #style_images_caffe do
      table.insert(style_blend_weights, 1.0)
    end
  else
    style_blend_weights = params.style_blend_weights:split(',') ------ nil 아니면, 넣어줌. 그리고, images_caffe랑 크기 같은지 확인
                                                                ---- 이때, weights:split(',')하면 ','마다 나눠서 문자열로 저장한
                                                             ---- table을 갖게 됨.{"1","2",..}이런식 따라서, 문자열을 수로 바꿔야함. (ㄱ)
    assert(#style_blend_weights == #style_images_caffe,
      '-style_blend_weights and -style_images must have the same number of elements')
  end
  -- Normalize the style blending weights so they sum to 1
  local style_blend_sum = 0
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = tonumber(style_blend_weights[i])     -- tonumber 이용해서, string table을 숫자 table로 바꿔줌 (ㄱ)!
    style_blend_sum = style_blend_sum + style_blend_weights[i]
  end
  for i = 1, #style_blend_weights do
    style_blend_weights[i] = style_blend_weights[i] / style_blend_sum  --전체크기로 normalize. 다합치면 1되게.
  end

    ---2단계. Layer들 build 및 loss 계산? 시작-----

  local content_layers = params.content_layers:split(",")
  local style_layers = params.style_layers:split(",")
  -- Which layer to use for the temporal loss. By default, it uses a pixel based loss, masked by the certainty
  --(indicated by initWeighted).
  local temporal_layers = params.temporal_weight > 0 and {'initWeighted'} or {}-- layer초기화 temporal_weight>0이면 initweighted,아니면 {}

  local style_losses = {}
  local contentLike_layers_indices = {}
  local contentLike_layers_type = {}

  local next_content_i, next_style_i, next_temporal_i = 1, 1, 1
  local current_layer_index = 1
  local net = nn.Sequential() -- net을 nn.sequential로 initialization !! Container가 Sequental. input 하나, output 하나.


	if params.spatial_weight > 0 then
		print("Setting up spatial consistency")
		print(current_layer_index,"is indicating space of spatial_loss_module. It will be inserted in main func")
		table.insert(contentLike_layers_indices, current_layer_index)
		table.insert(contentLike_layers_type,'spatial')
	end

    ---------!!!!! 중요!!! 어차피 temporal layer에 하나밖에 안들어가서, 이거 한번밖에 안 쓰임. 아래 for문에서, temporal_layer 따로 안쌓음!
    --                          temporal_layer에 여러개 넣는지 다시한번 확인해야하지만, 아닌 것 같음. 생각해보니 그럴 필요도 없음.
  -- Set up pixel based loss. initweighted거나 init이면, contentLike_indices에 current_layer_index 넣고, type에 flowweighted인지 flow인지.
	if temporal_layers[next_temporal_i] == 'init' or temporal_layers[next_temporal_i] == 'initWeighted'  then
    print("Setting up temporal consistency.")
    table.insert(contentLike_layers_indices, current_layer_index)

		print(current_layer_index,"is indicating space of  temporal_loss_module. It will be inserted in main func")

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
   	print("adding module to net. layer's name is TV_Loss")
    print( current_layer_index, "after inserting tv_loss module")
  end
  for i = 1, #cnn do -- cnn 은 받은 모듈. content_layer, style_layer, temporal_layer수만큼 network 쌓는 거 같음.
    if next_content_i <= #content_layers or next_style_i <= #style_layers or next_temporal_i <= #temporal_layers then
      local layer = cnn:get(i)
      local name = layer.name
      local layer_type = torch.type(layer)                    --- cnn layer를 차례대로 읽어와서, 이름, type 파악.
      local is_pooling = (layer_type == 'cudnn.SpatialMaxPooling' or layer_type == 'nn.SpatialMaxPooling')
      if is_pooling and params.pooling == 'avg' then --- check  whether layer is average pooling layer
        assert(layer.padW == 0 and layer.padH == 0)
        local kW, kH = layer.kW, layer.kH
        local dW, dH = layer.dW, layer.dH
        local avg_pool_layer = nn.SpatialAveragePooling(kW, kH, dW, dH):float()
        avg_pool_layer = MaybePutOnGPU(avg_pool_layer, params)
        local msg = 'Replacing max pooling at layer %d with average pooling'
        print(string.format(msg, i))
        net:add(avg_pool_layer)

				print("adding layer to net. layer's name is ", name, "type is ", layer_type)

      else
        net:add(layer)
				print("adding layer to net. layer's name is ", name, "type is ", layer_type)
				----- average pooling layer 아닌경우. 그냥 삽입 net:add(layer)
      end
      current_layer_index = current_layer_index + 1                   --- current_layer_index 1 증가
      if name == content_layers[next_content_i] then

				--- content layer일경우( 보통 ReLU 4_2 추가하고, 그 index 반환)
 	      print("The # of current_layer_index = ", current_layer_index)
		   	print("Setting up content layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'content')
        next_content_i = next_content_i + 1
      end
      if name == temporal_layers[next_temporal_i] then               -- VGG CNN의 name이 prevPlus 이따구일리가 없잖아 ㅠ 왜 여기있음?
        print("The # of current_layer_index = ", current_layer_index)
				print("Setting up temporal layer", i, ":", layer.name)
        table.insert(contentLike_layers_indices, current_layer_index)
        table.insert(contentLike_layers_type, 'prevPlusFlow')
        next_temporal_i = next_temporal_i + 1                --- content Layer인지, temproal Layer인지 판단해서 content_Like_layer에 추가
      end
      if name == style_layers[next_style_i] then
        print(current_layer_index, "after inserting style_layer",name)

        print("Setting up style layer  ", i, ":", layer.name)
        local gram = GramMatrix():float()
        gram = MaybePutOnGPU(gram, params)
        local target = nil
        for i = 1, #style_images_caffe do          --- 각 style image에 대해 gram 계산하고, style_blend_weight 곱해주고 target feature수로
          local target_features = net:forward(style_images_caffe[i]):clone()        --나눠주고, target_feature는 gram 구할 feature같음.
          local target_i = gram:forward(target_features):clone()                    -- target_i : i번째 style의 gram매트릭스들.
          target_i:div(target_features:nElement())
          target_i:mul(style_blend_weights[i])
          if i == 1 then
            target = target_i
          else
            target:add(target_i)
          end                                                                     -- target에 gram들 모아놓음 (하나면 하나, 여러개면 여러개)
        end
        local norm = params.normalize_gradients
        local loss_module = nn.StyleLoss(params.style_weight, target, norm):float()
        loss_module = MaybePutOnGPU(loss_module, params)
        net:add(loss_module)
        current_layer_index = current_layer_index + 1
				print(current_layer_index, "after inserting stlye_loss_module" )
				-- style_layer setting up 할때는, loss_module도 넣어주고
        table.insert(style_losses, loss_module)                        -- current_layer_index 1 추가
        next_style_i = next_style_i + 1
      end
    end
  end
  return net, style_losses, contentLike_layers_indices, contentLike_layers_type --ContentLikelayers_indices는 main에loss_indices로 리턴
end                                                                             -- type은 그에 해당하는 type들(content or prevPlusflow)

--
-- LOSS MODULES
--

-- Define an nn Module to compute content loss in-place                        -- Loss Module들을 nn.으로 부를수있게 정의하는 듯.
local ContentLoss, parent = torch.class('nn.ContentLoss', 'nn.Module')         -- torch.class() : user가 쉽게 새로운 classes 정의하게해줌
                                                                               -- 자세한 사항은 Torch_Study -> 'Torch 구조'에서 참고
function ContentLoss:__init(strength, target, normalize) --- nn의 Module을 inherit, 그 init도 받아옴.
  parent.__init(self)
  self.strength = strength
  self.target = target                                    -- target은 target image . content이므로 reLu 4_2
  self.normalize = normalize or false
  self.loss = 0
  self.crit = nn.MSECriterion()                          --- cirt = MSECriterion!
end

function ContentLoss:updateOutput(input)                      -- target과 비교해서 loss 계산할 input(사실 얜 output. 그래서 updateoutput임)
  if input:nElement() == self.target:nElement() then           -- main의 forward(x)가 얘를 자동으로 부름. -> loss도 알아서 계산, output수정
    self.loss = self.crit:forward(input, self.target) * self.strength -- loss 계산하려면 input(net의output, 4_2)와 target크기같아야함

	else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function ContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then                 --
    self.gradInput = self.crit:backward(input, self.target)          -- input 자체가 output( 이 모듈은 input이 곧 output이므로)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)                                     -- 이 모듈지날떄는,gradinput이 지금까지 쌓여왓던 gradOutput이랑 더해짐)
  return self.gradInput
end

-- Define an nn Module to compute content loss in-place
local WeightedContentLoss, parent = torch.class('nn.WeightedContentLoss', 'nn.Module')

function WeightedContentLoss:__init(strength, target, weights, normalize, loss_criterion) ---Flow에 Weight 줘서 뭔가 할때 쓰는 놈.
  parent.__init(self)
  self.strength = strength
  if weights ~= nil then
    -- Take square root of the weights, because of the way the weights are applied
    -- to the mean square error function. We want w*(error^2), but we can only
    -- do (w*error)^2 = w^2 * error^2 , Weight는 flow weight_tabel에서 받는듯. 이거는 자세히는 아직 모르니까 일단 패스. 근데 spatial에서안쓸듯.
    self.weights = torch.sqrt(weights)                                             --
    self.target = torch.cmul(target, self.weights)
  else
    self.target = target
    self.weights = nil
  end
  self.normalize = normalize or false
  self.loss = 0
  if loss_criterion == 'mse' then
    self.crit = nn.MSECriterion()
  elseif loss_criterion == 'smoothl1' then
    self.crit = nn.SmoothL1Criterion()
  else
    print('WARNING: Unknown flow loss criterion. Using MSE.')
    self.crit = nn.MSECriterion()
  end
end

function WeightedContentLoss:updateOutput(input)
  if input:nElement() == self.target:nElement() then
    self.loss = self.crit:forward(input, self.target) * self.strength
    if self.weights ~= nil then
      self.loss = self.crit:forward(torch.cmul(input, self.weights), self.target) * self.strength
    else
      self.loss = self.crit:forward(input, self.target) * self.strength
    end
  else
    print('WARNING: Skipping content loss')
  end
  self.output = input
  return self.output
end

function WeightedContentLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
    if self.weights ~= nil then
      self.gradInput = self.crit:backward(torch.cmul(input, self.weights), self.target)--loss에 고려되지 않는 부분의 weight항이 0임.
    else                                                                              -- 즉, input(image)의 고려하지않을부분 0곱해버리는듯
      self.gradInput = self.crit:backward(input, self.target)
    end
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


-- *** Todo : Define an nn Module to compute Spatial loss in-place
local SpatialLoss, parent = torch.class('nn.SpatialLoss', 'nn.Module')         -- torch.class() : user가 쉽게 새로운 classes 정의하게해줌
                                                                               -- 자세한 사항은 Torch_Study -> 'Torch 구조'에서 참고
function SpatialLoss:__init(target_img , flow_spatial, warpImage, flowWeights, params)

  parent.__init(self)
	self.params = params

  self.strength = params.spatial_weight
	self.weights=flowWeights

	--save_image(target_img, "a.png")
	--self.target = image.load("a.png", 3):float()
	self.target = target_img
		local w = self.target:size()[3]
		local y = self.target:narrow(3,w/2+1,w/2)
		y:copy(self.target:narrow(3,1,w/2))
	 self.target = preprocess(self.target):float()

    self.target = MaybePutOnGPU(self.target, self.params)


	--print(type(self.target))
	self.flow = flow_spatial
	self.warpImage = warpImage
	self.normalize = params.normalize_gradients or false
  self.loss = 0
  self.crit = nn.MSECriterion()
end

function SpatialLoss:updateOutput(input)                      -- target과 비교해서 loss 계산할 input(사실 얜 output. 그래서 updateoutput임)
  if input:nElement() == self.target:nElement() then
    --self.target = self.warpImage(input,self.flow) (이부분 채워야함 ! self.target을 정의하되, input을 소스로해서, optical_flow로 warp해야함.)
 --if self.weights ~= nil then
    -- Take square root of the weights, because of the way the weights are applied
    -- to the mean square error function. We want w*(error^2), but we can only
    -- do (w*error)^2 = w^2 * error^2
		--self.weights = torch.sqrt(self.weights)                                             --

		--print('spatial loss term available')
		--print(self.target)
		--print(input)
		--
	  ---	local w = self.target:size()[3]
	  --	local y = self.target:narrow(3,w/2+1,w/2)
	  --	y:copy(self.target:narrow(3,1,w/2))
	 	-- self.target = torch.cmul(target, self.weights)
   -- self.target = preprocess(self.target):float()
		--self.target = self.target:float()
   -- self.target = MaybePutOnGPU(self.target, self.params)
		self.loss = self.crit:forward(input, self.target) * self.strength -- loss 계산하려면 input(net의output, 4_2)와 target크기같아야함

else
    print('WARNING: Skipping SpatialLoss loss ; size missmatch between inputimg & targetimg')
  end
  self.output = input
  return self.output
end

function SpatialLoss:updateGradInput(input, gradOutput)
  if input:nElement() == self.target:nElement() then
		self.gradInput = self.crit:backward(input, self.target)          -- input 자체가 output( 이 모듈은 input이 곧 output이므로)
  end
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)                                     -- 이 모듈지날떄는,gradinput이 지금까지 쌓여왓던 gradOutput이랑 더해짐)
  return self.gradInput
end


-- Returns a network that computes the CxC Gram matrix from inputs
-- of size C x H x W
function GramMatrix()
  local net = nn.Sequential()
  net:add(nn.View(-1):setNumInputDims(2))
  local concat = nn.ConcatTable()
  concat:add(nn.Identity())
  concat:add(nn.Identity())
  net:add(concat)
  net:add(nn.MM(false, true))
  return net
end


-- Define an nn Module to compute style loss in-place
local StyleLoss, parent = torch.class('nn.StyleLoss', 'nn.Module')

function StyleLoss:__init(strength, target, normalize)
  parent.__init(self)
  self.normalize = normalize or false
  self.strength = strength
  self.target = target
  self.loss = 0

  self.gram = GramMatrix()
  self.G = nil
  self.crit = nn.MSECriterion()
end

function StyleLoss:updateOutput(input)
  self.G = self.gram:forward(input)
  self.G:div(input:nElement())
  self.loss = self.crit:forward(self.G, self.target)
  self.loss = self.loss * self.strength

	self.output = input
  return self.output
end

function StyleLoss:updateGradInput(input, gradOutput)
  local dG = self.crit:backward(self.G, self.target)
  dG:div(input:nElement())
  self.gradInput = self.gram:backward(input, dG)
  if self.normalize then
    self.gradInput:div(torch.norm(self.gradInput, 1) + 1e-8)
  end
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end


local TVLoss, parent = torch.class('nn.TVLoss', 'nn.Module')

function TVLoss:__init(strength)
  parent.__init(self)
  self.strength = strength
  self.x_diff = torch.Tensor()
  self.y_diff = torch.Tensor()
end

function TVLoss:updateOutput(input)
  self.output = input
  return self.output
end

-- TV loss backward pass inspired by kaishengtai/neuralart
function TVLoss:updateGradInput(input, gradOutput)
  self.gradInput:resizeAs(input):zero()
  local C, H, W = input:size(1), input:size(2), input:size(3)
  self.x_diff:resize(3, H - 1, W - 1)
  self.y_diff:resize(3, H - 1, W - 1)
  self.x_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.x_diff:add(-1, input[{{}, {1, -2}, {2, -1}}])
  self.y_diff:copy(input[{{}, {1, -2}, {1, -2}}])
  self.y_diff:add(-1, input[{{}, {2, -1}, {1, -2}}])
  self.gradInput[{{}, {1, -2}, {1, -2}}]:add(self.x_diff):add(self.y_diff)
  self.gradInput[{{}, {1, -2}, {2, -1}}]:add(-1, self.x_diff)
  self.gradInput[{{}, {2, -1}, {1, -2}}]:add(-1, self.y_diff)
  self.gradInput:mul(self.strength)
  self.gradInput:add(gradOutput)
  return self.gradInput
end

function getContentLossModuleForLayer(net, layer_idx, target_img, params)
  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do                                                  -- layer_idx 전 놈까지 net을 복사해서, 그 layer를 뽑아냄
    local layer = net:get(i)
    tmpNet:add(layer)
  end
  local target = tmpNet:forward(target_img):clone()                          -- target_img를 통과시켜서, layer_idx 에서 결과(target)뽑음
  local loss_module = nn.ContentLoss(params.content_weight, target, params.normalize_gradients):float()
  loss_module = MaybePutOnGPU(loss_module, params)
  return loss_module
end

function getWeightedContentLossModuleForLayer(net, layer_idx, target_img, params, weights) -- Temporal_Loss 구하기
  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do
    local layer = net:get(i)
    tmpNet:add(layer)
  end
  local target = tmpNet:forward(target_img):clone()
  local loss_module = nn.WeightedContentLoss(params.temporal_weight, target, weights,
      params.normalize_gradients, params.temporal_loss_criterion):float()
  loss_module = MaybePutOnGPU(loss_module, params)
  return loss_module
end
---  *************Todo ---
function getSpatialLossModuleForLayer(net, layer_idx, image,flow_spatial,warpImage,flowWeights, params)

  local tmpNet = nn.Sequential()
  for i = 1, layer_idx-1 do
    local layer = net:get(i)
    tmpNet:add(layer)
  end

  local target = tmpNet:forward(image):clone()

	local loss_module = nn.SpatialLoss(target,flow_spatial,warpImage,flowWeights,params):float() -- norm_grad : false of true
  loss_module = MaybePutOnGPU(loss_module, params)
  return loss_module
end



---
--- HELPER FUNCTIONS
---

function MaybePutOnGPU(obj, params)    --- 824_02 : GPU를 쓸건지 안 쓸건지. param 받아서, gpu 쓰면 obj의 cuda나 cl을 , 안쓰면 그냥 obj를 리턴
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
  img = img:double()
	-- print('jh--' , img:type())
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

function build_OutFilename(params, image_number, iterationOrRun) --- 0824 : format에 맞게 outputfile_name 정해줌. parma이 format가지는듯.
  local ext = paths.extname(params.output_image)
  local basename = paths.basename(params.output_image, ext)
  local fileNameBase = '%s%s-' .. params.number_format
  if iterationOrRun == -1 then
    return string.format(fileNameBase .. '.%s', -- s s d s 순서
      params.output_folder, basename, image_number, ext)
  else
    return string.format(fileNameBase .. '_%d.%s',  -- s s d d s 순서로 다음것들 입력.
      params.output_folder, basename, image_number, iterationOrRun, ext)
  end
end

function getFormatedFlowFileName(pattern, fromIndex, toIndex)   -- {}대신 fromIndex, []대신 toIndex
  local flowFileName = pattern
  flowFileName = string.gsub(flowFileName, '{(.-)}',
    function(a) return string.format(a, fromIndex) end )
  flowFileName = string.gsub(flowFileName, '%[(.-)%]',
    function(a) return string.format(a, toIndex) end )
  return flowFileName
end

function getFormatedFlowFileNameSpatial(pattern, Index)
	local flowFileName = pattern
	flowFileName = string.gsub(flowFileName, '{(.-)}',
		function(a) return string.format(a, Index) end )
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

function getStyleImages(params)       --- style_image의 list들을 차곡차곡 받는데, firstContentImg를 기준으로 삼아서 Scailing해서 집어넣는다.
  -- Needed to read content image size
  local firstContentImg = image.load(string.format(params.content_pattern, params.start_number), 3)
  local style_image_list = params.style_image:split(',')
  local style_images_caffe = {}
  for _, img_path in ipairs(style_image_list) do
    local img = image.load(img_path, 3)
    -- Scale the style image so that it's area equals the area of the content image multiplied by the style scale.
    local img_scale = math.sqrt(firstContentImg:size(2) * firstContentImg:size(3) / (img:size(3) * img:size(2))) -- Q : scaling을 왜
        * params.style_scale                                                                                     -- 이렇게 하지?
    img = image.scale(img, img:size(3) * img_scale, img:size(2) * img_scale, 'bilinear')
    print("Style image size: " .. img:size(3) .. " x " .. img:size(2))
    local img_caffe = preprocess(img):float()
    table.insert(style_images_caffe, img_caffe) --> preprocess 완료된 이미지들를 차례차례 하나씩 쌓는다.
  end

  for i = 1, #style_images_caffe do
     style_images_caffe[i] = MaybePutOnGPU(style_images_caffe[i], params) -- 모든 image들에대해 GPU를 달아준다.
  end

  return style_images_caffe
end
