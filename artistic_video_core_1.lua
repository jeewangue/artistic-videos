require 'optim'

-- modified to include a threshold for relative changes in the loss function as stopping criterion
local lbfgs_mod = require 'lbfgs'

---
--- MAIN FUNCTIONS
---

function runOptimization(params, net, content_losses, style_losses, temporal_losses,
    img, frameIdx, runIdx, max_iter)
  local isMultiPass = (runIdx ~= -1)

  -- Run it through the network once to get the proper size for the gradient
  -- All the gradients will come from the extra loss modules, so we just pass
  -- zeros into the top of the net on the backward pass.
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
      print(string.format('  Total loss: %f', loss))
    end
  end

  local function print_end(t)
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
    -- print informations
    maybe_print(t, loss, true)
  end

  local function maybe_save(t, isEnd)
    local should_save_intermed = params.save_iter > 0 and t % params.save_iter == 0
    local should_save_end = t == max_iter or isEnd
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
  local num_calls = 0
  local function feval(x)
    num_calls = num_calls + 1
    net:forward(x)
    local grad = net:backward(x, dy)
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
    maybe_print(num_calls, loss, false)
    -- Only need to print if single-pass algorithm is used.
    if not isMultiPass then
      maybe_save(num_calls, false)
    end

    collectgarbage()
    -- optim.lbfgs expects a vector for gradients
    return loss, grad:view(grad:nElement())
  end

  start_time = os.time()

  -- Run optimization.
  if params.optimizer == 'lbfgs' then
    print('Running optimization with L-BFGS')
    local x, losses = lbfgs_mod.optimize(feval, img, optim_state)
  elseif params.optimizer == 'adam' then
    print('Running optimization with ADAM')
    for t = 1, max_iter do
      local x, losses = optim.adam(feval, img, optim_state)
    end
  end

  end_time = os.time()
  elapsed_time = os.difftime(end_time-start_time)
  print("Running time: " .. elapsed_time .. "s")

  print_end(num_calls)
  maybe_save(num_calls, true)
end
