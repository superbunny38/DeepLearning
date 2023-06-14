import torch
def softmax(net_outputs,num_class = 10):
  denom = torch.sum(torch.exp(net_outputs),dim = 1)
  result = torch.zeros(net_outputs.size())
  for idx in range(num_class):
    result[:,idx] = torch.div(net_outputs[:,idx],torch.sum(torch.exp(net_outputs),dim = 1))
  return result

net_output = model(images))
softmax_out = softmax(net_output)
