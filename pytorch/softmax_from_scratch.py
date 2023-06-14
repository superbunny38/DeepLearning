import torch
def softmax(net_outputs):
  denom = torch.sum(torch.exp(net_outputs)).item()
  return net_outputs/denom

net_output = model(images))
softmax_out = softmax(net_output)
