import torch
#Categorical Cross Entropy (multi-class) from scratch
#CCE from scratch
#Writer: Chaeeun Ryu

def categorical_CE(target, softmax_output,num_class = 10):
  selected = softmax_output.gather(1,target.unsqueeze(0)).squeeze(0)
  epsilon = 1e-7
  CE = torch.log(torch.clip(selected,min = epsilon, max = 1.-epsilon))
  return - CE

ce = categorical_CE(targets, softmax_out)
