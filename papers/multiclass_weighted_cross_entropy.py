######### Multi-class Weighted Cross Entropy for Segmentation #########

def weighted_cce(logits, mask, class_weights = [0.1,4,2,4,1,3]):
    """"
    Version: Pytorch 1.12.0
    Author: Chaeeun Ryu
    mask: (BxHxW) B: BatchSize
    logits: (BxCxHxW) C: Number of Classes
    class_weights: weights per class (index 0: background)
    
    """
    mask = torch.unsqueeze(mask,1)
    criterion = SoftBCEWithLogitsLoss()
    loss = 0
    for idx,weight in enumerate(class_weights):
        bin_mask = (1*(mask[:,0,:,:]==idx)).to(torch.float64)
        bin_logits = logits[:,idx,:,:]
        loss += weight*criterion(bin_logits,bin_mask)
#         print(loss)
    return loss
