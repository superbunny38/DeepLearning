import torch
import matplotlib.pyplot as plt

'''
Function to visulize image from pytorch tensor
Writer: Chaeeun Ryu
'''
def show_img(img_tensor):
    if img_tensor.size()[0] == 3:
        plt.imshow((img_tensor).permute(1,2,0).detach().cpu().numpy())
        plt.show()
    else:
        plt.imshow((img_tensor).detach().cpu().numpy())
        plt.show()
