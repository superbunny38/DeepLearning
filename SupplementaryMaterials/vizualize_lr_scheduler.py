import matplotlib.pyplot as plt
import torch

'''
Code for Visualizing Learning rate schedulers

'''


model = torch.nn.Linear(2, 1)
ch_unet_optimizer = torch.optim.SGD(model.parameters(), args.LR,
                            momentum=args.MOMENTUM,
                            weight_decay=args.WEIGHTDECAY)
ch_unet_scheduler = CosineAnnealingLR(ch_unet_optimizer, T_max = args.T_MAX)

lrs = []
plt.title("My learning rate")

for i in range(args.EPOCHS):
    ch_unet_optimizer.step()
    lrs.append(ch_unet_optimizer.param_groups[0]["lr"])
#     print("Factor = ",i," , Learning Rate = ",optimizer.param_groups[0]["lr"])
    ch_unet_scheduler.step()

plt.plot(lrs)
