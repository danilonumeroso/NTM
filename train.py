from ntm.data import copy_task
from ntm.ntm import NTM
import torch
from torch import optim
from torch import nn
import numpy as np
import random
import time
import json

random_state = int(round(time.time() * 1000)%10000)
random_state = 620

random.seed(random_state)
torch.manual_seed(random_state)
np.random.seed(random_state)

no_batches = 10
batch_len  = 10
no_attributes = 4
memory_size = (10,5)
min_len_seq = 1
max_len_seq = 10

loss_curve = []
cost_curve = []
report_interval = 10
checkpoint_interval = 10

data = list(
    copy_task(no_attributes, (no_batches, batch_len), min_len_seq, max_len_seq)
)

#rms  = lambda: optim.RMSprop
#adam = lambda: optim.Adam

ntm = NTM(no_inputs=no_attributes+1,
          no_outputs=no_attributes,
          no_units=100,
          memory_size=memory_size
)

optimizer = optim.Adam(ntm.parameters(),
                       #momentum=0.9,
                       #alpha=0.95,
                       lr=1e-3)

criterion = nn.BCELoss()

no_epochs = 50
        
def clip_grads(net):
    parameters = list(filter(lambda param: param.grad is not None, net.parameters()))
    for param in parameters:
        param.grad.data.clamp_(-10, 10)

print("Using seed %d" %(random_state))
try:
    for epoch in range(no_epochs):
        for b_i, X_train, Y_train in data:
            for x_train, y_train in zip(X_train, Y_train):
                # dim(x) -> (1 x seq_len x no_attr)
                # dim(y) -> (1 x seq_len x no_attr)
    
                optimizer.zero_grad()
                ntm.reset()
            
                _, no_attr = x_train.size()
                seq_len, _ = y_train.size()
                # TRAIN BATCH
    
                # dim(x_) -> (seq_len x 1 x no_attr)
                # shape of x_ allows to fed the controller with the entire sequence
                # all at once.
                x_ = x_train.view(-1, 1, no_attr)
                for i in range(x_.size(0)):
                    ntm(x_[i])
                
                y_pred = torch.empty(y_train.size())

                assert seq_len == y_train.size(0)
                
                for i in range(seq_len):
                    empty_ = torch.zeros((1, no_attr))
                    y_pred[i] = ntm(empty_).squeeze(0)

            loss = criterion(y_pred, y_train)
            loss.backward()
            clip_grads(ntm)
            optimizer.step()
        
            y_pred = y_pred.data.apply_(lambda x: 0 if x < 0.5 else 1)
    
            # The cost is the number of error bits per sequence
            cost_curve.append(
                torch.sum(torch.abs(y_pred - y_train)).item() / batch_len
            )
        
            loss_curve.append(
                loss.item()
            )

            if b_i % report_interval == 0:
                mean_loss = np.array(loss_curve[-report_interval:]).mean()
                mean_cost = np.array(cost_curve[-report_interval:]).mean()
                print("Epoch %d Batch %d Loss: %.4f Cost: %.2f" %(epoch, b_i, mean_loss, mean_cost))

            if checkpoint_interval != 0 and (b_i % checkpoint_interval) == 0:
                base = "checkpoints/{}-{}-{}".format(random_state, epoch, b_i)
                jsonfn = base + ".json"
                modelfn = base + ".model"
                torch.save(ntm.state_dict(), modelfn)
                open(jsonfn, "wt").write(
                    json.dumps({
                        "loss": loss_curve,
                        "cost": cost_curve
                    })
                )
                print("Checkpoint")
    # END TRAIN BATCH
except KeyboardInterrupt:
    print("Training interrupted by user.")

