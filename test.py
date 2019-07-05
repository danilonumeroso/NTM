from ntm.data import copy_task
from ntm.ntm import NTM
import torch
import random
import numpy as np
import time
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import json

def test(ntm):
    ntm.reset()
    test = list(copy_task(no_attributes, (1, 1), min_len_seq, max_len_seq))
    for _, X_test, Y_test in test:
        for x_test, y_test in zip(X_test, Y_test):
            _, no_attr = x_test.size()
            seq_len, _ = y_test.size()
        
            x_ = x_test.view(-1, 1, no_attr)

            for i in range(x_.size(0)):
                ntm(x_[i])
            
            y_pred = torch.empty(y_test.size())
        
            for i in range(seq_len):
                empty_ = torch.zeros((1, no_attr))
                y_pred[i] = ntm(empty_).squeeze(0)

        y_pred = y_pred.data.apply_(lambda x: 0 if x < 0.5 else 1)
        cost = torch.sum(torch.abs(y_pred - y_test)).item()
        print("y_pred:")
        print(y_pred)
        print("y_test:")
        print(y_test)
        print("TestCost: %.2f" %(cost))

    return y_pred, y_test

no_attributes = 4
memory_size = (32,10)
min_len_seq = 1
max_len_seq = 3

loss_curve = []
cost_curve = []
report_interval = 100
checkpoint_interval = 1000

ntm = NTM(no_inputs=no_attributes+1,
          no_outputs=no_attributes,
          no_units=100,
          memory_size=memory_size
)


ntm.load_state_dict(torch.load("checkpoints/a.model"))
ntm.eval()

y_pred, y_test = test(ntm)

fig, axeslist = plt.subplots(ncols=2, nrows=1, gridspec_kw={'width_ratios': [1,1]})
figures = [('Inputs', y_test.transpose(0,1)), ('Outputs', y_pred.transpose(0,1))]
print(y_test.size())

for ind, (title, fig) in enumerate(figures):
    axeslist.ravel()[ind].imshow(fig, cmap='gray', interpolation='nearest')
    axeslist.ravel()[ind].set_title(title)

plt.sca(axeslist[1])
plt.tight_layout()
plt.show()
plt.savefig('heatmap', bbox_inches='tight')
            
with open('checkpoints/5770-5000-32.json') as json_file:  
    data = json.load(json_file)
    mean = []
    for start in range(0, len(data["loss"])-1, 1500):
        end = start + 1500
        if end > len(data["loss"]):
            end = len(data["loss"])
        mean.append(np.array(data["loss"][start:end]).mean())
    plt.plot(mean)
    plt.legend(["NTM w/LSTM controller"], fontsize = 'xx-large')
    plt.show()
