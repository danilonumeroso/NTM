import torch
from torch import nn

class Head(nn.Module):

    def __init__(self,
                 shared_memory,
                 no_units
    ):
        super(Head, self).__init__()

        self.memory_   = shared_memory
        rows, cols     = shared_memory.size_
        self.no_units_ = no_units
        self.prev_w_   = torch.zeros(rows)

        self.param_dims = [
            cols, # k
            1,    # beta
            1,    # g
            3,    # s
            1,    # gamma
            cols, # erase vector
            cols  # add vector
        ]

    def reset_states(self):
        rows, _ = self.memory_.size_
        self.prev_w_.detach_()
        self.prev_w_ = torch.zeros(rows)

    def _get_params(self, out):
        copy   = out.squeeze()
        params = []
        start = 0
        
        for dim in self.param_dims:
            end = start + dim
            params.append(copy[start:end])
            start = end

        return params
                 

class ReadHead(Head):
    def __init__(self,
                 shared_memory,
                 no_units
    ):
        super(ReadHead, self).__init__(shared_memory, no_units)

        self.param_dims = self.param_dims[:-2]

        # ReadHead is a linear transformation of the output of the
        # controller
        self.fc_r = nn.Linear(self.no_units_,
                              sum(self.param_dims))

        nn.init.xavier_uniform_(self.fc_r.weight)
        nn.init.normal_(self.fc_r.bias, std=0.5)

    def forward(self, input_):
        out = self.fc_r(input_)
        
        k, beta, g, s, gamma = self._get_params(out)

        r, w = self.memory_.read(k, beta, g, s, gamma, self.prev_w_)

        self.prev_w_ = w
        
        return r


class WriteHead(Head):
    def __init__(self,
                 shared_memory,
                 no_units
    ):
        super(WriteHead, self).__init__(shared_memory, no_units)

        # WriteHead is a linear transformation of the output of the
        # controller
        self.fc_w = nn.Linear(self.no_units_,
                              sum(self.param_dims))

        nn.init.xavier_uniform_(self.fc_w.weight)
        nn.init.normal_(self.fc_w.bias, std=0.5)

    def forward(self, input_):
        out = self.fc_w(input_)
        k, beta, g, s, gamma, e, a = self._get_params(out)

        e = torch.sigmoid(e)
        a = torch.tanh(a)
        
        w = self.memory_.write(k, beta, g, s, gamma, self.prev_w_, e, a)

        self.prev_w_ = w

        return w
        
