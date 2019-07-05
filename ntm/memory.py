import torch
from torch import nn
import torch.nn.functional as F

class Memory(nn.Module):

    def __init__(self,
                 size=(128,8)
    ):
        super(Memory, self).__init__()
        
        self.size_ = size

        self._mem = torch.empty(size)
        nn.init.normal_(self._mem, std=0.05)
        self.memory_ = self._mem.clone()

    def reset_states(self):
        self.memory_ = self._mem.clone()
        
    def _address(self, k, beta, g, s, gamma, prev_w):
        
        k = torch.tanh(k.view(1,self.size_[1]))
        beta = torch.relu(beta)
        g = torch.sigmoid(g)
        s = F.softmax(s, dim=0)
        gamma = torch.relu(gamma) + 1
        
        wc = self._similarity(k, beta)
        wg = self._interpolation(prev_w, wc, g)
        ws = self._shifting(wg, s)
        w  = self._sharpening(ws, gamma)
        return w

    def _similarity(self, k, beta):
        return F.softmax(
            F.cosine_similarity(self.memory_, k),
            dim=0
        )

    def _interpolation(self, prev_w, wc, g):
        return g * wc + (1 - g) * prev_w

    def _shifting(self, wg, s):
        return F.conv1d(
            torch.cat((wg[-1:], wg, wg[:1])).view(1, 1, -1),
            s.view(1, 1, -1)
        ).squeeze()

    def _sharpening(self, ws, gamma):
        w = ws ** gamma
        return torch.div(w, torch.sum(w))

    def read(self, k, beta, g, s, gamma, prev_w):
        w = self._address(k, beta, g, s, gamma, prev_w)
        return w @ self.memory_, w

    def write(self, k, beta, g, s, gamma, prev_w, e, a):
        w = self._address(k, beta, g, s, gamma, prev_w)
        
        # Reshaping the parameters
        w_ = w.unsqueeze(-1)
        e_ = e.unsqueeze(0)
        a_ = a.unsqueeze(0)
        
        erase = w_ @ e_
        add = w_ @ a_
        
        self.memory_ = self.memory_ * (1 - erase) + add
        return w
