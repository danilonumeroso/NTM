import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from .memory import Memory
from .head import ReadHead, WriteHead

class NTM(nn.Module):

    def __init__(self,
                 no_inputs=10,
                 no_outputs=10,
                 no_units=20,
                 memory_size=(128,20)
    ):
        super(NTM, self).__init__()

        self.no_inputs_  = no_inputs
        self.no_units_   = no_units
        self.memory_size = memory_size
        self.no_outputs_ = no_outputs
        
        _, cols = memory_size

        # Controller
        self.controller_ = nn.LSTM(
            input_size=no_inputs + cols,
            hidden_size=no_units
        )

        for param in self.controller_.parameters():
            if param.dim() == 1:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)

        #LSTMController(no_inputs + memory_size[1],
        #                                  no_units)
        self.fc = nn.Linear(no_units + cols, no_outputs)
        
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, std=0.5)
        # End Controller
        
        # Memory
        self.memory_ = Memory(memory_size)
        # End Memory

        # Heads
        self.read_head_  = ReadHead(self.memory_,
                                    no_units)
        
        self.write_head_ = WriteHead(self.memory_,
                                     no_units)

        self.previous_read = torch.randn(1, cols) * 0.01
        # End Heads
        
    def reset(self):
        self.previous_read = self.previous_read.detach()
        
        self.h_ = torch.randn(1,1,self.no_units_) * 0.05
        self.c_ = torch.randn(1,1,self.no_units_) * 0.05

        self.memory_.reset_states()
        self.write_head_.reset_states()
        self.read_head_.reset_states()

    def forward(self, input_):
        
        # Feeding the controller with the
        # input concatenated with previous
        # memory read
        input_ = torch.cat((input_, self.previous_read), dim=1)
        out_, (self.h_, self.c_) = self.controller_(input_.unsqueeze(0),
                                                    (self.h_, self.c_))

        # Read and Write
        reads = self.read_head_(out_)
        self.write_head_(out_)

        # Ensuring that the output is in [0, 1]
        out_ = torch.sigmoid(self.fc(
            torch.cat((out_.squeeze(), reads))
        ))
        
        self.previous_read = reads.unsqueeze(0)
        
        return out_.view(1, -1, self.no_outputs_)
        
