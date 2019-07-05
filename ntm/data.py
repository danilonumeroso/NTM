import numpy as np
import torch
import random

def copy_task(no_attributes,
              batches,
              min_len,
              max_len
):

    no_batches, batch_len = batches
    
    for i in range(no_batches):
        sequence_length = random.randint(min_len, max_len)
        sequence = np.random.choice([0,1],
                                    (batch_len, sequence_length, no_attributes))

        input_ = torch.zeros(batch_len, sequence_length+1, no_attributes + 1)
        input_[:, :-1, :-1] = torch.from_numpy(sequence)
        input_[:, -1, :] = 1
        output_ = torch.from_numpy(sequence).clone()

        yield i+1, input_, output_.float()
        
        
