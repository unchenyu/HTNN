'''
Tools on save weights, load weights and state logger.
'''

import numpy as np

def topK_error(predictions, labels, K=5):
    batch_size = labels.size(0)
    _, pred = predictions.topk(k=K, dim=1)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))
    correct_k = correct[:K].reshape(-1).float().sum(0)
    error = 1.0 - correct_k.mul_(1.0/batch_size)
    return error

class StatLogger:
    '''
    file writer to record various statistics
    '''

    def __init__(self, fpath):
        import os
        import os.path as pth

        self.fpath = fpath
        fdir = pth.split(fpath)[0]
        if len(fdir) > 0 and not pth.exists(fdir):
            os.makedirs(fdir)


    def report(self, epoch, **kwargs):
        import json
        with open(self.fpath, 'a') as fh:
            data = {
                'epoch': epoch
            }
            data.update(kwargs)
            fh.write(json.dumps(data, separators=(' ', ': ')) + "\n")
