import numpy as np

def grad_norm(self, grad_input, grad_output):
    '''
        Hook for traking norm of module gradient

        Example:

        D = torch.Module()
        D.register_backward_hook(hook_grad)

        ## Train and call Variable().backward()
        print(D.__grad_norm__)
    '''

    self.__grad_norm__ = np.mean([p.data.norm() for p in grad_input if p is not None])
