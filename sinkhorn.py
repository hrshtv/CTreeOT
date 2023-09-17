import torch
import torch.nn as nn

class Sinkhorn(nn.Module):

    def __init__(self, epsilon, max_steps, thresh):
        super(Sinkhorn, self).__init__()
        self.epsilon = epsilon
        self.max_steps = max_steps
        self.thresh = thresh

    def check(self, t, name = ''):
        if t.isnan().any():
            print(t)
            raise ValueError(f'tensor {name} has NaN')
        if t.isinf().any():
            print(t)
            raise ValueError(f'tensor {name} has inf')

    def pad_tensor(self, t, dims, value = 0):
        # t: [n,], dims: [l,r,t,b]
        return torch.nn.functional.pad(t, dims, mode = 'constant', value = value)

    def forward(self, C, dummy = True, scale = True):
        # C: [batch_size, n, m]

        n, m = C.shape

        if dummy: # pad with dummy nodes to make C an (n+m x n+m) square matrix
            C = self.pad_tensor(C, (0, n, 0, m), value = 0)
            n, m = C.shape

        p = torch.ones(n).cuda() / n
        q = torch.ones(m).cuda() / m

        logp = torch.log(p)
        logq = torch.log(q)

        u = torch.zeros_like(p) # [batch_size, n]
        v = torch.zeros_like(q) # [batch_size, m]

        # Sinkhorn iterations
        for step in range(self.max_steps):
            
            u_prev = u.detach().clone()
            
            log_pi = ((u.unsqueeze(-1) + v.unsqueeze(-2) - C) / self.epsilon)
            u = u + self.epsilon*(logp - torch.logsumexp(log_pi, dim = -1))
            log_pi = ((u.unsqueeze(-1) + v.unsqueeze(-2) - C) / self.epsilon)
            v = v + self.epsilon*(logq - torch.logsumexp(log_pi.t(), dim = -1))

            err = (u - u_prev).abs().max().item()
            if err < self.thresh and step > 5:
                break

        # print(f'Converged after: {step} iterations')

        log_pi = ((u.unsqueeze(-1) + v.unsqueeze(-2) - C) / self.epsilon)
        log_pi = log_pi.clamp(max = 0)
        
        if scale:
            pi = n*torch.exp(log_pi)
        else:
            pi = torch.exp(log_pi)

        # self.check(p)

        return pi