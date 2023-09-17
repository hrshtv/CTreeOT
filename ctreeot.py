import torch
from torch import nn

class CTreeOT(nn.Module):

    # assumes first node (0) is the root

    def __init__(self, max_steps, thresh, epsilon, lamda):
        super(CTreeOT, self).__init__()
        self.max_steps = max_steps
        self.thresh = thresh
        self.epsilon = epsilon
        self.lamda = lamda

    def check(self, t, name = ''):
        if t.isnan().any():
            print(f'tensor {name} has NaN')
            raise ValueError('Stop')
        if t.isinf().any():
            print(f'tensor {name} has inf')
            raise ValueError('Stop')

    def pad_tensor(self, t, dims, value = 0):
        # t: [n,], dims: [l,r,t,b]
        return torch.nn.functional.pad(t, dims, mode = 'constant', value = value)

    def get_aux_idxs(self, edges, temp):
        # auxiliary matrices required for CTreeOT
        dst = edges[..., 1]
        # compute to tensor
        to_dist = torch.cdist(
            dst.reshape(1, -1, 1), 
            temp
        ).squeeze(0).squeeze(-1)
        to = (to_dist == 0).float()
        # return
        return to

    def forward(self, E1f, E1b, E2f, cost, constr_f):
        
        device = cost.device
        n, m = cost.shape

        # add dummies
        cost = self.pad_tensor(cost, (0, n, 0, m), value = 0)
        constr_f = self.pad_tensor(constr_f, (0, n, 0, 0), value = 0)
        constr_f = self.pad_tensor(constr_f, (0, 0, 0, n), value = 1)

        n, m = cost.shape

        # get phi and psi
        phi = cost.t() # [m, n]
        psi_f = 1 - constr_f.unsqueeze(-1) # [m, m, 1], 1 if constr violated, above repeated for all edges
        psi_b = psi_f.permute(1, 0, 2)
        psi_f = self.lamda*psi_f
        psi_b = self.lamda*psi_b

        # get auxiliary index tensors used mainly for aggregation
        temp = torch.arange(n).reshape(1, -1, 1).float().to(device)
        to_f = self.get_aux_idxs(E1f, temp)
        to_b = self.get_aux_idxs(E1b, temp)
        to_f_t = to_f.t()
        to_b_t = to_b.t()

        # init log u, v, and messages, phi is [m, n]
        u = torch.zeros(1, n).float().to(device)
        v = torch.zeros(m, 1).float().to(device)
        msg_f = torch.zeros(m, E1f.shape[-2]).float().to(device) # messages, msg[i, j] = message_{sj \to tj}[i] 
        msg_b = torch.zeros(m, E1b.shape[-2]).float().to(device)
        sum_msg_f = torch.zeros_like(phi) # [m, n]
        sum_msg_b = torch.zeros_like(phi) # [m, n]

        # iterations
        for step in range(self.max_steps):

            u_prev = u.detach().clone()
            
            # note: phi is [m, n] tensor -- already transposed.

            # update u
            u_tilde = (sum_msg_f + sum_msg_b - phi/self.epsilon - v) # [m, n]
            u = torch.logsumexp(u_tilde, dim = -2, keepdim = True) # over m

            # update v
            v_tilde = (sum_msg_f + sum_msg_b - phi/self.epsilon - u) # [m, n]
            v = torch.logsumexp(v_tilde, dim = -1, keepdim = True) # over n

            # update msg_f
            term_f = ((phi/self.epsilon + u + v - sum_msg_f - sum_msg_b))@to_f_t
            msg_tilde_f = psi_f/self.epsilon + msg_b.unsqueeze(-2) # [m, m, num_edges_1] (j, j', e)
            msg_f = (0.5*(msg_f + term_f + torch.logsumexp(-msg_tilde_f, dim = -3)))
            sum_msg_f = msg_f@to_f

            # update msg_b
            term_b = ((phi/self.epsilon + u + v - sum_msg_f - sum_msg_b))@to_b_t
            msg_tilde_b = psi_b/self.epsilon + msg_f.unsqueeze(-2) # [m, m, num_edges_1] (j, j', e)
            msg_b = (0.5*(msg_b + term_b + torch.logsumexp(-msg_tilde_b, dim = -3)))
            sum_msg_b = msg_b@to_b

            # check convergence
            absdiff = (u - u_prev).abs().max()
            if absdiff < self.thresh and step > 5:
                break

        logp = (sum_msg_f + sum_msg_b - phi/self.epsilon - u - v) # [m, n]
        logp = logp.clamp(max = 0)
        p = torch.exp(logp).t() # [n, m]

        return p
