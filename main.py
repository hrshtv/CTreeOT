import sys
import math
import time
import torch
import random
import numpy as np
from tqdm import tqdm

from ctreeot import CTreeOT
from sinkhorn import Sinkhorn

def edges2A(edges, n):
    A = torch.zeros(n, n).to(edges.device)
    A[edges[:,0].long(), edges[:,1].long()] = 1
    return A

def loss(p, C, edges1, constr_f, lamda, epsilon):
    term1 = (p*C).sum()
    A1 = edges2A(edges1, p.shape[0])
    p1 = p.unsqueeze(0).unsqueeze(-2)
    p2 = p.unsqueeze(1).unsqueeze(-1)
    edge_mask = A1.unsqueeze(-1).unsqueeze(-1)
    term2 = (lamda*(1 - constr_f)*p1*p2*edge_mask).sum()
    term3 = epsilon*(p*(torch.log(p + 1e-12) - 1)).sum()
    return (term1 + term2 + term3).item()

def build_tree(n):
    edges = []
    free_nodes = [0]
    dists = [0]
    num_nodes = 1
    lvl = 0
    while True:
        total_pos_new = 2*len(free_nodes)
        total_new_add = min(total_pos_new, n - num_nodes)
        num_add = np.random.randint(1, total_new_add + 1)
        pos_new = np.random.choice(np.arange(0, total_pos_new), num_add, replace = False)
        parent_idx = np.int_(np.floor(pos_new / 2))
        nodes_new = num_nodes + np.arange(0, num_add)
        for pid, v in zip(parent_idx, nodes_new):
            edges.append((free_nodes[pid], v))
        free_nodes = np.copy(nodes_new)
        num_nodes += num_add
        dists = dists + ((lvl+1)*np.ones(nodes_new.shape)).tolist()
        lvl += 1
        if num_nodes >= n:
            break
    return torch.tensor(edges).float().cuda(), torch.tensor(dists).float().cuda()

def check_constraints_fast(p, d1, d2, thresh = 1e-6):
    p = p / max(p.max(), 1e-12)
    d1_self = (d1.unsqueeze(-1) < d1.unsqueeze(0)).float() # d1[i1] < d1[i2]
    d2_self = (d2.unsqueeze(-1) >= d2.unsqueeze(0)).float() * (1 - torch.eye(d2.shape[0]).to(p.device)) # d2[i1] >= d1[i2]
    constr = torch.einsum('ik,jl->ijkl', d1_self, d2_self) # constr[i,j,k,l] = (d1[i] < d1[j]) & (d2[k] >= d2[l])
    pp = (torch.einsum('ij,kl->ijkl', p, p) > thresh).long()
    num_t = constr.sum().item()
    num_v = (pp*constr).sum().item()
    return num_v

def pairwise_cosine_similarity(x, y):
    return torch.cosine_similarity(x.unsqueeze(-3), y.unsqueeze(-2), dim = -1)

def benchmark(n_min, n_max, n_step, R, alg, f, seed, epsilon, lamda = 1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    times = []
    vios = []
    losses = []
    for n in tqdm(range(n_min, n_max + 1, n_step)):
        time_n = 0
        vio_n = 0
        l_n = 0
        for _ in range(R):
            edges1, dists1 = build_tree(n)
            edges2, dists2 = build_tree(n)
            cost = 2*torch.rand(n, n).cuda() - 1
            cost = (cost + cost.t()) / 2
            constr_f = (dists2.unsqueeze(-1) < dists2.unsqueeze(-2)).float()
            old_constr_f = constr_f.detach().clone()
            edges1b = edges1.flip(dims = (-1,))
            if alg == 'sinkhorn':
                t1 = time.time()
                p = f(cost)
                t2 = time.time()
            elif 'ctreeot' in alg:
                t1 = time.time()
                p = f(edges1, edges1b, edges2, cost, constr_f)
                t2 = time.time()
            else:
                raise ValueError('Invalid algorithm specified')
            p = p[:n, :n]
            vio = check_constraints_fast(p, dists1, dists2)
            vio_n += vio
            l = loss(p, cost, edges1, old_constr_f, lamda, epsilon)
            l_n += l
            time_n += (t2 - t1)
        time_n = time_n / R
        vio_n = vio_n / R
        l_n = l_n / R
        times.append(time_n)
        vios.append(vio_n)
        losses.append(l_n)
    return times, vios, losses

if __name__ == '__main__':
    
    torch.use_deterministic_algorithms(True)

    alg = sys.argv[1]
    n_min = int(sys.argv[2])
    n_max = int(sys.argv[3])
    n_step = int(sys.argv[4])
    R = int(sys.argv[5])

    if alg == 'sinkhorn':
        epsilon = 1e-2
        lamda = int(1e3) # used for computing the loss, not used in sinkhorn
        f = Sinkhorn(epsilon = epsilon, max_steps = 100, thresh = 1e-4).cuda()
    elif 'ctreeot' in alg:
        epsilon = 1e-3
        lamda = int(1e3)
        f = CTreeOT(max_steps = 100, thresh = 1e-4, lamda = lamda, epsilon = epsilon).cuda()
    else:
        raise ValueError('Invalid algorithm specified')

    # Run benchmarking
    nodes_range = np.arange(n_min, n_max + 1, n_step)
    _ = benchmark(n_min, n_max, 2, 10, alg, f, seed = 0, epsilon = epsilon, lamda = lamda) # warmup
    times, vios, losses = benchmark(n_min, n_max, n_step, R, alg, f, seed = 42, epsilon = epsilon, lamda = lamda)

    alg = f'{alg}_{lamda}_{epsilon}'

    # Save benchmarking results
    np.save(f'results/nodes_range.npy', nodes_range)
    np.save(f'results/{alg}_times.npy', np.array(times, dtype = np.float32))
    np.save(f'results/{alg}_vios.npy', np.array(vios, dtype = np.float32))

    print('Done')
