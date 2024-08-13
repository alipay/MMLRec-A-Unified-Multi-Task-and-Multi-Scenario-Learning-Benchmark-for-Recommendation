import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random

class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


def gradnorm(model, task_loss, initial_task_loss):

    model.weights.grad.data = model.weights.grad.data * 0.0
    # get layer of shared weights
    W = model.expert_dnn

    # get the gradient norms for each of the tasks
    # G^{(i)}_w(t) 
    norms = []
    for i in range(len(task_loss)):
        # get the gradient of this task loss with respect to the shared parameters
        gygw = torch.autograd.grad(task_loss[i], W.parameters(), retain_graph=True)
        # compute the norm
        norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
    norms = torch.stack(norms)
    # compute the inverse training rate r_i(t) 
    # \curl{L}_i 
    if torch.cuda.is_available():
        loss_ratio = task_loss.data.cpu().numpy() / initial_task_loss
    else:
        loss_ratio = task_loss.data.numpy() / initial_task_loss
    # r_i(t)
    inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    # compute the mean norm \tilde{G}_w(t) 
    if torch.cuda.is_available():
        mean_norm = np.mean(norms.data.cpu().numpy())
    else:
        mean_norm = np.mean(norms.data.numpy())
    # compute the GradNorm loss 
    # this term has to remain constant
    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** args.alpha), requires_grad=False)
    if torch.cuda.is_available():
        constant_term = constant_term.cuda()
    #print('Constant term: {}'.format(constant_term))
    # this is the GradNorm loss itself
    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
    #print('GradNorm loss {}'.format(grad_norm_loss))

    # compute the gradient for the weights
    model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]


def cagrad():
    return

