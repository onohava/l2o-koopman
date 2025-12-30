import copy
from tqdm.auto import tqdm
from torch import optim
import numpy as np
import torch
from torch.autograd import Variable
from src.models.optimizer import Optimizer

USE_CUDA = torch.cuda.is_available()

def w(v):
    if USE_CUDA:
        return v.cuda()
    return v


def detach_var(v):
    var = w(Variable(v.data, requires_grad=True))
    var.retain_grad()
    return var


def _flatten_params(module):
    params = [p.detach().view(-1) for _, p in module.all_named_parameters()]
    if not params:
        return torch.empty(0)
    return torch.cat(params)


def do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it,
           out_mul=1.0, koopman_model=None, should_train=True):
    if should_train:
        opt_net.train()
    else:
        opt_net.eval()
        unroll = 1

    target = target_cls(training=should_train)
    optimizee = w(target_to_opt())

    n_params = sum(int(np.prod(p.size())) for p in optimizee.parameters())
    device = next(optimizee.parameters()).device
    hidden_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
    cell_states = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]

    all_losses_ever = []
    all_losses = None

    # Use the Factory Method to create the appropriate window
    if koopman_model is not None:
        kae_win = koopman_model.create_window(device)
    else:
        kae_win = None

    for iteration in range(1, optim_it + 1):
        loss = optimizee(target)

        # Accumulate gradients
        all_losses = loss if all_losses is None else (all_losses + loss)
        all_losses_ever.append(loss.data.cpu().numpy())

        # KAE/DMD Window Update
        psi_t = None
        if kae_win is not None:
            theta_flat = _flatten_params(optimizee).to(device)
            psi_t = kae_win.push_and_encode(theta_flat, float(loss.item()))

        loss.backward(retain_graph=should_train)

        offset = 0
        result_params = {}
        hidden_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]
        cell_states2 = [w(Variable(torch.zeros(n_params, opt_net.hidden_sz))) for _ in range(2)]

        for name, p in optimizee.all_named_parameters():
            cur_sz = int(np.prod(p.size()))
            gradients = detach_var(p.grad.view(cur_sz, 1))

            # Optimizer Input: [Grads | Latent State]
            if psi_t is not None:
                psi_row = psi_t.unsqueeze(0).expand(cur_sz, -1)
                inp = torch.cat([gradients, psi_row], dim=1)
            else:
                inp = gradients

            updates, new_hidden, new_cell = opt_net(
                inp,
                [h[offset:offset + cur_sz] for h in hidden_states],
                [c[offset:offset + cur_sz] for c in cell_states]
            )

            for i in range(len(new_hidden)):
                hidden_states2[i][offset:offset + cur_sz] = new_hidden[i]
                cell_states2[i][offset:offset + cur_sz] = new_cell[i]

            result_params[name] = p + updates.view(*p.size()) * out_mul
            result_params[name].retain_grad()
            offset += cur_sz

        if iteration % unroll == 0:
            if should_train:
                meta_opt.zero_grad()
                all_losses.backward()
                meta_opt.step()
            all_losses = None

            optimizee = w(target_to_opt(**{k: detach_var(v) for k, v in result_params.items()}))
            hidden_states = [detach_var(v) for v in hidden_states2]
            cell_states = [detach_var(v) for v in cell_states2]
        else:
            optimizee = w(target_to_opt(**result_params))
            hidden_states = hidden_states2
            cell_states = cell_states2

    return all_losses_ever


def fit_optimizer(target_cls, target_to_opt, preproc=False, unroll=20, optim_it=100, n_epochs=20, n_tests=100, lr=0.001,
                  out_mul=1.0, koopman_model=None, writer=None):
    latent_dim = 0
    if koopman_model is not None:
        latent_dim = koopman_model.get_latent_dim(preproc)

    opt_net = w(Optimizer(latent_dim, preproc=preproc))
    all_params = list(opt_net.parameters())
    if koopman_model is not None and isinstance(koopman_model, torch.nn.Module):
        all_params += list(koopman_model.parameters())
        koopman_model.train()

    meta_opt = optim.Adam(all_params, lr=lr)

    best_net = None
    best_loss = float('inf')

    for epoch in tqdm(range(n_epochs), 'epochs'):
        train_loss_accum = []
        for _ in tqdm(range(20), 'iterations', leave=False):
            loss_hist = do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul,
                               should_train=True, koopman_model=koopman_model)
            train_loss_accum.append(np.sum(loss_hist))

        avg_train = np.mean(train_loss_accum)

        test_losses = []
        for _ in range(n_tests):
            lh = do_fit(opt_net, meta_opt, target_cls, target_to_opt, unroll, optim_it, out_mul, should_train=False,
                        koopman_model=koopman_model)
            test_losses.append(np.sum(lh))

        avg_test = np.mean(test_losses)

        if writer:
            writer.add_scalar('Loss/Train', avg_train, epoch)
            writer.add_scalar('Loss/Test', avg_test, epoch)

        if avg_test < best_loss:
            print(f"New Best Loss: {avg_test:.4f}")
            best_loss = avg_test
            best_net = copy.deepcopy(opt_net.state_dict())

    return best_loss, best_net