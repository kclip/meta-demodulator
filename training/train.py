from __future__ import print_function
import torch
import numpy
from loss.cross_entropy_loss import cross_entropy_loss
from loss.cross_entropy_loss import cross_entropy_loss_test
from nets.meta_net import meta_net
import math
import os

def cavia_multi_inner_full_dep(args, iter_in_sampled_device, M, ind_d, iter_inner_loop, net, net_prime, train_set,
                                    s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter,
                                    if_cycle, if_online, ind_T, device):
    if if_online:
        s_total = train_set[:, :2]  # use whole data as training data for reptile
        y_total = train_set[:, 2:]
    else:
        s_total = train_set[ind_d, :, :2]  # use whole data as training data for reptile
        y_total = train_set[ind_d, :, 2:]

    net_prime.zero_grad()
    for f in net.parameters():
        f.start_para = f.data.clone()  # we need to keep this since we need to update finally (we don't touch net during inner)
        # make f_prime has same parameter with f (f is actual weight, f_prime is for inner loop optimization) (f stays still during the function)
    ind_f = 0
    #### make net_prime same as net #### we will use net_prime duing all inner loops ####
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f_prime.data = f.start_para
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    #### cavia - context para generation ####
    ### always start from 0 for each device & each epochs
    ### curr y shape : [mini_batch_size_meta, 2]
    context_para = torch.zeros(args.num_context_para, requires_grad=True)
    context_para = context_para.to(device)
    ##### keep using this para -> dependency everywhere...
    para_list_from_net_prime = list(map(lambda p: p[0], zip(net_prime.parameters())))
    for iter_idx_inner_loop in range(iter_inner_loop):
        if not iter_idx_inner_loop == iter_inner_loop-1:
            if not args.if_hess_vec_approx_for_one_inner_loop: # always sampling from 32 pilots
                if if_cycle: # disjoint sampling
                    num_cycles = int(K_TR + K_TE) // mini_batch_size_meta
                    cycle_ind = iter_idx_inner_loop % num_cycles  # repeat from 0~num_cycles-1
                    if iter_idx_inner_loop % num_cycles == 0:  # shuffle whenever all data has been visited
                        perm_idx = torch.randperm(int(K_TR + K_TE))
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[cycle_ind * mini_batch_size_meta + ind_in_minibatch]
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                else:
                    perm_idx = torch.randperm(int(
                        K_TR + K_TE))  # random sampling
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
            else: # sampling from 16 pilots for local update and sampling form 16 pilots for meta-update
                if not args.if_unify_K_tr_K_te:
                    if iter_idx_inner_loop == 0:
                        if if_online:
                            s_total = train_set[:K_TR, :2]
                            y_total = train_set[:K_TR, 2:]
                        else:
                            s_total = train_set[ind_d, :K_TR, :2]
                            y_total = train_set[ind_d, :K_TR, 2:]
                        perm_idx = torch.randperm(int(K_TR))
                        for ind_in_minibatch in range(mini_batch_size_meta):
                            mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                            s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    elif iter_idx_inner_loop == 1:
                        if if_online:
                            s_total = train_set[K_TR:, :2]  # use whole data as training data for reptile
                            y_total = train_set[K_TR:, 2:]
                        else:
                            s_total = train_set[ind_d, K_TR:, :2]  # use whole data as training data for reptile
                            y_total = train_set[ind_d, K_TR:, 2:]

                        if if_online:
                            print('we only use args.if_unify_K_tr_K_te = True for online case')
                            raise NotImplementedError

                        if args.mode_pilot_meta_training == 0:
                            perm_idx = perm_idx
                        elif args.mode_pilot_meta_training == 1:
                            perm_idx = torch.randperm(int(K_TE))
                        elif args.mode_pilot_meta_training == 2:
                            perm_idx = perm_idx[
                                       mini_batch_size_meta:]  # disjoint pilot set bet. meta-train, meta-test

                        for ind_in_minibatch in range(mini_batch_size_meta):
                            mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                            s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    else:
                        print('something wrong!')
                        raise NotImplementedError
            ##########################################################
            y_with_context = torch.zeros(mini_batch_size_meta, 2 + args.num_context_para)
            y_with_context = y_with_context.to(device)
            y_with_context[:,:2] = y
            y_with_context[:,2:] = context_para # copying slices seems no prob....

            net_meta_intermediate = meta_net(if_relu=args.if_relu)
            out = net_meta_intermediate(y_with_context, para_list_from_net_prime)
            ###################
            pure_loss = 0
            error_rate = 0
            pure_loss, error_rate = cross_entropy_loss(pure_loss, error_rate, M, s, out)
            loss = pure_loss
            if iter_idx_inner_loop == 0:
                if args.if_see_meta_inside_tensorboard:
                    writer_per_dev.add_scalar('loss_first_step', loss, iter)
                first_loss_curr = loss
            if iter_idx_inner_loop == iter_inner_loop - 1:
                if args.if_see_meta_inside_tensorboard:
                    writer_per_dev.add_scalar('loss_second_step', loss, iter)
                second_loss_curr = loss
            lr_alpha = lr_alpha_settings  # 0.1 #0.1
            if args.if_see_meta_inside_tensorboard:
                if iter == 0:
                    writer_per_dev.add_scalar('learning_rate_alpha', lr_alpha,
                                              iter)
            context_para_grad = torch.autograd.grad(loss, context_para, create_graph=True)
            context_para = context_para - lr_alpha * context_para_grad[0] # accumulating as with function of theta
        else:
            if not args.if_hess_vec_approx_for_one_inner_loop:
                if if_cycle: # disjoint sampling
                    num_cycles = int(K_TR + K_TE) // mini_batch_size_meta
                    cycle_ind = iter_idx_inner_loop % num_cycles
                    if iter_idx_inner_loop % num_cycles == 0:
                        perm_idx = torch.randperm(int(K_TR + K_TE))
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[cycle_ind * mini_batch_size_meta + ind_in_minibatch]
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                else:
                    perm_idx = torch.randperm(int(
                        K_TR + K_TE))  # random sampling
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
            else:
                if not args.if_unify_K_tr_K_te:
                    if iter_idx_inner_loop == 0:
                        if if_online:
                            s_total = train_set[:K_TR, :2]
                            y_total = train_set[:K_TR, 2:]
                        else:
                            s_total = train_set[ind_d, :K_TR, :2]
                            y_total = train_set[ind_d, :K_TR, 2:]
                        perm_idx = torch.randperm(int(K_TR))
                        for ind_in_minibatch in range(mini_batch_size_meta):
                            mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                            s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    elif iter_idx_inner_loop == 1:
                        if if_online:
                            s_total = train_set[K_TR:, :2]
                            y_total = train_set[K_TR:, 2:]
                        else:
                            s_total = train_set[ind_d, K_TR:, :2]
                            y_total = train_set[ind_d, K_TR:, 2:]
                        if if_online:
                            print('we only use args.if_unify_K_tr_K_te = True for online case')
                            raise NotImplementedError
                        if args.mode_pilot_meta_training == 0:
                            perm_idx = perm_idx
                        elif args.mode_pilot_meta_training == 1:
                            perm_idx = torch.randperm(int(K_TE))
                        elif args.mode_pilot_meta_training == 2:
                            perm_idx = perm_idx[
                                       mini_batch_size_meta:]  # disjoint pilot set bet. meta-train, meta-test
                        for ind_in_minibatch in range(mini_batch_size_meta):
                            mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                            s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    else:
                        print('something wrong!')
                        raise NotImplementedError
            ##########################################################
            y_with_context = torch.zeros(mini_batch_size_meta, 2 + args.num_context_para)
            y_with_context = y_with_context.to(device)
            y_with_context[:, :2] = y
            y_with_context[:, 2:] = context_para  # use not only data but all the gradients contained parameter
            ############ update network's parameter ###########
            out = net_meta_intermediate(y_with_context, para_list_from_net_prime)
            pure_loss = 0
            error_rate = 0
            pure_loss, error_rate = cross_entropy_loss(pure_loss, error_rate, M, s, out)
            loss = pure_loss
            if iter_idx_inner_loop == 0:
                if args.if_see_meta_inside_tensorboard:
                    writer_per_dev.add_scalar('loss_first_step', loss, iter)
                first_loss_curr = loss
            if iter_idx_inner_loop == iter_inner_loop - 1:
                if args.if_see_meta_inside_tensorboard:
                    writer_per_dev.add_scalar('loss_second_step', loss, iter)
                second_loss_curr = loss
            lr_alpha = lr_alpha_settings  # 0.1 #0.1
            if args.if_see_meta_inside_tensorboard:
                if iter == 0:
                    writer_per_dev.add_scalar('learning_rate_alpha', lr_alpha,
                                              iter)  # iter for in range ind_d what happens? maybe overwrite -> no prob
            loss.backward()  # backpropagation

    for f_prime in net_prime.parameters():
        f_prime.grad_actual_chain = f_prime.grad.data.clone()
    ind_f = 0
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f.actual_grad_curr = f_prime.grad_actual_chain.data.clone()
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    #### sum over num devs. ####
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = f.actual_grad_curr
        else:
            f.total_grad = f.total_grad + f.actual_grad_curr
    iter_in_sampled_device = iter_in_sampled_device + 1
    return iter_in_sampled_device, first_loss_curr, second_loss_curr

def fomaml_approx_multiple_inner_loop(args, iter_in_sampled_device, M, ind_d, iter_inner_loop, net, net_prime, train_set,
                                    s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter,
                                    if_cycle, if_online, ind_T):
    if if_online:
        s_total = train_set[:, :2]  # use whole data as training data for reptile
        y_total = train_set[:, 2:]
    else:
        s_total = train_set[ind_d, :, :2]  # use whole data as training data for reptile
        y_total = train_set[ind_d, :, 2:]

    for f in net.parameters():
        f.start_para = f.data.clone()  # we need to keep this since we need to update finally (we don't touch net during inner)
        # make f_prime has same parameter with f (f is actual weight, f_prime is for inner loop optimization) (f stays still during the function)
    ind_f = 0
    #### make net_prime same as net #### we will use net_prime duing all inner loops ####
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f_prime.data = f.start_para
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    for iter_idx_inner_loop in range(iter_inner_loop):
        if not args.if_hess_vec_approx_for_one_inner_loop:
            if if_cycle: # disjoint sampling
                num_cycles = int(K_TR + K_TE) // mini_batch_size_meta
                cycle_ind = iter_idx_inner_loop % num_cycles
                if iter_idx_inner_loop % num_cycles == 0:
                    perm_idx = torch.randperm(int(K_TR + K_TE))
                for ind_in_minibatch in range(mini_batch_size_meta):
                    mini_batch_idx = perm_idx[cycle_ind * mini_batch_size_meta + ind_in_minibatch]
                    s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
            else:
                perm_idx = torch.randperm(int(
                    K_TR + K_TE))  # random sampling
                for ind_in_minibatch in range(mini_batch_size_meta):
                    mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                    s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]

        else:
            if not args.if_unify_K_tr_K_te:
                if iter_idx_inner_loop == 0:
                    if if_online:
                        s_total = train_set[:K_TR, :2]
                        y_total = train_set[:K_TR, 2:]
                    else:
                        s_total = train_set[ind_d, :K_TR, :2]
                        y_total = train_set[ind_d, :K_TR, 2:]

                    perm_idx = torch.randperm(int(K_TR))
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                elif iter_idx_inner_loop == 1:
                    if if_online:
                        s_total = train_set[K_TR:, :2]
                        y_total = train_set[K_TR:, 2:]
                    else:
                        s_total = train_set[ind_d, K_TR:, :2]
                        y_total = train_set[ind_d, K_TR:, 2:]
                    if if_online:
                        print('we only use args.if_unify_K_tr_K_te = True for online case')
                        raise NotImplementedError
                    if args.mode_pilot_meta_training == 0:
                        perm_idx = perm_idx
                    elif args.mode_pilot_meta_training == 1:
                        perm_idx = torch.randperm(int(K_TE))
                    elif args.mode_pilot_meta_training == 2:
                        perm_idx = perm_idx[
                                   mini_batch_size_meta:]  # disjoint pilot set bet. meta-train, meta-test
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                else:
                    print('something wrong!')
                    raise NotImplementedError
        ###################
        net_prime.zero_grad()
        out = net_prime(
            y)  # inner loop update should be taken only temporarily.. net update should be done after seeing num_dev tasks
        ###################
        pure_loss = 0
        error_rate = 0
        pure_loss, error_rate = cross_entropy_loss(pure_loss, error_rate, M, s, out)
        loss = pure_loss
        if iter_idx_inner_loop == 0:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('loss_first_step', loss, iter)
            first_loss_curr = loss
        if iter_idx_inner_loop == iter_inner_loop - 1:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('loss_second_step', loss, iter)
            second_loss_curr = loss
        lr_alpha = lr_alpha_settings  # 0.1 #0.1
        if args.if_see_meta_inside_tensorboard:
            if iter == 0:
                writer_per_dev.add_scalar('learning_rate_alpha', lr_alpha,
                                          iter)  # iter for in range ind_d what happens? maybe overwrite -> no prob
        loss.backward()  # backpropagation
        ###### update #######
        for f_prime in net_prime.parameters():
            f_prime.data.sub_(f_prime.grad.data * lr_alpha)

    for f_prime in net_prime.parameters():
            f_prime.grad_actual_chain = f_prime.grad.data.clone()
    ind_f = 0
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f.actual_grad_curr = f_prime.grad_actual_chain.data.clone()
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    #### sum over num devs. ####
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = f.actual_grad_curr
        else:
            f.total_grad = f.total_grad + f.actual_grad_curr
    iter_in_sampled_device = iter_in_sampled_device + 1
    return iter_in_sampled_device, first_loss_curr, second_loss_curr

def maml_approx_multiple_inner_loop(args, iter_in_sampled_device, M, ind_d, iter_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T):
    if if_online:
        s_total = train_set[:, :2]
        y_total = train_set[:, 2:]
    else:
        s_total = train_set[ind_d, :, :2]
        y_total = train_set[ind_d, :, 2:]
    #### initalizae list for weight savings (should have iter_inner_loop) and gradient saving (should have iter_inner_loop)
    for f_prime in net_prime.parameters():
        f_prime.weight_footprint = [] # accumulating
        f_prime.grad_footprint = [] # accumulating
    f_prime.s_footprint = [] # only save to last layer
    f_prime.y_footprint = []

    for f in net.parameters():
        f.start_para = f.data.clone()  # we need to keep this since we need to update finally (we don't touch net during inner)
        # make f_prime has same parameter with f (f is actual weight, f_prime is for inner loop optimization) (f stays still during the function)
    ind_f = 0
    #### make net_prime same as net #### we will use net_prime duing all inner loops ####
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f_prime.data = f.start_para
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    for iter_idx_inner_loop in range(iter_inner_loop):
        if not args.if_hess_vec_approx_for_one_inner_loop:
            if if_cycle: # disjoint sampling
                num_cycles = int(K_TR+K_TE)//mini_batch_size_meta
                cycle_ind = iter_idx_inner_loop%num_cycles # repeat from 0~num_cycles-1
                if iter_idx_inner_loop%num_cycles == 0:
                    perm_idx = torch.randperm(int(K_TR + K_TE))
                for ind_in_minibatch in range(mini_batch_size_meta):
                    mini_batch_idx = perm_idx[cycle_ind*mini_batch_size_meta + ind_in_minibatch]
                    s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
            else:
                perm_idx = torch.randperm(int(K_TR + K_TE))  # random sampling
                for ind_in_minibatch in range(mini_batch_size_meta):
                    mini_batch_idx = perm_idx[ind_in_minibatch]
                    s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
        else:
            if not args.if_unify_K_tr_K_te:
                if iter_idx_inner_loop == 0:
                    if if_online:
                        s_total = train_set[:K_TR, :2]
                        y_total = train_set[:K_TR, 2:]
                    else:
                        s_total = train_set[ind_d, :K_TR, :2]
                        y_total = train_set[ind_d, :K_TR, 2:]
                    perm_idx = torch.randperm(int(K_TR))
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                elif iter_idx_inner_loop == 1:
                    if if_online:
                        s_total = train_set[K_TR:, :2]  # use whole data as training data for reptile
                        y_total = train_set[K_TR:, 2:]
                    else:
                        s_total = train_set[ind_d, K_TR:, :2]  # use whole data as training data for reptile
                        y_total = train_set[ind_d, K_TR:, 2:]
                    if if_online:
                        print('we only use args.if_unify_K_tr_K_te = True for online case')
                        raise NotImplementedError
                    if args.mode_pilot_meta_training == 0:
                        perm_idx = perm_idx
                    elif args.mode_pilot_meta_training == 1:
                        perm_idx = torch.randperm(int(K_TE))
                    elif args.mode_pilot_meta_training == 2:
                        perm_idx = perm_idx[
                                   mini_batch_size_meta:]  # disjoint pilot set bet. meta-train, meta-test
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                else:
                    print('something wrong!')
                    raise NotImplementedError
        ###################
        net_prime.zero_grad()
        out = net_prime(y) # inner loop update should be taken only temporarily.. net update should be done after seeing num_dev tasks
        ###################
        pure_loss = 0
        error_rate = 0
        pure_loss, error_rate = cross_entropy_loss(pure_loss, error_rate, M, s, out)
        loss = pure_loss
        if iter_idx_inner_loop == 0:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('loss_first_step', loss, iter)
            first_loss_curr = loss
        if iter_idx_inner_loop == iter_inner_loop - 1:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('loss_second_step', loss, iter)
            second_loss_curr = loss
        lr_alpha = lr_alpha_settings  # 0.1 #0.1
        if args.if_see_meta_inside_tensorboard:
            if iter == 0:
                writer_per_dev.add_scalar('learning_rate_alpha', lr_alpha, iter)  # iter for in range ind_d what happens? maybe overwrite -> no prob
        loss.backward() # backpropagation
        ##### save footprints #####
        for f_prime in net_prime.parameters():
            f_prime.weight_footprint.append(f_prime.data.clone())
            f_prime.grad_footprint.append(f_prime.grad.data.clone())
        f_prime.s_footprint.append(s.data.clone())
        f_prime.y_footprint.append(y.data.clone())
        ###### update #######
        for f_prime in net_prime.parameters():
            f_prime.data.sub_(f_prime.grad.data * lr_alpha)

    ########## Hessian-vecotor approximation by subtracting gradients at two points
    for ind_bp_inv in range(iter_inner_loop-1):
        ind_bp = iter_inner_loop - 2 - ind_bp_inv # ind_bp goes from n-1, n-2, ..., 0 like backprob
        if ind_bp_inv == 0:
            for f_prime in net_prime.parameters():
                f_prime.grad_next_chain = f_prime.grad.data.clone()
        else:
            for f_prime in net_prime.parameters():
                f_prime.grad_next_chain = f_prime.grad_actual_chain.data.clone()
        ### calculate gradient at point (\theta + g) ###
        for f_prime in net_prime.parameters():
            if args.if_use_handmade_epsilon_hessian:
                epsilon_hessian = args.epsilon_hessian
            else:
                machine_epsilon = numpy.finfo(numpy.float32).eps
                epsilon_hessian = (2 * pow(machine_epsilon, 0.5) * (1+ torch.norm(f_prime.weight_footprint[ind_bp])))/torch.norm(f_prime.grad_next_chain)
            f_prime.data = f_prime.weight_footprint[ind_bp] + epsilon_hessian * f_prime.grad_next_chain # last (n-th) is not used
        s = f_prime.s_footprint[ind_bp]
        y = f_prime.y_footprint[ind_bp]
        net_prime.zero_grad()
        out = net_prime(y)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s, out)
        loss.backward()
        #### now f_prime.grad has our desired grad
        #### now get hessian*vector approx. ####
        for name_prime, f_prime in net_prime.named_parameters():
            f_prime.hv_approx = (1 / epsilon_hessian) * (f_prime.grad.data - f_prime.grad_footprint[ind_bp])
            f_prime.grad_actual_chain = f_prime.grad_next_chain - lr_alpha * f_prime.hv_approx
    ind_f = 0
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f.actual_grad_curr = f_prime.grad_actual_chain.data.clone()
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    #### sum over num devs. ####
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = f.actual_grad_curr
        else:
            f.total_grad = f.total_grad + f.actual_grad_curr
    iter_in_sampled_device = iter_in_sampled_device + 1
    return iter_in_sampled_device, first_loss_curr, second_loss_curr

def maml_complex(args, ind_d, iter_in_sampled_device, M, net, net_prime, s, y, s_te, y_te, train_set, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, lr_beta_settings, if_use_same_seq_both_meta_train_test, device, iter, if_online, ind_T, mini_batch_size_meta_train_arg, mini_batch_size_meta_test_arg
                 ): # maml with full computation of Hessian matrix
    if args.if_unify_K_tr_K_te:
        if if_online:
            s_total = train_set[:K_TR, :2]
            y_total = train_set[:K_TR, 2:]
            perm_idx = torch.randperm(int(K_TR))
        else:
            s_total = train_set[ind_d, :, :2]
            y_total = train_set[ind_d, :, 2:]
            perm_idx = torch.randperm(int(K_TR+K_TE))

    else:
        if if_online:
            s_total = train_set[:K_TR, :2]
            y_total = train_set[:K_TR, 2:]
        else:
            s_total = train_set[ind_d, :K_TR, :2]
            y_total = train_set[ind_d, :K_TR, 2:]
        perm_idx = torch.randperm(int(K_TR))
    if args.if_mb_meta_change:
        mini_batch_size_meta_train = mini_batch_size_meta_train_arg
        mini_batch_size_meta_test = mini_batch_size_meta_test_arg
    else:
        mini_batch_size_meta_train = mini_batch_size_meta
        mini_batch_size_meta_test = mini_batch_size_meta
    for ind_in_minibatch in range(mini_batch_size_meta_train):
        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD

        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
    out = net(y)
    ###################
    pure_loss = 0
    error_rate = 0
    pure_loss, error_rate = cross_entropy_loss(pure_loss, error_rate, M, s, out)
    loss = pure_loss
    if args.if_see_meta_inside_tensorboard:
        writer_per_dev.add_scalar('loss_first_step', loss, iter)
    first_loss_curr = loss
    lr_alpha = lr_alpha_settings  # 0.1 #0.1
    if args.if_see_meta_inside_tensorboard:
        writer_per_dev.add_scalar('learn'
                                  'ing_rate_alpha', lr_alpha,
                                  iter)  # iter for in range ind_d what happens? maybe overwrite -> no prob
    for f in net.parameters():
        f.grad_inner = torch.autograd.grad(loss, f, create_graph=True)
        f.updated_para = f - lr_alpha * f.grad_inner[
            0]  # [0] since tuple, f.updated_para keeps the tree with f and f.grad_inner
        f.J_list = []  # for all the jacobians we need w.r.t. all the parameters in all layers
    for f in net.parameters():  # here f is for f
        # get Jacobian for all pairs of parameters
        # if #f = m, m*m number of Jac. needed
        for f_tilde in net.parameters():  # here f is for f~
            f_tilde.updated_para_flat = f_tilde.updated_para.view(1, -1)
            for index_updated_para in range(f_tilde.updated_para_flat.size()[1]):
                Jac = torch.autograd.grad(f_tilde.updated_para_flat[0][index_updated_para], f,
                                          create_graph=True)
                Jac = Jac[0].view(1, -1).data
                if index_updated_para == 0:
                    J = Jac
                else:
                    J = torch.cat((J, Jac), 0)
            J = torch.transpose(J, 0, 1)
            f.J_list.append(J)
    # now calcuate outer loop
    # first update para to updated_para
    ind_f = 0
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f_prime.data = f.updated_para
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    ################################################
    ## test in meta-training
    if args.if_unify_K_tr_K_te:
        if if_online:
            s_te_tot = train_set[:, :2]
            y_te_tot = train_set[:, 2:]
        else:
            s_te_tot = train_set[ind_d, :, :2]
            y_te_tot = train_set[ind_d, :, 2:]
    else:
        if if_online:
            s_te_tot = train_set[K_TR:, :2]
            y_te_tot = train_set[K_TR:, 2:]
        else:
            s_te_tot = train_set[ind_d, K_TR:, :2]
            y_te_tot = train_set[ind_d, K_TR:, 2:]
    if args.if_unify_K_tr_K_te:
        if args.mode_pilot_meta_training == 0:
            raise NotImplementedError
        elif args.mode_pilot_meta_training == 1:
            if if_online:
                perm_idx = torch.randperm(int(K_TE))
            else:
                perm_idx = torch.randperm(int(K_TR+K_TE)) # indep. sampling
        elif args.mode_pilot_meta_training == 2:
            if if_online:
                raise NotImplementedError # disjoint becomes weird since it will gonna decrease...to be 0
            else:
                perm_idx = perm_idx[mini_batch_size_meta_train:]  # disjoint sampling
    else:
        if if_online:
            print('we only use args.if_unify_K_tr_K_te = True for online case')
            raise NotImplementedError
        if args.mode_pilot_meta_training == 0:
            perm_idx = perm_idx
        elif args.mode_pilot_meta_training == 1:
            perm_idx = torch.randperm(int(K_TE))
        elif args.mode_pilot_meta_training == 2:
            perm_idx = perm_idx[mini_batch_size_meta_train:]  # disjoint pilot set bet. meta-train, meta-test
    for ind_in_minibatch in range(mini_batch_size_meta_test):
        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
        s_te[ind_in_minibatch] = s_te_tot[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
        y_te[ind_in_minibatch] = y_te_tot[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
    ######################
    out_te_prime = net_prime(y_te)  # jacobian
    ######################
    lr_beta = lr_beta_settings
    if args.if_see_meta_inside_tensorboard:
        writer_per_dev.add_scalar('learning_rate_beta', lr_beta, iter)
    pure_loss_te = 0
    error_rate_te = 0
    pure_loss_te, error_rate_te = cross_entropy_loss(pure_loss_te, error_rate_te, M, s_te, out_te_prime)
    loss_te = pure_loss_te
    if args.if_see_meta_inside_tensorboard:
        writer_per_dev.add_scalar('loss_second_step', loss_te, iter)
    second_loss_curr = loss_te
    outer_grad_list = []  # for outer grad saving for meaningless repitition of grad. calc.
    for f_tilde in net_prime.parameters():
        outer_grad = torch.autograd.grad(loss_te, f_tilde, create_graph=True)
        outer_grad_list.append(outer_grad[0])
    # now calculating actual gradient
    for f in net.parameters():
        for multivatiate_chain_idx in range(len(outer_grad_list)):
            outer_grad_flat_curr = outer_grad_list[multivatiate_chain_idx].view(-1, 1)
            Jac_curr = f.J_list[multivatiate_chain_idx]
            actual_grad_flat_curr = torch.matmul(Jac_curr, outer_grad_flat_curr)
            if multivatiate_chain_idx == 0:
                f.actual_grad = actual_grad_flat_curr
            else:
                f.actual_grad = f.actual_grad + actual_grad_flat_curr
        f.actual_grad = f.actual_grad.view(f.data.shape)
    ####################################################################################
    #### sum over num devs. ####
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = f.actual_grad
        else:
            f.total_grad = f.total_grad + f.actual_grad
    iter_in_sampled_device = iter_in_sampled_device + 1
    return  iter_in_sampled_device, first_loss_curr, second_loss_curr

def reptile(args, iter_in_sampled_device, M, ind_d, iter_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T):
    if if_online:
        s_total = train_set[:, :2] # use whole data as training data for reptile
        y_total = train_set[:, 2:]
    else:
        s_total = train_set[ind_d, :, :2]  # use whole data as training data for reptile
        y_total = train_set[ind_d, :, 2:]
    # make f_prime has same parameter with f (f is actual weight, f_prime is for inner loop optimization) (f stays still during the function)
    ind_f = 0
    for f in net.parameters():
        f.start_para = f.data.clone()
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f_prime.data = f.start_para
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    for iter_idx_inner_loop in range(iter_inner_loop):
        if not args.if_hess_vec_approx_for_one_inner_loop:
            if if_cycle: # disjoint sampling
                assert int(K_TR+K_TE)%mini_batch_size_meta == 0
                num_cycles = int(K_TR+K_TE)//mini_batch_size_meta
                cycle_ind = iter_idx_inner_loop%num_cycles # repeat from 0~num_cycles-1
                if iter_idx_inner_loop%num_cycles == 0: # only shuffle at the first time
                    perm_idx = torch.randperm(int(K_TR + K_TE))
                for ind_in_minibatch in range(mini_batch_size_meta):
                    mini_batch_idx = perm_idx[cycle_ind*mini_batch_size_meta + ind_in_minibatch]
                    s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
            else:
                perm_idx = torch.randperm(int(K_TR + K_TE))  # random sampling
                for ind_in_minibatch in range(mini_batch_size_meta):
                    mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                    s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
        else:
            if not args.if_unify_K_tr_K_te:
                if iter_idx_inner_loop == 0:
                    if if_online:
                        s_total = train_set[:K_TR, :2]
                        y_total = train_set[:K_TR, 2:]
                    else:
                        s_total = train_set[ind_d, :K_TR, :2]
                        y_total = train_set[ind_d, :K_TR, 2:]
                    perm_idx = torch.randperm(int(K_TR))
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                elif iter_idx_inner_loop == 1:
                    if if_online:
                        s_total = train_set[K_TR:, :2]
                        y_total = train_set[K_TR:, 2:]
                    else:
                        s_total = train_set[ind_d, K_TR:, :2]
                        y_total = train_set[ind_d, K_TR:, 2:]
                    if if_online:
                        print('we only use args.if_unify_K_tr_K_te = True for online case')
                        raise NotImplementedError
                    if args.mode_pilot_meta_training == 0:
                        perm_idx = perm_idx
                    elif args.mode_pilot_meta_training == 1:
                        perm_idx = torch.randperm(int(K_TE))
                    elif args.mode_pilot_meta_training == 2:
                        perm_idx = perm_idx[
                                   mini_batch_size_meta:]  # disjoint pilot set bet. meta-train, meta-test
                    for ind_in_minibatch in range(mini_batch_size_meta):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s[ind_in_minibatch] = s_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        y[ind_in_minibatch] = y_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                else:
                    print('something wrong!')
                    raise NotImplementedError

        ###################
        net_prime.zero_grad()
        out = net_prime(y) # inner loop update should be taken only temporarily.. net update should be done after seeing num_dev tasks
        ###################
        pure_loss = 0
        error_rate = 0
        pure_loss, error_rate = cross_entropy_loss(pure_loss, error_rate, M, s, out)
        loss = pure_loss
        if iter_idx_inner_loop == 0:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('loss_first_step', loss, iter)
            first_loss_curr = loss
        if iter_idx_inner_loop == iter_inner_loop - 1:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('loss_second_step', loss, iter)
            second_loss_curr = loss
        lr_alpha = lr_alpha_settings  # 0.1 #0.1
        if iter == 0:
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev.add_scalar('learning_rate_alpha', lr_alpha, iter)  # iter for in range ind_d what happens? maybe overwrite -> no prob
        loss.backward() # backpropagation
        for f_prime in net_prime.parameters():
            f_prime.data.sub_(f_prime.grad.data * lr_alpha)
    ### after inner loop update, calc the diff. between f and f_prime
    ind_f = 0
    for f in net.parameters():
        ind_f_prime = 0
        for f_prime in net_prime.parameters():
            if ind_f == ind_f_prime:
                f.diff_para = -(f_prime.data - f.data)
            ind_f_prime = ind_f_prime + 1
        ind_f = ind_f + 1
    #### sum over num devs. ####
    for f in net.parameters():
        if iter_in_sampled_device == 0:
            f.total_grad = f.diff_para
        else:
            f.total_grad = f.total_grad + f.diff_para
    iter_in_sampled_device = iter_in_sampled_device + 1
    return iter_in_sampled_device, first_loss_curr, second_loss_curr

def meta_train(M, num_epochs, num_dev, net, net_prime, train_set, K_TR, K_TE, device, writer_per_num_dev, writer_per_dev_tot, saved_model_PATH_meta_intermediate, lr_alpha_settings, lr_beta_settings, mini_batch_size_meta, if_use_same_seq_both_meta_train_test, sampled_device_num, jac_calc, reptile_inner_loop, if_cycle, args): #writer, writer_over_num_dev,
    _,len_of_pilots, _ = list(train_set.size())
    assert len_of_pilots == K_TR + K_TE
    if_online = False # for online, meta_train_online function
    ind_T = None
    for ind_d in range(num_dev):
        if args.if_see_meta_inside_tensorboard:
            writer_per_dev = writer_per_dev_tot[ind_d]
        if args.if_mb_meta_change:
            mini_batch_size_meta_train = args.mini_batch_size_meta_train
            mini_batch_size_meta_test = args.mini_batch_size_meta_test
        else:
            mini_batch_size_meta_train = mini_batch_size_meta
            mini_batch_size_meta_test = mini_batch_size_meta
        s = torch.zeros(mini_batch_size_meta_train, 2)
        y = torch.zeros(mini_batch_size_meta_train, 2)
        s = s.to(device)
        y = y.to(device)
        s_te = torch.zeros(mini_batch_size_meta_test, 2)
        y_te = torch.zeros(mini_batch_size_meta_test, 2)
        s_te = s_te.to(device)
        y_te = y_te.to(device)
    if args.if_continue_meta_training:
        net.load_state_dict(torch.load(args.path_for_continueing_meta_net))
        print('continued from' , args.path_for_continueing_meta_net)
        intermediate_PATH = saved_model_PATH_meta_intermediate + str(0) + 'continued_from_before' # initial
        torch.save(net.state_dict(), intermediate_PATH)
    else:
        intermediate_PATH = saved_model_PATH_meta_intermediate + str(0) # initial
        torch.save(net.state_dict(), intermediate_PATH)
    lr_beta = args.lr_beta_settings
    for iter in range(num_epochs):
        ###### avg. loss inside meta-training # sum over num_dev
        first_loss = 0
        second_loss = 0
        iter_in_sampled_device = 0
        if args.if_use_all_dev_in_one_epoch:
            if num_dev%sampled_device_num == 0:
                cycle_tot = num_dev//sampled_device_num
                if iter % cycle_tot == 0:
                    perm_idx_meta_device = torch.randperm(int(num_dev))
                cycle_ind = iter % cycle_tot
                sampled_device_idx_list = perm_idx_meta_device[cycle_ind * sampled_device_num : (cycle_ind+1) * sampled_device_num]
            elif (2*num_dev)%sampled_device_num == 0:
                cycle_tot = (2*num_dev) // sampled_device_num
                if iter % cycle_tot == 0:
                    perm_idx_meta_device_front = torch.randperm(int(num_dev))
                    perm_idx_meta_device_latter = perm_idx_meta_device_front.data
                    perm_idx_meta_device = torch.cat((perm_idx_meta_device_front, perm_idx_meta_device_latter),0)
                cycle_ind = iter % cycle_tot
                sampled_device_idx_list = perm_idx_meta_device[
                                          cycle_ind * sampled_device_num: (cycle_ind + 1) * sampled_device_num]
            else:
                raise NotImplementedError
        else:
            perm_idx_meta_device = torch.randperm(int(num_dev))
            sampled_device_idx_list = perm_idx_meta_device[:sampled_device_num]
        for ind_d in sampled_device_idx_list: # device
            ind_d = ind_d.numpy()
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev = writer_per_dev_tot[ind_d]
            else:
                writer_per_dev = []
            net.zero_grad()
            ## forward path
            if jac_calc == 1: # MAML with full Hessian computation
                iter_in_sampled_device, first_loss_curr, second_loss_curr = maml_complex(args, ind_d, iter_in_sampled_device, M, net, net_prime, s, y, s_te, y_te, train_set, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, lr_beta_settings, if_use_same_seq_both_meta_train_test, device, iter, if_online, ind_T, mini_batch_size_meta_train, mini_batch_size_meta_test)
            elif jac_calc == 2: # REPTILE
                iter_in_sampled_device, first_loss_curr, second_loss_curr = reptile(args, iter_in_sampled_device, M, ind_d, args.reptile_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 200: # MAML with approx Hv
                iter_in_sampled_device, first_loss_curr, second_loss_curr = maml_approx_multiple_inner_loop(args, iter_in_sampled_device, M, ind_d, args.maml_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 300: # FOMAML
                iter_in_sampled_device, first_loss_curr, second_loss_curr = fomaml_approx_multiple_inner_loop(args, iter_in_sampled_device, M, ind_d, args.maml_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 1001:  # CAVIA
                iter_in_sampled_device, first_loss_curr, second_loss_curr = cavia_multi_inner_full_dep(args, iter_in_sampled_device, M,
                                                                                  ind_d, args.maml_inner_loop, net,
                                                                                  net_prime, train_set, s, y, K_TR,
                                                                                  K_TE, mini_batch_size_meta,
                                                                                  writer_per_dev, lr_alpha_settings,
                                                                                  iter, if_cycle, if_online, ind_T,
                                                                                  device)
            first_loss = first_loss + first_loss_curr
            second_loss = second_loss + second_loss_curr
        first_loss = first_loss/sampled_device_num
        second_loss = second_loss/sampled_device_num
        for f in net.parameters():
            f.total_grad_prev = f.total_grad.data.clone()
            # f.total_grad: sum of gradients considering second derivative: MAML
            # f.total_grad: sum of delta_phi in Reptile
            # get expectation over sampled device num for each iteration
            # in Reptile, we need to consider minibatch in each sampled devices
            #f.total_grad = f.total_grad/sampled_device_num # average with num dev.s used in training
            f.total_grad = f.total_grad # sum over num dev.s used in training
            f.data.sub_(f.total_grad * lr_beta)
        if iter % 10 == 0: # 50...
            intermediate_PATH = saved_model_PATH_meta_intermediate + str(iter+1)
            torch.save(net.state_dict(), intermediate_PATH)
        print('epoch: ', iter, 'first loss: ', first_loss.data, 'second loss: ', second_loss.data)
        writer_per_num_dev.add_scalar('first_loss', first_loss, iter)
        writer_per_num_dev.add_scalar('second_loss', second_loss, iter)


def meta_train_online(M, num_epochs, num_dev, net, net_prime,  total_data_set, used_pilots_per_time, ind_T, K_TR_accum, K_TR, K_TE, device, writer_per_num_dev, writer_per_dev_tot, saved_model_PATH_meta_intermediate, lr_alpha_settings, lr_beta_settings, mini_batch_size_meta, if_use_same_seq_both_meta_train_test, sampled_device_num, jac_calc, reptile_inner_loop, if_cycle, args): #writer, writer_over_num_dev,
    if_online = True
    if args.if_see_meta_inside_tensorboard:
        for ind_d in range(num_dev):
            writer_per_dev = writer_per_dev_tot[ind_d]
    intermediate_PATH = saved_model_PATH_meta_intermediate + str(0) # initial
    torch.save(net.state_dict(), intermediate_PATH)

    for iter in range(num_epochs):
        first_loss = 0
        second_loss = 0
        iter_in_sampled_device = 0
        num_dev_curr = ind_T # we do not use current data as meta-training data but only for adaptation

        if args.if_use_all_dev_in_one_epoch:
            cycle_tot = num_dev_curr // sampled_device_num
            if iter % cycle_tot == 0:
                perm_idx_meta_device = torch.randperm(int(num_dev_curr))
            cycle_ind = iter % cycle_tot
        else:
            perm_idx_meta_device = torch.randperm(int(num_dev_curr)) # we need to use until current time's devices
        if sampled_device_num > num_dev_curr:
            sampled_device_num = 1
        sampled_device_idx_list = torch.zeros([sampled_device_num], dtype=torch.int)
        ind_sampled_device_idx_list = 0
        ind_perm_idx_meta_device = 0
        while ind_sampled_device_idx_list < sampled_device_num:
            if ind_perm_idx_meta_device == int(num_dev_curr): # currently do not have such devices
                ind_perm_idx_meta_device = -1
                break
            ind_d_tmp = perm_idx_meta_device[ind_perm_idx_meta_device]
            sampled_device_idx_list[ind_sampled_device_idx_list] = ind_d_tmp
            ind_sampled_device_idx_list += 1
            ind_perm_idx_meta_device += 1

        if ind_perm_idx_meta_device == -1:
            sampled_device_num = 1
            sampled_device_idx_list = torch.zeros([sampled_device_num], dtype=torch.int)
            ind_sampled_device_idx_list = 0
            ind_perm_idx_meta_device = 0
            while ind_sampled_device_idx_list < sampled_device_num:
                if ind_perm_idx_meta_device == int(num_dev_curr):  # currently do not have such devices
                    print('something wrong')
                    raise NotImplementedError
                ind_d_tmp = perm_idx_meta_device[ind_perm_idx_meta_device]
                sampled_device_idx_list[ind_sampled_device_idx_list] = ind_d_tmp
                ind_sampled_device_idx_list += 1
                ind_perm_idx_meta_device += 1

        for ind_d in sampled_device_idx_list: # device
            if ind_d == ind_T: # if it is different, the data comes from prev. so it has const. # of pilots
                print('we are not using current dataset as meta-training, something wrong!')
                raise NotImplementedError
            else:
                pass
            num_pilots = used_pilots_per_time[ind_d]
            K_TR = 0
            K_TE = num_pilots

            if mini_batch_size_meta > num_pilots:
                mini_batch_size_meta = num_pilots

            mini_batch_size_meta_train = mini_batch_size_meta
            mini_batch_size_meta_test = mini_batch_size_meta
            s = torch.zeros(mini_batch_size_meta_train, 2)
            y = torch.zeros(mini_batch_size_meta_train, 2)
            s = s.to(device)
            y = y.to(device)
            s_te = torch.zeros(mini_batch_size_meta_test, 2)
            y_te = torch.zeros(mini_batch_size_meta_test, 2)
            s_te = s_te.to(device)
            y_te = y_te.to(device)

            num_pilots = int(num_pilots)
            train_set = torch.zeros(num_pilots, 4)
            train_set = train_set.to(device)
            train_set = total_data_set[ind_d, :num_pilots, :] # rejection sampling is already done
            ind_d = ind_d.numpy()
            if args.if_see_meta_inside_tensorboard:
                writer_per_dev = writer_per_dev_tot[ind_d]
            else:
                writer_per_dev = []
            # refresh grad
            net.zero_grad()
            if jac_calc == 1: # corrected version
                iter_in_sampled_device, first_loss_curr, second_loss_curr = maml_complex(args, ind_d, iter_in_sampled_device, M, net, net_prime, s, y, s_te, y_te, train_set, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, lr_beta_settings, if_use_same_seq_both_meta_train_test, device, iter, if_online, ind_T, mini_batch_size_meta_train, mini_batch_size_meta_test)
            elif jac_calc == 2: # reptile
                iter_in_sampled_device, first_loss_curr, second_loss_curr = reptile(args, iter_in_sampled_device, M, ind_d, args.reptile_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 200:  # multiple inner maml with approx Hv
                iter_in_sampled_device, first_loss_curr, second_loss_curr = maml_approx_multiple_inner_loop(args,
                                                                                                            iter_in_sampled_device,
                                                                                                            M, ind_d,
                                                                                                            args.maml_inner_loop,
                                                                                                            net,
                                                                                                            net_prime,
                                                                                                            train_set,
                                                                                                            s, y, K_TR,
                                                                                                            K_TE,
                                                                                                            mini_batch_size_meta,
                                                                                                            writer_per_dev,
                                                                                                            lr_alpha_settings,
                                                                                                            iter,
                                                                                                            if_cycle,
                                                                                                            if_online,
                                                                                                            ind_T)

            elif jac_calc == 300:  # multiple inner fo-maml
                iter_in_sampled_device, first_loss_curr, second_loss_curr = fomaml_approx_multiple_inner_loop(args,
                                                                                                              iter_in_sampled_device,
                                                                                                              M, ind_d,
                                                                                                              args.maml_inner_loop,
                                                                                                              net,
                                                                                                              net_prime,
                                                                                                              train_set,
                                                                                                              s, y,
                                                                                                              K_TR,
                                                                                                              K_TE,
                                                                                                              mini_batch_size_meta,
                                                                                                              writer_per_dev,
                                                                                                              lr_alpha_settings,
                                                                                                              iter,
                                                                                                              if_cycle,
                                                                                                              if_online,
                                                                                                              ind_T)
            elif jac_calc == 1001:  # cavia
                iter_in_sampled_device, first_loss_curr, second_loss_curr = cavia_multi_inner_full_dep(args,
                                                                                                       iter_in_sampled_device,
                                                                                                       M,
                                                                                                       ind_d,
                                                                                                       args.maml_inner_loop,
                                                                                                       net,
                                                                                                       net_prime,
                                                                                                       train_set, s, y,
                                                                                                       K_TR,
                                                                                                       K_TE,
                                                                                                       mini_batch_size_meta,
                                                                                                       writer_per_dev,
                                                                                                       lr_alpha_settings,
                                                                                                       iter, if_cycle,
                                                                                                       if_online, ind_T,
                                                                                                       device)
            first_loss = first_loss + first_loss_curr
            second_loss = second_loss + second_loss_curr
        first_loss = first_loss/sampled_device_num
        second_loss = second_loss/sampled_device_num
        if iter % 10 == 0:
            print('epoch: ', iter, 'first loss: ', first_loss, 'second loss: ', second_loss)
        for f in net.parameters():
            # f.total_grad: sum of gradients considering second derivative: MAML
            # f.total_grad: sum of delta_phi in Reptile
            # get expectation over sampled device num for each iteration
            # in Reptile, we need to consider minibatch in each sampled devices
            #f.total_grad = f.total_grad/sampled_device_num # average with num dev.s used in training
            f.total_grad = f.total_grad # sum over num dev.s used in training
            f.data.sub_(f.total_grad * lr_beta_settings)
        if iter % 10 == 0: # 50...
            intermediate_PATH = saved_model_PATH_meta_intermediate + str(iter+1)
            torch.save(net.state_dict(), intermediate_PATH)

## For a certain device
def test_training(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH):
    num_dev_test_training_set, _, _ = test_training_set.size()
    if num_dev_test_training_set != 1: # for joint training
        print('something wrong for online')
        test_training_set_unified = torch.zeros(1, num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 4)
        for ind_dev in num_dev_test_training_set:
            test_training_set_unified[0, (ind_dev)*(K_TEST_TR + K_TEST_TE):(ind_dev+1)*(K_TEST_TR + K_TEST_TE), :] = test_training_set[ind_dev, :, :]
    else:
        test_training_set_unified = test_training_set
    if K_TEST_TR < mini_batch_size:
        mini_batch_size = 1
    mini_batch_num = K_TEST_TR/mini_batch_size
    s_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, :2]
    y_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, 2:]
    lr_alpha = lr_testtraining
    for epochs in range(num_epochs_test):
        if args.if_save_test_trained_net_per_epoch:
            if not os.path.exists(save_PATH + 'per_epochs' + '/'):
                os.mkdir(save_PATH + 'per_epochs' + '/')
            torch.save(net.state_dict(), save_PATH + 'per_epochs' + '/' + str(epochs))
        perm_idx = torch.randperm(int(K_TEST_TR))
        s_test = torch.zeros(mini_batch_size, 2)
        y_test = torch.zeros(mini_batch_size, 2)
        s_test = s_test.to(device)
        y_test = y_test.to(device)
        if args.if_test_train_no_permute:
            perm_idx_mini_batch_num = torch.randperm(int(mini_batch_num))
            mini_batch_idx_mb_num = perm_idx_mini_batch_num[0]  # SGD
            mini_batch_idx = mini_batch_idx_mb_num * mini_batch_size
            s_test = s_test_total[mini_batch_idx*1:(mini_batch_idx+mini_batch_size)*1 ,:]
            y_test= y_test_total[mini_batch_idx*1:(mini_batch_idx+mini_batch_size)*1 ,:]
        else:
            for ind_in_minibatch in range(mini_batch_size):
                mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                s_test[ind_in_minibatch] = s_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
                y_test[ind_in_minibatch] = y_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
        net.zero_grad()
        out = net(y_test)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s_test, out)
        loss.backward()
        for f in net.parameters():
            f.data.sub_(f.grad.data * lr_alpha)

def test_training_cavia(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH):
    num_dev_test_training_set, _, _ = test_training_set.size()
    if num_dev_test_training_set != 1: # for joint training
        print('something wrong for online')
        test_training_set_unified = torch.zeros(1, num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 4)
        for ind_dev in num_dev_test_training_set:
            test_training_set_unified[0, (ind_dev)*(K_TEST_TR + K_TEST_TE):(ind_dev+1)*(K_TEST_TR + K_TEST_TE), :] = test_training_set[ind_dev, :, :]
    else:
        test_training_set_unified = test_training_set
    if K_TEST_TR < mini_batch_size:
        mini_batch_size = 1
    mini_batch_num = K_TEST_TR/mini_batch_size
    s_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, :2]
    y_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, 2:]
    lr_alpha = lr_testtraining
    ################# cavia ###################
    #### cavia - context para generation ####
    ### always start from 0 for each device & each epochs
    ### curr y shape : [mini_batch_size_meta, 2]
    context_para = torch.zeros(args.num_context_para, requires_grad=True)
    context_para = context_para.to(device)
    for epochs in range(num_epochs_test):
        context_para_tmp = torch.zeros(args.num_context_para, requires_grad=True)
        context_para_tmp.data = context_para.data
        if args.if_save_test_trained_net_per_epoch:
            if not os.path.exists(save_PATH + 'per_epochs' + '/'):
                os.mkdir(save_PATH + 'per_epochs' + '/')
            torch.save(net.state_dict(), save_PATH + 'per_epochs' + '/' + str(epochs))
        perm_idx = torch.randperm(int(K_TEST_TR))
        s_test = torch.zeros(mini_batch_size, 2)
        y_test = torch.zeros(mini_batch_size, 2)
        s_test = s_test.to(device)
        y_test = y_test.to(device)
        if args.if_test_train_no_permute:
            perm_idx_mini_batch_num = torch.randperm(int(mini_batch_num))
            mini_batch_idx_mb_num = perm_idx_mini_batch_num[0]  # SGD
            mini_batch_idx = mini_batch_idx_mb_num * mini_batch_size
            s_test = s_test_total[mini_batch_idx*1:(mini_batch_idx+mini_batch_size)*1 ,:]
            y_test= y_test_total[mini_batch_idx*1:(mini_batch_idx+mini_batch_size)*1 ,:]
        else:
            for ind_in_minibatch in range(mini_batch_size):
                mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                s_test[ind_in_minibatch] = s_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
                y_test[ind_in_minibatch] = y_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]

        y_with_context = torch.zeros(mini_batch_size, 2 + args.num_context_para)
        y_with_context = y_with_context.to(device)
        y_with_context[:, :2] = y_test
        y_with_context[:, 2:] = context_para_tmp  # copying slices seems no prob....
        net.zero_grad()
        out = net(y_with_context)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s_test, out)
        ################# we do not update net.parameters() !!! ###################
        context_para_grad = torch.autograd.grad(loss, context_para_tmp, create_graph=True)
        context_para = context_para - lr_alpha * context_para_grad[0]  # accumulating as with function of theta
    return context_para

## For a certain device
def test_training_benchmark2(args, M, lr_bm2, mini_batch_size_bm2, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, writer, online_mode, device, saved_model_PATH_bm2_intermediate):
    writer_per_dev = writer
    # online_mode: 0: offline, 1: TOE, FTL
    if online_mode == 0:
        num_dev_test_training_set, _, _ = test_training_set.size()
        if num_dev_test_training_set != 1:
            test_training_set_unified = torch.zeros(1, num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 4)
            test_training_set_unified = test_training_set_unified.to(device)
            for ind_dev in range(num_dev_test_training_set):
                test_training_set_unified[0, (ind_dev)*(K_TEST_TR + K_TEST_TE):(ind_dev+1)*(K_TEST_TR + K_TEST_TE), :] = test_training_set[ind_dev, :, :]
        else:
            test_training_set_unified = test_training_set
        assert (num_dev_test_training_set*(K_TEST_TR + K_TEST_TE)) % mini_batch_size_bm2 == 0
        mini_batch_num = (num_dev_test_training_set*(K_TEST_TR + K_TEST_TE))/mini_batch_size_bm2
        s_train_val_tot = test_training_set_unified[0, :num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), :2]
        y_train_val_tot = test_training_set_unified[0, :num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 2:]

        total_number_of_joint_pilots = int(num_dev_test_training_set * (K_TEST_TR + K_TEST_TE))
    elif online_mode == 1:
        total_number_of_joint_pilots, _ = test_training_set.size()
        if mini_batch_size_bm2 > total_number_of_joint_pilots:
            mini_batch_size_bm2 = total_number_of_joint_pilots
        mini_batch_num = total_number_of_joint_pilots / mini_batch_size_bm2
        s_train_val_tot = test_training_set[:, :2]
        y_train_val_tot = test_training_set[:, 2:]
    lr_alpha = lr_bm2
    s_train_val = torch.zeros(mini_batch_size_bm2, 2)
    s_train_val = s_train_val.to(device)
    y_train_val = torch.zeros(mini_batch_size_bm2, 2)
    y_train_val = y_train_val.to(device)
    for epochs in range(num_epochs_test):
        if args.if_bm2_fully_joint:
            perm_idx = torch.randperm(total_number_of_joint_pilots)
            perm_curr = perm_idx[:mini_batch_size_bm2]
            for ind_for_bm2 in range(mini_batch_size_bm2):
                mini_batch_idx = perm_curr[ind_for_bm2]
                s_train_val[ind_for_bm2] = s_train_val_tot[mini_batch_idx:mini_batch_idx+1 ,:]
                y_train_val[ind_for_bm2] = y_train_val_tot[mini_batch_idx:mini_batch_idx+1 ,:]
        else:
            perm_idx = torch.randperm(int(mini_batch_num))
            mini_batch_idx = perm_idx[0]
            s_train_val = s_train_val_tot[
                          mini_batch_idx * mini_batch_size_bm2:(mini_batch_idx + 1) * mini_batch_size_bm2,
                          :]
            y_train_val = y_train_val_tot[
                          mini_batch_idx * mini_batch_size_bm2:(mini_batch_idx + 1) * mini_batch_size_bm2,
                          :]
        net.zero_grad()
        out = net(y_train_val)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s_train_val, out)
        loss.backward()
        for name, f in net.named_parameters():
            if not 'alpha' in name:
                f.data.sub_(f.grad.data * lr_alpha)
        if epochs % 10 == 0:  # 50...
            intermediate_PATH_bm2 = saved_model_PATH_bm2_intermediate + str(epochs + 1)
            torch.save(net.state_dict(), intermediate_PATH_bm2)
def q(x):
    return (1/2 * math.erfc(x/pow(2,0.5)))

def test_test_mul_dev(args, net_for_testtraining, num_test_iter, mini_batch_size, M, lr_testtraining, K_TEST_TR, K_TEST_TE, num_epochs_test, load_PATH, save_PATH, test_training_set, channel_set_genie, non_linearity_set_genie, if_theoretic_on, noise_variance, power, device):
    #################################
    total_error_num = 0
    total_pilot_num = 0
    for ind_dev in range(num_test_iter):
        # repeat for one device
        net = net_for_testtraining
        net.load_state_dict(torch.load(load_PATH))
        if K_TEST_TR > 0:
            if args.if_cavia:
                context_para_updated_for_curr_dev = test_training_cavia(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH)
            else:
                test_training(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH)
        elif K_TEST_TR == 0:
            context_para_updated_for_curr_dev = torch.zeros(args.num_context_para, requires_grad=True)
            context_para_updated_for_curr_dev = context_para_updated_for_curr_dev.to(device)

        #### REAL TEST ####
        net.zero_grad()
        s_test = test_training_set[0, K_TEST_TR:, :2]
        y_test = test_training_set[0, K_TEST_TR:, 2:]

        if args.if_cavia:
            y_with_context_test = torch.zeros(K_TEST_TE, 2 + args.num_context_para)
            y_with_context_test = y_with_context_test.to(device)
            y_with_context_test[:, :2] = y_test
            y_with_context_test[:, 2:] = context_para_updated_for_curr_dev  # copying slices seems no prob....
            out = net(y_with_context_test)
        else:
            out = net(y_test)
        ## loss function
        loss = 0
        error_rate = 0
        uncertainty_loss = 0
        uncertainty_loss, loss, error_rate = cross_entropy_loss_test(uncertainty_loss, loss, error_rate, M, s_test, out) # for reliability check in online scenario
        theoretical_bound = 0
        if if_theoretic_on and M == 5:
            if non_linearity_set_genie[0, 1] == 0:
                theoretical_bound = (3/2) * q((power/pow(noise_variance,0.5))*abs(channel_set_genie[0]))
            else:
                print('working!')
                l_0  = non_linearity_set_genie[0,0]
                l_1 = non_linearity_set_genie[0,1]
                amplitude = 1
                distorted_1 = l_0 * pow(amplitude, 1) / (1 + l_1 * pow(amplitude, 2))
                amplitude = 3
                distorted_3 = l_0 * pow(amplitude, 1) / (1 + l_1 * pow(amplitude, 2))
                d_1 = abs(channel_set_genie[0]) * distorted_1
                d_2 = (abs(channel_set_genie[0])*(distorted_3-distorted_1))/2
                sigma = pow(noise_variance, 0.5)
                Q_1 = 1/2 * math.erfc((1/pow(2,0.5)) * (d_2/sigma))
                Q_2 = 1/2 * math.erfc((1 / pow(2, 0.5)) * (d_1 + d_2 / sigma))
                Q_3 = 1 / 2 * math.erfc((1 / pow(2, 0.5)) * (d_1 / sigma))
                Q_4 = 1 / 2 * math.erfc((1 / pow(2, 0.5)) * (d_1 + 2*d_2 / sigma))
                theoretical_bound = Q_1+Q_2+(1/2)*Q_3+(1/2)*Q_4
        elif if_theoretic_on and M == 4:
            print('we do not consider theoretic values due to hardware non-linearity')
            raise NotImplementedError
        elif if_theoretic_on and M == 16:
            print('we do not consider theoretic values due to hardware non-linearity')
            raise NotImplementedError
        num_error_tmp = error_rate * K_TEST_TE
        total_error_num = total_error_num + num_error_tmp
        total_pilot_num = total_pilot_num + K_TEST_TE
    final_loss = loss
    torch.save(net.state_dict(), save_PATH)
    return uncertainty_loss, final_loss, total_error_num, total_pilot_num, theoretical_bound