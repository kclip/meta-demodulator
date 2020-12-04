from __future__ import print_function
import torch
import numpy
from loss.cross_entropy_loss import cross_entropy_loss
from loss.cross_entropy_loss import cross_entropy_loss_test
from nets.meta_net import meta_net
import math
import os
from numpy.linalg import inv
import scipy.io as sio
from data_gen.data_set import iq_imbalance


def cavia(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, iter_inner_loop, net, net_prime, train_set,
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
    if args.meta_training_query_mode == 10:
        assert M == 16
        total_data_num = s_total.shape[0]
        assert total_data_num % M == 0
        if mini_batch_size_meta > 16:
            one_block_size = mini_batch_size_meta
        else:
            one_block_size = 16

        perm_idx_for_support = torch.randperm(int(total_data_num / one_block_size))[0]
        perm_idx_for_query = torch.randperm(int((total_data_num - one_block_size) // K_TE))[0]  # for query
    else:
        pass
    #### cavia - context para generation ####
    ### always start from 0 for each device & each epochs
    ### curr y shape : [mini_batch_size_meta, 2]
    context_para = torch.zeros(args.num_context_para, requires_grad=True)
    context_para = context_para.to(device)
    ##### keep using this para -> dependency everywhere...
    para_list_from_net_prime = list(map(lambda p: p[0], zip(net_prime.parameters())))
    for iter_idx_inner_loop in range(iter_inner_loop):
        if not iter_idx_inner_loop == iter_inner_loop-1:
            if (args.meta_training_query_mode == 1) or (args.meta_training_query_mode == 2):
                s = s_total[:mini_batch_size_meta, :]
                y = y_total[:mini_batch_size_meta, :]
            elif args.meta_training_query_mode == 10:
                s = s_total[
                    perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta,
                    :]
                y = y_total[
                    perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta,
                    :]
            else:
                raise NotImplementedError
            ##########################################################
            y_with_context = torch.zeros(mini_batch_size_meta, 2 + args.num_context_para)
            y_with_context = y_with_context.to(device)
            y_with_context[:,:2] = y
            y_with_context[:,2:] = context_para # copying slices seems no prob....
            net_meta_intermediate = meta_net(if_relu=args.if_relu)
            out = net_meta_intermediate(y_with_context, para_list_from_net_prime, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
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
            if args.meta_training_query_mode == 1:
                s = s_total[mini_batch_size_meta:, :]
                y = y_total[mini_batch_size_meta:, :]
            elif args.meta_training_query_mode == 2: # use all for query set (for online case)
                s = s_total[:, :]
                y = y_total[:, :]
            elif args.meta_training_query_mode == 10:
                s = torch.cat([s_total[:perm_idx_for_support * one_block_size],
                               s_total[(perm_idx_for_support + 1) * one_block_size:]])
                y = torch.cat([y_total[:perm_idx_for_support * one_block_size],
                               y_total[(perm_idx_for_support + 1) * one_block_size:]])
                s = s[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
                y = y[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
            else:
                raise NotImplementedError

            ##########################################################
            y_with_context = torch.zeros(y.shape[0], 2 + args.num_context_para)
            y_with_context = y_with_context.to(device)
            y_with_context[:, :2] = y
            y_with_context[:, 2:] = context_para  # use not only data but all the gradients contained parameter
            ############ update network's parameter ###########
            out = net_meta_intermediate(y_with_context, para_list_from_net_prime, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
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

    if args.if_see_cos_similarity:
        ### cos sim.
        if ind_d % 2 == 0:  # positive channel
            for f in net.parameters():
                if iter_in_sampled_device_positive_ch == 0:
                    f.total_grad_positive_ch = f.actual_grad_curr
                else:
                    f.total_grad_positive_ch = f.total_grad_positive_ch + f.actual_grad_curr
            iter_in_sampled_device_positive_ch = iter_in_sampled_device_positive_ch + 1
        else:
            for f in net.parameters():
                if iter_in_sampled_device_negative_ch == 0:
                    f.total_grad_negative_ch = f.actual_grad_curr
                else:
                    f.total_grad_negative_ch = f.total_grad_negative_ch + f.actual_grad_curr
            iter_in_sampled_device_negative_ch = iter_in_sampled_device_negative_ch + 1
    else:
        iter_in_sampled_device_negative_ch = None
        iter_in_sampled_device_positive_ch = None

    return iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch,first_loss_curr, second_loss_curr

def fomaml(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, iter_inner_loop, net, net_prime, train_set,
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

    if args.meta_training_query_mode == 10:
        assert M == 16
        total_data_num = s_total.shape[0]
        assert total_data_num % M == 0
        if mini_batch_size_meta > 16:
            one_block_size = mini_batch_size_meta
        else:
            one_block_size = 16

        perm_idx_for_support = torch.randperm(int(total_data_num/one_block_size))[0]
        perm_idx_for_query = torch.randperm(int((total_data_num-one_block_size) // K_TE) )[0]  # for query
    else:
        pass

    for iter_idx_inner_loop in range(iter_inner_loop):
        if not (iter_idx_inner_loop == iter_inner_loop - 1):
            if (args.meta_training_query_mode == 1) or (args.meta_training_query_mode == 2):
                s = s_total[:mini_batch_size_meta, :]
                y = y_total[:mini_batch_size_meta, :]
            elif args.meta_training_query_mode == 10:
                s = s_total[perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta, :]
                y = y_total[perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta, :]
            else:
                raise NotImplementedError
        else:
            if args.meta_training_query_mode == 1:
                s = s_total[mini_batch_size_meta:, :]
                y = y_total[mini_batch_size_meta:, :]
            elif args.meta_training_query_mode == 2:
                s = s_total[:, :]
                y = y_total[:, :]
            elif args.meta_training_query_mode == 10:
                s = torch.cat([s_total[:perm_idx_for_support * one_block_size], s_total[(perm_idx_for_support + 1) * one_block_size:]])
                y = torch.cat([y_total[:perm_idx_for_support * one_block_size], y_total[(perm_idx_for_support + 1) * one_block_size:]])
                s = s[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
                y = y[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
            else:
                raise NotImplementedError
        ###################
        net_prime.zero_grad()
        out = net_prime(y, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)  # inner loop update should be taken only temporarily.. net update should be done after seeing num_dev tasks
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
    ### cos sim.
    if ind_d % 2 == 0: # positive channel
        for f in net.parameters():
            if iter_in_sampled_device_positive_ch == 0:
                f.total_grad_positive_ch = f.actual_grad_curr
            else:
                f.total_grad_positive_ch = f.total_grad_positive_ch + f.actual_grad_curr
        iter_in_sampled_device_positive_ch = iter_in_sampled_device_positive_ch + 1
    else:
        for f in net.parameters():
            if iter_in_sampled_device_negative_ch == 0:
                f.total_grad_negative_ch = f.actual_grad_curr
            else:
                f.total_grad_negative_ch = f.total_grad_negative_ch + f.actual_grad_curr
        iter_in_sampled_device_negative_ch = iter_in_sampled_device_negative_ch + 1

    return iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr

def maml(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, iter_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T):
    if if_online:
        s_total = train_set[:, :2]
        y_total = train_set[:, 2:]
    else:
        s_total = train_set[ind_d, :, :2]
        y_total = train_set[ind_d, :, 2:]
    #print('total data', 's', s_total, 'y', y_total)
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
    if args.meta_training_query_mode == 10:
        assert M == 16
        total_data_num = s_total.shape[0]
        assert total_data_num % M == 0
        if mini_batch_size_meta > 16:
            one_block_size = mini_batch_size_meta
        else:
            one_block_size = 16

        perm_idx_for_support = torch.randperm(int(total_data_num / one_block_size))[0]
        perm_idx_for_query = torch.randperm(int((total_data_num - one_block_size) // K_TE))[0]  # for query
    else:
        pass

    for iter_idx_inner_loop in range(iter_inner_loop):
        if not (iter_idx_inner_loop == iter_inner_loop - 1):
            if (args.meta_training_query_mode == 1) or (args.meta_training_query_mode == 2):
                s = s_total[:mini_batch_size_meta, :]
                y = y_total[:mini_batch_size_meta, :]
            elif args.meta_training_query_mode == 10:
                s = torch.zeros(mini_batch_size_meta, 2)
                y = torch.zeros(mini_batch_size_meta, 2)
                if args.if_rand_sampling_during_meta_training_inner_update: # here
                    dfdf
                    if iter_idx_inner_loop > 0:
                        s_tmp_whole = s_total[
                            perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + one_block_size,
                            :]
                        y_tmp_whole = y_total[
                            perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + one_block_size,
                            :]
                        perm_idx_for_rand_sample = torch.randperm(int(one_block_size))  # random sampling
                        for ind_in_minibatch in range(mini_batch_size_meta):
                            mini_batch_idx = perm_idx_for_rand_sample[ind_in_minibatch]
                            s[ind_in_minibatch] = s_tmp_whole[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y[ind_in_minibatch] = y_tmp_whole[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                    else:
                        s = s_total[
                            perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta,
                            :]
                        y = y_total[
                            perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta,
                            :]
                else:
                    s = s_total[
                        perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta,
                        :]
                    y = y_total[
                        perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta,
                        :]
                #print('iter', iter_idx_inner_loop, 's', s)
            else:
                raise NotImplementedError
        else:
            if args.meta_training_query_mode == 1:
                s = s_total[mini_batch_size_meta:, :]
                y = y_total[mini_batch_size_meta:, :]
            elif args.meta_training_query_mode == 2:
                s = s_total[:, :]
                y = y_total[:, :]
            elif args.meta_training_query_mode == 10:
                s = torch.cat([s_total[:perm_idx_for_support * one_block_size], s_total[(perm_idx_for_support + 1) * one_block_size:]])
                y = torch.cat([y_total[:perm_idx_for_support * one_block_size], y_total[(perm_idx_for_support + 1) * one_block_size:]])
                s = s[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
                y = y[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
            else:
                raise NotImplementedError
        ###################
        net_prime.zero_grad()
        out = net_prime(y, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first) # inner loop update should be taken only temporarily.. net update should be done after seeing num_dev tasks
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
        out = net_prime(y, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
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

    if args.if_see_cos_similarity:
        ### cos sim.
        if ind_d % 2 == 0:  # positive channel
            for f in net.parameters():
                if iter_in_sampled_device_positive_ch == 0:
                    f.total_grad_positive_ch = f.actual_grad_curr
                else:
                    f.total_grad_positive_ch = f.total_grad_positive_ch + f.actual_grad_curr
            iter_in_sampled_device_positive_ch = iter_in_sampled_device_positive_ch + 1
        else:
            for f in net.parameters():
                if iter_in_sampled_device_negative_ch == 0:
                    f.total_grad_negative_ch = f.actual_grad_curr
                else:
                    f.total_grad_negative_ch = f.total_grad_negative_ch + f.actual_grad_curr
            iter_in_sampled_device_negative_ch = iter_in_sampled_device_negative_ch + 1
    else:
        iter_in_sampled_device_positive_ch = None
        iter_in_sampled_device_negative_ch = None



    return iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr

def reptile(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, iter_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T):
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
    if args.meta_training_query_mode == 10:
        assert M == 16
        total_data_num = s_total.shape[0]
        assert total_data_num % M == 0
        if mini_batch_size_meta > 16:
            one_block_size = mini_batch_size_meta
        else:
            one_block_size = 16

        perm_idx_for_support = torch.randperm(int(total_data_num / one_block_size))[0]
        perm_idx_for_query = torch.randperm(int((total_data_num - one_block_size) // K_TE))[0]  # for query
    else:
        pass

    for iter_idx_inner_loop in range(iter_inner_loop):
        if not (iter_idx_inner_loop == iter_inner_loop - 1):
            if (args.meta_training_query_mode == 1) or (args.meta_training_query_mode == 2):
                s = s_total[:mini_batch_size_meta, :]
                y = y_total[:mini_batch_size_meta, :]
            elif args.meta_training_query_mode == 10:
                s = s_total[perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta, :]
                y = y_total[perm_idx_for_support * one_block_size: perm_idx_for_support * one_block_size + mini_batch_size_meta, :]
            else:
                raise NotImplementedError
        else:
            if args.meta_training_query_mode == 1:
                s = s_total[mini_batch_size_meta:, :]
                y = y_total[mini_batch_size_meta:, :]
            elif args.meta_training_query_mode == 2:
                s = s_total[:, :]
                y = y_total[:, :]
            elif args.meta_training_query_mode == 10:
                s = torch.cat([s_total[:perm_idx_for_support * one_block_size], s_total[(perm_idx_for_support + 1) * one_block_size:]])
                y = torch.cat([y_total[:perm_idx_for_support * one_block_size], y_total[(perm_idx_for_support + 1) * one_block_size:]])
                s = s[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
                y = y[perm_idx_for_query * K_TE: perm_idx_for_query * K_TE + K_TE, :]
            else:
                raise NotImplementedError

        ###################
        net_prime.zero_grad()
        out = net_prime(y, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first) # inner loop update should be taken only temporarily.. net update should be done after seeing num_dev tasks
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

    ### cos sim.
    if ind_d % 2 == 0: # positive channel
        for f in net.parameters():
            if iter_in_sampled_device_positive_ch == 0:
                f.total_grad_positive_ch = f.diff_para
            else:
                f.total_grad_positive_ch = f.total_grad_positive_ch + f.diff_para
        iter_in_sampled_device_positive_ch = iter_in_sampled_device_positive_ch + 1
    else:
        for f in net.parameters():
            if iter_in_sampled_device_negative_ch == 0:
                f.total_grad_negative_ch = f.diff_para
            else:
                f.total_grad_negative_ch = f.total_grad_negative_ch + f.diff_para
        iter_in_sampled_device_negative_ch = iter_in_sampled_device_negative_ch + 1


    return iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr

def meta_train(M, num_epochs, num_dev, net, net_prime, train_set, K_TR, K_TE, device, writer_per_num_dev, writer_per_dev_tot, saved_model_PATH_meta_intermediate, lr_alpha_settings, lr_beta_settings, mini_batch_size_meta, if_use_same_seq_both_meta_train_test, sampled_device_num, jac_calc, reptile_inner_loop, if_cycle, args, test_set_array, other_things): #writer, writer_over_num_dev,
    _,len_of_pilots, _ = list(train_set.size())

    # only for args.toy_get_performance_during_meta_training  == True
    if args.toy_get_performance_during_meta_training:
        save_test_result_dict_during_meta_training = {}
        first_loss_array = []
        second_loss_array = []
        test_ser_array = []
        pos_grad_norm_array = []
        neg_grad_norm_array = []
        total_grad_norm_array = []
        cos_sim_array = []
    else:
        pass

    if args.meta_train_with_adam:
        meta_optimiser = torch.optim.Adam(net.parameters(), args.lr_beta_settings)
    else:
        pass

    smoothed_second_loss_array = [] # for stopping criteria
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
        intermediate_PATH = saved_model_PATH_meta_intermediate + 'very_initial' # initial
        torch.save(net.state_dict(), intermediate_PATH)
    lr_beta = args.lr_beta_settings
    loss_for_stopping_criteria = 9999999999

    if args.if_fix_te_num_during_meta_training:
        K_TEST_TE = args.K_TEST_TE
    else:
        K_TEST_TE = 100
    for iter in range(num_epochs):
        if iter % 5000 == 0:
            print('curr iter', iter)
        # get performance per meta-training
        if args.toy_get_performance_during_meta_training:
            if iter % args.toy_check_ser_period_during_meta_training == 0:
                intermediate_PATH = saved_model_PATH_meta_intermediate + str(iter)  # initial
                torch.save(net.state_dict(), intermediate_PATH)
                if not args.if_no_meta:
                    curr_iter_model_PATH_meta = intermediate_PATH
                    curr_iter_model_PATH_benchmark2 = None
                else:
                    assert args.if_no_bm2 == True
                    assert args.if_no_bm1 == True
                    raise NotImplementedError


                num_pilots = args.meta_test_pilot_num
                common_dir = other_things[0]
                net_for_testtraining = other_things[1]
                noise_variance = other_things[2]
                power = other_things[3]

                channel_set_genie_over_meta_test_devs = test_set_array[0]
                non_linearity_set_genie_over_meta_test_devs = test_set_array[1]
                test_training_set_over_meta_test_devs = test_set_array[2]

                if_redo_test = True
                print('curr num te', K_TEST_TE)
                while if_redo_test == True:
                    total_error_rate_bm2_curr_iter, total_error_rate_meta_curr_iter = test_per_dev_during_meta_training(args, iter, num_pilots, channel_set_genie_over_meta_test_devs,
                                                      non_linearity_set_genie_over_meta_test_devs,
                                                      test_training_set_over_meta_test_devs, K_TEST_TE, args.max_pilot_test, device,
                                                      common_dir, num_dev, net_for_testtraining, args.mini_batch_size_test_train, M,
                                                      args.lr_testtraining, curr_iter_model_PATH_benchmark2, curr_iter_model_PATH_meta,
                                                      noise_variance, power)
                    if_redo_test = False
                    print('epoccs', iter, 'ser', total_error_rate_meta_curr_iter)
                    # adjust K_TEST_TE based on current ser

                    if args.if_fix_te_num_during_meta_training:
                        pass
                    else:
                        ########
                        if total_error_rate_meta_curr_iter < 0.0002:
                            if K_TEST_TE < 1000000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_meta_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 1000000
                                if_redo_test = True
                            else:
                                pass
                        elif total_error_rate_meta_curr_iter < 0.002:
                            if K_TEST_TE < 100000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_meta_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 100000
                                if_redo_test = True
                            else:
                                pass
                        elif total_error_rate_meta_curr_iter < 0.02:
                            if K_TEST_TE < 10000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_meta_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 10000
                                if_redo_test = True
                            else:
                                pass
                        elif total_error_rate_meta_curr_iter < 0.2:
                            if K_TEST_TE < 1000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_meta_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 1000
                                if_redo_test = True
                            else:
                                pass
                        else:
                            pass

                        if total_error_rate_meta_curr_iter > 0.3:
                            K_TEST_TE = 100
                        elif total_error_rate_meta_curr_iter > 0.2:
                            if K_TEST_TE > 100:
                                K_TEST_TE = 100
                        elif total_error_rate_meta_curr_iter > 0.02:
                            if K_TEST_TE > 1000:
                                K_TEST_TE = 1000
                        elif total_error_rate_meta_curr_iter > 0.002:
                            if K_TEST_TE > 10000:
                                K_TEST_TE = 10000
                        elif total_error_rate_meta_curr_iter > 0.0002:
                            if K_TEST_TE > 100000:
                                K_TEST_TE = 100000
                        else:
                            pass
                test_ser_array.append(total_error_rate_meta_curr_iter)
                writer_per_num_dev.add_scalar('ser_during_meta_training', total_error_rate_meta_curr_iter, iter)
            else:
                pass
        else:
            pass

        first_loss = 0
        second_loss = 0
        iter_in_sampled_device = 0
        iter_in_sampled_device_positive_ch = 0
        iter_in_sampled_device_negative_ch = 0
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
                raise NotImplementedError # deprecated
            elif jac_calc == 2: # REPTILE
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = reptile(args, iter_in_sampled_device,iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, args.reptile_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 100:  # pytorch maml
                raise NotImplementedError # not dealing in this version, instead we use Hessian-vector approximation at hand
            elif jac_calc == 200: # MAML with approx Hv
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = maml(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, args.maml_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 300: # FOMAML
                iter_in_sampled_device,  iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = fomaml(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, args.maml_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 1001:  # CAVIA
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = cavia(args, iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M,
                                                                                  ind_d, args.maml_inner_loop, net,
                                                                                  net_prime, train_set, s, y, K_TR,
                                                                                  K_TE, mini_batch_size_meta,
                                                                                  writer_per_dev, lr_alpha_settings,
                                                                                  iter, if_cycle, if_online, ind_T,
                                                                                  device)
            first_loss = first_loss + first_loss_curr
            second_loss = second_loss + second_loss_curr
        if iter % 5000 == 0:
            # get cos sim. bet pos grad and neg grad
            if args.if_see_cos_similarity:
                if args.if_cavia:
                    total_size_fc_vector = 244
                else:
                    total_size_fc_vector = 214
                fc_vector_positive = torch.zeros(total_size_fc_vector)
                fc_vector_negative = torch.zeros(total_size_fc_vector)
                ind_f = 0
                for f in net.parameters():
                    if ind_f == 0:
                        if args.if_cavia:
                            first_layer_size = 90
                        else:
                            first_layer_size = 60
                        fc_vector_negative[:first_layer_size] = f.total_grad_negative_ch.reshape(1,first_layer_size).squeeze()
                        fc_vector_positive[:first_layer_size] = f.total_grad_positive_ch.reshape(1, first_layer_size).squeeze()
                    elif ind_f == 1:
                        fc_vector_negative[first_layer_size:first_layer_size+30] = f.total_grad_negative_ch.reshape(1, 30).squeeze()
                        fc_vector_positive[first_layer_size:first_layer_size+30] = f.total_grad_positive_ch.reshape(1, 30).squeeze()
                    elif ind_f == 2:
                        fc_vector_negative[first_layer_size+30:first_layer_size+150] = f.total_grad_negative_ch.reshape(1, 120).squeeze()
                        fc_vector_positive[first_layer_size+30:first_layer_size+150] = f.total_grad_positive_ch.reshape(1, 120).squeeze()
                    else:
                        fc_vector_negative[first_layer_size+150:] = f.total_grad_negative_ch.reshape(1, 4).squeeze()
                        fc_vector_positive[first_layer_size+150:] = f.total_grad_positive_ch.reshape(1, 4).squeeze()
                    ind_f += 1

                cossim = torch.nn.CosineSimilarity(dim=0, eps=1e-6)
                cos_sim_curr = cossim(fc_vector_positive, fc_vector_negative)

                pos_grad_norm = torch.norm(fc_vector_positive)
                neg_grad_norm = torch.norm(fc_vector_negative)
                total_grad_norm = torch.norm(fc_vector_negative+fc_vector_positive)

                writer_per_num_dev.add_scalar('cos similarity', float(cos_sim_curr.data), iter)
                writer_per_num_dev.add_scalar('pos. ch. grad norm', float(pos_grad_norm.data), iter)
                writer_per_num_dev.add_scalar('neg. ch. grad norm', float(neg_grad_norm.data), iter)
                writer_per_num_dev.add_scalar('tot. grad norm', float(total_grad_norm.data), iter)

                pos_grad_norm_array.append(float(pos_grad_norm.data))
                neg_grad_norm_array.append(float(neg_grad_norm.data))
                total_grad_norm_array.append(float(total_grad_norm.data))
                cos_sim_array.append(float(cos_sim_curr.data))
            else:
                pass

        first_loss = first_loss/sampled_device_num
        second_loss = second_loss/sampled_device_num

        if second_loss.data < loss_for_stopping_criteria:
            loss_for_stopping_criteria = second_loss.data
            intermediate_PATH_overwrite_best = saved_model_PATH_meta_intermediate + 'best_model_based_on_meta_training_loss'
            torch.save(net.state_dict(), intermediate_PATH_overwrite_best)
        else:
            pass

        if args.toy_get_performance_during_meta_training:
            first_loss_array.append(float(first_loss.data))
            second_loss_array.append(float(second_loss.data))
        else:
            pass

        if args.meta_train_with_adam:
            meta_optimiser.zero_grad()
            for f in net.parameters():
                f.grad = f.total_grad.data.clone()/sampled_device_num
            meta_optimiser.step()
        else:
            for f in net.parameters():
                #f.total_grad_prev = f.total_grad.data.clone()
                # f.total_grad: sum of gradients considering second derivative: MAML
                # f.total_grad: sum of delta_phi in Reptile
                # get expectation over sampled device num for each iteration
                # in Reptile, we need to consider minibatch in each sampled devices
                f.total_grad = f.total_grad.data.clone()/sampled_device_num # average with num dev.s used in training
                #f.total_grad = f.total_grad # sum over num dev.s used in training
                f.data.sub_(f.total_grad * lr_beta)
        if iter % args.toy_check_ser_period_during_meta_training == 0:
            print('epoch: ', iter, 'first loss: ', float(first_loss.data), 'second loss: ', float(second_loss.data))
            writer_per_num_dev.add_scalar('first_loss', float(first_loss), iter)
            writer_per_num_dev.add_scalar('second_loss', float(second_loss), iter)
            writer_per_num_dev.add_scalar('best loss', float(loss_for_stopping_criteria), iter)
            print('curr best loss', loss_for_stopping_criteria)
            save_test_result_dict_during_meta_training['ser'] = test_ser_array
            save_test_result_dict_during_meta_training['first_loss'] = first_loss_array
            save_test_result_dict_during_meta_training['second_loss'] = second_loss_array

            save_test_result_dict_during_meta_training['pos_grad_norm'] = pos_grad_norm_array
            save_test_result_dict_during_meta_training['neg_grad_norm'] = neg_grad_norm_array
            save_test_result_dict_during_meta_training['total_grad_norm'] = total_grad_norm_array
            save_test_result_dict_during_meta_training['cos_similarity'] = cos_sim_array


            accum_result_during_meta_training = common_dir + 'during_meta_training_result/curr_iter/' + str(
                iter) + '/' + 'test_result.mat'

            os.makedirs(common_dir + 'during_meta_training_result/curr_iter/' + str(
                iter) + '/')

            sio.savemat(accum_result_during_meta_training, save_test_result_dict_during_meta_training)
        if torch.isnan(second_loss.data ):
            print('second loss nan so we stop learning here')
            break


def meta_train_online(M, num_epochs, num_dev, net, net_prime,  total_data_set, used_pilots_per_time, ind_T, K_TR_accum, K_TR, K_TE, device, writer_per_num_dev, writer_per_dev_tot, saved_model_PATH_meta_intermediate, lr_alpha_settings, lr_beta_settings, mini_batch_size_meta, if_use_same_seq_both_meta_train_test, sampled_device_num, jac_calc, reptile_inner_loop, if_cycle, args): #writer, writer_over_num_dev,
    if_online = True
    if args.if_see_meta_inside_tensorboard:
        for ind_d in range(num_dev):
            writer_per_dev = writer_per_dev_tot[ind_d]
    intermediate_PATH = saved_model_PATH_meta_intermediate + str(0) # initial
    torch.save(net.state_dict(), intermediate_PATH)

    if args.meta_train_with_adam:
        meta_optimiser = torch.optim.Adam(net.parameters(), args.lr_beta_settings)
    else:
        pass

    for iter in range(num_epochs):
        first_loss = 0
        second_loss = 0
        iter_in_sampled_device = 0
        iter_in_sampled_device_positive_ch = 0
        iter_in_sampled_device_negative_ch = 0
        num_dev_curr = ind_T # we do not use current data as meta-training data but only for adaptation

        if args.if_use_all_dev_in_one_epoch:
            raise NotImplementedError # deprecated
            cycle_tot = num_dev_curr // sampled_device_num
            if iter % cycle_tot == 0:
                perm_idx_meta_device = torch.randperm(int(num_dev_curr))
            cycle_ind = iter % cycle_tot
        else:
            perm_idx_meta_device = torch.randperm(int(num_dev_curr)) # we need to use until current time's devices
        if sampled_device_num > num_dev_curr:
            sampled_device_num = num_dev_curr #1
        sampled_device_idx_list = torch.zeros([sampled_device_num], dtype=torch.int)
        ind_sampled_device_idx_list = 0
        ind_perm_idx_meta_device = 0
        while ind_sampled_device_idx_list < sampled_device_num:
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
            if jac_calc == 1: # deprecated (Hessian matrix explicitly computation)
                raise NotImplementedError
            elif jac_calc == 2: # reptile
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = reptile(args, iter_in_sampled_device,iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, M, ind_d, args.reptile_inner_loop, net, net_prime, train_set, s, y, K_TR, K_TE, mini_batch_size_meta, writer_per_dev, lr_alpha_settings, iter, if_cycle, if_online, ind_T)
            elif jac_calc == 200:  # multiple inner maml with approx Hv
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = maml(args,
                                                                                                            iter_in_sampled_device,iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch,
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
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = fomaml(args,
                                                                                                              iter_in_sampled_device,iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch,
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
                iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch, first_loss_curr, second_loss_curr = cavia(args,
                                                                                                       iter_in_sampled_device, iter_in_sampled_device_positive_ch, iter_in_sampled_device_negative_ch,
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
        if iter % 100 == 0:
            print('epoch: ', iter, 'first loss: ', first_loss, 'second loss: ', second_loss)

        if args.meta_train_with_adam:
            meta_optimiser.zero_grad()
            for f in net.parameters():
                f.grad = f.total_grad.data.clone()/sampled_device_num
            meta_optimiser.step()
        else:
            for f in net.parameters():
                # f.total_grad: sum of gradients considering second derivative: MAML
                # f.total_grad: sum of delta_phi in Reptile
                # get expectation over sampled device num for each iteration
                # in Reptile, we need to consider minibatch in each sampled devices
                f.total_grad = f.total_grad/sampled_device_num # average with num dev.s used in training
                # sum over num dev.s used in training
                f.data.sub_(f.total_grad * lr_beta_settings)
        if iter % 100 == 0: # 50...
            intermediate_PATH = saved_model_PATH_meta_intermediate + str(iter+1)
            torch.save(net.state_dict(), intermediate_PATH)

## For a certain device
def test_training(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH):
    if args.if_test_training_adam:
        test_training_optimiser = torch.optim.Adam(net.parameters(), lr_testtraining/args.if_test_train_fix_seq_than_adapt_lr)
    else:
        pass

    if args.if_use_stopping_criteria_during_test_training:
        best_loss = 99999999

    num_dev_test_training_set, _, _ = test_training_set.size()
    if num_dev_test_training_set != 1: # for joint training
        print('something wrong for online')
        test_training_set_unified = torch.zeros(1, num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 4)
        for ind_dev in num_dev_test_training_set:
            test_training_set_unified[0, (ind_dev)*(K_TEST_TR + K_TEST_TE):(ind_dev+1)*(K_TEST_TR + K_TEST_TE), :] = test_training_set[ind_dev, :, :]
    else:
        test_training_set_unified = test_training_set
    if K_TEST_TR < mini_batch_size:
        mini_batch_size = K_TEST_TR
    if M == 16:
        if mini_batch_size > 16:
            mini_batch_size = 16


    mini_batch_num = K_TEST_TR/mini_batch_size
    s_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, :2]
    y_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, 2:]
    lr_alpha = lr_testtraining
    for epochs in range(num_epochs_test):
        if args.if_test_training_adam:
            test_training_optimiser.zero_grad()
        if args.if_save_test_trained_net_per_epoch:
            if not os.path.exists(save_PATH + 'per_epochs' + '/'):
                os.mkdir(save_PATH + 'per_epochs' + '/')
            torch.save(net.state_dict(), save_PATH + 'per_epochs' + '/' + str(epochs))
        perm_idx = torch.randperm(int(K_TEST_TR))
        if epochs == 0:
            s_test = torch.zeros(args.mini_batch_size_meta, 2)
            y_test = torch.zeros(args.mini_batch_size_meta, 2)
        else:
            s_test = torch.zeros(mini_batch_size, 2)
            y_test = torch.zeros(mini_batch_size, 2)
        s_test = s_test.to(device)
        y_test = y_test.to(device)

        if args.if_test_train_fix_seq_than_adapt_randomly:
            if epochs == 0:
                s_test = s_test_total[:args.mini_batch_size_meta, :]
                y_test = y_test_total[:args.mini_batch_size_meta, :]
            else: # random sampling
                if mini_batch_size == 16:
                    num_constellations = K_TEST_TR // mini_batch_size
                    remainder_constellations = K_TEST_TR % mini_batch_size  # we have this additional pilots for remainder_constellations pilots
                    for ind_in_minibatch in range(mini_batch_size):
                        if ind_in_minibatch < remainder_constellations:
                            per_ind_num_const = torch.randperm(int(num_constellations+1))[0]
                            mini_batch_idx = ind_in_minibatch + mini_batch_size * per_ind_num_const
                            s_test[ind_in_minibatch] = s_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y_test[ind_in_minibatch] = y_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        else:
                            per_ind_num_const = torch.randperm(int(num_constellations))[0]
                            mini_batch_idx = ind_in_minibatch + mini_batch_size*per_ind_num_const
                            s_test[ind_in_minibatch] = s_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y_test[ind_in_minibatch] = y_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]

                else: # use all
                    for ind_in_minibatch in range(mini_batch_size):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s_test[ind_in_minibatch] = s_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
                        y_test[ind_in_minibatch] = y_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
                lr_alpha = lr_testtraining/args.if_test_train_fix_seq_than_adapt_lr

        elif args.if_test_train_no_permute:
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
        out = net(y_test, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s_test, out)

        if args.if_use_stopping_criteria_during_test_training:
            if loss < best_loss:
                best_loss = float(loss)
                #print('epoch', epochs, 'best loss', best_loss)
                torch.save(net.state_dict(), save_PATH + 'best_model_based_on_test_training_loss')


        loss.backward()
        if epochs % 1000 == 9999:
            print('epochs', epochs, 'loss', float(loss))
        if args.if_test_training_adam:
            if args.if_adam_after_sgd:
                if epochs == 0:
                    for f in net.parameters():
                        f.data.sub_(f.grad.data * lr_alpha)
                else:
                    test_training_optimiser.step()
            else:
                test_training_optimiser.step()
        else:
            for f in net.parameters():
                f.data.sub_(f.grad.data * lr_alpha)

def test_training_cavia(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH):
    num_dev_test_training_set, _, _ = test_training_set.size()
    if args.if_use_stopping_criteria_during_test_training:
        best_loss = 99999999
        best_epoch_ind = -99 # dummy
        context_para_list = torch.zeros(num_epochs_test,
                                        args.num_context_para)  # save all and select lowest training loss
    if num_dev_test_training_set != 1: # for joint training
        print('something wrong for online')
        test_training_set_unified = torch.zeros(1, num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 4)
        for ind_dev in num_dev_test_training_set:
            test_training_set_unified[0, (ind_dev)*(K_TEST_TR + K_TEST_TE):(ind_dev+1)*(K_TEST_TR + K_TEST_TE), :] = test_training_set[ind_dev, :, :]
    else:
        test_training_set_unified = test_training_set
    if K_TEST_TR < mini_batch_size:
        mini_batch_size = K_TEST_TR #1
    if M == 16:
        if mini_batch_size > 16:
            mini_batch_size = 16

    mini_batch_num = K_TEST_TR/mini_batch_size
    s_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, :2]
    y_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, 2:]
    lr_alpha = lr_testtraining
    ################# cavia ###################
    #### cavia - context para generation ####
    ### always start from 0 for each device & each epochs
    ### curr y shape : [mini_batch_size_meta, 2]

    if args.if_test_training_adam:
        m = torch.zeros(num_epochs_test, args.num_context_para)
        v = torch.zeros(num_epochs_test, args.num_context_para)
        beta_1 = 0.9
        beta_2 = 0.999
        eps_adam = 1e-08




    context_para = torch.zeros(args.num_context_para, requires_grad=True)
    context_para = context_para.to(device)
    for epochs in range(num_epochs_test):
        #print(context_para_list)
        context_para_tmp = torch.zeros(args.num_context_para, requires_grad=True)
        context_para_tmp.data = context_para.data
        if args.if_save_test_trained_net_per_epoch:
            if not os.path.exists(save_PATH + 'per_epochs' + '/'):
                os.mkdir(save_PATH + 'per_epochs' + '/')
            torch.save(net.state_dict(), save_PATH + 'per_epochs' + '/' + str(epochs))
        perm_idx = torch.randperm(int(K_TEST_TR))
        if epochs == 0:
            s_test = torch.zeros(args.mini_batch_size_meta, 2)
            y_test = torch.zeros(args.mini_batch_size_meta, 2)
        else:
            s_test = torch.zeros(mini_batch_size, 2)
            y_test = torch.zeros(mini_batch_size, 2)
        s_test = s_test.to(device)
        y_test = y_test.to(device)

        if args.if_test_train_fix_seq_than_adapt_randomly:
            if epochs == 0:
                s_test = s_test_total[:args.mini_batch_size_meta, :]
                y_test = y_test_total[:args.mini_batch_size_meta, :]
            else: # random sampling
                if mini_batch_size == 16:
                    num_constellations = K_TEST_TR // mini_batch_size # we have this number of set for 16 pilots
                    remainder_constellations = K_TEST_TR % mini_batch_size # we have this additional pilots for remainder_constellations pilots
                    for ind_in_minibatch in range(mini_batch_size):
                        if ind_in_minibatch < remainder_constellations:
                            per_ind_num_const = torch.randperm(int(num_constellations+1))[0]
                            mini_batch_idx = ind_in_minibatch + mini_batch_size * per_ind_num_const
                            s_test[ind_in_minibatch] = s_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y_test[ind_in_minibatch] = y_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                        else:
                            per_ind_num_const = torch.randperm(int(num_constellations))[0]
                            mini_batch_idx = ind_in_minibatch + mini_batch_size*per_ind_num_const
                            s_test[ind_in_minibatch] = s_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]
                            y_test[ind_in_minibatch] = y_test_total[mini_batch_idx * 1:(mini_batch_idx + 1) * 1, :]

                else: # use all
                    for ind_in_minibatch in range(mini_batch_size):
                        mini_batch_idx = perm_idx[ind_in_minibatch]  # SGD
                        s_test[ind_in_minibatch] = s_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
                        y_test[ind_in_minibatch] = y_test_total[mini_batch_idx*1:(mini_batch_idx+1)*1 ,:]
                lr_alpha = lr_testtraining/args.if_test_train_fix_seq_than_adapt_lr
        elif args.if_test_train_no_permute:
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

        if epochs == 0:
            y_with_context = torch.zeros(min(K_TEST_TR, args.mini_batch_size_meta), 2 + args.num_context_para)
        else:
            y_with_context = torch.zeros(mini_batch_size, 2 + args.num_context_para)
        y_with_context = y_with_context.to(device)
        y_with_context[:, :2] = y_test
        y_with_context[:, 2:] = context_para_tmp  # copying slices seems no prob....
        net.zero_grad()
        out = net(y_with_context, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s_test, out)
        if args.if_use_stopping_criteria_during_test_training:
            context_para_list[epochs, :] = context_para.data
            if loss < best_loss:
                best_loss = float(loss)
                best_epoch_ind = epochs
        context_para_grad = torch.autograd.grad(loss, context_para_tmp, create_graph=True)
        if args.if_test_training_adam:
            g_t = context_para_grad.data.clone()
            m = beta_1*m+ (1-beta_1)*g_t
            v = beta_2*v + (1-beta_2)*g_t*g_t
            m_hat = m/(1-pow(beta_1,(epochs+1)))
            v_hat = v/(1-pow(beta_2,(epochs+1)))
            context_para = context_para - lr_alpha * m_hat/(pow(v_hat,0.5)+eps_adam)
        else:
            ################# we do not update net.parameters() !!! ###################
            context_para = context_para - lr_alpha * context_para_grad[0]  # accumulating as with function of theta
    if args.if_use_stopping_criteria_during_test_training:
        return context_para_list[best_epoch_ind, :]
    else:
        return context_para



### mmse channel estimator
## For a certain device
def mmse_channel_estimation(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH):
    num_dev_test_training_set, _, _ = test_training_set.size()
    assert num_dev_test_training_set == 1
    test_training_set_unified = test_training_set
    s_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, :2]
    y_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, 2:]
    ### A gen.
    A = numpy.zeros((K_TEST_TR, 1), dtype=complex)
    s_test_total_to_cpu = s_test_total.cpu().numpy()
    y_test_total_to_cpu = y_test_total.cpu().numpy()
    for ind_pilot in range(K_TEST_TR):
        A[ind_pilot, 0] = s_test_total_to_cpu[ind_pilot, 0] + s_test_total_to_cpu[ind_pilot, 1] * 1j

    received_signal_total = numpy.zeros((K_TEST_TR, 1), dtype=complex)
    for ind_pilot in range(K_TEST_TR):
        received_signal_total[ind_pilot, 0] = y_test_total_to_cpu[ind_pilot, 0] + y_test_total_to_cpu[ind_pilot, 1] * 1j

    I = numpy.zeros((K_TEST_TR, K_TEST_TR), dtype=complex)
    for i in range(K_TEST_TR):
        I[i, i] = 1
    ## mmse estimator
    ch_var_complex = 1
    power = args.power
    noise_var_complex = (pow(power, 2) * (10)) / (pow(10, args.SNR_db / 10))

    cov_h_y = ch_var_complex * numpy.conj(numpy.transpose(A))

    cov_y = ch_var_complex * numpy.matmul(A, numpy.conj(numpy.transpose(A))) + noise_var_complex * I

    W = numpy.matmul(cov_h_y, inv(cov_y))
    estiamated_channel = numpy.matmul(W, received_signal_total)

    estiamated_channel_tensor = torch.FloatTensor([0,0])
    estiamated_channel_tensor[0] = numpy.real(estiamated_channel)[0][0]
    estiamated_channel_tensor[1] = numpy.imag(estiamated_channel)[0][0]
    print('est ch', estiamated_channel_tensor)
    return estiamated_channel_tensor

### mmse channel estimator
## For a certain device
def mmse_channel_estimation_effective_channel_with_iq(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH):
    num_dev_test_training_set, _, _ = test_training_set.size()

    assert num_dev_test_training_set == 1
    test_training_set_unified = test_training_set

    s_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, :2]
    y_test_total = test_training_set_unified[0, :num_dev_test_training_set * K_TEST_TR, 2:]
    ### A gen.
    #A = numpy.zeros((K_TEST_TR, 1), dtype=complex)
    A = numpy.zeros((K_TEST_TR, 2))

    s_test_total_to_cpu = s_test_total.cpu().numpy()
    y_test_total_to_cpu = y_test_total.cpu().numpy()

    I = numpy.zeros((K_TEST_TR, K_TEST_TR))

    for i in range(K_TEST_TR):
        I[i, i] = 1

    A = s_test_total_to_cpu

    power = args.power
    noise_var_complex = (pow(power, 2) * (10)) / (pow(10, args.SNR_db / 10))
    noise_var = noise_var_complex/2

    y_1 = y_test_total_to_cpu[:,0]
    y_2 = y_test_total_to_cpu[:,1]

    C_z = noise_var * I

    inv_mat_tmp = inv(numpy.matmul(A, numpy.transpose(A)) + C_z)
    W = numpy.matmul(numpy.transpose(A), inv_mat_tmp)
    est_channel_1 = numpy.matmul(W, y_1)
    est_channel_2 = numpy.matmul(W, y_2)

    estiamated_channel_tensor = torch.zeros(2,2)
    estiamated_channel_tensor[:,0] = torch.from_numpy(est_channel_1)
    estiamated_channel_tensor[:, 1] = torch.from_numpy(est_channel_2)


    print('est ch', torch.transpose(estiamated_channel_tensor, 0, 1))
    return torch.transpose(estiamated_channel_tensor, 0, 1)


## For a certain device
def test_training_benchmark2(args, M, lr_bm2, mini_batch_size_bm2, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, writer, online_mode, device, saved_model_PATH_bm2_intermediate, other_things, test_set_array):
    writer_per_dev = writer
    # online_mode: 0: offline, 1: TOE, FTL

    if args.meta_train_with_adam:
        joint_optimiser = torch.optim.Adam(net.parameters(), lr_bm2)
    else:
        pass

    if args.toy_get_performance_during_meta_training:
        save_test_result_dict_during_joint_training = {}
        test_ser_array = []
        loss_array = []


    if online_mode == 0:
        num_dev_test_training_set, _, _ = test_training_set.size()
        if num_dev_test_training_set != 1:
            test_training_set_unified = torch.zeros(1, num_dev_test_training_set * (K_TEST_TR + K_TEST_TE), 4)
            test_training_set_unified = test_training_set_unified.to(device)
            for ind_dev in range(num_dev_test_training_set):
                test_training_set_unified[0, (ind_dev)*(K_TEST_TR + K_TEST_TE):(ind_dev+1)*(K_TEST_TR + K_TEST_TE), :] = test_training_set[ind_dev, :, :]
        else:
            test_training_set_unified = test_training_set

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

    loss_for_best = 9999999
    if args.if_fix_te_num_during_meta_training:
        K_TEST_TE = args.K_TEST_TE
    else:
        K_TEST_TE = 100

    for epochs in range(num_epochs_test):
        if epochs % 10000 == 0:
            print('curr iter', epochs)
        if args.meta_train_with_adam:
            joint_optimiser.zero_grad()
        else:
            pass
        # get test loss during meta-training
        # get performance per meta-training
        if args.toy_get_performance_during_meta_training:
            if epochs % args.toy_check_ser_period_during_meta_training == 0:
                intermediate_PATH = saved_model_PATH_bm2_intermediate + str(epochs)  # initial
                torch.save(net.state_dict(), intermediate_PATH)
                if not args.if_no_bm2:
                    curr_iter_model_PATH_meta = None
                    curr_iter_model_PATH_benchmark2 = intermediate_PATH
                else:
                    assert args.if_no_bm2 == True
                    assert args.if_no_meta == True
                    raise NotImplementedError

                num_pilots = args.meta_test_pilot_num
                common_dir = other_things[0]
                net_for_testtraining = other_things[1]
                noise_variance = other_things[2]
                power = other_things[3]

                channel_set_genie_over_meta_test_devs = test_set_array[0]
                non_linearity_set_genie_over_meta_test_devs = test_set_array[1]
                test_training_set_over_meta_test_devs = test_set_array[2]

                if_redo_test = True
                print('curr num te', K_TEST_TE)
                while if_redo_test == True:
                    total_error_rate_bm2_curr_iter, total_error_rate_meta_curr_iter = test_per_dev_during_meta_training(
                        args, epochs, num_pilots, channel_set_genie_over_meta_test_devs,
                        non_linearity_set_genie_over_meta_test_devs,
                        test_training_set_over_meta_test_devs, K_TEST_TE, args.max_pilot_test, device,
                        common_dir, args.num_dev, net_for_testtraining, args.mini_batch_size_test_train, M,
                        args.lr_testtraining, curr_iter_model_PATH_benchmark2, curr_iter_model_PATH_meta,
                        noise_variance, power)
                    if_redo_test = False


                    #####
                    ########
                    if args.if_fix_te_num_during_meta_training:
                        pass
                    else:
                        if total_error_rate_bm2_curr_iter < 0.0002:
                            if K_TEST_TE < 1000000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_bm2_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 1000000
                                if_redo_test = True
                            else:
                                pass
                        elif total_error_rate_bm2_curr_iter < 0.002:
                            if K_TEST_TE < 100000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_bm2_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 100000
                                if_redo_test = True
                            else:
                                pass
                        elif total_error_rate_bm2_curr_iter < 0.02:
                            if K_TEST_TE < 10000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_bm2_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 10000
                                if_redo_test = True
                            else:
                                pass
                        elif total_error_rate_bm2_curr_iter < 0.2:
                            if K_TEST_TE < 1000:
                                print('curr num test', K_TEST_TE, 'curr ser', total_error_rate_bm2_curr_iter,
                                      'so redo exp')
                                K_TEST_TE = 1000
                                if_redo_test = True
                            else:
                                pass
                        else:
                            pass

                        if total_error_rate_bm2_curr_iter > 0.3:
                            K_TEST_TE = 100
                        elif total_error_rate_bm2_curr_iter > 0.2:
                            if K_TEST_TE > 100:
                                K_TEST_TE = 100
                        elif total_error_rate_bm2_curr_iter > 0.02:
                            if K_TEST_TE > 1000:
                                K_TEST_TE = 1000
                        elif total_error_rate_bm2_curr_iter > 0.002:
                            if K_TEST_TE > 10000:
                                K_TEST_TE = 10000
                        elif total_error_rate_bm2_curr_iter > 0.0002:
                            if K_TEST_TE > 100000:
                                K_TEST_TE = 100000
                        else:
                            pass

                test_ser_array.append(total_error_rate_bm2_curr_iter)
                writer_per_dev.add_scalar('ser_during_joint_training', total_error_rate_bm2_curr_iter, epochs)
            else:
                pass
        else:
            pass







        if args.if_bm2_fully_joint:
            perm_idx = torch.randperm(total_number_of_joint_pilots)
            perm_curr = perm_idx[:mini_batch_size_bm2]
            for ind_for_bm2 in range(mini_batch_size_bm2):
                mini_batch_idx = perm_curr[ind_for_bm2]
                s_train_val[ind_for_bm2] = s_train_val_tot[mini_batch_idx:mini_batch_idx+1 ,:]
                y_train_val[ind_for_bm2] = y_train_val_tot[mini_batch_idx:mini_batch_idx+1 ,:]
        else:
            assert (num_dev_test_training_set*(K_TEST_TR + K_TEST_TE)) % mini_batch_size_bm2 == 0
            perm_idx = torch.randperm(int(mini_batch_num))
            mini_batch_idx = perm_idx[0]
            s_train_val = s_train_val_tot[
                          mini_batch_idx * mini_batch_size_bm2:(mini_batch_idx + 1) * mini_batch_size_bm2,
                          :]
            y_train_val = y_train_val_tot[
                          mini_batch_idx * mini_batch_size_bm2:(mini_batch_idx + 1) * mini_batch_size_bm2,
                          :]
        net.zero_grad()
        out = net(y_train_val, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
        loss = 0
        error_rate = 0
        loss, error_rate = cross_entropy_loss(loss, error_rate, M, s_train_val, out)

        if args.toy_get_performance_during_meta_training:
            loss_array.append(float(loss))
        if args.toy_get_performance_during_meta_training:
            if epochs % args.toy_check_ser_period_during_meta_training == 0:
                writer_per_dev.add_scalar('loss', float(loss), epochs)
                print('epochs: ', epochs, 'bm2 loss', float(loss))
                print('curr best loss', loss_for_best)
                save_test_result_dict_during_joint_training['ser'] = test_ser_array
                save_test_result_dict_during_joint_training['second_loss'] = loss_array

                accum_result_during_joint_training = common_dir + 'during_joint_training_result/curr_iter/' + str(
                    epochs) + '/' + 'test_result.mat'

                os.makedirs(common_dir + 'during_joint_training_result/curr_iter/' + str(
                    epochs) + '/')

                sio.savemat(accum_result_during_joint_training, save_test_result_dict_during_joint_training)


        if loss < loss_for_best:
            loss_for_best = loss
            intermediate_PATH_bm2_best_training_loss = saved_model_PATH_bm2_intermediate + 'best_model_based_on_meta_training_loss'
            torch.save(net.state_dict(), intermediate_PATH_bm2_best_training_loss)
        loss.backward()

        if args.meta_train_with_adam:
            joint_optimiser.step()
        else:
            for name, f in net.named_parameters():
                if not 'alpha' in name:
                    f.data.sub_(f.grad.data * lr_alpha)

def q(x):
    return (1/2 * math.erfc(x/pow(2,0.5)))

def test_test_mul_dev(args, net_for_testtraining, num_test_iter, mini_batch_size, M, lr_testtraining, K_TEST_TR, K_TEST_TE, num_epochs_test, load_PATH, save_PATH, test_training_set, if_theoretic_on, noise_variance, power, device, genie_set):
    #################################
    total_error_num = 0
    total_pilot_num = 0

    if (if_theoretic_on == True) and (M == 16):
        print('MMSE channel estimator with maximum likelihood')
        if_conven_commun = True
    elif (if_theoretic_on == True) and (M == 5):
        print('sign estimator with decision region')
        if_conven_commun = True
        args.num_epochs_test_bm1 = 0
    else:
        if_conven_commun = False

    for ind_dev in range(num_test_iter):
        # repeat for one device
        net = net_for_testtraining
        net.load_state_dict(torch.load(load_PATH))
        if K_TEST_TR > 0:
            if args.if_cavia:
                context_para_updated_for_curr_dev = test_training_cavia(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH)
            elif if_conven_commun:
                if M == 16:
                    if args.if_perfect_csi:
                        if args.if_awgn:
                            est_h = 1
                        else:
                            est_h = genie_set[0][0]
                            print(genie_set[0])
                            print(genie_set[0][0])
                    else:
                        if args.mmse_considering_iq:
                            est_h = mmse_channel_estimation_effective_channel_with_iq(args, M, lr_testtraining, mini_batch_size, net,
                                                            test_training_set, K_TEST_TR, K_TEST_TE,
                                                            num_epochs_test, device, save_PATH)
                        else:
                            est_h = mmse_channel_estimation(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE,
                                          num_epochs_test, device, save_PATH)
                else:
                    pass
            else:
                test_training(args, M, lr_testtraining, mini_batch_size, net, test_training_set, K_TEST_TR, K_TEST_TE, num_epochs_test, device, save_PATH)
                if args.if_use_stopping_criteria_during_test_training:
                    best_model_based_on_training_loss = save_PATH + 'best_model_based_on_test_training_loss'
                    net.load_state_dict(torch.load(best_model_based_on_training_loss))
                else:
                    pass
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
            out = net(y_with_context_test, args.if_deconv,  args.if_RTN, args.if_cavia, args.if_tanh_first)
        else:
            out = net(y_test, args.if_deconv, args.if_RTN, args.if_cavia, args.if_tanh_first)
        ## loss function
        if not if_conven_commun:
            loss = 0
            error_rate = 0
            uncertainty_loss = 0
            uncertainty_loss, loss, error_rate = cross_entropy_loss_test(uncertainty_loss, loss, error_rate, M, s_test, out) # for reliability check in online scenario
            theoretical_bound = 0

            num_error_tmp = error_rate * K_TEST_TE
            total_error_num = total_error_num + num_error_tmp
            total_pilot_num = total_pilot_num + K_TEST_TE

        else:
            loss = 0
            error_rate = 0
            uncertainty_loss = 0
            theoretical_bound = 0
        if if_theoretic_on and M == 5:
            assert K_TEST_TR == 1
            out_conv = torch.zeros(y_test.shape[0], M)
            s_tr = test_training_set[0, :K_TEST_TR, :2]
            y_tr = test_training_set[0, :K_TEST_TR, 2:]
            if y_tr[0,0]/s_tr[0,0] > 0:
                h_est = 1
            else:
                h_est = -1
            print(y_tr[0,0]/s_tr[0,0], h_est)

            success = 0

            for ind_mb in range(y_test.shape[0]):
                #print(ind_mb)
                ch_conv_y = h_est * y_test[ind_mb,0]
                if ch_conv_y < -2:
                    if s_test[ind_mb, 0] == -3:
                        success += 1
                    else:
                        pass
                elif -2 <= ch_conv_y < 0:
                    if s_test[ind_mb,0] == -1:
                        success += 1
                    else:
                        pass
                elif 0 <= ch_conv_y < 2:
                    if s_test[ind_mb, 0] == 1:
                        success += 1
                    else:
                        pass
                else:
                    if s_test[ind_mb, 0] == 3:
                        success += 1
                    else:
                        pass
            total_error_num = y_test.shape[0] - success
            total_pilot_num = y_test.shape[0]
            print('curr error num', total_error_num)
            print('total pilot num', total_pilot_num)
            theoretical_bound = 1 - success/y_test.shape[0]
            print('opt', theoretical_bound)



        elif if_theoretic_on and M == 4:
            print('we do not consider theoretic values due to hardware non-linearity')
            raise NotImplementedError
        elif if_theoretic_on and M == 16:
            print('optimal receiver with mmse channel estimator with maximum likelihood')
            out_conv = torch.zeros(y_test.shape[0], M)
            for ind_mb in range(y_test.shape[0]):
                for symb_ind in range(M):
                    cand_y = torch.FloatTensor([0, 0])
                    ####
                    if symb_ind % 16 == 0:
                        cand_s = torch.FloatTensor([-3, -3])
                    elif symb_ind % 16 == 1:
                        cand_s = torch.FloatTensor([-3, 1])
                    elif symb_ind % 16 == 2:
                        cand_s = torch.FloatTensor([1, 1])
                    elif symb_ind % 16 == 3:
                        cand_s = torch.FloatTensor([1, -3])

                    elif symb_ind % 16 == 4:
                        cand_s = torch.FloatTensor([-3, 3])
                    elif symb_ind % 16 == 5:
                        cand_s = torch.FloatTensor([3, 1])
                    elif symb_ind % 16 == 6:
                        cand_s = torch.FloatTensor([1, -1])
                    elif symb_ind % 16 == 7:
                        cand_s = torch.FloatTensor([-1, -3])

                    elif symb_ind % 16 == 8:
                        cand_s = torch.FloatTensor([3, 3])
                    elif symb_ind % 16 == 9:
                        cand_s = torch.FloatTensor([3, -1])
                    elif symb_ind % 16 == 10:
                        cand_s = torch.FloatTensor([-1, -1])
                    elif symb_ind % 16 == 11:
                        cand_s = torch.FloatTensor([-1, 3])

                    elif symb_ind % 16 == 12:
                        cand_s = torch.FloatTensor([3, -3])
                    elif symb_ind % 16 == 13:
                        cand_s = torch.FloatTensor([-3, -1])
                    elif symb_ind % 16 == 14:
                        cand_s = torch.FloatTensor([-1, 1])
                    elif symb_ind % 16 == 15:
                        cand_s = torch.FloatTensor([1, 3])

                    if args.if_perfect_iq_imbalance_knowledge:
                        #cand_s[0] = cand_s[0] * genie_set[1][0,0]
                        cand_s = iq_imbalance(cand_s, genie_set[1][0,0], genie_set[1][0,1], device)
                        if args.if_awgn:
                            cand_y[0] = cand_s[0]
                            cand_y[1] = cand_s[1]
                        else:
                            cand_y[0] = est_h[0] * cand_s[0] - est_h[1] * cand_s[1]
                            cand_y[1] = est_h[0] * cand_s[1] + est_h[1] * cand_s[0]
                    else:
                        if args.if_awgn:
                            cand_y[0] = cand_s[0]
                            cand_y[1] = cand_s[1]
                        else:
                            if args.mmse_considering_iq:
                                cand_y = torch.matmul(est_h, cand_s)
                            else:
                                cand_y[0] = est_h[0] * cand_s[0] - est_h[1] * cand_s[1]
                                cand_y[1] = est_h[0] * cand_s[1] + est_h[1] * cand_s[0]
                        #print(y_test[ind_mb], cand_y)

                    out_conv[ind_mb, symb_ind] = 1 / torch.norm(
                        y_test[ind_mb] - cand_y.to(device))  # since norm smaller, better
            loss_conv = 0
            error_rate_conv = 0
            uncertainty_loss_conv = 0
            uncertainty_loss_conv, loss_conv, theoretical_bound = cross_entropy_loss_test(uncertainty_loss_conv, loss_conv, error_rate_conv, M, s_test,
                                                                         out_conv)
            total_pilot_num = y_test.shape[0]
            total_error_num = total_pilot_num * theoretical_bound


    final_loss = float(loss)
    torch.save(net.state_dict(), save_PATH)
    return float(uncertainty_loss), float(final_loss), float(total_error_num), float(total_pilot_num), float(theoretical_bound)


def test_per_dev_during_meta_training(args, iter, num_pilots, channel_set_genie_over_meta_test_devs, non_linearity_set_genie_over_meta_test_devs, test_training_set_over_meta_test_devs, K_TEST_TE, max_num_pilots, device, common_dir, num_dev, net_for_testtraining, mini_batch_size, M, lr_testtraining, saved_model_PATH_benchmark2, saved_model_PATH_meta, noise_variance, power):
    #print('num_pilots used for meta-test during meta-training: ', num_pilots)
    num_test_iter = 1  # iter for one device (fix channel gain, dist. coefficients, but generate data set each time)
    # currently no iteration for a one device
    genie_set = None # no need for machine learning
    K_TEST_TR = num_pilots
    total_error_num_bm2 = 0
    total_pilot_num_bm2 = 0
    total_error_num_meta = 0
    total_pilot_num_meta = 0

    assert args.if_no_bm1 == True

    total_error_rate_bm2 = None
    total_error_rate_meta = None


    for i in range(args.num_devices_for_test):
        print('curr dev for test: ', i)
        channel_set_genie = channel_set_genie_over_meta_test_devs[i]
        non_linearity_set_genie = non_linearity_set_genie_over_meta_test_devs[i]
        tmp_test_training_set = test_training_set_over_meta_test_devs[i]
        tmp_test_training_set_only_train = tmp_test_training_set[:, :K_TEST_TR, :]
        tmp_test_training_set_only_val = tmp_test_training_set[:, max_num_pilots:max_num_pilots + K_TEST_TE, :]
        test_training_set = torch.zeros(1, K_TEST_TR + K_TEST_TE, 4)
        test_training_set = test_training_set.to(device)
        test_training_set[:, :K_TEST_TR, :] = tmp_test_training_set_only_train
        test_training_set[:, K_TEST_TR:, :] = tmp_test_training_set_only_val
        dir_benchmark2 = common_dir + 'saved_model/' + 'during_meta_training/' + 'after_meta_testing_set/' + 'benchmark2/' + 'num pilots:' + str(
            num_pilots) + '/' + 'num_dev:' + str(
            num_dev) + '/' + 'curr_iter/' + str(iter) + '/ind_dev_test_train:' + str(i) + '/'
        dir_meta = common_dir + 'saved_model/' + 'during_meta_training/' + 'after_meta_testing_set/' + 'meta/' + 'num pilots:' + str(
            num_pilots) + '/' + 'num_dev:' + str(
            num_dev) + '/' + 'curr_iter/' + str(iter) + '/ind_dev_test_train:' + str(i) + '/'
        if not os.path.exists(dir_benchmark2):
            os.makedirs(dir_benchmark2)
        if not os.path.exists(dir_meta):
            os.makedirs(dir_meta)
        save_PATH_benchmark2 = dir_benchmark2 + 'joint_trained_net'
        save_PATH_meta = dir_meta + 'meta_net'

        ###################################
        if not args.if_no_bm2:
            num_epochs_test = args.num_epochs_test_bm2
            uncertainty_loss_tmp_bm2, final_loss_tmp_bm2, total_error_num_tmp_bm2, total_pilot_num_tmp_bm2, theoretical_bound_bm2 = test_test_mul_dev(
                args, net_for_testtraining,
                num_test_iter, mini_batch_size, M, lr_testtraining,
                K_TEST_TR,
                K_TEST_TE, num_epochs_test,
                saved_model_PATH_benchmark2,
                save_PATH_benchmark2,
                test_training_set, False, noise_variance, power, device, genie_set)
        if not args.if_no_meta:
            num_epochs_test_meta = args.num_epochs_test_meta
            uncertainty_loss_tmp_meta, final_loss_tmp_meta, total_error_num_tmp_meta, total_pilot_num_tmp_meta, theoretical_bound_meta = test_test_mul_dev(
                args, net_for_testtraining,
                num_test_iter, mini_batch_size, M, lr_testtraining,
                K_TEST_TR,
                K_TEST_TE, num_epochs_test_meta,
                saved_model_PATH_meta,
                save_PATH_meta,
                test_training_set, False, noise_variance,
                power, device, genie_set)
        if not args.if_no_bm2:
            total_error_num_bm2 = total_error_num_bm2 + total_error_num_tmp_bm2
            total_pilot_num_bm2 = total_pilot_num_bm2 + total_pilot_num_tmp_bm2
        if not args.if_no_meta:
            total_error_num_meta = total_error_num_meta + total_error_num_tmp_meta
            total_pilot_num_meta = total_pilot_num_meta + total_pilot_num_tmp_meta
    ######
    if not args.if_no_bm2:
        total_error_rate_bm2 = total_error_num_bm2 / total_pilot_num_bm2
    if not args.if_no_meta:
        total_error_rate_meta = total_error_num_meta / total_pilot_num_meta

    return total_error_rate_bm2, total_error_rate_meta




