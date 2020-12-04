from __future__ import print_function
import torch

def generating_symbol(M, device_for_data):
    device = device_for_data
    Bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))  # equal prob.
    if M == 2:  # BPSK
        symb = Bern.sample()
        symb = symb.to(device)
        if symb == 0:
            symb = -1
        symb = torch.FloatTensor([symb, 0])
    elif M == 4:  # 4-QAM
        symb1 = Bern.sample()
        symb2 = Bern.sample()
        symb1 = symb1.to(device)
        symb2 = symb2.to(device)
        if symb1 == 0:
            symb1 = -1
        if symb2 == 0:
            symb2 = -1
        symb = torch.FloatTensor([symb1, symb2])
    elif M == 5:  # 4-PAM
        symb1 = Bern.sample()
        symb2 = Bern.sample()
        symb1 = symb1.to(device)
        symb2 = symb2.to(device)
        if symb1 == 0:
            symb1 = -1
        if symb2 == 0:
            symb2 = -1
        symb1 = symb1 * 2
        symb = torch.FloatTensor([symb1 + symb2, 0])  # -3, -1,, 1, 3 : 4-PAM
    elif M == 16:  # 16-QAM
        symb1 = Bern.sample()
        symb2 = Bern.sample()
        symb3 = Bern.sample()
        symb4 = Bern.sample()
        symb1 = symb1.to(device)
        symb2 = symb2.to(device)
        symb3 = symb3.to(device)
        symb4 = symb4.to(device)
        if symb1 == 0:
            symb1 = -1
        if symb2 == 0:
            symb2 = -1
        if symb3 == 0:
            symb3 = -1
        if symb4 == 0:
            symb4 = -1
        symb3 = symb3 * 2
        symb4 = symb4 * 2
        symb = torch.FloatTensor([symb1 + symb3, symb2 + symb4])
    else:
        NotImplementedError()
    symb = symb.to(device)
    return symb

def generating_channel(var_array, device_for_data):  # channel (rayleigh)
    device = device_for_data
    h = torch.empty(len(var_array), 2)
    for i in range(len(var_array)):
        Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), var_array[i] * torch.eye(2))
        h_tmp = Chan.sample()
        h[i] = h_tmp
    h = h.to(device)
    return h

def generating_noise_dist(var):
    Noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), var * torch.eye(2))
    return Noise

def generating_distortion(mean_array, device_for_data):  # hardware's non-ideality
    device = device_for_data
    l = torch.empty(len(mean_array), 1)
    for i in range(len(mean_array)):
        Lambda = torch.distributions.normal.Normal(torch.tensor([mean_array[i]]),
                                                   torch.tensor([0.0]))  # if we want to give rand. for coefficients
        l_tmp = Lambda.sample()
        l[i] = l_tmp
    l = l.to(device)
    return l


def iq_imbalance(s, epsilon, delta, device):
    s = s.to(device)
    s_distorted = torch.FloatTensor([0, 0])
    s_distorted = s_distorted.to(device)
    s_distorted[0] = (1 + epsilon) * torch.cos(delta) * s[0] - (1 + epsilon) * torch.sin(delta) * s[1]
    s_distorted[1] = (1 - epsilon) * torch.cos(delta) * s[1] - (1 - epsilon) * torch.sin(delta) * s[0]
    return s_distorted


def generating_training_set(curr_dev_char, K_TR, K_TE, num_dev, M, var_array, var, mean_array0, mean_array1,
                            mean_array3, mean_array5, writer_per_dev_tot, if_cali, if_symm, meta_train_version,
                            if_reuse_metatrain_pilot_symbols, power, args, device):
    ## generating training dataset
    train_set = torch.zeros(num_dev, K_TR + K_TE, 4)  # 4 = 2+2, s and y
    train_set = train_set.to(device)

    channel_set_for_vis = torch.zeros(num_dev, 2)
    channel_set_for_vis = channel_set_for_vis.to(device)

    non_linearity_set_genie = torch.zeros(num_dev, 2)
    non_linearity_set_genie = non_linearity_set_genie.to(device)

    if curr_dev_char is None:
        print(
            'in meta-training, we are generating new channel and distortion for test (if you are in online, something wrong!!!!')
        h = generating_channel(var_array, device)
        l_0 = generating_distortion(mean_array0, device)
        l_1 = generating_distortion(mean_array1, device)
    else:
        h = curr_dev_char[0]
        l_0 = curr_dev_char[1]
        l_1 = curr_dev_char[2]

    for ind_d in range(num_dev):
        print('curr dev', ind_d)
        for i in range(K_TR + K_TE):
            s_tmp = generating_symbol(M, device)
            if if_reuse_metatrain_pilot_symbols:
                if M == 5:
                    if i % 4 == 0:
                        s_tmp = torch.FloatTensor([-3, 0])
                    elif i % 4 == 1:
                        s_tmp = torch.FloatTensor([3, 0])
                    elif i % 4 == 2:
                        s_tmp = torch.FloatTensor([-1, 0])
                    elif i % 4 == 3:
                        s_tmp = torch.FloatTensor([1, 0])
                elif M == 4:
                    print('currenlty only considering 4-PAM (M=5) and 16-QAM (M=16)')
                    raise NotImplementedError
                elif M == 16:
                    if i % 16 == 0:
                        s_tmp = torch.FloatTensor([-3, -3])
                    elif i % 16 == 1:
                        s_tmp = torch.FloatTensor([-3, 1])
                    elif i % 16 == 2:
                        s_tmp = torch.FloatTensor([1, 1])
                    elif i % 16 == 3:
                        s_tmp = torch.FloatTensor([1, -3])

                    elif i % 16 == 4:
                        s_tmp = torch.FloatTensor([-3, 3])
                    elif i % 16 == 5:
                        s_tmp = torch.FloatTensor([3, 1])
                    elif i % 16 == 6:
                        s_tmp = torch.FloatTensor([1, -1])
                    elif i % 16 == 7:
                        s_tmp = torch.FloatTensor([-1, -3])

                    elif i % 16 == 8:
                        s_tmp = torch.FloatTensor([3, 3])
                    elif i % 16 == 9:
                        s_tmp = torch.FloatTensor([3, -1])
                    elif i % 16 == 10:
                        s_tmp = torch.FloatTensor([-1, -1])
                    elif i % 16 == 11:
                        s_tmp = torch.FloatTensor([-1, 3])

                    elif i % 16 == 12:
                        s_tmp = torch.FloatTensor([3, -3])
                    elif i % 16 == 13:
                        s_tmp = torch.FloatTensor([-3, -1])
                    elif i % 16 == 14:
                        s_tmp = torch.FloatTensor([-1, 1])
                    elif i % 16 == 15:
                        s_tmp = torch.FloatTensor([1, 3])
            if (M == 16) and (not if_cali):
                if i == 0:  # for computing avg_power_ratio
                    no_distortion_avg_power = 10 * pow(args.power, 2)
                    if args.if_iq_imbalance:
                        distorted_1 = iq_imbalance(torch.FloatTensor([1, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_2 = iq_imbalance(torch.FloatTensor([3, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_3 = iq_imbalance(torch.FloatTensor([1, 3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_4 = iq_imbalance(torch.FloatTensor([3, 3]), l_0[ind_d], l_1[ind_d], device)

                        distorted_5 = iq_imbalance(torch.FloatTensor([-1, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_6 = iq_imbalance(torch.FloatTensor([-3, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_7 = iq_imbalance(torch.FloatTensor([-1, 3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_8 = iq_imbalance(torch.FloatTensor([-3, 3]), l_0[ind_d], l_1[ind_d], device)

                        distorted_9 = iq_imbalance(torch.FloatTensor([1, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_10 = iq_imbalance(torch.FloatTensor([3, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_11 = iq_imbalance(torch.FloatTensor([1, -3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_12 = iq_imbalance(torch.FloatTensor([3, -3]), l_0[ind_d], l_1[ind_d], device)

                        distorted_13 = iq_imbalance(torch.FloatTensor([-1, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_14 = iq_imbalance(torch.FloatTensor([-3, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_15 = iq_imbalance(torch.FloatTensor([-1, -3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_16 = iq_imbalance(torch.FloatTensor([-3, -3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_amplitude_1 = pow(distorted_1[0], 2) + pow(distorted_1[1], 2)
                        distorted_amplitude_2 = pow(distorted_2[0], 2) + pow(distorted_2[1], 2)
                        distorted_amplitude_3 = pow(distorted_3[0], 2) + pow(distorted_3[1], 2)
                        distorted_amplitude_4 = pow(distorted_4[0], 2) + pow(distorted_4[1], 2)

                        distorted_amplitude_5 = pow(distorted_5[0], 2) + pow(distorted_5[1], 2)
                        distorted_amplitude_6 = pow(distorted_6[0], 2) + pow(distorted_6[1], 2)
                        distorted_amplitude_7 = pow(distorted_7[0], 2) + pow(distorted_7[1], 2)
                        distorted_amplitude_8 = pow(distorted_8[0], 2) + pow(distorted_8[1], 2)

                        distorted_amplitude_9 = pow(distorted_9[0], 2) + pow(distorted_9[1], 2)
                        distorted_amplitude_10 = pow(distorted_10[0], 2) + pow(distorted_10[1], 2)
                        distorted_amplitude_11 = pow(distorted_11[0], 2) + pow(distorted_11[1], 2)
                        distorted_amplitude_12 = pow(distorted_12[0], 2) + pow(distorted_12[1], 2)

                        distorted_amplitude_13 = pow(distorted_13[0], 2) + pow(distorted_13[1], 2)
                        distorted_amplitude_14 = pow(distorted_14[0], 2) + pow(distorted_14[1], 2)
                        distorted_amplitude_15 = pow(distorted_15[0], 2) + pow(distorted_15[1], 2)
                        distorted_amplitude_16 = pow(distorted_16[0], 2) + pow(distorted_16[1], 2)

                        distorted_avg_power = (distorted_amplitude_1 + distorted_amplitude_2 +
                                               distorted_amplitude_3 + distorted_amplitude_4 + distorted_amplitude_5 + distorted_amplitude_6 +
                                               distorted_amplitude_7 + distorted_amplitude_8 + distorted_amplitude_9 + distorted_amplitude_10 +
                                               distorted_amplitude_11 + distorted_amplitude_12 + distorted_amplitude_13 + distorted_amplitude_14 +
                                               distorted_amplitude_15 + distorted_amplitude_16) / 16
                        power = pow((no_distortion_avg_power / distorted_avg_power), 0.5)
                    else:
                        print('for now, we are only considering I/Q imbalance hardward imperfection')
                        raise NotImplementedError
                else:
                    pass  # power = args.power
            else:
                pass
            train_set[ind_d, i, :2] = s_tmp  # generating symbol # original constellation point without distortion
            Noise = generating_noise_dist(var)
            n = Noise.sample()
            n = n.to(device)
            if args.if_iq_imbalance:
                s_tmp = iq_imbalance(s_tmp, l_0[ind_d], l_1[ind_d], device)
                s_tmp[0] = s_tmp[0] * power
                s_tmp[1] = s_tmp[1] * power
            else:
                # only for Toy example
                squared_amplitude = pow(s_tmp[0], 2) + pow(s_tmp[1],
                                                           2)  # amplitude of the complex symbol, s_tmp[0] as real and s_tmp[1] as img
                amplitude = pow(squared_amplitude, 0.5)
                sin_phase = s_tmp[1] / amplitude
                cos_phase = s_tmp[0] / amplitude
            if if_cali:  # no distortion for Toy scenario
                distorted_amplitude = amplitude * power
                s_tmp[0] = distorted_amplitude * cos_phase
                s_tmp[1] = distorted_amplitude * sin_phase
                if if_symm:  # channel with +- 1
                    if ind_d % 2 == 0:
                        h[ind_d][0] = 1
                        h[ind_d][1] = 0
                    else:
                        h[ind_d][0] = -1
                        h[ind_d][1] = 0
            else:
                if args.if_iq_imbalance:
                    pass
                else:
                    print('for now, we are only considering I/Q imbalance hardward imperfection')
                    raise NotImplementedError
            if args.if_awgn:
                train_set[ind_d, i, 2] = s_tmp[0]
                train_set[ind_d, i, 3] = s_tmp[1]
            else: # fading channel
                train_set[ind_d, i, 2] = h[ind_d][0] * s_tmp[0] - h[ind_d][1] * s_tmp[1]
                train_set[ind_d, i, 3] = h[ind_d][0] * s_tmp[1] + h[ind_d][1] * s_tmp[0]
            if M == 2 or M == 5:  # For the case with only real symbols
                if meta_train_version == 1:  # 1 is channel with -1 and 1 w.p. 0.5 each and fix always for different num pilots
                    if ind_d % 2 == 0:
                        h_abs = 1
                    else:
                        h_abs = -1
                    train_set[ind_d, i, 2] = h_abs * s_tmp[0]
                else:
                    h_abs_squared = pow(h[ind_d][0], 2) + pow(h[ind_d][1], 2)
                    h_abs = pow(h_abs_squared, 0.5)  # now it follows Rayleigh even for BPSK, PAM
                    h_abs = h_abs * (h[ind_d][0] / abs(h[ind_d][0]))
                    train_set[ind_d, i, 2] = h_abs * s_tmp[0]
            train_set[ind_d, i, 2:] = train_set[ind_d, i, 2:] + n  # adding noise
            if M == 2 or M == 5:
                train_set[ind_d, i, 3] = 0  # giving 0 to the machine since only Real symbols

            channel_set_for_vis[ind_d] = h[ind_d]  # deprecated
            non_linearity_set_genie[ind_d, 0] = l_0[ind_d] # deprecated
            non_linearity_set_genie[ind_d, 1] = l_1[ind_d] # deprecated
    return train_set, channel_set_for_vis, non_linearity_set_genie


def generating_test_set(curr_dev_char, K_TEST_TR, K_TEST_TE, num_dev, M, var_array, var, mean_array0, mean_array1,
                        mean_array3, mean_array5, ind_dev, if_cali, if_symm, test_train_version,
                        if_reuse_testtrain_pilot_symbols, power, device, args):
    ## generating test dataset
    if num_dev != 1:
        NotImplementedError('num_dev should be 1')
    test_set = torch.zeros(num_dev, K_TEST_TR + K_TEST_TE, 4)  # 3 = 2+1, s and y
    test_set = test_set.to(device)
    channel_set_genie = torch.zeros(num_dev)
    channel_set_genie = channel_set_genie.to(device)

    channel_set_for_vis = torch.zeros(num_dev, 2)
    channel_set_for_vis = channel_set_for_vis.to(device)

    non_linearity_set_genie = torch.zeros(num_dev, 2)
    non_linearity_set_genie = non_linearity_set_genie.to(device)

    if curr_dev_char is None:
        print('we are generating new channel and distortion for test (if you are in online, something wrong!!!!')
        h = generating_channel(var_array, device)
        l_0 = generating_distortion(mean_array0, device)
        l_1 = generating_distortion(mean_array1, device)
    else:
        h = curr_dev_char[0]
        l_0 = curr_dev_char[1]
        l_1 = curr_dev_char[2]

    for ind_d in range(num_dev):  # only one dev
        print('curr dev', ind_d)
        for i in range(K_TEST_TR + K_TEST_TE):
            s_tmp = generating_symbol(M, device)
            if if_reuse_testtrain_pilot_symbols:
                if i < K_TEST_TR:
                    if M == 5:
                        if i % 4 == 0:
                            s_tmp = torch.FloatTensor([-3, 0])
                        elif i % 4 == 1:
                            s_tmp = torch.FloatTensor([3, 0])
                        elif i % 4 == 2:
                            s_tmp = torch.FloatTensor([-1, 0])
                        elif i % 4 == 3:
                            s_tmp = torch.FloatTensor([1, 0])
                    elif M == 4:
                        if i % 4 == 0:
                            s_tmp = torch.FloatTensor([-1, -1])
                        elif i % 4 == 1:
                            s_tmp = torch.FloatTensor([1, 1])
                        elif i % 4 == 2:
                            s_tmp = torch.FloatTensor([-1, 1])
                        elif i % 4 == 3:
                            s_tmp = torch.FloatTensor([1, -1])
                    elif M == 16:
                        if i % 16 == 0:
                            s_tmp = torch.FloatTensor([-3, -3])
                        elif i % 16 == 1:
                            s_tmp = torch.FloatTensor([-3, 1])
                        elif i % 16 == 2:
                            s_tmp = torch.FloatTensor([1, 1])
                        elif i % 16 == 3:
                            s_tmp = torch.FloatTensor([1, -3])

                        elif i % 16 == 4:
                            s_tmp = torch.FloatTensor([-3, 3])
                        elif i % 16 == 5:
                            s_tmp = torch.FloatTensor([3, 1])
                        elif i % 16 == 6:
                            s_tmp = torch.FloatTensor([1, -1])
                        elif i % 16 == 7:
                            s_tmp = torch.FloatTensor([-1, -3])

                        elif i % 16 == 8:
                            s_tmp = torch.FloatTensor([3, 3])
                        elif i % 16 == 9:
                            s_tmp = torch.FloatTensor([3, -1])
                        elif i % 16 == 10:
                            s_tmp = torch.FloatTensor([-1, -1])
                        elif i % 16 == 11:
                            s_tmp = torch.FloatTensor([-1, 3])

                        elif i % 16 == 12:
                            s_tmp = torch.FloatTensor([3, -3])
                        elif i % 16 == 13:
                            s_tmp = torch.FloatTensor([-3, -1])
                        elif i % 16 == 14:
                            s_tmp = torch.FloatTensor([-1, 1])
                        elif i % 16 == 15:
                            s_tmp = torch.FloatTensor([1, 3])

            ######
            if (M == 16) and (not if_cali):
                if i == 0:  # for computing avg_power_ratio
                    ####
                    no_distortion_avg_power = 10 * pow(args.power, 2)
                    if args.if_iq_imbalance:
                        distorted_1 = iq_imbalance(torch.FloatTensor([1, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_2 = iq_imbalance(torch.FloatTensor([3, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_3 = iq_imbalance(torch.FloatTensor([1, 3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_4 = iq_imbalance(torch.FloatTensor([3, 3]), l_0[ind_d], l_1[ind_d], device)

                        distorted_5 = iq_imbalance(torch.FloatTensor([-1, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_6 = iq_imbalance(torch.FloatTensor([-3, 1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_7 = iq_imbalance(torch.FloatTensor([-1, 3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_8 = iq_imbalance(torch.FloatTensor([-3, 3]), l_0[ind_d], l_1[ind_d], device)

                        distorted_9 = iq_imbalance(torch.FloatTensor([1, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_10 = iq_imbalance(torch.FloatTensor([3, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_11 = iq_imbalance(torch.FloatTensor([1, -3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_12 = iq_imbalance(torch.FloatTensor([3, -3]), l_0[ind_d], l_1[ind_d], device)

                        distorted_13 = iq_imbalance(torch.FloatTensor([-1, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_14 = iq_imbalance(torch.FloatTensor([-3, -1]), l_0[ind_d], l_1[ind_d], device)
                        distorted_15 = iq_imbalance(torch.FloatTensor([-1, -3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_16 = iq_imbalance(torch.FloatTensor([-3, -3]), l_0[ind_d], l_1[ind_d], device)
                        distorted_amplitude_1 = pow(distorted_1[0], 2) + pow(distorted_1[1], 2)
                        distorted_amplitude_2 = pow(distorted_2[0], 2) + pow(distorted_2[1], 2)
                        distorted_amplitude_3 = pow(distorted_3[0], 2) + pow(distorted_3[1], 2)
                        distorted_amplitude_4 = pow(distorted_4[0], 2) + pow(distorted_4[1], 2)

                        distorted_amplitude_5 = pow(distorted_5[0], 2) + pow(distorted_5[1], 2)
                        distorted_amplitude_6 = pow(distorted_6[0], 2) + pow(distorted_6[1], 2)
                        distorted_amplitude_7 = pow(distorted_7[0], 2) + pow(distorted_7[1], 2)
                        distorted_amplitude_8 = pow(distorted_8[0], 2) + pow(distorted_8[1], 2)

                        distorted_amplitude_9 = pow(distorted_9[0], 2) + pow(distorted_9[1], 2)
                        distorted_amplitude_10 = pow(distorted_10[0], 2) + pow(distorted_10[1], 2)
                        distorted_amplitude_11 = pow(distorted_11[0], 2) + pow(distorted_11[1], 2)
                        distorted_amplitude_12 = pow(distorted_12[0], 2) + pow(distorted_12[1], 2)

                        distorted_amplitude_13 = pow(distorted_13[0], 2) + pow(distorted_13[1], 2)
                        distorted_amplitude_14 = pow(distorted_14[0], 2) + pow(distorted_14[1], 2)
                        distorted_amplitude_15 = pow(distorted_15[0], 2) + pow(distorted_15[1], 2)
                        distorted_amplitude_16 = pow(distorted_16[0], 2) + pow(distorted_16[1], 2)

                        distorted_avg_power = (distorted_amplitude_1 + distorted_amplitude_2 +
                                               distorted_amplitude_3 + distorted_amplitude_4 + distorted_amplitude_5 + distorted_amplitude_6 +
                                               distorted_amplitude_7 + distorted_amplitude_8 + distorted_amplitude_9 + distorted_amplitude_10 +
                                               distorted_amplitude_11 + distorted_amplitude_12 + distorted_amplitude_13 + distorted_amplitude_14 +
                                               distorted_amplitude_15 + distorted_amplitude_16) / 16
                        power = pow((no_distortion_avg_power / distorted_avg_power), 0.5)
                    else:
                        print('for now, we are only considering I/Q imbalance hardward imperfection')
                        raise NotImplementedError
                else:
                    pass  # power = args.power
            else:
                pass

            test_set[ind_d, i, :2] = s_tmp  # generating symbol
            Noise = generating_noise_dist(var)
            n = Noise.sample()
            n = n.to(device)

            if args.if_iq_imbalance:
                s_tmp = iq_imbalance(s_tmp, l_0[ind_d], l_1[ind_d], device)
                s_tmp[0] = s_tmp[0] * power
                s_tmp[1] = s_tmp[1] * power
            else:
                squared_amplitude = pow(s_tmp[0], 2) + pow(s_tmp[1],
                                                           2)  # amplitude of the complex symbol, s_tmp[0] as real and s_tmp[1] as img
                amplitude = pow(squared_amplitude, 0.5)
                sin_phase = s_tmp[1] / amplitude
                cos_phase = s_tmp[0] / amplitude
            if if_cali:
                distorted_amplitude = amplitude * power
                l_0[ind_d] = 0
                l_1[ind_d] = 0
                s_tmp[0] = distorted_amplitude * cos_phase
                s_tmp[1] = distorted_amplitude * sin_phase
                if if_symm:  # always this when if_cali -> toy
                    if ind_dev % 2 == 0:
                        h[ind_d][0] = 1
                        h[ind_d][1] = 0
                    else:
                        h[ind_d][0] = -1
                        h[ind_d][1] = 0
            else:
                if args.if_iq_imbalance:
                    pass
                else:
                    print('for now, we are only considering I/Q imbalance hardward imperfection')
                    raise NotImplementedError

            if args.if_awgn:
                test_set[ind_d, i, 2] = s_tmp[0]
                test_set[ind_d, i, 3] = s_tmp[1]
            else:
                test_set[ind_d, i, 2] = h[ind_d][0] * s_tmp[0] - h[ind_d][1] * s_tmp[1]
                test_set[ind_d, i, 3] = h[ind_d][0] * s_tmp[1] + h[ind_d][1] * s_tmp[0]

            if M == 2 or M == 5:
                if test_train_version == 1:  # # 1 is channel with -1 and 1 w.p. 0.5 each and fix always for different num pilots
                    if ind_dev % 2 == 0:
                        h_abs = 1
                    else:
                        h_abs = -1
                    test_set[ind_d, i, 2] = h_abs * s_tmp[0]
                else:
                    h_abs_squared = pow(h[ind_d][0], 2) + pow(h[ind_d][1], 2)
                    h_abs = pow(h_abs_squared, 0.5)  # now it follows Rayleigh even for BPSK, PAM
                    h_abs = h_abs * (h[ind_d][0] / abs(h[ind_d][0]))  # sign follows h_real
                    test_set[ind_d, i, 2] = h_abs * s_tmp[0]
            test_set[ind_d, i, 2:] = test_set[ind_d, i, 2:] + n  # adding noise
            if M == 2 or M == 5:
                test_set[ind_d, i, 3] = 0  # giving 0 to the machine
        if M == 2 or M == 5:
            channel_set_genie[ind_d] = h_abs
        elif M == 16 or M == 4:
            h_abs_squared = pow(h[ind_d][0], 2) + pow(h[ind_d][1], 2)
            h_abs = pow(h_abs_squared, 0.5)
            h_abs = h_abs * (h[ind_d][0] / abs(h[ind_d][0]))
            channel_set_genie[ind_d] = h_abs # deprecated
        channel_set_for_vis[ind_d] = h[ind_d]  # deprecated
        if if_cali:
            non_linearity_set_genie[ind_d, 0] = 1 # deprecated
            non_linearity_set_genie[ind_d, 1] = 0 # deprecated
        else:
            non_linearity_set_genie[ind_d, 0] = l_0[ind_d] # deprecated
            non_linearity_set_genie[ind_d, 1] = l_1[ind_d] # deprecated
    return test_set, channel_set_genie, channel_set_for_vis, non_linearity_set_genie


def generating_online_training_set(args, curr_dev_char, total_data_set, K_TR_accum, K_TR, K_TE, ind_T, M, var_array,
                                   var, mean_array0, mean_array1, mean_array3, mean_array5,
                                   writer_per_dev_tot, if_cali, if_symm, meta_train_version,
                                   if_reuse_metatrain_pilot_symbols,
                                   power, device):  # contains meta-train, meta-test, which is also used for test-train
    # we can just set K_TE = 0 and make K_TR as our # training pilot
    # K_TR here is current iteration's adding pilot number (1 is most reasonable thinking now...)
    ## generating training dataset
    assert K_TE == 0
    if K_TE != 0:
        print('for online, K_TE should be 0')
        raise NotImplementedError

    print('generate training set for %d dev' % ind_T)

    num_dev = args.total_online_time
    train_set = torch.zeros(num_dev, K_TR + K_TE, 4)  # 3 = 2+2, s and y
    train_set = train_set.to(device)

    if K_TR_accum == 0:
        assert curr_dev_char == None
        print('generate channel and distortion for curr dev')
        h = generating_channel(var_array, device)
        l_0 = generating_distortion(mean_array0, device)
        l_1 = generating_distortion(mean_array1, device)
        l_3 = generating_distortion(mean_array3, device)
        l_5 = generating_distortion(mean_array5, device)
        curr_dev_char = [h, l_0, l_1, l_3, l_5]
    else:
        print('load prev. generated channel and distortion for curr dev')
        h = curr_dev_char[0]
        l_0 = curr_dev_char[1]
        l_1 = curr_dev_char[2]
        l_3 = curr_dev_char[3]
        l_5 = curr_dev_char[4]

    ind_d = ind_T
    for i in range(K_TR + K_TE):
        i_curr = i + K_TR_accum
        s_tmp = generating_symbol(M, device)
        if if_reuse_metatrain_pilot_symbols:
            if M == 5:
                if i_curr % 4 == 0:
                    s_tmp = torch.FloatTensor([-3, 0])
                elif i_curr % 4 == 1:
                    s_tmp = torch.FloatTensor([-1, 0])
                elif i_curr % 4 == 2:
                    s_tmp = torch.FloatTensor([1, 0])
                elif i_curr % 4 == 3:
                    s_tmp = torch.FloatTensor([3, 0])
            elif M == 4:
                if i_curr % 4 == 0:
                    s_tmp = torch.FloatTensor([-1, -1])
                elif i_curr % 4 == 1:
                    s_tmp = torch.FloatTensor([1, 1])
                elif i_curr % 4 == 2:
                    s_tmp = torch.FloatTensor([-1, 1])
                elif i_curr % 4 == 3:
                    s_tmp = torch.FloatTensor([1, -1])
            elif M == 16:
                if i % 16 == 0:
                    s_tmp = torch.FloatTensor([-3, -3])
                elif i % 16 == 1:
                    s_tmp = torch.FloatTensor([-3, 1])
                elif i % 16 == 2:
                    s_tmp = torch.FloatTensor([1, 1])
                elif i % 16 == 3:
                    s_tmp = torch.FloatTensor([1, -3])

                elif i % 16 == 4:
                    s_tmp = torch.FloatTensor([-3, 3])
                elif i % 16 == 5:
                    s_tmp = torch.FloatTensor([3, 1])
                elif i % 16 == 6:
                    s_tmp = torch.FloatTensor([1, -1])
                elif i % 16 == 7:
                    s_tmp = torch.FloatTensor([-1, -3])

                elif i % 16 == 8:
                    s_tmp = torch.FloatTensor([3, 3])
                elif i % 16 == 9:
                    s_tmp = torch.FloatTensor([3, -1])
                elif i % 16 == 10:
                    s_tmp = torch.FloatTensor([-1, -1])
                elif i % 16 == 11:
                    s_tmp = torch.FloatTensor([-1, 3])

                elif i % 16 == 12:
                    s_tmp = torch.FloatTensor([3, -3])
                elif i % 16 == 13:
                    s_tmp = torch.FloatTensor([-3, -1])
                elif i % 16 == 14:
                    s_tmp = torch.FloatTensor([-1, 1])
                elif i % 16 == 15:
                    s_tmp = torch.FloatTensor([1, 3])

        if (M == 16) and (not if_cali):
            if i == 0:  # for computing avg_power_ratio
                no_distortion_avg_power = 10 * pow(args.power, 2)
                if args.if_iq_imbalance:
                    distorted_1 = iq_imbalance(torch.FloatTensor([1, 1]), l_0[0], l_1[0], device)
                    distorted_2 = iq_imbalance(torch.FloatTensor([3, 1]), l_0[0], l_1[0], device)
                    distorted_3 = iq_imbalance(torch.FloatTensor([1, 3]), l_0[0], l_1[0], device)
                    distorted_4 = iq_imbalance(torch.FloatTensor([3, 3]), l_0[0], l_1[0], device)

                    distorted_5 = iq_imbalance(torch.FloatTensor([-1, 1]), l_0[0], l_1[0], device)
                    distorted_6 = iq_imbalance(torch.FloatTensor([-3, 1]), l_0[0], l_1[0], device)
                    distorted_7 = iq_imbalance(torch.FloatTensor([-1, 3]), l_0[0], l_1[0], device)
                    distorted_8 = iq_imbalance(torch.FloatTensor([-3, 3]), l_0[0], l_1[0], device)

                    distorted_9 = iq_imbalance(torch.FloatTensor([1, -1]), l_0[0], l_1[0], device)
                    distorted_10 = iq_imbalance(torch.FloatTensor([3, -1]), l_0[0], l_1[0], device)
                    distorted_11 = iq_imbalance(torch.FloatTensor([1, -3]), l_0[0], l_1[0], device)
                    distorted_12 = iq_imbalance(torch.FloatTensor([3, -3]), l_0[0], l_1[0], device)

                    distorted_13 = iq_imbalance(torch.FloatTensor([-1, -1]), l_0[0], l_1[0], device)
                    distorted_14 = iq_imbalance(torch.FloatTensor([-3, -1]), l_0[0], l_1[0], device)
                    distorted_15 = iq_imbalance(torch.FloatTensor([-1, -3]), l_0[0], l_1[0], device)
                    distorted_16 = iq_imbalance(torch.FloatTensor([-3, -3]), l_0[0], l_1[0], device)
                    distorted_amplitude_1 = pow(distorted_1[0], 2) + pow(distorted_1[1], 2)
                    distorted_amplitude_2 = pow(distorted_2[0], 2) + pow(distorted_2[1], 2)
                    distorted_amplitude_3 = pow(distorted_3[0], 2) + pow(distorted_3[1], 2)
                    distorted_amplitude_4 = pow(distorted_4[0], 2) + pow(distorted_4[1], 2)

                    distorted_amplitude_5 = pow(distorted_5[0], 2) + pow(distorted_5[1], 2)
                    distorted_amplitude_6 = pow(distorted_6[0], 2) + pow(distorted_6[1], 2)
                    distorted_amplitude_7 = pow(distorted_7[0], 2) + pow(distorted_7[1], 2)
                    distorted_amplitude_8 = pow(distorted_8[0], 2) + pow(distorted_8[1], 2)

                    distorted_amplitude_9 = pow(distorted_9[0], 2) + pow(distorted_9[1], 2)
                    distorted_amplitude_10 = pow(distorted_10[0], 2) + pow(distorted_10[1], 2)
                    distorted_amplitude_11 = pow(distorted_11[0], 2) + pow(distorted_11[1], 2)
                    distorted_amplitude_12 = pow(distorted_12[0], 2) + pow(distorted_12[1], 2)

                    distorted_amplitude_13 = pow(distorted_13[0], 2) + pow(distorted_13[1], 2)
                    distorted_amplitude_14 = pow(distorted_14[0], 2) + pow(distorted_14[1], 2)
                    distorted_amplitude_15 = pow(distorted_15[0], 2) + pow(distorted_15[1], 2)
                    distorted_amplitude_16 = pow(distorted_16[0], 2) + pow(distorted_16[1], 2)

                    distorted_avg_power = (distorted_amplitude_1 + distorted_amplitude_2 +
                                           distorted_amplitude_3 + distorted_amplitude_4 + distorted_amplitude_5 + distorted_amplitude_6 +
                                           distorted_amplitude_7 + distorted_amplitude_8 + distorted_amplitude_9 + distorted_amplitude_10 +
                                           distorted_amplitude_11 + distorted_amplitude_12 + distorted_amplitude_13 + distorted_amplitude_14 +
                                           distorted_amplitude_15 + distorted_amplitude_16) / 16
                    power = pow((no_distortion_avg_power / distorted_avg_power), 0.5)
                else:
                    print('for now, we are only considering I/Q imbalance hardward imperfection')
                    raise NotImplementedError
            else:
                pass  # power = args.power
        else:
            pass

        train_set[ind_d, i, :2] = s_tmp  # generating symbol
        Noise = generating_noise_dist(var)
        n = Noise.sample()
        n = n.to(device)

        if args.if_iq_imbalance:
            # print('original s', s_tmp)
            s_tmp = iq_imbalance(s_tmp, l_0[0], l_1[0], device)
            s_tmp[0] = s_tmp[0] * power
            s_tmp[1] = s_tmp[1] * power
            # print('iq imbalance s', s_tmp, 'with distortion', l_0[ind_d], 'with power', power)
        else:
            squared_amplitude = pow(s_tmp[0], 2) + pow(s_tmp[1],
                                                       2)  # amplitude of the complex symbol, s_tmp[0] as real and s_tmp[1] as img
            amplitude = pow(squared_amplitude, 0.5)
            sin_phase = s_tmp[1] / amplitude
            cos_phase = s_tmp[0] / amplitude
        if if_cali:  # no distortion
            distorted_amplitude = amplitude * power
            s_tmp[0] = distorted_amplitude * cos_phase
            s_tmp[1] = distorted_amplitude * sin_phase
            # n = 0
            if if_symm:  # channel with +- 1
                if ind_d % 2 == 0:
                    h[0] = 1
                else:
                    h[0] = -1
        else:
            if args.if_iq_imbalance:
                pass
            else:
                print('for now, we are only considering I/Q imbalance hardward imperfection')
                raise NotImplementedError

        if i == 0:
            sum_power = pow(s_tmp[0], 2) + pow(s_tmp[1], 2)
        else:
            sum_power += pow(s_tmp[0], 2) + pow(s_tmp[1], 2)
        if i == 15:
            print('tr set avg power', sum_power / 16, 'should be same with', 10 * pow(args.power, 2))

        if args.if_awgn:
            train_set[ind_d, i, 2] = s_tmp[0]
            train_set[ind_d, i, 3] = s_tmp[1]
        else:
            train_set[ind_d, i, 2] = h[0][0] * s_tmp[0] - h[0][1] * s_tmp[1]
            train_set[ind_d, i, 3] = h[0][0] * s_tmp[1] + h[0][1] * s_tmp[0]
        if M == 2 or M == 5:  # For the case with only real symbols
            if meta_train_version == 1:  # # 1 is channel with -1 and 1 w.p. 0.5 each and fix always for different num pilots
                if ind_d % 2 == 0:
                    h_abs = 1
                else:
                    h_abs = -1
                train_set[ind_d, i, 2] = h_abs * s_tmp[0]
            else:
                h_abs_squared = pow(h[0][0], 2) + pow(h[0][1], 2)
                h_abs = pow(h_abs_squared, 0.5)  # now it follows Rayleigh even for BPSK, PAM
                h_abs = h_abs * (h[0][0] / abs(h[0][0]))
                train_set[ind_d, i, 2] = h_abs * s_tmp[0]
        train_set[ind_d, i, 2:] = train_set[ind_d, i, 2:] + n  # adding noise
        if M == 2 or M == 5:
            train_set[ind_d, i, 3] = 0  # giving 0 to the machine since only Real symbols
    total_data_set[ind_T, K_TR_accum:K_TR_accum + K_TR, :] = train_set[ind_d, :, :]
    return total_data_set, curr_dev_char


def complex_mul(h, x):  # h fixed on batch, x has multiple batch
    # assert len(h.shape) == 1
    if len(h.shape) == 1:
        if x.shape[1] == 2:
            y = torch.zeros(x.shape[0], 2, dtype=torch.float)
            y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
            y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
        else:  # due to cavia, additional input, we multiply
            y = torch.zeros(x.shape[0], x.shape[1], dtype=torch.float)
            y[:, 0] = x[:, 0] * h[0] - x[:, 1] * h[1]
            y[:, 1] = x[:, 0] * h[1] + x[:, 1] * h[0]
            y[:, 2:] = x[:, 2:]
    elif len(h.shape) == 2:  # for RTN
        assert x.shape[1] == 2
        assert x.shape[0] == h.shape[0]
        y = torch.zeros(x.shape[0], 2, dtype=torch.float)
        y[:, 0] = x[:, 0] * h[:, 0] - x[:, 1] * h[:, 1]
        y[:, 1] = x[:, 0] * h[:, 1] + x[:, 1] * h[:, 0]
    else:
        raise NotImplementedError
    return y

