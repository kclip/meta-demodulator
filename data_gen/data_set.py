from __future__ import print_function
import torch

def generating_symbol(M, device_for_data):
    device = device_for_data
    Bern = torch.distributions.bernoulli.Bernoulli(torch.tensor([0.5]))  # equal prob.
    if M == 2: # BPSK
        symb = Bern.sample()
        symb = symb.to(device)
        if symb == 0:
            symb = -1
        symb = torch.FloatTensor([symb, 0])
    elif M == 4: # 4-QAM
        symb1 = Bern.sample()
        symb2 = Bern.sample()
        symb1 = symb1.to(device)
        symb2 = symb2.to(device)
        if symb1 == 0:
            symb1 = -1
        if symb2 == 0:
            symb2 = -1
        symb = torch.FloatTensor([symb1, symb2])
    elif M == 5: # 4-PAM
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

    elif M == 16: # 16-QAM
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
        symb3 = symb3*2
        symb4 = symb4*2
        symb = torch.FloatTensor([symb1 + symb3, symb2+ symb4])
    else:
        NotImplementedError()
    symb = symb.to(device)
    return symb

def generating_channel(var_array, device_for_data): #channel (rayleigh)
    device = device_for_data
    h = torch.empty(len(var_array),2)
    for i in range(len(var_array)):
        Chan = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), var_array[i] * torch.eye(2))
        h_tmp = Chan.sample()
        h[i] = h_tmp
    h = h.to(device)
    return h


def generating_noise_dist(var):
    Noise = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(2), var * torch.eye(2))
    return Noise

def generating_distortion(mean_array, device_for_data): # hardware's non-ideality
    device = device_for_data
    l = torch.empty(len(mean_array), 1)
    for i in range(len(mean_array)):
        Lambda = torch.distributions.normal.Normal(torch.tensor([mean_array[i]]), torch.tensor([0.0])) # if we want to give rand. for coefficients
        l_tmp = Lambda.sample()
        l[i] = l_tmp
    l = l.to(device)
    return l

def generating_training_set(K_TR, K_TE, num_dev, M, var_array, var, mean_array0, mean_array1, mean_array3, mean_array5, writer_per_dev_tot, if_cali, if_symm, meta_train_version, if_reuse_metatrain_pilot_symbols, power, args, device):
    ## generating training dataset
    train_set = torch.zeros(num_dev, K_TR + K_TE, 4)  # 4 = 2+2, s and y
    train_set = train_set.to(device)
    h = generating_channel(var_array, device)
    l_0 = generating_distortion(mean_array0, device)
    l_1 = generating_distortion(mean_array1, device)
    l_3 = generating_distortion(mean_array3, device)
    l_5 = generating_distortion(mean_array5, device)
    for ind_d in range(num_dev):
        for i in range(K_TR + K_TE):
            s_tmp = generating_symbol(M, device)
            if if_reuse_metatrain_pilot_symbols:
                if M == 5:
                    if i % 4 == 0:
                        s_tmp = torch.FloatTensor([-3, 0])
                    elif i % 4 == 1:
                        s_tmp = torch.FloatTensor([-1, 0])
                    elif i % 4 == 2:
                        s_tmp = torch.FloatTensor([1, 0])
                    elif i % 4 == 3:
                        s_tmp = torch.FloatTensor([3, 0])
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
                        s_tmp = torch.FloatTensor([-1, -1])
                    elif i % 16 == 2:
                        s_tmp = torch.FloatTensor([1, 1])
                    elif i % 16 == 3:
                        s_tmp = torch.FloatTensor([3, 3])
                        
                    elif i % 16 == 4:
                        s_tmp = torch.FloatTensor([-3, -1])
                    elif i % 16 == 5:
                        s_tmp = torch.FloatTensor([-1, 3])
                    elif i % 16 == 6:
                        s_tmp = torch.FloatTensor([1, -3])
                    elif i % 16 == 7:
                        s_tmp = torch.FloatTensor([3, 1])
                        
                    elif i % 16 == 8:
                        s_tmp = torch.FloatTensor([-3, 3])
                    elif i % 16 == 9:
                        s_tmp = torch.FloatTensor([-1, 1])
                    elif i % 16 == 10:
                        s_tmp = torch.FloatTensor([1, -1])
                    elif i % 16 == 11:
                        s_tmp = torch.FloatTensor([3, -3])
                    
                    elif i % 16 == 12:
                        s_tmp = torch.FloatTensor([-3, 1])
                    elif i % 16 == 13:
                        s_tmp = torch.FloatTensor([-1, -3])
                    elif i % 16 == 14:
                        s_tmp = torch.FloatTensor([1, 3])
                    elif i % 16 == 15:
                        s_tmp = torch.FloatTensor([3, -1])

            train_set[ind_d, i, :2] = s_tmp  # generating symbol
            Noise = generating_noise_dist(var)
            n = Noise.sample()
            n = n.to(device)
            squared_amplitude = pow(s_tmp[0],2) + pow(s_tmp[1],2) # amplitude of the complex symbol, s_tmp[0] as real and s_tmp[1] as img
            amplitude = pow(squared_amplitude, 0.5)
            sin_phase = s_tmp[1]/amplitude
            cos_phase = s_tmp[0]/amplitude
            amplitude = amplitude * power
            distorted_amplitude = l_0[ind_d]*pow(amplitude,1)/(1+l_1[ind_d]*pow(amplitude,2)) # l_0 is alpha and l_1 is beta  #+ l_1[ind_d]*pow(amplitude, 1) + l_3[ind_d] * pow(amplitude, 3) + l_5[ind_d] * pow(amplitude, 5)
            distorted_amplitude = torch.abs(distorted_amplitude) # avoid - amplitude # check exactly for the def of dist.
            if if_cali: # no distortion
                distorted_amplitude = amplitude
                s_tmp[0] = distorted_amplitude * cos_phase
                s_tmp[1] = distorted_amplitude * sin_phase
                if if_symm: # channel with +- 1
                    if ind_d%2 == 0:
                        h[ind_d] = 1
                    else:
                        h[ind_d] = -1
            else:
                s_tmp[0] = distorted_amplitude * cos_phase
                s_tmp[1] = distorted_amplitude * sin_phase
            train_set[ind_d, i, 2] = h[ind_d][0] * s_tmp[0] - h[ind_d][1] * s_tmp[1]
            train_set[ind_d, i, 3] = h[ind_d][0] * s_tmp[1] + h[ind_d][1] * s_tmp[0]
            if M == 2 or M == 5: # For the case with only real symbols
                if meta_train_version == 1:  # # 1 is channel with -1 and 1 w.p. 0.5 each and fix always for different num pilots
                    if ind_d % 2 == 0:
                        h_abs = 1
                    else:
                        h_abs = -1
                    train_set[ind_d, i, 2] = h_abs * s_tmp[0]
                else:
                    h_abs_squared = pow(h[ind_d][0],2) + pow(h[ind_d][1],2)
                    h_abs = pow(h_abs_squared,0.5) # now it follows Rayleigh even for BPSK, PAM
                    h_abs = h_abs * (h[ind_d][0]/abs(h[ind_d][0]))
                    train_set[ind_d, i, 2] = h_abs * s_tmp[0]
            train_set[ind_d, i, 2:] = train_set[ind_d, i, 2:] + n # adding noise
            if M == 2 or M == 5:
                train_set[ind_d, i, 3] = 0 # giving 0 to the machine since only Real symbols
    return train_set


def generating_test_set(curr_dev_char, K_TEST_TR, K_TEST_TE, num_dev, M, var_array, var, mean_array0, mean_array1, mean_array3, mean_array5, ind_dev, if_cali, if_symm, test_train_version, if_reuse_testtrain_pilot_symbols, power, device):
    ## generating test dataset
    if num_dev != 1:
        NotImplementedError('num_dev should be 1')
    test_set = torch.zeros(num_dev, K_TEST_TR + K_TEST_TE, 4)  # 3 = 2+1, s and y
    test_set = test_set.to(device)
    channel_set_genie = torch.zeros(num_dev)
    channel_set_genie = channel_set_genie.to(device)

    channel_set_for_vis = torch.zeros(num_dev,2)
    channel_set_for_vis = channel_set_for_vis.to(device)

    non_linearity_set_genie = torch.zeros(num_dev,2)
    non_linearity_set_genie = non_linearity_set_genie.to(device)

    if curr_dev_char is None:
        print('we are generating new channel and distortion for test (if you are in online, something wrong!!!!')
        h = generating_channel(var_array, device)
        l_0 = generating_distortion(mean_array0, device)
        l_1 = generating_distortion(mean_array1, device)
        l_3 = generating_distortion(mean_array3, device)
        l_5 = generating_distortion(mean_array5, device)
    else:
        h = curr_dev_char[0]
        l_0 = curr_dev_char[1]
        l_1 = curr_dev_char[2]
        l_3 = curr_dev_char[3]
        l_5 = curr_dev_char[4]

    for ind_d in range(num_dev): # only one dev
        for i in range(K_TEST_TR + K_TEST_TE):
            s_tmp = generating_symbol(M, device)
            if if_reuse_testtrain_pilot_symbols:
                if i < K_TEST_TR:
                    if M == 5:
                        if i % 4 == 0:
                            s_tmp = torch.FloatTensor([-3, 0])
                        elif i % 4 == 1:
                            s_tmp = torch.FloatTensor([-1, 0])
                        elif i % 4 == 2:
                            s_tmp = torch.FloatTensor([1, 0])
                        elif i % 4 == 3:
                            s_tmp = torch.FloatTensor([3, 0])
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
                            s_tmp = torch.FloatTensor([1, 1])
                        elif i % 16 == 1:
                            s_tmp = torch.FloatTensor([1, -3])
                        elif i % 16 == 2:
                            s_tmp = torch.FloatTensor([-3, 1])
                        elif i % 16 == 3:
                            s_tmp = torch.FloatTensor([3, 3])

                        elif i % 16 == 4:
                            s_tmp = torch.FloatTensor([3, 1])
                        elif i % 16 == 5:
                            s_tmp = torch.FloatTensor([-1, -1])
                        elif i % 16 == 6:
                            s_tmp = torch.FloatTensor([-1, -3])
                        elif i % 16 == 7:
                            s_tmp = torch.FloatTensor([-3, -3])

                        elif i % 16 == 8:
                            s_tmp = torch.FloatTensor([3, -1])
                        elif i % 16 == 9:
                            s_tmp = torch.FloatTensor([-1, 3])
                        elif i % 16 == 10:
                            s_tmp = torch.FloatTensor([-1, 1])
                        elif i % 16 == 11:
                            s_tmp = torch.FloatTensor([1, -1])

                        elif i % 16 == 12:
                            s_tmp = torch.FloatTensor([-3, -1])
                        elif i % 16 == 13:
                            s_tmp = torch.FloatTensor([1, 3])
                        elif i % 16 == 14:
                            s_tmp = torch.FloatTensor([-3, 3])
                        elif i % 16 == 15:
                            s_tmp = torch.FloatTensor([3, -3])

            test_set[ind_d, i, :2] = s_tmp  # generating symbol
            Noise = generating_noise_dist(var)
            n = Noise.sample()
            n = n.to(device)
            squared_amplitude = pow(s_tmp[0], 2) + pow(s_tmp[1], 2)  # amplitude of the complex symbol, s_tmp[0] as real and s_tmp[1] as img
            amplitude = pow(squared_amplitude, 0.5)
            sin_phase = s_tmp[1] / amplitude
            cos_phase = s_tmp[0] / amplitude
            amplitude = amplitude * power
            distorted_amplitude = l_0[ind_d]*pow(amplitude,1)/(1+l_1[ind_d]*pow(amplitude,2))     #l_0[ind_d] + l_1[ind_d] * pow(amplitude, 1) + l_3[ind_d] * pow(amplitude, 3) + l_5[ind_d] * pow(amplitude, 5)
            distorted_amplitude = torch.abs(distorted_amplitude)  # avoid - amplitude # check exactly for the def of dist.
            if if_cali:
                distorted_amplitude = amplitude
                l_0[ind_d] = 0
                l_1[ind_d] = 0
                s_tmp[0] = distorted_amplitude * cos_phase
                s_tmp[1] = distorted_amplitude * sin_phase
                if if_symm: # always this when if_cali -> toy
                    if ind_dev%2 == 0:
                        h[ind_d] = 1
                    else:
                        h[ind_d] = -1
            else:
                s_tmp[0] = distorted_amplitude * cos_phase
                s_tmp[1] = distorted_amplitude * sin_phase
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
                    h_abs_squared = pow(h[ind_d][0],2) + pow(h[ind_d][1],2)
                    h_abs = pow(h_abs_squared,0.5) # now it follows Rayleigh even for BPSK, PAM
                    h_abs = h_abs * (h[ind_d][0] / abs(h[ind_d][0])) # sign follows h_real
                    test_set[ind_d, i, 2] = h_abs * s_tmp[0]
            test_set[ind_d, i, 2:] = test_set[ind_d, i, 2:] + n # adding noise
            if M == 2 or M == 5:
                test_set[ind_d, i, 3] = 0 # giving 0 to the machine
        if M == 2 or M == 5:
            channel_set_genie[ind_d] = h_abs
        elif M == 16 or M == 4:
            h_abs_squared = pow(h[ind_d][0], 2) + pow(h[ind_d][1], 2)
            h_abs = pow(h_abs_squared, 0.5)
            h_abs = h_abs * (h[ind_d][0] / abs(h[ind_d][0]))
            channel_set_genie[ind_d] = h_abs
        channel_set_for_vis[ind_d] = h[ind_d] # no need...
        if if_cali:
            non_linearity_set_genie[ind_d, 0] = 1
            non_linearity_set_genie[ind_d, 1] = 0
        else:
            non_linearity_set_genie[ind_d, 0] = l_0[ind_d]
            non_linearity_set_genie[ind_d, 1] = l_1[ind_d]
    return test_set, channel_set_genie, channel_set_for_vis, non_linearity_set_genie


def generating_online_training_set(args, curr_dev_char, total_data_set, K_TR_accum, K_TR, K_TE, ind_T, M, var_array, var, mean_array0, mean_array1, mean_array3, mean_array5,
                            writer_per_dev_tot, if_cali, if_symm, meta_train_version, if_reuse_metatrain_pilot_symbols,
                            power, device): # contains meta-train, meta-test, which is also used for test-train
    # we can just set K_TE = 0 and make K_TR as our # training pilot
    # K_TR here is current iteration's adding pilot number (1 is most reasonable thinking now...)
    ## generating training dataset
    assert K_TE == 0
    if K_TE != 0:
        print('for online, K_TE should be 0')
        raise NotImplementedError

    print('generate training set for %d dev' %ind_T)

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
    # writer_per_dev = writer_per_dev_tot[ind_d]
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
                if i_curr % 16 == 0:
                    s_tmp = torch.FloatTensor([1, 1])
                elif i_curr % 16 == 1:
                    s_tmp = torch.FloatTensor([1, -3])
                elif i_curr % 16 == 2:
                    s_tmp = torch.FloatTensor([-3, 1])
                elif i_curr % 16 == 3:
                    s_tmp = torch.FloatTensor([3, 3])

                elif i_curr % 16 == 4:
                    s_tmp = torch.FloatTensor([3, 1])
                elif i_curr % 16 == 5:
                    s_tmp = torch.FloatTensor([-1, -1])
                elif i_curr % 16 == 6:
                    s_tmp = torch.FloatTensor([-1, -3])
                elif i_curr % 16 == 7:
                    s_tmp = torch.FloatTensor([-3, -3])

                elif i_curr % 16 == 8:
                    s_tmp = torch.FloatTensor([3, -1])
                elif i_curr % 16 == 9:
                    s_tmp = torch.FloatTensor([-1, 3])
                elif i_curr % 16 == 10:
                    s_tmp = torch.FloatTensor([-1, 1])
                elif i_curr % 16 == 11:
                    s_tmp = torch.FloatTensor([1, -1])

                elif i_curr % 16 == 12:
                    s_tmp = torch.FloatTensor([-3, -1])
                elif i_curr % 16 == 13:
                    s_tmp = torch.FloatTensor([1, 3])
                elif i_curr % 16 == 14:
                    s_tmp = torch.FloatTensor([-3, 3])
                elif i_curr % 16 == 15:
                    s_tmp = torch.FloatTensor([3, -3])

        train_set[ind_d, i, :2] = s_tmp  # generating symbol
        Noise = generating_noise_dist(var)
        n = Noise.sample()
        n = n.to(device)
        squared_amplitude = pow(s_tmp[0], 2) + pow(s_tmp[1],2)  # amplitude of the complex symbol, s_tmp[0] as real and s_tmp[1] as img
        amplitude = pow(squared_amplitude, 0.5)
        sin_phase = s_tmp[1] / amplitude
        cos_phase = s_tmp[0] / amplitude
        amplitude = amplitude * power
        distorted_amplitude = l_0[0] * pow(amplitude, 1) / (1 + l_1[0] * pow(amplitude,
                                                                                     2))  # l_0 is alpha and l_1 is beta  #+ l_1[ind_d]*pow(amplitude, 1) + l_3[ind_d] * pow(amplitude, 3) + l_5[ind_d] * pow(amplitude, 5)
        distorted_amplitude = torch.abs(
            distorted_amplitude)  # avoid - amplitude # check exactly for the def of dist.
        if if_cali:  # no distortion
            distorted_amplitude = amplitude
            s_tmp[0] = distorted_amplitude * cos_phase
            s_tmp[1] = distorted_amplitude * sin_phase
            # n = 0
            if if_symm:  # channel with +- 1
                if ind_d % 2 == 0:
                    h[0] = 1
                else:
                    h[0] = -1
        else:
            s_tmp[0] = distorted_amplitude * cos_phase
            s_tmp[1] = distorted_amplitude * sin_phase
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
    total_data_set[ind_T, K_TR_accum:K_TR_accum + K_TR,:] = train_set[ind_d, :, :]
    return total_data_set, curr_dev_char

