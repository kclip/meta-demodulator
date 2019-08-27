from __future__ import print_function
import torch
import numpy
from nets.deeper_linear import deeper_linear_net
from nets.deeper_linear import deeper_linear_net_prime
from data_gen.data_set import generating_training_set
from data_gen.data_set import generating_test_set
from training.train import meta_train
from training.train import test_test_mul_dev
from training.train import test_training_benchmark2
from torch.utils.tensorboard import SummaryWriter
import datetime
import pickle
import os
import argparse
import scipy.io as sio


def parse_args():
    parser = argparse.ArgumentParser(description='Meta Learning for Few Shot Learning')
    parser.add_argument('--mini_batch_size_meta', dest='mini_batch_size_meta', default=4, type=int) # mini batch size for meta-training
    parser.add_argument('--mini_batch_size_test_train', dest ='mini_batch_size_test_train', default=1, type=int) # mini batch size for training (adaptation) during meta-test 
    parser.add_argument('--mini_batch_size_bm2', dest='mini_batch_size_bm2', default=4, type=int) # mini batch size for joint training (bm2 stands for joint training)
    parser.add_argument('--mini_batch_size_meta_device', dest='mini_batch_size_meta_device', default=20, type=int) # number of meta-devices to consider for one meta-update
    parser.add_argument('--SNR_db', dest ='SNR_db', default=15, type=int) # SNR in db
    parser.add_argument('--lr_benchmark2', dest='lr_benchmark2', default=0.001, type=float) # learning rate for joint training
    parser.add_argument('--lr_testtraining', dest='lr_testtraining', default=0.001, type=float) # learning rate used for training (adaptation) during meta-test
    parser.add_argument('--lr_beta_settings', dest='lr_beta_settings', default=0.001, type=float) # learning rate used for meta-update
    parser.add_argument('--lr_alpha_settings', dest='lr_alpha_settings', default=0.001, type=float) # learning rate used during inner update
    parser.add_argument('--K_TR', dest='K_TR', default=16, type=int) # number of pilots for meta-training (training, support set, for local update)
    parser.add_argument('--K_TE', dest='K_TE', default=16, type=int) # number of pilots for meta-training (test, query set, for meta-update)
    parser.add_argument('--num_dev', dest='num_dev', default=100, type=int) # number of meta-devices
    parser.add_argument('--num_epochs_meta', dest='num_epochs_meta', default=60000, type=int) # number of epochs for meta-training
    parser.add_argument('--num_epochs_bm2', dest='num_epochs_bm2', default=300000, type=int) # number of epochs for joint training
    parser.add_argument('--K_TEST_TE', dest='K_TEST_TE', default=1000, type=int) # number of payload symbols for meta-test (number of pilots used for meta-test can be controlled in the main file - num_pilots_list (line 414))
    parser.add_argument('--num_devices_for_test', dest='num_devices_for_test', default=100, type=int) # number of devices for meta-test 
    parser.add_argument('--path_for_common_dir', dest='path_for_common_dir', default='/default_folder/default_subfolder/', type=str) # default path
    parser.add_argument('--path_for_meta_trained_net', dest='path_for_meta_trained_net', default=None, type=str)# to use previously generated meta-trained net
    parser.add_argument('--path_for_meta_training_set', dest='path_for_meta_training_set', default=None, type=str) # to use previously generated meta-training set
    parser.add_argument('--path_for_meta_test_set', dest='path_for_meta_test_set', default=None, type=str) # to use previously generated meta-test set
    parser.add_argument('--if_tanh', dest='if_relu', action='store_false', default=True)  # Relu or tanh
    parser.add_argument('--num_neurons_first', dest='num_neurons_first', default=10, type=int)  # number of neurons in first hidden layer
    parser.add_argument('--num_neurons_second', dest='num_neurons_second', default=None, type=int)  # number of neurons in second hidden layer
    parser.add_argument('--num_neurons_third', dest='num_neurons_third', default=None, type=int)  # number of neurons in third hidden layer
    parser.add_argument('--num_hidden_layer', dest='num_hidden_layer', default=1, type=int) # number of hidden layer (should coincide with num_neurons_first, num_neurons_second, num_neurons_third) considering at least one hidden layer, at most three hidden layer here

    parser.add_argument('--if_no_distortion', dest='if_no_distortion', action='store_true', default=False) # no hardware's non idealities
    parser.add_argument('--if_param_set_1', dest='if_param_set_1', action='store_true', default=False) # initialize all parameters 1
    parser.add_argument('--if_bias_set_0', dest='if_bias_set_0', action='store_true', default=False) # initialize bias as 0
    parser.add_argument('--version_of_channel_train', dest='version_of_channel_train', default=2, type=int) # only for PAM case #2 is channel with rayleigh #1 # 1 is channel with -1 and 1 w.p. 0.5 each
    parser.add_argument('--version_of_channel_test', dest='version_of_channel_test', default=2, type=int) #only for PAM case #2 is channel with rayleigh #1 # 1 is channel with -1 and 1 w.p. 0.5 each
    parser.add_argument('--modulation_order', dest='modulation_order', default=16, type=int) # 5: 4-PAM, 16: 16QAM
    parser.add_argument('--jac_calc', dest='jac_calc', default=200, type=int) # determines which meta-training solution to use: 1: MAML with full hessian computation (only works for inner loop = 2), 2: REPTILE, 200: MAML with hessian-vector approx. 300: FOMAML, 1001: CAVIA 
    parser.add_argument('--max_pilot_test', dest='max_pilot_test', default=1600, type=int) # maximum number of training set for meta-test 
    parser.add_argument('--reptile_inner_loop', dest='reptile_inner_loop', default=10, type=int) # number of local update + meta update for reptile
    parser.add_argument('--maml_inner_loop', dest='maml_inner_loop', default=8, type=int) # number of local update + meta update for maml, cavia, fomaml
    parser.add_argument('--if_cycle', dest='if_cycle', action='store_true', default=False) # sampling in a disjoint way for meta-training minibatch
    parser.add_argument('--if_mb_meta_change', dest='if_mb_meta_change', action='store_true',
                        default=False) # using different minibatch size for training and testing during meta-training
    parser.add_argument('--mini_batch_size_meta_train', dest='mini_batch_size_meta_train', default=1, type=int) # minibatch size for training (support set) during meta-training if 'if_mb_meta_change=True'
    parser.add_argument('--mini_batch_size_meta_test', dest='mini_batch_size_meta_test', default=1, type=int) # minibatch size for testing (query set) during meta-training if 'if_mb_meta_change=True'
    parser.add_argument('--mode_pilot_meta_training', dest='mode_pilot_meta_training', default=0, type=int) #pilot sequence for minibatch in training set & test set in meta-training # 0: fix seq. 1: random seq. 2: disjoint seq. 
    parser.add_argument('--if_unify_K_tr_K_te', dest='if_unify_K_tr_K_te', action='store_true', default=False) # do not divide meta-training set into two part
    parser.add_argument('--if_see_meta_inside_tensorboard', dest='if_see_meta_inside_tensorboard', action='store_true',
                        default=False)  # plot during meta-learning
    parser.add_argument('--if_test_train_no_permute', dest='if_test_train_no_permute', action='store_true', default=False) # if true, do not shuffle the sequence of pilots to make one minibatch
    parser.add_argument('--if_not_use_all_dev_in_one_epoch', dest='if_use_all_dev_in_one_epoch',
                        action='store_false',
                        default=True) # default: use all other devs before using same dev
    parser.add_argument('--if_not_bm2_fully_joint', dest='if_bm2_fully_joint',
                        action='store_false',
                        default=True) # if joint training has no division accounting for meta-device, if false: one minibatch only has data for one meta-device
    parser.add_argument('--if_continue_meta_training', dest='if_continue_meta_training',
                        action='store_true',
                        default=False) # if we want to continue meta-training from saved net
    parser.add_argument('--path_for_continueing_meta_net', dest='path_for_continueing_meta_net', default=None, type=str) # path of the saved net for continue training
    parser.add_argument('--power', dest='power',
                        default=0.5, type=float) # (args.power)^2 = actual power (energy)
    parser.add_argument('--if_fix_random_seed', dest='if_fix_random_seed',
                        action='store_true',
                        default=False) # fix random seed
    parser.add_argument('--random_seed', dest='random_seed',
                        default=1, type=int) # random seed
    parser.add_argument('--if_use_cuda', dest='if_use_cuda', action='store_true',
                        default=False)  # if use cuda
    parser.add_argument('--num_epochs_test_bm1', dest='num_epochs_test_bm1',
                        default=100000, type=int) # number of epochs for adaptation in meta-test fixed initialization case (bm1 stands for fixed initialization case)
    parser.add_argument('--num_epochs_test_bm2', dest='num_epochs_test_bm2',
                        default=100000, type=int) # number of epochs for adaptation in meta-test joint training case 
    parser.add_argument('--num_epochs_test_meta', dest='num_epochs_test_meta',
                        default=10000, type=int) # number of epochs for adaptation in meta-test meta-training case
    parser.add_argument('--if_save_test_trained_net_per_epoch', dest='if_save_test_trained_net_per_epoch', action='store_true', default=False) # if we want to save adapted net during adaptation during meta-test
    parser.add_argument('--if_no_bm1', dest='if_no_bm1',
                        action='store_true', default=False) # if we do not want to run fixed initialization case
    parser.add_argument('--if_no_bm2', dest='if_no_bm2',
                        action='store_true', default=False) # if we do not want to run joint training case
    parser.add_argument('--if_use_handmade_epsilon_hessian', dest='if_use_handmade_epsilon_hessian',
                        action='store_true', default=False)  # default: automatically calculated epsilon #Andrei, Neculai. "Accelerated conjugate gradient algorithm with finite difference Hessian/vector product approximation for unconstrained optimization." Journal of Computational and Applied Mathematics 230.2 (2009): 570-582.

    parser.add_argument('--epsilon_hessian', dest='epsilon_hessian',
                        default=0.1, type=float)  # for Hessian-vector approx. & using handmade epsilon hessian (only works when argsif_use_handmade_epsilon_hessian = True)

    parser.add_argument('--if_hess_vec_approx_for_one_inner_loop', dest='if_hess_vec_approx_for_one_inner_loop',
                        action='store_true', default=False) # do not unify pilots in meta-training set but to use in a divided manner

    parser.add_argument('--if_cavia', dest='if_cavia',
                        action='store_true', default=False) # if we want to run cavia (should be used with jac_calc = 1001)

    parser.add_argument('--num_context_para', dest='num_context_para',
                        default=10, type=int) # number of context parameters for cavia

    parser.add_argument('--path_for_bm2_net', dest='path_for_bm2_net', default=None, type=str) # path for saved joint trained net

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    print('Called with args:')
    print(args)
    if_cali = args.if_no_distortion
    if_symm = True # only active when args.if_no_distortion = True, binary channel
    if_bias = True # whether use bias for neural network
    if_reuse_testtrain_pilot_symbols = True # using fixed sequence & forcing successive M pilots compose constellation S
    if_reuse_metatrain_pilot_symbols = True # using fixed sequence & forcing successive M pilots compose constellation S
    meta_train_version = args.version_of_channel_train # only for PAM case #2 is channel with rayleigh #1 # 1 is channel with -1 and 1 w.p. 0.5 each
    test_train_version = args.version_of_channel_test # only for PAM case #2 is channel with rayleigh #1 # 1 is channel with -1 and 1 w.p. 0.5 each

    if_init_param_1 = args.if_param_set_1
    if_init_param_bias_0 = args.if_bias_set_0

    if args.mode_pilot_meta_training == 0:
        if_use_same_seq_both_meta_train_test = True
    elif args.mode_pilot_meta_training == 1:
        if_use_same_seq_both_meta_train_test = False
    else:
        if_use_same_seq_both_meta_train_test = None
    mini_batch_size_meta = args.mini_batch_size_meta
    mini_batch_size = args.mini_batch_size_test_train  # 2
    mini_batch_size_bm2 = args.mini_batch_size_bm2  # 2
    if args.num_dev < args.mini_batch_size_meta_device:
        sampled_device_num = 1
    else:
        sampled_device_num = args.mini_batch_size_meta_device
    SNR_db = args.SNR_db
    lr_benchmark2 = args.lr_benchmark2
    lr_testtraining = args.lr_testtraining
    lr_beta_settings = args.lr_beta_settings
    lr_alpha_settings = args.lr_alpha_settings
    K_TR = args.K_TR
    K_TE = args.K_TE
    num_dev_list = [args.num_dev]

    if args.path_for_meta_trained_net is not None:
        num_epochs = 1
    else:
        num_epochs = args.num_epochs_meta

    num_epochs_bm2 = args.num_epochs_bm2
    K_TEST_TE = args.K_TEST_TE
    num_devices_for_test = args.num_devices_for_test
    path_for_common_dir = args.path_for_common_dir
    path_for_meta_trained_net = args.path_for_meta_trained_net
    path_for_meta_training_set = args.path_for_meta_training_set
    path_for_meta_test_set = args.path_for_meta_test_set
    M = args.modulation_order
    if_relu = args.if_relu
    jac_calc = args.jac_calc
    reptile_inner_loop = args.reptile_inner_loop
    if_cycle = args.if_cycle
    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    common_dir = './' + path_for_common_dir + curr_time + '/'
    net_dir_for_test_result = common_dir + 'test_result/'
    net_dir_over_num_pilots_bm1 = common_dir + 'w.r.t_pilots_bm1'
    net_dir_over_num_pilots_theoretic = common_dir + 'w.r.t_pilots_theoretic'
    writer_over_num_pilots_bm1 = SummaryWriter(net_dir_over_num_pilots_bm1)
    writer_over_num_pilots_theoretic = SummaryWriter(net_dir_over_num_pilots_theoretic)
    net_dir_over_num_pilots_bm2 = common_dir + 'w.r.t_pilots_bm2'
    writer_over_num_pilots_bm2 = SummaryWriter(net_dir_over_num_pilots_bm2)
    net_dir_over_num_pilots_meta = common_dir + 'w.r.t_pilots_meta'
    writer_over_num_pilots_meta = SummaryWriter(net_dir_over_num_pilots_meta)
    if args.if_use_cuda:
        device = torch.device("cuda")
    else:    
        device = torch.device("cpu")

    if args.if_fix_random_seed:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_seed)

    if M == 16 or M == 5:
        if args.num_neurons_third is not None:
            net_name = 'deep_linear_net_with_3_hidden_layer'
            assert args.num_hidden_layer == 3
        elif args.num_neurons_second is not None:
            net_name = 'deep_linear_net_with_2_hidden_layer'
            net_name = 'deeper_linear_net'
            assert args.num_hidden_layer == 2
        elif args.num_neurons_first is not None:
            net_name = 'deep_linear_net_with_1_hidden_layer'
            assert args.num_hidden_layer == 1
        else:
            raise NotImplementedError
        M_tmp = M
        net = deeper_linear_net(args = args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first, num_neurons_second=args.num_neurons_second, num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net.cuda()
        net_prime = deeper_linear_net_prime(args = args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first, num_neurons_second=args.num_neurons_second, num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net_prime.cuda()
        net_for_testtraining = deeper_linear_net(args = args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first, num_neurons_second=args.num_neurons_second, num_neurons_third=args.num_neurons_third, if_bias=if_bias,
                                               if_relu=if_relu)  # dummy net for loading learned network
        if args.if_use_cuda:
            net_for_testtraining.cuda()
        net_benchmark = deeper_linear_net(args = args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first, num_neurons_second=args.num_neurons_second, num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net_benchmark.cuda()
        net_benchmark_2 = deeper_linear_net(args = args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first, num_neurons_second=args.num_neurons_second, num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net_benchmark_2.cuda()
    else:
        raise NotImplementedError
    i = 0
    if if_init_param_1 and if_init_param_bias_0:
        for f in net.parameters():
            if i%2 == 0:
                f.data.fill_(1)
            else:
                f.data.fill_(0)
            i = i + 1
    elif if_init_param_1 and not if_init_param_bias_0:
        for f in net.parameters():
            f.data.fill_(1)

    init_model_PATH = common_dir + 'init_model/' + net_name  # random weight used for benchmark 1
    if not os.path.exists(common_dir + 'init_model/'):
        os.mkdir(common_dir + 'init_model/')
    torch.save(net.state_dict(), init_model_PATH)  # always start from this init. realization

    for num_dev in num_dev_list: # in case we want set different number of meta-devices, currently just do for once for given whole meta-devices
        net.load_state_dict(torch.load(init_model_PATH))
        net_prime.load_state_dict(torch.load(init_model_PATH))
        ## For a certain device
        ## meta-training set generation
        #################################### settings for training dataset ###############################
        var_array = numpy.ones(num_dev, dtype=numpy.float64)
        ## get noise variance w.r.t. given SNR, we are fixing power
        if M == 5:
            power = args.power
            noise_variance = pow(10, numpy.log10(5*pow(power,2)) - SNR_db / 10)
        elif M == 4:
            power = args.power
            noise_variance_real_and_im = 2 * pow(power,2)/(pow(10,SNR_db/10))
            noise_variance = noise_variance_real_and_im/2
        elif M == 16:
            power = args.power
            noise_variance_real_and_im = (pow(power,2) * (10))/(pow(10,SNR_db/10))
            noise_variance = noise_variance_real_and_im/2
        channel_variance = 0.5
        distortion_range0 = 0
        distortion_range1 = 0.05
        distortion_range3 = 0
        distortion_range5 = 0
        distortion_mean0 = 4
        distortion_mean1 = 0.1
        distortion_mean3 = 0
        distortion_mean5 = 0
        var = noise_variance  # variance of noise
        var_array = channel_variance * var_array  # variance of channel
        mean_array0 = numpy.random.uniform(distortion_mean0 - distortion_range0, distortion_mean0 + distortion_range0,
                                           num_dev)
        mean_array1 = numpy.random.uniform(distortion_mean1 - distortion_range1, distortion_mean1 + distortion_range1,
                                           num_dev)
        mean_array3 = numpy.random.uniform(distortion_mean3 - distortion_range3, distortion_mean3 + distortion_range3,
                                           num_dev)
        mean_array5 = numpy.random.uniform(distortion_mean5 - distortion_range5, distortion_mean5 + distortion_range5,
                                           num_dev)
        name_of_the_net_for_net_dir = 'num_dev:' + str(num_dev) + 'M_order:' + str(M) + 'model_type:' + net_name + 'noise_variance:' + str(
            noise_variance) + 'channel_variance:' + str(channel_variance)
        net_dir_meta = common_dir + 'TB/' + 'meta_training_set/' + 'num_dev/' + name_of_the_net_for_net_dir

        net_dir_per_num_dev = net_dir_meta + 'total dev num:' + str(num_dev)

        if args.path_for_meta_trained_net is None: # if we are loading previous set (eval mode), no need to see writer in meta-training
            writer_per_num_dev = SummaryWriter(net_dir_per_num_dev)

        # for writer per each device in current number of devices set
        if args.path_for_meta_trained_net is None: # only used for meta-training
            writer_per_dev_tot = []
            if args.if_see_meta_inside_tensorboard:
                for ind_writer in range(num_dev):
                    net_dir_per_dev = net_dir_meta + 'total dev num:' + str(num_dev) + 'dev index:' + str(ind_writer)
                    writer_per_dev_tmp = SummaryWriter(net_dir_per_dev)
                    writer_per_dev_tot.append(writer_per_dev_tmp)
        if path_for_meta_training_set is None:
            # only generate meta-training set if path is None
            writer_per_dev_tot = []
            train_set = generating_training_set(K_TR, K_TE, num_dev, M, var_array, var, mean_array0, mean_array1, mean_array3,
                                                mean_array5,
                                                writer_per_dev_tot, if_cali, if_symm, meta_train_version, if_reuse_metatrain_pilot_symbols, power, args, device)
            name_of_current_train_set = common_dir + 'meta_training_set/' + 'num_dev:' + str(
                num_dev) + '/' + name_of_the_net_for_net_dir + '.pckl'
            if not os.path.exists(common_dir + 'meta_training_set/' + 'num_dev:' + str(num_dev) + '/'):
                os.makedirs(common_dir + 'meta_training_set/' + 'num_dev:' + str(num_dev) + '/')
            f_for_train_set = open(name_of_current_train_set, 'wb')
            pickle.dump(train_set, f_for_train_set)
            f_for_train_set.close()

        if path_for_meta_training_set is not None:
            print('load previously generated meta-training set')
            meta_training_set_path = path_for_meta_training_set
            f_meta_set = open(meta_training_set_path, 'rb')
            train_set = pickle.load(f_meta_set)
            f_meta_set.close()
        else:
            print('generate new meta-training set')

        # meta-training
        ######################################################################################################
        print('start meta training for %d training device' % num_dev)
        print('start meta training for %d epochs' % num_epochs)
        print('start meta training for %d pilots (meta training)' % K_TR)
        print('start meta training for %d pilots (meta test)' % K_TE)
        # meta_train(num_epochs)
        name_of_the_net_for_meta = 'meta' + name_of_the_net_for_net_dir

        saved_model_PATH_meta_intermediate = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'meta/' + 'num_dev:' + str(
            num_dev) + '/' + name_of_the_net_for_meta + '/'

        if not os.path.exists(saved_model_PATH_meta_intermediate):
            os.makedirs(saved_model_PATH_meta_intermediate)
        if M == 17:
            raise NotImplementedError
        else:
            if args.path_for_meta_trained_net is None:
                #only do meta-train when no net given
                meta_train(M, num_epochs, num_dev, net, net_prime, train_set, K_TR, K_TE, device, writer_per_num_dev,
                           writer_per_dev_tot, saved_model_PATH_meta_intermediate, lr_alpha_settings, lr_beta_settings,
                           mini_batch_size_meta, if_use_same_seq_both_meta_train_test, sampled_device_num, jac_calc, reptile_inner_loop, if_cycle, args)

        print('end meta training for %d training device' % num_dev)

        ## saving the trained model

        saved_model_PATH_meta = saved_model_PATH_meta_intermediate + name_of_the_net_for_meta
        if not os.path.exists(saved_model_PATH_meta_intermediate):
            os.makedirs(
                saved_model_PATH_meta_intermediate)
        if not os.path.exists(saved_model_PATH_meta_intermediate+'/real/'):
            os.makedirs(
                saved_model_PATH_meta_intermediate+'/real/')
        if not os.path.exists(saved_model_PATH_meta_intermediate +'/im/'):
            os.makedirs(
                saved_model_PATH_meta_intermediate+'/im/')
        torch.save(net.state_dict(), saved_model_PATH_meta)
        if path_for_meta_trained_net is not None:
            print('load previously meta-trained network')
            saved_model_PATH_meta = path_for_meta_trained_net
        else:
            print('use meta training from raw')

        # Joint training (BENCHMARK 2)
        ######################################################################################################
        net_benchmark_2.load_state_dict(torch.load(init_model_PATH))  # initialize
        train_set_for_benchmark2 = train_set
        name_of_the_net_for_benchmark2 = 'benchmark2 ' + name_of_the_net_for_net_dir
        dir_for_writer_benchmark2 = common_dir + 'TB/' + name_of_the_net_for_benchmark2
        writer_benchmark2 = SummaryWriter(dir_for_writer_benchmark2)
        online_mode = 0 # offline
        saved_model_PATH_bm2_intermediate = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(
                num_dev) + '/' + name_of_the_net_for_benchmark2 + '/'
        if not os.path.exists(saved_model_PATH_bm2_intermediate):
            os.makedirs(saved_model_PATH_bm2_intermediate)
        if not args.if_no_bm2:
            if args.path_for_bm2_net is None:
                test_training_benchmark2(args, M, lr_benchmark2, mini_batch_size_bm2, net_benchmark_2, train_set_for_benchmark2, K_TR, K_TE, num_epochs_bm2,
                                         writer_benchmark2, online_mode, device, saved_model_PATH_bm2_intermediate)
            else:
                net_benchmark_2.load_state_dict(torch.load(args.path_for_bm2_net))
        if not args.if_no_bm2:
            saved_model_PATH_benchmark2 = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(
                num_dev) + '/' + name_of_the_net_for_benchmark2 + 'final'
            if not os.path.exists(common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(
                    num_dev) + '/'):
                os.makedirs(
                    common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(num_dev) + '/')
            if not os.path.exists(common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(
                    num_dev) + '/' + 'real/'):
                os.makedirs(
                    common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(num_dev) + '/' +'real/')
            if not os.path.exists(common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(
                    num_dev) + '/' + 'im/'):
                os.makedirs(
                    common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark2/' + 'num_dev:' + str(num_dev) + '/' +'im/')
            torch.save(net_benchmark_2.state_dict(),
                       saved_model_PATH_benchmark2)  # getting initial point by conventional training

        # Fixed initialization (BENCHMARK 1)
        ######################################################################################################
        net_benchmark.load_state_dict(torch.load(init_model_PATH))
        name_of_the_net_for_benchmark1 = 'benchmark1 ' + name_of_the_net_for_net_dir
        saved_model_PATH_benchmark1 = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark1/' + 'num_dev:' + str(
            num_dev) + '/' + name_of_the_net_for_benchmark1
        if not os.path.exists(common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark1/' + 'num_dev:' + str(
                num_dev) + '/'):
            os.makedirs(
                common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'benchmark1/' + 'num_dev:' + str(num_dev) + '/')
        torch.save(net_benchmark.state_dict(),
                   saved_model_PATH_benchmark1)  # getting initial point by conventional training
        ############################## Number of pilots list for meta-test ###################################
        num_pilots_list = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
        ######################################################################################################
        # for saving experiment results
        max_num_pilots = args.max_pilot_test
        test_error_rate_list_meta = []
        test_error_rate_list_bm1 = []
        test_error_rate_list_bm2 = []

        test_training_set_over_meta_test_devs = []
        channel_set_genie_over_meta_test_devs = []
        channel_set_for_vis_over_meta_test_devs = []
        non_linearity_set_genie_over_meta_test_devs = []

        # generating meta-test set (pilots and payload data for each meta-test devices)
        for i in range(num_devices_for_test):
            var_array = [channel_variance]
            var = noise_variance
            mean_array0 = numpy.random.uniform(distortion_mean0 - distortion_range0,
                                               distortion_mean0 + distortion_range0,
                                               1)
            mean_array1 = numpy.random.uniform(distortion_mean1 - distortion_range1,
                                               distortion_mean1 + distortion_range1,
                                               1)
            mean_array3 = numpy.random.uniform(distortion_mean3 - distortion_range3,
                                               distortion_mean3 + distortion_range3,
                                               1)
            mean_array5 = numpy.random.uniform(distortion_mean5 - distortion_range5,
                                               distortion_mean5 + distortion_range5,
                                               1)

            ## generating test training dataset
            K_TEST_TR = max_num_pilots
            if path_for_meta_test_set is None:
                curr_dev_char = None # for online
                print('generating meta-test set') #only generate meta-test set if path is None
                test_training_set, channel_set_genie, channel_set_for_vis, non_linearity_set_genie = generating_test_set(curr_dev_char, K_TEST_TR, K_TEST_TE, 1,
                                                                                                    M, var_array, var,
                                                                                                    mean_array0,
                                                                                                    mean_array1,
                                                                                                    mean_array3,
                                                                                                    mean_array5,
                                                                                                    i,
                                                                                                    if_cali, if_symm,
                                                                                                    test_train_version,
                                                                                                    if_reuse_testtrain_pilot_symbols,
                                                                                                    power, device)
                test_training_set_over_meta_test_devs.append(test_training_set)
                channel_set_genie_over_meta_test_devs.append(channel_set_genie)
                channel_set_for_vis_over_meta_test_devs.append(channel_set_for_vis)
                non_linearity_set_genie_over_meta_test_devs.append(non_linearity_set_genie)

        if not os.path.exists(common_dir + 'meta_test_set/'):
            os.makedirs(common_dir + 'meta_test_set/')


        name_of_current_channel_set = common_dir + 'meta_test_set/' + 'channel_set_meta_test.pckl'
        f_for_channel_meta_test = open(name_of_current_channel_set, 'wb')
        pickle.dump(channel_set_for_vis_over_meta_test_devs, f_for_channel_meta_test)
        f_for_channel_meta_test.close()

        name_of_current_channel_abs_set = common_dir + 'meta_test_set/' + 'channel_set_abs_meta_test.pckl'
        f_for_channel_abs_meta_test = open(name_of_current_channel_abs_set, 'wb')
        pickle.dump(channel_set_genie_over_meta_test_devs, f_for_channel_abs_meta_test)
        f_for_channel_abs_meta_test.close()

        name_of_current_testtraining_set= common_dir + 'meta_test_set/' + 'testtraining_set.pckl'
        f_for_test_train_set = open(name_of_current_testtraining_set, 'wb')
        pickle.dump(test_training_set_over_meta_test_devs, f_for_test_train_set)
        f_for_test_train_set.close()

        name_of_current_nonlinear_set = common_dir + 'meta_test_set/' + 'nonlinear_set.pckl'
        f_for_nonlinearset = open(name_of_current_nonlinear_set, 'wb')
        pickle.dump(non_linearity_set_genie_over_meta_test_devs, f_for_nonlinearset)
        f_for_nonlinearset.close()

        if path_for_meta_test_set is not None:
            print('load previously generated meta-test set')
            meta_test_channel_path = path_for_meta_test_set + '/' + 'channel_set_meta_test.pckl'
            meta_test_channel_abs_path = path_for_meta_test_set + '/' + 'channel_set_abs_meta_test.pckl'
            meta_test_nonlinear_path = path_for_meta_test_set + '/' + 'nonlinear_set.pckl'
            meta_test_testtrain_set_path =  path_for_meta_test_set + '/' + 'testtraining_set.pckl'

            f_meta_test_channel = open(meta_test_channel_path, 'rb')
            f_meta_test_channel_abs = open(meta_test_channel_abs_path, 'rb')
            f_meta_test_nonlinear = open(meta_test_nonlinear_path, 'rb')
            f_meta_test_testtrain_set = open(meta_test_testtrain_set_path, 'rb')

            channel_set_genie_over_meta_test_devs = pickle.load(f_meta_test_channel_abs)
            channel_set_for_vis_over_meta_test_devs = pickle.load(f_meta_test_channel)
            non_linearity_set_genie_over_meta_test_devs = pickle.load(f_meta_test_nonlinear)
            test_training_set_over_meta_test_devs = pickle.load(f_meta_test_testtrain_set)

            f_meta_test_channel.close()
            f_meta_test_nonlinear.close()
            f_meta_test_testtrain_set.close()
        else:
            print('generate new meta-test set')
        uncertainty_loss_array_bm1 = torch.zeros(max(num_pilots_list)+1,
                                                  num_devices_for_test)
        error_rate_array_bm1 = torch.zeros(max(num_pilots_list)+1, num_devices_for_test)
        uncertainty_loss_array_bm2 = torch.zeros(max(num_pilots_list)+1,
                                                  num_devices_for_test)
        error_rate_array_bm2 = torch.zeros(max(num_pilots_list)+1, num_devices_for_test)
        uncertainty_loss_array_meta = torch.zeros(max(num_pilots_list)+1, num_devices_for_test)
        error_rate_array_meta = torch.zeros(max(num_pilots_list)+1, num_devices_for_test)

        for num_pilots in num_pilots_list:
            save_test_result_dict = {}
            print('num_pilots: ', num_pilots)
            num_test_iter = 1  # iter for one device (fix channel gain, dist. coefficients, but generate data set each time)
            # currently no iteration for a one device
            K_TEST_TR = num_pilots
            total_error_num_bm1 = 0
            total_pilot_num_bm1 = 0
            total_error_num_bm2 = 0
            total_pilot_num_bm2 = 0
            total_error_num_meta = 0
            total_pilot_num_meta = 0
            theoretical_bound = 0
            final_loss_bm1 = 0
            final_loss_bm2 = 0
            final_loss_meta = 0
            for i in range(num_devices_for_test):
                print('curr dev for test: ', i)
                channel_set_genie = channel_set_genie_over_meta_test_devs[i]
                non_linearity_set_genie = non_linearity_set_genie_over_meta_test_devs[i]
                tmp_test_training_set = test_training_set_over_meta_test_devs[i]
                tmp_test_training_set_only_train = tmp_test_training_set[:, :K_TEST_TR, :]
                tmp_test_training_set_only_val = tmp_test_training_set[:, max_num_pilots:, :]
                test_training_set = torch.zeros(1, K_TEST_TR + K_TEST_TE, 4)
                test_training_set = test_training_set.to(device)
                test_training_set[:, :K_TEST_TR, :] = tmp_test_training_set_only_train
                test_training_set[:, K_TEST_TR:, :] = tmp_test_training_set_only_val
                dir_benchmark = common_dir + 'saved_model/' + 'after_meta_testing_set/' + 'benchmark1/' + 'num pilots:' + str(num_pilots) + '/' +  'num_dev:' + str(
                    num_dev) + '/' + 'ind_dev_test_train:' + str(i) + '/'
                dir_benchmark2 = common_dir + 'saved_model/' + 'after_meta_testing_set/' + 'benchmark2/' + 'num pilots:' + str(num_pilots) + '/' + 'num_dev:' + str(
                    num_dev) + '/' + 'ind_dev_test_train:' + str(i) + '/'
                dir_meta = common_dir + 'saved_model/' + 'after_meta_testing_set/' + 'meta/' + 'num pilots:' + str(num_pilots) + '/' +  'num_dev:' + str(
                    num_dev) + '/' + 'ind_dev_test_train:' + str(i) + '/'
                if not os.path.exists(dir_benchmark):
                    os.makedirs(dir_benchmark)
                if not os.path.exists(dir_benchmark2):
                    os.makedirs(dir_benchmark2)
                if not os.path.exists(dir_meta):
                    os.makedirs(dir_meta)
                save_PATH_benchmark = dir_benchmark + name_of_the_net_for_benchmark1
                save_PATH_benchmark2 = dir_benchmark2 + name_of_the_net_for_benchmark2
                save_PATH_meta = dir_meta + name_of_the_net_for_meta

                ###################################
                if not args.if_no_bm1:
                    num_epochs_test = args.num_epochs_test_bm1
                    uncertainty_loss_tmp_bm1, final_loss_tmp_bm1, total_error_num_tmp_bm1, total_pilot_num_tmp_bm1, theoretical_bound_bm1= test_test_mul_dev(args, net_for_testtraining,
                                                                                             num_test_iter, mini_batch_size, M, lr_testtraining,
                                                                                             K_TEST_TR,
                                                                                             K_TEST_TE, num_epochs_test,
                                                                                             saved_model_PATH_benchmark1,
                                                                                             save_PATH_benchmark,
                                                                                             test_training_set, channel_set_genie, non_linearity_set_genie, True, noise_variance, power, device)

                    print('bm1 curr dev error rate: ', (total_error_num_tmp_bm1 / total_pilot_num_tmp_bm1))

                if not args.if_no_bm2:
                    num_epochs_test = args.num_epochs_test_bm2
                    uncertainty_loss_tmp_bm2, final_loss_tmp_bm2, total_error_num_tmp_bm2, total_pilot_num_tmp_bm2, theoretical_bound_bm2 = test_test_mul_dev(args, net_for_testtraining,
                                                                                             num_test_iter, mini_batch_size, M, lr_testtraining,
                                                                                             K_TEST_TR,
                                                                                             K_TEST_TE, num_epochs_test,
                                                                                             saved_model_PATH_benchmark2,
                                                                                             save_PATH_benchmark2,
                                                                                             test_training_set, channel_set_genie, non_linearity_set_genie, False, noise_variance, power, device)

                    print('bm2 curr dev error rate: ', (total_error_num_tmp_bm2 / total_pilot_num_tmp_bm2))

                num_epochs_test_meta = args.num_epochs_test_meta
                uncertainty_loss_tmp_meta, final_loss_tmp_meta, total_error_num_tmp_meta, total_pilot_num_tmp_meta, theoretical_bound_meta = test_test_mul_dev(
                    args, net_for_testtraining,
                    num_test_iter, mini_batch_size, M, lr_testtraining,
                    K_TEST_TR,
                    K_TEST_TE, num_epochs_test_meta,
                    saved_model_PATH_meta,
                    save_PATH_meta,
                    test_training_set, channel_set_genie, non_linearity_set_genie, False, noise_variance,
                    power, device)
                print('epcohs_meta: ', num_epochs_test_meta)
                print('meta curr dev error rate: ', (total_error_num_tmp_meta / total_pilot_num_tmp_meta))
                if not args.if_no_bm1:
                    total_error_num_bm1 = total_error_num_bm1 + total_error_num_tmp_bm1
                    total_pilot_num_bm1 = total_pilot_num_bm1 + total_pilot_num_tmp_bm1
                    error_rate_array_bm1[num_pilots, i] = total_error_num_tmp_bm1/total_pilot_num_tmp_bm1
                    uncertainty_loss_array_bm1[num_pilots, i] = uncertainty_loss_tmp_bm1
                    theoretical_bound = theoretical_bound + theoretical_bound_bm1  # average for all dev.s
                if not args.if_no_bm2:
                    total_error_num_bm2 = total_error_num_bm2 + total_error_num_tmp_bm2
                    total_pilot_num_bm2 = total_pilot_num_bm2 + total_pilot_num_tmp_bm2
                    error_rate_array_bm2[num_pilots, i] = total_error_num_tmp_bm2 / total_pilot_num_tmp_bm2
                    uncertainty_loss_array_bm2[num_pilots, i] = uncertainty_loss_tmp_bm2
                total_error_num_meta = total_error_num_meta + total_error_num_tmp_meta
                total_pilot_num_meta = total_pilot_num_meta + total_pilot_num_tmp_meta
                error_rate_array_meta[num_pilots, i] = total_error_num_tmp_meta / total_pilot_num_tmp_meta
                uncertainty_loss_array_meta[num_pilots, i] = uncertainty_loss_tmp_meta
            if not args.if_no_bm1:
                total_error_rate_bm1 = total_error_num_bm1 / total_pilot_num_bm1
            if not args.if_no_bm2:
                total_error_rate_bm2 = total_error_num_bm2 / total_pilot_num_bm2
            total_error_rate_meta = total_error_num_meta / total_pilot_num_meta
            theoretical_bound = theoretical_bound/num_devices_for_test

            if not args.if_no_bm1:
                writer_over_num_pilots_bm1.add_scalar('total_error_rate w.r.t. num pilots', total_error_rate_bm1, num_pilots)
                test_error_rate_list_bm1.append(total_error_rate_bm1)
            if not args.if_no_bm2:
                writer_over_num_pilots_bm2.add_scalar('total_error_rate w.r.t. num pilots', total_error_rate_bm2, num_pilots)
                test_error_rate_list_bm2.append(total_error_rate_bm2)
            writer_over_num_pilots_meta.add_scalar('total_error_rate w.r.t. num pilots', total_error_rate_meta, num_pilots)
            test_error_rate_list_meta.append(total_error_rate_meta)
            writer_over_num_pilots_theoretic.add_scalar('total_error_rate w.r.t. num pilots', theoretical_bound, num_pilots)

            print('with pilot ', num_pilots, ' average error rate with meta-net: ', total_error_rate_meta)

            save_test_result_dict['bm1_error_rate_wrt_pilot'] = test_error_rate_list_bm1
            save_test_result_dict['bm2_error_rate_wrt_pilot'] = test_error_rate_list_bm2
            save_test_result_dict['meta_error_rate_wrt_pilot'] = test_error_rate_list_meta

            save_test_result_dict['bm1_error_rate_array'] = error_rate_array_bm1.data.numpy()
            save_test_result_dict['bm2_error_rate_array'] = error_rate_array_bm2.data.numpy()
            save_test_result_dict['meta_error_rate_array'] = error_rate_array_meta.data.numpy()

            save_test_result_dict['meta_uncertainty_loss_array'] = uncertainty_loss_array_meta.data.numpy()
            save_test_result_dict['bm1_uncertainty_loss_array'] = uncertainty_loss_array_bm1.data.numpy()
            save_test_result_dict['bm2_uncertainty_loss_array'] = uncertainty_loss_array_bm2.data.numpy()

            # save result as matfile
            # save result per num_pilots as accumulated dataset
            accum_result_up_to_curr_pilot_path = net_dir_for_test_result + 'curr_num_pilots' + str(num_pilots) + '/' + 'test_result.mat'

            os.makedirs(net_dir_for_test_result + 'curr_num_pilots' + str(num_pilots) + '/')

            sio.savemat(accum_result_up_to_curr_pilot_path, save_test_result_dict)

            accum_result_up_to_curr_pilot_path_pickle = net_dir_for_test_result + 'curr_num_pilots' + str(
                num_pilots) + '/' + 'test_result.pckl'
            f_for_test_result = open(accum_result_up_to_curr_pilot_path_pickle, 'wb')
            pickle.dump(save_test_result_dict, f_for_test_result)
            f_for_test_result.close()




