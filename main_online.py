from __future__ import print_function
import torch
import numpy
from nets.deeper_linear import deeper_linear_net
from nets.deeper_linear import deeper_linear_net_prime
from data_gen.data_set import generating_online_training_set
from data_gen.data_set import generating_test_set
from training.train import meta_train_online
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
    parser.add_argument('--mini_batch_size_test_train', dest='mini_batch_size_test_train', default=4, type=int) # mini batch size for training (adaptation) during meta-test
    parser.add_argument('--mini_batch_size_bm2', dest='mini_batch_size_bm2', default=4, type=int) # mini batch size for joint training
    parser.add_argument('--mini_batch_size_meta_device', dest='mini_batch_size_meta_device', default=20, type=int) # number of meta-devices to consider for one meta-update
    parser.add_argument('--SNR_db', dest='SNR_db', default=20, type=int) # SNR in db
    parser.add_argument('--lr_beta_settings', dest='lr_beta_settings', default=0.001, type=float) # learning rate used for meta-update
    parser.add_argument('--lr_alpha_settings', dest='lr_alpha_settings', default=0.1, type=float) # learning rate used during inner update
    parser.add_argument('--num_epochs_meta', dest='num_epochs_meta', default=500, type=int) # number of epochs for meta-training
    parser.add_argument('--K_TEST_TE', dest='K_TEST_TE', default=10000, type=int) # number of payload symbols for meta-test (number of pilots used for meta-test can be controlled in the main file - num_pilots_list (line 413))
    parser.add_argument('--path_for_common_dir', dest='path_for_common_dir',
                        default='/default_folder/default_subfolder/', type=str) # default path
    parser.add_argument('--path_for_meta_training_set', dest='path_for_meta_training_set', default=None, type=str) # to use previously generated meta-training set (as if we are facing same situation for fair comparison in online)
    parser.add_argument('--path_for_meta_test_set', dest='path_for_meta_test_set', default=None, type=str) # to use previously generated meta-test set (payload) (as if we are facing same situation for fair comparison in online)
    parser.add_argument('--if_tanh', dest='if_relu', action='store_false', default=True) # Relu or tanh
    parser.add_argument('--num_neurons_first', dest='num_neurons_first', default=10, type=int)  # number of neurons in first hidden layer
    parser.add_argument('--num_neurons_second', dest='num_neurons_second', default=None, type=int)  # number of neurons in second hidden layer
    parser.add_argument('--num_neurons_third', dest='num_neurons_third', default=None, type=int)  # number of neurons in third hidden layer
    parser.add_argument('--num_hidden_layer', dest='num_hidden_layer', default=1, type=int) # number of hidden layer (should coincide with num_neurons_first, num_neurons_second, num_neurons_third) considering at least one hidden layer, at most three hidden layer here
    parser.add_argument('--if_no_distortion', dest='if_no_distortion', action='store_true', default=False) # no hardware's non idealities
    parser.add_argument('--if_param_set_1', dest='if_param_set_1', action='store_true', default=False) # initialize all parameters 1
    parser.add_argument('--if_bias_set_0', dest='if_bias_set_0', action='store_true', default=False) # initialize bias as 0
    parser.add_argument('--version_of_channel_train', dest='version_of_channel_train', default=2, type=int) # only for PAM case #2 is channel with rayleigh #1 # 1 is channel with -1 and 1 w.p. 0.5 each
    parser.add_argument('--version_of_channel_test', dest='version_of_channel_test', default=2, type=int) # only for PAM case #2 is channel with rayleigh #1 # 1 is channel with -1 and 1 w.p. 0.5 each
    parser.add_argument('--modulation_order', dest='modulation_order', default=16, type=int) # 5: 4-PAM, 16: 16QAM
    parser.add_argument('--jac_calc', dest='jac_calc', default=200, type=int) # determines which meta-training solution to use: 1: MAML with full hessian computation (only works for inner loop = 2), 2: REPTILE, 200: MAML with hessian-vector approx. 300: FOMAML, 1001: CAVIA
    parser.add_argument('--reptile_inner_loop', dest='reptile_inner_loop', default=2, type=int) # number of local update + meta update for reptile
    parser.add_argument('--maml_inner_loop', dest='maml_inner_loop', default=2, type=int) # number of local update + meta update for maml, cavia, fomaml
    parser.add_argument('--if_cycle', dest='if_cycle', action='store_true', default=False) # sampling in a disjoint way for meta-training minibatch
    parser.add_argument('--if_mb_meta_change', dest='if_mb_meta_change', action='store_true',
                        default=False) # using different minibatch size for training and testing during meta-training
    parser.add_argument('--mini_batch_size_meta_train', dest='mini_batch_size_meta_train', default=4, type=int) # minibatch size for training (support set) during meta-training if 'if_mb_meta_change=True'
    parser.add_argument('--mini_batch_size_meta_test', dest='mini_batch_size_meta_test', default=4, type=int) # minibatch size for testing (query set) during meta-training if 'if_mb_meta_change=True'
    parser.add_argument('--mode_pilot_meta_training', dest='mode_pilot_meta_training', default=0,
                        type=int)  #pilot sequence for minibatch in training set & test set in meta-training # 0: fix seq. 1: random seq. 2: disjoint seq.
    parser.add_argument('--if_see_meta_inside_tensorboard', dest='if_see_meta_inside_tensorboard', action='store_true',
                        default=False)  # plot during meta-learning
    parser.add_argument('--if_test_train_no_permute', dest='if_test_train_no_permute', action='store_true',
                        default=None) # if true, do not shuffle the sequence of pilots to make one minibatch
    parser.add_argument('--if_not_use_all_dev_in_one_epoch', dest='if_use_all_dev_in_one_epoch',
                        action='store_false',
                        default=True) # default: use all other devs before using same dev
    parser.add_argument('--if_not_bm2_fully_joint', dest='if_bm2_fully_joint',
                        action='store_false',
                        default=True)  # if joint training has no division accounting for meta-device, if false: one minibatch only has data for one meta-device
    parser.add_argument('--if_continue_meta_training', dest='if_continue_meta_training',
                        action='store_true',
                        default=False) # if we want to continue meta-training from saved net
    parser.add_argument('--path_for_continueing_meta_net', dest='path_for_continueing_meta_net', default=None, type=str) # path of the saved net for continue training
    parser.add_argument('--power', dest='power',
                        default=1, type=float)  # (args.power)^2 = actual power (energy)
    parser.add_argument('--if_fix_random_seed', dest='if_fix_random_seed',
                        action='store_true',
                        default=False) # fix random seed
    parser.add_argument('--random_seed', dest='random_seed',
                        default=1, type=int) # random seed
    parser.add_argument('--if_use_cuda', dest='if_use_cuda', action='store_true',
                        default=False)  # if use cuda
    parser.add_argument('--if_save_test_trained_net_per_epoch', dest='if_save_test_trained_net_per_epoch',
                        action='store_true', default=False) # if we want to save adapted net during adaptation during meta-test
    parser.add_argument('--if_use_handmade_epsilon_hessian', dest='if_use_handmade_epsilon_hessian',
                        action='store_true',
                        default=False)  # default: automatically calculated epsilon #Andrei, Neculai. "Accelerated conjugate gradient algorithm with finite difference Hessian/vector product approximation for unconstrained optimization." Journal of Computational and Applied Mathematics 230.2 (2009): 570-582.
    parser.add_argument('--epsilon_hessian', dest='epsilon_hessian',
                        default=0.1, type=float)  # for Hessian-vector approx. & using handmade epsilon hessian (only works when argsif_use_handmade_epsilon_hessian = True)
    parser.add_argument('--if_hess_vec_approx_for_one_inner_loop', dest='if_hess_vec_approx_for_one_inner_loop',
                        action='store_true',
                        default=False)  # do not unify pilots in meta-training set but to use in a divided manner
    parser.add_argument('--if_cavia', dest='if_cavia',
                        action='store_true', default=False) # if we want to run cavia (should be used with jac_calc = 1001)
    parser.add_argument('--num_context_para', dest='num_context_para',
                        default=10, type=int) # number of context parameters for cavia

    parser.add_argument('--if_test_train_fix_seq_than_adapt_randomly',
                        dest='if_test_train_fix_seq_than_adapt_randomly',
                        action='store_true', default=False)

    parser.add_argument('--if_test_train_fix_seq_than_adapt_lr', dest='if_test_train_fix_seq_than_adapt_lr',
                        default=100,
                        type=int)

    parser.add_argument('--if_see_cos_similarity',
                        dest='if_see_cos_similarity',
                        action='store_true', default=False)



    ########## for online ################
    parser.add_argument('--if_not_unify_K_tr_K_te', dest='if_unify_K_tr_K_te', action='store_false',
                        default=True)  # do not divide meta-training set into two part
    parser.add_argument('--total_online_time', dest='total_online_time', default=1000,
                        type=int) # total number of slots for online scenario
    parser.add_argument('--max_num_pilots_for_train', dest='max_num_pilots_for_train', default=32,
                        type=int) # maximum number of pilots (P in paper)
    parser.add_argument('--performance_threshold_accur', dest='performance_threshold_accur', default=0.05,
                        type=float) # prescribed value used for adaptive pilots number selection scheme (reliability check)
    parser.add_argument('--adding_pilots_num_per_iter', dest='adding_pilots_num_per_iter', default=4,
                        type=int) # increased number of pilots during reliability check
    parser.add_argument('--num_epochs_ftl', dest='num_epochs_ftl', default=10000, type=int) # number of epochs for training for joint training (ftl stands for joint training)

    parser.add_argument('--lr_testtraining_meta', dest='lr_testtraining_meta', default=0.1, type=float) # learning rate for adaptation during meta-test for meta-trained network
    parser.add_argument('--lr_testtraining_tfs', dest='lr_testtraining_tfs', default=0.005, type=float) # learning rate for fixed initialization (tfs stands for fixed initialization)
    parser.add_argument('--lr_testtraining_ftl', dest='lr_testtraining_ftl', default=0.005, type=float) # learning rate for adaptation for joint trained network

    parser.add_argument('--num_epochs_test_meta', dest='num_epochs_test_meta', default=1000, type=int) # number of epochs for adaptation during meta-test for meta-trained network
    parser.add_argument('--num_epochs_test_tfs', dest='num_epochs_test_tfs', default=1000, type=int) # number of epochs for adaptation for fixed initialization
    parser.add_argument('--num_epochs_test_ftl', dest='num_epochs_test_ftl', default=1000, type=int) # nubmer of epcchos for adaptation for joint trainined network

    parser.add_argument('--lr_ftl', dest='lr_ftl', default=0.001, type=float) # learning rate during training for joint training (learning rate used for joint training)
    parser.add_argument('--mini_batch_size_ftl', dest='mini_batch_size_ftl', default=16, type=int) # mini batch size for joint training (minibatch size during joint training)
    parser.add_argument('--if_save_whole_max_pilots', dest='if_save_whole_max_pilots', action='store_true',
                        default=False)  # we are not actually using the whole pilots but following efficiency, but generate for comparison with several experiments
    parser.add_argument('--mode_for_selecting_online_schemes', dest='mode_for_selecting_online_schemes', default=0, type=int) # 0: FTML, 1: TFS, 3: FTL,
    parser.add_argument('--if_use_tensorboard', dest='if_use_tensorboard',
                        action='store_true',
                        default=False) # whether use tensorboard to visualize during online learning
    parser.add_argument('--if_no_adaptive_pilot_number_scheme_used', dest='if_no_adaptive_pilot_number_scheme_used',
                        action='store_true',
                        default=False) # whether to use adaptive pilot number selection scheme (default: using adaptive scheme)
    parser.add_argument('--if_not_adap_criteria_use_uncertainty_loss', dest='if_adap_criteria_use_uncertainty_loss',
                        action='store_false',
                        default=True) # whether to check reliability without decoder for adaptive pilot number selection scheme (default: without decoder)
    parser.add_argument('--if_tfs_start_from_middle', dest='if_tfs_start_from_middle',
                        action='store_true',
                        default=False) # for fixed initialization starting from middle of online scenario (since fixed initialization does not anything online)
    parser.add_argument('--tfs_start_ind_T', dest='tfs_start_ind_T', default=0, type=int) # for fixed initialization starting from middle of online schenrio, starting time index
    parser.add_argument('--K_TR_max_ref_tfs', dest='K_TR_max_ref_tfs', default=32, type=int) # for reference curve for fixed initialization, can be chosen from 0,1, ..., args.max_num_pilots_for_train

    ## offline
    parser.add_argument('--if_realistic_setting',
                        dest='if_realistic_setting',
                        action='store_true', default=False)

    parser.add_argument('--if_awgn',
                        dest='if_awgn',
                        action='store_true', default=False)
    parser.add_argument('--if_perfect_iq_imbalance_knowledge',
                        dest='if_perfect_iq_imbalance_knowledge',
                        action='store_true', default=False)

    parser.add_argument('--if_perfect_csi',
                        dest='if_perfect_csi',
                        action='store_true', default=False)
    parser.add_argument('--if_conven_commun',
                        dest='if_conven_commun',
                        action='store_true', default=False) # use with bm1 with M=16
    parser.add_argument('--if_test_training_adam',
                        dest='if_test_training_adam',
                        action='store_true', default=False)

    parser.add_argument('--if_adam_after_sgd',
                        dest='if_adam_after_sgd',
                        action='store_true', default=False)

    parser.add_argument('--if_joint_or_tfs',
                        dest='if_joint_or_tfs',
                        action='store_true', default=False) # simple adapt

    parser.add_argument('--toy_get_performance_during_meta_training',
                        dest='toy_get_performance_during_meta_training',
                        action='store_true', default=False)

    parser.add_argument('--if_use_stopping_criteria_during_test_training',
                        dest='if_use_stopping_criteria_during_test_training',
                        action='store_false', default=None)


    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = parse_args()

    if args.if_realistic_setting:
        args.if_test_train_fix_seq_than_adapt_randomly = True
        if args.if_use_stopping_criteria_during_test_training == None:
            args.if_use_stopping_criteria_during_test_training = True
        else:
            pass

        args.if_fix_te_num_during_meta_training = False
        args.if_no_distortion = False
        args.if_use_all_dev_in_one_epoch = False
        args.meta_train_with_adam = True
        if args.if_test_train_no_permute == None:
            args.if_test_train_no_permute = True
        else:
            pass
        args.SNR_db = 20
        args.lr_alpha_settings = 0.1
        args.lr_beta_settings = 0.001
        args.lr_benchmark2 = 0.001
        args.power = 1
        args.modulation_order = 16
        args.K_TEST_TE = 10000
        args.if_use_stopping_criteria_during_meta_training = False
        args.num_hidden_layer = 3
        args.num_neurons_first = 10
        args.num_neurons_second = 30
        args.num_neurons_third = 30
        args.num_context_para = 10
        args.if_fix_random_seed = True
        args.if_iq_imbalance = True
        args.meta_training_query_mode = 2

        if args.mode_for_selecting_online_schemes == 0: # 0: FTML, 1: TFS, 3: FTL,
            args.if_test_train_fix_seq_than_adapt_lr = 20
            args.mini_batch_size_meta = 4
            args.mini_batch_size_test_train = 16
        elif args.mode_for_selecting_online_schemes == 1: # this as conventional training... do not have to be fair...
            args.if_test_train_fix_seq_than_adapt_lr = 1
            args.mini_batch_size_test_train = 16
            args.mini_batch_size_meta = 16
            args.num_epochs_test_tfs = 100000
            args.lr_testtraining_tfs = 0.001
        elif args.mode_for_selecting_online_schemes == 3:
            args.if_test_train_fix_seq_than_adapt_lr = 1
            args.mini_batch_size_test_train = 16
            args.mini_batch_size_meta = 16
            args.lr_testtraining_ftl = 0.005
        else:
            raise NotImplementedError
        args.if_use_tensorboard = False

    else:
        raise NotImplementedError

    print('Called with args:')
    print(args)
    assert args.if_adap_criteria_use_uncertainty_loss == True
    if_cali = args.if_no_distortion
    if_symm = True # only active when args.if_no_distortion = True, binary channel
    if_bias = True # whether use bias for neural network
    if_reuse_testtrain_pilot_symbols = True # using fixed sequence & forcing successive M pilots compose constellation S
    if_reuse_metatrain_pilot_symbols = True # using fixed sequence & forcing successive M pilots compose constellation S
    meta_train_version = args.version_of_channel_train
    test_train_version = args.version_of_channel_test

    if_init_param_1 = args.if_param_set_1
    if_init_param_bias_0 = args.if_bias_set_0

    if args.mode_pilot_meta_training == 0:
        if_use_same_seq_both_meta_train_test = True
    elif args.mode_pilot_meta_training == 1:
        if_use_same_seq_both_meta_train_test = False
    else:
        if_use_same_seq_both_meta_train_test = None
    mini_batch_size_meta = args.mini_batch_size_meta
    mini_batch_size = args.mini_batch_size_test_train
    mini_batch_size_bm2 = args.mini_batch_size_bm2
    SNR_db = args.SNR_db
    lr_beta_settings = args.lr_beta_settings
    lr_alpha_settings = args.lr_alpha_settings
    K_TR = 0 # for online, no need
    K_TE = 0 # for online, no need
    K_TEST_TE = args.K_TEST_TE
    path_for_common_dir = args.path_for_common_dir
    path_for_meta_training_set = args.path_for_meta_training_set
    path_for_meta_test_set = args.path_for_meta_test_set
    M = args.modulation_order
    if_relu = args.if_relu

    jac_calc = args.jac_calc
    reptile_inner_loop = args.reptile_inner_loop
    if_cycle = args.if_cycle

    curr_time = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    common_dir = './online/' + path_for_common_dir + curr_time + '/'

    net_dir_for_test_result = common_dir + 'test_result/'

    net_dir_over_round = common_dir + 'w.r.t_dev (round)'
    writer_over_round = SummaryWriter(net_dir_over_round)

    net_dir_over_time= common_dir + 'w.r.t_time'
    writer_over_time = SummaryWriter(net_dir_over_time)

    net_dir_over_num_pilots_theoretic = common_dir + 'w.r.t_pilots_theoretic'
    writer_over_num_pilots_theoretic = SummaryWriter(net_dir_over_num_pilots_theoretic)

    if args.if_use_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    if args.if_fix_random_seed:
        torch.manual_seed(args.random_seed)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.random_seed)
    #################### initial setting

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
        net = deeper_linear_net(args=args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first,
                                num_neurons_second=args.num_neurons_second, num_neurons_third=args.num_neurons_third,
                                if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net.cuda()
        net_prime = deeper_linear_net_prime(args=args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first,
                                            num_neurons_second=args.num_neurons_second,
                                            num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net_prime.cuda()
        net_for_testtraining = deeper_linear_net(args=args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first,
                                                 num_neurons_second=args.num_neurons_second,
                                                 num_neurons_third=args.num_neurons_third, if_bias=if_bias,
                                                 if_relu=if_relu)  # dummy net for loading learned network
        if args.if_use_cuda:
            net_for_testtraining.cuda()

        net_tfs = deeper_linear_net(args=args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first,
                                    num_neurons_second=args.num_neurons_second,
                                    num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net_tfs.cuda()

        net_ftl = deeper_linear_net(args=args, m_ary=M_tmp, num_neurons_first=args.num_neurons_first,
                                    num_neurons_second=args.num_neurons_second,
                                    num_neurons_third=args.num_neurons_third, if_bias=if_bias, if_relu=if_relu)
        if args.if_use_cuda:
            net_ftl.cuda()

    else:
        raise NotImplementedError


    #########writers per scheme############
    if args.mode_for_selecting_online_schemes == 1:
        net_dir_over_num_pilots_tfs = common_dir + net_name +'w.r.t_pilots_tfs'
        writer_over_num_pilots_tfs = SummaryWriter(net_dir_over_num_pilots_tfs)

    if args.mode_for_selecting_online_schemes == 3:
        net_dir_over_num_pilots_ftl = common_dir + net_name + 'w.r.t_pilots_ftl'
        writer_over_num_pilots_ftl = SummaryWriter(net_dir_over_num_pilots_ftl)

    if args.mode_for_selecting_online_schemes == 0:
        net_dir_over_num_pilots_meta = common_dir + net_name + 'w.r.t_pilots_meta'
        writer_over_num_pilots_meta = SummaryWriter(net_dir_over_num_pilots_meta)

    #################### def. of network
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

    #################### weight first initialization (need to be checked for 'train from scratch'

    init_model_PATH = common_dir + 'init_model/' + net_name  # random weight used for benchmark 1
    if not os.path.exists(common_dir + 'init_model/'):
        os.mkdir(common_dir + 'init_model/')
    torch.save(net.state_dict(), init_model_PATH)  # always start from this init. realization
    #################### save initial weight

    #################### basic settings for IoT devs.
    channel_variance = 0.5
    num_dev = args.total_online_time
    K_TR_max = args.max_num_pilots_for_train
    K_TE_max = 0

    total_data_set = torch.zeros(num_dev, K_TR_max + K_TE_max, 4) # empty which will be filled as time goes on
    total_data_set = total_data_set.to(device)

    efficiency = numpy.zeros(num_dev, dtype=int)  # empty which will be filled as time goes on
    used_pilots_per_time = numpy.zeros(num_dev, dtype=int)  # empty which will be filled as time goes on

    # initialize net for the very first time
    net.load_state_dict(torch.load(init_model_PATH))
    net_prime.load_state_dict(torch.load(init_model_PATH))
    net_tfs.load_state_dict(torch.load(init_model_PATH))
    net_ftl.load_state_dict(torch.load(init_model_PATH))

    # test data set save for each ind_T
    test_training_set_over_meta_test_devs = []
    channel_set_genie_over_meta_test_devs = []
    channel_set_for_vis_over_meta_test_devs = []
    non_linearity_set_genie_over_meta_test_devs = []

    time_count = -1 # to start from 0
    count_success = 0
    total_transmitted_pilots = 0
    success_num_array_per_round = []
    tx_pilots_array_per_round = []
    min_error_rate_array_per_round = []
    actual_error_rate_array_per_round = []
    conven_error_rate_array_per_round = []
    actual_error_rate_array_per_round_at_first_reliability_success = []
    save_test_result_dict = {}

    ### follows beta distribution for iq imbalance
    basic_distribution_for_amplitude = torch.distributions.beta.Beta(5, 2)  # between 0 and 1
    basic_distribution_for_phase = torch.distributions.beta.Beta(5, 2)  # between 0 and 1

    for ind_T in range(args.total_online_time):
        if args.if_tfs_start_from_middle:
            ind_T += args.tfs_start_ind_T

        num_epochs = args.num_epochs_meta
        num_epochs_ftl = args.num_epochs_ftl
        prob_error_array_meta = []
        prob_error_array_ftl = []
        prob_error_array_tfs = []
        uncertainty_loss_array_meta = []
        uncertainty_loss_array_ftl = []
        uncertainty_loss_array_tfs = []
        iter_for_finetuing = 0
        # current device setting
        var_array = numpy.ones(1, dtype=numpy.float64)
        noise_variance = pow(10, numpy.log10(5) - SNR_db / 10)
        if M == 5:
            power = args.power
            noise_variance = pow(10, numpy.log10(5 * pow(power, 2)) - SNR_db / 10)
        elif M == 4:
            power = args.power
            noise_variance_real_and_im = 2 * pow(power, 2) / (pow(10, SNR_db / 10))
            noise_variance = noise_variance_real_and_im / 2
        elif M == 16:
            power = args.power
            noise_variance_real_and_im = (pow(power, 2) * (10)) / (pow(10, SNR_db / 10))
            noise_variance = noise_variance_real_and_im / 2

        var = noise_variance  # variance of noise
        var_array = channel_variance * var_array  # variance of channel

        ######## iq imbalance

        mean_array0 = []
        mean_array1 = []
        for ind_dev_meta_train in range(1): # for curr dev.
            ampl_beta_rv = basic_distribution_for_amplitude.sample()
            phase_beta_rv = basic_distribution_for_phase.sample()
            ampl_distortion_curr_dev = ampl_beta_rv * 0.15  # we need to multiply with max value: 0.15
            phase_distortion_curr_dev = phase_beta_rv * (numpy.pi / 180) * 15  # 15 degree
            mean_array0.append(ampl_distortion_curr_dev)
            mean_array1.append(phase_distortion_curr_dev)
        mean_array3 = []  # deprecated
        mean_array5 = []  # deprecated

        name_of_the_net_for_net_dir = 'total_time:' + str(args.total_online_time) + 'M_order:' + str(M)  + 'model_type:' + net_name + 'noise_variance:' + str(
            noise_variance) + 'channel_variance:' + str(channel_variance)
        net_dir_meta = common_dir + 'TB/' + 'meta_training_set/' + name_of_the_net_for_net_dir

        writer_per_dev_tot = []
        K_TR_accum = 0 # used also as idx for adding new pilots to total_data_set
        if ind_T == 0:
            K_TR_max_curr = K_TR_max
        else:
            if args.if_no_adaptive_pilot_number_scheme_used:
                K_TR_max_curr = K_TR_max
            else: # with adaptive pilot number selection scheme
                if efficiency[ind_T-1] == -1:
                    K_TR_max_curr = K_TR_max
                else:
                    K_TR_max_curr = efficiency[ind_T-1] # use prev. num_pilots effic. as curr. required num_pilots
        if args.if_tfs_start_from_middle: # means making ref. curve
            K_TR_max_curr = args.K_TR_max_ref_tfs
            assert args.adding_pilots_num_per_iter == args.K_TR_max_ref_tfs
        used_pilots_per_time[ind_T] = K_TR_max_curr
        name_of_current_used_pilots = common_dir + 'used_pilots_per_time/' + 'num_dev:' + str(
            num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/' + name_of_the_net_for_net_dir + '.pckl'

        if not os.path.exists(common_dir + 'used_pilots_per_time/' + 'num_dev:' + str(
            num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/'):
            os.makedirs(common_dir + 'used_pilots_per_time/' + 'num_dev:' + str(
            num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/')
        f_for_used_pilots_per_time = open(name_of_current_used_pilots, 'wb')
        pickle.dump(used_pilots_per_time, f_for_used_pilots_per_time)
        f_for_used_pilots_per_time.close()
        efficiency_curr_dev = []
        curr_dev_char = None # first None
        ##### generate train and test set for once (practical commun.)
        ##### mode for comparison (generate all train set) and mode for realistic (generate only desired # pilots)

        if path_for_meta_training_set is not None:
            # load sequentially (enabling almost simultaneous comparison)
            path_common_dir_existing = path_for_meta_training_set # load common dir where desired meta-training set is in

            path_for_current_train_set_existing = path_common_dir_existing + '/' + 'meta_training_set/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/' + name_of_the_net_for_net_dir + '.pckl'
            f_current_train_set_existing = open(path_for_current_train_set_existing, 'rb')
            total_data_set = pickle.load(f_current_train_set_existing)
            f_current_train_set_existing.close()

        else:
            # generate meta-training set only when there is no desired existing meta-training set
            if args.if_save_whole_max_pilots:
                K_TR_curr = K_TR_max # generate whole pilots (even though we do not transmit.. just for comparison)
                K_TE_curr = 0  # we are now generating only one data set (not like offline case which we had generated two sets)
                total_data_set, curr_dev_char = generating_online_training_set(args, curr_dev_char, total_data_set,
                                                                               K_TR_accum, K_TR_curr, K_TE_curr, ind_T, M,
                                                                               var_array,
                                                                               var, mean_array0, mean_array1, mean_array3,
                                                                               mean_array5,
                                                                               writer_per_dev_tot, if_cali, if_symm,
                                                                               meta_train_version,
                                                                               if_reuse_metatrain_pilot_symbols,
                                                                               power, device)

            else:
                K_TR_curr = used_pilots_per_time[ind_T] # generate whole pilots once (send all desired pilots at once) #args.adding_pilots_num_per_iter
                K_TE_curr = 0 # we are now generating only one data set (not like offline case which we had generated two sets)
                total_data_set, curr_dev_char = generating_online_training_set(args, curr_dev_char, total_data_set,
                                                                               K_TR_accum, K_TR_curr, K_TE_curr, ind_T, M,
                                                                               var_array,
                                                                               var, mean_array0, mean_array1, mean_array3,
                                                                               mean_array5,
                                                                               writer_per_dev_tot, if_cali, if_symm,
                                                                               meta_train_version,
                                                                               if_reuse_metatrain_pilot_symbols,
                                                                               power, device)

        ################## save total_data_set for current time, even if we have loaded from existing, just save for convenience

        name_of_current_train_set = common_dir + 'meta_training_set/' + 'num_dev:' + str(
            num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/' + name_of_the_net_for_net_dir + '.pckl'

        if os.path.exists(name_of_current_train_set):
            print('you are trying to overwrite existing meta-training set!!')
            raise NotImplementedError


        if not os.path.exists(common_dir + 'meta_training_set/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/'):
            os.makedirs(common_dir + 'meta_training_set/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/')
        f_for_train_set = open(name_of_current_train_set, 'wb')
        pickle.dump(total_data_set, f_for_train_set)
        f_for_train_set.close()

        ############ generate real-test set # payload data

        if path_for_meta_test_set is not None: # load from existing
            path_common_dir_existing_test = path_for_meta_test_set

            path_for_current_test_set_existing = path_common_dir_existing_test + '/' + 'real_test_set/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/' + name_of_the_net_for_net_dir + '.pckl'
            f_current_test_set_existing = open(path_for_current_test_set_existing, 'rb')
            test_training_whole_list = pickle.load(f_current_test_set_existing)
            f_current_test_set_existing.close()

            test_training_set = test_training_whole_list[0]
            channel_set_genie = test_training_whole_list[1]
            channel_set_for_vis = test_training_whole_list[2]
            non_linearity_set_genie = test_training_whole_list[3]

            curr_dev_csi = None
            curr_dev_nonlinearity = None
            if args.if_perfect_csi:
                curr_dev_csi = channel_set_for_vis
            if args.if_perfect_iq_imbalance_knowledge:
                curr_dev_nonlinearity = non_linearity_set_genie

            genie_set = [curr_dev_csi, curr_dev_nonlinearity]


        else:
            #### generate real-test set for once ###
            K_TEST_TR = 0  # we do not need test_train set in online
            test_training_set, channel_set_genie, channel_set_for_vis, non_linearity_set_genie = generating_test_set(
                curr_dev_char, K_TEST_TR, K_TEST_TE, 1,
                M, var_array, var,
                mean_array0,
                mean_array1,
                mean_array3,
                mean_array5,
                ind_T,
                if_cali, if_symm,
                test_train_version,
                if_reuse_testtrain_pilot_symbols,
                power, device, args)
            test_training_set_over_meta_test_devs.append(test_training_set)
            channel_set_genie_over_meta_test_devs.append(channel_set_genie)
            channel_set_for_vis_over_meta_test_devs.append(channel_set_for_vis)
            non_linearity_set_genie_over_meta_test_devs.append(non_linearity_set_genie)

            curr_dev_csi = None
            curr_dev_nonlinearity = None
            if args.if_perfect_csi:
                curr_dev_csi = channel_set_for_vis
            if args.if_perfect_iq_imbalance_knowledge:
                curr_dev_nonlinearity = non_linearity_set_genie

            genie_set = [curr_dev_csi, curr_dev_nonlinearity]

            ########################################
            test_training_whole_list = [test_training_set, channel_set_genie, channel_set_for_vis,
                                        non_linearity_set_genie]

        ################## save all for the testing for current time, even if we have loaded from existing, just save for convenience

        name_of_current_test_set = common_dir + 'real_test_set/' + 'num_dev:' + str(
            num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/' + name_of_the_net_for_net_dir + '.pckl'

        if os.path.exists(name_of_current_test_set):
            print('you are trying to overwrite existing test set!!!')
            raise NotImplementedError


        if not os.path.exists(common_dir + 'real_test_set/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/'):
            os.makedirs(common_dir + 'real_test_set/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/')
        f_for_test_set = open(name_of_current_test_set, 'wb')
        pickle.dump(test_training_whole_list, f_for_test_set)
        f_for_test_set.close()

        K_TR_max_tmp = K_TR_max
        K_TE_max_tmp = 0
        train_set_tmp = torch.zeros(K_TR_max_tmp + K_TE_max_tmp, 4)
        train_set_tmp = train_set_tmp.to(device)
        pilot_total_set_num = K_TR_max_tmp // 16 # rej. samping per 16 pilots, o.w. same pilot repeated
        ####### start training -> fine-tuning -> performance check repetition #########

        error_rate_for_efficiency_list_for_min_error_rate = []
        count_for_actual_error_rate_array_per_round_at_first_reliability_success = 0
        while K_TR_accum < K_TR_max_curr: # if we add more than 1, it could be larger than max...
            time_count += 1
            print('current acutal time: ', time_count)
            print('curr char: ', curr_dev_char)
            K_TR_accum = K_TR_accum + args.adding_pilots_num_per_iter # # of pilots for meta-train and fine-tuning

            print('start meta training with %d training device' % ind_T)
            print('start meta training for %d epochs' % num_epochs)
            print('start meta training with %d pilots (meta training)' % K_TR_accum)
            print('start meta training with %d pilots (in fact using this for all meta test)' % used_pilots_per_time[ind_T])
            # meta_train(num_epochs)
            name_of_the_net_for_meta = 'meta' + name_of_the_net_for_net_dir

            saved_model_PATH_meta_intermediate = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'meta/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_T:' + str(ind_T) + '/' + 'total_pilots_accum:' + str(K_TR_accum) + '/' + name_of_the_net_for_meta + '/'

            if not os.path.exists(saved_model_PATH_meta_intermediate):
                os.makedirs(saved_model_PATH_meta_intermediate)
            writer_per_num_dev = []


            if args.mini_batch_size_meta_device > ind_T + 1: # ind_T + 1: total num_dev
                sampled_device_num = ind_T + 1 #1
            else:
                sampled_device_num = args.mini_batch_size_meta_device



            ############ meta-online #############
            if iter_for_finetuing == 0:
                if args.mode_for_selecting_online_schemes == 0:
                    if ind_T == 0:
                        pass
                    else:
                        meta_train_online(M, num_epochs, num_dev, net, net_prime, total_data_set, used_pilots_per_time, ind_T, K_TR_accum, K_TR, K_TE, device,
                                   writer_per_num_dev,
                                   writer_per_dev_tot, saved_model_PATH_meta_intermediate, lr_alpha_settings,
                                   lr_beta_settings,
                                   mini_batch_size_meta, if_use_same_seq_both_meta_train_test, sampled_device_num, jac_calc,
                                   reptile_inner_loop, if_cycle, args)
                    saved_model_PATH_meta = saved_model_PATH_meta_intermediate + name_of_the_net_for_meta
                    torch.save(net.state_dict(), saved_model_PATH_meta)
                    print('current pilots: %d' % K_TR_accum)
                    print('end meta training for with %d training device' % (ind_T + 1))

            ############# train from scratch ####################
            if args.mode_for_selecting_online_schemes == 1:
                net_tfs.load_state_dict(torch.load(init_model_PATH))
                saved_model_PATH_tfs_intermediate = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'tfs/' + 'num_dev:' + str(
                    num_dev) + '/' + 'ind_T:' + str(ind_T) + '/' + 'total_pilots_accum:' + str(
                    K_TR_accum) + '/' + name_of_the_net_for_meta + '/'
                if not os.path.exists(saved_model_PATH_tfs_intermediate):
                    os.makedirs(saved_model_PATH_tfs_intermediate)
                name_of_the_net_for_tfs = 'tfs ' + name_of_the_net_for_net_dir
                saved_model_PATH_tfs = saved_model_PATH_tfs_intermediate + name_of_the_net_for_tfs
                torch.save(net_tfs.state_dict(), saved_model_PATH_tfs)
                print('current pilots: %d' % K_TR_accum)
                print('end tfs for with %d training device' % (ind_T + 1))


            ############## follow the leader (FTL) #############
            if iter_for_finetuing == 0:
                if args.mode_for_selecting_online_schemes == 3:
                    name_of_the_net_for_ftl = 'ftl ' + name_of_the_net_for_net_dir
                    writer_ftl = []
                    online_mode = 1
                    saved_model_PATH_ftl_intermediate = common_dir + 'saved_model/' + 'before_meta_testing_set/' + 'ftl/' + 'num_dev:' + str(
                        num_dev) + '/' + 'ind_T:' + str(ind_T) + '/' + 'total_pilots_accum:' + str(
                        K_TR_accum) + '/' + name_of_the_net_for_meta + '/'
                    if not os.path.exists(saved_model_PATH_ftl_intermediate):
                        os.makedirs(saved_model_PATH_ftl_intermediate)
                    saved_model_PATH_ftl = saved_model_PATH_ftl_intermediate + name_of_the_net_for_ftl
                    if ind_T == 0:
                        net_ftl.load_state_dict(torch.load(init_model_PATH))
                        torch.save(net_ftl.state_dict(), saved_model_PATH_ftl)
                    else:
                        for time_index in range(ind_T):
                            if time_index < ind_T:
                                curr_pilots_for_joint = used_pilots_per_time[time_index]
                                curr_data_for_joint = total_data_set[time_index, :curr_pilots_for_joint, :]
                                if time_index == 0:
                                    train_set_ftl = curr_data_for_joint
                                else:
                                    train_set_ftl = torch.cat((train_set_ftl, curr_data_for_joint), 0)
                            else:  # time_index == ind_T
                                print('something wrong')
                                raise NotImplementedError

                        test_training_benchmark2(args, M, args.lr_ftl, args.mini_batch_size_ftl, net_ftl,
                                                 train_set_ftl, K_TR, K_TE, num_epochs_ftl,
                                                 writer_ftl, online_mode, device, saved_model_PATH_ftl_intermediate, None, None)

                        torch.save(net_ftl.state_dict(), saved_model_PATH_ftl)

                        print('current pilots: %d' % curr_pilots_for_joint)
                        print('end FTL for with %d training device' % (ind_T + 1))


            num_test_iter = 1
            # currently no iteration for a one device

            K_TEST_TR = K_TR_accum
            tmp_test_set = test_training_set[0, :, :]
            tmp_training_set = total_data_set[ind_T, :K_TR_accum, :]

            test_training_set_curr = torch.zeros(1, K_TEST_TR + K_TEST_TE, 4)  # 3 = 2+1, s and y
            test_training_set_curr = test_training_set_curr.to(device)
            test_training_set_curr[0, :K_TEST_TR, :] = tmp_training_set
            test_training_set_curr[0, K_TEST_TR:, :] = tmp_test_set

            dir_tfs = common_dir + 'saved_model/' + 'after_meta_testing_set/' + 'tfs/' + 'num pilots:' + str(
                K_TEST_TR) + '/' + 'num_dev:' + str(
                ind_T + 1) + '/'
            dir_ftl = common_dir + 'saved_model/' + 'after_meta_testing_set/' + 'ftl/' + 'num pilots:' + str(
                K_TEST_TR) + '/' + 'num_dev:' + str(
                ind_T + 1) + '/'
            dir_meta = common_dir + 'saved_model/' + 'after_meta_testing_set/' + 'meta/' + 'num pilots:' + str(
                K_TEST_TR) + '/' + 'num_dev:' + str(
                ind_T + 1) + '/'
            if not os.path.exists(dir_tfs):
                os.makedirs(dir_tfs)
            if not os.path.exists(dir_ftl):
                os.makedirs(dir_ftl)
            if not os.path.exists(dir_meta):
                os.makedirs(dir_meta)

            ############## meta ############

            if args.mode_for_selecting_online_schemes == 0:
                save_PATH_meta = dir_meta + name_of_the_net_for_meta
                print('start finetuning & testing for meta with %d pilots' %K_TR_accum)
                uncertainty_loss_tmp_meta, final_loss_tmp_meta, total_error_num_tmp_meta, total_pilot_num_tmp_meta, theoretical_bound_meta = test_test_mul_dev(
                    args, net_for_testtraining,
                    num_test_iter, mini_batch_size, M, args.lr_testtraining_meta,
                    K_TEST_TR,
                    K_TEST_TE, args.num_epochs_test_meta,
                    saved_model_PATH_meta,
                    save_PATH_meta,
                    test_training_set_curr, False, noise_variance,
                    power, device, genie_set)
                error_rate_meta = total_error_num_tmp_meta/total_pilot_num_tmp_meta
                prob_error_array_meta.append(error_rate_meta)
                uncertainty_loss_array_meta.append(uncertainty_loss_tmp_meta)

                if args.if_use_tensorboard:
                    writer_over_num_pilots_meta.add_scalar('test_error_rate w.r.t. time', error_rate_meta,
                                                          time_count)
                    writer_over_num_pilots_theoretic.add_scalar('test_error_rate w.r.t. time', theoretical_bound_meta,
                                                           time_count)


            ############### TFS #############

            if args.mode_for_selecting_online_schemes == 1:
                save_PATH_tfs = dir_tfs + name_of_the_net_for_tfs
                print('start finetuning & testing for TFS with %d pilots' %K_TR_accum)
                uncertainty_loss_tmp_tfs, final_loss_tmp_tfs, total_error_num_tmp_tfs, total_pilot_num_tmp_tfs, theoretical_bound_tfs = test_test_mul_dev(
                    args, net_for_testtraining,
                    num_test_iter, mini_batch_size, M, args.lr_testtraining_tfs,
                    K_TEST_TR,
                    K_TEST_TE, args.num_epochs_test_tfs,
                    saved_model_PATH_tfs,
                    save_PATH_tfs,
                    test_training_set_curr, args.if_conven_commun, noise_variance,
                    power, device, genie_set)

                error_rate_tfs = total_error_num_tmp_tfs / total_pilot_num_tmp_tfs
                prob_error_array_tfs.append(error_rate_tfs)
                uncertainty_loss_array_tfs.append(uncertainty_loss_tmp_tfs)
                if args.if_use_tensorboard:
                    writer_over_num_pilots_tfs.add_scalar('test_error_rate w.r.t. time', error_rate_tfs,
                                                           time_count)
                    writer_over_num_pilots_theoretic.add_scalar('test_error_rate w.r.t. time', theoretical_bound_tfs,
                                                           time_count)
                    writer_over_num_pilots_theoretic.add_scalar('channel_abs', channel_set_genie,
                                                                time_count)

            ############### FTL #############

            if args.mode_for_selecting_online_schemes == 3:
                save_PATH_ftl = dir_ftl + name_of_the_net_for_ftl
                print('start finetuning & testing for FTL with %d pilots' %K_TR_accum)
                uncertainty_loss_tmp_ftl, final_loss_tmp_ftl, total_error_num_tmp_ftl, total_pilot_num_tmp_ftl, theoretical_bound_ftl = test_test_mul_dev(
                    args, net_for_testtraining,
                    num_test_iter, mini_batch_size, M, args.lr_testtraining_ftl,
                    K_TEST_TR,
                    K_TEST_TE, args.num_epochs_test_ftl,
                    saved_model_PATH_ftl,
                    save_PATH_ftl,
                    test_training_set_curr, False, noise_variance,
                    power, device, genie_set)
                error_rate_ftl = total_error_num_tmp_ftl / total_pilot_num_tmp_ftl
                prob_error_array_ftl.append(error_rate_ftl)
                uncertainty_loss_array_ftl.append(uncertainty_loss_tmp_ftl)
                if args.if_use_tensorboard:
                    writer_over_num_pilots_ftl.add_scalar('test_error_rate w.r.t. time', error_rate_ftl,
                                                          time_count)
                    writer_over_num_pilots_theoretic.add_scalar('test_error_rate w.r.t. time', theoretical_bound_ftl,
                                                                time_count)

            ############### get efficiency #################
            if args.mode_for_selecting_online_schemes == 0:
                uncertainty_loss_for_efficiency = uncertainty_loss_tmp_meta
                error_rate_for_efficiency =error_rate_meta

            elif args.mode_for_selecting_online_schemes == 1:
                uncertainty_loss_for_efficiency = uncertainty_loss_tmp_tfs
                error_rate_for_efficiency = error_rate_tfs

            elif args.mode_for_selecting_online_schemes == 3:
                uncertainty_loss_for_efficiency = uncertainty_loss_tmp_ftl
                error_rate_for_efficiency = error_rate_ftl
            else:
                raise NotImplementedError
            #############
            if args.if_adap_criteria_use_uncertainty_loss:
                criteria_value_for_adaptive = uncertainty_loss_for_efficiency
            else:
                criteria_value_for_adaptive = error_rate_for_efficiency
            ### get array of error_rate
            error_rate_for_efficiency_list_for_min_error_rate.append(error_rate_for_efficiency)

            if criteria_value_for_adaptive <= args.performance_threshold_accur:
                efficiency_curr_dev.append(K_TR_accum)
                efficiency_curr = K_TR_accum
                if count_for_actual_error_rate_array_per_round_at_first_reliability_success == 0: # get error rate at first success (rel.)
                    actual_error_rate_array_per_round_at_first_reliability_success.append(error_rate_for_efficiency)
                    count_for_actual_error_rate_array_per_round_at_first_reliability_success += 1
                    min_effic_tmp_cross_check = K_TR_accum
            else:
                efficiency_curr_dev.append(9999)  # not fulfilled
                efficiency_curr = 9999

            if args.if_use_tensorboard:
                writer_over_time.add_scalar('dev index w.r.t. time', ind_T,
                                                       time_count)
                writer_over_time.add_scalar('efficiency w.r.t. time', efficiency_curr,
                                                       time_count)
                writer_over_time.add_scalar('used pilots w.r.t. time', used_pilots_per_time[ind_T],
                                            time_count)

            print('time count!!!!!!!', time_count)
            print('error rate now: ', error_rate_for_efficiency)
            iter_for_finetuing += 1
        ##################################################
        min_effic_tmp = numpy.amin(efficiency_curr_dev)

        if min_effic_tmp == 9999:
            efficiency[ind_T] = -1  # not fulfilled at once the threshold for all the pilots
            success_num_array_per_round.append(0)
        else:
            efficiency[ind_T] = min_effic_tmp
            success_num_array_per_round.append(1)
            count_success += 1

        total_transmitted_pilots += used_pilots_per_time[ind_T]
        avg_transmitted_pilots = total_transmitted_pilots / (ind_T + 1)
        tx_pilots_array_per_round.append(used_pilots_per_time[ind_T])

        min_error_rate_array_per_round.append(min(error_rate_for_efficiency_list_for_min_error_rate))
        actual_error_rate_array_per_round.append(error_rate_for_efficiency_list_for_min_error_rate[-1])

        if args.mode_for_selecting_online_schemes == 1:
            conven_error_rate_array_per_round.append(theoretical_bound_tfs)

        if count_for_actual_error_rate_array_per_round_at_first_reliability_success == 0:
            actual_error_rate_array_per_round_at_first_reliability_success.append(error_rate_for_efficiency)
            min_effic_tmp_cross_check = 9999

        assert min_effic_tmp_cross_check == min_effic_tmp # check breaking at rel. success & getting min value

        if args.if_use_tensorboard:
            writer_over_round.add_scalar('efficiency w.r.t. round', efficiency[ind_T],
                                                       ind_T)
            writer_over_round.add_scalar('success w.r.t. round', count_success,
                                         ind_T)
            writer_over_round.add_scalar('avg_transmitted_pilots w.r.t. round', avg_transmitted_pilots,
                                         ind_T)
            writer_over_time.add_scalar('used pilots w.r.t. round', used_pilots_per_time[ind_T],
                                        ind_T)

        save_test_result_dict['total_success_per_threshold'] = success_num_array_per_round
        save_test_result_dict['total_pilot'] = tx_pilots_array_per_round
        save_test_result_dict['minimum_total_error_rate_deprecated'] = min_error_rate_array_per_round
        save_test_result_dict['conven_error_rate'] = conven_error_rate_array_per_round # mmse or optimal # if mmse, fix num pilots as tfs ref curve...
        save_test_result_dict['actual_total_error_rate_with_total_pilot_always_totla_pilot'] = actual_error_rate_array_per_round
        save_test_result_dict['actual_total_error_rate_with_total_pilot_cut_at_rel_succ'] = actual_error_rate_array_per_round_at_first_reliability_success
        save_test_result_dict['error_rate_per_pilot_meta'] = prob_error_array_meta
        save_test_result_dict['error_rate_per_pilot_ftl'] = prob_error_array_ftl
        save_test_result_dict['error_rate_per_pilot_tfs'] = prob_error_array_tfs

        save_test_result_dict['uncertainty_loss_per_pilot_meta'] = uncertainty_loss_array_meta
        save_test_result_dict['uncertainty_loss_per_pilot_ftl'] = uncertainty_loss_array_ftl
        save_test_result_dict['uncertainty_loss_per_pilot_tfs'] = uncertainty_loss_array_tfs

        net_dir_for_test_result = common_dir + 'test_result/'
        accum_result_up_to_curr_pilot_path = net_dir_for_test_result + 'curr_time_idx' + str(
            ind_T) + '/' + 'test_result.mat'

        os.makedirs(net_dir_for_test_result + 'curr_time_idx' + str(
            ind_T) + '/')

        sio.savemat(accum_result_up_to_curr_pilot_path, save_test_result_dict)

        accum_result_up_to_curr_pilot_path_pickle = net_dir_for_test_result + 'curr_time_idx' + str(
            ind_T) + '/' + 'test_result.pckl'
        f_for_test_result = open(accum_result_up_to_curr_pilot_path_pickle, 'wb')
        pickle.dump(save_test_result_dict, f_for_test_result)
        f_for_test_result.close()


        ####### efficiency save #######
        name_of_current_efficiency = common_dir + 'efficiency/' + 'num_dev:' + str(
            num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/' + name_of_the_net_for_net_dir + '.pckl'

        if not os.path.exists(common_dir + 'efficiency/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/'):
            os.makedirs(common_dir + 'efficiency/' + 'num_dev:' + str(
                num_dev) + '/' + 'ind_dev:' + str(ind_T) + '/')
        f_for_efficiency = open(name_of_current_efficiency, 'wb')
        pickle.dump(efficiency, f_for_efficiency)
        f_for_efficiency.close()

    # save all test set
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

    name_of_current_testtraining_set = common_dir + 'meta_test_set/' + 'testtraining_set.pckl'
    f_for_test_train_set = open(name_of_current_testtraining_set, 'wb')
    pickle.dump(test_training_set_over_meta_test_devs, f_for_test_train_set)
    f_for_test_train_set.close()

    name_of_current_nonlinear_set = common_dir + 'meta_test_set/' + 'nonlinear_set.pckl'
    f_for_nonlinearset = open(name_of_current_nonlinear_set, 'wb')
    pickle.dump(non_linearity_set_genie_over_meta_test_devs, f_for_nonlinearset)
    f_for_nonlinearset.close()
