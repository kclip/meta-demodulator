python ../../../../../../../main_offline.py --K_TR 0 --K_TE 3200 --maml_inner_loop 2 --mini_batch_size_bm2 16 --mini_batch_size_test_train 16 --meta_test_pilot_num 8 --max_pilot_test 1600  --if_no_bm1 --if_no_bm2 --toy_get_performance_during_meta_training --toy_check_ser_period_during_meta_training 9990000 --jac_calc 200 --if_realistic_setting --num_dev 200  --if_fix_random_seed --path_for_common_dir 'realistic/iq_beta/first_exp_rev_final/joint/mb_16/rev/dev/200/' --path_for_meta_trained_net '/home/hdd1/logs/tsp_meta_demod_first_revision/realistic/iq_beta/first_exp_rev_final/joint/mb_16/rev/dev/200/time/saved_model/before_meta_testing_set/benchmark2/num_dev:200/benchmark2num_dev:200M_order:16model_type:deep_linear_net_with_3_hidden_layernoise_variance:0.05channel_variance:0.5/best_model_based_on_meta_training_loss'