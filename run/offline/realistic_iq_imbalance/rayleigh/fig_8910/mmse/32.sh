python ../../../../../../main_offline.py  --K_TR 0 --K_TE 16  --mini_batch_size_meta 3 --mini_batch_size_test_train 16  --meta_test_pilot_num 32 --max_pilot_test 1600 --mini_batch_size_bm2 3 --num_epochs_test_bm1 100000 --if_conven_commun --if_no_meta --if_no_bm2 --toy_get_performance_during_meta_training --toy_check_ser_period_during_meta_training 100 --jac_calc 200 --maml_inner_loop 2 --if_realistic_setting  --num_dev 1  --mini_batch_size_meta_device 5 --if_fix_random_seed --path_for_common_dir 'realistic/iq_beta/feasibility_check/tfs/pilots/32/'