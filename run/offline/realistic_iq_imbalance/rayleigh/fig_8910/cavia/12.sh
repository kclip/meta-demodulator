python ../../../../../../main_offline.py  --K_TR 0 --K_TE 160  --mini_batch_size_meta 12 --mini_batch_size_test_train 12  --meta_test_pilot_num 12 --max_pilot_test 1600 --mini_batch_size_bm2 12 --num_epochs_test_bm1 0 --if_no_bm1 --if_no_bm2 --toy_get_performance_during_meta_training --toy_check_ser_period_during_meta_training 1000   --jac_calc 1001 --if_cavia  --maml_inner_loop 2 --if_realistic_setting  --num_dev 1000  --mini_batch_size_meta_device 5 --if_fix_random_seed --path_for_common_dir 'realistic/iq_beta/first_exp_rev_final/cavia/inner_1/pilot_12/'