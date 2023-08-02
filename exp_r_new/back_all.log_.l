2023-07-31 19:31:05,313 - logger name:exp/ECL-PatchTST2023-07-31-19:31:05.313545/ECL-PatchTST.log
2023-07-31 19:31:05,314 - params : {'conf': 'ECL-PatchTST', 'data_name': 'ETTh2', 'iteration': 1, 'load': True, 'build_graph': False, 'same_init': True, 'grad_norm': False, 'refiner_no_residual': False, 'root_path': '', 'exp_path': 'exp/', 'val_test_mix': False, 'pred_len': 96, 'noise_rate': 0.5, 'device': device(type='cuda', index=0), 'test_model_path': '/Disk/fhyega/code/BASE/exp/ECL-PatchTST2023-07-23-21:59:42.618606/0/0.0379_epoch_25.pkl', 'idx': -1, 'aligner': 0, 'always_align': 1, 'refiner': 0, 'refiner_block_num': 1, 'enhance': 0, 'enhance_type': 0, 'seed': 42033, 'batch_size': 128, 'jitter_sigma': 0.1, '/* model related args*/': '//', 'model_name': 'PatchTST', 'seq_len': 336, 'label_len': 48, 'features': 'M', 'target': 'OT', 'graph_input': False, 'individual': False, 'e_layers': 3, 'd_layers': 1, 'factor': 1, 'n_heads': 16, 'd_model': 128, 'd_ff': 256, 'dropout': 0.2, 'fc_dropout': 0.2, 'head_dropout': 0.0, 'patch_len': 16, 'stride': 8, 'des': 'Exp', 'padding_patch': 'end', 'revin': 1, 'affine': 0, 'subtract_last': 0, 'decomposition': 0, 'kernel_size': 25, 'output_attention': 0, 'embed_type': 0, 'activation': 'gelu', 'lradj': 'TST', 'distil': 1, 'linear_output': 1, 'pct_start': 0.2, '/*train related args*/': '//', 'train': True, 'epoch': 30, 'lr': 0.0001, 'loss': 'mse', '/*dataset related args*/': '//', 'save_data_path': 'data/ECL/', 'data_process': True, 'embed': 'timeF', 'freq': 'h', 'begin_phase': 0, 'end_phase': 1, 'phase_len': 26304, 'val_ratio': 0.33, 'graph_size': 321, '/*strategy related args*/': '//', 'strategy': 'incremental', 'increase': False, 'detect': False, 'detect_strategy': 'feature', 'replay': False, 'replay_strategy': 'random', 'repaly_num_samples': 100, 'ewc': False, 'ewc_strategy': 'ewc', 'ewc_lambda': 0.0001, 'subgraph_train': False, 'num_hops': 2, 'logname': 'ECL-PatchTST', 'time': '2023-07-31-19:31:05.313545', 'path': 'exp/ECL-PatchTST2023-07-31-19:31:05.313545', 'num_workers': 4, 'start_train': 0, 'train_mode': 'pretrain', 'get_score': False, 'use_cm': True, 'logger': <Logger __main__ (INFO)>}
2023-07-31 19:31:05,314 - [*] phase 0 start training
0 26304
train 8209
val 2785
test 2785
2023-07-31 19:31:05,506 - [*] phase 0 Dataset load!
2023-07-31 19:31:06,340 - [*] phase 0 Training start
train 8209