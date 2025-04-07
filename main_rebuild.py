import time
# import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2, 3'
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
import torch.utils.data
from model_object import models
from Config import parse_args
from utils.observer import RuntimeObserver
from utils.api import *
from thop import profile, clever_format
#99480885
#
def prepare_to_train(mri_dir, pet_dir, cli_dir, csv_file, batch_size, model_index,
                     seed, device, data_parallel, n_splits, others_params):
    global experiment_settings
    assert torch.cuda.is_available(), "Please ensure codes are executed on cuda."
    try:
        experiment_settings = models[model_index]
    except KeyError:
        print('model not in model_object!')
    torch.cuda.empty_cache()

    # 初始化数据集
    # if model_index == 'IMF':
    #     dataset = MriPetDatasetWithTowLabel(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # elif model_index == 'MDL':
    #     dataset = MriPetDatasetWithTwoInput(mri_dir, pet_dir, csv_file, valid_group=("pMCI", "sMCI"))
    # else:
    #     dataset = MriPetDataset(mri_dir, pet_dir, cli_dir, csv_file, valid_group=("pMCI", "sMCI"))
    dataset = experiment_settings['dataset'](mri_dir, pet_dir, cli_dir, csv_file,
                                             resize_shape=experiment_settings['shape'],
                                             valid_group=("pMCI", "sMCI"))
    torch.manual_seed(seed)

    # K折交叉验证
    # kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    labels = [data[3] for data in dataset]  # 假设dataset[i]的第3项是label
    # 存储每个fold的评估指标
    metrics = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'balanceAccuracy':[],
        'Specificity': [],
        'auc': [],
        'f1': [],
    }
    # 训练日志和监控
    current_time = str(datetime.now().strftime('%Y-%m-%d_%H-%M'))
    target_dir = f'./{others_params.checkpoints_dir}_{current_time}/'
    Path(target_dir).mkdir(exist_ok=True)
    # for fold, (train_index, test_index) in enumerate(kf.split(dataset)):
    for fold, (train_index, test_index) in enumerate(skf.split(dataset, labels), 1):
        observer = RuntimeObserver(log_dir=target_dir, device=device, name=experiment_settings['Name'], seed=seed)
        observer.log(f'Fold {fold}/{5}')
        train_sampler = torch.utils.data.SubsetRandomSampler(train_index)
        val_sampler = torch.utils.data.SubsetRandomSampler(test_index)
        trainDataLoader = torch.utils.data.DataLoader(dataset, sampler=train_sampler, batch_size=batch_size,
                                                      num_workers=4, drop_last=True)
        testDataLoader = torch.utils.data.DataLoader(dataset, sampler=val_sampler, batch_size=batch_size,
                                                     num_workers=4, drop_last=True)
        # 分割数据集
        # train_dataset = torch.utils.data.Subset(dataset, train_index)
        # test_dataset = torch.utils.data.Subset(dataset, test_index)
        #
        # trainDataLoader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
        #                                               num_workers=4, drop_last=True)
        # testDataLoader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
        #                                              num_workers=4)

        observer.log(f'[DEBUG] Observer init successfully, program start @{current_time}\n')
        # 模型加载
        _model = experiment_settings['Model']
        if model_index == 'MDL':
            model = _model(model_depth=18, in_planes=1, num_classes=2)
        elif model_index == 'RLAD':
            _, model = _model()
        else:
            print(f"The name of model will run {_model}")
            model = _model()
        # 使用 DataParallel 进行多GPU训练
        # if torch.cuda.device_count() > 1 and data_parallel == 1:
        #     observer.log("Using " + str(torch.cuda.device_count()) + " GPUs for training.\n")
        #     model = torch.nn.DataParallel(model)

        observer.log(f'Use model : {str(experiment_settings)}\n')
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        observer.log("\n===============================================\n")
        observer.log("model parameters: " + str(num_params))
        observer.log("\n===============================================\n")

        # 超参数设置
        optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'])
        # 定义一个filter，只传入requires_grad=True的模型参数
        # optimizer = experiment_settings['Optimizer'](filter(lambda p: p.requires_grad, model.parameters()),
        #                                              experiment_settings['Lr'])

        # MDL
        # optimizer = experiment_settings['Optimizer'](model.parameters(), lr=experiment_settings['Lr'],
        #                                              weight_decay=experiment_settings['weight_decay'],
        #                                              momentum=experiment_settings['momentum'])

        # RLAD
        # optimizer = experiment_settings['Optimizer'](model.parameters(), experiment_settings['Lr'],
        #                                              weight_decay=experiment_settings['weight_decay'])

        scheduler = experiment_settings['Scheduler'](optimizer, others_params)

        if 'w1' in experiment_settings:
            criterion = experiment_settings['Loss'](w1=experiment_settings['w1'], w2=experiment_settings['w2'])
        elif experiment_settings['Name'] == 'MDL_Net':
            criterion = experiment_settings['Loss'](label_smoothing=experiment_settings['label_smoothing'])
        else:
            criterion = experiment_settings['Loss']()

        print("Prepare completed for fold {}! Launch training!\U0001F680".format(fold))

        # 启动训练
        _run = experiment_settings['Run']
        _run(observer, others_params.epochs, trainDataLoader, testDataLoader, model, device,
             optimizer, criterion, scheduler, fold)

        # 收集评估指标
        metrics['accuracy'].append(observer.best_dicts['Accuracy'])
        metrics['precision'].append(observer.best_dicts['Precision'])
        metrics['recall'].append(observer.best_dicts['Recall'])
        metrics['Specificity'].append(observer.best_dicts['Specificity'])
        metrics['balanceAccuracy'].append(observer.best_dicts['BalanceAccuracy'])
        metrics['auc'].append(observer.best_dicts['AuRoc'])
        metrics['f1'].append(observer.best_dicts['F1'])
    print("Cross-validation training completed for all folds.")
    return metrics

if __name__ == "__main__":
    args = parse_args()
    print(args)
    time_start = time.time()
    best_metrics = prepare_to_train(mri_dir=args.mri_dir, pet_dir=args.pet_dir, cli_dir=args.cli_dir,
                                    csv_file=args.csv_file, batch_size=args.batch_size, model_index=args.model,
                                    seed=args.seed, device=args.device, data_parallel=args.data_parallel,
                                    n_splits=args.n_splits, others_params=args)
    time_over = time.time()
    use_time = time_over - time_start
    # 计算小时、分钟和秒
    hours = use_time // 3600
    minutes = (use_time % 3600) // 60
    seconds = use_time % 60
    # 打印总训练时间
    times_result = f'Total time: {hours}h {minutes}m {seconds}s'
    print(f"Metrics:{best_metrics}")
    print(times_result)
    with open(args.logs, 'a') as f:
        f.write(f"Metrics:{best_metrics} \n ")
        f.write(times_result + '\n')

