import os
import json
import time
import types
import inspect
import argparse
import datetime
import numpy as np
from pathlib import Path
import albumentations as albu
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter
import IMDLBenCo.training_scripts.utils.misc as misc

# Register for generate function or CLASS from string
from IMDLBenCo.registry import MODELS, POSTFUNCS

from IMDLBenCo.datasets import ManiDataset, JsonDataset, BalancedDataset
from IMDLBenCo.transforms import RandomCopyMove, RandomInpainting

from IMDLBenCo.evaluation import PixelF1, ImageF1, ImageAccuracy, ImageAUC # TODO You can select evaluator you like here

from IMDLBenCo.training_scripts.tester import test_one_epoch
from IMDLBenCo.training_scripts.trainer import train_one_epoch
from IMDLBenCo.model_zoo.mvss_net.mvssnet import MVSSNet
from ForensicHub.tasks.deepfake.wrapper.wrappers import DeepfakeOutputWrapper, BencoOutputWrapper
from ForensicHub import MODELS
from ForensicHub.tasks.deepfake.datasets.get_loaders import prepare_testing_data,prepare_training_data
import yaml
def get_args_parser():
    parser = argparse.ArgumentParser('IMDLBenCo training launch!', add_help=True)

    # -------------------------------
    # Model name
    parser.add_argument('--model', default=None, type=str,
                        help='The name of applied model', required=True)
    
    # 可以接受label的模型是否接受label输入，并启用相关的loss。
    parser.add_argument('--if_predict_label', action='store_true',
                        help='Does the model that can accept labels actually take label input and enable the corresponding loss function?')
    # ----Dataset parameters 数据集相关的参数----
    parser.add_argument('--image_size', default=512, type=int,
                        help='image size of the images in datasets')
    
    parser.add_argument('--if_padding', action='store_true',
                        help='padding all images to same resolution.')
    
    parser.add_argument('--if_resizing', action='store_true', 
                        help='resize all images to same resolution.')
    # If edge mask activated
    parser.add_argument('--edge_mask_width', default=None, type=int,
                        help='Edge broaden size (in pixels) for edge maks generator.')
    parser.add_argument('--data_path', default='/root/Dataset/CASIA2.0/', type=str,
                        help='dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    parser.add_argument('--test_data_path', default='/root/Dataset/CASIA1.0', type=str,
                        help='test dataset path, should be our json_dataset or mani_dataset format. Details are in readme.md')
    # ------------------------------------
    # training related
    parser.add_argument('--batch_size', default=1, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--test_batch_size', default=2, type=int,
                        help="batch size for testing")
    parser.add_argument('--epochs', default=200, type=int)
    # Test related
    parser.add_argument('--no_model_eval', action='store_true', 
                        help='Do not use model.eval() during testing.')
    parser.add_argument('--test_period', default=4, type=int,
                        help="how many epoch per testing one time")
    
    # 一个epoch在tensorboard中打几个loss的data point
    parser.add_argument('--log_per_epoch_count', default=20, type=int,
                        help="how many loggings (data points for loss) per testing epoch in Tensorboard")
    
    parser.add_argument('--find_unused_parameters', action='store_true',
                        help='find_unused_parameters for DDP. Mainly solve issue for model with image-level prediction but not activate during training.')
    
    # 不启用AMP（自动精度）进行训练
    parser.add_argument('--if_not_amp', action='store_false',
                        help='Do not use automatic precision.')
    parser.add_argument('--accum_iter', default=16, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--warmup_epochs', type=int, default=4, metavar='N',
                        help='epochs to warmup LR')

    # ----输出的日志相关的参数----------
    # ----output related parameters----
    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    # -----------------------
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint, input the path of a ckpt.')

    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--num_workers', default=1, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    args, remaining_args = parser.parse_known_args()
    # 获取对应的模型类
    # Get the corresponding model class
    model_class = MODELS.get(args.model)

    # 根据模型类动态创建参数解析器
    # Dynamically create a parameter parser based on the model class
    model_parser = misc.create_argparser(model_class)
    model_args = model_parser.parse_args(remaining_args)

    return args, model_args

def main(args, model_args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    print("=====Model args:=====")
    print("{}".format(model_args).replace(', ', ',\n'))
    device = torch.device(args.device)
    
    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)



    detector_path = '/mnt/data1/xuekang/workspace/DeepfakeBench/DeepfakeBench/training/config/config/detector/sbi.yaml'
    detector_path = '/mnt/data1/xuekang/workspace/DeepfakeBench/DeepfakeBench/training/config/config/detector/facexray.yaml'
    # detector_path = '/mnt/data1/xuekang/workspace/Wrapper/mvss.yaml'
    with open(detector_path, 'r') as f:
        config = yaml.safe_load(f)
    with open('/mnt/data1/xuekang/workspace/DeepfakeBench/DeepfakeBench/training/config/train_config.yaml', 'r') as f:
        config2 = yaml.safe_load(f)
    import pdb;pdb.set_trace()
    if 'label_dict' in config:
        config2['label_dict']=config['label_dict']
    config.update(config2)
    config['dataset_json_folder'] = '/mnt/data1/xuekang/workspace/DeepfakeBench/DeepfakeBench/preprocessing/dataset_json'

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        config['ddp'] = True
    else:
        config['ddp'] = False
    import pdb;pdb.set_trace()
    model_class = DETECTOR[config['model_name']]
    # model = model_class()
    # model = DeepfakeOutputWrapper(model,config)
    model = MVSSNet()
    model = BencoOutputWrapper(model)
    config['train_batchSize'] = args.batch_size
    config['test_batchSize'] = args.test_batch_size
    config['resolution'] = args.image_size
    data_loader_train = prepare_training_data(config)
    # prepare the testing data loader
    test_dataloaders = prepare_testing_data(config)
    

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    
    # ========define the model directly==========
    # model = IML_ViT(
    #     vit_pretrain_path = model_args.vit_pretrain_path,
    #     predict_head_norm= model_args.predict_head_norm,
    #     edge_lambda = model_args.edge_lambda
    # )
    
    # --------------- or -------------------------
    # Init model with registry
    # model = MODELS.get(args.model)
    # Filt usefull args
    # if isinstance(model,(types.FunctionType, types.MethodType)):
    #     model_init_params = inspect.signature(model).parameters
    # else:
    #     model_init_params = inspect.signature(model.__init__).parameters
    # combined_args = {k: v for k, v in vars(args).items() if k in model_init_params}
    # for k, v in vars(model_args).items():
    #     if k in model_init_params and k not in combined_args:
    #         combined_args[k] = v
    # model = model(**combined_args)
    # ============================================

    """
    TODO Set the evaluator you want to use
    You can use PixelF1, ImageF1, or any other evaluator you like.
    Available evaluators are in: https://github.com/scu-zjz/IMDLBenCo/blob/main/IMDLBenCo/evaluation/__init__.py
    """    
    evaluator_list = [
        # ImageAccuracy(threshold=0.5),
        ImageAUC(),
        # ImageF1(threshold=0.5)
    ]
    
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
    
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module
    
    # TODO You can set optimizer settings here
    args.opt='AdamW'
    args.betas=(0.9, 0.999)
    args.momentum=0.9
    optimizer  = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_evaluate_metric_value = 0
    for epoch in range(args.start_epoch, args.epochs):
        # if args.distributed:
        #     data_loader_train.sampler.set_epoch(epoch)
        # train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=None,
            log_per_epoch_count=args.log_per_epoch_count,
            args=args
        )
        
        # # saving checkpoint
        if args.output_dir and (epoch % 25 == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            
        optimizer.zero_grad()
        # test for one epoch
        if epoch % args.test_period == 0 or epoch + 1 == args.epochs:
            values = {} # dict of dict (dataset_name: {metric_name: metric_value})
            # test across all datasets in the `test_data_loaders' dict
            for test_dataset_name, test_dataloader in test_dataloaders.items():
                print(f'!!!Start Test: {test_dataset_name}',len(test_dataloader))
                test_stats = test_one_epoch(
                    model, 
                    data_loader = test_dataloader, 
                    evaluator_list=evaluator_list,
                    device = device, 
                    epoch = epoch,
                    name = test_dataset_name, 
                    log_writer=None,
                    args = args,
                    is_test = False,
                )
                one_metric_value = {}
                # Read the metric value from the test_stats dict
                for evaluator in evaluator_list:
                    evaluate_metric_value = test_stats[evaluator.name]
                    one_metric_value[evaluator.name] = evaluate_metric_value
                values[test_dataset_name] = one_metric_value

            metrics_dict = {metric: {dataset: values[dataset][metric] for dataset in values} for metric in {m for d in values.values() for m in d}}
            # Calculate the mean of each metric across all datasets
            metric_means = {metric: np.mean(list(datasets.values())) for metric, datasets in metrics_dict.items()}
            # Calculate the mean of all metrics
            evaluate_metric_value = np.mean(list(metric_means.values()))
            
            # Store the best metric value 
            if evaluate_metric_value > best_evaluate_metric_value :
                best_evaluate_metric_value = evaluate_metric_value
                print(f"Best {' '.join([evaluator.name for evaluator in evaluator_list])} = {best_evaluate_metric_value}")
                # Save the best only after 20 epoch. TODO you can modify this.
                misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch)
            else:
                print(f"Average {' '.join([evaluator.name for evaluator in evaluator_list])} = {evaluate_metric_value}")
            # Log the metrics to Tensorboard
            if log_writer is not None:
                for metric, datasets in metrics_dict.items():
                    log_writer.add_scalars(f'{metric}_Metric', datasets, epoch)
                log_writer.add_scalar('Average', evaluate_metric_value, epoch)
            log_stats =  {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,}
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch,}
        
        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, model_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args)