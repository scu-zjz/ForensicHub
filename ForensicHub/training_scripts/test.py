import os
import json
import time
import argparse
import datetime
import numpy as np
from pathlib import Path
import timm.optim.optim_factory as optim_factory
from torch.utils.tensorboard import SummaryWriter

import ForensicHub.training_scripts.utils.misc as misc
from ForensicHub.registry import DATASETS, MODELS, POSTFUNCS, TRANSFORMS, EVALUATORS, build_from_registry
from ForensicHub.common.evaluation import PixelF1, ImageF1
from IMDLBenCo.training_scripts.tester import test_one_epoch
from ForensicHub.common.utils.yaml import load_yaml_config, split_config, add_attr


def get_args_parser():
    parser = argparse.ArgumentParser('ForensicHub benchmark training launch!', add_help=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    # model_args, train_dataset_args, transform_args are dict type, test_dataset_args is dict list.
    args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = split_config(config)
    add_attr(args, output_dir=args.log_dir)
    return args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args


def main(args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args):
    # init parameters for distributed training
    misc.init_distributed_mode(args)
    import torch.multiprocessing
    torch.multiprocessing.set_sharing_strategy('file_system')
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("=====args:=====")
    print("{}".format(args).replace(', ', ',\n'))
    device = torch.device(args.device)

    transform = build_from_registry(TRANSFORMS, transform_args)
    test_transform = transform.get_test_transform()
    post_transform = transform.get_post_transform()

    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
    else:
        global_rank = 0

    print("Test transform: ", test_transform)
    print("Post transform: ", post_transform)

    # Init model with registry
    model = build_from_registry(MODELS, model_args)

    # Init evaluators
    evaluator_list = []
    for eva_args in evaluator_args:
        evaluator_list.append(build_from_registry(EVALUATORS, eva_args))
    print(f"Evaluators: {evaluator_list}")

    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    model_without_ddp = model
    print("Model = %s" % str(model_without_ddp))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=False)
        model_without_ddp = model.module

    start_time = time.time()
    # get post function (if have)
    post_function_name = f"{model_args['name']}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None

    dataset_dict = {}
    dataset_logger = {}
    test_dataset_list = {}
    for test_args in test_dataset_args:
        test_args["init_config"].update({
            "post_funcs": post_function,
            "common_transform": test_transform,
            "post_transform": post_transform
        })
        test_dataset_list[test_args["dataset_name"]] = build_from_registry(DATASETS, test_args)
    for t_args, dataset in zip(test_dataset_args, test_dataset_list.values()):
        print(f"Test dataset: {t_args['dataset_name']}\n{str(dataset)}\n")

    # Start go through each datasets:
    for dataset_name, test_dataset in test_dataset_list.items():
        args.full_log_dir = os.path.join(args.log_dir, dataset_name)

        if global_rank == 0 and args.full_log_dir is not None:
            os.makedirs(args.full_log_dir, exist_ok=True)
            log_writer = SummaryWriter(log_dir=args.full_log_dir)
        else:
            log_writer = None
        dataset_logger[dataset_name] = log_writer

        # Sampler
        if args.distributed:
            sampler_test = torch.utils.data.DistributedSampler(
                test_dataset,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=True
            )
            print("Sampler_test = %s" % str(sampler_test))
        else:
            sampler_test = torch.utils.data.RandomSampler(test_dataset)

        dataloader_test = torch.utils.data.DataLoader(
            test_dataset,
            sampler=sampler_test,
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        dataset_dict[dataset_name] = dataloader_test
    print("dataset_dict", dataset_dict)

    chkpt_list = os.listdir(args.checkpoint_path)
    print(chkpt_list)
    chkpt_pair = [(int(chkpt.split('-')[1].split('.')[0]), chkpt) for chkpt in chkpt_list if chkpt.endswith(".pth")]
    chkpt_pair.sort(key=lambda x: x[0])
    print("sorted checkpoint pairs in the ckpt dir: ", chkpt_pair)
    for epoch, chkpt_dir in chkpt_pair:
        if chkpt_dir.endswith(".pth"):
            print("Loading checkpoint: %s" % chkpt_dir)
            ckpt = os.path.join(args.checkpoint_path, chkpt_dir)
            ckpt = torch.load(ckpt, map_location='cuda')
            model.module.load_state_dict(ckpt['model'])

            for dataset_name, dataloader_test in dataset_dict.items():
                print("Testing on dataset: %s" % dataset_name)
                test_stats = test_one_epoch(
                    model=model,
                    data_loader=dataloader_test,
                    evaluator_list=evaluator_list,
                    device=device,
                    epoch=epoch,
                    name="normal",
                    log_writer=dataset_logger[dataset_name],
                    args=args
                )
                log_stats = {
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    'epoch': epoch
                }
                if args.full_log_dir and misc.is_main_process():
                    if dataset_logger[dataset_name] is not None:
                        dataset_logger[dataset_name].flush()
                    with open(os.path.join(args.full_log_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
        local_time = time.time() - start_time
        local_time_str = str(datetime.timedelta(seconds=int(local_time)))
        print(f'Testing on ckpt {chkpt_dir} takes {local_time_str}')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Total testing time {}'.format(total_time_str))
    exit(0)


if __name__ == '__main__':
    args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args)
