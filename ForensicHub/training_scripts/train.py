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
from IMDLBenCo.training_scripts.trainer import train_one_epoch
from ForensicHub.common.utils.yaml import load_yaml_config, split_config, add_attr


def get_args_parser():
    parser = argparse.ArgumentParser('ForensicHub benchmark training launch!', add_help=True)
    parser.add_argument("--config", type=str, help="Path to YAML config file", required=True)

    args = parser.parse_args()
    config = load_yaml_config(args.config)

    # model_args, train_dataset_args, transform_args are dict type, test_dataset_args is dict list.
    args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = split_config(config)
    add_attr(args, output_dir=args.log_dir)
    add_attr(args, if_not_amp=not args.use_amp)
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

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    misc.seed_torch(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    transform = build_from_registry(TRANSFORMS, transform_args)
    train_transform = transform.get_train_transform()
    test_transform = transform.get_test_transform()
    post_transform = transform.get_post_transform()

    print("Train transform: ", train_transform)
    print("Test transform: ", test_transform)
    print("Post transform: ", post_transform)

    # get post function (if have)
    post_function_name = f"{model_args['name']}_post_func".lower()
    print(f"Post function check: {post_function_name}")
    print(POSTFUNCS)
    if POSTFUNCS.has(post_function_name):
        post_function = POSTFUNCS.get(post_function_name)
    else:
        post_function = None

    train_dataset_args["init_config"].update({
        "post_funcs": post_function,
        "common_transform": train_transform,
        "post_transform": post_transform
    })
    train_dataset = build_from_registry(DATASETS, train_dataset_args)

    test_dataset_list = {}
    for test_args in test_dataset_args:
        test_args["init_config"].update({
            "post_funcs": post_function,
            "common_transform": test_transform,
            "post_transform": post_transform
        })
        test_dataset_list[test_args["dataset_name"]] = build_from_registry(DATASETS, test_args)

    print(f"Train dataset: {train_dataset_args['dataset_name']}.")
    print(len(train_dataset))
    print(f"Test dataset: {[args['dataset_name'] for args in test_dataset_args]}.")
    print([len(dataset) for dataset in test_dataset_list.values()])

    test_sampler = {}
    if args.distributed:
        num_tasks = misc.get_world_size()
        global_rank = misc.get_rank()
        sampler_train = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=num_tasks, rank=global_rank, shuffle=True
        )
        for test_dataset_name, dataset_test in test_dataset_list.items():
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test,
                num_replicas=num_tasks,
                rank=global_rank,
                shuffle=False,
                drop_last=True
            )
            test_sampler[test_dataset_name] = sampler_test
        print("Sampler_train = %s" % str(sampler_train))
        print("Sampler_test = %s" % str(sampler_test))
    else:
        sampler_train = torch.utils.data.RandomSampler(train_dataset)
        for test_dataset_name, dataset_test in test_dataset_list.items():
            sampler_test = torch.utils.data.RandomSampler(dataset_test)
            test_sampler[test_dataset_name] = sampler_test

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    data_loader_train = torch.utils.data.DataLoader(
        train_dataset, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )

    test_dataloaders = {}
    for test_dataset_name in test_sampler.keys():
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset_list[test_dataset_name], sampler=test_sampler[test_dataset_name],
            batch_size=args.test_batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=True,
        )
        test_dataloaders[test_dataset_name] = test_dataloader

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

    eff_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()

    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * eff_batch_size / 256

    print("base lr: %.2e" % (args.lr * 256 / eff_batch_size))
    print("actual lr: %.2e" % args.lr)

    print("accumulate grad iterations: %d" % args.accum_iter)
    print("effective batch size: %d" % eff_batch_size)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=args.find_unused_parameters)
        model_without_ddp = model.module

    # TODO You can set optimizer settings here
    args.opt = 'AdamW'
    args.betas = (0.9, 0.999)
    args.momentum = 0.9
    optimizer = optim_factory.create_optimizer(args, model_without_ddp)
    print(optimizer)
    loss_scaler = misc.NativeScalerWithGradNormCount()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_evaluate_metric_value = 0
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # train for one epoch
        train_stats = train_one_epoch(
            model, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            log_writer=log_writer,
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
            values = {}  # dict of dict (dataset_name: {metric_name: metric_value})
            # test across all datasets in the `test_data_loaders' dict
            for test_dataset_name, test_dataloader in test_dataloaders.items():
                print(f'!!!Start Test: {test_dataset_name}', len(test_dataloader))
                test_stats = test_one_epoch(
                    model,
                    data_loader=test_dataloader,
                    evaluator_list=evaluator_list,
                    device=device,
                    epoch=epoch,
                    name=test_dataset_name,
                    log_writer=log_writer,
                    args=args,
                    is_test=False,
                )
                one_metric_value = {}
                # Read the metric value from the test_stats dict
                for evaluator in evaluator_list:
                    evaluate_metric_value = test_stats[evaluator.name]
                    one_metric_value[evaluator.name] = evaluate_metric_value
                values[test_dataset_name] = one_metric_value

            metrics_dict = {metric: {dataset: values[dataset][metric] for dataset in values} for metric in
                            {m for d in values.values() for m in d}}
            # Calculate the mean of each metric across all datasets
            metric_means = {metric: np.mean(list(datasets.values())) for metric, datasets in metrics_dict.items()}
            # Calculate the mean of all metrics
            evaluate_metric_value = np.mean(list(metric_means.values()))

            # Store the best metric value
            if evaluate_metric_value > best_evaluate_metric_value:
                best_evaluate_metric_value = evaluate_metric_value
                print(
                    f"Best {' '.join([evaluator.name for evaluator in evaluator_list])} = {best_evaluate_metric_value}")
                # Save the best only after record epoch.
                if epoch >= args.record_epoch:
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
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         **{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch, }
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args = get_args_parser()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args, model_args, train_dataset_args, test_dataset_args, transform_args, evaluator_args)
