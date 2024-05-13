import numpy as np
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
from time import time

import wandb
from dynamics_training_loop import configure_epd_models, configure_val_dataset, configure_train_dataset_and_loader, \
    epd_train_step, epd_train_step_with_checkpoint, prepare_training, validate_loop, to_device, \
    configure_loss, configure_optimizers, dump_state,  parse_args_and_config, configure_residual_stat
from utils.steps_scheduler import CurriculumSampler
import gc
from datetime import timedelta


def main(args, config):
    # this train loop uses a curriculum schedule of rollout steps
    # according the the curriculum, the rollout steps will be increased

    assert torch.cuda.is_available(), "Training currently requires at least one GPU."
    # retrieve specified gpu id from config
    torch.backends.cudnn.deterministic = True
    torch.set_float32_matmul_precision('high')
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ['NCCL_BLOCKING_WAIT'] = '0'

    # Setup DDP:
    dist.init_process_group("nccl", timeout=timedelta(seconds=7200000),)
    rank = dist.get_rank()
    # visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
    # print(f"Visible devices: {visible_devices}")
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)

    # print free memory on this device
    # free_memory = torch.cuda.get_device_properties(device).total_memory - torch.cuda.memory_reserved(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # prepare wandb logging
    if rank == 0:
        wandb.init(project=config.project_name,
                   config=config)
    logger, log_dir = prepare_training(args, config)
    model = configure_epd_models(config)    # train from scratch
    model.to(torch.device(device))

    # use the standard linear warmup + cosine annealing schedule
    optim, sched = configure_optimizers(config, model)

    # optionally resume from checkpoint
    global_step = 0

    if config.training.resume_from_checkpoint:
        # currently only resume model state
        checkpoint = torch.load(config.training.resume_checkpoint_path, map_location=torch.device(device))
        model.load_state_dict(checkpoint['model'])
        optim.load_state_dict(checkpoint['optimizer'])
        sched.load_state_dict(checkpoint['scheduler'])
        global_step = checkpoint['global_step']
        if rank == 0:
            print(f"Resumed from checkpoint at global step {checkpoint['global_step']}.")
        torch.cuda.empty_cache()

    model_ddp = DDP(model, device_ids=[rank])

    # compile model, usually much faster
    # torch._dynamo.config.optimize_ddp = False   # this prevents error in some cases and in fact does not slow down the process

    if config.training.use_compile:  # currently does not work with gradient checkpointing
        model_ddp_cpd = torch.compile(model_ddp)
    else:
        model_ddp_cpd = model_ddp

    if rank == 0:
        print('Building datasets...')

    # construct curriculum sampler
    curriculum_scheduler = CurriculumSampler(
        values=config.training.curriculum_values,
        milestone=config.training.curriculum_milestone
    )

    train_dataset, train_dataloader =\
        configure_train_dataset_and_loader(config.training.curriculum_values[0]+1,
                                           config.training.init_batch_size,
                                           config)

    residual_normalizer = configure_residual_stat(config)
    # to device
    residual_normalizer = residual_normalizer.to(device)

    # only load valid data on rank 0
    if rank == 0:
        valsteps = config.data.valsteps
        val_dataset = configure_val_dataset(valsteps, config)
    else:
        val_dataset = None

    training_iter = iter(train_dataloader)

    max_steps = config.training.max_steps

    if rank == 0:
        logger.info(f"max_steps: {max_steps}")
        logger.info("Starting training loop...")

    # construct loss function
    training_loss_module = configure_loss(config)
    training_loss_module.to(device)

    grad_post_proc = lambda x: nn.utils.clip_grad_norm_(model_ddp_cpd.parameters(),
                                                        config.training.max_grad_norm)

    start_time = time()

    while global_step < max_steps:

        rollout_steps, has_changed = curriculum_scheduler.get_value(global_step)
        if has_changed:  # reconstruct the dataloader
            del train_dataloader, train_dataset
            # garbage collection
            torch.cuda.empty_cache()
            gc.collect()

            train_dataset, train_dataloader = configure_train_dataset_and_loader(rollout_steps+1,
                                                                                 config.training.final_batch_size,
                                                                                 config)

            # if compile, re-compile the model to account for batch size change
            if config.training.use_compile and \
                config.training.init_batch_size != config.training.final_batch_size:
                model_ddp_cpd = torch.compile(model_ddp)   # re-compile

            training_iter = iter(train_dataloader)
            if rank == 0:
                logger.info(f"Rollout steps changed to {rollout_steps} at global step {global_step}.")

        try:
            batch = next(training_iter)
        except StopIteration:
            training_iter = iter(train_dataloader)
            batch = next(training_iter)

        # retrieve things from batch
        batch = to_device(batch, device)
        surface_in_feat, surface_target_feat, multi_level_in_feat, multi_level_target_feat, constants = batch

        if rollout_steps > config.training.gradient_checkpointing_segment_size:
            loss = \
                epd_train_step_with_checkpoint(model_ddp_cpd, surface_in_feat, surface_target_feat,
                                              multi_level_in_feat, multi_level_target_feat,
                                              constants,
                                              optim, sched, training_loss_module, grad_post_proc,
                                              residual_normalizer,
                                               config.training.gradient_checkpointing_segment_size)
        else:
            loss = \
                epd_train_step(model_ddp_cpd, surface_in_feat, surface_target_feat,
                                                      multi_level_in_feat, multi_level_target_feat,
                                                      constants,
                                                      optim, sched, training_loss_module,
                                                      grad_post_proc,
                                                    residual_normalizer)

        if global_step % config.training.ckpt_every == 0:
            # only do this on rank zero
            if rank == 0:
                dump_state(model_ddp, optim, sched, global_step, log_dir)

            dist.barrier()

        if global_step % config.training.validate_every == 0:
            # only do this on rank zero
            if rank == 0:
                validate_loop(model_ddp.module, config.data.val_timestamps,
                              logger, global_step, val_dataset,
                              config.training.val_batch_size, config, device)
                gc.collect()

            dist.barrier()

        global_step += 1
        if rank == 0:
            if global_step % config.training.print_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = config.training.print_every / (end_time - start_time)
                print(
                    f' Step: {global_step}'
                    f' Pred Loss: {np.round(loss, 4)}'
                    f' LR: {np.round(optim.param_groups[0]["lr"], 6)}'
                    f' Steps/sec: {np.round(steps_per_sec, 3)}'
                    f' ETA: {np.round((max_steps - global_step) / steps_per_sec / 3600, 3)}h'
                    f' Rollout steps: {rollout_steps}'
                )
                start_time = time()
            wandb.log({
                'loss': loss,   # to match with previous experiments
                'lr': optim.param_groups[0]['lr'],
                'rollout_steps': rollout_steps
            })

    if rank == 0:
        dump_state(model_ddp, optim, sched, global_step, log_dir)

        validate_loop(model_ddp.module, config.data.val_timestamps,
                      logger, global_step, val_dataset,
                      config.training.val_batch_size, config, device)
    dist.barrier()
    logger.info('Training finished...')
    wandb.finish()

    dist.destroy_process_group()


if __name__ == "__main__":
    args, config = parse_args_and_config()
    main(args, config)
