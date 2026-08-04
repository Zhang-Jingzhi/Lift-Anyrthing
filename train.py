import os
import argparse
import csv
import torch
import wandb
from datetime import datetime
from omegaconf import OmegaConf
from dataset.CMapDataset import create_dataloader
from dataset.BimanualPairDataset import create_bimanual_dataloader
from model.tro_graph import RobotGraph


def save_checkpoint(checkpoint, path):
    """Write checkpoints atomically so an evaluator never sees a partial file."""
    temporary_path = f"{path}.tmp"
    torch.save(checkpoint, temporary_path)
    os.replace(temporary_path, path)


def append_metric_row(path, row):
    exists = os.path.isfile(path)
    with open(path, "a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def prepare_input(batch, device):

    robot_pc_initial = batch['robot_pc_initial']
    for batch_robot_pc_initial in robot_pc_initial:
        for link_name, link_pc in batch_robot_pc_initial.items():
            batch_robot_pc_initial[link_name] = link_pc.to(device)

    robot_pc_target = batch['robot_pc_target']
    for batch_robot_pc_target in robot_pc_target:
        for link_name, link_pc in batch_robot_pc_target.items():
            batch_robot_pc_target[link_name] = link_pc.to(device)

    batch['object_pc'] = batch['object_pc'].to(device)
    batch['object_pc_normal'] = batch['object_pc_normal'].to(device)
    batch['initial_q'] = [x.to(device) for x in batch['initial_q']]
    batch['target_q'] = [x.to(device) for x in batch['target_q']]
    batch['initial_se3'] = [x.to(device) for x in batch['initial_se3']]
    batch['target_se3'] = [x.to(device) for x in batch['target_se3']]
    batch['initial_vec'] = [x.to(device) for x in batch['initial_vec']]
    batch['target_vec'] = [x.to(device) for x in batch['target_vec']]
    
    return batch


def train(config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    save_dir = config.train.save_dir
    os.makedirs(save_dir, exist_ok=True)
    
    wandb.init(
        project=config.train.project_name,
        config=OmegaConf.to_container(config, resolve=True),
        name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        mode=config.train.get("wandb_mode", None),
    )

    print("Building dataloader...")
    if config.dataset.get("type", "cmap") == "bimanual_pairs":
        dataloader = create_bimanual_dataloader(config.dataset)
    else:
        dataloader = create_dataloader(config.dataset, is_train=True)
    print("Building model...")
    model = RobotGraph(**config.model).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params {total_params}")
    wandb.config.update({"total_params": total_params})

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.train.lr
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=config.train.lr_step,
        gamma=config.train.lr_gamma
    )

    warm_start_from = config.train.get("warm_start_from", False)
    if config.train.resume_from and warm_start_from:
        raise ValueError(
            "Use either resume_from or warm_start_from, not both"
        )
    if config.train.resume_from:
        ckpt = torch.load(config.train.resume_from)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        start_epoch = ckpt["epoch"]
        print(f"Resumed from {config.train.resume_from} at epoch {start_epoch}")
    elif warm_start_from:
        ckpt = torch.load(warm_start_from, map_location=device)
        incompatible = model.load_state_dict(
            ckpt["model_state"],
            strict=False,
        )
        print(
            f"Warm-started model weights from {warm_start_from}; "
            f"missing={list(incompatible.missing_keys)}, "
            f"unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
        start_epoch = 0
    else:
        start_epoch = 0

    num_epochs = config.train.epochs
    metric_path = os.path.join(save_dir, "batch_metrics.csv")
    stop_after_epoch = config.train.get("stop_after_epoch", None)
    max_batches_per_epoch = config.train.get("max_batches_per_epoch", None)
    log_interval = config.train.get("log_interval", 100)
    save_latest_every_epoch = config.train.get("save_latest_every_epoch", False)
    for epoch in range(start_epoch, num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs} started.", flush=True)
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_id, batch in enumerate(dataloader):

            batch = prepare_input(batch, device)
            loss_dict = model(batch)
            loss = loss_dict['loss_total']
            optimizer.zero_grad()
            loss.backward()
            grad_clip_norm = config.train.get("grad_clip_norm", None)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    float(grad_clip_norm),
                )
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1

            log_data = {k: v.item() for k, v in loss_dict.items()}
            log_data.update({"lr": scheduler.get_last_lr()[0]})
            wandb.log(log_data)
            append_metric_row(
                metric_path,
                {
                    "epoch": epoch + 1,
                    "batch": num_batches,
                    **log_data,
                },
            )
            if num_batches == 1 or num_batches % log_interval == 0:
                print(
                    f"Epoch {epoch + 1}/{num_epochs} "
                    f"batch {num_batches}/{len(dataloader)} "
                    f"loss={loss.item():.6f}",
                    flush=True
                )
            if max_batches_per_epoch is not None and num_batches >= max_batches_per_epoch:
                break

        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches
        wandb.log({"epoch_avg_loss": avg_epoch_loss})
        print(
            f"Epoch {epoch + 1}/{num_epochs} completed: "
            f"avg_loss={avg_epoch_loss:.6f}",
            flush=True
        )

        checkpoint = {
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict()
        }
        if save_latest_every_epoch:
            os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)
            save_checkpoint(
                checkpoint,
                os.path.join(save_dir, "ckpt", "latest.pth"),
            )

        if (epoch + 1) % config.train.save_interval == 0:
            os.makedirs(os.path.join(save_dir, "ckpt"), exist_ok=True)
            ckpt_path = os.path.join(save_dir, "ckpt", f"{epoch+1}.pth")
            save_checkpoint(checkpoint, ckpt_path)
            wandb.save(ckpt_path)

        if stop_after_epoch is not None and epoch + 1 >= stop_after_epoch:
            print(
                f"Reached requested stop epoch {stop_after_epoch}; "
                "checkpoint saved.",
                flush=True,
            )
            break
    wandb.finish()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="config file")
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="override train.resume_from",
    )
    parser.add_argument(
        "--warm-start-from",
        type=str,
        default=None,
        help="load model weights only and start a fresh optimizer",
    )
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        default=None,
        help="stop cleanly after saving this epoch",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="override train.epochs",
    )
    parser.add_argument(
        "--max-batches-per-epoch",
        type=int,
        default=None,
        help="override train.max_batches_per_epoch",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=None,
        help="override train.save_dir",
    )
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    if args.resume_from is not None:
        config.train.resume_from = args.resume_from
    if args.warm_start_from is not None:
        config.train.warm_start_from = args.warm_start_from
    if args.stop_after_epoch is not None:
        config.train.stop_after_epoch = args.stop_after_epoch
    if args.epochs is not None:
        config.train.epochs = args.epochs
    if args.max_batches_per_epoch is not None:
        config.train.max_batches_per_epoch = args.max_batches_per_epoch
    if args.save_dir is not None:
        config.train.save_dir = args.save_dir
    train(config)
