"""Train AR-FlowMatching model."""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.rcParams['font.family'] = ['Comic Sans MS', 'FZKaTong-M19S']
plt.rcParams['axes.unicode_minus'] = False

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import torch.backends.cudnn as cudnn
import argparse
import os
from pathlib import Path
from tqdm import tqdm
import time

from arflow import ARFlowMatching, EulerSampler, SigmoidTimeField, EMAModel
from arflow import get_timestamped_output_dir, save_args_to_txt
from data_loader import get_unlabeled_dataloaders, get_labeled_dataloaders, get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Train AR-FlowMatching')

    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'cifar10', 'celeba'],
                        help='Dataset name')
    parser.add_argument('--data_root', type=str, default='./data/MNIST',
                        help='Data root directory')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='Data loading workers')
    parser.add_argument('--use_augmentation', action='store_true',
                        help='Enable data augmentation')
    parser.add_argument('--backbone', type=str, default='spade_unet',
                        choices=['spade_unet'],
                        help='Model architecture')
    parser.add_argument('--base_channels', type=int, default=64,
                        help='U-Net base channels')
    parser.add_argument('--channel_mult', type=int, nargs='+', default=[1, 2, 4, 8, 16],
                        help='U-Net channel multiplier list')
    parser.add_argument('--num_res_blocks', type=int, default=2,
                        help='Residual blocks per layer')
    parser.add_argument('--spade_hidden_nc', type=int, default=128,
                        help='SPADE hidden dimension')
    parser.add_argument('--label_embed_dim', type=int, default=128,
                        help='Label embedding dimension')
    parser.add_argument('--time_embed_dim', type=int, default=256,
                        help='Time embedding dimension')
    parser.add_argument('--time_start_delay', type=float, default=0.3,
                        help='Edge start delay')
    parser.add_argument('--time_power', type=float, default=1.0,
                        help='Distance power exponent')
    parser.add_argument('--time_k', type=float, default=2.0,
                        help='Sigmoid steepness')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='LR warmup epochs')
    parser.add_argument('--eval_every', type=int, default=10,
                        help='Evaluate every N epochs')
    parser.add_argument('--sample_steps', type=int, default=30,
                        help='Inference steps')
    parser.add_argument('--num_samples', type=int, default=36,
                        help='Number of generated samples')
    parser.add_argument('--output_dir', type=str, default='./outputs',
                        help='Output directory')
    parser.add_argument('--log_dir', type=str, default='./logs',
                        help='Log directory')
    parser.add_argument('--save_every', type=int, default=20,
                        help='Save every N epochs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Checkpoint path to resume')
    parser.add_argument('--use_labels', action='store_true',
                        help='Use label conditioning')
    parser.add_argument('--num_classes', type=int, default=10,
                        help='Number of classes')
    parser.add_argument('--use_cfg', action='store_true',
                        help='Use Classifier-Free Guidance')
    parser.add_argument('--cfg_drop_prob', type=float, default=0.1,
                        help='CFG label drop probability')
    parser.add_argument('--use_ema', action='store_true',
                        help='Use EMA')
    parser.add_argument('--ema_decay', type=float, default=0.9999,
                        help='EMA decay rate')
    parser.add_argument('--ema_update_after', type=int, default=0,
                        help='EMA delayed start step')
    parser.add_argument('--ema_update_every', type=int, default=10,
                        help='Update EMA every N steps')
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='Gradient clipping (0=no clipping)')
    parser.add_argument('--min_lr_ratio', type=float, default=0.01,
                        help='Cosine schedule min LR ratio')
    parser.add_argument('--attention_heads', type=int, default=4,
                        help='Attention heads')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Residual block dropout')
    parser.add_argument('--use_amp', action='store_true',
                        help='Use mixed precision (requires CUDA)')

    return parser.parse_args()


def get_lr_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.1):
    """Cosine LR scheduler with warmup and min LR floor."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        else:
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine = 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159265359))).item()
            return cosine * (1 - min_lr_ratio) + min_lr_ratio

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_checkpoint(model, optimizer, scheduler, epoch, args, filepath, ema_model=None):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'args': vars(args),
        'use_ema': ema_model is not None
    }
    if ema_model is not None:
        checkpoint['ema_state_dict'] = ema_model.state_dict()

    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer, scheduler, filepath, ema_model=None):
    """Load training checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Restore EMA state from checkpoint
    if ema_model is not None and 'ema_state_dict' in checkpoint:
        ema_model.load_state_dict(checkpoint['ema_state_dict'])

    return checkpoint['epoch']


def visualize_time_field(model, save_path, num_steps=5, image_size=28):
    """Visualize time field evolution."""
    import matplotlib.pyplot as plt
    import numpy as np

    device = next(model.parameters()).device
    time_field = model.time_field

    taus = torch.linspace(0, 1, num_steps, device=device)
    time_maps = []
    for tau in taus:
        t_map = time_field.get_time_map(tau, (1, 1, image_size, image_size))
        time_maps.append(t_map.squeeze().cpu().numpy())

    fig, axes = plt.subplots(1, num_steps, figsize=(num_steps * 2, 2))
    for i, (ax, t_map) in enumerate(zip(axes, time_maps)):
        im = ax.imshow(t_map, cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'tau={taus[i]:.2f}')
        ax.axis('off')
    plt.suptitle('Time Field Evolution (Center -> Edge)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def sample_images(model, sampler, num_samples, device, save_path,
                   in_channels=1, image_size=28):
    """Generate and save sample images."""
    import matplotlib.pyplot as plt
    import numpy as np

    samples = sampler.sample_with_intermediate(
        shape=(num_samples, in_channels, image_size, image_size),
        num_intermediate=5
    )
    final_samples = samples[-1]

    grid_size = int(np.ceil(np.sqrt(num_samples)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < len(final_samples):
            if in_channels == 1:
                img = final_samples[i].squeeze().cpu().numpy()
                ax.imshow(img, cmap='gray_r', vmin=-1, vmax=1)
            else:
                img = final_samples[i].permute(1, 2, 0).cpu().numpy()
                img = (img + 1) / 2
                ax.imshow(img.clip(0, 1))
        ax.axis('off')

    plt.suptitle('Generated Samples')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def sample_trajectory(model, sampler, device, save_path,
                      in_channels=1, image_size=28):
    """Save single-sample generation trajectory."""
    import matplotlib.pyplot as plt

    samples = sampler.sample_with_intermediate(
        shape=(1, in_channels, image_size, image_size),
        num_intermediate=8
    )

    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.flatten()
    total_steps = len(samples) - 1

    for i, (ax, sample) in enumerate(zip(axes, samples)):
        if in_channels == 1:
            img = sample[0, 0].cpu().numpy()
            ax.imshow(img, cmap='gray_r', vmin=-1, vmax=1)
        else:
            img = sample[0].permute(1, 2, 0).cpu().numpy()
            img = (img + 1) / 2
            ax.imshow(img.clip(0, 1))
        progress = i / total_steps * 100
        ax.set_title(f'Progress {progress:.0f}%')
        ax.axis('off')

    plt.suptitle('Generation Trajectory (Noise -> Clear)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100)
    plt.close()


def train_epoch(model, train_loader, optimizer, device, epoch, use_labels=False, ema_model=None, start_step=0, grad_clip=1.0, scaler=None):
    """Train one epoch. Returns (avg_loss, avg_active_ratio, step)."""
    model.train()
    total_loss = 0
    total_active_ratio = 0
    num_batches = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    step = start_step

    if use_labels:
        for x_1, labels in pbar:
            x_1 = x_1.to(device)
            labels = labels.to(device)
            batch_size = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            tau = torch.rand(batch_size, device=device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    loss, info = model.get_loss(x_0, x_1, tau, labels=labels)

                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    # Anomalous gradient detection: NaN or >10000x clip threshold
                    if torch.isnan(grad_norm) or grad_norm > grad_clip * 10000:
                        print(f"\nWarning: Anomalous gradient norm {grad_norm:.2f}, skipping update")
                        scaler.update()
                        continue
                else:
                    scaler.unscale_(optimizer)

                scaler.step(optimizer)
                scaler.update()
            else:
                loss, info = model.get_loss(x_0, x_1, tau, labels=labels)
                loss.backward()

                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    if torch.isnan(grad_norm):
                        print(f"\nWarning: Anomalous gradient norm {grad_norm:.2f}, skipping update")
                        continue

                optimizer.step()

            if ema_model is not None:
                ema_model.update(model, step)
            step += 1

            total_loss += loss.item()
            total_active_ratio += info['active_ratio']
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'active': f"{info['active_ratio']:.2%}"
            })
    else:
        for x_1 in pbar:
            x_1 = x_1.to(device)
            batch_size = x_1.shape[0]
            x_0 = torch.randn_like(x_1)
            tau = torch.rand(batch_size, device=device)

            optimizer.zero_grad()

            if scaler is not None:
                with autocast('cuda'):
                    loss, info = model.get_loss(x_0, x_1, tau)

                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                    # Anomalous gradient detection
                    if torch.isnan(grad_norm) or grad_norm > grad_clip * 10000:
                        print(f"\nWarning: Anomalous gradient norm {grad_norm:.2f}, skipping update")
                        scaler.update()
                        continue
                scaler.step(optimizer)
                scaler.update()
            else:
                loss, info = model.get_loss(x_0, x_1, tau)
                loss.backward()

                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()

            if ema_model is not None:
                ema_model.update(model, step)
            step += 1

            total_loss += loss.item()
            total_active_ratio += info['active_ratio']
            num_batches += 1

            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'active': f"{info['active_ratio']:.2%}"
            })

    avg_loss = total_loss / num_batches
    avg_active = total_active_ratio / num_batches

    return avg_loss, avg_active, step


def main():
    args = parse_args()

    dataset_name = args.dataset.upper()

    if args.resume:
        output_dir = Path(args.resume).parent
        print(f"Resuming training, output: {output_dir}")
    else:
        output_dir = get_timestamped_output_dir(args.output_dir, dataset_name)

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir / f'run_{time.strftime("%Y%m%d_%H%M%S")}')

    save_args_to_txt(args, output_dir / 'args.txt', script_name=f'ARFM-{dataset_name}')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        cudnn.benchmark = True

    if args.use_amp and not torch.cuda.is_available():
        print("Warning: AMP requires CUDA, disabled")
        args.use_amp = False

    if args.dataset == 'cifar10':
        in_channels = 3
        image_size = 32
        default_data_root = './data/CIFAR10'
        print(f"Dataset: CIFAR-10 (32x32 RGB)")
    elif args.dataset == 'celeba':
        in_channels = 3
        image_size = 64
        default_data_root = './data/celeba'
        print(f"Dataset: CelebA (64x64 RGB, 40 attributes)")
    else:
        in_channels = 1
        image_size = 28
        default_data_root = './data/MNIST'
        print(f"Dataset: MNIST (28x28 Grayscale)")

    if args.data_root == './data/MNIST':
        args.data_root = default_data_root

    if args.use_labels:
        if args.dataset == 'celeba':
            num_classes = 40
        else:
            num_classes = args.num_classes
    else:
        num_classes = 0

    model = ARFlowMatching(
        backbone=args.backbone,
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=args.base_channels,
        channel_mult=tuple(args.channel_mult),
        num_res_blocks=args.num_res_blocks,
        spade_hidden_nc=args.spade_hidden_nc,
        time_embed_dim=args.time_embed_dim,
        label_embed_dim=args.label_embed_dim,
        time_start_delay=args.time_start_delay,
        time_power=args.time_power,
        time_k=args.time_k,
        num_classes=num_classes,
        use_cfg=args.use_cfg if args.use_labels else False,
        cfg_drop_prob=args.cfg_drop_prob if args.use_labels else 0.0,
        attention_heads=args.attention_heads,
        dropout=args.dropout
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {num_params:,} params, SPADE U-Net base_ch={args.base_channels}, mult={args.channel_mult}")
    if args.use_labels:
        print(f"Conditional mode: num_classes={num_classes}, CFG={args.use_cfg}")

    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            model,
            decay=args.ema_decay,
            update_after_step=args.ema_update_after,
            update_every=args.ema_update_every,
            device=device
        )

    train_loader, test_loader = get_dataloaders(
        dataset=args.dataset,
        root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        labeled=args.use_labels,
        use_augmentation=args.use_augmentation
    )
    print(f"Device: {device} | Batches: {len(train_loader)}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = get_lr_scheduler(
        optimizer,
        args.warmup_epochs,
        args.epochs,
        min_lr_ratio=args.min_lr_ratio
    )

    scaler = GradScaler('cuda') if args.use_amp else None
    sampler = EulerSampler(model, num_steps=args.sample_steps)

    start_epoch = 0
    start_step = 0
    if args.resume:
        start_epoch = load_checkpoint(model, optimizer, scheduler, args.resume, ema_model) + 1

    visualize_time_field(model, output_dir / 'time_field.png', image_size=image_size)

    best_loss = float('inf')

    for epoch in range(start_epoch, args.epochs):
        train_loss, active_ratio, start_step = train_epoch(
            model, train_loader, optimizer, device, epoch + 1,
            use_labels=args.use_labels,
            ema_model=ema_model,
            start_step=start_step,
            grad_clip=args.grad_clip,
            scaler=scaler
        )

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('LR', current_lr, epoch)
        writer.add_scalar('ActiveRatio', active_ratio, epoch)

        print(f"Epoch {epoch+1}/{args.epochs}: "
              f"Loss={train_loss:.4f}, LR={current_lr:.6f}, "
              f"Active={active_ratio:.2%}")

        if (epoch + 1) % args.eval_every == 0:
            # EMA save/restore: swap EMA params in for evaluation, restore after
            if ema_model is not None:
                ema_model.store(model)
                ema_model.copy_to(model)

            model.eval()

            sample_images(model, sampler, args.num_samples, device,
                          output_dir / f'samples_epoch_{epoch+1}.png',
                          in_channels=in_channels, image_size=image_size)
            sample_trajectory(model, sampler, device,
                              output_dir / f'trajectory_epoch_{epoch+1}.png',
                              in_channels=in_channels, image_size=image_size)

            model.train()

            if ema_model is not None:
                ema_model.restore(model)

        if train_loss < best_loss:
            best_loss = train_loss
            save_checkpoint(model, optimizer, scheduler, epoch, args,
                            output_dir / 'best_model.pt', ema_model)
        if (epoch + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, args,
                            output_dir / f'checkpoint_epoch_{epoch+1}.pt', ema_model)

    save_checkpoint(model, optimizer, scheduler, args.epochs - 1, args,
                   output_dir / 'final_model.pt', ema_model)
    print("Training completed!")


if __name__ == '__main__':
    main()
