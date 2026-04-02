"""Generate images using trained model."""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.rcParams['font.family'] = ['Comic Sans MS', 'FZKaTong-M19S']
plt.rcParams['axes.unicode_minus'] = False
import torch
import argparse
import numpy as np
from pathlib import Path

from arflow import ARFlowMatching, EulerSampler, EMAModel
from arflow import get_timestamped_output_dir, save_args_to_txt
from arflow.solver import compute_ar_order_metric, ConditionalEulerSampler

CIFAR10_CLASSES = [
    'Airplane(0)', 'Automobile(1)', 'Bird(2)', 'Cat(3)', 'Deer(4)',
    'Dog(5)', 'Frog(6)', 'Horse(7)', 'Ship(8)', 'Truck(9)'
]


def parse_args():
    parser = argparse.ArgumentParser(description='AR-FlowMatching sampling')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--output_dir', type=str, default='./samples',
                        help='Output directory')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='Number of generated samples')
    parser.add_argument('--num_steps', type=int, default=30,
                        help='Inference steps')
    parser.add_argument('--batch_size', type=int, default=50,
                        help='Batch size')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (auto, cpu, cuda)')
    parser.add_argument('--save_trajectory', action='store_true',
                        help='Save generation trajectory')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['mnist', 'cifar10', 'celeba'],
                        help='Dataset type')
    parser.add_argument('--label', type=int, default=None,
                        help='Class to generate (0-9)')
    parser.add_argument('--attr', type=str, default=None,
                        help='CelebA attribute indices (comma-separated)')
    parser.add_argument('--cfg_scale', type=float, default=1.5,
                        help='CFG scale (1.0=no CFG)')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA, use original model')

    return parser.parse_args()


def load_model(checkpoint_path, device, dataset=None, use_ema=True):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args = checkpoint['args']

    has_ema = checkpoint.get('use_ema', False) and 'ema_state_dict' in checkpoint

    if dataset is None:
        dataset = args.get('dataset', 'mnist')

    if dataset == 'cifar10':
        in_channels = 3
        image_size = 32
    elif dataset == 'celeba':
        in_channels = 3
        image_size = 64
    else:
        in_channels = 1
        image_size = 28

    num_classes = args.get('num_classes', 0)
    use_cfg = args.get('use_cfg', False)
    cfg_drop_prob = args.get('cfg_drop_prob', 0.0)

    base_channels = args.get('base_channels', 64)
    channel_mult = args.get('channel_mult', (1, 2, 4, 8, 16))
    num_res_blocks = args.get('num_res_blocks', 2)
    spade_hidden_nc = args.get('spade_hidden_nc', 128)
    label_embed_dim = args.get('label_embed_dim', 128)
    time_embed_dim = args.get('time_embed_dim', 256)

    model = ARFlowMatching(
        backbone=args.get('backbone', 'spade_unet'),
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        spade_hidden_nc=spade_hidden_nc,
        label_embed_dim=label_embed_dim,
        time_embed_dim=time_embed_dim,
        time_start_delay=args.get('time_start_delay', 0.3),
        time_power=args.get('time_power', 2.0),
        time_k=args.get('time_k', 8.0),
        num_classes=num_classes,
        use_cfg=use_cfg,
        cfg_drop_prob=cfg_drop_prob
    ).to(device)

    state_dict = checkpoint['model_state_dict']

    # Remove compatibility cache keys
    for key in ['network.pos_embed_cache', 'network.pos_embed_grid_size',
                'pos_embed_cache', 'pos_embed_grid_size']:
        state_dict.pop(key, None)

    model.load_state_dict(state_dict, strict=True)
    model.eval()

    if has_ema and use_ema:
        ema = EMAModel(model, decay=args.get('ema_decay', 0.9999), device=device)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model)

    return model, in_channels, image_size, dataset


@torch.no_grad()
def generate_samples(model, sampler, num_samples, batch_size, device, save_trajectory=False,
                     labels=None, attr_labels=None, cfg_scale=1.0, in_channels=1, image_size=28):
    """Generate samples in batches."""
    all_samples = []
    num_generated = 0

    use_conditional = (labels is not None) or (attr_labels is not None)
    use_cfg = use_conditional and cfg_scale > 1.0 and hasattr(sampler, 'sample_with_cfg')

    while num_generated < num_samples:
        current_batch = min(batch_size, num_samples - num_generated)

        batch_labels = None
        if use_conditional:
            if labels is not None:
                batch_labels = labels[num_generated:num_generated + current_batch]
                batch_labels = torch.tensor(batch_labels, device=device)
            elif attr_labels is not None:
                batch_labels = attr_labels[num_generated:num_generated + current_batch]
                if not isinstance(batch_labels, torch.Tensor):
                    batch_labels = torch.tensor(batch_labels, device=device)
                else:
                    batch_labels = batch_labels.to(device)

        if save_trajectory and num_generated == 0:
            if use_cfg:
                samples, trajectory = sampler.sample_with_cfg(
                    shape=(current_batch, in_channels, image_size, image_size),
                    labels=batch_labels, cfg_scale=cfg_scale, return_trajectory=True
                )
            elif use_conditional:
                samples, trajectory = sampler.sample_with_labels(
                    shape=(current_batch, in_channels, image_size, image_size),
                    labels=batch_labels, return_trajectory=True
                )
            else:
                trajectory = sampler.sample_with_intermediate(
                    shape=(current_batch, in_channels, image_size, image_size),
                    num_intermediate=10
                )
                samples = trajectory[-1]
            all_samples.append(samples)
            trajectory_data = trajectory
        else:
            if use_cfg:
                samples = sampler.sample_with_cfg(
                    shape=(current_batch, in_channels, image_size, image_size),
                    labels=batch_labels, cfg_scale=cfg_scale
                )
            elif use_conditional:
                samples = sampler.sample_with_labels(
                    shape=(current_batch, in_channels, image_size, image_size),
                    labels=batch_labels
                )
            else:
                samples = sampler.sample(
                    shape=(current_batch, in_channels, image_size, image_size)
                )
            all_samples.append(samples)

        num_generated += current_batch

    final_samples = torch.cat(all_samples, dim=0)[:num_samples]

    if save_trajectory:
        return final_samples, trajectory_data
    return final_samples, None


def save_samples_grid(samples, save_path, title='Generated Samples', in_channels=1):
    """Save sample grid."""
    num_samples = samples.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    fig, axes = plt.subplots(grid_size, grid_size, figsize=(grid_size, grid_size))
    axes = axes.flatten()

    for i in range(grid_size * grid_size):
        ax = axes[i]
        if i < num_samples:
            if in_channels == 1:
                img = samples[i].squeeze().cpu().numpy()
                ax.imshow(img, cmap='gray_r', vmin=-1, vmax=1)
            else:
                img = samples[i].permute(1, 2, 0).cpu().numpy()
                img = (img + 1) / 2
                ax.imshow(img.clip(0, 1))
        ax.axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_trajectory(trajectory, save_path, in_channels=1):
    """Save generation trajectory visualization."""
    num_steps = len(trajectory)
    fig, axes = plt.subplots(2, (num_steps + 1) // 2, figsize=((num_steps + 1) // 2 * 3, 6))
    axes = axes.flatten()

    for i in range(num_steps):
        ax = axes[i]
        if in_channels == 1:
            img = trajectory[i][0, 0].cpu().numpy()
            ax.imshow(img, cmap='gray_r', vmin=-1, vmax=1)
        else:
            img = trajectory[i][0].permute(1, 2, 0).cpu().numpy()
            img = (img + 1) / 2
            ax.imshow(img.clip(0, 1))
        ax.set_title(f'Step {i}')
        ax.axis('off')

    plt.suptitle('Generation Trajectory (Noise to Clear)')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def compute_ar_metrics(trajectory):
    """Compute AR characteristic metrics."""
    return compute_ar_order_metric(trajectory)


def visualize_time_field_evolution(model, save_path, image_size=28):
    """Visualize time field evolution."""
    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    taus = [0.0, 0.25, 0.5, 0.75, 1.0]
    device = next(model.parameters()).device

    for i, tau in enumerate(taus):
        ax = axes[i]
        t_map = model.time_field.get_time_map(
            torch.tensor([tau]), (1, 1, image_size, image_size)
        )
        img = t_map[0, 0].cpu().numpy()
        im = ax.imshow(img, cmap='RdBu_r', vmin=0, vmax=1)
        ax.set_title(f'\u03c4={tau}')
        ax.axis('off')

    plt.suptitle('Time Field Evolution (Red=Clear, Blue=Noise)')
    plt.colorbar(im, ax=axes, fraction=0.02)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    checkpoint_name = Path(args.checkpoint).stem
    output_dir = get_timestamped_output_dir(args.output_dir, f"sample_{checkpoint_name}")

    save_args_to_txt(args, output_dir / 'args.txt', script_name='ARFM-Sample')

    use_ema = not args.no_ema
    model, in_channels, image_size, dataset = load_model(args.checkpoint, device, args.dataset, use_ema)

    labels = None
    attr_labels = None

    if dataset == 'celeba':
        if args.attr is not None:
            attr_indices = [int(x.strip()) for x in args.attr.split(',')]
            attr_vector = torch.zeros(args.num_samples, 40, dtype=torch.long)
            for idx in attr_indices:
                if 0 <= idx < 40:
                    attr_vector[:, idx] = 1
            attr_labels = attr_vector
    else:
        if args.label is not None:
            if not (0 <= args.label <= 9):
                raise ValueError(f"Label must be between 0-9, current value: {args.label}")
            labels = [args.label] * args.num_samples

    supports_conditional = model.num_classes > 0
    if (args.label is not None or args.attr is not None) and not supports_conditional:
        print("Warning: Model does not support conditional generation (num_classes=0), ignoring condition")
        labels = None
        attr_labels = None

    if supports_conditional and args.cfg_scale > 1.0 and model.use_cfg:
        sampler = ConditionalEulerSampler(model, num_steps=args.num_steps)
    else:
        sampler = EulerSampler(model, num_steps=args.num_steps)

    visualize_time_field_evolution(model, output_dir / 'time_field.png', image_size=image_size)

    samples, trajectory = generate_samples(
        model, sampler, args.num_samples, args.batch_size, device, args.save_trajectory,
        labels=labels, attr_labels=attr_labels, cfg_scale=args.cfg_scale,
        in_channels=in_channels, image_size=image_size
    )

    if args.label is not None:
        sample_filename = f'samples_label_{args.label}.png'
        trajectory_filename = f'trajectory_label_{args.label}.png'
        trajectory_single_filename = f'trajectory_single_label_{args.label}.png'
        title = f'Generated Samples ({args.num_steps} steps, class={args.label}'
    else:
        sample_filename = 'samples.png'
        trajectory_filename = 'trajectory.png'
        trajectory_single_filename = 'trajectory_single.png'
        title = f'Generated Samples ({args.num_steps} steps'

    if args.cfg_scale > 1.0 and supports_conditional and model.use_cfg:
        title += f', CFG={args.cfg_scale}'
    title += ')'

    save_samples_grid(samples, output_dir / sample_filename, title=title, in_channels=in_channels)

    if args.save_trajectory and trajectory is not None:
        save_trajectory(trajectory, output_dir / trajectory_filename, in_channels=in_channels)

    if args.save_trajectory and trajectory is not None:
        fig, axes = plt.subplots(1, len(trajectory), figsize=(len(trajectory) * 2, 2))
        for i, sample in enumerate(trajectory):
            if in_channels == 1:
                img = sample[0, 0].cpu().numpy()
                axes[i].imshow(img, cmap='gray_r', vmin=-1, vmax=1)
            else:
                img = sample[0].permute(1, 2, 0).cpu().numpy()
                img = (img + 1) / 2
                axes[i].imshow(img.clip(0, 1))
            axes[i].set_title(f'Step {i}')
            axes[i].axis('off')
        plt.tight_layout()
        plt.savefig(output_dir / trajectory_single_filename, dpi=150, bbox_inches='tight')
        plt.close()


if __name__ == '__main__':
    main()
