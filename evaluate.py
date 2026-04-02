"""Evaluate AR-FlowMatching model using FID and other metrics."""

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='whitegrid')
plt.rcParams['font.family'] = ['Comic Sans MS', 'FZKaTong-M19S']
plt.rcParams['axes.unicode_minus'] = False
import torch
import torch.nn.functional as F
import argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore

from arflow import ARFlowMatching, EulerSampler, ConditionalEulerSampler, EMAModel
from arflow import get_timestamped_output_dir, save_args_to_txt
from data_loader import get_mnist_dataloaders, get_dataloaders


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate AR-FlowMatching model')

    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Model checkpoint path')
    parser.add_argument('--data_root', type=str, default=None,
                        help='Data root directory')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=['mnist', 'cifar10', 'celeba'],
                        help='Dataset type')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_samples', type=int, default=3000,
                        help='Number of generated samples')
    parser.add_argument('--num_steps', type=int, default=30,
                        help='Inference steps')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device')
    parser.add_argument('--output_dir', type=str, default='./eval_results',
                        help='Output directory')
    parser.add_argument('--per_class_fid', action='store_true',
                        help='Compute per-class FID')
    parser.add_argument('--cfg_scale', type=float, default=1.5,
                        help='CFG scale (1.0=no CFG)')
    parser.add_argument('--no_ema', action='store_true',
                        help='Disable EMA, use original model')

    return parser.parse_args()


def load_model(checkpoint_path, device, dataset=None, use_ema=True):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    args_dict = checkpoint['args']

    has_ema = checkpoint.get('use_ema', False) and 'ema_state_dict' in checkpoint

    if dataset is None:
        dataset = args_dict.get('dataset', 'mnist')

    if dataset == 'cifar10':
        in_channels = 3
        image_size = 32
        default_data_root = './data/CIFAR10'
    elif dataset == 'celeba':
        in_channels = 3
        image_size = 64
        default_data_root = './data/celeba'
    else:
        in_channels = 1
        image_size = 28
        default_data_root = './data/MNIST'

    num_classes = args_dict.get('num_classes', 0)
    use_cfg = args_dict.get('use_cfg', False)
    cfg_drop_prob = args_dict.get('cfg_drop_prob', 0.0)

    base_channels = args_dict.get('base_channels', 64)
    channel_mult = args_dict.get('channel_mult', (1, 2, 4, 8, 16))
    num_res_blocks = args_dict.get('num_res_blocks', 2)
    spade_hidden_nc = args_dict.get('spade_hidden_nc', 128)
    label_embed_dim = args_dict.get('label_embed_dim', 128)
    time_embed_dim = args_dict.get('time_embed_dim', 256)

    model = ARFlowMatching(
        backbone=args_dict.get('backbone', 'spade_unet'),
        in_channels=in_channels,
        out_channels=in_channels,
        base_channels=base_channels,
        channel_mult=channel_mult,
        num_res_blocks=num_res_blocks,
        spade_hidden_nc=spade_hidden_nc,
        label_embed_dim=label_embed_dim,
        time_embed_dim=time_embed_dim,
        time_start_delay=args_dict.get('time_start_delay', 0.3),
        time_power=args_dict.get('time_power', 2.0),
        time_k=args_dict.get('time_k', 8.0),
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
        ema = EMAModel(model, decay=args_dict.get('ema_decay', 0.9999), device=device)
        ema.load_state_dict(checkpoint['ema_state_dict'])
        ema.copy_to(model)

    return model, in_channels, image_size, dataset, default_data_root


@torch.no_grad()
def generate_for_eval(model, num_samples, batch_size, num_steps, device,
                      labels=None, cfg_scale=1.0, in_channels=1, image_size=28):
    """Generate samples for evaluation."""
    all_samples = []
    num_generated = 0

    use_conditional = model.num_classes > 0 and labels is not None
    use_cfg = use_conditional and cfg_scale > 1.0 and model.use_cfg

    if use_conditional:
        sampler = ConditionalEulerSampler(model, num_steps=num_steps)
    else:
        sampler = EulerSampler(model, num_steps=num_steps)

    with tqdm(total=num_samples, unit="img") as pbar:
        while num_generated < num_samples:
            current_batch = min(batch_size, num_samples - num_generated)

            if use_conditional:
                batch_labels = labels[num_generated:num_generated + current_batch]
                if not isinstance(batch_labels, torch.Tensor):
                    batch_labels = torch.tensor(batch_labels, device=device)
                else:
                    batch_labels = batch_labels.to(device)

                if use_cfg:
                    samples = sampler.sample_with_cfg(
                        shape=(current_batch, in_channels, image_size, image_size),
                        labels=batch_labels,
                        cfg_scale=cfg_scale
                    )
                else:
                    samples = sampler.sample_with_labels(
                        shape=(current_batch, in_channels, image_size, image_size),
                        labels=batch_labels
                    )
            else:
                samples = sampler.sample(shape=(current_batch, in_channels, image_size, image_size))

            all_samples.append(samples)
            num_generated += current_batch
            pbar.update(current_batch)

    return torch.cat(all_samples, dim=0)[:num_samples]


def preprocess_for_inception(images, dataset='mnist'):
    """Preprocess images to Inception v3 input format (3x299x299)."""
    images = (images + 1) / 2
    images = torch.clamp(images, 0, 1)
    if dataset == 'mnist':
        images = images.repeat(1, 3, 1, 1)
    images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    return images


def compute_fid(real_images, generated_images, device, dataset='mnist'):
    """Compute FID score."""
    real_images = preprocess_for_inception(real_images, dataset=dataset)
    generated_images = preprocess_for_inception(generated_images, dataset=dataset)

    real_images = (real_images * 255).clamp(0, 255).to(torch.uint8)
    generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)

    fid = FrechetInceptionDistance(feature=2048).to(device)

    for i in range(0, len(real_images), 100):
        fid.update(real_images[i:i+100], real=True)
    for i in range(0, len(generated_images), 100):
        fid.update(generated_images[i:i+100], real=False)

    return fid.compute().item()


def compute_mnist_fid(real_images, generated_images, device, batch_size=100, dataset='mnist'):
    """Compute FID score with batched processing."""
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)

    def preprocess_tensor(imgs):
        if dataset == 'mnist':
            imgs = imgs.repeat(1, 3, 1, 1)
        imgs = (imgs + 1.0) / 2.0
        imgs = (imgs * 255.0).clamp(0, 255).to(torch.uint8)
        return imgs

    num_real = len(real_images)
    num_real_batches = (num_real + batch_size - 1) // batch_size
    with tqdm(total=num_real_batches, unit="batch") as pbar:
        for i in range(0, num_real, batch_size):
            batch_real = real_images[i : i + batch_size].to(device)
            batch_real = preprocess_tensor(batch_real)
            fid.update(batch_real, real=True)
            pbar.update(1)

    num_gen = len(generated_images)
    num_gen_batches = (num_gen + batch_size - 1) // batch_size
    with tqdm(total=num_gen_batches, unit="batch") as pbar:
        for i in range(0, num_gen, batch_size):
            batch_gen = generated_images[i : i + batch_size].to(device)
            batch_gen = preprocess_tensor(batch_gen)
            fid.update(batch_gen, real=False)
            pbar.update(1)

    fid_score = fid.compute()
    fid.reset()
    return fid_score.item()


def compute_per_class_fid(real_images_by_class, generated_images_by_class, device, dataset='mnist'):
    """Compute FID score for each class."""
    fid_scores = {}

    for class_id in tqdm(range(10)):
        real = real_images_by_class.get(class_id)
        generated = generated_images_by_class.get(class_id)

        if real is None or len(real) == 0:
            continue
        if generated is None or len(generated) == 0:
            continue
        if len(real) < 50 or len(generated) < 50:
            continue

        fid = compute_fid(real, generated, device, dataset=dataset)
        fid_scores[f'class_{class_id}'] = fid

    return fid_scores


def compute_is(generated_images, device):
    """Compute Inception Score."""
    generated_images = preprocess_for_inception(generated_images)
    generated_images = (generated_images * 255).clamp(0, 255).to(torch.uint8)

    inception = InceptionScore(feature=2048).to(device)
    for i in range(0, len(generated_images), 100):
        inception.update(generated_images[i:i+100])

    is_mean, is_std = inception.compute()
    return is_mean.item(), is_std.item()


def compute_pixel_space_metrics(real_images, generated_images):
    """Compute pixel-space metrics (mean/std, sharpness)."""
    real_mean = real_images.mean()
    real_std = real_images.std()
    gen_mean = generated_images.mean()
    gen_std = generated_images.std()

    def compute_sharpness(images):
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32, device=images.device).view(1, 1, 3, 3)
        edges = F.conv2d(images, laplacian_kernel, padding=1)
        return edges.abs().mean().item()

    real_sharpness = compute_sharpness(real_images)
    gen_sharpness = compute_sharpness(generated_images)

    return {
        'real_mean': real_mean.item(),
        'real_std': real_std.item(),
        'gen_mean': gen_mean.item(),
        'gen_std': gen_std.item(),
        'mean_diff': abs(real_mean.item() - gen_mean.item()),
        'std_diff': abs(real_std.item() - gen_std.item()),
        'real_sharpness': real_sharpness,
        'gen_sharpness': gen_sharpness,
    }


def compute_ar_characteristics(model, device, num_samples=100,
                               in_channels=1, image_size=28):
    """Compute AR characteristic metrics via center-edge clarity evolution."""
    sampler = EulerSampler(model, num_steps=20)
    trajectory = sampler.sample_with_intermediate(
        shape=(num_samples, in_channels, image_size, image_size),
        num_intermediate=10
    )

    center_size = image_size // 4
    h_start = (image_size - center_size) // 2
    w_start = (image_size - center_size) // 2
    edge_size = max(2, image_size // 8)

    center_clarity = []
    edge_clarity = []

    for samples in trajectory:
        center = samples[:, :, h_start:h_start+center_size, w_start:w_start+center_size]
        center_clarity.append(center.var().item())

        edge = torch.cat([
            samples[:, :, :edge_size, :].reshape(-1),
            samples[:, :, -edge_size:, :].reshape(-1),
            samples[:, :, :, :edge_size].reshape(-1),
            samples[:, :, :, -edge_size:].reshape(-1)
        ])
        edge_clarity.append(edge.var().item())

    center_leads = sum(c > e for c, e in zip(center_clarity, edge_clarity))
    ar_score = center_leads / len(center_clarity)

    return {
        'ar_order_score': ar_score,
        'center_clarity_final': center_clarity[-1],
        'edge_clarity_final': edge_clarity[-1],
        'center_clarity_trajectory': center_clarity,
        'edge_clarity_trajectory': edge_clarity
    }


def evaluate(model, data_root, num_samples, batch_size, num_steps, device, output_dir,
             per_class_fid=False, cfg_scale=1.0, dataset='mnist',
             in_channels=1, image_size=28):
    """Complete evaluation pipeline."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    supports_conditional = model.num_classes > 0

    if dataset == 'celeba' and per_class_fid:
        print("Note: CelebA uses multi-label attributes, disabling per-class FID analysis")
        per_class_fid = False

    _, test_loader = get_dataloaders(
        dataset=dataset,
        root=data_root,
        batch_size=batch_size,
        num_workers=4,
        labeled=True
    )

    real_images = []
    if per_class_fid and supports_conditional:
        num_classes = 10 if dataset != 'celeba' else 40
        real_images_by_class = {i: [] for i in range(num_classes)}
        for images, labels in test_loader:
            images = images.to(device)
            if dataset == 'celeba':
                for img, attr_vector in zip(images, labels):
                    active_attrs = (attr_vector == 1).nonzero(as_tuple=True)[0]
                    if len(active_attrs) == 1:
                        class_id = active_attrs[0].item()
                        real_images_by_class[class_id].append(img)
            else:
                for img, label in zip(images, labels):
                    class_id = label.item()
                    real_images_by_class[class_id].append(img)
            real_images.append(images.cpu())
            if sum(len(v) for v in real_images_by_class.values()) >= num_samples * 2:
                break
        for class_id in real_images_by_class:
            if len(real_images_by_class[class_id]) > 0:
                real_images_by_class[class_id] = torch.stack(real_images_by_class[class_id])
    else:
        for images, _ in test_loader:
            real_images.append(images)
            if len(real_images) * batch_size >= num_samples:
                break

    real_images_all = torch.cat(real_images, dim=0)[:num_samples].to(device)

    if supports_conditional:
        if per_class_fid:
            generated_images_by_class = {}
            num_classes = 10 if dataset != 'celeba' else 40
            samples_per_class = num_samples // num_classes

            for class_id in range(num_classes):
                if dataset == 'celeba':
                    attr_vector = torch.zeros(samples_per_class, 40, dtype=torch.long)
                    attr_vector[:, class_id] = 1
                    labels = attr_vector
                else:
                    labels = [class_id] * samples_per_class

                generated = generate_for_eval(
                    model, samples_per_class, batch_size, num_steps, device,
                    labels=labels, cfg_scale=cfg_scale,
                    in_channels=in_channels, image_size=image_size
                )
                generated_images_by_class[class_id] = generated

            generated_images_all = torch.cat(list(generated_images_by_class.values()), dim=0)
        else:
            if dataset == 'celeba':
                labels = torch.randint(0, 2, (num_samples, 40), dtype=torch.long)
            else:
                labels = torch.randint(0, 10, (num_samples,)).tolist()
            generated_images_all = generate_for_eval(
                model, num_samples, batch_size, num_steps, device,
                labels=labels, cfg_scale=cfg_scale,
                in_channels=in_channels, image_size=image_size
            )
    else:
        generated_images_all = generate_for_eval(
            model, num_samples, batch_size, num_steps, device,
            in_channels=in_channels, image_size=image_size
        )

    # Overall FID
    fid_score = compute_mnist_fid(real_images_all, generated_images_all, device, dataset=dataset)
    results['FID'] = fid_score
    print(f"\n{'='*50}")
    print(f"Overall FID score: {fid_score:.4f}")
    print(f"{'='*50}\n")

    # Per-class FID
    if per_class_fid and supports_conditional:
        per_class_results = compute_per_class_fid(
            real_images_by_class, generated_images_by_class, device, dataset=dataset
        )
        results.update(per_class_results)

        if len(per_class_results) > 0:
            avg_fid = sum(per_class_results.values()) / len(per_class_results)
            results['FID_per_class_avg'] = avg_fid
            print(f"\n{'='*50}")
            print(f"Average per-class FID: {avg_fid:.4f}")
            print(f"{'='*50}\n")

    # AR characteristic metrics
    ar_metrics = compute_ar_characteristics(model, device, num_samples=100,
                                            in_channels=in_channels, image_size=image_size)
    results.update(ar_metrics)
    print(f"{'='*50}")
    print("AR characteristic metrics:")
    print(f"  AR order score: {ar_metrics['ar_order_score']:.2%}")
    print(f"  Final center clarity: {ar_metrics['center_clarity_final']:.4f}")
    print(f"  Final edge clarity: {ar_metrics['edge_clarity_final']:.4f}")
    print(f"{'='*50}\n")

    # Save results
    import json
    with open(output_dir / 'metrics.json', 'w') as f:
        serializable_results = {}
        for k, v in results.items():
            if isinstance(v, (np.integer, np.floating)):
                serializable_results[k] = float(v)
            elif isinstance(v, list):
                serializable_results[k] = [float(x) for x in v]
            else:
                serializable_results[k] = v
        json.dump(serializable_results, f, indent=2)

    return results


def main():
    args = parse_args()

    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    print(f"Using device: {device}")

    use_ema = not args.no_ema
    model, in_channels, image_size, dataset, default_data_root = load_model(
        args.checkpoint, device, args.dataset, use_ema
    )

    checkpoint_name = Path(args.checkpoint).stem
    output_dir = get_timestamped_output_dir(args.output_dir, f"eval_{checkpoint_name}")

    save_args_to_txt(args, output_dir / 'args.txt', script_name='ARFM-Evaluate')

    data_root = args.data_root if args.data_root else default_data_root

    evaluate(
        model=model,
        data_root=data_root,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        num_steps=args.num_steps,
        device=device,
        output_dir=str(output_dir),
        per_class_fid=args.per_class_fid,
        cfg_scale=args.cfg_scale,
        dataset=dataset,
        in_channels=in_channels,
        image_size=image_size
    )


if __name__ == '__main__':
    main()
