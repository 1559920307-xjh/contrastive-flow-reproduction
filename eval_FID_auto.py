import os, torch, torchvision, gc
from torch.utils.data import DataLoader
from torchvision import transforms, utils
from tqdm import tqdm
from diffusers.models import AutoencoderKL
from models.sit import SiT_models
from samplers import euler_sampler
from torchmetrics.image.fid import FrechetInceptionDistance
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output_dir', default='./fid_eval_auto')
parser.add_argument('--batch_size', type=int, default=10)
parser.add_argument('--num_samples', type=int, default=2000)
parser.add_argument('--cfg_scale', type=float, default=4.0)
parser.add_argument('--real_data_dir', default='/root/autodl-tmp/imagenette2-320-preprocessed/images')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CLASSES = 10
LATENT_SIZE = 32  # 256 // 8

# 1. 真实图像统计
print("Processing real images ...")
transform = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(256), transforms.ToTensor()])
dataset = torchvision.datasets.ImageFolder(args.real_data_dir, transform=transform)
real_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
fid = FrechetInceptionDistance(normalize=True).to(device)
for batch in tqdm(real_loader, desc='Real stats'):
    fid.update(batch[0].to(device), real=True)

# 2. 加载检查点并推断模型结构
print(f"Loading checkpoint: {args.checkpoint}")
ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
ema_state = ckpt['ema']

# 推断 z_dims（从 projectors 键）
z_dims = []
decoder_hidden_size = None
for key in ema_state.keys():
    if key.startswith('projectors.') and key.endswith('.4.weight'):
        # projectors.i.4.weight 的形状是 (z_dim, projector_dim) 或类似，我们取输出维度
        z_dim = ema_state[key].shape[0]
        z_dims.append(z_dim)
    if key == 'final_layer.adaLN_modulation.1.weight':
        # 该线性层输入维度即为解码器隐藏层维度
        decoder_hidden_size = ema_state[key].shape[1]

# 如果没有投影器，z_dims 保持空
if not z_dims:
    z_dims = []
# 如果没有检测到 decoder_hidden_size，设为 None，模型会自动调整
print(f"Detected z_dims: {z_dims}, decoder_hidden_size: {decoder_hidden_size}")

# 3. 构建模型
model = SiT_models['SiT-S/2'](
    input_size=LATENT_SIZE,
    num_classes=NUM_CLASSES,
    use_cfg=True,
    z_dims=z_dims,
    encoder_depth=8,
    fused_attn=True,
    qk_norm=False,
    decoder_hidden_size=decoder_hidden_size
).to(device)

model.load_state_dict(ema_state)
model.eval()

# 4. VAE
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()

# 5. 生成图像并计算 FID
all_grid_images = []
grid_size = min(64, args.num_samples)
print(f"Generating {args.num_samples} samples with cfg_scale={args.cfg_scale} ...")
with torch.no_grad():
    for start in tqdm(range(0, args.num_samples, args.batch_size)):
        cur_bs = min(args.batch_size, args.num_samples - start)
        ys = torch.randint(0, NUM_CLASSES, (cur_bs,), device=device)
        xT = torch.randn((cur_bs, 4, LATENT_SIZE, LATENT_SIZE), device=device)
        samples = euler_sampler(
            model, xT, ys,
            num_steps=50, cfg_scale=args.cfg_scale, path_type="linear"
        )["samples"].to(torch.float32)
        samples = vae.decode((samples - 0.18215) / 0.18215).sample
        samples = (samples + 1) / 2
        fid.update(samples, real=False)

        if len(all_grid_images) < grid_size:
            remain = grid_size - len(all_grid_images)
            for i in range(min(cur_bs, remain)):
                all_grid_images.append(samples[i].cpu())

fid_value = fid.compute().item()
print(f"\nFID = {fid_value:.2f}")

# 保存结果
with open(os.path.join(args.output_dir, 'fid.txt'), 'w') as f:
    f.write(f"FID: {fid_value:.4f}\n")

if all_grid_images:
    grid = utils.make_grid(torch.stack(all_grid_images), nrow=8, padding=2)
    utils.save_image(grid, os.path.join(args.output_dir, 'grid.png'))
print(f"Results saved in {args.output_dir}")