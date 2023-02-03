import torch
from torchvision import transforms as T

import os
import argparse

## customized
from net import DDPM, ContextUnet


def get_args():
    parser = argparse.ArgumentParser(description="Generate 100 samples for 10 labels.")
    parser.add_argument(
        "--model", "-m", type=str, default=None, help="Model path to load"
    )
    parser.add_argument(
        "--output_dir", "-o", type=str, default=None, help="Model path to load"
    )
    return parser.parse_args()


def save_img(imgs, label=None, name=None, save_dir=None):
    reverse_tfm = T.Compose(
        [
            T.Resize(28),
            T.Lambda(lambda t: (t + 1) / 2),  ## unnormalize to 0~1
            T.Lambda(lambda t: t * 255),
            T.Lambda(lambda t: t.to(torch.uint8)),
            T.ToPILImage(),
        ]
    )

    if len(imgs.shape) == 4:
        ## take the first image of the batch
        for idx in range(imgs.size(0)):
            img = imgs[idx, :, :, :]
            img = reverse_tfm(img)
            name = idx + 1
            if name < 10:
                name = f"00{name}"
            elif name < 100:
                name = f"0{name}"
            else:
                name = str(name)
            img.save(os.path.join(save_dir, f"{label}_{name}.png"))
            print(f"image saved as {label}_{name}.png")
    else:
        imgs = reverse_tfm(imgs)
        imgs.save(os.path.join(save_dir, f"{label}_{name}.png"))
        print(f"image saved as {label}_{name}.png")


def test(
    model_path,
    device,
    output_dir,
    n_T=300,
    img_size=28,
    n_feat=256,
    num_classes=10,
    guide_w=2.0,
):

    ddpm = DDPM(
        nn_model=ContextUnet(in_channels=3, n_feat=n_feat, num_classes=num_classes),
        betas=(1e-4, 0.02),
        n_T=n_T,
        device=device,
        drop_prob=0.1,
    )
    state = torch.load(model_path, device)
    ddpm.load_state_dict(state["ddpm"])
    ddpm.to(device)

    ddpm.eval()
    with torch.no_grad():
        for i in range(10):
            print(f"sampling label {i}")
            x_gen = ddpm.eval_sample(
                100, (3, img_size, img_size), label=i, guide_w=guide_w
            )
            save_img(x_gen, label=i, save_dir=output_dir)


if __name__ == "__main__":
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"using device {device}")

    test(args.model, device, args.output_dir)
