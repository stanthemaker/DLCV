import torch
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torch import nn
from torchvision import transforms as T
import torch.nn.functional as F
from net import DDPM, ContextUnet
from torchvision import utils
import matplotlib.pyplot as plt
import subprocess
from tqdm.auto import tqdm
import numpy as np

MEAN = [0.5, 0.5, 0.5]  # [1, 1, 1]
STD = [0.5, 0.5, 0.5]  # [1, 1, 1]


def load_checkpoint(checkpoint_path, model):
    state = torch.load(checkpoint_path, map_location="cuda")
    model.load_state_dict(state["state_dict"])
    print("model loaded from %s" % checkpoint_path)


class DATA(Dataset):
    def __init__(self, path):
        self.img_dir = path

        self.data = []
        self.labels = []
        for filename in os.listdir(self.img_dir):
            self.data.append(os.path.join(self.img_dir, filename))
            self.labels.append(int(filename[0]))

        self.transform = T.Compose(
            [
                T.ToTensor(),  # (H,W,C)->(C,H,W), [0,255]->[0, 1.0] RGB->RGB
                T.Normalize(MEAN, STD),
            ]
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        img = Image.open(img_path).convert("RGB")

        return self.transform(img), self.labels[idx]


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def check_acc(output_dir):
    net = Classifier()
    path = "../input/hw2-data/Classifier.pth"
    load_checkpoint(path, net)

    # GPU enable
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # print('Device used:', device)
    net = net.to(device)

    data_loader = torch.utils.data.DataLoader(
        DATA(output_dir), batch_size=32, num_workers=0, shuffle=False
    )

    correct = 0
    total = 0
    net.eval()
    # print('===> start evaluation ...')
    with torch.no_grad():
        for idx, (imgs, labels) in enumerate(data_loader):
            imgs, labels = imgs.to(device), labels.to(device)
            output = net(imgs)
            _, pred = torch.max(output, 1)
            correct += (pred == labels).detach().sum().item()
            total += len(pred)
    acc = float(correct) / total
    print("acc = {} (correct/total = {}/{})".format(acc, correct, total))

    return acc


class Trainer(object):
    def __init__(
        self,
        img_size,
        dataset,
        num_epochs,
        batch_size,
        n_T,
        device,
        output_dir,
        sample_dir,
        ckpt_dir,
        n_feat=256,
        num_classes=10,
        lr=1e-4,
        save_model=True,
        guide_weights=[0.0, 0.5, 2.0],  ## different strength of generative guidance
        #         classifier_path = "../input/hw2-data/Classifier.pth"
    ):

        super(Trainer, self).__init__()

        self.img_size = img_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.n_T = n_T
        self.device = device
        self.output_dir = output_dir
        self.sample_dir = sample_dir
        self.ckpt_dir = ckpt_dir
        self.num_classes = num_classes
        self.lr = lr
        self.guide_weights = guide_weights

        #         betas = linear_beta_schedule(n_T)
        self.ddpm = DDPM(
            nn_model=ContextUnet(
                in_channels=3, n_feat=n_feat, num_classes=self.num_classes
            ),
            betas=(1e-4, 0.02),
            n_T=self.n_T,
            device=self.device,
            drop_prob=0.1,
        )
        self.ddpm.to(self.device)

        self.dataset = dataset
        self.dataloader = DataLoader(
            self.dataset, batch_size=self.batch_size, shuffle=True, num_workers=0
        )

        self.optimizer = torch.optim.Adam(
            self.ddpm.parameters(), lr=self.lr, betas=(0.9, 0.99)
        )

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: 1 - epoch / self.num_epochs
        )
        #         self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='max', factor = 0.5, patience=3, min_lr = 1e-5)

        self.epoch = 0

    def save_checkpoint(self):
        data = {
            "ddpm": self.ddpm.state_dict(),
            "opt": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "epoch": self.epoch,
        }

        torch.save(data, os.path.join(self.ckpt_dir, "2130_checkpoint.pt"))

    def report(self, num_samples, guide_w, save_dir):
        reverse_tfm = T.Compose(
            [
                T.Resize(28),
                T.Lambda(lambda t: (t + 1) / 2),  ## unnormalize to 0~1
                T.Lambda(lambda t: t * 255),
                T.Lambda(lambda t: t.to(torch.uint8)),
            ]
        )
        to_img = T.ToPILImage()
        with torch.no_grad():
            imgs = []
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            for sample_time in range(num_samples):
                print(f"sampleing {sample_time}")
                if sample_time == 0:
                    x_gen, x_store = self.ddpm.sample(
                        10, (3, self.img_size, self.img_size), guide_w=guide_w
                    )
                else:
                    x_gen, _ = self.ddpm.sample(
                        10, (3, self.img_size, self.img_size), guide_w=guide_w
                    )
                x_gen = reverse_tfm(x_gen)
                x_gen = x_gen.detach().cpu()
                imgs.append(x_gen)

            imgs = torch.cat(imgs)
            imgs = utils.make_grid(imgs, nrow=10)
            # plt.imshow(imgs.permute(1, 2, 0))
            # plt.show()

            imgs = to_img(imgs)
            imgs.save(os.path.join(save_dir, "report10x10.png"))

            ## different time steps of the first image
            plt.figure(figsize=(15, 15))
            plt.axis("off")
            b_size = len(x_store)
            x_store = torch.cat(x_store)
            x_store = x_store.view(b_size, 3, 28, 28)
            x_store = reverse_tfm(x_store)
            x_store = utils.make_grid(x_store, nrow=10)
            # plt.imshow(x_store.permute(1, 2, 0))
            # plt.show()
            x_store = to_img(x_store)
            x_store.save(os.path.join(save_dir, "report_first.png"))

    def load_checkpoint(self, pt_path):
        data = torch.load(pt_path, self.device)
        self.ddpm.load_state_dict(data["ddpm"])
        self.optimizer.load_state_dict(data["opt"])
        self.scheduler.load_state_dict(data["scheduler"])
        self.epoch = data["epoch"]

    def check_acc(self, script):
        subprocess.run(script)

    def train(self):

        best_acc = 0

        for epoch in range(self.num_epochs):
            self.epoch += 1
            sample_counter = 0

            ## train
            self.ddpm.train()
            train_loss = []
            for batch in tqdm(self.dataloader):

                self.optimizer.zero_grad()
                imgs, c = batch
                imgs = imgs.float()

                loss = self.ddpm(imgs.to(self.device), c.to(self.device))
                loss.backward()
                train_loss.append(loss.item())

                self.optimizer.step()

            train_loss = np.sum(train_loss) / self.batch_size
            print(f"[Epoch {epoch} |Train loss: {train_loss:.4f}]")

            ## sample
            self.ddpm.eval()

            with torch.no_grad():
                ## sample noise
                num_samples = 10

                acc = 0
                avg_acc = 0
                for w_idx, w in enumerate(self.guide_weights):
                    plt.figure(figsize=(15, 15))
                    plt.axis("off")
                    x_gen, _ = self.ddpm.sample(
                        num_samples, (3, self.img_size, self.img_size), guide_w=w
                    )
                    gen_samples = []
                    #                     for k in range(n_classes):
                    #                         for j in range(int(num_samples/self.num_classes)):
                    #                             try:
                    #                                 idx = torch.squeeze((c == k).nonzero())[j]
                    #                             except:
                    #                                 idx = 0
                    #                             x_real[k+(j*n_classes)] = imgs[idx]
                    #                     x_all = torch.cat([x_gen, x_real])
                    #                     x_all = torchvision.utils.make_grid(x_all, nrow = self.num_classes)

                    for idx in range(x_gen.size(0)):
                        img = x_gen[idx, :, :, :]
                        plt.subplot(1, num_samples, idx + 1)
                        # img = show_img(img, idx, self.output_dir)
                        gen_samples.append(img)

                    # plt.show()

                    acc = check_acc(self.output_dir)
                    avg_acc += acc
                    if acc > 0.8:

                        sample_counter += 1
                        if sample_counter < 10:
                            name = f"00{sample_counter}"
                        elif sample_counter < 100:
                            name = f"0{sample_counter}"
                        else:
                            name = str(sample_counter)

                        for idx, sample in enumerate(gen_samples):
                            sample.save(
                                os.path.join(self.sample_dir, f"{idx}_{name}.png")
                            )

            self.scheduler.step(acc)
            avg_acc /= 3

            if avg_acc > best_acc:
                best_acc = avg_acc
                self.save_checkpoint()
                print(f"saving model at epoch {epoch}")
