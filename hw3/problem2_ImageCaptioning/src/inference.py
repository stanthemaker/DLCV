import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from torch import Tensor
from model import Transformer
from trainer import seed_everything
from dataset import InfDataset, Infcollate_padd
from tokenizers import Tokenizer
import numpy as np
from torch.nn import functional as F
import torchvision.transforms.functional as TF
import os
import argparse
import json

# from torchsummary import summary
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tok",
        "-t",
        type=str,
        default="/home/stan/hw3-stanthemaker/hw3_data/caption_tokenizer.json",
        help="tokenizer path to load",
    )
    parser.add_argument(
        "--model",
        "-m",
        type=str,
        default="/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/ckpt/1116-1743_test.pth",
        # default="/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/ckpt/1116-1832_beam_k_5.pth",
        help="Model path to load",
    )
    parser.add_argument("--cuda", "-c", type=str, default="cuda", help="choose a cuda")
    parser.add_argument(
        "--json",
        "-j",
        type=str,
        default="/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/output/pred.json",
        help="json path to output",
    )
    parser.add_argument(
        "--image_dir",
        "-i",
        type=str,
        default="/home/stan/hw3-stanthemaker/hw3_data/p2_data/images/val",
        help="image dir",
    )
    return parser.parse_args()


config = {
    "seed": 1314520,
    "max_len": 55,
    "batch_size": 32,
    "num_epochs": 100,
    "val_interval": 3,
    "pad_id": 0,
    "vocab_size": 18202,
    "d_model": 768,  # 768
    "dec_ff_dim": 2048,
    "dec_n_layers": 6,
    "dec_n_heads": 12,  # 12 # 8
    "dropout": 0.1,
    "max_norm": 0.1,
}

args = get_args()
device = torch.device(args.cuda if (torch.cuda.is_available()) else "cpu")
pred_file = args.json
print(f"using device {device}")

SEED = config["seed"]
seed_everything(SEED)

print("Builing datasets...")
test_set = InfDataset(
    args.image_dir,
    384,
)
test_loader = DataLoader(
    test_set,
    collate_fn=Infcollate_padd(config["max_len"], config["pad_id"]),
    batch_size=1,
    pin_memory=True,
    shuffle=True,
    num_workers=0,
)

print("Loading transformer...")

transformer = Transformer(
    config["vocab_size"],
    config["d_model"],
    config["dec_ff_dim"],
    config["dec_n_layers"],
    config["dec_n_heads"],
    config["max_len"] - 1,
).to(device)

state = torch.load(args.model, map_location=device)
transformer_state = state["models"]
transformer.load_state_dict(transformer_state)
seq_mask = []


def create_caption_and_mask(start_token, max_length, device):
    caption_template = torch.zeros((1, max_length), dtype=torch.long, device=device)
    mask_template = torch.ones((1, max_length), dtype=torch.bool)

    caption_template[:, 0] = start_token
    mask_template[:, 0] = False

    return caption_template, mask_template


EOS = 3
BOS = 2
start_token = torch.tensor(BOS)
caption, _ = create_caption_and_mask(start_token, config["max_len"], device)
pred_caps = {}
print("Inferencing...")
transformer.eval()
tokenizer = Tokenizer.from_file(args.tok)

pb = tqdm(test_loader, leave=False, total=len(test_loader))
pb.unit = "step"
for step, (imgs, names) in enumerate(pb):
    imgs: Tensor  # images [B, 3, 224, 224]
    imgs = imgs.to(device)
    max_len = 55
    imgs: Tensor  # images [1, 3, 256, 256]

    k = 5
    start = torch.full(
        size=(1, 1),
        fill_value=BOS,
        dtype=torch.long,
        device=device,
    )

    with torch.no_grad():
        imgs_enc = transformer.encoder(imgs)
        imgs_enc = transformer.match_size(imgs_enc)
        logits, attns = transformer.decoder(start, imgs_enc.permute(1, 0, 2))
        logits = transformer.predictor(logits).permute(1, 0, 2).contiguous()

        logits: Tensor  # [k=1, 1, vsc]
        attns: Tensor  # [ln, k=1, hn, S=1, is]

        log_prob = F.log_softmax(logits, dim=2)  # [1, 1, k]
        log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)  # [1, 1, k]
        current_preds = torch.cat([start.expand(k, 1), indxs_topk.view(k, 1)], dim=1)

    seq_preds = []
    seq_log_probs = []
    seq_attns = []
    last_word_id = 0
    num_repeat_word = 0
    while k > 0 and current_preds.nelement():
        with torch.no_grad():
            imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
            # print('size of current_pred', current_preds.size())
            # print('size of imgs_expand', imgs_expand.size())
            # [k, is, ie]
            logits, attns = transformer.decoder(
                current_preds, imgs_expand.permute(1, 0, 2)
            )
            logits = transformer.predictor(logits).permute(1, 0, 2).contiguous()
            # logits: [k, S, vsc]
            # attns: # [ln, k, hn, S, is]
            # get last layer, mean across transformer heads
            # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
            # # [k, S, h, w]
            # attns = attns[:, -1].view(k, 1, h, w)  # current word

            # next word prediction
            log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(1)
            # log_prob: [k, vsc]
            log_prob = log_prob + log_prob_topk.view(k, 1)
            # top k probs in log_prob[k, vsc]
            log_prob_topk, indxs_topk = log_prob.view(-1).topk(k, sorted=True)
            # indxs_topk are a flat indecies, convert them to 2d indecies:
            # i.e the top k in all log_prob: get indecies: K, next_word_id
            prev_seq_k, next_word_id = np.unravel_index(
                indxs_topk.cpu(), log_prob.size()
            )
            next_word_id = torch.as_tensor(next_word_id).to(device).view(k, 1)
            # prev_seq_k [k], next_word_id [k]
            # if current_preds.size(1) > (max_len - 2):
            #     next_word_id = end_point
            # print(next_word_id.shape)
            current_preds = torch.cat((current_preds[prev_seq_k], next_word_id), dim=1)

        # find predicted sequences that ends
        if current_preds.size(1) > (max_len - 2):
            next_word_id[:, :] = EOS

        seqs_end = (next_word_id == EOS).view(-1)
        if torch.any(seqs_end) or current_preds.size(1) > (max_len - 2):
            # print("sequence ended")
            seq_preds.extend(seq.tolist() for seq in current_preds[seqs_end])
            # print(seq_preds.len())
            seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
            # get last layer, mean across transformer heads
            h, w = 1, 145
            attns = attns[-1].mean(dim=1).view(k, -1, h, w)
            # [k, S, h, w]
            seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

            k -= torch.sum(seqs_end)
            current_preds = current_preds[~seqs_end]
            log_prob_topk = log_prob_topk[~seqs_end]
            # current_attns = current_attns[~seqs_end]

    # Sort predicted captions according to seq_log_probs
    specials = [config["pad_id"], BOS, EOS]
    # print(len(seq_preds), len(seq_attns), len(seq_log_probs))
    seq_preds, seq_attns, seq_log_probs = zip(
        *sorted(
            zip(seq_preds, seq_attns, seq_log_probs),
            key=lambda tup: -tup[2],
        )
    )
    pred = tokenizer.decode_batch(seq_preds)
    pred = pred[0]
    if step % 20 == 0:
        print(pred)
    img_name = names[0]
    pred_caps[img_name] = pred

with open(
    pred_file,
    "w",
) as json_out:
    json.dump(pred_caps, json_out)
