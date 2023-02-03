from datetime import datetime
from tqdm.auto import tqdm
import json
import numpy as np
import os, random
import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
from evaluate_score import getScore
from torch.nn import functional as F


BOS = 2
EOS = 3


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class Trainer:
    def __init__(
        self,
        transformer,
        device,
        epochs: int,
        val_interval: int,
        max_norm: float,
        checkpoints_path: str,
        pad_id: int,
        loadModel: str,
        max_len=55,
    ) -> None:

        # Some parameters
        self.train = True  # train or val
        self.device = device
        self.epochs_num = epochs - 1  # epoch count start from 0
        self.epoch = 0
        self.loadModel = loadModel
        self.val_interval = val_interval  # validate the model evey (n) epochs
        # stop trianing if the model doesn't improve for n-validation epochs
        # number of validation epochs in which model doesn't improve
        self.bad_epochs_num = 0
        # number of validation epochs to wait before decreases the lr if model
        # does not improve
        # start tune embeddings after n training epochs have beed passed
        self.pad_id = pad_id

        # criterion, optims and schedulers
        self.criterion = nn.CrossEntropyLoss(ignore_index=pad_id).to(device)
        # self.criterion = LabelSmoothing(54, pad_id, 0.4).to(device)

        self.transformer = transformer.to(self.device)
        self.transformer_optim = Adam(
            self.transformer.parameters(), lr=1e-4, weight_decay=1e-5
        )
        self.transformer_scheduler = ReduceLROnPlateau(
            self.transformer_optim, factor=0.9, patience=1, mode="max"
        )

        if self.loadModel:
            self.load_checkpoint(self.loadModel)
        # Some coeffecient
        self.max_norm = max_norm  # gradient clip coeffecient
        self.grad_clip_c = 5
        self.max_len = max_len

        self.checkpoints_path = checkpoints_path

    def loss_fn(self, logits: Tensor, targets: Tensor) -> Tensor:
        v_sz = logits.size()[-1]
        targets = targets.contiguous()
        loss = self.criterion(logits.view(-1, v_sz), targets.view(-1))
        return loss

    def remove_pad(self, tensor, mask):
        out = []
        max_len = tensor.size(1)
        is3d = len(tensor.size()) == 3

        if is3d:
            tensor = tensor.permute(0, 2, 1).contiguous().view(-1, max_len)
            mask = mask.permute(0, 2, 1).contiguous().view(-1, max_len)

        for i in range(tensor.size(0)):
            unpad = list(torch.masked_select(tensor[i], mask=mask[i]))
            unpad = [int(e) for e in unpad]
            out.append(unpad)

        if is3d:
            out = [out[i : i + 5] for i in range(0, len(out), 5)]

        return out

    def clip_gradient(self):
        for group in self.transformer_optim.param_groups:
            for param in group["params"]:
                if param.grad is not None:
                    param.grad.data.clamp_(-self.grad_clip_c, self.grad_clip_c)

    def set_phase(self) -> None:
        if not self.train:
            self.train = True  # toggle if val
        else:
            # validate every "val_interval" epoch
            self.train = bool(self.epoch % self.val_interval)

    def load_checkpoint(self, load_path):
        print("loading model from ", load_path)
        # load checkopoint
        state = torch.load(load_path, self.device)
        transformer_state = state["models"]
        transformer_optim_state = state["optims"]
        transformer_scheduler_state = state["schedulers"]
        # state["optim"]

        # load state dicts
        self.transformer.load_state_dict(transformer_state)
        self.transformer_optim.load_state_dict(transformer_optim_state)
        self.transformer_scheduler.load_state_dict(transformer_scheduler_state)
        # self.optim.load_state_dict()

        self.transformer = self.transformer.to(self.device)

        # set some parameters
        self.train = state["phase"]
        self.epoch = state["epoch"]
        self.bad_epochs_num = state["bad_epochs_num"]

        self.set_phase()  # set train or vall phase
        self.epoch += 1 * self.train

        return

    def save_checkpoint(self, name, model):

        Transformer_state = model.state_dict()
        transformer_optim_state = self.transformer_optim.state_dict()
        transformer_scheduler_state = self.transformer_scheduler.state_dict()

        state = {
            "models": Transformer_state,
            "optims": transformer_optim_state,
            "schedulers": transformer_scheduler_state,
            "phase": self.train,
            "epoch": self.epoch,
            "bad_epochs_num": self.bad_epochs_num,
        }

        # set save path
        torch.save(state, os.path.join(self.checkpoints_path, f"{name}.pth"))

    def run_train(self, name, transformer, tokenizer, data_iters, SEED: int):
        # Sizes:
        # B:   batch_size
        # is:  image encode size^2: image seq len: [default=196]
        # vsc: vocab_size: vsz
        # lm:  max_len: [default=52]
        # cn:  number of captions: [default=5]
        # hn:  number of transformer heads: [default=8]
        # ln:  number of layers
        # k:   Beam Size

        # some preparations:
        time = datetime.now().strftime("%m%d-%H%M_")
        train_name = time + name

        phases = ["val", "train"]  # to determine the current phase
        seed_everything(SEED)
        best_clip = 0
        best_cider = 0

        # start
        main_pb = tqdm(range(self.epochs_num))
        while self.epoch <= self.epochs_num:

            main_pb.set_description(f"epoch: {self.epoch:02d}")

            es = False  # early stopping
            lr_r = False  # reduce lr flag

            if self.train:
                transformer.train()
                data_iter = data_iters[0]
                # img_dir = "/home/eegroup/ee50525/b09901015/hw3-WEIoct/hw3_data/p2_data/images/train"
                # cap_file = "/home/eegroup/ee50525/b09901015/hw3-WEIoct/hw3_data/p2_data/train.json"
                # fine tune the embeddings layer after some epochs and add the
                # parameters to the optimizer
            else:
                transformer.eval()
                data_iter = data_iters[1]
                img_dir = "/home/stan/hw3-stanthemaker/hw3_data/p2_data/images/val"
                cap_file = "/home/stan/hw3-stanthemaker/hw3_data/p2_data/val.json"
                val_clip = []
                val_cider = []
                pred_caps = {}
                predictions = []
                answers = []

            # Iterate over data
            # prgress bar
            pb = tqdm(data_iter, leave=False, total=len(data_iter))
            pb.unit = "step"

            for step, (imgs, names, cptns_all) in enumerate(pb):
                imgs: Tensor  # images [B, 3, 384, 384]
                cptns_all: Tensor  # all 5 captions [B, lm, cn=5]

                # set progress bar description and metrics
                pb.set_description(f"{phases[self.train]}: Step-{step+1:<4d}")

                # move data to device, and random selected cptns
                imgs = imgs.to(self.device)
                # random selected cptns: [B, lm]
                idx = np.random.randint(0, cptns_all.size(-1))
                cptns = cptns_all[:, :, idx].to(self.device)
                # cptns = cptns_all.to(self.device)

                if self.train:

                    # zero the parameter gradients
                    self.transformer_optim.zero_grad()
                    # self.optim.zero_grad()

                    logits, _ = transformer(imgs, cptns[:, :-1])  ## original
                    # print('logits', logits)
                    # logits = transformer(imgs, cptns[:, :-1]) ## attention is all you need

                    logits: Tensor  # [B, lm - 1, vsz]
                    attns: Tensor  # [ln, B, hn, lm, is]

                    # loss calc, backward
                    loss = self.loss_fn(logits, cptns[:, 1:])
                    # in train, gradient clip + update weights
                    loss.backward()
                    # nn.utils.clip_grad_norm_(self.transformer.parameters(), self.max_norm)
                    self.clip_gradient()
                    self.transformer_optim.step()
                    # self.optim.step()
                else:

                    with torch.no_grad():
                        # logits, _ = transformer(imgs, cptns[:, :-1])
                        # logits = transformer(imgs, cptns[:, :-1])
                        caption = torch.zeros((1, self.max_len), dtype=torch.long)
                        caption[:, 0] = BOS

                        for i in range(55 - 1):
                            caption = caption.to(self.device)
                            predictions, _ = transformer(imgs, caption)
                            predictions = predictions[:, i, :]
                            predicted_id = torch.argmax(predictions, axis=-1)

                            caption[:, i + 1] = predicted_id[0]

                            if predicted_id[0] == EOS:
                                break
                    pred = tokenizer.decode_batch(caption.tolist())
                    pred = pred[0]
                    # print(pred)
                    img_name = names[0]
                    pred_caps[img_name] = pred
                    # --------------- beam search -------------#
                    # max_len = 55
                    # imgs: Tensor  # images [1, 3, 256, 256]
                    # cptns_all: Tensor  # all 5 captions [1, lm, cn=5]

                    # k = 5
                    # # seq_preds = beam_search(imgs, k, transformer, max_len, self.device)
                    # # start: [1, 1]
                    # start = torch.full(
                    #     size=(1, 1),
                    #     fill_value=BOS,
                    #     dtype=torch.long,
                    #     device=self.device,
                    # )

                    # with torch.no_grad():
                    #     imgs_enc = transformer.encoder(imgs)
                    #     imgs_enc = transformer.match_size(imgs_enc)
                    #     logits, attns = transformer.decoder(
                    #         start, imgs_enc.permute(1, 0, 2)
                    #     )
                    #     logits = (
                    #         transformer.predictor(logits).permute(1, 0, 2).contiguous()
                    #     )

                    #     logits: Tensor  # [k=1, 1, vsc]
                    #     attns: Tensor  # [ln, k=1, hn, S=1, is]

                    #     log_prob = F.log_softmax(logits, dim=2)
                    #     log_prob_topk, indxs_topk = log_prob.topk(k, sorted=True)
                    #     # log_prob_topk [1, 1, k]
                    #     # indices_topk [1, 1, k]
                    #     current_preds = torch.cat(
                    #         [start.expand(k, 1), indxs_topk.view(k, 1)], dim=1
                    #     )
                    #     # current_preds: [k, S]

                    #     # get last layer, mean across transformer heads
                    #     # attns = attns[-1].mean(dim=1).view(1, 1, h, w)  # [k=1, s=1, h, w]
                    #     # current_attns = attns.repeat_interleave(repeats=k, dim=0)
                    #     # [k, s=1, h, w]

                    # # print("current_preds", current_preds.size())
                    # # print("k", k)
                    # seq_preds = []
                    # seq_log_probs = []
                    # seq_attns = []
                    # while (
                    #     current_preds.size(1) <= (max_len - 2)
                    #     and k > 0
                    #     and current_preds.nelement()
                    # ):
                    #     with torch.no_grad():
                    #         print("prediction len", current_preds.size(1))
                    #         print("running beam search")
                    #         # print("seq_preds", seq_preds)
                    #         # print("seq_attns", seq_attns)
                    #         # print("seq_log_probs", seq_log_probs)

                    #         imgs_expand = imgs_enc.expand(k, *imgs_enc.size()[1:])
                    #         # print('size of current_pred', current_preds.size())
                    #         # print('size of imgs_expand', imgs_expand.size())
                    #         # [k, is, ie]
                    #         logits, attns = transformer.decoder(
                    #             current_preds, imgs_expand.permute(1, 0, 2)
                    #         )
                    #         logits = (
                    #             transformer.predictor(logits)
                    #             .permute(1, 0, 2)
                    #             .contiguous()
                    #         )
                    #         # logits: [k, S, vsc]
                    #         # attns: # [ln, k, hn, S, is]
                    #         # get last layer, mean across transformer heads
                    #         # attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                    #         # # [k, S, h, w]
                    #         # attns = attns[:, -1].view(k, 1, h, w)  # current word

                    #         # next word prediction
                    #         log_prob = F.log_softmax(logits[:, -1:, :], dim=-1).squeeze(
                    #             1
                    #         )
                    #         # log_prob: [k, vsc]
                    #         log_prob = log_prob + log_prob_topk.view(k, 1)
                    #         # top k probs in log_prob[k, vsc]
                    #         log_prob_topk, indxs_topk = log_prob.view(-1).topk(
                    #             k, sorted=True
                    #         )
                    #         # indxs_topk are a flat indecies, convert them to 2d indecies:
                    #         # i.e the top k in all log_prob: get indecies: K, next_word_id
                    #         prev_seq_k, next_word_id = np.unravel_index(
                    #             indxs_topk.cpu(), log_prob.size()
                    #         )
                    #         next_word_id = (
                    #             torch.as_tensor(next_word_id).to(self.device).view(k, 1)
                    #         )
                    #         # prev_seq_k [k], next_word_id [k]

                    #         current_preds = torch.cat(
                    #             (current_preds[prev_seq_k], next_word_id), dim=1
                    #         )

                    #     # find predicted sequences that ends
                    #     seqs_end = (next_word_id == EOS).view(-1)
                    #     if torch.any(seqs_end):
                    #         print("sequence ended")
                    #         seq_preds.extend(
                    #             seq.tolist() for seq in current_preds[seqs_end]
                    #         )
                    #         seq_log_probs.extend(log_prob_topk[seqs_end].tolist())
                    #         # get last layer, mean across transformer heads
                    #         h, w = 1, 145
                    #         attns = attns[-1].mean(dim=1).view(k, -1, h, w)
                    #         # [k, S, h, w]
                    #         seq_attns.extend(attns[prev_seq_k][seqs_end].tolist())

                    #         k -= torch.sum(seqs_end)
                    #         current_preds = current_preds[~seqs_end]
                    #         log_prob_topk = log_prob_topk[~seqs_end]
                    #         # current_attns = current_attns[~seqs_end]

                    # # Sort predicted captions according to seq_log_probs
                    # specials = [self.pad_id, BOS, EOS]
                    # seq_preds, seq_attns, seq_log_probs = zip(
                    #     *sorted(
                    #         zip(seq_preds, seq_attns, seq_log_probs),
                    #         key=lambda tup: -tup[2],
                    #     )
                    # )

                    # print(seq_preds)
                    # pred = tokenizer.decode_batch(seq_preds)
                    # pred = pred[0]
                    # # print(pred)
                    # img_name = names[0]
                    # pred_caps[img_name] = pred

            # self.metrics_tracker.update(phases[self.train])  # save metrics
            pred_file = f"/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/output/{train_name}.json"
            if not self.train:
                with open(
                    pred_file,
                    "w",
                ) as json_out:
                    json.dump(pred_caps, json_out)

                print("calculating scores...")
                val_cider, val_clip = getScore(pred_file)
                print("finished calculating scores")

                with open(
                    f"/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/log/{train_name}_log.txt",
                    "a",
                ) as f:
                    f.write(
                        f"[ {self.epoch + 1:03d}/{self.epochs_num:03d} ] loss = {loss:.4f} | CIDEr score: {val_cider:.4f} | CLIP score: {val_clip:.4f}\n"
                    )
                    print(
                        f"[ {self.epoch + 1:03d}/{self.epochs_num:03d} ] loss = {loss:.4f} | CIDEr score: {val_cider:.4f} | CLIP score: {val_clip:.4f}]"
                    )

                if val_cider > best_cider and val_clip > best_clip:
                    best_cider = val_cider
                    best_clip = val_clip
                    self.save_checkpoint(train_name, transformer)
                    with open(
                        f"/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/log/{train_name}_log.txt",
                        "a",
                    ) as f:
                        f.write(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]\n"
                        )
                        print(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )

                if val_clip > best_clip and val_cider > 0.87:
                    best_clip = val_clip
                    self.save_checkpoint(train_name, transformer)
                    with open(
                        f"/home/eegroup/ee50525/b09901015/hw3-WEIoct/p2/log/{train_name}_log.txt",
                        "a",
                    ) as f:
                        f.write(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]\n"
                        )
                        print(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )
                if val_cider > best_cider and val_clip > 0.70:
                    best_cider = val_cider
                    self.save_checkpoint(train_name, transformer)
                    with open(
                        f"/home/stan/hw3-stanthemaker/problem2_ImageCaptioning/log/{train_name}_log.txt",
                        "a",
                    ) as f:
                        f.write(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]\n"
                        )
                        print(
                            f"[ saving model at epoch {self.epoch} with CLIP score: {val_clip:.4f} | CIDEr score {val_cider:.4f} ]"
                        )

                self.transformer_scheduler.step(best_cider + best_clip)

            # epoch ended
            self.set_phase()  # set train or val phase
            self.epoch += 1 * self.train
            pb.close()  # close progress bar
            if self.train:
                main_pb.update(1)
            if es:  # early stopping
                main_pb.close()
                print(f"Early stop training at epoch {self.epoch}")
                break
