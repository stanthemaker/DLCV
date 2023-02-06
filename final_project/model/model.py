import torch, os, math, logging
import torch.nn as nn
import torch.optim
import torch.utils.data
from torch.nn.init import normal, constant
from model.resnet import resnet18
from model.resse import ResNetSE
import timm
from torch import einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from model.module import Attention, PreNorm, FeedForward

from torchaudio.models.wav2vec2.utils import import_huggingface_model # for audio encoder
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

logger = logging.getLogger(__name__)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.norm = nn.LayerNorm(dim)
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)


class VisionEncoder(nn.Module):
    def __init__(self, pretrained_model="vit_base_resnet26d_224", feature_dim=256):
        super(VisionEncoder, self).__init__()
        self.vit = timm.create_model(pretrained_model, pretrained=True)
        self.match_size = nn.Linear(1024, feature_dim)

    def forward(self, x):
        x = self.vit.forward_features(x)
        x = self.match_size(x)
        return x


class ViViT(nn.Module):
    def __init__(
        self,
        args,
        device,
        image_size=224,
        patch_size=16,
        num_classes=2,
        num_frames=400,
        dim=512,
        depth=4,
        heads=3,
        pool="cls",
        in_channels=3,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
        scale_dim=4,
        
        
    ):
        super().__init__()

        assert pool in {
            "cls",
            "mean",
        }, "pool type must be either cls (cls token) or mean (mean pooling)"

        assert (
            image_size % patch_size == 0
        ), "Image dimensions must be divisible by the patch size."
        num_patches = (image_size // patch_size) ** 2
        patch_dim = in_channels * patch_size**2
        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b t c (h p1) (w p2) -> b t (h w) (p1 p2 c)",
                p1=patch_size,
                p2=patch_size,
            ),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_frames, num_patches + 1, dim)
        )
        self.space_token = nn.Parameter(torch.randn(1, 1, dim))
        self.space_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim, dropout
        )
        # self.space_transformer = VisionEncoder(feature_dim=dim*scale_dim)

        self.temporal_token = nn.Parameter(torch.randn(1, 1, dim))
        self.temporal_transformer = Transformer(
            dim, depth, heads, dim_head, dim * scale_dim, dropout
        )
        # self.temporal_transformer = VisionEncoder(feature_dim=dim*scale_dim)

        self.device = device
        self.audio_encoder_args = args.audio_encoder
        if self.audio_encoder_args == "ResNetSE":
            self.audio_encoder = ResNetSE()
        
        elif self.audio_encoder_args == "huggingface":
            self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
            original = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
            self.audio_encoder = import_huggingface_model(original)
            self.audio_matcher = nn.Linear(32*400, 512)

        self.dropout = nn.Dropout(emb_dropout)
        self.pool = pool

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim * 2),
            nn.Linear(dim * 2, dim // 2),
            nn.ReLU(),
            nn.Linear(dim // 2, num_classes),
        )

    def forward(self, x, audio):
        # [b, t, c, h, w]
        x = self.to_patch_embedding(x)
        # [b, t, (h/patch_size)*(w/patch_size), dim]
        b, t, n, _ = x.shape
        # logger.info(f"batch size: {b}, frames: {t}, n: {n}")

        cls_space_tokens = repeat(self.space_token, "() n d -> b t n d", b=b, t=t)
        x = torch.cat((cls_space_tokens, x), dim=2)
        x += self.pos_embedding[:, :t, : (n + 1)]
        x = self.dropout(x)
        # [b, t, (h/patch_size)*(w/patch_size) + 1, dim]

        x = rearrange(x, "b t n d -> (b t) n d")
        # [b*t, n, d]
        x = self.space_transformer(x)
        # [b*t, n, d]
        x = rearrange(x[:, 0], "(b t) ... -> b t ...", b=b)
        # [b, t, d]

        cls_temporal_tokens = repeat(self.temporal_token, "() n d -> b n d", b=b)
        x = torch.cat((cls_temporal_tokens, x), dim=1)
        # [b, t+1, d]

        x = self.temporal_transformer(x)
        # [b, t+1, d]
        x = x.mean(dim=1) if self.pool == "mean" else x[:, 0]
        
        x += torch.randn(x.size()).to(self.device)
        # [b, d]
        print(x.size())

        audio_out = None
        if self.audio_encoder_args == "ResNetSE":
            audio_out = self.audio_encoder(audio)
        
        elif self.audio_encoder_args == "huggingface":
            # input_values = self.processor(audio, return_tensors="pt", padding="longest").input_values
            audio_out, _ = self.audio_encoder(audio)
            b, t, _ = audio_out.shape
            target = torch.zeros(b, 400, 32).to(self.device)
            target[:, :t, :] = audio_out
            target = target.view(b, 400*32)
            audio_out = self.audio_matcher(target)
        
        
        
        print("audio feature:", audio_out.size())

        whole_feature = torch.cat((x, audio_out), dim=1)

        return self.mlp_head(whole_feature)


class BaselineLSTM(nn.Module):
    def __init__(self, args):
        super(BaselineLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = (
            256  # the dimension of the CNN feature to represent each frame
        )

        self.video_encoder = resnet18(pretrained=False)

        self.video_encoder.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(
            self.img_feature_dim,
            self.img_feature_dim,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )

        self.audio_encoder = ResNetSE()

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer1 = nn.Linear(4 * self.img_feature_dim, 128)

        self.last_layer2 = nn.Linear(128, 2)

        for param in self.parameters():
            param.requires_grad = True

        self._init_parameters()

        self.load_checkpoint()

    def forward(self, video, audio):
        N, D, C, H, W = video.shape

        video_out = self.video_encoder(video.view(N * D, C, H, W))
        video_out = video_out.view(N, D, self.img_feature_dim)

        lstm_out, _ = self.lstm(video_out)
        lstm_out = lstm_out[:, -1, :]

        audio_out = self.audio_encoder(audio)

        output = self.last_layer1(torch.cat((lstm_out, audio_out), dim=1))

        output = self.last_layer2(output)

        return output

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f"loading checkpoint {self.args.checkpoint}")
                state = torch.load(
                    self.args.checkpoint, map_location=f"cuda:{self.args.device_id}"
                )
                if "module" in list(state["state_dict"].keys())[0]:
                    state_dict = {k[7:]: v for k, v in state["state_dict"].items()}
                else:
                    state_dict = state["state_dict"]
                self.load_state_dict(state_dict)

    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear or nn.Conv2d):
                m.weight.data = nn.init.xavier_uniform_(
                    m.weight.data, gain=nn.init.calculate_gain("relu")
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        # elif isinstance(m, nn.BatchNorm2d):
        #     nn.init.constant_(m.weight, 1)
        #     nn.init.constant_(m.bias, 0)


class GazeLSTM(nn.Module):
    def __init__(self, args):
        super(GazeLSTM, self).__init__()
        self.args = args
        self.img_feature_dim = (
            256  # the dimension of the CNN feature to represent each frame
        )

        self.video_encoder = resnet18(pretrained=True)

        self.video_encoder.fc2 = nn.Linear(1000, self.img_feature_dim)

        self.lstm = nn.LSTM(
            self.img_feature_dim,
            self.img_feature_dim,
            bidirectional=True,
            num_layers=2,
            batch_first=True,
        )

        # The linear layer that maps the LSTM with the 3 outputs
        self.last_layer = nn.Linear(2 * self.img_feature_dim, 3)
        self.load_checkpoint()

    def forward(self, video):

        video_out = self.video_encoder(video.view((-1, 3) + video.size()[-2:]))

        video_out = video_out.view(video.size(0), 7, self.img_feature_dim)

        lstm_out, _ = self.lstm(video_out)
        lstm_out = lstm_out[:, 3, :]
        output = self.last_layer(lstm_out).view(-1, 3)

        angular_output = output[:, :2]
        angular_output[:, 0:1] = math.pi * nn.Tanh()(angular_output[:, 0:1])
        angular_output[:, 1:2] = (math.pi / 2) * nn.Tanh()(angular_output[:, 1:2])

        var = math.pi * nn.Sigmoid()(output[:, 2:3])
        var = var.view(-1, 1).expand(var.size(0), 2)

        return angular_output, var

    def load_checkpoint(self):
        if self.args.checkpoint is not None:
            if os.path.exists(self.args.checkpoint):
                logger.info(f"loading checkpoint {self.args.checkpoint}")
                map_loc = (
                    f"cuda:{self.args.device_id}" if torch.cuda.is_available() else "cpu"
                )
                state = torch.load(self.args.checkpoint, map_location=map_loc)
                if "module" in list(state["state_dict"].keys())[0]:
                    state_dict = {k[7:]: v for k, v in state["state_dict"].items()}
                else:
                    state_dict = state["state_dict"]
                self.load_state_dict(state_dict)


if __name__ == "__main__":
    video_test = torch.FloatTensor(4, 21, 224, 224)
    model_test = BaselineLSTM()
    out = model_test.forward(video_test)
