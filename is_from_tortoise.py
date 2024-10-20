from tortoise.models.classifier import AudioMiniEncoderWithClassifierHead
import torch
import torch.nn.functional as F


def classify_audio_clip(waveform):
    waveform = waveform[:22000]
    classifier = AudioMiniEncoderWithClassifierHead(
        2,
        spec_dim=1,
        embedding_dim=512,
        depth=5,
        downsample_factor=4,
        resnet_blocks=2,
        attn_blocks=4,
        num_attn_heads=4,
        base_channels=32,
        dropout=0,
        kernel_size=5,
        distribute_zero_label=False,
    )
    classifier.load_state_dict(
        torch.load("classifier.pth", map_location=torch.device("cpu"))
    )
    clip = clip.cpu().unsqueeze(0)
    results = F.softmax(classifier(clip), dim=-1)
    return results[0][0]
