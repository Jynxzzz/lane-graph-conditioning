# tools/encoder/__init__.py
# from .latent_path_token import LatentPathEncoder
# from .neighbor_token import NeighborEncoder
from .simple_token import SimpleEncoder

ENCODER_REGISTRY = {
    "simple_token": SimpleEncoder,
    # "neighbor_token": NeighborEncoder,
    # "latent_path_token": LatentPathEncoder,
}


def build_encoder(name, **kwargs):
    if name not in ENCODER_REGISTRY:
        raise ValueError(f"未知 encoder：{name}")
    return ENCODER_REGISTRY[name](**kwargs)  # ✅ 返回 encoder 实例
