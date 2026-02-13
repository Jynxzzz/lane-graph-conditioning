from .lstm_baseline import LSTMBaseline
from .lane_conditioned_lstm import LaneConditionedLSTM
from .dual_supervised_lstm import DualSupervisedLSTM
from .transformer_baseline import TransformerBaseline
from .transformer_lane_cond import TransformerLaneCond
from .transformer_dual_sup import TransformerDualSup
from .multimodal_lstm import MultiModalLSTMBaseline
from .multimodal_lane_cond import MultiModalLaneCond
from .flow_matching import LaneFlowNet


MODEL_REGISTRY = {
    "lstm_baseline": LSTMBaseline,
    "lane_conditioned": LaneConditionedLSTM,
    "dual_supervised": DualSupervisedLSTM,
    "tf_baseline": TransformerBaseline,
    "tf_lane_cond": TransformerLaneCond,
    "tf_dual_sup": TransformerDualSup,
    "multimodal_lstm_baseline": MultiModalLSTMBaseline,
    "multimodal_lane_cond": MultiModalLaneCond,
    "flow_matching": LaneFlowNet,
}


def build_model(name, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[name](**kwargs)
