from .conv import VQAConv, VQAConvConfig
from .film import VQAFiLM, VQAFiLMConfig
from .question_only import VQAQuestionOnly, VQAQuestionOnlyConfig
from .relnet import VQARelNet, VQARelNetConfig
from .shared_workspace import VQAWorkspaceTransformer, VQAWorkspaceTransformerConfig
from .transformer import VQATransformer, VQATransformerConfig

MODELS: dict[str, tuple[type, type]] = {
    'transformer': (VQATransformerConfig, VQATransformer),
    'shared_workspace': (VQAWorkspaceTransformerConfig, VQAWorkspaceTransformer),
    'relnet': (VQARelNetConfig, VQARelNet),
    'film': (VQAFiLMConfig, VQAFiLM),
    'conv': (VQAConvConfig, VQAConv),
    'question_only': (VQAQuestionOnlyConfig, VQAQuestionOnly),
}
