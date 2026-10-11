from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .config_ramnet import RAMNetConfig
from .modeling_ramnet import RAMNetModel, RAMNetForCausalLM

AutoConfig.register(RAMNetConfig.model_type, RAMNetConfig, exist_ok=True)
AutoModel.register(RAMNetConfig, RAMNetModel, exist_ok=True)
AutoModelForCausalLM.register(RAMNetConfig, RAMNetForCausalLM, exist_ok=True)

__all__ = ['RAMNetConfig', 'RAMNetModel', 'RAMNetForCausalLM', ]
