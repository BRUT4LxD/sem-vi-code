from dataclasses import dataclass, field
from typing import Optional

from torch.nn import Module


@dataclass
class LoadedModel:
    """
    Typed result from loading an ImageNette or binary classification model.
    
    Replaces the untyped dict previously returned by load_model_imagenette / load_model_binary.
    """
    model: Optional[Module] = None
    model_name: Optional[str] = None
    checkpoint: Optional[dict] = None
    training_state: Optional[dict] = field(default_factory=dict)
    device: str = 'cuda'
    success: bool = False
    error: Optional[str] = None
