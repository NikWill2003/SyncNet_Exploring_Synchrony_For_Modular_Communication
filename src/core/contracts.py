from __future__ import annotations

from typing import NotRequired, TypedDict

from torch import Tensor

class VQABatch(TypedDict):
    images: Tensor                    
    questions: Tensor                   
    answers: Tensor                     


class VQAOutput(TypedDict):
    logits: Tensor                     
    traces: NotRequired[dict]
    metrics: NotRequired[dict]          
