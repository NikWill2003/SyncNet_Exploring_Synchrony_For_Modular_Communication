from typing import Any

import numpy as np
import torch

from .spec import *

COLOURS_KEY = list(COLOURS.keys())


def to_cpu_tensor(x: Any) -> torch.Tensor:
    return x.detach().cpu() if isinstance(x, torch.Tensor) else torch.as_tensor(x)


def one_hot_idx(x: torch.Tensor, start: int, end: int) -> int:
    return int(x[start:end].argmax().item())


def get_question(question: torch.Tensor | np.ndarray) -> torch.Tensor:
    q = to_cpu_tensor(question)

    if q.shape != (QUESTION_SIZE,):
        raise ValueError(f'Expected question shape {(QUESTION_SIZE,)}, got {tuple(q.shape)}.')

    return q


def answer_idx(answer: torch.Tensor | np.ndarray | int) -> int:
    a = to_cpu_tensor(answer)

    if a.numel() == 1:
        return int(a.item())

    if a.shape == (ANSWER_SIZE,) and (a.dtype.is_floating_point or int((a != 0).sum().item()) == 1):
        return int(a.argmax().item())

    raise ValueError(f'Could not decode answer with shape {tuple(a.shape)}.')


def translate_question(question: torch.Tensor | np.ndarray) -> str:
    q = get_question(question)

    colour_1 = COLOURS_KEY[one_hot_idx(q, 0, 6)]
    q_type = one_hot_idx(q, Q_TYPE_IDX, Q_TYPE_IDX + 3)
    subtype = one_hot_idx(q, SUB_Q_TYPE_IDX, SUB_Q_TYPE_IDX + 3)

    if q_type == 0:
        if subtype == 0:
            return f'What shape is the {colour_1} object?'
        if subtype == 1:
            return f'Is the {colour_1} object on the left side of the image?'
        if subtype == 2:
            return f'Is the {colour_1} object in the top half of the image?'

    if q_type == 1:
        if subtype == 0:
            return f'What shape is the object closest to the {colour_1} object?'
        if subtype == 1:
            return f'What shape is the object furthest from the {colour_1} object?'
        if subtype == 2:
            return f'How many objects have the same shape as the {colour_1} object, excluding itself?'

    if q_type == 2:
        colour_2 = COLOURS_KEY[one_hot_idx(q, 6, 12)]

        if subtype == 0:
            return f'How many objects lie inside the box between the {colour_1} object and the {colour_2} object?'
        if subtype == 1:
            return f'Is there any object on the band between the {colour_1} object and the {colour_2} object?'
        if subtype == 2:
            return f'How many objects form an obtuse triangle with the {colour_1} object and the {colour_2} object?'

    raise ValueError(f'Unknown question type/subtype: q_type={q_type}, subtype={subtype}.')


def translate_answer(
    answer: torch.Tensor | np.ndarray | int,
    question: torch.Tensor | np.ndarray | None = None,
) -> str:
    idx = answer_idx(answer)

    if question is None:
        return ANSWERS[idx]

    q = get_question(question)

    q_type = one_hot_idx(q, Q_TYPE_IDX, Q_TYPE_IDX + 3)
    subtype = one_hot_idx(q, SUB_Q_TYPE_IDX, SUB_Q_TYPE_IDX + 3)

    count_question = (q_type == 1 and subtype == 2) or (q_type == 2 and subtype in {0, 2})

    if count_question:
        if idx < COUNT_OFFSET:
            raise ValueError(f'count question with non-count answer index {idx}')
        return COUNT_ANSWERS[idx]

    return ANSWERS[idx]