from __future__ import annotations

import random
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from ..base import save_split
from PIL import Image, ImageDraw, ImageFont
from omegaconf import OmegaConf

from .spec import (
    SHAPE_TO_IDX,
    IMG_SIZE, MAX_OBJECT_SIZE, MIN_OBJECT_SIZE, NUM_OBJECTS,
    RELATIONS, SHAPES,
)
from .translate import encode_question

if TYPE_CHECKING:
    from .spec import SqoopDataConfig

_FONT_PATH = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}
_REPEAT_STEP = 2 * len(RELATIONS)
_MIN_NUM_OBJECTS = 4
_MAX_EXAMPLE_ATTEMPTS = 2000
_MAX_FILL_RESETS = 20


def _cfg_get(cfg, key: str, default):
    try:
        val = getattr(cfg, key)
    except Exception:
        return default
    return default if val is None else val


def _font_for_size(target_px: int) -> ImageFont.FreeTypeFont:
    if target_px not in _FONT_CACHE:
        pt = target_px
        for _ in range(6):
            f = ImageFont.truetype(_FONT_PATH, pt)
            _, t, _, b = f.getbbox('A')
            h = b - t
            if h == target_px or pt <= 4:
                break
            pt = max(4, round(pt * target_px / max(h, 1)))
        _FONT_CACHE[target_px] = ImageFont.truetype(_FONT_PATH, pt)
    return _FONT_CACHE[target_px]


class _Object:
    __slots__ = ('size', 'font', 'pos', 'shape')

    def __init__(self, size: int, pos=None, shape=None):
        self.size = size
        self.font = _font_for_size(size)
        self.pos = pos
        self.shape: str | None = shape

    def overlap(self, other: '_Object') -> bool:
        assert (self.pos is not None) and (other.pos is not None)
        min_dist = (self.size + other.size) // 2 + 1
        return (abs(self.pos[0] - other.pos[0]) < min_dist and
                abs(self.pos[1] - other.pos[1]) < min_dist)

    def relate(self, rel: str, other: '_Object') -> bool:
        assert (self.pos is not None) and (other.pos is not None)
        if rel == 'left_of':
            return self.pos[0] < other.pos[0]
        if rel == 'right_of':
            return self.pos[0] > other.pos[0]
        if rel == 'above':
            return self.pos[1] > other.pos[1]
        if rel == 'below':
            return self.pos[1] < other.pos[1]
        raise ValueError(rel)


def _get_random_spot(
        rng: np.random.RandomState,
        objects: list[_Object],
        img_size: int,
        min_obj: int,
        max_obj: int,
        rel: str | None = None,
        rel_holds: bool = False,
        rel_obj: int = 0,
        ) -> _Object | None:
    assert all(o.pos is not None for o in objects)
    size = int(rng.randint(min_obj, max_obj + 1))
    obj = _Object(size)

    min_c = obj.size // 2 + 1
    max_c = img_size - obj.size // 2 - 1

    if rel is not None:
        anchor = objects[rel_obj].pos
        assert anchor is not None
        if not rel_holds:
            max_cx = anchor[0] if rel == 'left_of' else max_c
            min_cx = anchor[0] if rel == 'right_of' else min_c
            max_cy = anchor[1] if rel == 'below' else max_c
            min_cy = anchor[1] if rel == 'above' else min_c
        else:
            min_cx = anchor[0] if rel == 'left_of' else min_c
            max_cx = anchor[0] if rel == 'right_of' else max_c
            min_cy = anchor[1] if rel == 'below' else min_c
            max_cy = anchor[1] if rel == 'above' else max_c
    else:
        min_cx = min_cy = min_c
        max_cx = max_cy = max_c

    if min_cx >= max_cx or min_cy >= max_cy:
        return None

    for _ in range(10):
        x = int(rng.randint(min_cx, max_cx))
        y = int(rng.randint(min_cy, max_cy))
        obj.pos = (x, y)
        if (any(abs(x - o.pos[0]) < 5 for o in objects) or # type: ignore
                any(abs(y - o.pos[1]) < 5 for o in objects)): # type: ignore
            continue
        if any(obj.overlap(o) for o in objects):
            continue
        return obj
    return None


def _fill_scene(
        rng: np.random.RandomState,
        objects: list[_Object],
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        restrict: bool,
        ) -> list[_Object] | None:

    orig = list(objects)
    restricted = {o.shape for o in orig} if restrict else set()
    allowed = [s for s in SHAPES if s not in restricted]
    if not allowed:
        raise ValueError('no distractor shapes available')

    out = list(orig)
    failures = 0
    resets = 0
    while len(out) < num_objects:
        shape = allowed[int(rng.randint(len(allowed)))]
        new = _get_random_spot(rng, out, img_size, min_obj, max_obj)
        if new is None:
            failures += 1
            if failures == 10:
                out = list(orig)
                failures = 0
                resets += 1
                if resets > _MAX_FILL_RESETS:
                    return None
            continue
        new.shape = shape
        out.append(new)
    return out


def _draw(objects: list[_Object], img_size: int) -> np.ndarray:
    assert all(o.pos is not None for o in objects)

    img = Image.new('RGB', (img_size, img_size))
    for obj in objects:
        assert (obj.pos is not None) and (obj.shape is not None)
        glyph = Image.new('RGBA', (obj.size + 4, obj.size + 4))
        d = ImageDraw.Draw(glyph)
        _, t, _, _ = obj.font.getbbox(obj.shape)
        d.text((0, -t), obj.shape, font=obj.font, fill='green')
        img.paste(
            glyph,
            (obj.pos[0] - glyph.size[0] // 2, obj.pos[1] - glyph.size[1] // 2),
            glyph,
        )
    return np.asarray(img, dtype=np.uint8)


def _gen_example(
        pair: tuple[str, str],
        rel: str,
        label: bool,
        rng: np.random.RandomState,
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        restrict_positive: bool = False,
        ):
    x, y = pair
    if label:
        obj1 = _get_random_spot(rng, [], img_size, min_obj, max_obj)
        if obj1 is None:
            return None
        obj2 = _get_random_spot(rng, [obj1], img_size, min_obj, max_obj)
        if not obj2 or not obj1.relate(rel, obj2):
            return None
        obj1.shape, obj2.shape = x, y
        scene = _fill_scene(
            rng, [obj1, obj2], num_objects, img_size, min_obj, max_obj,
            restrict=restrict_positive,
        )
        if scene is None:
            return None
    else:
        obj1 = _get_random_spot(rng, [], img_size, min_obj, max_obj)
        if obj1 is None:
            return None
        obj2 = _get_random_spot(
            rng, [obj1], img_size, min_obj, max_obj,
            rel=rel, rel_holds=False,
        )
        if not obj2 or obj1.relate(rel, obj2):
            return None
        obj1.shape, obj2.shape = x, y
        scene = _fill_scene(
            rng, [obj1, obj2], num_objects, img_size, min_obj, max_obj,
            restrict=True,
        )
        if scene is None:
            return None
        # hard negative: require x rel y' and x' rel y to hold
        obj3, obj4 = scene[2], scene[3]
        if not obj1.relate(rel, obj4):
            return None
        if not obj3.relate(rel, obj2):
            return None

    return _draw(scene, img_size), encode_question(x, rel, y), int(label)


def _repeats_for(target: int, n_pairs: int, label: str) -> int:

    if target <= 0:
        raise ValueError(f'{label} must be positive, got {target}')
    if n_pairs <= 0:
        raise ValueError(f'{label}: no pairs to distribute over')
    repeats = (target // n_pairs)
    repeats -= repeats % _REPEAT_STEP
    if repeats <= 0:
        raise ValueError(
            f'{label}={target:,} over {n_pairs} pairs leaves under '
            f'{_REPEAT_STEP} examples per pair, which cannot be balanced. '
            f'Raise it to at least {n_pairs * _REPEAT_STEP:,}.'
        )
    return repeats


def _build_schedule(
        pairs_unique: list[tuple[str, str]],
        repeats: int,
        rng: np.random.RandomState,
        ) -> list[tuple[tuple[str, str], str, bool]]:

    per_cell = repeats // len(RELATIONS)
    schedule: list[tuple[tuple[str, str], str, bool]] = []
    for pair in pairs_unique:
        for rel in RELATIONS:
            for k in range(per_cell):
                schedule.append((pair, rel, k < per_cell // 2))

    order = rng.permutation(len(schedule))
    schedule = [schedule[i] for i in order]

    pos = Counter()
    tot = Counter()
    for pair, rel, label in schedule:
        tot[(pair, rel)] += 1
        pos[(pair, rel)] += int(label)
    for cell, n in tot.items():
        assert pos[cell] * 2 == n, f'unbalanced cell {cell}: {pos[cell]}/{n}'

    return schedule


def _gen_split(
        schedule: list[tuple[tuple[str, str], str, bool]],
        seed: int,
        num_objects: int,
        img_size: int,
        min_obj: int,
        max_obj: int,
        restrict_positive: bool = False,
        ) -> dict[str, np.ndarray]:
    rng = np.random.RandomState(seed)
    n = len(schedule)

    images = np.empty((n, img_size, img_size, 3), dtype=np.uint8)
    questions = np.empty((n, 3), dtype=np.int64)
    answers = np.empty((n,), dtype=np.int64)

    total_attempts = 0
    worst_attempts = 0

    for i, (pair, rel, label) in enumerate(schedule):
        for attempt in range(1, _MAX_EXAMPLE_ATTEMPTS + 1):
            out = _gen_example(
                pair, rel, label, rng,
                num_objects, img_size, min_obj, max_obj,
                restrict_positive=restrict_positive,
            )
            if out is not None:
                break
        else:
            raise RuntimeError(f'could not generate example {i} ')
        total_attempts += attempt
        worst_attempts = max(worst_attempts, attempt)
        images[i], questions[i], answers[i] = out

    return {'images': images, 'questions': questions, 'answers': answers}


def prepare_sqoop(data_cfg: 'SqoopDataConfig') -> None:
    cfg = data_cfg
    out_dir = Path(cfg.root) / cfg.dir
    out_dir.mkdir(parents=True, exist_ok=True)

    base_seed = int(cfg.seed)
    rhs = int(cfg.rhs_variety)

    num_objects = NUM_OBJECTS
    if num_objects < _MIN_NUM_OBJECTS:
        raise ValueError(
            f'num_objects={num_objects} but the hard-negative construction '
        )

    restrict_positive = bool(_cfg_get(cfg, 'restrict_positive', False))

    py_rng = random.Random(base_seed)
    all_pairs = {(x, y) for x in SHAPES for y in SHAPES if x != y}
    unseen = set(all_pairs)
    train_pairs_unique: list[tuple[str, str]] = []
    for i, x in enumerate(SHAPES):
        ys = py_rng.sample(SHAPES[:i] + SHAPES[i + 1:], rhs)
        for y in ys:
            unseen.remove((x, y))
            train_pairs_unique.append((x, y))

    left = sorted(unseen)
    py_rng.shuffle(left)
    val_slice = len(left) // 2
    val_unseen_pairs = left[:val_slice]
    test_unseen_pairs = left[val_slice:]

    n_train = int(cfg.train_size)
    n_eval = int(cfg.test_size)

    repeats = _repeats_for(n_train, len(train_pairs_unique), 'train_size')
    rep_seen = _repeats_for(n_eval, len(train_pairs_unique),
                            'test_size (val_seen)')

    iid = not val_unseen_pairs and not test_unseen_pairs
    if iid:
        print(f'sqoop rhs={rhs}: IID control')
        rep_val = rep_test = 0
    else:
        rep_val = _repeats_for(n_eval, len(val_unseen_pairs),
                               'test_size (val_unseen)')
        rep_test = _repeats_for(n_eval, len(test_unseen_pairs),
                                'test_size (test_unseen)')

    sched_rng = np.random.RandomState(base_seed)
    schedules = {
        'train': (
            _build_schedule(train_pairs_unique, repeats, sched_rng),
            base_seed + 1,
        ),
        'val_seen': (
            _build_schedule(train_pairs_unique, rep_seen, sched_rng),
            base_seed + 2,
        ),
    }
    if iid:
        schedules['test_unseen'] = (
            _build_schedule(train_pairs_unique, rep_seen, sched_rng),
            base_seed + 4,
        )
    else:
        schedules['val_unseen'] = (
            _build_schedule(val_unseen_pairs, rep_val, sched_rng),
            base_seed + 3,
        )
        schedules['test_unseen'] = (
            _build_schedule(test_unseen_pairs, rep_test, sched_rng),
            base_seed + 4,
        )

    print(
        f'sqoop rhs={rhs}: {len(train_pairs_unique)} train pairs '
        f'({len(schedules["train"][0]):,} train ex at {repeats}/pair, '
        f'target {n_train:,}), '
        f'{len(left)} unseen pairs '
        f'({len(schedules["val_unseen"][0]) if not iid else 0} val_unseen / '
        f'{len(schedules["test_unseen"][0]) if not iid else 0} test_unseen ex), '
        f'{len(schedules["val_seen"][0])} val_seen ex; '
        f'restrict_positive={restrict_positive}'
    )

    for name, (schedule, seed) in schedules.items():
        print(f'building {name} ({len(schedule)} examples)...')
        arrays = _gen_split(
            schedule, seed,
            num_objects=num_objects,
            img_size=IMG_SIZE,
            min_obj=MIN_OBJECT_SIZE,
            max_obj=MAX_OBJECT_SIZE,
            restrict_positive=restrict_positive,
        )
        save_split(arrays, out_dir / f'{name}.npz')
        print(f'saved {name}.npz to {out_dir}')

    if is_dataclass(cfg):
        manifest = asdict(cfg)
    else:
        manifest = OmegaConf.to_container(cfg, resolve=True)
    manifest['restrict_positive'] = restrict_positive # type: ignore
    manifest['repeats_per_pair_effective'] = repeats # type: ignore
    manifest['train_size_realised'] = (  # type: ignore
        len(schedules['train'][0])
    )
    OmegaConf.save(OmegaConf.create(manifest), out_dir / 'manifest.yaml')