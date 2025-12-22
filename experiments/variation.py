import copy
import random
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

from experiments.random_camera import RandomCamera
from experiments.random_lighting import RandomLighting


_TEMPLATES = {
    "grasp": [
        "pick [OBJECT]",
        "grab [OBJECT]",
        "can you pick up [OBJECT]",
        "fetch [OBJECT]",
        "get [OBJECT]",
        "lift [OBJECT]",
        "take [OBJECT]",
        "retrieve [OBJECT]",
        "let's pick up [OBJECT]",
        "would you grab [OBJECT]",
    ],
    "move": [
        "Take [OBJECT A] to [OBJECT B]",
        "Bring [OBJECT A] close to [OBJECT B]",
        "Position [OBJECT A] near [OBJECT B]",
        "Move [OBJECT A] closer to [OBJECT B]",
        "Put [OBJECT A] by [OBJECT B]",
        "Place [OBJECT A] near [OBJECT B]",
        "Set [OBJECT A] next to [OBJECT B]",
        "Can you move [OBJECT A] near [OBJECT B]",
        "Shift [OBJECT A] near [OBJECT B]",
        "Let's move [OBJECT A] near [OBJECT B]",
    ],
    "put-on": [
        "place [OBJECT A] on [OBJECT B]",
        "set [OBJECT A] on [OBJECT B]",
        "move [OBJECT A] onto [OBJECT B]",
        "position [OBJECT A] on [OBJECT B]",
        "put [OBJECT A] onto [OBJECT B]",
        "could you put [OBJECT A] on [OBJECT B]",
        "let's put [OBJECT A] on [OBJECT B]",
        "please place [OBJECT A] on [OBJECT B]",
        "can you place [OBJECT A] on [OBJECT B]",
        "would you move [OBJECT A] onto [OBJECT B]",
    ],
    "put-in": [
        "take [OBJECT] into the yellow basket",
        "bring [OBJECT] into the yellow basket",
        "place [OBJECT] in the yellow basket",
        "move [OBJECT] inside the yellow basket",
        "put [OBJECT] inside the yellow basket",
        "drop [OBJECT] into the yellow basket",
        "insert [OBJECT] into the yellow basket",
        "can you put [OBJECT] into the yellow basket",
        "please put [OBJECT] into the yellow basket",
        "let's put [OBJECT] into the yellow basket",
    ],
}


def _get_instruction_obj_name(name: str) -> str:
    parts = name.split("_")
    rm_list = {
        "opened",
        "light",
        "generated",
        "modified",
        "objaverse",
        "bridge",
        "baked",
        "v2",
    }
    cleaned = []
    for word in parts:
        if word.endswith("cm"):
            continue
        if word not in rm_list:
            cleaned.append(word)
    return " ".join(cleaned)


def _random_xy(rng: np.random.RandomState, xy_range: Tuple[float, float]) -> List[float]:
    return [float(rng.uniform(*xy_range)), float(rng.uniform(*xy_range))]


def _random_z_quat(rng: np.random.RandomState) -> List[float]:
    yaw = float(rng.uniform(-np.pi, np.pi))
    return [float(np.cos(yaw / 2.0)), 0.0, 0.0, float(np.sin(yaw / 2.0))]


class Variation:
    def __init__(
        self,
        seed: Optional[int] = None,
        camera_base: Optional[str] = None,
        lighting_direction: Optional[str] = None,
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.camera_fuzzer = RandomCamera(camera_base) if camera_base else None
        self.lighting_fuzzer = RandomLighting(lighting_direction)

    def generate(
        self,
        options: Dict[str, Any],
        background_choices: Optional[Iterable[str]] = None,
        occlusion_choices: Optional[Iterable[Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a new test combo by changing background, lighting, occlusion, and camera."""
        mutated = copy.deepcopy(options)

        lighting_options = self.lighting_fuzzer.generate_options()
        mutated.update(lighting_options)

        if self.camera_fuzzer:
            camera_options = self.camera_fuzzer.generate_options()
            mutated.update(camera_options)

        if background_choices:
            bg = self.py_rng.choice(list(background_choices))
            key = self._choose_existing_key(mutated, ["background", "background_id", "scene_id", "scene_name"])
            mutated[key] = bg

        if occlusion_choices:
            occ = self.py_rng.choice(list(occlusion_choices))
            key = self._choose_existing_key(mutated, ["occlusion_cfgs", "occlusion", "occluder_cfgs"])
            mutated[key] = occ

        return mutated

    def crossover(self, parent_a: Dict[str, Any], parent_b: Dict[str, Any]) -> Dict[str, Any]:
        """Fuse two test combos into a new one."""
        return self._crossover_dict(parent_a, parent_b)

    def expand(
        self,
        options: Dict[str, Any],
        distractor_pool: Iterable[str],
        num: int = 1,
        xy_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> Dict[str, Any]:
        """Add distractor objects to expand a test combo."""
        mutated = copy.deepcopy(options)
        existing = set(mutated.get("distractor_model_ids", []))
        choices = [obj for obj in distractor_pool if obj not in existing]
        if not choices or num <= 0:
            return mutated

        selected = self.py_rng.sample(choices, min(num, len(choices)))
        mutated.setdefault("distractor_model_ids", [])
        mutated.setdefault("distractor_obj_init_options", {})
        for obj in selected:
            mutated["distractor_model_ids"].append(obj)
            mutated["distractor_obj_init_options"][obj] = {
                "init_xy": _random_xy(self.rng, xy_range),
                "init_rot_quat": _random_z_quat(self.rng),
            }
        return mutated

    def rephrase(self, task: str, options: Dict[str, Any]) -> Dict[str, Any]:
        """Rewrite task instruction while preserving the semantics."""
        mutated = copy.deepcopy(options)
        if task not in _TEMPLATES:
            return mutated

        try:
            if task == "grasp":
                obj = _get_instruction_obj_name(options["model_id"])
                instruction = self.py_rng.choice(_TEMPLATES[task]).replace("[OBJECT]", obj)
            elif task == "move":
                ids = options["model_ids"]
                obj_a = _get_instruction_obj_name(ids[options["source_obj_id"]])
                obj_b = _get_instruction_obj_name(ids[options["target_obj_id"]])
                instruction = (
                    self.py_rng.choice(_TEMPLATES[task])
                    .replace("[OBJECT A]", obj_a)
                    .replace("[OBJECT B]", obj_b)
                )
            elif task == "put-on":
                obj_a = _get_instruction_obj_name(options["model_ids"][0])
                obj_b = _get_instruction_obj_name(options["model_ids"][1])
                instruction = (
                    self.py_rng.choice(_TEMPLATES[task])
                    .replace("[OBJECT A]", obj_a)
                    .replace("[OBJECT B]", obj_b)
                )
            elif task == "put-in":
                obj = _get_instruction_obj_name(options["model_ids"][0])
                instruction = self.py_rng.choice(_TEMPLATES[task]).replace("[OBJECT]", obj)
            else:
                return mutated
        except (KeyError, IndexError, TypeError):
            return mutated

        mutated["task_instruction"] = instruction
        return mutated

    def reset(
        self,
        options: Dict[str, Any],
        xy_range: Tuple[float, float] = (-0.5, 0.5),
    ) -> Dict[str, Any]:
        """Rearrange object positions in a test combo."""
        mutated = copy.deepcopy(options)

        obj_opts = mutated.get("obj_init_options")
        if isinstance(obj_opts, dict):
            if "init_xy" in obj_opts:
                obj_opts["init_xy"] = _random_xy(self.rng, xy_range)
            else:
                for name, cfg in obj_opts.items():
                    if isinstance(cfg, dict):
                        cfg["init_xy"] = _random_xy(self.rng, xy_range)
                        obj_opts[name] = cfg
            mutated["obj_init_options"] = obj_opts

        if "distractor_obj_init_options" in mutated:
            for name, cfg in mutated["distractor_obj_init_options"].items():
                if isinstance(cfg, dict):
                    cfg["init_xy"] = _random_xy(self.rng, xy_range)
                    mutated["distractor_obj_init_options"][name] = cfg

        return mutated

    def _crossover_dict(self, a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        merged: Dict[str, Any] = {}
        keys = set(a.keys()) | set(b.keys())
        for key in keys:
            if key in a and key in b:
                va, vb = a[key], b[key]
                if isinstance(va, dict) and isinstance(vb, dict):
                    merged[key] = self._crossover_dict(va, vb)
                else:
                    merged[key] = copy.deepcopy(self.py_rng.choice([va, vb]))
            else:
                merged[key] = copy.deepcopy(a.get(key, b.get(key)))
        return merged

    @staticmethod
    def _choose_existing_key(options: Dict[str, Any], candidates: List[str]) -> str:
        for key in candidates:
            if key in options:
                return key
        return candidates[0]
