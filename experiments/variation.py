import copy
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

try:
    from experiments.random_camera import RandomCamera
    from experiments.random_lighting import RandomLighting
except ModuleNotFoundError:
    # Allow running this file directly from the experiments directory.
    from random_camera import RandomCamera
    from random_lighting import RandomLighting

_COLOR_BACKGROUNDS = {
    "red": (255, 0, 0),
    "blue": (0, 0, 255),
    "green": (0, 255, 0),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "brown": (165, 42, 42),
    "gray": (128, 128, 128),
    "black": (0, 0, 0),
    "white": (255, 255, 255),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
    "olive": (128, 128, 0),
    "maroon": (128, 0, 0),
    "gold": (255, 215, 0),
    "lime": (191, 255, 0),
    "sky": (135, 206, 235),
}

_COLOR_BG_DIR = Path(__file__).parent / "assets" / "color_backgrounds"
_COLOR_OCCLUSION_DIR = Path(__file__).parent / "assets" / "color_occlusions"


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

def _ensure_color_patch(path: Path, rgb: Tuple[int, int, int], size: int = 128) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = np.zeros((size, size, 3), dtype=np.uint8)
    bgr[:, :, 0] = rgb[2]
    bgr[:, :, 1] = rgb[1]
    bgr[:, :, 2] = rgb[0]
    cv2.imwrite(str(path), bgr)


class Variation:
    def __init__(
        self,
        seed: Optional[int] = None,
        camera_base: Optional[str] = None,
        lighting_direction: Optional[str] = None,
        model_pool: Optional[Iterable[str]] = None,
        distractor_count_range: Tuple[int, int] = (1, 3),
        robot_init_xy_range: Tuple[float, float] = (0.30, 0.45),
        robot_init_yaw_range: Tuple[float, float] = (-np.pi, np.pi),
        distractor_xy_offset_range: Tuple[float, float] = (-0.3, 0.3),
        target_xy_offset_range: Tuple[float, float] = (-0.15, 0.15),
    ) -> None:
        self.rng = np.random.RandomState(seed)
        self.py_rng = random.Random(seed)
        self.camera_fuzzer = RandomCamera(camera_base) if camera_base else None
        self.lighting_fuzzer = RandomLighting(lighting_direction)
        self.model_pool = list(model_pool) if model_pool else None
        self.distractor_count_range = distractor_count_range
        self.robot_init_xy_range = robot_init_xy_range
        self.robot_init_yaw_range = robot_init_yaw_range
        self.distractor_xy_offset_range = distractor_xy_offset_range
        self.target_xy_offset_range = target_xy_offset_range

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

        choices = list(background_choices) if background_choices else list(_COLOR_BACKGROUNDS.keys())
        if choices and all(isinstance(c, str) for c in choices) and all(c in _COLOR_BACKGROUNDS for c in choices):
            color_name = self.py_rng.choice(choices)
            rgb = _COLOR_BACKGROUNDS[color_name]
            bg_path = self._get_color_background_path(color_name, rgb, mutated)
            mutated["rgb_overlay_path"] = str(bg_path)
            mutated.setdefault("rgb_overlay_cameras", ["overhead_camera"])
            camera_cfgs = mutated.get("camera_cfgs")
            if not isinstance(camera_cfgs, dict):
                camera_cfgs = {}
            camera_cfgs["add_segmentation"] = True
            mutated["camera_cfgs"] = camera_cfgs
        elif choices:
            bg = self.py_rng.choice(choices)
            key = self._choose_existing_key(mutated, ["background", "background_id", "scene_id", "scene_name"])
            mutated[key] = bg

        occ_color = self.py_rng.choice(list(_COLOR_BACKGROUNDS.keys()))
        rgb = _COLOR_BACKGROUNDS[occ_color]
        occ_path = _COLOR_OCCLUSION_DIR / f"{occ_color}.png"
        _ensure_color_patch(occ_path, rgb)
        mutated["occlusion_cfgs"] = {
            "paths": [str(occ_path)],
            "alpha": 1.0,
            "scale_range": [0.2, 0.45],
            "position": "random",
            "cameras": ["overhead_camera"],
        }

        self._randomize_robot(mutated)
        self._randomize_objects(mutated)

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

    @staticmethod
    def _get_color_background_path(color_name: str, rgb: Tuple[int, int, int], options: Dict[str, Any]) -> Path:
        _COLOR_BG_DIR.mkdir(parents=True, exist_ok=True)
        camera_cfgs = options.get("camera_cfgs", {})
        width = int(camera_cfgs.get("width", 256)) if isinstance(camera_cfgs, dict) else 256
        height = int(camera_cfgs.get("height", 256)) if isinstance(camera_cfgs, dict) else 256
        path = _COLOR_BG_DIR / f"{color_name}_{width}x{height}.png"
        if not path.exists():
            bgr = np.zeros((height, width, 3), dtype=np.uint8)
            bgr[:, :, 0] = rgb[2]
            bgr[:, :, 1] = rgb[1]
            bgr[:, :, 2] = rgb[0]
            cv2.imwrite(str(path), bgr)
        return path

    def _randomize_objects(self, options: Dict[str, Any]) -> None:
        if not any(k in options for k in ("model_id", "distractor_model_ids", "distractor_obj_init_options")):
            return

        pool = self._get_model_pool(options)
        if not pool:
            return

        model_id = self.py_rng.choice(pool)
        options["model_id"] = model_id

        robot_xy = None
        robot_opts = options.get("robot_init_options")
        if isinstance(robot_opts, dict):
            robot_xy = robot_opts.get("init_xy")

        if robot_xy is not None and len(robot_xy) == 2:
            dx = float(self.rng.uniform(*self.target_xy_offset_range))
            dy = float(self.rng.uniform(*self.target_xy_offset_range))
            init_xy = [float(robot_xy[0]) + dx, float(robot_xy[1]) + dy]
        else:
            init_xy = _random_xy(self.rng, (-0.2, 0.2))
        options["obj_init_options"] = {
            "init_xy": init_xy,
            "orientation": _random_z_quat(self.rng),
        }

        pool_no_target = [m for m in pool if m != model_id]
        if not pool_no_target:
            pool_no_target = pool

        min_k, max_k = self.distractor_count_range
        if max_k < min_k:
            min_k, max_k = max_k, min_k
        max_k = min(max_k, len(pool_no_target))
        if max_k < min_k:
            min_k = max_k
        num = self.py_rng.randint(min_k, max_k + 1) if max_k > 0 else 0

        if num > 0:
            if len(pool_no_target) >= num:
                distractors = self.py_rng.sample(pool_no_target, num)
            else:
                distractors = [self.py_rng.choice(pool_no_target) for _ in range(num)]
        else:
            distractors = []

        options["distractor_model_ids"] = distractors
        options["distractor_obj_init_options"] = {}
        for obj in distractors:
            if robot_xy is not None and len(robot_xy) == 2:
                dx = float(self.rng.uniform(*self.distractor_xy_offset_range))
                dy = float(self.rng.uniform(*self.distractor_xy_offset_range))
                init_xy = [float(robot_xy[0]) + dx, float(robot_xy[1]) + dy]
            else:
                init_xy = _random_xy(self.rng, (-0.5, 0.5))
            options["distractor_obj_init_options"][obj] = {
                "init_xy": init_xy,
                "init_rot_quat": _random_z_quat(self.rng),
            }

    def _get_model_pool(self, options: Dict[str, Any]) -> List[str]:
        if self.model_pool:
            return self.model_pool
        pool = options.get("model_pool")
        if isinstance(pool, (list, tuple)) and pool:
            return list(pool)

        model_json = options.get("model_json") or options.get("model_json_path")
        if isinstance(model_json, str):
            model_path = Path(model_json)
            if model_path.exists():
                return self._load_model_json(model_path)

        default_path = Path(__file__).parent.parent / "ManiSkill2_real2sim" / "data" / "ycb-dataset" / "info_ycb.json"
        if default_path.exists():
            return self._load_model_json(default_path)

        return []

    @staticmethod
    def _load_model_json(path: Path) -> List[str]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            return list(data.keys())
        except (OSError, json.JSONDecodeError):
            return []

    def _randomize_robot(self, options: Dict[str, Any]) -> None:
        robot_opts = options.get("robot_init_options")
        if not isinstance(robot_opts, dict):
            robot_opts = {}

        init_xy = _random_xy(self.rng, self.robot_init_xy_range)

        yaw = float(self.rng.uniform(*self.robot_init_yaw_range))
        init_rot_quat = [
            float(np.cos(yaw / 2.0)),
            0.0,
            0.0,
            float(np.sin(yaw / 2.0)),
        ]

        robot_opts["init_xy"] = init_xy
        robot_opts["init_rot_quat"] = init_rot_quat
        options["robot_init_options"] = robot_opts

if __name__ == "__main__":
    variation = Variation(seed=123, camera_base=None)
    base_options = {
        "model_id": "placeholder",
        "camera_cfgs": {"width": 256, "height": 256},
    }
    result = variation.generate(base_options)
    print(json.dumps(result, indent=2, sort_keys=True))
