"""Adapter to local sam3 repository for text and prompt-guided mask prediction."""
from pathlib import Path
from typing import Optional
import logging
import numpy as np
import sys

from .types import PromptRegion


def _add_local_sam3_repo_to_path() -> None:
    repo_root = Path(__file__).resolve().parents[2] / "sam3"
    repo_root_str = str(repo_root)
    if repo_root.exists() and repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)


build_sam3_image_model = None
Sam3Processor = None


def _import_local_sam3():
    global build_sam3_image_model, Sam3Processor
    if build_sam3_image_model is not None and Sam3Processor is not None:
        return

    _add_local_sam3_repo_to_path()
    try:
        from sam3.model_builder import build_sam3_image_model as _build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor as _Sam3Processor
    except Exception as e:
        raise RuntimeError(f"Local sam3 package not available for import: {e}") from e

    build_sam3_image_model = _build_sam3_image_model
    Sam3Processor = _Sam3Processor


class SAM3Adapter:
    def __init__(
        self,
        checkpoint: Optional[str] = None,
        device: str = "cuda",
        enable_inst_interactivity: bool = True,
    ):
        self.checkpoint = checkpoint
        self.device = device
        self.model = None
        self.processor = None
        self.predictor = None
        self.enable_inst_interactivity = enable_inst_interactivity

    def load(self, bpe_path: Optional[str] = None) -> None:
        _import_local_sam3()
        logging.info("Building sam3 image model (device=%s)...", self.device)
        self.model = build_sam3_image_model(
            bpe_path=bpe_path,
            device=self.device,
            eval_mode=True,
            checkpoint_path=self.checkpoint,
            enable_inst_interactivity=self.enable_inst_interactivity,
        )
        self.processor = Sam3Processor(self.model, device=self.device)
        self.predictor = getattr(self.model, "inst_interactive_predictor", None)
        if self.predictor is None:
            logging.warning(
                "SAM3 interactive predictor not available; prompt-guided refinement may be limited"
            )

    def supports_text_prompt(self) -> bool:
        return self.model is not None and self.processor is not None

    def supports_box_prompt(self) -> bool:
        return (
            self.model is not None
            and hasattr(self.model, "predict_inst")
            and self.predictor is not None
        )

    def supports_point_prompt(self) -> bool:
        return self.supports_box_prompt()

    def _ensure_text_ready(self) -> None:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized")

    def _ensure_inst_ready(self, prompt_mode: str) -> bool:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not initialized")
        if not hasattr(self.model, "predict_inst") or self.predictor is None:
            logging.warning(
                "SAM3 local interface does not support %s prompts; skipping prompt-guided refinement",
                prompt_mode,
            )
            return False
        return True

    def _select_best_mask_and_score(
        self, masks, scores
    ) -> tuple[Optional[np.ndarray], Optional[float]]:
        masks_np = np.asarray(masks)
        scores_np = np.asarray(scores)
        if masks_np.size == 0 or scores_np.size == 0:
            return None, None

        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        elif masks_np.ndim == 2:
            masks_np = masks_np[None, ...]

        if scores_np.ndim > 1:
            scores_np = scores_np.reshape(-1)

        best_idx = int(np.argmax(scores_np))
        return masks_np[best_idx].astype(bool), float(scores_np[best_idx])

    def predict_with_text(self, pil_image, text_prompt: str):
        self._ensure_text_ready()
        inference_state = self.processor.set_image(pil_image)
        inference_state = self.processor.set_text_prompt(
            prompt=text_prompt, state=inference_state
        )

        masks = inference_state.get("masks")
        boxes = inference_state.get("boxes")
        scores = inference_state.get("scores")
        if masks is None:
            return (
                np.zeros((0, pil_image.size[1], pil_image.size[0]), dtype=bool),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32),
            )

        masks_np = masks.detach().cpu().numpy()
        if masks_np.ndim == 4 and masks_np.shape[1] == 1:
            masks_np = masks_np[:, 0]
        scores_np = (
            scores.detach().float().cpu().numpy()
            if scores is not None
            else np.zeros((len(masks_np),), dtype=np.float32)
        )
        boxes_np = (
            boxes.detach().float().cpu().numpy()
            if boxes is not None
            else np.zeros((0, 4), dtype=np.float32)
        )
        return masks_np.astype(bool), scores_np, boxes_np

    def refine_with_prompts(
        self, pil_image, prompt_regions: list[PromptRegion], prompt_mode: str
    ):
        if prompt_mode == "text":
            raise ValueError("Use predict_with_text for text prompt mode")
        if not self._ensure_inst_ready(prompt_mode):
            return []

        inference_state = self.processor.set_image(pil_image)
        results = []
        for prompt in prompt_regions:
            point_coords = None
            point_labels = None
            box = None

            if prompt_mode in {"box", "box_point"}:
                box = np.array([prompt.bbox_xyxy], dtype=np.float32)
                if not self.supports_box_prompt():
                    logging.warning(
                        "Box prompts are not supported by local SAM3 interface; skipping prompt %s",
                        prompt.prompt_id,
                    )
                    continue
            if prompt_mode in {"point", "box_point"}:
                point_coords = np.array([prompt.center_point_xy], dtype=np.float32)
                point_labels = np.array([1], dtype=np.int32)
                if not self.supports_point_prompt():
                    logging.warning(
                        "Point prompts are not supported by local SAM3 interface; skipping prompt %s",
                        prompt.prompt_id,
                    )
                    continue

            multimask_output = prompt_mode == "point"
            masks, scores, _ = self.model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                box=box,
                multimask_output=multimask_output,
            )
            best_mask, best_score = self._select_best_mask_and_score(masks, scores)
            if best_mask is None:
                continue

            results.append(
                {
                    "prompt": prompt,
                    "mask": best_mask,
                    "score": best_score,
                    "prompt_type": prompt_mode,
                }
            )

        return results
