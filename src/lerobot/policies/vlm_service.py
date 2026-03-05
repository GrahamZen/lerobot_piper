import json
import logging
import os
import re
import time
from datetime import datetime
from typing import Any

import torch
from google import genai
from google.genai import types
from PIL import Image
from torchvision.transforms import ToPILImage

logger = logging.getLogger(__name__)


class VLMService:
    def __init__(
        self,
        video_path: str,
        api_key: str = None,
        model_name: str = "gemini-2.5-pro",
    ):
        """
        Initialize the VLM service, upload the demo video, and create a chat session with memory.
        """
        # If api_key is not provided, read it from environment variables.
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        masked_key = "None" if not self.api_key else f"{self.api_key[:6]}...{self.api_key[-4:]}"
        self.model_name = model_name
        self.last_message: dict[str, Any] | None = None
        self.debug_history: list[dict[str, Any]] = []
        self.debug_session_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_session_dirname = f"vlm_debug_{self.debug_session_timestamp}"
        self.debug_save_dir = "logs"
        self.debug_session_path: str | None = None
        self.saved_record_count = 0
        self.log_dir = os.path.join(self.debug_save_dir, "vlm_service", self.debug_session_dirname)
        self.log_file_path = os.path.join(self.log_dir, "vlm_service.log")
        self.logger = self._setup_logger(model_name=model_name, video_path=video_path, masked_key=masked_key)
        self.to_pil = ToPILImage()
        if self.api_key is None:
            self.logger.warning("API key is empty.")
            self.client = None
            return
        self.client = genai.Client(api_key=self.api_key)

        self.logger.info("Uploading reference video: %s...", video_path)
        self.logger.info("[VLMService.__init__] Uploading reference video...")
        self.video_file = self.client.files.upload(file=video_path)
        self.logger.info(
            "[VLMService.__init__] Upload submitted | file_name=%s | state=%s",
            self.video_file.name,
            self.video_file.state.name,
        )

        # Poll until video processing is complete (Gemini needs a few seconds to process frames in the cloud).
        poll_count = 0
        while self.video_file.state.name == "PROCESSING":
            self.logger.info("Waiting for video processing...")
            poll_count += 1
            self.logger.info("[VLMService.__init__] Polling video state... attempt=%d", poll_count)
            time.sleep(3)
            self.video_file = self.client.files.get(name=self.video_file.name)
            self.logger.info("[VLMService.__init__] Video state=%s", self.video_file.state.name)

        if self.video_file.state.name == "FAILED":
            self.logger.error("[VLMService.__init__] Video processing failed.")
            raise RuntimeError("Video processing failed on Gemini servers.")
        self.logger.info("Video ready!")
        self.logger.info("[VLMService.__init__] Video ready.")

        # Create a clean chat session without using system_instruction.
        self.chat = self.client.chats.create(
            model=self.model_name, config=types.GenerateContentConfig(temperature=0.0)
        )
        self.logger.info("[VLMService.__init__] Chat session created.")

        initial_prompt = """
    This is a task demonstration video for a robot. Please carefully observe the standard workflow. Next, I will send you a stitched three-view image at a failure timestamp (from left to right: left arm view, top camera view, right arm view), plus several stitched three-view images corresponding to candidate rollback timestamps in chronological order.

    You need to choose a safe and reasonable rollback state and provide a detailed analysis.

    [Core rollback principle: safety over efficiency]
    1. Complete subtask boundary: a complete subtask includes "start from standby position -> approach target -> execute action (grasp/place) -> retract to standby position".
    2. When a subtask fails (for example, left-arm grasping), you must roll back to the state **before that entire subtask started**.
    3. In other words, the robot arms should be in a fully idle, retracted, or globally standby "absolutely safe position" (for example, right after the previous task has fully completed, and before the new arm starts moving toward the target).
    4. You must **never** choose an intermediate preparation state where an arm has already moved over the target and is about to descend, because that lacks safe replanning space. It is better to sacrifice some efficiency and travel farther to ensure no interference.

    [Important output format constraints]
    To ensure accuracy, you must respond in the following order:
    1. Step one: first provide a detailed comparison between the current failure state and each candidate rollback point, analyzing both robot arms' positions and task completion status.
    2. Step two: at the end of your analysis, wrap your final selected index with <FINAL_ANSWER> tags (for example: <FINAL_ANSWER>3</FINAL_ANSWER>).

    If you understand the task, reply only with: "I understand. Please send the images."
"""

        # Send the video to the model as initial memory.
        warmup_response = self.chat.send_message([self.video_file, initial_prompt])
        warmup_text = (warmup_response.text or "").strip() if hasattr(warmup_response, "text") else ""
        self.logger.info("[VLMService.__init__] Warmup reply: %s", warmup_text)

    def _setup_logger(self, model_name: str, video_path: str, masked_key: str) -> logging.Logger:
        os.makedirs(self.log_dir, exist_ok=True)
        logger_name = f"{__name__}.{id(self)}"
        instance_logger = logging.getLogger(logger_name)
        instance_logger.setLevel(logging.INFO)
        instance_logger.propagate = False

        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(formatter)

        file_handler = logging.FileHandler(self.log_file_path, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)

        instance_logger.handlers.clear()
        instance_logger.addHandler(stream_handler)
        instance_logger.addHandler(file_handler)

        instance_logger.info(
            "[VLMService.__init__] Start | model=%s | video_path=%s | api_key=%s",
            model_name,
            video_path,
            masked_key,
        )
        instance_logger.info("[VLMService.__init__] Log file=%s", self.log_file_path)
        return instance_logger

    def _stitch_three_views(
        self, left_tensor: torch.Tensor, top_tensor: torch.Tensor, right_tensor: torch.Tensor
    ) -> Image.Image:
        """
        Stitch three view tensors into one PIL image (left to right: left, top, right).
        Expected input tensor shape is (C, H, W), with optional leading batch dim (1, C, H, W).
        """

        def _prepare_view_tensor(view_tensor: torch.Tensor, view_name: str) -> torch.Tensor:
            if view_tensor.dim() == 4:
                if view_tensor.size(0) != 1:
                    logger.warning(
                        "%s has batch size %d; using the first frame for stitching.",
                        view_name,
                        view_tensor.size(0),
                    )
                view_tensor = view_tensor[0]
            if view_tensor.dim() != 3:
                raise ValueError(
                    f"{view_name} must have shape (C, H, W) or (B, C, H, W), got {tuple(view_tensor.shape)}"
                )

            if view_tensor.is_floating_point():
                if view_tensor.min() < 0:
                    mean = torch.tensor([0.485, 0.456, 0.406], device=view_tensor.device).view(3, 1, 1)
                    std = torch.tensor([0.229, 0.224, 0.225], device=view_tensor.device).view(3, 1, 1)
                    view_tensor = view_tensor * std + mean

                view_tensor = torch.clamp(view_tensor, 0.0, 1.0)

            return view_tensor

        left_tensor = _prepare_view_tensor(left_tensor, "left_tensor")
        top_tensor = _prepare_view_tensor(top_tensor, "top_tensor")
        right_tensor = _prepare_view_tensor(right_tensor, "right_tensor")

        if (
            left_tensor.shape[-2:] != top_tensor.shape[-2:]
            or left_tensor.shape[-2:] != right_tensor.shape[-2:]
        ):
            raise ValueError(
                "All view tensors must have identical spatial size. "
                f"Got left={tuple(left_tensor.shape)}, top={tuple(top_tensor.shape)}, right={tuple(right_tensor.shape)}"
            )

        img_left = self.to_pil(left_tensor)
        img_top = self.to_pil(top_tensor)
        img_right = self.to_pil(right_tensor)

        w, h = img_left.size
        # Create a canvas with width=3x and height=1x.
        composite = Image.new("RGB", (w * 3, h))
        composite.paste(img_left, (0, 0))
        composite.paste(img_top, (w, 0))
        composite.paste(img_right, (w * 2, 0))

        # Resize to reduce token usage and upload latency.
        composite = composite.resize((max(1, w * 3 // 2), max(1, h // 2)))
        return composite

    def _get_three_views(
        self,
        views: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        required_view_keys = (
            "observation.images.left",
            "observation.images.middle",
            "observation.images.right",
        )
        missing_keys = [k for k in required_view_keys if k not in views]
        if missing_keys:
            self.logger.warning(
                "[VLMService.select_checkpoint_index] Missing view keys: %s | available=%s",
                missing_keys,
                list(views.keys())[:8],
            )
            return None
        return (
            views["observation.images.left"],
            views["observation.images.middle"],
            views["observation.images.right"],
        )

    def _build_message_contents(
        self,
        batch: dict[str, torch.Tensor],
        checkpoint_queue: list[tuple[Any, ...]],
        episode: int | None = None,
        step: int | None = None,
    ) -> tuple[list[Any] | None, list[int]]:
        current_views = self._get_three_views(batch)
        if current_views is None:
            return None, []

        current_img = self._stitch_three_views(*current_views)
        self.logger.info(
            "[VLMService.select_checkpoint_index] Current stitched image size=%s", current_img.size
        )

        message_contents: list[Any] = [
            "This is the stitched three-view image at the failure timestamp:",
            current_img,
        ]

        valid_candidate_indices: list[int] = []
        for idx, queue_entry in enumerate(checkpoint_queue):
            if len(queue_entry) < 3 or not isinstance(queue_entry[2], dict):
                self.logger.info(
                    "[VLMService.select_checkpoint_index] Skip queue entry idx=%d | reason=legacy_or_invalid_format",
                    idx,
                )
                continue

            checkpoint_views = queue_entry[2]
            three_views = self._get_three_views(checkpoint_views)
            if three_views is None:
                self.logger.info(
                    "[VLMService.select_checkpoint_index] Skip queue entry idx=%d | reason=missing_views",
                    idx,
                )
                continue

            step_num = queue_entry[0] if len(queue_entry) > 0 else "unknown"
            chkpt_img = self._stitch_three_views(*three_views)
            message_contents.append(
                f"This is candidate rollback point #{idx} (corresponding to step={step_num}):"
            )
            message_contents.append(chkpt_img)
            valid_candidate_indices.append(idx)
            self.logger.info(
                "[VLMService.select_checkpoint_index] Added candidate idx=%d | stitched_size=%s",
                idx,
                chkpt_img.size,
            )

        message_contents.append(
            "Please fully analyze the current failure state and all candidate rollback points first, "
            "then output <FINAL_ANSWER>index</FINAL_ANSWER> on the last line."
        )
        return message_contents, valid_candidate_indices

    def _create_debug_record(
        self,
        message_contents: list[Any],
        episode: int | None = None,
        step: int | None = None,
    ) -> dict[str, Any]:
        cloned_parts: list[Any] = []
        for part in message_contents:
            if isinstance(part, Image.Image):
                cloned_parts.append(part.copy())
            else:
                cloned_parts.append(part)

        record: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "episode": episode,
            "step": step,
            "request_parts": cloned_parts,
            "response_text": None,
            "selected_index": None,
            "error": None,
        }
        self.last_message = record
        self.debug_history.append(record)
        return record

    def save_debug_history(self, output_dir: str | None = None) -> str:
        """
        Persist VLM debug history to a session-stable folder.

        The first call decides the base output directory. All later calls in the same
        VLMService instance reuse the same session folder and only append new records.

        Each request is saved under a dedicated record subfolder with:
        - text parts as .txt
        - image parts as .png
        - response text
        - metadata json
        """
        if output_dir is None:
            output_dir = self.debug_save_dir

        if self.debug_session_path is None:
            os.makedirs(output_dir, exist_ok=True)
            self.debug_session_path = os.path.join(output_dir, self.debug_session_dirname)

        os.makedirs(self.debug_session_path, exist_ok=True)

        if self.saved_record_count > len(self.debug_history):
            self.saved_record_count = 0

        manifest: dict[str, Any] = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "model_name": self.model_name,
            "session_timestamp": self.debug_session_timestamp,
            "session_dirname": self.debug_session_dirname,
            "total_records": len(self.debug_history),
            "records": [],
        }

        for record_index in range(self.saved_record_count, len(self.debug_history)):
            record = self.debug_history[record_index]
            record_dir_name = f"record_{record_index:04d}"
            record_dir = os.path.join(self.debug_session_path, record_dir_name)
            os.makedirs(record_dir, exist_ok=True)

            part_items: list[dict[str, Any]] = []
            request_parts = record.get("request_parts", [])
            for part_index, part in enumerate(request_parts):
                if isinstance(part, Image.Image):
                    file_name = f"part_{part_index:03d}.png"
                    part.save(os.path.join(record_dir, file_name))
                    part_items.append(
                        {
                            "index": part_index,
                            "type": "image",
                            "file": file_name,
                            "size": list(part.size),
                            "mode": part.mode,
                        }
                    )
                else:
                    file_name = f"part_{part_index:03d}.txt"
                    with open(os.path.join(record_dir, file_name), "w", encoding="utf-8") as f:
                        f.write(str(part))
                    part_items.append(
                        {
                            "index": part_index,
                            "type": "text",
                            "file": file_name,
                        }
                    )

            response_text = record.get("response_text")
            if response_text is not None:
                with open(os.path.join(record_dir, "response.txt"), "w", encoding="utf-8") as f:
                    f.write(str(response_text))

            record_meta = {
                "timestamp": record.get("timestamp"),
                "episode": record.get("episode"),
                "step": record.get("step"),
                "selected_index": record.get("selected_index"),
                "error": record.get("error"),
                "request_part_count": len(request_parts),
                "request_parts": part_items,
                "has_response": response_text is not None,
            }
            with open(os.path.join(record_dir, "meta.json"), "w", encoding="utf-8") as f:
                json.dump(record_meta, f, ensure_ascii=False, indent=2)

        self.saved_record_count = len(self.debug_history)

        for record_index in range(len(self.debug_history)):
            record_dir_name = f"record_{record_index:04d}"
            meta_name = "meta.json"
            manifest["records"].append(
                {
                    "index": record_index,
                    "dir": record_dir_name,
                    "meta": meta_name,
                }
            )

        with open(os.path.join(self.debug_session_path, "manifest.json"), "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

        self.logger.info(
            "[VLMService.save_debug_history] Session dir=%s | total_records=%d",
            self.debug_session_path,
            len(self.debug_history),
        )
        return self.debug_session_path

    def _parse_selected_index(self, reply: str, checkpoint_indices: list[int]) -> int:
        final_answer_match = re.search(
            r"<FINAL_ANSWER>\s*(-?\d+)\s*</FINAL_ANSWER>", reply, flags=re.IGNORECASE
        )
        if final_answer_match:
            selected_index = int(final_answer_match.group(1))
            parse_source = "final_answer_tag"
        else:
            first_line = reply.split("\n", 1)[0] if reply else ""
            first_line_match = re.search(r"-?\d+", first_line)
            if first_line_match:
                selected_index = int(first_line_match.group())
                parse_source = "first_line"
            else:
                global_match = re.search(r"-?\d+", reply)
                selected_index = int(global_match.group()) if global_match else checkpoint_indices[0]
                parse_source = "global_fallback" if global_match else "default_index"

        self.logger.info(
            "[VLMService.select_checkpoint_index] Parsed source=%s | parsed_index=%d",
            parse_source,
            selected_index,
        )

        if selected_index not in checkpoint_indices:
            self.logger.warning(
                "VLM selected invalid index %d. Defaulting to %d.",
                selected_index,
                checkpoint_indices[0],
            )
            self.logger.warning(
                "[VLMService.select_checkpoint_index] Parsed index %d invalid. Fallback to %d.",
                selected_index,
                checkpoint_indices[0],
            )
            selected_index = checkpoint_indices[0]

        return selected_index

    def select_checkpoint_index(
        self,
        batch: dict[str, torch.Tensor],
        checkpoint_queue: list[tuple[Any, ...]],
        episode: int | None = None,
        step: int | None = None,
    ) -> int:
        """
        Call the VLM for failure analysis and return selected checkpoint queue index.

        Args:
            batch: Current inference batch containing three camera views.
            checkpoint_queue: Raw checkpoint queue entries from FailureMetrics.
                Expected entry format is `(checkpoint_step, action, checkpoint_views)`.
                Legacy two-field entries are also accepted.
            episode: Optional episode id for logging/context.
            step: Optional process step id for logging/context.
        Returns:
            Selected checkpoint index in the checkpoint queue.
        """
        self.logger.info(
            "[VLMService.select_checkpoint_index] Called | episode=%s | step=%s | queue_len=%d",
            episode,
            step,
            len(checkpoint_queue),
        )

        if not checkpoint_queue:
            self.logger.info("[VLMService.select_checkpoint_index] Empty checkpoint queue. Return 0.")
            return 0

        checkpoint_indices = list(range(len(checkpoint_queue)))
        self.logger.info("[VLMService.select_checkpoint_index] Candidate indices=%s", checkpoint_indices)

        message_contents, valid_candidate_indices = self._build_message_contents(
            batch=batch,
            checkpoint_queue=checkpoint_queue,
            episode=episode,
            step=step,
        )
        if message_contents is None:
            self.logger.warning(
                "Missing current observation views in batch; fallback to the first checkpoint."
            )
            self.logger.warning(
                "[VLMService.select_checkpoint_index] Invalid current batch views. Fallback to index 0."
            )
            return checkpoint_indices[0]

        candidate_count = len(valid_candidate_indices)
        if candidate_count == 0:
            self.logger.warning(
                "No candidate checkpoint views available for VLM; fallback to the first checkpoint."
            )
            self.logger.warning(
                "[VLMService.select_checkpoint_index] No valid candidates. Fallback to index 0."
            )
            return checkpoint_indices[0]

        if candidate_count < 5:
            self.logger.warning(
                "[VLMService.select_checkpoint_index] Fewer than 5 valid candidates: %d",
                candidate_count,
            )

        self.logger.info(
            "[VLMService.select_checkpoint_index] Sending request to VLM | message_parts=%d | valid_candidates=%d",
            len(message_contents),
            candidate_count,
        )

        debug_record = self._create_debug_record(
            message_contents=message_contents,
            episode=episode,
            step=step,
        )

        # 3. Send request.
        if self.client is None:
            self.logger.warning("VLM client is not initialized.")
            debug_record["error"] = "client_not_initialized"
            return checkpoint_indices[0]
        try:
            response = self.chat.send_message(message_contents)
            reply = response.text.strip()
            self.logger.info("[VLMService.select_checkpoint_index] Raw VLM reply:\n%s", reply)
            debug_record["response_text"] = reply

            selected_index = self._parse_selected_index(reply, valid_candidate_indices)
            debug_record["selected_index"] = selected_index

            self.logger.info("[VLMService.select_checkpoint_index] Final selected_index=%d", selected_index)
            return selected_index

        except Exception as e:
            self.logger.error("VLM call failed: %s", e)
            debug_record["error"] = str(e)
            self.logger.exception("[VLMService.select_checkpoint_index] Exception")
            self.logger.warning(
                "[VLMService.select_checkpoint_index] Fallback selected_index=%d",
                checkpoint_indices[0],
            )
            return checkpoint_indices[0]
