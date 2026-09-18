"""
backend/services/encoder.py
===========================
Phase 1.8: RemoteCLIP Encoder Service for Satellite Imagery
===========================================================
PS Sections: 2.2.4 (Multimodal & Semantic Image Retrieval)

Provides a singleton `RemoteCLIPEncoder` that loads RemoteCLIP ViT-B-32 weights
and generates 512-dimensional L2-normalized vector embeddings for text queries
and satellite tile image arrays / files.
"""

import io
import logging
import os
from typing import List, Union, Optional
import numpy as np
from PIL import Image
import torch

# Suppress initial open_clip random skeleton initialization warning
logging.getLogger("root").setLevel(logging.ERROR)
import open_clip

logger = logging.getLogger("RemoteCLIPEncoder")

MODEL_NAME = "ViT-B-32"
EMBEDDING_DIM = 512


class RemoteCLIPEncoder:
    """
    Singleton / Service class wrapper around fine-tuned RemoteCLIP ViT-B-32.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        if device is None:
            env_dev = os.getenv("DEVICE")
            if env_dev:
                self.device = env_dev
            else:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        if checkpoint_path is None:
            env_path = os.getenv("REMOTECLIP_CHECKPOINT_PATH")
            if env_path and os.path.exists(env_path):
                checkpoint_path = env_path
            else:
                # Check default project locations
                base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
                candidates = [
                    os.path.join(base_dir, "models", "retrieval", "RemoteCLIP-ViT-B-32.pt")
                ]
                for cand in candidates:
                    if os.path.exists(cand):
                        checkpoint_path = cand
                        break

                if checkpoint_path is None:
                    checkpoint_path = candidates[0]

        self.checkpoint_path = checkpoint_path
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"RemoteCLIP checkpoint not found at: {self.checkpoint_path}. Attempting download from Hugging Face (chendelong/RemoteCLIP)...")
            try:
                from huggingface_hub import hf_hub_download
                ckpt_dir = os.path.dirname(self.checkpoint_path)
                os.makedirs(ckpt_dir, exist_ok=True)
                downloaded = hf_hub_download(
                    repo_id="chendelong/RemoteCLIP",
                    filename="RemoteCLIP-ViT-B-32.pt",
                    local_dir=ckpt_dir,
                    local_dir_use_symlinks=False
                )
                self.checkpoint_path = downloaded
                logger.info(f"RemoteCLIP checkpoint successfully downloaded to: {self.checkpoint_path}")
            except Exception as e:
                logger.error(f"Failed to auto-download RemoteCLIP checkpoint: {e}")
                raise FileNotFoundError(
                    f"RemoteCLIP checkpoint not found at: {self.checkpoint_path} and could not be downloaded from Hugging Face: {e}"
                )

        logger.info(f"Initializing RemoteCLIP {MODEL_NAME} architecture on device '{self.device}'...")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(MODEL_NAME)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

        logger.info(f"Loading weights from {self.checkpoint_path}...")
        ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        self.model.load_state_dict(ckpt, strict=True)
        self.model = self.model.to(self.device).eval()
        logger.info("RemoteCLIP encoder initialized successfully!")

    def encode_text(
        self,
        text: Union[str, List[str]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode text prompt(s) into 512-dimensional L2-normalized vector embedding(s).
        """
        is_single = isinstance(text, str)
        prompts = [text] if is_single else text

        text_tokens = self.tokenizer(prompts).to(self.device)

        with torch.no_grad():
            text_features = self.model.encode_text(text_tokens)
            # L2 normalization
            text_features /= text_features.norm(dim=-1, keepdim=True)

        embeddings = text_features.cpu().numpy().tolist()

        if is_single:
            return embeddings[0]
        return embeddings

    def encode_image(
        self,
        image_input: Union[str, Image.Image, np.ndarray, bytes, io.BytesIO, List[Union[str, Image.Image, np.ndarray, bytes, io.BytesIO]]]
    ) -> Union[List[float], List[List[float]]]:
        """
        Encode satellite image(s) into 512-dimensional L2-normalized vector embedding(s).
        Supports file paths, PIL Images, numpy RGB arrays ((3, H, W) or (H, W, 3)), or lists.
        """
        is_single = not isinstance(image_input, list)
        items = [image_input] if is_single else image_input

        pil_images = []
        for item in items:
            pil_img = self._to_pil_image(item)
            pil_images.append(pil_img)

        # Apply OpenCLIP preprocessing
        preprocessed_tensors = [self.preprocess(img) for img in pil_images]
        batch_tensor = torch.stack(preprocessed_tensors).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(batch_tensor)
            # L2 normalization
            image_features /= image_features.norm(dim=-1, keepdim=True)

        embeddings = image_features.cpu().numpy().tolist()

        if is_single:
            return embeddings[0]
        return embeddings

    def _to_pil_image(self, item: Union[str, Image.Image, np.ndarray, bytes, io.BytesIO]) -> Image.Image:
        """Helper to convert various image input formats into a 3-channel RGB PIL Image."""
        if isinstance(item, Image.Image):
            return item.convert("RGB")
        elif isinstance(item, np.ndarray):
            # If shape is (3, H, W), transpose to (H, W, 3)
            if item.ndim == 3 and item.shape[0] == 3:
                arr = np.transpose(item, (1, 2, 0))
            else:
                arr = item
            if arr.dtype != np.uint8:
                # If float in [0, 1] or uint16
                if np.max(arr) <= 1.0:
                    arr = (arr * 255.0).astype(np.uint8)
                else:
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        elif isinstance(item, str):
            if not os.path.exists(item):
                raise FileNotFoundError(f"Image file not found at: {item}")
            return Image.open(item).convert("RGB")
        elif isinstance(item, (bytes, bytearray)):
            return Image.open(io.BytesIO(item)).convert("RGB")
        elif isinstance(item, io.BytesIO):
            item.seek(0)
            return Image.open(item).convert("RGB")
        else:
            raise TypeError(f"Unsupported image input type: {type(item)}")


# Application-wide shared singleton instance
_global_encoder: Optional[RemoteCLIPEncoder] = None


def get_encoder() -> RemoteCLIPEncoder:
    """
    Obtain shared singleton instance of RemoteCLIPEncoder.
    Loads model weights into memory once.
    """
    global _global_encoder
    if _global_encoder is None:
        _global_encoder = RemoteCLIPEncoder()
    return _global_encoder


# ============================================================
# Phase 2.1 — Query Encoding Top-Level Functions
# ============================================================

def encode_query_text(text: str) -> List[float]:
    """
    Encode an analyst natural language query text into a 512-dim L2-normalized vector
    with RemoteCLIP aerial prompt template ensembling for maximum cross-modal alignment.
    """
    encoder = get_encoder()
    clean_text = text.strip()
    if not clean_text:
        return [0.0] * EMBEDDING_DIM

    # Multi-template aerial prompt expansion (boosts cross-modal alignment)
    prompts = [
        clean_text,
        f"satellite imagery of {clean_text}",
        f"aerial view of {clean_text}",
        f"satellite photo showing {clean_text}"
    ]
    embs = encoder.encode_text(prompts)
    embs_arr = np.array(embs)
    avg_vec = np.mean(embs_arr, axis=0)
    norm = np.linalg.norm(avg_vec)
    if norm > 0:
        avg_vec = avg_vec / norm
    return avg_vec.tolist()


def encode_query_image(
    image_input: Union[str, Image.Image, np.ndarray, bytes, io.BytesIO]
) -> List[float]:
    """
    Encode a reference query satellite image (path, bytes, PIL, or numpy array)
    into a 512-dim L2-normalized vector. Reuses the application singleton encoder.
    """
    encoder = get_encoder()
    return encoder.encode_image(image_input)

