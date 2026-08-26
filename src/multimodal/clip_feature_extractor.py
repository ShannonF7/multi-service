try:
    # Prefer open_clip for performance
    import open_clip
    _BACKEND = "open_clip"
except Exception:
    try:
        from transformers import CLIPProcessor, CLIPModel
        _BACKEND = "transformers"
    except Exception:
        _BACKEND = None

from PIL import Image
import torch
import os
import numpy as np


class ClipFeatureExtractor:
    def __init__(self, model_name: str = "ViT-B-32", device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if _BACKEND == "open_clip":
            # model name mapping for open_clip: (model, pretrained)
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s34b_b79k')
            self.model.to(self.device)
            self.model.eval()
            self._backend = 'open_clip'
        elif _BACKEND == "transformers":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            self.model.eval()
            self._backend = 'transformers'
        else:
            raise RuntimeError("No CLIP backend available. Install 'open_clip_torch' or 'transformers'.")

    def extract(self, image_input) -> list:
        """Accepts PIL Image or image path. Returns normalized 512-d vector (list[float])."""
        try:
            if isinstance(image_input, str):
                img = Image.open(image_input).convert('RGB')
            else:
                img = image_input.convert('RGB')

            if self._backend == 'open_clip':
                x = self.preprocess(img).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    features = self.model.encode_image(x)
            else:
                inputs = self.processor(images=img, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = self.model.get_image_features(**inputs)
                    features = outputs

            features = features / features.norm(dim=-1, keepdim=True)
            vec = features.cpu().numpy().flatten().tolist()
            return vec
        except Exception as e:
            raise


_instance = None


def get_clip_extractor(model_name: str = "ViT-B-32"):
    global _instance
    if _instance is None:
        _instance = ClipFeatureExtractor(model_name=model_name)
    return _instance
