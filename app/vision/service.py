import os
import logging
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import torch
import numpy as np
import rasterio

from .model import LightweightUNet

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FloodSegmentationService:
    """
    Production-style service for flood segmentation inference.
    Handles data preprocessing, model inference, and postprocessing.
    """
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the service, optionally loading a model from disk.
        """
        self.model_path = model_path
        self.model: Optional[LightweightUNet] = None
        self.dummy_mode = False
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        logger.info(f"Initializing FloodSegmentationService on device: {self.device}")
        self.load_model()

    def load_model(self):
        """
        Loads the PyTorch model into memory.
        """
        logger.info("Loading model...")
        if self.model_path and os.path.exists(self.model_path):
            try:
                checkpoint = torch.load(self.model_path, map_location=self.device)
                
                # Initialize model architecture
                self.model = LightweightUNet(n_channels=2, n_classes=1).to(self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.eval()
                
                logger.info(f"Model loaded from {self.model_path}")
            except Exception as e:
                logger.error(f"Failed to load model from {self.model_path}: {e}")
                raise
        else:
            logger.warning("No valid model path provided. Running in dummy mode.")
            self.dummy_mode = True

    def preprocess(self, s1_chip_path: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Reads the S1 chip from disk and preprocesses it for the model.
        
        Args:
            s1_chip_path (str): Path to the S1 .tif file.
            
        Returns:
            Tuple[torch.Tensor, Dict[str, Any]]: The preprocessed image tensor and metadata.
        """
        from app.vision.transforms import Compose, ToTensor, NormalizeS1
        
        logger.debug(f"Preprocessing S1 chip: {s1_chip_path}")
        if not os.path.exists(s1_chip_path):
            logger.error(f"File not found: {s1_chip_path}")
            raise FileNotFoundError(f"S1 chip path does not exist: {s1_chip_path}")

        try:
            with rasterio.open(s1_chip_path) as src:
                image = src.read()
                meta = src.meta.copy()

            logger.info(f"Trace [1] rasterio src.read() shape: {image.shape}")

            # Ensure we use identical preprocessing as the training path
            dummy_mask = np.zeros((1, image.shape[1], image.shape[2]), dtype=np.uint8)
            transform = Compose([ToTensor(), NormalizeS1()])
            image_tensor, _ = transform((image, dummy_mask))
            image_tensor = image_tensor.to(self.device)
            
            logger.info(f"Trace [2] after transforms ToTensor() shape: {image_tensor.shape}")

            # Extract useful metadata
            metadata = {
                'file_path': s1_chip_path,
                'width': meta.get('width'),
                'height': meta.get('height'),
                'crs': meta.get('crs') if meta.get('crs') else None,
                'bands': meta.get('count'),
                'original_shape': image.shape,
                'tensor_shape': tuple(image_tensor.shape)
            }
            
            final_tensor = image_tensor.unsqueeze(0)
            logger.info(f"Trace [3] final tensor unsqueeze(0) shape: {final_tensor.shape}")
            
            return final_tensor, metadata # Add batch dimension

        except Exception as e:
            logger.error(f"Error during preprocessing: {e}")
            raise

    def predict(self, image_tensor: torch.Tensor) -> torch.Tensor:
        """
        Executes the forward pass of the model.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed input tensor of shape (B, C, H, W).
            
        Returns:
            torch.Tensor: The predicted mask probabilities or logits.
        """
        logger.debug("Running prediction model.")
        try:
            if self.dummy_mode or self.model is None:
                # Simulate model prediction (batch, classes, height, width)
                # Create a random mask between 0 and 1
                b, c, h, w = image_tensor.shape
                output = torch.rand((b, 1, h, w), device=self.device)
                
                # We can introduce a fake 'flood' bias based on the input signal to make it slightly deterministic
                # e.g., if sum of tensor > 0
                return output
            else:
                with torch.no_grad():
                    return self.model(image_tensor)
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def postprocess(self, raw_output: torch.Tensor, metadata: Dict[str, Any], image_tensor: Optional[torch.Tensor] = None, debug: bool = False) -> Dict[str, Any]:
        """
        Processes the raw model output into human-readable results.
        
        Args:
            raw_output (torch.Tensor): Output from the predict function.
            metadata (Dict[str, Any]): Metadata from the preprocessing step.
            
        Returns:
            Dict[str, Any]: Structured inference results.
        """
        logger.debug("Postprocessing prediction results.")
        try:
            # Apply sigmoid because output is logits
            # Threshold to get binary mask
            probabilities = torch.sigmoid(raw_output).squeeze().cpu().numpy()
            binary_mask = (probabilities > 0.5).astype(np.uint8)
            
            total_pixels = binary_mask.size
            water_pixels = int(binary_mask.sum())
            water_percentage = (water_pixels / total_pixels) * 100 if total_pixels > 0 else 0.0
            
            # Confidence scaled to 0-100
            if water_pixels > 0:
                confidence_raw = float(probabilities[binary_mask == 1].mean())
            else:
                confidence_raw = float((1.0 - probabilities[binary_mask == 0]).mean())
                
            confidence = confidence_raw * 100.0
            flood_extent_pct = water_percentage
                
            flood_detected = water_percentage > 5.0 # Example threshold: > 5% water

            # Severity calculation
            if flood_extent_pct < 5.0:
                severity = "LOW"
            elif flood_extent_pct < 15.0:
                severity = "MODERATE"
            elif flood_extent_pct < 30.0:
                severity = "HIGH"
            else:
                severity = "SEVERE"

            # Risk Score
            risk_score = min(100.0, (flood_extent_pct / 30.0) * confidence)
            
            # Analyst Narrative
            explanation = f"Flooding was detected across approximately {flood_extent_pct:.1f}% of the analyzed area. Based on the observed extent, the event is classified as {severity.capitalize()} Severity. Confidence is {confidence:.1f}%."

            result = {
                "flood_detected": bool(flood_detected),
                "water_percentage": round(water_percentage, 2), # Legacy
                "flood_extent_pct": round(flood_extent_pct, 2),
                "severity": severity,
                "risk_score": round(risk_score, 2),
                "confidence": round(confidence, 2),
                "explanation": explanation,
                "mask_shape": list(binary_mask.shape),
                "metadata": metadata,
                "checkpoint_loaded": not self.dummy_mode,
                "model_name": self.model.__class__.__name__ if self.model else "DummyModel",
                "prediction_min": float(probabilities.min()),
                "prediction_max": float(probabilities.max()),
                "prediction_mean": float(probabilities.mean()),
                "flood_pixels": water_pixels,
                "flood_percentage": round(water_percentage, 2)
            }

            if debug:
                import matplotlib.pyplot as plt
                out_dir = "outputs/vision"
                os.makedirs(out_dir, exist_ok=True)
                
                mask_path = os.path.join(out_dir, "prediction_mask.png")
                plt.imsave(mask_path, binary_mask, cmap='gray')
                
                heatmap_path = os.path.join(out_dir, "confidence_heatmap.png")
                plt.imsave(heatmap_path, probabilities, cmap='jet')
                
                overlay_path = os.path.join(out_dir, "overlay.png")
                if image_tensor is not None:
                    img = image_tensor.squeeze(0).cpu().numpy()
                    if img.ndim == 3:
                        img = img[0] # use first channel
                    img_min, img_max = img.min(), img.max()
                    if img_max > img_min:
                        img = (img - img_min) / (img_max - img_min)
                    
                    plt.figure(figsize=(10, 10))
                    plt.imshow(img, cmap='gray')
                    # Create an RGBA mask for the red overlay
                    mask_rgba = np.zeros((*binary_mask.shape, 4))
                    mask_rgba[binary_mask == 1] = [1, 0, 0, 0.4] # Red with 40% opacity
                    plt.imshow(mask_rgba)
                    plt.axis('off')
                    plt.savefig(overlay_path, bbox_inches='tight', pad_inches=0)
                    plt.close()
                else:
                    overlay_path = mask_path

                result["explainability_outputs"] = {
                    "prediction_mask": mask_path,
                    "confidence_heatmap": heatmap_path,
                    "overlay": overlay_path
                }

            return result
        except Exception as e:
            logger.error(f"Error during postprocessing: {e}")
            raise

    def infer(self, s1_chip_path: str, debug: bool = False) -> Dict[str, Any]:
        """
        End-to-end inference pipeline: preprocess -> predict -> postprocess.
        
        Args:
            s1_chip_path (str): Path to the S1 image.
            
        Returns:
            Dict[str, Any]: Final prediction results.
        """
        logger.info(f"Starting inference for: {s1_chip_path}")
        try:
            image_tensor, metadata = self.preprocess(s1_chip_path)
            raw_output = self.predict(image_tensor)
            results = self.postprocess(raw_output, metadata, image_tensor=image_tensor if debug else None, debug=debug)
            logger.info(f"Inference completed successfully. Flood detected: {results['flood_detected']}")
            return results
        except Exception as e:
            logger.error(f"Inference pipeline failed: {e}")
            return {"error": str(e), "file": s1_chip_path}

    def infer_change(self, before_path: str, after_path: str) -> Dict[str, Any]:
        """
        Performs change detection between two images.
        """
        logger.info(f"Starting change detection: {before_path} -> {after_path}")
        try:
            # Predict on before image
            img_before, meta_before = self.preprocess(before_path)
            raw_before = self.predict(img_before)
            
            # Predict on after image
            img_after, meta_after = self.preprocess(after_path)
            raw_after = self.predict(img_after)
            
            # Postprocess both to get masks
            prob_before = torch.sigmoid(raw_before).squeeze().cpu().numpy()
            mask_before = (prob_before > 0.5).astype(np.uint8)
            
            prob_after = torch.sigmoid(raw_after).squeeze().cpu().numpy()
            mask_after = (prob_after > 0.5).astype(np.uint8)
            
            # Calculate changes
            total_pixels = mask_after.size
            if total_pixels == 0:
                raise ValueError("Images have no pixels.")
                
            flood_expansion_mask = (mask_after == 1) & (mask_before == 0)
            flood_reduction_mask = (mask_after == 0) & (mask_before == 1)
            changed_mask = flood_expansion_mask | flood_reduction_mask
            
            expansion_pixels = flood_expansion_mask.sum()
            reduction_pixels = flood_reduction_mask.sum()
            changed_pixels = changed_mask.sum()
            
            flood_expansion_pct = (expansion_pixels / total_pixels) * 100
            flood_reduction_pct = (reduction_pixels / total_pixels) * 100
            changed_area_pct = (changed_pixels / total_pixels) * 100
            
            # Save change mask
            out_dir = "outputs/vision"
            os.makedirs(out_dir, exist_ok=True)
            mask_path = os.path.join(out_dir, "change_mask.png")
            
            # Create a colored mask: 
            # Red for expansion, Green for reduction, Blue for stable water
            import matplotlib.pyplot as plt
            colored_mask = np.zeros((*mask_after.shape, 3))
            colored_mask[flood_expansion_mask] = [1, 0, 0] # Red = Expansion
            colored_mask[flood_reduction_mask] = [0, 1, 0] # Green = Reduction
            stable_water = (mask_after == 1) & (mask_before == 1)
            colored_mask[stable_water] = [0, 0, 1] # Blue = Stable Water
            
            plt.imsave(mask_path, colored_mask)
            
            results = {
                "changed_area": float(changed_area_pct),
                "flood_expansion": float(flood_expansion_pct),
                "flood_reduction": float(flood_reduction_pct),
                "change_percentage": float(changed_area_pct),
                "change_mask_path": mask_path,
                "status": "success"
            }
            logger.info("Change detection completed successfully.")
            return results
        except Exception as e:
            logger.error(f"Change detection pipeline failed: {e}")
            return {"error": str(e), "status": "error"}

