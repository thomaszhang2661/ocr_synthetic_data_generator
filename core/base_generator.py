"""
Base generator class for OCR synthetic data generation
"""
import os
import json
import random
import string
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Tuple, Optional, Union
from pathlib import Path
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm import tqdm

from PIL import Image, ImageFont
import numpy as np

from ..config.settings import Config, GeneratorConfig
from ..utils.image_processor import ImageProcessor
from ..utils.data_augmentation import DataAugmentation


class BaseGenerator(ABC):
    """Abstract base class for all data generators"""
    
    def __init__(self, config: GeneratorConfig):
        """
        Initialize base generator
        
        Args:
            config: Generator configuration
        """
        self.config = config
        
        # Setup logging
        import logging
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
        
        # Initialize image processor and augmentation
        self.image_processor = ImageProcessor()
        self.data_augmentation = DataAugmentation()
        
        # Load fonts
        self.fonts = self._load_fonts()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the generator"""
        logger = logging.getLogger(f"{self.__class__.__name__}")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_fonts(self) -> List[str]:
        """Load available fonts from system paths"""
        fonts = []
        font_extensions = ('.ttf', '.otf', '.TTF', '.OTF')
        
        for path in Config.get_font_paths():
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for file in files:
                        if file.endswith(font_extensions):
                            fonts.append(os.path.join(root, file))
        
        if not fonts:
            self.logger.warning("No fonts found, using system default")
            # Try to use a default system font
            try:
                font = ImageFont.load_default()
                fonts = ["default"]
            except:
                raise RuntimeError("No fonts available")
        
        self.logger.info(f"Loaded {len(fonts)} fonts")
        return fonts
    
    def _get_random_font(self, size: int) -> ImageFont.FreeTypeFont:
        """Get a random font with specified size"""
        if self.fonts == ["default"]:
            return ImageFont.load_default()
        
        font_path = random.choice(self.fonts)
        try:
            return ImageFont.truetype(font_path, size)
        except Exception as e:
            self.logger.warning(f"Failed to load font {font_path}: {e}")
            return ImageFont.load_default()
    
    @abstractmethod
    def generate_text(self, length: int = None) -> str:
        """Generate text content for the image"""
        pass
    
    @abstractmethod
    def create_base_image(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """Create base image with text"""
        pass
    
    def apply_augmentations(self, image: Image.Image) -> Image.Image:
        """Apply data augmentations to image"""
        if not self.config.augmentation:
            return image
        
        return self.data_augmentation.random_augmentation_pipeline(image)
    
    def post_process_image(self, image: Image.Image) -> Image.Image:
        """Post-process the generated image"""
        # Crop whitespace
        image = self.image_processor.crop_whitespace(image)
        
        # Resize to target size if specified
        if hasattr(self.config, 'target_size') and self.config.target_size:
            image = self.image_processor.resize_with_aspect_ratio(
                image, self.config.target_size
            )
        
        return image
    
    def generate_single_sample(self, text: str = None) -> Tuple[Image.Image, str]:
        """
        Generate a single training sample
        
        Args:
            text: Text to render (generated if None)
            
        Returns:
            Tuple of (image, text_label)
        """
        try:
            # Generate text if not provided
            if text is None:
                text = self.generate_text()
            
            # Get random font
            font = self._get_random_font(self.config.font_size)
            
            # Create base image
            image = self.create_base_image(text, font)
            
            # Apply augmentations
            if self.config.augmentation:
                image = self.apply_augmentations(image)
            
            # Post-process
            image = self.post_process_image(image)
            
            return image, text
            
        except Exception as e:
            self.logger.error(f"Error generating sample: {e}")
            raise
    
    def generate_batch(self, 
                      batch_size: int,
                      output_dir: str,
                      name_prefix: str = "sample",
                      save_labels: bool = True) -> List[Dict[str, Any]]:
        """
        Generate a batch of training samples
        
        Args:
            batch_size: Number of samples to generate
            output_dir: Directory to save images
            name_prefix: Prefix for image filenames
            save_labels: Whether to save label file
            
        Returns:
            List of sample metadata
        """
        os.makedirs(output_dir, exist_ok=True)
        samples = []
        
        self.logger.info(f"Generating {batch_size} samples...")
        
        for i in tqdm(range(batch_size), desc="Generating samples"):
            try:
                image, text = self.generate_single_sample()
                
                # Save image
                filename = f"{name_prefix}_{i:06d}.{self.config.output_format}"
                image_path = os.path.join(output_dir, filename)
                
                if self.config.output_format.lower() in ['jpg', 'jpeg']:
                    image.save(image_path, format='JPEG', quality=Config.DEFAULT_QUALITY)
                else:
                    image.save(image_path)
                
                # Prepare sample metadata
                sample = {
                    "filename": filename,
                    "text": text,
                    "path": image_path,
                    "config": self.config.to_dict()
                }
                samples.append(sample)
                
            except Exception as e:
                self.logger.error(f"Failed to generate sample {i}: {e}")
                continue
        
        # Save labels file
        if save_labels:
            labels_path = os.path.join(output_dir, "labels.json")
            with open(labels_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Saved labels to {labels_path}")
        
        self.logger.info(f"Generated {len(samples)} samples in {output_dir}")
        return samples
    
    def generate_parallel(self,
                         total_samples: int,
                         output_dir: str,
                         batch_size: int = 1000,
                         max_workers: int = None) -> List[Dict[str, Any]]:
        """
        Generate samples using parallel processing
        
        Args:
            total_samples: Total number of samples to generate
            output_dir: Directory to save images
            batch_size: Size of each batch
            max_workers: Number of parallel workers
            
        Returns:
            List of all sample metadata
        """
        if max_workers is None:
            max_workers = Config.MAX_WORKERS or os.cpu_count()
        
        # Calculate number of batches
        num_batches = (total_samples + batch_size - 1) // batch_size
        
        all_samples = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, total_samples)
                current_batch_size = end_idx - start_idx
                
                batch_output_dir = os.path.join(output_dir, f"batch_{batch_idx:03d}")
                
                future = executor.submit(
                    self.generate_batch,
                    current_batch_size,
                    batch_output_dir,
                    f"sample_{batch_idx:03d}",
                    True
                )
                futures.append(future)
            
            # Collect results
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc="Processing batches"):
                try:
                    samples = future.result()
                    all_samples.extend(samples)
                except Exception as e:
                    self.logger.error(f"Batch failed: {e}")
        
        # Save combined labels
        combined_labels_path = os.path.join(output_dir, "all_labels.json")
        with open(combined_labels_path, 'w', encoding='utf-8') as f:
            json.dump(all_samples, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Generated {len(all_samples)} total samples")
        return all_samples
    
    def get_sample_statistics(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Get statistics about generated samples"""
        if not samples:
            return {}
        
        texts = [sample['text'] for sample in samples]
        text_lengths = [len(text) for text in texts]
        
        return {
            "total_samples": len(samples),
            "avg_text_length": np.mean(text_lengths),
            "min_text_length": min(text_lengths),
            "max_text_length": max(text_lengths),
            "unique_texts": len(set(texts)),
            "config": samples[0]['config'] if samples else {}
        }