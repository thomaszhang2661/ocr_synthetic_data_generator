"""
Configuration settings for OCR Data Generator
"""
import os
from typing import List, Dict, Any

class Config:
    """Main configuration class"""
    
    # Data generation settings
    DEFAULT_IMAGE_SIZE = (64, 512)  # height, width
    DEFAULT_FONT_SIZE = 32
    DEFAULT_OUTPUT_FORMAT = "jpg"
    DEFAULT_QUALITY = 95
    
    # Augmentation settings
    ROTATION_RANGE = (-5, 5)  # degrees
    PERSPECTIVE_STRENGTH = 0.1
    BRIGHTNESS_RANGE = (0.7, 1.3)
    CONTRAST_RANGE = (0.8, 1.2)
    
    # Text generation settings
    MIN_TEXT_LENGTH = 1
    MAX_TEXT_LENGTH = 20
    
    # Character sets
    DIGITS = "0123456789"
    ENGLISH_LETTERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    PUNCTUATION = ".,!?;:-()[]{}\"'"
    
    # Supported languages
    SUPPORTED_LANGUAGES = ["en", "zh", "digits"]
    
    # Font paths (to be configured by user)
    FONT_PATHS = {
        "system": "/System/Library/Fonts/",
        "user": os.path.expanduser("~/Library/Fonts/"),
        "custom": "./assets/fonts/"
    }
    
    # Background settings
    USE_BACKGROUND = True
    BACKGROUND_OPACITY = 0.3
    
    # Processing settings
    MULTIPROCESSING = True
    MAX_WORKERS = None  # None means use all available cores
    
    @classmethod
    def get_font_paths(cls) -> List[str]:
        """Get all available font paths"""
        paths = []
        for path in cls.FONT_PATHS.values():
            if os.path.exists(path):
                paths.append(path)
        return paths
    
    @classmethod
    def validate(cls) -> bool:
        """Validate configuration"""
        # Check if at least one font path exists
        if not cls.get_font_paths():
            print("Warning: No font paths found")
            return False
        return True


class GeneratorConfig:
    """Configuration for specific generators"""
    
    def __init__(self, 
                 language: str = "en",
                 font_size: int = Config.DEFAULT_FONT_SIZE,
                 image_size: tuple = Config.DEFAULT_IMAGE_SIZE,
                 output_format: str = Config.DEFAULT_OUTPUT_FORMAT,
                 augmentation: bool = True,
                 **kwargs):
        self.language = language
        self.font_size = font_size
        self.image_size = image_size
        self.output_format = output_format
        self.augmentation = augmentation
        self.extra_params = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            "language": self.language,
            "font_size": self.font_size,
            "image_size": self.image_size,
            "output_format": self.output_format,
            "augmentation": self.augmentation,
            **self.extra_params
        }