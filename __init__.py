"""
OCR Data Generator Package

A comprehensive toolkit for generating synthetic OCR training data.

Main Features:
1. Character-based line image composition (original core functionality)
2. Font-based text generation 
3. Advanced data augmentation
4. Multi-language support

Original Core Feature:
- Load individual character images from dictionary
- Compose line images using corpus text
- Generate annotated handwriting-style training data

Author: Zhang Jian (based on original project)
Date: 2024-09-19
"""

__version__ = "1.0.0"
__author__ = "Zhang Jian"

# Import original core functionality (character composition)
from .core.line_composer import (
    CharacterImageLoader,
    CorpusProcessor, 
    LineImageComposer,
    HandwritingLineGenerator
)

# Import generator framework
from .core.base_generator import BaseGenerator
from .generators.text_generator import TextGenerator
from .generators.digit_generator import DigitGenerator
from .generators.chinese_generator import ChineseGenerator

# Import utilities
from .utils.image_processor import ImageProcessor
from .utils.data_augmentation import DataAugmentation

__all__ = [
    # Original core functionality
    "CharacterImageLoader",
    "CorpusProcessor",
    "LineImageComposer", 
    "HandwritingLineGenerator",
    
    # Generator framework
    "BaseGenerator",
    "TextGenerator", 
    "DigitGenerator",
    "ChineseGenerator",
    
    # Utilities
    "ImageProcessor",
    "DataAugmentation"
]