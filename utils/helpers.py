"""
Utility functions for OCR data generation
"""
import os
import json
import random
from typing import List, Dict, Any, Tuple, Optional
from pathlib import Path
import shutil
import logging


class FontManager:
    """Manage font files and selections"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    def scan_fonts(directories: List[str]) -> List[str]:
        """
        Scan directories for font files
        
        Args:
            directories: List of directories to scan
            
        Returns:
            List of font file paths
        """
        font_extensions = ('.ttf', '.otf', '.TTF', '.OTF', '.ttc', '.TTC')
        fonts = []
        
        for directory in directories:
            if os.path.exists(directory):
                for root, dirs, files in os.walk(directory):
                    for file in files:
                        if file.endswith(font_extensions):
                            fonts.append(os.path.join(root, file))
        
        return fonts
    
    @staticmethod
    def filter_fonts_by_language(font_paths: List[str], language: str) -> List[str]:
        """
        Filter fonts suitable for specific language
        
        Args:
            font_paths: List of font file paths
            language: Language code ('en', 'zh', etc.)
            
        Returns:
            Filtered list of suitable fonts
        """
        if language == 'zh':
            # Chinese font keywords
            chinese_keywords = [
                'chinese', 'cjk', 'han', 'zh', 'cn', 'simsun', 'simhei', 
                'kaiti', 'fangsong', 'yahei', 'mingti', 'song', 'hei'
            ]
            suitable_fonts = []
            
            for font_path in font_paths:
                font_name = os.path.basename(font_path).lower()
                if any(keyword in font_name for keyword in chinese_keywords):
                    suitable_fonts.append(font_path)
            
            # If no specific Chinese fonts found, return all (might work)
            return suitable_fonts if suitable_fonts else font_paths
        
        # For other languages, return all fonts
        return font_paths


class DatasetManager:
    """Manage dataset creation and organization"""
    
    def __init__(self, base_path: str):
        """
        Initialize dataset manager
        
        Args:
            base_path: Base directory for datasets
        """
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
    
    def create_dataset_structure(self, dataset_name: str) -> str:
        """
        Create standard dataset directory structure
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            Path to dataset directory
        """
        dataset_path = self.base_path / dataset_name
        
        # Create directories
        directories = [
            dataset_path,
            dataset_path / 'images',
            dataset_path / 'labels',
            dataset_path / 'train',
            dataset_path / 'val',
            dataset_path / 'test'
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Created dataset structure at {dataset_path}")
        return str(dataset_path)
    
    def split_dataset(self, 
                     image_dir: str,
                     output_dir: str,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15) -> Dict[str, List[str]]:
        """
        Split dataset into train/val/test sets
        
        Args:
            image_dir: Directory containing images
            output_dir: Output directory for split datasets
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            test_ratio: Ratio for test set
            
        Returns:
            Dictionary with file lists for each split
        """
        # Validate ratios
        if abs(train_ratio + val_ratio + test_ratio - 1.0) > 0.01:
            raise ValueError("Ratios must sum to 1.0")
        
        # Get all image files
        image_files = []
        for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
            image_files.extend(Path(image_dir).glob(f'*{ext}'))
            image_files.extend(Path(image_dir).glob(f'*{ext.upper()}'))
        
        # Shuffle files
        random.shuffle(image_files)
        
        # Calculate split sizes
        total_files = len(image_files)
        train_size = int(total_files * train_ratio)
        val_size = int(total_files * val_ratio)
        
        # Split files
        train_files = image_files[:train_size]
        val_files = image_files[train_size:train_size + val_size]
        test_files = image_files[train_size + val_size:]
        
        # Create output directories
        output_path = Path(output_dir)
        train_dir = output_path / 'train'
        val_dir = output_path / 'val'
        test_dir = output_path / 'test'
        
        for directory in [train_dir, val_dir, test_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Copy files to appropriate directories
        splits = {
            'train': self._copy_files(train_files, train_dir),
            'val': self._copy_files(val_files, val_dir),
            'test': self._copy_files(test_files, test_dir)
        }
        
        self.logger.info(f"Split dataset: {len(train_files)} train, "
                        f"{len(val_files)} val, {len(test_files)} test")
        
        return splits
    
    def _copy_files(self, files: List[Path], target_dir: Path) -> List[str]:
        """Copy files to target directory"""
        copied_files = []
        
        for file_path in files:
            target_path = target_dir / file_path.name
            shutil.copy2(file_path, target_path)
            copied_files.append(str(target_path))
        
        return copied_files
    
    def create_annotation_file(self, 
                              samples: List[Dict[str, Any]], 
                              output_path: str,
                              format_type: str = 'json') -> str:
        """
        Create annotation file in various formats
        
        Args:
            samples: List of sample metadata
            output_path: Output file path
            format_type: Format type ('json', 'csv', 'txt')
            
        Returns:
            Path to created annotation file
        """
        if format_type == 'json':
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(samples, f, ensure_ascii=False, indent=2)
        
        elif format_type == 'csv':
            import csv
            with open(output_path, 'w', newline='', encoding='utf-8') as f:
                if samples:
                    writer = csv.DictWriter(f, fieldnames=samples[0].keys())
                    writer.writeheader()
                    writer.writerows(samples)
        
        elif format_type == 'txt':
            with open(output_path, 'w', encoding='utf-8') as f:
                for sample in samples:
                    f.write(f"{sample['filename']}\t{sample['text']}\n")
        
        self.logger.info(f"Created annotation file: {output_path}")
        return output_path


class QualityController:
    """Quality control for generated data"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def validate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Validate generated samples for quality
        
        Args:
            samples: List of sample metadata
            
        Returns:
            Validation report
        """
        report = {
            'total_samples': len(samples),
            'valid_samples': 0,
            'invalid_samples': 0,
            'issues': []
        }
        
        for i, sample in enumerate(samples):
            issues = []
            
            # Check if image file exists
            if 'path' in sample and not os.path.exists(sample['path']):
                issues.append(f"Image file not found: {sample['path']}")
            
            # Check text content
            if 'text' in sample:
                text = sample['text']
                if not text or not text.strip():
                    issues.append("Empty text content")
                elif len(text) > 100:  # Reasonable limit
                    issues.append(f"Text too long ({len(text)} chars)")
            
            # Check image dimensions if possible
            if 'path' in sample and os.path.exists(sample['path']):
                try:
                    from PIL import Image
                    with Image.open(sample['path']) as img:
                        width, height = img.size
                        if width < 10 or height < 10:
                            issues.append(f"Image too small: {width}x{height}")
                        elif width > 5000 or height > 1000:
                            issues.append(f"Image too large: {width}x{height}")
                except Exception as e:
                    issues.append(f"Cannot open image: {e}")
            
            if issues:
                report['invalid_samples'] += 1
                report['issues'].append({
                    'sample_index': i,
                    'filename': sample.get('filename', 'unknown'),
                    'issues': issues
                })
            else:
                report['valid_samples'] += 1
        
        # Calculate quality metrics
        if report['total_samples'] > 0:
            report['quality_score'] = report['valid_samples'] / report['total_samples']
        else:
            report['quality_score'] = 0.0
        
        self.logger.info(f"Quality validation: {report['valid_samples']}/{report['total_samples']} "
                        f"samples valid ({report['quality_score']:.2%})")
        
        return report
    
    def remove_invalid_samples(self, 
                              samples: List[Dict[str, Any]], 
                              validation_report: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Remove invalid samples based on validation report
        
        Args:
            samples: Original samples list
            validation_report: Report from validate_samples
            
        Returns:
            Filtered list of valid samples
        """
        invalid_indices = {issue['sample_index'] for issue in validation_report['issues']}
        valid_samples = [sample for i, sample in enumerate(samples) 
                        if i not in invalid_indices]
        
        self.logger.info(f"Removed {len(invalid_indices)} invalid samples")
        return valid_samples


class ConfigurationHelper:
    """Helper for managing configurations"""
    
    @staticmethod
    def create_config_template(output_path: str):
        """Create a configuration template file"""
        template = {
            "dataset_name": "my_ocr_dataset",
            "output_directory": "./output",
            "total_samples": 1000,
            "generators": {
                "digits": {
                    "enabled": True,
                    "samples": 400,
                    "config": {
                        "min_length": 6,
                        "max_length": 12,
                        "font_size": 32,
                        "cell_width": 40,
                        "add_separators": True
                    }
                },
                "english": {
                    "enabled": True,
                    "samples": 400,
                    "config": {
                        "min_length": 3,
                        "max_length": 20,
                        "font_size": 28,
                        "include_punctuation": True
                    }
                },
                "chinese": {
                    "enabled": True,
                    "samples": 200,
                    "config": {
                        "min_length": 2,
                        "max_length": 8,
                        "font_size": 32,
                        "include_punctuation": True
                    }
                }
            },
            "augmentation": {
                "enabled": True,
                "probability": 0.7,
                "rotation_range": [-5, 5],
                "perspective_strength": 0.1
            },
            "output": {
                "format": "jpg",
                "quality": 95,
                "annotation_format": "json"
            }
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(template, f, indent=2, ensure_ascii=False)
        
        print(f"Configuration template created: {output_path}")
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """Load configuration from file"""
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        required_fields = ['dataset_name', 'output_directory', 'generators']
        
        for field in required_fields:
            if field not in config:
                print(f"Missing required field: {field}")
                return False
        
        if not config['generators']:
            print("No generators configured")
            return False
        
        return True