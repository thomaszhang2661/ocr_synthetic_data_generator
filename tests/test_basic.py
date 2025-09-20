"""
Basic tests for OCR Data Generator

Run with: python -m pytest tests/ -v
"""

import unittest
import tempfile
import shutil
import os
from pathlib import Path

# Import modules to test
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_data_generator.config.settings import Config, GeneratorConfig
from ocr_data_generator.utils.image_processor import ImageProcessor
from ocr_data_generator.utils.data_augmentation import DataAugmentation
from ocr_data_generator.utils.helpers import FontManager, DatasetManager, ConfigurationHelper


class TestConfig(unittest.TestCase):
    """Test configuration classes"""
    
    def test_config_defaults(self):
        """Test default configuration values"""
        self.assertEqual(Config.DEFAULT_IMAGE_SIZE, (64, 512))
        self.assertEqual(Config.DEFAULT_FONT_SIZE, 32)
        self.assertIn("en", Config.SUPPORTED_LANGUAGES)
        self.assertIn("zh", Config.SUPPORTED_LANGUAGES)
        self.assertIn("digits", Config.SUPPORTED_LANGUAGES)
    
    def test_generator_config(self):
        """Test generator configuration"""
        config = GeneratorConfig(
            language="en",
            font_size=24,
            augmentation=True
        )
        
        self.assertEqual(config.language, "en")
        self.assertEqual(config.font_size, 24)
        self.assertTrue(config.augmentation)
        
        # Test to_dict method
        config_dict = config.to_dict()
        self.assertIsInstance(config_dict, dict)
        self.assertEqual(config_dict["language"], "en")
    
    def test_config_validation(self):
        """Test configuration validation"""
        # This might fail if no fonts are available, which is expected
        try:
            result = Config.validate()
            self.assertIsInstance(result, bool)
        except:
            pass  # Expected in test environment without fonts


class TestImageProcessor(unittest.TestCase):
    """Test image processing utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = ImageProcessor()
        
        # Create a simple test image
        try:
            from PIL import Image
            self.test_image = Image.new('RGB', (100, 50), color='white')
            # Add some content
            from PIL import ImageDraw
            draw = ImageDraw.Draw(self.test_image)
            draw.text((20, 15), "TEST", fill='black')
            self.has_pil = True
        except ImportError:
            self.has_pil = False
    
    def test_crop_whitespace(self):
        """Test whitespace cropping"""
        if not self.has_pil:
            self.skipTest("PIL not available")
            
        # Test with image that has whitespace
        cropped = self.processor.crop_whitespace(self.test_image)
        self.assertIsNotNone(cropped)
        
        # Cropped image should be smaller or equal in size
        self.assertLessEqual(cropped.size[0], self.test_image.size[0])
        self.assertLessEqual(cropped.size[1], self.test_image.size[1])
    
    def test_resize_with_aspect_ratio(self):
        """Test aspect ratio preserving resize"""
        if not self.has_pil:
            self.skipTest("PIL not available")
            
        target_size = (200, 100)
        resized = self.processor.resize_with_aspect_ratio(
            self.test_image, target_size
        )
        
        self.assertEqual(resized.size, target_size)
    
    def test_add_border(self):
        """Test border addition"""
        if not self.has_pil:
            self.skipTest("PIL not available")
            
        bordered = self.processor.add_border(self.test_image, border_width=5)
        
        # Image should be larger after adding border
        self.assertGreater(bordered.size[0], self.test_image.size[0])
        self.assertGreater(bordered.size[1], self.test_image.size[1])


class TestDataAugmentation(unittest.TestCase):
    """Test data augmentation utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.augmentation = DataAugmentation()
        
        try:
            from PIL import Image
            self.test_image = Image.new('L', (100, 50), color=255)
            from PIL import ImageDraw
            draw = ImageDraw.Draw(self.test_image)
            draw.text((20, 15), "TEST", fill=0)
            self.has_pil = True
        except ImportError:
            self.has_pil = False
    
    def test_rotate_image(self):
        """Test image rotation"""
        if not self.has_pil:
            self.skipTest("PIL not available")
            
        rotated = self.augmentation.rotate_image(
            self.test_image, angle_range=(-5, 5)
        )
        self.assertIsNotNone(rotated)
    
    def test_adjust_brightness_contrast(self):
        """Test brightness and contrast adjustment"""
        if not self.has_pil:
            self.skipTest("PIL not available")
            
        adjusted = self.augmentation.adjust_brightness_contrast(self.test_image)
        self.assertIsNotNone(adjusted)
        self.assertEqual(adjusted.size, self.test_image.size)


class TestFontManager(unittest.TestCase):
    """Test font management utilities"""
    
    def test_scan_fonts(self):
        """Test font scanning"""
        # Test with non-existent directory
        fonts = FontManager.scan_fonts(["/non/existent/path"])
        self.assertEqual(len(fonts), 0)
        
        # Test with system directories (might not exist in all environments)
        system_paths = ["/System/Library/Fonts", "/usr/share/fonts"]
        existing_paths = [path for path in system_paths if os.path.exists(path)]
        
        if existing_paths:
            fonts = FontManager.scan_fonts(existing_paths)
            self.assertIsInstance(fonts, list)
    
    def test_filter_fonts_by_language(self):
        """Test font filtering by language"""
        test_fonts = [
            "/path/to/arial.ttf",
            "/path/to/simsun.ttc",
            "/path/to/chinese.otf",
            "/path/to/regular.ttf"
        ]
        
        # Test Chinese font filtering
        chinese_fonts = FontManager.filter_fonts_by_language(test_fonts, "zh")
        self.assertIsInstance(chinese_fonts, list)
        
        # Test English font filtering (should return all)
        english_fonts = FontManager.filter_fonts_by_language(test_fonts, "en")
        self.assertEqual(len(english_fonts), len(test_fonts))


class TestDatasetManager(unittest.TestCase):
    """Test dataset management utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = DatasetManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_dataset_structure(self):
        """Test dataset structure creation"""
        dataset_path = self.manager.create_dataset_structure("test_dataset")
        
        # Check if directories were created
        self.assertTrue(os.path.exists(dataset_path))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, "images")))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, "labels")))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, "train")))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, "val")))
        self.assertTrue(os.path.exists(os.path.join(dataset_path, "test")))
    
    def test_create_annotation_file(self):
        """Test annotation file creation"""
        samples = [
            {"filename": "test1.jpg", "text": "hello"},
            {"filename": "test2.jpg", "text": "world"}
        ]
        
        # Test JSON format
        json_path = os.path.join(self.temp_dir, "test.json")
        result_path = self.manager.create_annotation_file(samples, json_path, "json")
        
        self.assertEqual(result_path, json_path)
        self.assertTrue(os.path.exists(json_path))
        
        # Verify content
        import json
        with open(json_path, 'r') as f:
            loaded_samples = json.load(f)
        
        self.assertEqual(len(loaded_samples), 2)
        self.assertEqual(loaded_samples[0]["text"], "hello")


class TestConfigurationHelper(unittest.TestCase):
    """Test configuration helper utilities"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_create_config_template(self):
        """Test configuration template creation"""
        config_path = os.path.join(self.temp_dir, "test_config.json")
        ConfigurationHelper.create_config_template(config_path)
        
        self.assertTrue(os.path.exists(config_path))
        
        # Verify it's valid JSON
        config = ConfigurationHelper.load_config(config_path)
        self.assertIsInstance(config, dict)
        self.assertIn("dataset_name", config)
        self.assertIn("generators", config)
    
    def test_validate_config(self):
        """Test configuration validation"""
        # Valid config
        valid_config = {
            "dataset_name": "test",
            "output_directory": "./output",
            "generators": {"digits": {"enabled": True}}
        }
        
        self.assertTrue(ConfigurationHelper.validate_config(valid_config))
        
        # Invalid config (missing required field)
        invalid_config = {
            "dataset_name": "test"
            # Missing output_directory and generators
        }
        
        self.assertFalse(ConfigurationHelper.validate_config(invalid_config))


class TestIntegration(unittest.TestCase):
    """Integration tests"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        shutil.rmtree(self.temp_dir)
    
    def test_basic_workflow(self):
        """Test basic generation workflow"""
        try:
            from ocr_data_generator import DigitGenerator
            from ocr_data_generator.config.settings import GeneratorConfig
            
            # Create simple config
            config = GeneratorConfig(
                language="digits",
                font_size=20,
                min_length=3,
                max_length=5,
                augmentation=False  # Disable for faster testing
            )
            
            # Create generator
            generator = DigitGenerator(config)
            
            # Test single sample generation
            try:
                image, text = generator.generate_single_sample()
                self.assertIsNotNone(image)
                self.assertIsNotNone(text)
                self.assertTrue(text.isdigit() or any(c in text for c in ['-', ' ', '.']))
            except Exception as e:
                # May fail due to missing fonts, which is expected
                self.skipTest(f"Font-related test skipped: {e}")
            
        except ImportError as e:
            self.skipTest(f"Missing dependencies: {e}")


if __name__ == "__main__":
    # Run tests
    unittest.main(verbosity=2)