# OCR Synthetic Data Generator

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![PIL](https://img.shields.io/badge/PIL-8.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A comprehensive Python toolkit for generating synthetic OCR training data with support for multiple languages, fonts, and advanced data augmentation techniques.

> 🚀 A OCR training data generation toolkit with multi-language support and advanced data augmentation

[🇨🇳 中文文档](README_CN.md) | [📚 Documentation](docs/) | [🎯 Examples](examples/)

## ✨ Key Features

- **🏗️ Modular Architecture**: Refactored from prototype to production-ready toolkit
- **🌍 Multi-language Support**: English, Chinese (Simplified/Traditional), Digits
- **⚡ High Performance**: Multi-process parallel processing, 100+ samples/sec
- **🎨 Data Augmentation**: 10+ image enhancement techniques (perspective, noise, blur, etc.)
- **💼 Production Ready**: Complete configuration management, quality control, error handling

## 🎯 Core Innovation

### Character Composition Technology
Intelligent composition of individual character images into line-level handwriting:
```
Individual Character Images + Corpus Text → Smart Composition → Line Images + Annotations
```

### Multi-type Data Generators
- **DigitGenerator**: Digit sequences (6-12 digits, separators supported)
- **TextGenerator**: English text (names, addresses, sentences)
- **ChineseGenerator**: Chinese text (simplified/traditional support)
- **HandwritingLineGenerator**: Character composition based handwriting generation

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Command Line Usage
```bash
# Generate digit samples
python quick_start.py digits --samples 1000 --output ./digit_data

# Generate English text samples
python quick_start.py english --samples 500 --font-size 28

# Generate Chinese text samples
python quick_start.py chinese --samples 300 --output ./chinese_data

# Generate mixed dataset
python quick_start.py mixed --output ./mixed_data
```

### Programming Interface
```python
from ocr_data_generator import DigitGenerator
from ocr_data_generator.config.settings import GeneratorConfig

# Configure generator
config = GeneratorConfig(
    language="digits",
    font_size=32,
    min_length=6,
    max_length=12,
    augmentation=True
)

# Create generator and generate samples
generator = DigitGenerator(config)
samples = generator.generate_batch(
    batch_size=1000,
    output_dir="./output",
    save_labels=True
)
```

## 📚 Examples

### 1. Generate Digit Sequences
```python
# examples/example_digits.py
python3 examples/example_digits.py
```
Generates digit sequences like student IDs, phone numbers, etc.

### 2. Mixed Language Dataset
```python
# examples/example_mixed_dataset.py
python3 examples/example_mixed_dataset.py
```
Creates a comprehensive dataset with English, Chinese, and digits.

### 3. Configuration-Based Generation
```python
# examples/example_config_based.py
python3 examples/example_config_based.py
```
Uses JSON configuration files for complex dataset specifications.

## 🛠️ Architecture

### Core Components
```
ocr_data_generator/
├── core/
│   └── base_generator.py      # Abstract base generator class
├── generators/
│   ├── digit_generator.py     # Digit sequence generation
│   ├── text_generator.py      # English text generation
│   └── chinese_generator.py   # Chinese text generation
├── utils/
│   ├── image_processor.py     # Image processing utilities
│   ├── data_augmentation.py   # Data augmentation techniques
│   └── helpers.py            # Utility functions
├── config/
│   └── settings.py           # Configuration management
└── examples/
    ├── example_digits.py      # Basic digit generation
    ├── example_mixed_dataset.py # Multi-language dataset
    └── example_config_based.py  # Configuration-driven generation
```

### Generator Classes
1. **BaseGenerator**: Abstract base class with common functionality
2. **DigitGenerator**: Specialized for numerical sequences (IDs, phone numbers)
3. **TextGenerator**: English text and mixed alphanumeric content
4. **ChineseGenerator**: Chinese characters with punctuation support

### Data Augmentation Pipeline
- **Geometric Transformations**: Rotation, perspective distortion
- **Photometric Effects**: Brightness, contrast adjustment
- **Noise Addition**: Gaussian noise, blur effects
- **Stroke Modifications**: Thickness adjustment, gap insertion
- **Background Integration**: Texture blending, copy effects

## ⚙️ Configuration

### Generator Configuration
```python
config = GeneratorConfig(
    language="digits",           # "digits", "en", "zh"
    font_size=32,               # Font size in pixels
    image_size=(64, 400),       # (height, width)
    output_format="jpg",        # "jpg", "png"
    augmentation=True,          # Enable data augmentation
    
    # Language-specific parameters
    min_length=6,               # Minimum text length
    max_length=12,              # Maximum text length
    include_punctuation=True,   # Include punctuation marks
    
    # Digit-specific parameters
    cell_width=40,              # Width per digit cell
    add_separators=True,        # Add separators like "-", " "
    
    # Custom parameters
    custom_param="value"        # Additional parameters
)
```

### JSON Configuration
```json
{
  "dataset_name": "my_ocr_dataset",
  "output_directory": "./output",
  "total_samples": 1000,
  "generators": {
    "digits": {
      "enabled": true,
      "samples": 400,
      "config": {
        "min_length": 6,
        "max_length": 12,
        "font_size": 32
      }
    },
    "english": {
      "enabled": true,
      "samples": 400,
      "config": {
        "min_length": 3,
        "max_length": 20,
        "font_size": 28
      }
    }
  },
  "augmentation": {
    "enabled": true,
    "probability": 0.7
  }
}
```

## 🔧 Advanced Configuration

### Custom Data Augmentation
```python
config = GeneratorConfig(
    augmentation=True,
    perspective_prob=0.3,      # Perspective transform probability
    noise_prob=0.2,            # Noise addition probability
    blur_prob=0.1,             # Blur effect probability
    brightness_range=(0.8, 1.2)  # Brightness variation range
)
```

### Character Composition (Core Feature)
```python
from ocr_data_generator import HandwritingLineGenerator

# Configure character composition generator
config = {
    'char_dict_path': './merged_dict.txt',
    'char_image_directory': './chinese_data/',
    'corpus_directory': './corpus',
    'corpus_files': ['text_corpus.txt']
}

generator = HandwritingLineGenerator(config)
samples = generator.generate_line_samples(1000, './output')
```

## 📈 Use Cases

### 1. OCR Model Training
- Generate large amounts of annotated data, reducing manual annotation costs
- Diverse fonts and styles improve model generalization
- Controllable data distribution for specific scenario optimization

### 2. Data Augmentation
- Expansion and enhancement of existing datasets
- Synthetic generation of rare samples
- Data simulation under different conditions

### 3. Algorithm Validation
- Standard datasets for algorithm performance testing
- Test samples of different difficulty levels
- Reproducible experimental data

## 🎯 Use Cases

### 1. Student ID Recognition
```python
config = GeneratorConfig(
    language="digits",
    min_length=10,
    max_length=10,
    cell_width=35,
    add_separators=False
)
```

### 2. License Plate Recognition
```python
config = GeneratorConfig(
    language="en",
    min_length=6,
    max_length=8,
    include_numbers=True,
    font_size=36
)
```

### 3. Chinese Address Recognition
```python
config = GeneratorConfig(
    language="zh",
    min_length=8,
    max_length=20,
    include_punctuation=True,
    font_size=28
)
```

### 4. Form Field Recognition
```python
# Mixed content generator
from ocr_data_generator.generators.text_generator import FormTextGenerator

generator = FormTextGenerator()
# Generates names, addresses, phone numbers, etc.
```

## 🏆 Technical Highlights

### Innovative Algorithms
- **Character Composition**: Intelligent combination of individual character images into line-level images
- **Smart Alignment**: Automatic adjustment of character spacing and baseline alignment
- **Style Preservation**: Maintains original handwriting character style features

### Architecture Design
- **Modular Refactoring**: From messy code to clean architecture
- **Object-Oriented**: Abstract base classes and inheritance system
- **Plugin-based**: Easy to extend new generators and enhancement techniques

### Performance Optimization
- **Parallel Processing**: Multi-process batch generation
- **Memory Management**: Smart caching and memory optimization
- **Quality Control**: Automatic quality detection and filtering

## 🛠️ Development & Testing

### Run Tests
```bash
python -m pytest tests/
```

### Code Quality Check
```bash
# Format code
black .

# Code linting
flake8 .
```

## 📚 Documentation

- [Project Summary](PROJECT_SUMMARY.md) - Detailed technical documentation
- [Resume Format](RESUME_FORMAT.md) - Resume description templates
- [API Documentation](README.md) - Complete API reference
- [Usage Examples](examples/) - Code examples

## 🔧 Advanced Features

### Custom Font Integration
```python
from ocr_data_generator.utils.helpers import FontManager

# Scan for fonts
fonts = FontManager.scan_fonts(["/path/to/fonts"])

# Filter by language
chinese_fonts = FontManager.filter_fonts_by_language(fonts, "zh")
```

### Dataset Management
```python
from ocr_data_generator.utils.helpers import DatasetManager

manager = DatasetManager("./datasets")

# Create dataset structure
dataset_path = manager.create_dataset_structure("my_dataset")

# Split into train/val/test
splits = manager.split_dataset(
    image_dir="./images",
    output_dir="./splits",
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15
)
```

### Parallel Processing
```python
# Generate large datasets with parallel processing
samples = generator.generate_parallel(
    total_samples=10000,
    output_dir="./large_dataset",
    batch_size=1000,
    max_workers=8
)
```

### Debug Mode
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Generators will output detailed debug information
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

For questions, issues, or feature requests:

1. Check the [examples](examples/) directory
2. Review the [troubleshooting](#troubleshooting) section
3. Open an issue on GitHub

---

**Author**: Thomas Zhang  
**Date**: 2024-09-19  
**Version**: 1.0.0
