#!/usr/bin/env python3
"""
Quick Start Script for OCR Data Generator

This script provides a simple command-line interface to quickly generate
OCR training data without writing any code.
"""

import argparse
import sys
import os
import json
from pathlib import Path

# Add the parent directory to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from generators.digit_generator import DigitGenerator
    from generators.text_generator import TextGenerator
    from generators.chinese_generator import ChineseGenerator
    from config.settings import GeneratorConfig
    from ocr_data_generator.utils.helpers import DatasetManager, QualityController
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install Pillow numpy opencv-python tqdm")
    sys.exit(1)


def create_parser():
    """Create command line argument parser"""
    parser = argparse.ArgumentParser(
        description="OCR Synthetic Data Generator - Quick Start",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Generate 1000 digit samples:
    python quick_start.py digits --samples 1000 --output ./digit_data
    
  Generate English text with custom settings:
    python quick_start.py english --samples 500 --font-size 28 --min-length 5
    
  Generate Chinese text:
    python quick_start.py chinese --samples 300 --output ./chinese_data
    
  Generate mixed dataset:
    python quick_start.py mixed --output ./mixed_data
        """
    )
    
    # Subcommands for different generators
    subparsers = parser.add_subparsers(dest='generator', help='Generator type')
    
    # Common arguments
    def add_common_args(parser):
        parser.add_argument('--samples', type=int, default=100,
                          help='Number of samples to generate (default: 100)')
        parser.add_argument('--output', type=str, default='./output',
                          help='Output directory (default: ./output)')
        parser.add_argument('--font-size', type=int, default=32,
                          help='Font size in pixels (default: 32)')
        parser.add_argument('--format', choices=['jpg', 'png'], default='jpg',
                          help='Output format (default: jpg)')
        parser.add_argument('--augmentation', action='store_true',
                          help='Enable data augmentation')
        parser.add_argument('--no-augmentation', dest='augmentation', action='store_false',
                          help='Disable data augmentation')
        parser.set_defaults(augmentation=True)
    
    # Digits generator
    digits_parser = subparsers.add_parser('digits', help='Generate digit sequences')
    add_common_args(digits_parser)
    digits_parser.add_argument('--min-length', type=int, default=6,
                              help='Minimum digit length (default: 6)')
    digits_parser.add_argument('--max-length', type=int, default=12,
                              help='Maximum digit length (default: 12)')
    digits_parser.add_argument('--cell-width', type=int, default=40,
                              help='Width per digit cell (default: 40)')
    digits_parser.add_argument('--separators', action='store_true',
                              help='Add separators like - and spaces')
    
    # English text generator
    english_parser = subparsers.add_parser('english', help='Generate English text')
    add_common_args(english_parser)
    english_parser.add_argument('--min-length', type=int, default=3,
                               help='Minimum text length (default: 3)')
    english_parser.add_argument('--max-length', type=int, default=20,
                               help='Maximum text length (default: 20)')
    english_parser.add_argument('--punctuation', action='store_true', default=True,
                               help='Include punctuation')
    english_parser.add_argument('--no-punctuation', dest='punctuation', action='store_false',
                               help='Exclude punctuation')
    
    # Chinese text generator
    chinese_parser = subparsers.add_parser('chinese', help='Generate Chinese text')
    add_common_args(chinese_parser)
    chinese_parser.add_argument('--min-length', type=int, default=3,
                               help='Minimum text length (default: 3)')
    chinese_parser.add_argument('--max-length', type=int, default=10,
                               help='Maximum text length (default: 10)')
    chinese_parser.add_argument('--punctuation', action='store_true', default=True,
                               help='Include Chinese punctuation')
    
    # Mixed dataset generator
    mixed_parser = subparsers.add_parser('mixed', help='Generate mixed language dataset')
    mixed_parser.add_argument('--output', type=str, default='./mixed_output',
                            help='Output directory (default: ./mixed_output)')
    mixed_parser.add_argument('--digits', type=int, default=200,
                            help='Number of digit samples (default: 200)')
    mixed_parser.add_argument('--english', type=int, default=200,
                            help='Number of English samples (default: 200)')
    mixed_parser.add_argument('--chinese', type=int, default=100,
                            help='Number of Chinese samples (default: 100)')
    mixed_parser.add_argument('--augmentation', action='store_true', default=True,
                            help='Enable data augmentation')
    
    return parser


def generate_digits(args):
    """Generate digit sequences"""
    print(f"🔢 Generating {args.samples} digit samples...")
    
    config = GeneratorConfig(
        language="digits",
        font_size=args.font_size,
        output_format=args.format,
        augmentation=args.augmentation,
        min_length=args.min_length,
        max_length=args.max_length,
        cell_width=args.cell_width,
        add_separators=args.separators
    )
    
    generator = DigitGenerator(config)
    samples = generator.generate_batch(
        batch_size=args.samples,
        output_dir=args.output,
        name_prefix="digit",
        save_labels=True
    )
    
    return samples


def generate_english(args):
    """Generate English text"""
    print(f"🔤 Generating {args.samples} English text samples...")
    
    config = GeneratorConfig(
        language="en",
        font_size=args.font_size,
        output_format=args.format,
        augmentation=args.augmentation,
        min_length=args.min_length,
        max_length=args.max_length,
        include_punctuation=args.punctuation
    )
    
    generator = TextGenerator(config)
    samples = generator.generate_batch(
        batch_size=args.samples,
        output_dir=args.output,
        name_prefix="text",
        save_labels=True
    )
    
    return samples


def generate_chinese(args):
    """Generate Chinese text"""
    print(f"🀄 Generating {args.samples} Chinese text samples...")
    
    config = GeneratorConfig(
        language="zh",
        font_size=args.font_size,
        output_format=args.format,
        augmentation=args.augmentation,
        min_length=args.min_length,
        max_length=args.max_length,
        include_punctuation=args.punctuation
    )
    
    generator = ChineseGenerator(config)
    samples = generator.generate_batch(
        batch_size=args.samples,
        output_dir=args.output,
        name_prefix="chinese",
        save_labels=True
    )
    
    return samples


def generate_mixed(args):
    """Generate mixed language dataset"""
    print(f"🌍 Generating mixed language dataset...")
    print(f"   Digits: {args.digits}")
    print(f"   English: {args.english}")
    print(f"   Chinese: {args.chinese}")
    
    # Setup dataset structure
    dataset_manager = DatasetManager(args.output)
    dataset_path = dataset_manager.create_dataset_structure("mixed_dataset")
    
    all_samples = []
    
    # Generate digits
    if args.digits > 0:
        digit_config = GeneratorConfig(
            language="digits",
            font_size=32,
            augmentation=args.augmentation
        )
        digit_generator = DigitGenerator(digit_config)
        digit_samples = digit_generator.generate_batch(
            batch_size=args.digits,
            output_dir=os.path.join(dataset_path, "digits"),
            name_prefix="digit",
            save_labels=True
        )
        all_samples.extend(digit_samples)
    
    # Generate English
    if args.english > 0:
        english_config = GeneratorConfig(
            language="en",
            font_size=28,
            augmentation=args.augmentation
        )
        english_generator = TextGenerator(english_config)
        english_samples = english_generator.generate_batch(
            batch_size=args.english,
            output_dir=os.path.join(dataset_path, "english"),
            name_prefix="text",
            save_labels=True
        )
        all_samples.extend(english_samples)
    
    # Generate Chinese
    if args.chinese > 0:
        chinese_config = GeneratorConfig(
            language="zh",
            font_size=30,
            augmentation=args.augmentation
        )
        chinese_generator = ChineseGenerator(chinese_config)
        chinese_samples = chinese_generator.generate_batch(
            batch_size=args.chinese,
            output_dir=os.path.join(dataset_path, "chinese"),
            name_prefix="chinese",
            save_labels=True
        )
        all_samples.extend(chinese_samples)
    
    # Save combined annotations
    dataset_manager.create_annotation_file(
        all_samples,
        os.path.join(dataset_path, "all_annotations.json"),
        format_type="json"
    )
    
    return all_samples


def main():
    """Main function"""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.generator:
        parser.print_help()
        return
    
    print("🚀 OCR Data Generator - Quick Start")
    print("=" * 50)
    
    try:
        # Generate samples based on command
        if args.generator == 'digits':
            samples = generate_digits(args)
        elif args.generator == 'english':
            samples = generate_english(args)
        elif args.generator == 'chinese':
            samples = generate_chinese(args)
        elif args.generator == 'mixed':
            samples = generate_mixed(args)
        else:
            print(f"❌ Unknown generator: {args.generator}")
            return
        
        # Quality check
        print(f"\n🔍 Performing quality check...")
        qc = QualityController()
        report = qc.validate_samples(samples)
        
        print(f"\n✅ Generation completed successfully!")
        print(f"📊 Statistics:")
        print(f"   Total samples: {len(samples)}")
        print(f"   Quality score: {report['quality_score']:.2%}")
        print(f"   Valid samples: {report['valid_samples']}")
        print(f"   Invalid samples: {report['invalid_samples']}")
        
        if hasattr(args, 'output'):
            print(f"📂 Output directory: {args.output}")
        
        # Show some examples
        print(f"\n📝 Example generated texts:")
        for i in range(min(5, len(samples))):
            print(f"   {i+1}. {samples[i]['text']}")
        
        print(f"\n💡 Next steps:")
        print(f"   1. Check the output directory for generated images")
        print(f"   2. Review the labels.json file for annotations")
        print(f"   3. Use the data for training your OCR models")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()