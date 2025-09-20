#!/usr/bin/env python3
"""
Simple example: Generate digit sequences for OCR training

This example demonstrates how to use the OCR Data Generator to create
digit sequence images suitable for training OCR models.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_data_generator import DigitGenerator
from ocr_data_generator.config.settings import GeneratorConfig


def main():
    """Generate digit sequence samples"""
    print("🔢 OCR Data Generator - Digit Example")
    print("=" * 50)
    
    # Configure digit generator
    config = GeneratorConfig(
        language="digits",
        font_size=32,
        image_size=(64, 400),  # height, width
        output_format="jpg",
        augmentation=True,
        # Digit-specific parameters
        min_length=6,
        max_length=12,
        cell_width=40,
        add_separators=True,
        separator_chars=['-', ' ']
    )
    
    # Create generator
    generator = DigitGenerator(config)
    
    # Set output directory
    output_dir = "./output/digit_samples"
    
    print(f"📁 Output directory: {output_dir}")
    print(f"🎯 Generating samples...")
    
    # Generate samples
    samples = generator.generate_batch(
        batch_size=100,
        output_dir=output_dir,
        name_prefix="digit",
        save_labels=True
    )
    
    # Show statistics
    stats = generator.get_sample_statistics(samples)
    print(f"\n📊 Generation Statistics:")
    print(f"   Total samples: {stats['total_samples']}")
    print(f"   Avg text length: {stats['avg_text_length']:.1f}")
    print(f"   Min text length: {stats['min_text_length']}")
    print(f"   Max text length: {stats['max_text_length']}")
    print(f"   Unique texts: {stats['unique_texts']}")
    
    print(f"\n✅ Successfully generated {len(samples)} digit samples!")
    print(f"📂 Check the '{output_dir}' directory for generated images and labels.json")
    
    # Show some example texts
    print(f"\n📝 Example generated texts:")
    for i in range(min(5, len(samples))):
        print(f"   {i+1}. {samples[i]['text']}")


if __name__ == "__main__":
    main()