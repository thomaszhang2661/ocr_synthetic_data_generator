#!/usr/bin/env python3
"""
Advanced example: Generate mixed-language OCR dataset

This example shows how to create a comprehensive OCR dataset
with multiple languages and data types.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_data_generator import DigitGenerator, TextGenerator, ChineseGenerator
from ocr_data_generator.config.settings import GeneratorConfig
from ocr_data_generator.utils.helpers import DatasetManager, QualityController


def main():
    """Generate mixed-language OCR dataset"""
    print("🌍 OCR Data Generator - Mixed Language Example")
    print("=" * 60)
    
    # Setup dataset manager
    dataset_manager = DatasetManager("./output")
    dataset_path = dataset_manager.create_dataset_structure("mixed_ocr_dataset")
    
    print(f"📁 Dataset path: {dataset_path}")
    
    # Quality controller
    quality_controller = QualityController()
    
    all_samples = []
    
    # 1. Generate digit sequences
    print(f"\n🔢 Generating digit sequences...")
    digit_config = GeneratorConfig(
        language="digits",
        font_size=28,
        output_format="jpg",
        augmentation=True,
        min_length=8,
        max_length=15,
        cell_width=35
    )
    
    digit_generator = DigitGenerator(digit_config)
    digit_samples = digit_generator.generate_batch(
        batch_size=300,
        output_dir=os.path.join(dataset_path, "digits"),
        name_prefix="digit",
        save_labels=True
    )
    all_samples.extend(digit_samples)
    
    # 2. Generate English text
    print(f"\n🔤 Generating English text...")
    text_config = GeneratorConfig(
        language="en",
        font_size=24,
        output_format="jpg",
        augmentation=True,
        min_length=5,
        max_length=25,
        include_punctuation=True
    )
    
    text_generator = TextGenerator(text_config)
    text_samples = text_generator.generate_batch(
        batch_size=300,
        output_dir=os.path.join(dataset_path, "english"),
        name_prefix="text",
        save_labels=True
    )
    all_samples.extend(text_samples)
    
    # 3. Generate Chinese text
    print(f"\n🀄 Generating Chinese text...")
    chinese_config = GeneratorConfig(
        language="zh",
        font_size=30,
        output_format="jpg",
        augmentation=True,
        min_length=3,
        max_length=10,
        include_punctuation=True
    )
    
    chinese_generator = ChineseGenerator(chinese_config)
    chinese_samples = chinese_generator.generate_batch(
        batch_size=200,
        output_dir=os.path.join(dataset_path, "chinese"),
        name_prefix="chinese",
        save_labels=True
    )
    all_samples.extend(chinese_samples)
    
    # 4. Quality validation
    print(f"\n🔍 Performing quality validation...")
    validation_report = quality_controller.validate_samples(all_samples)
    
    print(f"   Quality score: {validation_report['quality_score']:.2%}")
    print(f"   Valid samples: {validation_report['valid_samples']}")
    print(f"   Invalid samples: {validation_report['invalid_samples']}")
    
    if validation_report['invalid_samples'] > 0:
        print(f"   Issues found: {len(validation_report['issues'])}")
        # Remove invalid samples
        all_samples = quality_controller.remove_invalid_samples(all_samples, validation_report)
    
    # 5. Create dataset splits
    print(f"\n📊 Creating dataset splits...")
    
    # Combine all images into single directory for splitting
    combined_dir = os.path.join(dataset_path, "all_images")
    os.makedirs(combined_dir, exist_ok=True)
    
    import shutil
    for sample in all_samples:
        src_path = sample['path']
        dst_path = os.path.join(combined_dir, sample['filename'])
        shutil.copy2(src_path, dst_path)
        sample['path'] = dst_path  # Update path in metadata
    
    # Split dataset
    splits = dataset_manager.split_dataset(
        image_dir=combined_dir,
        output_dir=os.path.join(dataset_path, "splits"),
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15
    )
    
    # 6. Create comprehensive annotations
    print(f"\n📝 Creating annotation files...")
    
    # Save all samples metadata
    dataset_manager.create_annotation_file(
        all_samples,
        os.path.join(dataset_path, "annotations.json"),
        format_type="json"
    )
    
    dataset_manager.create_annotation_file(
        all_samples,
        os.path.join(dataset_path, "annotations.csv"),
        format_type="csv"
    )
    
    dataset_manager.create_annotation_file(
        all_samples,
        os.path.join(dataset_path, "annotations.txt"),
        format_type="txt"
    )
    
    # 7. Generate statistics
    print(f"\n📈 Dataset Statistics:")
    print(f"   Total samples: {len(all_samples)}")
    print(f"   Digit samples: {len(digit_samples)}")
    print(f"   English samples: {len(text_samples)}")
    print(f"   Chinese samples: {len(chinese_samples)}")
    print(f"   Train split: {len(splits['train'])}")
    print(f"   Val split: {len(splits['val'])}")
    print(f"   Test split: {len(splits['test'])}")
    
    # Language distribution
    language_stats = {}
    for sample in all_samples:
        lang = sample['config']['language']
        language_stats[lang] = language_stats.get(lang, 0) + 1
    
    print(f"\n🌐 Language Distribution:")
    for lang, count in language_stats.items():
        percentage = (count / len(all_samples)) * 100
        print(f"   {lang}: {count} samples ({percentage:.1f}%)")
    
    print(f"\n✅ Successfully created mixed-language OCR dataset!")
    print(f"📂 Dataset location: {dataset_path}")
    print(f"📊 Quality score: {validation_report['quality_score']:.2%}")
    
    # Generate summary report
    summary = {
        "dataset_name": "mixed_ocr_dataset",
        "total_samples": len(all_samples),
        "quality_score": validation_report['quality_score'],
        "language_distribution": language_stats,
        "splits": {split: len(files) for split, files in splits.items()},
        "generated_files": [
            "annotations.json",
            "annotations.csv", 
            "annotations.txt"
        ]
    }
    
    import json
    with open(os.path.join(dataset_path, "dataset_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"📋 Dataset summary saved to: {os.path.join(dataset_path, 'dataset_summary.json')}")


if __name__ == "__main__":
    main()