#!/usr/bin/env python3
"""
Configuration-based example: Generate dataset from config file

This example demonstrates how to use configuration files to 
generate OCR datasets with predefined settings.
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ocr_data_generator import DigitGenerator, TextGenerator, ChineseGenerator
from ocr_data_generator.config.settings import GeneratorConfig
from ocr_data_generator.utils.helpers import ConfigurationHelper, DatasetManager


def create_config_file():
    """Create example configuration file"""
    config_path = "./example_config.json"
    ConfigurationHelper.create_config_template(config_path)
    print(f"📄 Created configuration template: {config_path}")
    return config_path


def generate_from_config(config_path: str):
    """Generate dataset based on configuration file"""
    
    # Load configuration
    config = ConfigurationHelper.load_config(config_path)
    
    if not ConfigurationHelper.validate_config(config):
        raise ValueError("Invalid configuration")
    
    print(f"📋 Loaded configuration: {config['dataset_name']}")
    print(f"📁 Output directory: {config['output_directory']}")
    
    # Setup dataset
    dataset_manager = DatasetManager(config['output_directory'])
    dataset_path = dataset_manager.create_dataset_structure(config['dataset_name'])
    
    all_samples = []
    
    # Process each generator
    for gen_name, gen_config in config['generators'].items():
        if not gen_config.get('enabled', False):
            continue
            
        print(f"\n🔄 Processing {gen_name} generator...")
        
        # Create generator config
        generator_config = GeneratorConfig(
            language=gen_name if gen_name != 'english' else 'en',
            **gen_config['config']
        )
        
        # Set augmentation based on global config
        if 'augmentation' in config:
            generator_config.augmentation = config['augmentation'].get('enabled', True)
        
        # Create appropriate generator
        if gen_name == 'digits':
            generator = DigitGenerator(generator_config)
        elif gen_name == 'english':
            generator = TextGenerator(generator_config)
        elif gen_name == 'chinese':
            generator = ChineseGenerator(generator_config)
        else:
            print(f"⚠️  Unknown generator: {gen_name}")
            continue
        
        # Generate samples
        samples = generator.generate_batch(
            batch_size=gen_config['samples'],
            output_dir=os.path.join(dataset_path, gen_name),
            name_prefix=gen_name,
            save_labels=True
        )
        
        all_samples.extend(samples)
        print(f"✅ Generated {len(samples)} {gen_name} samples")
    
    # Create final annotations
    output_config = config.get('output', {})
    annotation_format = output_config.get('annotation_format', 'json')
    
    annotation_path = os.path.join(dataset_path, f"all_annotations.{annotation_format}")
    dataset_manager.create_annotation_file(all_samples, annotation_path, annotation_format)
    
    # Generate summary
    summary = {
        "dataset_name": config['dataset_name'],
        "total_samples": len(all_samples),
        "configuration": config,
        "generators_used": list(config['generators'].keys()),
        "output_path": dataset_path
    }
    
    with open(os.path.join(dataset_path, "generation_summary.json"), 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n🎉 Dataset generation complete!")
    print(f"📂 Location: {dataset_path}")
    print(f"📊 Total samples: {len(all_samples)}")
    print(f"📄 Annotations: {annotation_path}")
    
    return dataset_path, all_samples


def main():
    """Main function"""
    print("⚙️  OCR Data Generator - Configuration Example")
    print("=" * 60)
    
    # Step 1: Create configuration template
    print("\n1️⃣  Creating configuration template...")
    config_path = create_config_file()
    
    print(f"\n📝 Please review and modify the configuration file: {config_path}")
    print("   You can adjust:")
    print("   - Number of samples for each generator")
    print("   - Text length ranges")
    print("   - Font sizes")
    print("   - Augmentation settings")
    print("   - Output formats")
    
    # Ask user if they want to continue with default config
    response = input(f"\nContinue with default configuration? (y/n): ")
    
    if response.lower() not in ['y', 'yes']:
        print(f"Please modify {config_path} and run this script again.")
        return
    
    # Step 2: Generate dataset from configuration
    print(f"\n2️⃣  Generating dataset from configuration...")
    
    try:
        dataset_path, samples = generate_from_config(config_path)
        
        # Step 3: Show usage examples
        print(f"\n3️⃣  Usage Examples:")
        print(f"\n📖 How to use this dataset:")
        print(f"   1. Training data location: {dataset_path}")
        print(f"   2. Annotations file: {dataset_path}/all_annotations.json")
        print(f"   3. Total samples: {len(samples)}")
        
        print(f"\n💡 Next steps:")
        print(f"   - Split dataset into train/val/test sets")
        print(f"   - Load annotations in your training script")
        print(f"   - Apply additional preprocessing if needed")
        
        print(f"\n🐍 Python code to load annotations:")
        print(f"```python")
        print(f"import json")
        print(f"with open('{dataset_path}/all_annotations.json', 'r') as f:")
        print(f"    annotations = json.load(f)")
        print(f"for sample in annotations:")
        print(f"    image_path = sample['path']")
        print(f"    text_label = sample['text']")
        print(f"    # Process your training data...")
        print(f"```")
        
    except Exception as e:
        print(f"❌ Error generating dataset: {e}")
        raise


if __name__ == "__main__":
    main()