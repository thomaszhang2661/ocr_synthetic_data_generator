#!/usr/bin/env python3
"""
字符拼接行图像生成示例

这个示例展示了项目的核心功能：
1. 从单个汉字图片字典加载字符
2. 使用语料库文本生成行图像
3. 输出带标注的训练数据

这是您原始项目的主要功能实现。
"""

import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ocr_data_generator import (
        CharacterImageLoader, 
        CorpusProcessor, 
        LineImageComposer,
        HandwritingLineGenerator
    )
except ImportError as e:
    print(f"❌ 导入错误: {e}")
    print("请确保安装了必要的依赖:")
    print("pip install Pillow numpy opencv-python tqdm")
    sys.exit(1)


def demo_character_composition():
    """演示字符拼接功能"""
    print("🚀 字符拼接行图像生成演示")
    print("=" * 50)
    
    # 配置路径（需要根据您的实际路径调整）
    config = {
        # 字符字典文件 - 包含字符到编码的映射
        'char_dict_path': '../chinese_pseudo/merged_dict.txt',
        
        # 单个字符图片目录 - 包含所有单字符图片
        'char_image_directory': '../chinese_pseudo/chinese_data1018/data_自动化所',
        
        # 语料库目录 - 包含文本文件
        'corpus_directory': '../chinese_pseudo/corpus',
        
        # 语料库文件列表
        'corpus_files': ['all_corpus_standard.txt'],
        
        # 背景图片目录（可选）
        'background_directory': '../chinese_pseudo/backgrounds',
        
        # 是否添加背景和数据增强
        'add_background': True,
        'apply_augmentation': True
    }
    
    # 检查必要文件是否存在
    required_files = [
        config['char_dict_path'],
        config['char_image_directory']
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  以下必要文件/目录不存在:")
        for file_path in missing_files:
            print(f"   {file_path}")
        print()
        print("💡 这是一个演示脚本，请根据您的实际文件路径调整config中的路径设置")
        print("   主要需要:")
        print("   1. 字符字典文件（char : code 格式）")
        print("   2. 单个字符图片目录")
        print("   3. 语料库文本文件（可选）")
        return
    
    try:
        print("🔧 初始化字符拼接生成器...")
        
        # 创建生成器
        generator = HandwritingLineGenerator(config)
        
        print("📊 系统信息:")
        print(f"   可用字符数: {len(generator.character_loader.get_available_characters())}")
        print(f"   可用字体数: {len(generator.character_loader.get_available_fonts())}")
        print(f"   语料库行数: {len(generator.corpus_processor.corpus_lines)}")
        
        # 生成一些示例
        print()
        print("📝 生成示例行图像...")
        
        samples = generator.generate_line_samples(
            num_samples=10,
            output_directory='./demo_output',
            use_corpus=True
        )
        
        print()
        print("✅ 生成完成!")
        print(f"📊 统计:")
        print(f"   成功生成: {len(samples)} 个样本")
        print(f"   输出目录: ./demo_output")
        
        if samples:
            print()
            print("📝 生成的文本示例:")
            for i, sample in enumerate(samples[:5], 1):
                print(f"   {i}. {sample['text']}")
                print(f"      图像尺寸: {sample['image_size']}")
        
        print()
        print("🎯 核心功能说明:")
        print("   ✓ 单字符图片加载和管理")
        print("   ✓ 语料库文本处理")
        print("   ✓ 字符图片拼接成行")
        print("   ✓ 背景添加和数据增强")
        print("   ✓ 自动标注生成")
        
    except Exception as e:
        print(f"❌ 执行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


def demo_individual_components():
    """演示各个组件的独立功能"""
    print()
    print("🔧 组件功能演示")
    print("=" * 30)
    
    try:
        # 演示字符加载器
        print("1. 字符图片加载器演示:")
        
        # 模拟配置（使用相对路径）
        char_dict_path = '../chinese_pseudo/merged_dict.txt'
        char_image_dir = '../chinese_pseudo/chinese_data1018/data_自动化所'
        
        if os.path.exists(char_dict_path) and os.path.exists(char_image_dir):
            loader = CharacterImageLoader(char_dict_path, char_image_dir)
            
            available_chars = loader.get_available_characters()[:10]  # 前10个字符
            print(f"   可用字符示例: {available_chars}")
            
            # 尝试获取一个字符的图片
            if available_chars:
                char = available_chars[0]
                img = loader.get_character_image(char)
                if img is not None:
                    print(f"   字符 '{char}' 图片尺寸: {img.shape}")
        else:
            print("   (需要实际字符数据文件)")
        
        print()
        print("2. 语料库处理器演示:")
        
        corpus_dir = '../chinese_pseudo/corpus'
        if os.path.exists(corpus_dir):
            processor = CorpusProcessor(corpus_dir)
            processor.load_corpus_files(['all_corpus_standard.txt'])
            
            if processor.corpus_lines:
                random_line = processor.get_random_line()
                print(f"   随机语料行: {random_line[:50]}...")
                print(f"   总行数: {len(processor.corpus_lines)}")
        else:
            print("   (需要实际语料库文件)")
        
        print()
        print("3. 行图像合成器演示:")
        print("   (需要配合字符加载器使用)")
        print("   主要功能: 字符拼接、背景添加、数据增强")
        
    except Exception as e:
        print(f"   演示过程中出现错误: {e}")


if __name__ == "__main__":
    print("🎯 这是您原始项目的核心功能实现")
    print("   核心理念: 单字符图片 + 语料库文本 → 行图像")
    print()
    
    # 主要演示
    demo_character_composition()
    
    # 组件演示
    demo_individual_components()
    
    print()
    print("💡 使用说明:")
    print("   1. 准备单个字符图片字典")
    print("   2. 准备字符编码映射文件")  
    print("   3. 准备语料库文本文件")
    print("   4. 运行生成器创建训练数据")
    print()
    print("📁 这个实现完全基于您的原始项目思路：")
    print("   输入: 单字符图片 + 语料库")
    print("   处理: 字符拼接 + 增强")
    print("   输出: 带标注的行图像")