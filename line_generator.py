#!/usr/bin/env python3
"""
Handwriting Line Generator - 基于字符图片拼接生成行图像

这个脚本实现了您的原始项目核心功能：
1. 从单个汉字图片字典加载字符图像
2. 使用语料库文本生成行图像
3. 支持背景添加和数据增强

用法示例：
    python line_generator.py --chars ./chinese_data1018/data_自动化所 
                           --dict ./merged_dict.txt 
                           --corpus ./corpus 
                           --output ./line_output 
                           --samples 1000
"""

import argparse
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from ocr_data_generator.core.line_composer import HandwritingLineGenerator
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("Please install required dependencies:")
    print("pip install Pillow numpy opencv-python tqdm")
    sys.exit(1)


def create_parser():
    """创建命令行参数解析器"""
    parser = argparse.ArgumentParser(
        description="Handwriting Line Generator - 基于字符图片拼接生成行图像",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  基础用法（生成1000个行图像）:
    python line_generator.py --chars ./chinese_data1018/data_自动化所 
                           --dict ./merged_dict.txt 
                           --corpus ./corpus 
                           --samples 1000
  
  使用自定义配置:
    python line_generator.py --chars ./chinese_data1018/data_自动化所 
                           --dict ./merged_dict.txt 
                           --corpus ./corpus 
                           --corpus-files all_corpus_standard.txt
                           --background ./backgrounds
                           --samples 500
                           --output ./my_output
        """
    )
    
    # 必需参数
    parser.add_argument('--chars', required=True, type=str,
                       help='字符图片目录路径（包含单个汉字图片）')
    parser.add_argument('--dict', required=True, type=str,
                       help='字符字典文件路径（char : code 格式）')
    
    # 可选参数
    parser.add_argument('--corpus', type=str, default='./corpus',
                       help='语料库目录路径（默认: ./corpus）')
    parser.add_argument('--corpus-files', nargs='+', 
                       default=['all_corpus_standard.txt'],
                       help='语料库文件名列表（默认: all_corpus_standard.txt）')
    parser.add_argument('--background', type=str,
                       help='背景图片目录（可选）')
    parser.add_argument('--output', type=str, default='./line_output',
                       help='输出目录（默认: ./line_output）')
    parser.add_argument('--samples', type=int, default=100,
                       help='生成样本数量（默认: 100）')
    
    # 增强选项
    parser.add_argument('--no-background', action='store_true',
                       help='禁用背景添加')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='禁用数据增强')
    parser.add_argument('--random-text', action='store_true',
                       help='使用随机文本而不是语料库')
    
    return parser


def validate_paths(args):
    """验证输入路径"""
    errors = []
    
    if not os.path.exists(args.chars):
        errors.append(f"字符图片目录不存在: {args.chars}")
    
    if not os.path.exists(args.dict):
        errors.append(f"字符字典文件不存在: {args.dict}")
    
    if not args.random_text and not os.path.exists(args.corpus):
        errors.append(f"语料库目录不存在: {args.corpus}")
    
    if args.background and not os.path.exists(args.background):
        errors.append(f"背景图片目录不存在: {args.background}")
    
    if errors:
        print("❌ 路径验证失败:")
        for error in errors:
            print(f"   {error}")
        return False
    
    return True


def main():
    """主函数"""
    parser = create_parser()
    args = parser.parse_args()
    
    print("🚀 Handwriting Line Generator")
    print("=" * 50)
    print(f"基于字符图片拼接生成行图像")
    print(f"字符图片目录: {args.chars}")
    print(f"字符字典文件: {args.dict}")
    print(f"语料库目录: {args.corpus}")
    print(f"输出目录: {args.output}")
    print(f"生成样本数: {args.samples}")
    print()
    
    # 验证路径
    if not validate_paths(args):
        sys.exit(1)
    
    try:
        # 创建配置
        config = {
            'char_dict_path': args.dict,
            'char_image_directory': args.chars,
            'corpus_directory': args.corpus,
            'corpus_files': args.corpus_files,
            'add_background': not args.no_background,
            'apply_augmentation': not args.no_augmentation
        }
        
        if args.background:
            config['background_directory'] = args.background
        
        print("🔧 初始化生成器...")
        generator = HandwritingLineGenerator(config)
        
        print("📝 开始生成行图像...")
        samples = generator.generate_line_samples(
            num_samples=args.samples,
            output_directory=args.output,
            use_corpus=not args.random_text
        )
        
        print()
        print("✅ 生成完成!")
        print(f"📊 统计信息:")
        print(f"   成功生成: {len(samples)} 个样本")
        print(f"   输出目录: {args.output}")
        
        if samples:
            # 显示一些示例
            print(f"📝 示例文本:")
            for i, sample in enumerate(samples[:5]):
                print(f"   {i+1}. {sample['text']}")
            
            # 统计信息
            avg_length = sum(len(s['text']) for s in samples) / len(samples)
            print(f"📏 平均文本长度: {avg_length:.1f} 字符")
        
        print()
        print("💡 下一步:")
        print("   1. 检查输出目录中的生成图像")
        print("   2. 查看 labels.json 文件获取标注信息")
        print("   3. 将数据用于OCR模型训练")
        
    except Exception as e:
        print(f"❌ 生成过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()