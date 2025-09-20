"""
OCR Line Image Composer - 核心字符拼接器

这个模块的核心功能是：
1. 从单个字符图片字典中加载字符图像
2. 根据语料库文本将字符图片拼接成行图像
3. 添加背景、变换效果等增强
4. 生成带标注的训练数据

作者：基于原始项目重构
"""

import os
import numpy as np
from PIL import Image, ImageDraw
import random
import json
from typing import Dict, List, Tuple, Optional
import cv2
from tqdm import tqdm


class CharacterImageLoader:
    """单字符图片加载器和管理器"""
    
    def __init__(self, char_dict_path: str, image_directory: str):
        """
        初始化字符图片加载器
        
        Args:
            char_dict_path: 字符字典文件路径 (char : code 格式)
            image_directory: 字符图片目录路径
        """
        self.char_dict_path = char_dict_path
        self.image_directory = image_directory
        self.char_dict = {}
        self.char_dict_reverse = {}
        self.char_images = {}  # {char: {font_style: image_array}}
        
        self._load_char_dict()
        self._load_character_images()
    
    def _load_char_dict(self):
        """加载字符字典"""
        print("Loading character dictionary...")
        with open(self.char_dict_path, 'r', encoding='utf-8') as f:
            for line in f:
                if ' : ' in line:
                    char, code = line.strip().split(' : ')
                    self.char_dict[char] = int(code)
                    self.char_dict_reverse[int(code)] = char
        print(f"Loaded {len(self.char_dict)} characters")
    
    def _load_character_images(self):
        """加载字符图片到内存"""
        print("Loading character images...")
        
        if not os.path.exists(self.image_directory):
            print(f"Directory {self.image_directory} not found.")
            return
        
        # 遍历图片文件
        files = os.listdir(self.image_directory)
        image_files = [f for f in files if f.endswith(('.jpg', '.png'))]
        
        for filename in tqdm(image_files, desc="Loading character images"):
            try:
                # 解析文件名：font_style_char_code.jpg
                parts = filename.split('_')
                if len(parts) >= 2:
                    font_style = parts[0]
                    char_code = parts[1].split('.')[0]
                    
                    # 根据编码获取字符
                    if char_code.isdigit():
                        char = self.char_dict_reverse.get(int(char_code))
                        if char:
                            filepath = os.path.join(self.image_directory, filename)
                            image = Image.open(filepath).convert('L')
                            
                            if char not in self.char_images:
                                self.char_images[char] = {}
                            self.char_images[char][font_style] = np.array(image)
            except Exception as e:
                print(f"Error loading {filename}: {e}")
        
        print(f"Loaded images for {len(self.char_images)} characters")
    
    def get_character_image(self, char: str, font_style: Optional[str] = None) -> Optional[np.ndarray]:
        """
        获取指定字符的图片
        
        Args:
            char: 目标字符
            font_style: 字体风格，如果为None则随机选择
            
        Returns:
            字符图片的numpy数组，如果不存在则返回None
        """
        if char not in self.char_images:
            return None
        
        available_fonts = list(self.char_images[char].keys())
        if not available_fonts:
            return None
        
        if font_style is None or font_style not in available_fonts:
            font_style = random.choice(available_fonts)
        
        return self.char_images[char][font_style]
    
    def get_available_characters(self) -> List[str]:
        """获取可用字符列表"""
        return list(self.char_images.keys())
    
    def get_available_fonts(self) -> List[str]:
        """获取所有可用字体风格"""
        all_fonts = set()
        for char_fonts in self.char_images.values():
            all_fonts.update(char_fonts.keys())
        return list(all_fonts)


class CorpusProcessor:
    """语料库处理器"""
    
    def __init__(self, corpus_directory: str):
        """
        初始化语料库处理器
        
        Args:
            corpus_directory: 语料库文件目录
        """
        self.corpus_directory = corpus_directory
        self.corpus_lines = []
        
    def load_corpus_files(self, file_list: List[str]):
        """
        加载指定的语料库文件
        
        Args:
            file_list: 语料库文件名列表
        """
        print("Loading corpus files...")
        self.corpus_lines = []
        
        for filename in file_list:
            filepath = os.path.join(self.corpus_directory, filename)
            if os.path.exists(filepath):
                with open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            self.corpus_lines.append(line)
        
        print(f"Loaded {len(self.corpus_lines)} lines from corpus")
    
    def get_random_line(self) -> str:
        """获取随机语料行"""
        if not self.corpus_lines:
            return ""
        return random.choice(self.corpus_lines)
    
    def get_line_by_index(self, index: int) -> str:
        """根据索引获取语料行"""
        if 0 <= index < len(self.corpus_lines):
            return self.corpus_lines[index]
        return ""
    
    def filter_available_text(self, text: str, available_chars: List[str]) -> str:
        """
        过滤文本，只保留有图片的字符
        
        Args:
            text: 原始文本
            available_chars: 可用字符列表
            
        Returns:
            过滤后的文本
        """
        filtered_chars = []
        for char in text:
            if char in available_chars or char.isspace():
                filtered_chars.append(char)
        return ''.join(filtered_chars)


class LineImageComposer:
    """行图像合成器 - 核心功能类"""
    
    def __init__(self, character_loader: CharacterImageLoader):
        """
        初始化行图像合成器
        
        Args:
            character_loader: 字符图片加载器
        """
        self.character_loader = character_loader
        self.background_images = []
    
    def load_background_images(self, background_directory: str):
        """加载背景图片"""
        if not os.path.exists(background_directory):
            print(f"Background directory {background_directory} not found")
            return
        
        print("Loading background images...")
        files = os.listdir(background_directory)
        for filename in files:
            if filename.endswith(('.jpg', '.png')):
                filepath = os.path.join(background_directory, filename)
                try:
                    bg_image = Image.open(filepath).convert('L')
                    self.background_images.append(bg_image)
                except Exception as e:
                    print(f"Error loading background {filepath}: {e}")
        
        print(f"Loaded {len(self.background_images)} background images")
    
    def compose_line_image(self, text: str, font_style: Optional[str] = None, 
                          add_background: bool = True, 
                          apply_augmentation: bool = True) -> Tuple[Optional[Image.Image], str]:
        """
        将文本合成为行图像
        
        Args:
            text: 要合成的文本
            font_style: 字体风格
            add_background: 是否添加背景
            apply_augmentation: 是否应用数据增强
            
        Returns:
            (合成的图像, 实际使用的文本)
        """
        if not text.strip():
            return None, ""
        
        # 过滤可用字符
        available_chars = self.character_loader.get_available_characters()
        filtered_text = ""
        char_images = []
        
        for char in text:
            if char.isspace():
                # 处理空格 - 添加空白区域
                space_width = 20  # 空格宽度
                space_height = 32  # 默认高度
                space_image = Image.new('L', (space_width, space_height), 255)
                char_images.append(np.array(space_image))
                filtered_text += char
            else:
                char_img = self.character_loader.get_character_image(char, font_style)
                if char_img is not None:
                    char_images.append(char_img)
                    filtered_text += char
        
        if not char_images:
            return None, ""
        
        # 拼接字符图片
        line_image = self._concatenate_character_images(char_images)
        
        if line_image is None:
            return None, ""
        
        # 裁剪空白
        line_image = self._crop_whitespace(line_image)
        
        # 添加背景
        if add_background and self.background_images:
            line_image = self._add_background(line_image)
        
        # 应用数据增强
        if apply_augmentation:
            line_image = self._apply_augmentation(line_image)
        
        return line_image, filtered_text
    
    def _concatenate_character_images(self, char_images: List[np.ndarray]) -> Optional[Image.Image]:
        """拼接字符图片"""
        if not char_images:
            return None
        
        # 统一高度到最大高度
        max_height = max(img.shape[0] for img in char_images)
        
        # 调整每个字符图片的高度
        resized_images = []
        total_width = 0
        
        for img in char_images:
            # 如果图片高度小于最大高度，居中放置
            if img.shape[0] < max_height:
                pad_top = (max_height - img.shape[0]) // 2
                pad_bottom = max_height - img.shape[0] - pad_top
                img = np.pad(img, ((pad_top, pad_bottom), (0, 0)), mode='constant', constant_values=255)
            
            resized_images.append(img)
            total_width += img.shape[1]
        
        # 创建合并图像
        line_array = np.full((max_height, total_width), 255, dtype=np.uint8)
        
        current_x = 0
        for img in resized_images:
            line_array[:, current_x:current_x + img.shape[1]] = img
            current_x += img.shape[1]
        
        return Image.fromarray(line_array)
    
    def _crop_whitespace(self, image: Image.Image) -> Image.Image:
        """裁剪空白区域"""
        gray_image = image.convert('L')
        image_array = np.array(gray_image)
        threshold = 230
        
        # 计算非白色像素的边界
        horizontal_sum = np.sum(image_array < threshold, axis=1)
        vertical_sum = np.sum(image_array < threshold, axis=0)
        
        # 找边界
        top = np.argmax(horizontal_sum > 0)
        bottom = len(horizontal_sum) - np.argmax(horizontal_sum[::-1] > 0)
        left = np.argmax(vertical_sum > 0)
        right = len(vertical_sum) - np.argmax(vertical_sum[::-1] > 0)
        
        # 添加小量随机边距
        margin = 3
        top = max(0, top - random.randint(0, margin))
        bottom = min(image_array.shape[0], bottom + random.randint(0, margin))
        left = max(0, left - random.randint(0, margin))
        right = min(image_array.shape[1], right + random.randint(0, margin))
        
        return image.crop((left, top, right, bottom))
    
    def _add_background(self, image: Image.Image) -> Image.Image:
        """添加背景"""
        if not self.background_images:
            return image
        
        background = random.choice(self.background_images)
        return self._blend_with_background(image, background)
    
    def _blend_with_background(self, foreground: Image.Image, background: Image.Image, 
                              threshold: int = 180) -> Image.Image:
        """将前景图像与背景图像混合"""
        fg_array = np.array(foreground)
        fg_height, fg_width = fg_array.shape
        
        # 调整背景尺寸
        if background.size[0] < fg_width or background.size[1] < fg_height:
            background = background.resize((fg_width, fg_height), Image.Resampling.LANCZOS)
        
        bg_array = np.array(background)
        
        # 创建结果图像
        result_array = bg_array[:fg_height, :fg_width].copy()
        
        # 将前景的非白色部分覆盖到背景上
        mask = fg_array < threshold
        result_array[mask] = fg_array[mask]
        
        return Image.fromarray(result_array)
    
    def _apply_augmentation(self, image: Image.Image) -> Image.Image:
        """应用数据增强"""
        image_array = np.array(image)
        
        # 随机应用增强
        if random.random() < 0.3:
            # 亮度调整
            brightness_factor = random.uniform(0.8, 1.2)
            image_array = np.clip(image_array * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.2:
            # 添加噪声
            noise = np.random.normal(0, 5, image_array.shape)
            image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)
        
        if random.random() < 0.1:
            # 轻微模糊
            image_array = cv2.GaussianBlur(image_array, (3, 3), 0.5)
        
        return Image.fromarray(image_array)


class HandwritingLineGenerator:
    """手写体行图像生成器 - 主要接口类"""
    
    def __init__(self, config: dict):
        """
        初始化生成器
        
        Args:
            config: 配置字典
        """
        self.config = config
        
        # 初始化组件
        self.character_loader = CharacterImageLoader(
            char_dict_path=config['char_dict_path'],
            image_directory=config['char_image_directory']
        )
        
        self.corpus_processor = CorpusProcessor(
            corpus_directory=config['corpus_directory']
        )
        
        self.line_composer = LineImageComposer(self.character_loader)
        
        # 加载语料库
        if 'corpus_files' in config:
            self.corpus_processor.load_corpus_files(config['corpus_files'])
        
        # 加载背景图片
        if 'background_directory' in config:
            self.line_composer.load_background_images(config['background_directory'])
    
    def generate_line_samples(self, num_samples: int, output_directory: str, 
                            use_corpus: bool = True) -> List[dict]:
        """
        生成行图像样本
        
        Args:
            num_samples: 生成样本数量
            output_directory: 输出目录
            use_corpus: 是否使用语料库文本
            
        Returns:
            生成样本的元数据列表
        """
        os.makedirs(output_directory, exist_ok=True)
        samples_metadata = []
        
        print(f"Generating {num_samples} line samples...")
        
        for i in tqdm(range(num_samples), desc="Generating samples"):
            try:
                # 获取文本
                if use_corpus and self.corpus_processor.corpus_lines:
                    text = self.corpus_processor.get_random_line()
                    # 限制长度
                    if len(text) > 20:
                        start_idx = random.randint(0, max(0, len(text) - 20))
                        text = text[start_idx:start_idx + 20]
                else:
                    # 生成随机文本
                    available_chars = self.character_loader.get_available_characters()
                    if not available_chars:
                        continue
                    
                    text_length = random.randint(3, 15)
                    text = ''.join(random.choices(available_chars, k=text_length))
                
                # 生成图像
                line_image, actual_text = self.line_composer.compose_line_image(
                    text=text,
                    font_style=None,  # 随机选择字体
                    add_background=self.config.get('add_background', True),
                    apply_augmentation=self.config.get('apply_augmentation', True)
                )
                
                if line_image is not None and actual_text:
                    # 保存图像
                    filename = f"line_{i:06d}.jpg"
                    filepath = os.path.join(output_directory, filename)
                    line_image.save(filepath, quality=90)
                    
                    # 记录元数据
                    metadata = {
                        'filename': filename,
                        'text': actual_text,
                        'path': filepath,
                        'original_text': text,
                        'image_size': line_image.size
                    }
                    samples_metadata.append(metadata)
                
            except Exception as e:
                print(f"Error generating sample {i}: {e}")
        
        # 保存标注文件
        labels_path = os.path.join(output_directory, 'labels.json')
        with open(labels_path, 'w', encoding='utf-8') as f:
            json.dump(samples_metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(samples_metadata)} samples in {output_directory}")
        return samples_metadata


def create_sample_config():
    """创建示例配置"""
    return {
        'char_dict_path': './merged_dict.txt',
        'char_image_directory': './chinese_data1018/data_自动化所',
        'corpus_directory': './corpus',
        'corpus_files': ['all_corpus_standard.txt'],
        'background_directory': './backgrounds',
        'add_background': True,
        'apply_augmentation': True
    }


if __name__ == "__main__":
    # 示例使用
    config = create_sample_config()
    
    generator = HandwritingLineGenerator(config)
    
    # 生成样本
    samples = generator.generate_line_samples(
        num_samples=100,
        output_directory='./output_lines',
        use_corpus=True
    )
    
    print(f"Generated {len(samples)} line image samples")