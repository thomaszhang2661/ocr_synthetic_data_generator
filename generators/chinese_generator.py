"""
Chinese text generator for OCR training data
"""
import random
import os
from typing import List, Optional, Dict, Any
from PIL import Image, ImageDraw, ImageFont

from ..core.base_generator import BaseGenerator
from ..config.settings import Config, GeneratorConfig


class ChineseGenerator(BaseGenerator):
    """Generator for Chinese text sequences"""
    
    def __init__(self, config: GeneratorConfig = None, char_dict: dict = None):
        """
        Initialize Chinese generator
        
        Args:
            config: Generator configuration
            char_dict: Dictionary of Chinese characters to use
        """
        if config is None:
            config = GeneratorConfig(language="zh")
        
        super().__init__(config)
        
        # Chinese-specific settings
        self.min_length = getattr(config, 'min_length', 1)
        self.max_length = getattr(config, 'max_length', 10)
        self.char_dict = char_dict or self._load_default_chars()
        self.char_list = list(self.char_dict.keys()) if self.char_dict else []
        
        # Punctuation settings
        self.include_punctuation = getattr(config, 'include_punctuation', True)
        self.chinese_punctuation = ['。', '，', '？', '！', '；', '：', '"', '"', ''', ''', '（', '）', '、']
        
        if not self.char_list:
            self.logger.warning("No Chinese characters loaded")
    
    def _load_default_chars(self) -> Dict[str, int]:
        """Load default Chinese character set"""
        # Common Chinese characters (simplified)
        common_chars = [
            '的', '一', '是', '了', '我', '不', '人', '在', '他', '有', '这', '个', '上', '们', '来', '到',
            '时', '大', '地', '为', '子', '中', '你', '说', '生', '国', '年', '着', '就', '那', '和', '要',
            '她', '出', '也', '得', '里', '后', '自', '以', '会', '家', '可', '下', '而', '过', '天', '去',
            '能', '对', '小', '多', '然', '于', '心', '学', '么', '之', '都', '好', '看', '起', '发', '当',
            '没', '成', '只', '如', '事', '把', '还', '用', '第', '样', '道', '想', '作', '种', '开', '美',
            '乖', '让', '相', '本', '关', '老', '头', '手', '高', '一', '三', '两', '长', '明', '行', '见',
            '问', '名', '主', '进', '示', '己', '所', '次', '白', '应', '女', '其', '最', '感', '现', '向',
            '外', '由', '此', '如', '前', '性', '华', '通', '先', '回', '实', '内', '情', '必', '全', '常',
            '满', '战', '南', '与', '什', '至', '些', '度', '家', '电', '力', '里', '如', '水', '化', '高',
            '自', '二', '理', '起', '小', '物', '现', '实', '加', '量', '都', '两', '体', '制', '机', '当',
            '使', '点', '从', '业', '本', '去', '把', '性', '好', '应', '开', '它', '合', '还', '因', '由',
            '其', '些', '然', '前', '外', '天', '政', '四', '日', '那', '社', '义', '事', '平', '形', '相',
            '全', '表', '间', '样', '与', '关', '各', '重', '新', '线', '内', '数', '正', '心', '反', '你',
            '明', '看', '原', '又', '么', '利', '比', '或', '但', '质', '气', '第', '向', '道', '命', '此',
            '变', '条', '只', '没', '结', '解', '问', '意', '建', '月', '公', '无', '系', '军', '很', '情',
            '者', '最', '立', '代', '想', '已', '通', '并', '提', '直', '题', '党', '程', '展', '五', '果',
            '料', '象', '员', '革', '位', '入', '常', '文', '总', '次', '品', '式', '活', '设', '及', '管',
            '特', '件', '长', '求', '老', '头', '基', '资', '边', '流', '路', '级', '少', '图', '山', '统',
            '接', '知', '较', '将', '组', '见', '计', '别', '她', '手', '角', '期', '根', '论', '运', '农',
            '指', '几', '九', '区', '强', '放', '决', '西', '被', '干', '做', '必', '战', '先', '回', '则',
            '任', '取', '据', '处', '队', '南', '给', '色', '光', '门', '即', '保', '治', '北', '造', '百',
            '规', '热', '领', '七', '海', '口', '东', '导', '器', '压', '志', '世', '金', '增', '争', '济',
            '阶', '油', '思', '术', '极', '交', '受', '联', '什', '认', '六', '共', '权', '收', '证', '改',
            '清', '美', '再', '采', '转', '更', '单', '风', '切', '打', '白', '教', '速', '花', '带', '安',
            '场', '身', '车', '例', '真', '务', '具', '万', '每', '目', '至', '达', '走', '积', '示', '议',
            '声', '报', '斗', '完', '类', '八', '离', '华', '名', '确', '才', '科', '张', '信', '马', '节',
            '话', '米', '整', '空', '元', '况', '今', '集', '温', '传', '土', '许', '步', '群', '广', '石',
            '记', '需', '段', '研', '界', '拉', '林', '律', '叫', '且', '究', '观', '越', '织', '装', '影',
            '算', '低', '持', '音', '众', '书', '布', '复', '容', '儿', '须', '际', '商', '非', '验', '连',
            '断', '深', '难', '近', '矿', '千', '周', '委', '素', '技', '备', '半', '办', '青', '省', '列',
            '习', '响', '约', '支', '般', '史', '感', '劳', '便', '团', '往', '酸', '历', '市', '克', '何',
            '除', '消', '构', '府', '称', '太', '准', '精', '值', '号', '率', '族', '维', '划', '选', '标',
            '写', '存', '候', '毛', '亲', '快', '效', '斯', '院', '查', '江', '型', '眼', '王', '按', '格',
            '养', '易', '置', '派', '层', '片', '始', '却', '专', '状', '育', '厂', '京', '识', '适', '属',
            '圆', '包', '火', '住', '调', '满', '县', '局', '照', '参', '红', '细', '引', '听', '该', '铁',
            '价', '严', '龙', '飞'
        ]
        
        return {char: i for i, char in enumerate(common_chars)}
    
    def generate_text(self, length: int = None) -> str:
        """
        Generate random Chinese text
        
        Args:
            length: Length of text (random if None)
            
        Returns:
            Random Chinese text string
        """
        if not self.char_list:
            raise ValueError("No Chinese characters available")
        
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        # Choose generation method
        method = random.choice(['random_chars', 'word_combinations', 'sentences'])
        
        if method == 'random_chars':
            return self._generate_random_chars(length)
        elif method == 'word_combinations':
            return self._generate_word_combinations(length)
        else:
            return self._generate_sentences(length)
    
    def _generate_random_chars(self, length: int) -> str:
        """Generate random Chinese characters"""
        chars = random.choices(self.char_list, k=length)
        
        # Optionally add punctuation
        if self.include_punctuation and length > 2 and random.random() < 0.3:
            # Insert punctuation at random position
            punct_pos = random.randint(1, length - 1)
            punct = random.choice(self.chinese_punctuation)
            chars.insert(punct_pos, punct)
        
        return ''.join(chars)
    
    def _generate_word_combinations(self, target_length: int) -> str:
        """Generate combinations that form word-like structures"""
        # Common Chinese word patterns
        word_patterns = [
            lambda: self._get_random_chars(2),  # Two-character words
            lambda: self._get_random_chars(3),  # Three-character words
            lambda: self._get_random_chars(4),  # Four-character words
            lambda: self._get_random_chars(1),  # Single characters
        ]
        
        result = []
        current_length = 0
        
        while current_length < target_length:
            # Choose word pattern
            pattern = random.choices(
                word_patterns,
                weights=[0.5, 0.2, 0.1, 0.2]  # Favor 2-character words
            )[0]
            
            word = pattern()
            if current_length + len(word) <= target_length:
                result.append(word)
                current_length += len(word)
            else:
                # Add partial word if needed
                remaining = target_length - current_length
                if remaining > 0:
                    result.append(word[:remaining])
                break
        
        return ''.join(result)
    
    def _generate_sentences(self, target_length: int) -> str:
        """Generate sentence-like Chinese text"""
        # Common sentence patterns in Chinese
        result = []
        current_length = 0
        
        while current_length < target_length:
            # Generate a clause
            clause_length = random.randint(2, min(6, target_length - current_length))
            clause = self._get_random_chars(clause_length)
            result.append(clause)
            current_length += len(clause)
            
            # Add punctuation if appropriate
            if (current_length < target_length - 1 and 
                random.random() < 0.4 and 
                self.include_punctuation):
                punct = random.choice(['，', '。', '？', '！'])
                result.append(punct)
                current_length += 1
        
        text = ''.join(result)
        return text[:target_length]
    
    def _get_random_chars(self, count: int) -> str:
        """Get random Chinese characters"""
        return ''.join(random.choices(self.char_list, k=count))
    
    def create_base_image(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """
        Create base image with Chinese text
        
        Args:
            text: Chinese text to render
            font: Font object
            
        Returns:
            PIL Image with rendered text
        """
        # Calculate text dimensions
        temp_img = Image.new('RGB', (1, 1))
        temp_draw = ImageDraw.Draw(temp_img)
        bbox = temp_draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Chinese characters are typically square-ish, so adjust dimensions
        char_size = self.config.font_size
        estimated_width = len(text) * char_size
        estimated_height = char_size + 20  # Some padding
        
        # Use estimated dimensions if text bbox failed
        if text_width <= 0:
            text_width = estimated_width
        if text_height <= 0:
            text_height = estimated_height
        
        # Add padding
        padding = 10
        width = text_width + 2 * padding
        height = max(text_height + 2 * padding, estimated_height)
        
        # Create image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Calculate text position
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, font=font, fill='black')
        
        return image
    
    def create_line_image(self, texts: List[str], font: ImageFont.FreeTypeFont) -> Image.Image:
        """
        Create image with multiple Chinese texts in a line
        
        Args:
            texts: List of Chinese text pieces
            font: Font object
            
        Returns:
            PIL Image with line of text
        """
        if not texts:
            return Image.new('RGB', (100, 50), color='white')
        
        # Calculate total dimensions
        char_size = self.config.font_size
        total_chars = sum(len(text) for text in texts)
        spacing = char_size // 4  # Space between text pieces
        
        width = total_chars * char_size + (len(texts) - 1) * spacing + 40  # padding
        height = char_size + 40  # padding
        
        # Create image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Draw texts
        x_offset = 20  # left padding
        y = height // 2
        
        for text in texts:
            # Draw text
            draw.text((x_offset, y), text, font=font, fill='black', anchor='lm')
            
            # Calculate width of current text for next offset
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            x_offset += text_width + spacing
        
        return image
    
    def generate_poem_like_text(self, num_lines: int = 4, chars_per_line: int = 7) -> List[str]:
        """
        Generate poem-like Chinese text
        
        Args:
            num_lines: Number of lines in poem
            chars_per_line: Characters per line
            
        Returns:
            List of text lines
        """
        lines = []
        for _ in range(num_lines):
            line = self._get_random_chars(chars_per_line)
            lines.append(line)
        
        return lines
    
    def generate_address_text(self) -> str:
        """Generate Chinese address-like text"""
        # Common Chinese address components
        provinces = ['北京', '上海', '广东', '浙江', '江苏', '山东', '河北', '河南', '湖北', '湖南']
        cities = ['市', '县', '区']
        streets = ['街', '路', '巷', '大道', '小区']
        
        province = random.choice(provinces)
        city_suffix = random.choice(cities)
        street_suffix = random.choice(streets)
        
        # Generate components
        city_name = self._get_random_chars(2) + city_suffix
        street_name = self._get_random_chars(random.randint(2, 4)) + street_suffix
        number = str(random.randint(1, 999)) + '号'
        
        return f"{province}{city_name}{street_name}{number}"
    
    def generate_name_text(self) -> str:
        """Generate Chinese name-like text"""
        # Common Chinese surnames
        surnames = ['王', '李', '张', '刘', '陈', '杨', '黄', '赵', '周', '吴', '徐', '孙', '朱', '马', '胡']
        
        surname = random.choice(surnames)
        
        # Generate given name (1-2 characters)
        given_name_length = random.choice([1, 2])
        given_name = self._get_random_chars(given_name_length)
        
        return surname + given_name


class TraditionalChineseGenerator(ChineseGenerator):
    """Generator for Traditional Chinese characters"""
    
    def __init__(self, config: GeneratorConfig = None):
        # Load traditional characters if available
        traditional_chars = self._load_traditional_chars()
        super().__init__(config, traditional_chars)
    
    def _load_traditional_chars(self) -> Dict[str, int]:
        """Load traditional Chinese character set"""
        # This would load traditional Chinese characters
        # For demo purposes, using a small subset
        traditional_chars = [
            '的', '一', '是', '了', '我', '不', '人', '在', '他', '有', '這', '個', '上', '們', '來', '到',
            '時', '大', '地', '為', '子', '中', '你', '說', '生', '國', '年', '著', '就', '那', '和', '要',
            '她', '出', '也', '得', '裡', '後', '自', '以', '會', '家', '可', '下', '而', '過', '天', '去',
            '能', '對', '小', '多', '然', '於', '心', '學', '麼', '之', '都', '好', '看', '起', '發', '當',
            '沒', '成', '只', '如', '事', '把', '還', '用', '第', '樣', '道', '想', '作', '種', '開', '美'
        ]
        
        return {char: i for i, char in enumerate(traditional_chars)}