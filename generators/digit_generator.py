"""
Digital number generator for OCR training data
"""
import random
import string
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from ..core.base_generator import BaseGenerator
from ..config.settings import Config, GeneratorConfig


class DigitGenerator(BaseGenerator):
    """Generator for digital number sequences"""
    
    def __init__(self, config: GeneratorConfig = None):
        """
        Initialize digit generator
        
        Args:
            config: Generator configuration
        """
        if config is None:
            config = GeneratorConfig(language="digits")
        
        super().__init__(config)
        
        # Digit-specific settings
        self.min_length = getattr(config, 'min_length', 1)
        self.max_length = getattr(config, 'max_length', 15)
        self.cell_width = getattr(config, 'cell_width', 40)
        self.add_separators = getattr(config, 'add_separators', False)
        self.separator_chars = getattr(config, 'separator_chars', ['-', ' ', '.'])
        
    def generate_text(self, length: int = None) -> str:
        """
        Generate random digit sequence
        
        Args:
            length: Length of digit sequence (random if None)
            
        Returns:
            Random digit string
        """
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        digits = ''.join(random.choices(Config.DIGITS, k=length))
        
        # Optionally add separators
        if self.add_separators and length > 3:
            # Add separator at random positions
            separator_positions = random.sample(range(1, length), 
                                              random.randint(1, min(2, length-1)))
            
            result = []
            for i, digit in enumerate(digits):
                if i in separator_positions:
                    result.append(random.choice(self.separator_chars))
                result.append(digit)
            
            return ''.join(result)
        
        return digits
    
    def create_base_image(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """
        Create base image with digit text
        
        Args:
            text: Digit text to render
            font: Font object
            
        Returns:
            PIL Image with rendered text
        """
        # Calculate image dimensions
        char_count = len([c for c in text if c.isdigit()])
        separator_count = len(text) - char_count
        
        # Use individual cell approach for better control
        width = self.cell_width * char_count + separator_count * (self.cell_width // 2)
        height = getattr(self.config, 'image_height', 64)
        
        # Create main image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        x_offset = 0
        
        for char in text:
            if char.isdigit():
                # Create cell for digit
                cell_image = self._create_digit_cell(char, font)
                cell_width = cell_image.size[0]
                
                # Paste digit cell
                y_offset = (height - cell_image.size[1]) // 2
                image.paste(cell_image, (x_offset, y_offset))
                x_offset += cell_width
                
            else:
                # Handle separator
                separator_width = self.cell_width // 2
                text_x = x_offset + separator_width // 4
                text_y = height // 2
                
                draw.text((text_x, text_y), char, font=font, fill='black', anchor='mm')
                x_offset += separator_width
        
        return image
    
    def _create_digit_cell(self, digit: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """
        Create individual cell for a digit with variations
        
        Args:
            digit: Single digit character
            font: Font object
            
        Returns:
            PIL Image of digit cell
        """
        cell_height = getattr(self.config, 'image_height', 64)
        
        # Create cell image
        cell_image = Image.new('L', (self.cell_width, cell_height), color=255)
        draw = ImageDraw.Draw(cell_image)
        
        # Add slight random positioning for naturalness
        x_jitter = random.randint(-2, 2)
        y_jitter = random.randint(-2, 2)
        
        # Calculate text position
        text_x = self.cell_width // 2 + x_jitter
        text_y = cell_height // 2 + y_jitter
        
        # Draw digit
        draw.text((text_x, text_y), digit, font=font, fill=0, anchor='mm')
        
        # Apply random scaling
        if random.random() < 0.3:  # 30% chance
            scale_factor = random.uniform(0.9, 1.1)
            new_width = int(self.cell_width * scale_factor)
            new_height = int(cell_height * scale_factor)
            cell_image = cell_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Pad back to original size
            if new_width != self.cell_width or new_height != cell_height:
                padded = Image.new('L', (self.cell_width, cell_height), 255)
                x_pad = (self.cell_width - new_width) // 2
                y_pad = (cell_height - new_height) // 2
                padded.paste(cell_image, (x_pad, y_pad))
                cell_image = padded
        
        return cell_image
    
    def create_student_id(self, id_length: int = 10) -> str:
        """
        Generate student ID format numbers
        
        Args:
            id_length: Length of student ID
            
        Returns:
            Student ID string
        """
        # Common student ID patterns
        patterns = [
            lambda: ''.join(random.choices('123456789', k=id_length)),  # All digits
            lambda: f"20{random.randint(18, 24)}" + ''.join(random.choices(Config.DIGITS, k=id_length-4)),  # Year prefix
            lambda: f"{random.randint(1, 9)}" + ''.join(random.choices(Config.DIGITS, k=id_length-1)),  # No leading zero
        ]
        
        return random.choice(patterns)()
    
    def create_handwritten_style(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """
        Create handwritten-style digit image using MNIST-like approach
        
        Args:
            text: Digit text
            font: Font object (may be ignored for handwritten style)
            
        Returns:
            Handwritten-style image
        """
        # This would integrate with MNIST data if available
        # For now, create a more casual printed style
        
        width = len(text) * self.cell_width
        height = getattr(self.config, 'image_height', 64)
        
        image = Image.new('L', (width, height), color=255)
        draw = ImageDraw.Draw(image)
        
        x_offset = 0
        for digit in text:
            if digit.isdigit():
                # Add more variation for handwritten feel
                x_jitter = random.randint(-5, 5)
                y_jitter = random.randint(-3, 3)
                rotation = random.uniform(-3, 3)
                
                # Create temporary digit image
                temp_img = Image.new('L', (self.cell_width, height), 255)
                temp_draw = ImageDraw.Draw(temp_img)
                
                text_x = self.cell_width // 2 + x_jitter
                text_y = height // 2 + y_jitter
                
                temp_draw.text((text_x, text_y), digit, font=font, fill=0, anchor='mm')
                
                # Apply rotation
                if abs(rotation) > 0.5:
                    temp_img = temp_img.rotate(rotation, fillcolor=255)
                
                # Paste to main image
                image.paste(temp_img, (x_offset, 0))
                x_offset += self.cell_width
        
        return image


class PrintedDigitGenerator(DigitGenerator):
    """Specialized generator for printed digit sequences"""
    
    def __init__(self, config: GeneratorConfig = None):
        if config is None:
            config = GeneratorConfig(language="digits")
        
        # Set printed-specific defaults
        config.extra_params.update({
            'cell_width': 40,
            'add_borders': True,
            'add_separators': True
        })
        
        super().__init__(config)
    
    def create_base_image(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """Create printed-style digit image with borders"""
        image = super().create_base_image(text, font)
        
        # Add border if requested
        if self.config.extra_params.get('add_borders', False):
            image = self.image_processor.add_border(image, border_width=2)
        
        return image


class HandwrittenDigitGenerator(DigitGenerator):
    """Specialized generator for handwritten digit sequences"""
    
    def __init__(self, config: GeneratorConfig = None, mnist_data: dict = None):
        if config is None:
            config = GeneratorConfig(language="digits")
        
        super().__init__(config)
        self.mnist_data = mnist_data or {}
    
    def create_base_image(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """Create handwritten-style image"""
        if self.mnist_data:
            return self._create_from_mnist(text)
        else:
            return self.create_handwritten_style(text, font)
    
    def _create_from_mnist(self, text: str) -> Image.Image:
        """Create image using MNIST data if available"""
        # Implementation would use pre-loaded MNIST digit images
        # This is a placeholder for the actual MNIST integration
        return self.create_handwritten_style(text, self._get_random_font(self.config.font_size))