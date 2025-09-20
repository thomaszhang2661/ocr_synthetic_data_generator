"""
Text generator for English text OCR training data
"""
import random
import string
from typing import List, Optional
from PIL import Image, ImageDraw, ImageFont

from ..core.base_generator import BaseGenerator
from ..config.settings import Config, GeneratorConfig


class TextGenerator(BaseGenerator):
    """Generator for English text sequences"""
    
    def __init__(self, config: GeneratorConfig = None):
        """
        Initialize text generator
        
        Args:
            config: Generator configuration
        """
        if config is None:
            config = GeneratorConfig(language="en")
        
        super().__init__(config)
        
        # Text-specific settings
        self.min_length = getattr(config, 'min_length', 1)
        self.max_length = getattr(config, 'max_length', 20)
        self.include_punctuation = getattr(config, 'include_punctuation', True)
        self.include_numbers = getattr(config, 'include_numbers', True)
        self.word_list = getattr(config, 'word_list', None)
        
        # Character sets
        self.characters = Config.ENGLISH_LETTERS
        if self.include_numbers:
            self.characters += Config.DIGITS
        if self.include_punctuation:
            self.characters += Config.PUNCTUATION
    
    def generate_text(self, length: int = None) -> str:
        """
        Generate random text sequence
        
        Args:
            length: Length of text (random if None)
            
        Returns:
            Random text string
        """
        if length is None:
            length = random.randint(self.min_length, self.max_length)
        
        # Choose generation method
        method = random.choice(['random_chars', 'random_words', 'mixed'])
        
        if method == 'random_chars':
            return self._generate_random_characters(length)
        elif method == 'random_words':
            return self._generate_random_words(length)
        else:
            return self._generate_mixed_content(length)
    
    def _generate_random_characters(self, length: int) -> str:
        """Generate random character sequence"""
        return ''.join(random.choices(self.characters, k=length))
    
    def _generate_random_words(self, target_length: int) -> str:
        """Generate random words to approximate target length"""
        if self.word_list:
            words = self.word_list
        else:
            # Common English words
            words = [
                'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
                'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
                'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy',
                'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use', 'word', 'work',
                'first', 'would', 'there', 'could', 'water', 'after', 'where', 'right',
                'think', 'little', 'world', 'years', 'still', 'place', 'young', 'great',
                'never', 'again', 'school', 'family', 'important', 'different', 'another'
            ]
        
        result = []
        current_length = 0
        
        while current_length < target_length:
            word = random.choice(words)
            if current_length + len(word) <= target_length:
                result.append(word)
                current_length += len(word)
                
                # Add space if not at end
                if current_length < target_length - 1:
                    result.append(' ')
                    current_length += 1
            else:
                # Add partial word if needed
                remaining = target_length - current_length
                if remaining > 0:
                    result.append(word[:remaining])
                break
        
        return ''.join(result)
    
    def _generate_mixed_content(self, length: int) -> str:
        """Generate mixed content with words, numbers, and punctuation"""
        result = []
        current_length = 0
        
        while current_length < length:
            # Choose content type
            content_type = random.choices(
                ['word', 'number', 'punctuation'],
                weights=[0.7, 0.2, 0.1]
            )[0]
            
            if content_type == 'word':
                word_length = random.randint(2, min(8, length - current_length))
                word = ''.join(random.choices(Config.ENGLISH_LETTERS, k=word_length))
                result.append(word)
                current_length += len(word)
                
            elif content_type == 'number':
                num_length = random.randint(1, min(4, length - current_length))
                number = ''.join(random.choices(Config.DIGITS, k=num_length))
                result.append(number)
                current_length += len(number)
                
            else:  # punctuation
                if current_length < length:
                    punct = random.choice(Config.PUNCTUATION)
                    result.append(punct)
                    current_length += 1
            
            # Add space sometimes
            if current_length < length - 1 and random.random() < 0.3:
                result.append(' ')
                current_length += 1
        
        text = ''.join(result)
        return text[:length]  # Ensure exact length
    
    def create_base_image(self, text: str, font: ImageFont.FreeTypeFont) -> Image.Image:
        """
        Create base image with text
        
        Args:
            text: Text to render
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
        
        # Add padding
        padding = 20
        width = text_width + 2 * padding
        height = max(text_height + 2 * padding, self.config.image_size[0])
        
        # Create image
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Calculate text position
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, font=font, fill='black')
        
        return image
    
    def generate_address_text(self) -> str:
        """Generate address-like text"""
        # Street numbers
        street_number = random.randint(1, 9999)
        
        # Street names
        street_names = [
            'Main St', 'Oak Ave', 'First St', 'Second St', 'Park Ave', 'Elm St',
            'Washington St', 'Maple Ave', 'Cedar St', 'Pine St', 'Lake Ave',
            'Hill St', 'Church St', 'School St', 'High St', 'Mill St'
        ]
        
        street_name = random.choice(street_names)
        
        return f"{street_number} {street_name}"
    
    def generate_name_text(self) -> str:
        """Generate name-like text"""
        first_names = [
            'John', 'Jane', 'Michael', 'Sarah', 'David', 'Lisa', 'Robert', 'Mary',
            'James', 'Patricia', 'William', 'Jennifer', 'Richard', 'Elizabeth',
            'Joseph', 'Linda', 'Thomas', 'Barbara', 'Christopher', 'Susan'
        ]
        
        last_names = [
            'Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller',
            'Davis', 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez',
            'Wilson', 'Anderson', 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin'
        ]
        
        first = random.choice(first_names)
        last = random.choice(last_names)
        
        return f"{first} {last}"
    
    def generate_license_plate(self) -> str:
        """Generate license plate text"""
        patterns = [
            lambda: ''.join(random.choices(Config.ENGLISH_LETTERS[:26], k=3)) + 
                   ''.join(random.choices(Config.DIGITS, k=3)),
            lambda: ''.join(random.choices(Config.DIGITS, k=3)) + 
                   ''.join(random.choices(Config.ENGLISH_LETTERS[:26], k=3)),
            lambda: ''.join(random.choices(Config.ENGLISH_LETTERS[:26], k=2)) + 
                   ''.join(random.choices(Config.DIGITS, k=4)),
        ]
        
        return random.choice(patterns)()


class FormTextGenerator(TextGenerator):
    """Specialized generator for form-like text"""
    
    def __init__(self, config: GeneratorConfig = None):
        if config is None:
            config = GeneratorConfig(language="en")
        
        super().__init__(config)
    
    def generate_text(self, length: int = None) -> str:
        """Generate form-appropriate text"""
        form_types = [
            'name', 'address', 'license_plate', 'phone', 'email', 'id_number'
        ]
        
        form_type = random.choice(form_types)
        
        if form_type == 'name':
            return self.generate_name_text()
        elif form_type == 'address':
            return self.generate_address_text()
        elif form_type == 'license_plate':
            return self.generate_license_plate()
        elif form_type == 'phone':
            return self._generate_phone_number()
        elif form_type == 'email':
            return self._generate_email()
        else:
            return self._generate_id_number()
    
    def _generate_phone_number(self) -> str:
        """Generate phone number"""
        patterns = [
            lambda: f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(200, 999)}-{random.randint(200, 999)}-{random.randint(1000, 9999)}",
            lambda: f"{random.randint(200, 999)}.{random.randint(200, 999)}.{random.randint(1000, 9999)}",
        ]
        
        return random.choice(patterns)()
    
    def _generate_email(self) -> str:
        """Generate email address"""
        names = ['john', 'jane', 'mike', 'sarah', 'alex', 'chris', 'kelly', 'jordan']
        domains = ['gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com', 'company.com']
        
        name = random.choice(names)
        domain = random.choice(domains)
        
        # Sometimes add numbers
        if random.random() < 0.3:
            name += str(random.randint(1, 999))
        
        return f"{name}@{domain}"
    
    def _generate_id_number(self) -> str:
        """Generate ID number"""
        return ''.join(random.choices(Config.DIGITS, k=random.randint(6, 12)))