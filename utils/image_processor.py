"""
Image processing utilities for OCR data generation
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont, ImageFilter
from typing import Tuple, Optional, Union
import random


class ImageProcessor:
    """Handles basic image processing operations"""
    
    @staticmethod
    def crop_whitespace(image: Image.Image, 
                       threshold: int = 200,
                       margin: int = 2) -> Image.Image:
        """
        Remove whitespace around the image content
        
        Args:
            image: PIL Image object
            threshold: Pixel value threshold for content detection
            margin: Margin to keep around content
            
        Returns:
            Cropped PIL Image
        """
        # Convert to grayscale for processing
        if image.mode != 'L':
            gray = image.convert('L')
        else:
            gray = image
            
        img_array = np.array(gray)
        
        # Find content boundaries
        horizontal_sum = np.sum(img_array < threshold, axis=1)
        vertical_sum = np.sum(img_array < threshold, axis=0)
        
        # Find non-empty rows and columns
        rows = np.where(horizontal_sum > 0)[0]
        cols = np.where(vertical_sum > 0)[0]
        
        if len(rows) == 0 or len(cols) == 0:
            return image  # Return original if no content found
        
        # Get boundaries with margin
        top = max(0, rows[0] - margin)
        bottom = min(img_array.shape[0], rows[-1] + margin + 1)
        left = max(0, cols[0] - margin)
        right = min(img_array.shape[1], cols[-1] + margin + 1)
        
        # Crop the original image
        return image.crop((left, top, right, bottom))
    
    @staticmethod
    def resize_with_aspect_ratio(image: Image.Image,
                                target_size: Tuple[int, int],
                                background_color: int = 255) -> Image.Image:
        """
        Resize image while maintaining aspect ratio
        
        Args:
            image: PIL Image object
            target_size: (width, height) target size
            background_color: Background color for padding
            
        Returns:
            Resized PIL Image
        """
        target_width, target_height = target_size
        
        # Calculate scaling factor
        img_width, img_height = image.size
        scale = min(target_width / img_width, target_height / img_height)
        
        # Calculate new size
        new_width = int(img_width * scale)
        new_height = int(img_height * scale)
        
        # Resize image
        resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Create new image with target size
        result = Image.new(image.mode, target_size, background_color)
        
        # Center the resized image
        x_offset = (target_width - new_width) // 2
        y_offset = (target_height - new_height) // 2
        result.paste(resized, (x_offset, y_offset))
        
        return result
    
    @staticmethod
    def add_border(image: Image.Image,
                   border_width: int = 2,
                   border_color: int = 0) -> Image.Image:
        """
        Add border around image
        
        Args:
            image: PIL Image object
            border_width: Width of border in pixels
            border_color: Color of border (0=black, 255=white)
            
        Returns:
            Image with border
        """
        width, height = image.size
        
        # Create new image with border
        new_width = width + 2 * border_width
        new_height = height + 2 * border_width
        new_image = Image.new(image.mode, (new_width, new_height), 255)
        
        # Paste original image in center
        new_image.paste(image, (border_width, border_width))
        
        # Draw border
        draw = ImageDraw.Draw(new_image)
        draw.rectangle(
            [(border_width - 1, border_width - 1), 
             (width + border_width, height + border_width)],
            outline=border_color,
            width=border_width
        )
        
        return new_image
    
    @staticmethod
    def simulate_copy_effect(image: Image.Image,
                           contrast_factor: float = 0.8,
                           brightness_offset: int = 20) -> Image.Image:
        """
        Simulate photocopying effects
        
        Args:
            image: PIL Image object
            contrast_factor: Contrast adjustment factor
            brightness_offset: Brightness adjustment offset
            
        Returns:
            Image with copy effects
        """
        # Convert to numpy array
        img_array = np.array(image)
        
        # Adjust contrast and brightness
        img_array = img_array.astype(np.float32)
        img_array = (img_array - 127.5) * contrast_factor + 127.5 + brightness_offset
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def add_noise(image: Image.Image,
                  noise_level: float = 0.1) -> Image.Image:
        """
        Add random noise to image
        
        Args:
            image: PIL Image object
            noise_level: Intensity of noise (0.0 to 1.0)
            
        Returns:
            Image with noise
        """
        img_array = np.array(image)
        
        # Generate noise
        noise = np.random.normal(0, noise_level * 255, img_array.shape)
        
        # Add noise to image
        noisy_array = img_array.astype(np.float32) + noise
        noisy_array = np.clip(noisy_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(noisy_array)
    
    @staticmethod
    def blend_with_background(foreground: Image.Image,
                            background: Image.Image,
                            alpha: float = 0.8) -> Image.Image:
        """
        Blend foreground image with background
        
        Args:
            foreground: Foreground PIL Image
            background: Background PIL Image
            alpha: Blending factor (1.0 = only foreground, 0.0 = only background)
            
        Returns:
            Blended image
        """
        # Ensure both images are the same size
        fg_width, fg_height = foreground.size
        background = background.resize((fg_width, fg_height), Image.Resampling.LANCZOS)
        
        # Convert to RGBA for blending
        if foreground.mode != 'RGBA':
            foreground = foreground.convert('RGBA')
        if background.mode != 'RGBA':
            background = background.convert('RGBA')
        
        # Blend images
        blended = Image.blend(background, foreground, alpha)
        
        return blended.convert('RGB')
    
    @staticmethod 
    def create_text_mask(text: str,
                        font: ImageFont.FreeTypeFont,
                        image_size: Tuple[int, int]) -> Image.Image:
        """
        Create a mask image for text
        
        Args:
            text: Text to render
            font: Font object
            image_size: (width, height) of output image
            
        Returns:
            Binary mask image
        """
        width, height = image_size
        
        # Create mask image
        mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(mask)
        
        # Calculate text position
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        x = (width - text_width) // 2
        y = (height - text_height) // 2
        
        # Draw text
        draw.text((x, y), text, font=font, fill=255)
        
        return mask