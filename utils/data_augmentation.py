"""
Data augmentation utilities for OCR training data
"""
import numpy as np
import cv2
from PIL import Image, ImageDraw
from typing import Tuple, Optional
import random
import math


class DataAugmentation:
    """Handles various data augmentation techniques"""
    
    @staticmethod
    def rotate_image(image: Image.Image,
                    angle_range: Tuple[float, float] = (-5, 5),
                    background_color: int = 255) -> Image.Image:
        """
        Rotate image by random angle
        
        Args:
            image: PIL Image object
            angle_range: (min_angle, max_angle) in degrees
            background_color: Background color for rotation
            
        Returns:
            Rotated image
        """
        angle = random.uniform(angle_range[0], angle_range[1])
        return image.rotate(angle, 
                          resample=Image.Resampling.BICUBIC,
                          expand=True,
                          fillcolor=background_color)
    
    @staticmethod
    def apply_perspective_transform(image: Image.Image,
                                  strength: float = 0.1) -> Image.Image:
        """
        Apply random perspective transformation
        
        Args:
            image: PIL Image object
            strength: Transformation strength (0.0 to 1.0)
            
        Returns:
            Transformed image
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Define source points (corners of the image)
        src_points = np.float32([
            [0, 0],
            [width-1, 0],
            [0, height-1],
            [width-1, height-1]
        ])
        
        # Add random displacement to corners
        max_displacement = strength * min(width, height)
        dst_points = src_points.copy()
        
        for i in range(4):
            dx = random.uniform(-max_displacement, max_displacement)
            dy = random.uniform(-max_displacement, max_displacement)
            dst_points[i][0] = np.clip(dst_points[i][0] + dx, 0, width-1)
            dst_points[i][1] = np.clip(dst_points[i][1] + dy, 0, height-1)
        
        # Apply perspective transformation
        matrix = cv2.getPerspectiveTransform(src_points, dst_points)
        transformed = cv2.warpPerspective(img_array, matrix, (width, height),
                                        borderValue=255)
        
        return Image.fromarray(transformed)
    
    @staticmethod
    def adjust_brightness_contrast(image: Image.Image,
                                 brightness_range: Tuple[float, float] = (0.8, 1.2),
                                 contrast_range: Tuple[float, float] = (0.8, 1.2)) -> Image.Image:
        """
        Randomly adjust brightness and contrast
        
        Args:
            image: PIL Image object
            brightness_range: (min, max) brightness multipliers
            contrast_range: (min, max) contrast multipliers
            
        Returns:
            Adjusted image
        """
        img_array = np.array(image).astype(np.float32)
        
        # Random brightness and contrast factors
        brightness = random.uniform(brightness_range[0], brightness_range[1])
        contrast = random.uniform(contrast_range[0], contrast_range[1])
        
        # Apply adjustments
        img_array = img_array * brightness
        img_array = (img_array - 127.5) * contrast + 127.5
        
        # Clip values
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def add_elastic_deformation(image: Image.Image,
                              alpha: float = 20,
                              sigma: float = 5) -> Image.Image:
        """
        Apply elastic deformation to simulate handwriting variations
        
        Args:
            image: PIL Image object
            alpha: Deformation strength
            sigma: Smoothness of deformation
            
        Returns:
            Deformed image
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        # Generate random displacement fields
        dx = np.random.uniform(-1, 1, (height, width)) * alpha
        dy = np.random.uniform(-1, 1, (height, width)) * alpha
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (0, 0), sigma)
        dy = cv2.GaussianBlur(dy, (0, 0), sigma)
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        x_new = np.clip(x + dx, 0, width-1).astype(np.float32)
        y_new = np.clip(y + dy, 0, height-1).astype(np.float32)
        
        # Apply deformation
        deformed = cv2.remap(img_array, x_new, y_new, 
                           interpolation=cv2.INTER_LINEAR,
                           borderValue=255)
        
        return Image.fromarray(deformed)
    
    @staticmethod
    def adjust_stroke_thickness(image: Image.Image,
                              method: str = "random",
                              kernel_size: int = 2) -> Image.Image:
        """
        Adjust stroke thickness using morphological operations
        
        Args:
            image: PIL Image object
            method: "dilate", "erode", or "random"
            kernel_size: Size of morphological kernel
            
        Returns:
            Image with adjusted stroke thickness
        """
        img_array = np.array(image)
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        if method == "random":
            method = random.choice(["dilate", "erode"])
        
        if method == "dilate":
            result = cv2.dilate(img_array, kernel, iterations=1)
        elif method == "erode":
            result = cv2.erode(img_array, kernel, iterations=1)
        else:
            result = img_array
            
        return Image.fromarray(result)
    
    @staticmethod
    def add_random_gaps(image: Image.Image,
                       num_gaps: int = None,
                       gap_size_range: Tuple[int, int] = (2, 8)) -> Image.Image:
        """
        Add random gaps to simulate broken strokes
        
        Args:
            image: PIL Image object
            num_gaps: Number of gaps to add (random if None)
            gap_size_range: (min, max) gap sizes
            
        Returns:
            Image with gaps
        """
        img_array = np.array(image)
        height, width = img_array.shape[:2]
        
        if num_gaps is None:
            num_gaps = random.randint(1, 3)
        
        for _ in range(num_gaps):
            # Random gap position and size
            gap_size = random.randint(gap_size_range[0], gap_size_range[1])
            x = random.randint(0, max(0, width - gap_size))
            y = random.randint(0, height - 1)
            
            # Create gap by setting pixels to white
            img_array[y, x:x+gap_size] = 255
        
        return Image.fromarray(img_array)
    
    @staticmethod
    def apply_blur(image: Image.Image,
                  blur_type: str = "gaussian",
                  strength: float = 1.0) -> Image.Image:
        """
        Apply blur effects
        
        Args:
            image: PIL Image object
            blur_type: "gaussian", "motion", or "random"
            strength: Blur strength
            
        Returns:
            Blurred image
        """
        if blur_type == "random":
            blur_type = random.choice(["gaussian", "motion"])
        
        if blur_type == "gaussian":
            sigma = strength
            img_array = np.array(image)
            blurred = cv2.GaussianBlur(img_array, (0, 0), sigma)
            return Image.fromarray(blurred)
        
        elif blur_type == "motion":
            # Create motion blur kernel
            kernel_size = int(strength * 5) + 3
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size//2, :] = 1.0
            kernel = kernel / kernel_size
            
            img_array = np.array(image)
            blurred = cv2.filter2D(img_array, -1, kernel)
            return Image.fromarray(blurred)
        
        return image
    
    @staticmethod
    def random_augmentation_pipeline(image: Image.Image,
                                   prob_each: float = 0.5) -> Image.Image:
        """
        Apply random combination of augmentations
        
        Args:
            image: PIL Image object
            prob_each: Probability of applying each augmentation
            
        Returns:
            Augmented image
        """
        # Apply augmentations with given probability
        if random.random() < prob_each:
            image = DataAugmentation.rotate_image(image)
        
        if random.random() < prob_each:
            image = DataAugmentation.apply_perspective_transform(image)
        
        if random.random() < prob_each:
            image = DataAugmentation.adjust_brightness_contrast(image)
        
        if random.random() < prob_each:
            image = DataAugmentation.adjust_stroke_thickness(image)
        
        if random.random() < prob_each * 0.3:  # Lower probability for gaps
            image = DataAugmentation.add_random_gaps(image)
        
        if random.random() < prob_each * 0.3:  # Lower probability for blur
            image = DataAugmentation.apply_blur(image)
        
        return image