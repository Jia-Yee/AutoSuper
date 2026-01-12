import cv2
import numpy as np
import random
from datetime import datetime
import os


def create_matrix_with_enhanced_text(target_text="AutoSuper", output_file='matrix_enhanced_text.mp4', duration=10, fps=30, height=720, width=1280):
    """
    Creates a Matrix-style digital rain effect with large embedded text that is more prominent
    by adding a bright outline around the characters
    
    Args:
        target_text: Text to embed in the digital rain using large repeated characters with enhanced visibility
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
        height: Height of the output video (default 720)
        width: Width of the output video (default 1280)
    """
    
    # Define video properties
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize columns for the digital rain
    font_size = 16
    cols = width // font_size
    drops = [0] * cols  # Initialize drops position for each column
    
    # Total frames to generate
    total_frames = duration * fps
    
    # Define character patterns for each letter (7x10 grid for larger size)
    char_patterns = {
        'A': [
            "  000  ",
            " 00 00 ",
            "00   00",
            "00   00",
            "0000000",
            "00   00",
            "00   00"
        ],
        'U': [
            "00   00",
            "00   00", 
            "00   00",
            "00   00",
            "00   00",
            "00   00",
            " 00000 "
        ],
        'T': [
            "0000000",
            "   00   ",
            "   00   ",
            "   00   ",
            "   00   ",
            "   00   ",
            "   00   "
        ],
        'O': [
            " 00000 ",
            "00   00",
            "00   00",
            "00   00", 
            "00   00",
            "00   00",
            " 00000 "
        ],
        'S': [
            " 000000",
            "00     ",
            "00     ",
            " 00000 ",
            "     00",
            "     00",
            "000000 "
        ],
        'P': [
            "000000 ",
            "00   00",
            "00   00",
            "000000 ",
            "00     ",
            "00     ",
            "00     "
        ],
        'E': [
            "0000000",
            "00     ",
            "00     ",
            "000000 ",
            "00     ",
            "00     ",
            "0000000"
        ],
        'R': [
            "000000 ",
            "00   00",
            "00   00", 
            "000000 ",
            "00 00  ",
            "00  00 ",
            "00   00"
        ],
        ' ': [  # Space
            "       ",
            "       ",
            "       ",
            "       ",
            "       ",
            "       ",
            "       "
        ]
    }
    
    # Calculate position to center the text
    char_width = 7  # Pattern width
    char_height = 7  # Pattern height
    spacing = 2  # Space between characters
    total_width = len(target_text) * (char_width + spacing) - spacing
    start_col = (cols - total_width) // 2  # Center the text horizontally
    
    # Calculate vertical position (centered)
    start_row = (height // font_size) // 2 - char_height // 2
    
    # Pre-calculate colors for each character position to maintain consistency
    char_colors = {}
    for idx, char in enumerate(target_text.upper()):
        # Generate base color based on character and position
        hue_base = (idx * 30) % 180  # Different base hue for each character position
        char_colors[idx] = hue_base
    
    for frame_idx in range(total_frames):
        # Create black background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Set up font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Update and draw drops for each column
        for i in range(cols):
            # Calculate position
            x = i * font_size
            y = drops[i] * font_size
            
            # Check if this position is part of our target text pattern (including outline)
            is_in_pattern = False
            is_outline = False
            pattern_char = None
            char_idx = -1
            
            # Calculate which character position in the target text this column might be part of
            for idx, char in enumerate(target_text.upper()):
                if char in char_patterns:
                    # Calculate the column range for this character (including outline padding)
                    char_start_col = start_col + idx * (char_width + spacing)
                    char_end_col = char_start_col + char_width
                    
                    # First check if this is inside the character pattern
                    if char_start_col <= i < char_end_col:
                        # Calculate which row in the pattern this corresponds to
                        row_in_pattern = int((y // font_size) - start_row)
                        
                        if 0 <= row_in_pattern < char_height:
                            # Get the pattern for this character
                            pattern = char_patterns[char]
                            # Get the specific position in the pattern
                            pattern_row = pattern[row_in_pattern]
                            pattern_col = i - char_start_col
                            
                            if 0 <= pattern_col < len(pattern_row) and pattern_row[pattern_col] == '0':
                                is_in_pattern = True
                                pattern_char = char  # Use the character itself
                                char_idx = idx
                                break
                    
                    # If not in the main character, check if it's in the outline area
                    if not is_in_pattern:
                        # Check if this position is near the character (outline effect)
                        outline_start_col = max(0, char_start_col - 1)
                        outline_end_col = min(cols, char_end_col + 1)
                        outline_start_row = max(0, start_row - 1)
                        outline_end_row = min(height // font_size, start_row + char_height + 1)
                        
                        if (outline_start_col <= i < outline_end_col and 
                            outline_start_row <= (y // font_size) < outline_end_row):
                            # Check if this is adjacent to an actual character pixel
                            for r in range(max(0, row_in_pattern-1), min(char_height, row_in_pattern+2)):
                                for c in range(max(0, i - char_start_col - 1), 
                                              min(char_width, i - char_start_col + 2)):
                                    if 0 <= r < len(char_patterns[char]) and 0 <= c < len(char_patterns[char][r]):
                                        if char_patterns[char][r][c] == '0':
                                            is_outline = True
                                            pattern_char = '#'
                                            char_idx = idx
                                            break
                                if is_outline:
                                    break
                            if is_outline:
                                break
            
            if is_in_pattern or is_outline:
                # Calculate color based on character index and time for overall color shift
                base_hue = char_colors[char_idx]
                # Add time-based shift to make colors cycle smoothly
                time_shift = int((frame_idx / fps) * 20) % 180  # Shift colors over time
                hue = (base_hue + time_shift) % 180
                
                # Ensure full saturation and brightness for consistently bright colors
                saturation = 255  
                value = 255       
                
                # Convert HSV to BGR for OpenCV
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                
                # Display the character in consistently bright color for the whole character
                cv2.putText(img, pattern_char, (x, y), font, font_scale, tuple(bgr_color), thickness)
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - keeping them bright too
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text pattern (including outline)
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    is_trail_in_pattern = False
                    is_trail_outline = False
                    trail_pattern_char = None
                    trail_char_idx = -1
                    
                    for idx, char_t in enumerate(target_text.upper()):
                        if char_t in char_patterns:
                            # Calculate the column range for this character (including outline padding)
                            char_start_col = start_col + idx * (char_width + spacing)
                            char_end_col = char_start_col + char_width
                            
                            # First check if this is inside the character pattern
                            if char_start_col <= i < char_end_col:
                                # Calculate which row in the pattern this corresponds to
                                pattern_row_idx = trail_row_pos - start_row
                                
                                if 0 <= pattern_row_idx < char_height:
                                    # Get the pattern for this character
                                    pattern = char_patterns[char_t]
                                    # Get the specific position in the pattern
                                    pattern_row = pattern[pattern_row_idx]
                                    pattern_col = i - char_start_col
                                    
                                    if 0 <= pattern_col < len(pattern_row) and pattern_row[pattern_col] == '0':
                                        is_trail_in_pattern = True
                                        trail_pattern_char = char_t  # Use the character itself
                                        trail_char_idx = idx
                                        break
                            
                            # If not in the main character, check if it's in the outline area
                            if not is_trail_in_pattern:
                                # Check if this position is near the character (outline effect)
                                outline_start_col = max(0, char_start_col - 1)
                                outline_end_col = min(cols, char_end_col + 1)
                                outline_start_row = max(0, start_row - 1)
                                outline_end_row = min(height // font_size, start_row + char_height + 1)
                                
                                if (outline_start_col <= i < outline_end_col and 
                                    outline_start_row <= trail_row_pos < outline_end_row):
                                    # Check if this is adjacent to an actual character pixel
                                    for r in range(max(0, pattern_row_idx-1), min(char_height, pattern_row_idx+2)):
                                        for c in range(max(0, i - char_start_col - 1), 
                                                      min(char_width, i - char_start_col + 2)):
                                            if 0 <= r < len(char_patterns[char_t]) and 0 <= c < len(char_patterns[char_t][r]):
                                                if char_patterns[char_t][r][c] == '0':
                                                    is_trail_outline = True
                                                    trail_pattern_char = '#'
                                                    trail_char_idx = idx
                                                    break
                                        if is_trail_outline:
                                            break
                                    if is_trail_outline:
                                        break
                    
                    if is_trail_in_pattern or is_trail_outline:
                        # Calculate consistent bright color for trailing characters
                        base_hue = char_colors[trail_char_idx]
                        time_shift = int((frame_idx / fps) * 20) % 180
                        hue = (base_hue + time_shift) % 180
                        
                        # Keep brightness high but reduce saturation slightly for trail effect
                        saturation = 220  # High but slightly less than main text
                        value = 255       # Keep maximum brightness
                        
                        hsv_color = np.uint8([[[hue, saturation, value]]])
                        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                        
                        cv2.putText(img, trail_pattern_char, (x, y - j * font_size), font, font_scale, 
                                   tuple(bgr_color), thickness)
                    else:
                        # Fade from bright green to dark green for non-text characters
                        intensity = max(50, 255 - j * 20)
                        char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)
                        cv2.putText(img, char, (x, y - j * font_size), font, font_scale, 
                                   (0, intensity, 0), thickness)
            
            # Reset drop if it reaches bottom or randomly
            if y > height or random.random() > 0.975:
                drops[i] = 0
            
            # Move drop down
            drops[i] += 1
        
        # Write the frame to video
        out.write(img)
    
    # Release everything
    out.release()
    print(f"Matrix effect with enhanced text saved as {output_file}")


if __name__ == "__main__":
    print("选择效果:")
    print("1. 增强大文本的数字雨效果（显示自定义文本）")
    
    choice = input("请输入选择 (1): ")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if choice == "1":
        custom_text = input("请输入要在数字雨中显示的文本 (默认为 'AutoSuper'): ").strip()
        if not custom_text:
            custom_text = "AutoSuper"
        
        height_input = input("请输入视频高度 (默认为 720): ").strip()
        width_input = input("请输入视频宽度 (默认为 1280): ").strip()
        
        height = int(height_input) if height_input else 720
        width = int(width_input) if width_input else 1280
        
        output_filename = f"matrix_enhanced_text_{timestamp}.mp4"
        create_matrix_with_enhanced_text(target_text=custom_text, output_file=output_filename, height=height, width=width)
    else:
        print("无效选择，运行增强大文本效果")
        custom_text = "AutoSuper"
        output_filename = f"matrix_enhanced_text_{timestamp}.mp4"
        create_matrix_with_enhanced_text(target_text=custom_text, output_file=output_filename)