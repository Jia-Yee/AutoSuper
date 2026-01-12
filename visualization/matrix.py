import numpy as np
import cv2
import random
import time
from datetime import datetime
import os

def create_matrix_effect(text_lines, output_file='matrix_effect.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect video
    
    Args:
        text_lines: List of strings to display as falling characters
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize columns for the digital rain
    font_size = 16
    cols = width // font_size
    drops = [0] * cols  # Initialize drops position for each column
    
    # Total frames to generate
    total_frames = duration * fps
    
    for frame_idx in range(total_frames):
        # Create black background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Set up font properties
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        
        # Update and draw drops for each column
        for i in range(cols):
            # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
            char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
            
            # Calculate position
            x = i * font_size
            y = drops[i] * font_size
            
            # Draw the character - bright green for head of trail, darker green for tail
            cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - fade effect
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Fade from bright green to dark green
                    intensity = max(50, 255 - j * 20)
                    char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)
                    cv2.putText(img, char, (x, y - j * font_size), font, font_scale, 
                               (0, intensity, 0), thickness)
            
            # Reset drop if it reaches bottom or randomly
            if y > height or random.random() > 0.975:
                drops[i] = 0
            
            # Move drop down
            drops[i] += 1
        
        # Add the text lines occasionally in brighter color
        if frame_idx % (fps * 2) < fps:  # Show text for 1 second every 2 seconds
            for idx, line in enumerate(text_lines):
                if line.strip():  # Only draw non-empty lines
                    text_y = height // 2 + idx * 30
                    cv2.putText(img, line, (width//2 - len(line)*7, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Write the frame to video
        out.write(img)
    
    # Release everything
    out.release()
    print(f"Video saved as {output_file}")


def scroll_text_effect(text_lines, output_file='matrix_scroll_effect.mp4', duration=10, fps=30):
    """
    Creates a scrolling text effect similar to The Matrix intro
    """
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Prepare text for scrolling
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.0
    thickness = 2
    
    # Calculate text size to center it
    text_sizes = []
    positions = []
    for line in text_lines:
        if line.strip():
            size = cv2.getTextSize(line, font, font_scale, thickness)[0]
            text_sizes.append(size)
            # Start position off-screen at the bottom
            positions.append([width//2 - size[0]//2, height + len(positions)*50])
    
    total_frames = duration * fps
    # Scroll speed
    scroll_speed = 2.0
    
    for frame_idx in range(total_frames):
        # Create black background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw digital rain in background
        font_size = 16
        cols = width // font_size
        # Simulate rain effect in background
        for i in range(int(cols * 0.3)):  # Sparse background rain
            col = random.randint(0, cols - 1)
            char = random.choice(['0', '1'])
            x = col * font_size
            y = (frame_idx // 5 + col) % (height // font_size) * font_size
            cv2.putText(img, char, (x, y), font, 0.5, (0, 100, 0), 1)
        
        # Update and draw text lines
        for i, line in enumerate(text_lines):
            if line.strip():
                # Update position (move upward)
                positions[i][1] -= int(scroll_speed)
                
                # Draw the text
                cv2.putText(img, line, tuple(positions[i]), font, font_scale, (0, 255, 0), thickness)
                
                # Reset if text scrolls completely off screen
                if positions[i][1] < -50:
                    positions[i][1] = height + len(text_lines)*50
        
        # Write the frame to video
        out.write(img)
    
    # Release everything
    out.release()
    print(f"Scrolling video saved as {output_file}")


def combined_effect(text_lines, output_file='matrix_combined_effect.mp4', duration=15, fps=30):
    """
    Creates a combination of both effects - digital rain with scrolling text
    """
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize columns for the digital rain
    font_size = 16
    cols = width // font_size
    drops = [0] * cols  # Initialize drops position for each column
    
    # Prepare text for scrolling
    font = cv2.FONT_HERSHEY_SIMPLEX
    scroll_positions = []
    for i, line in enumerate(text_lines):
        if line.strip():
            # Start position off-screen at the bottom
            scroll_positions.append([width//2 - len(line)*10, height + i*50])
    
    total_frames = duration * fps
    scroll_speed = 1.5
    
    for frame_idx in range(total_frames):
        # Create black background
        img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Digital rain effect
        for i in range(cols):
            # Get a random character
            char = random.choice('01' + ''.join([chr(j) for j in range(33, 126)])*2)
            
            # Calculate position
            x = i * font_size
            y = drops[i] * font_size
            
            # Draw the character - bright green for head of trail, darker green for tail
            cv2.putText(img, char, (x, y), font, 0.5, (0, 255, 0), 1)
            
            # Draw trailing characters - fade effect
            for j in range(1, 15):  # 15 characters in trail for stronger effect
                if y - j * font_size > 0:
                    # Fade from bright green to dark green
                    intensity = max(30, 255 - j * 15)
                    char = random.choice('01' + ''.join([chr(j) for j in range(33, 126)])*2)
                    cv2.putText(img, char, (x, y - j * font_size), font, 0.5, 
                               (0, min(255, intensity), 0), 1)
            
            # Randomly reset drop or if it reaches bottom
            if y > height or random.random() > 0.97:
                drops[i] = 0
            
            # Move drop down
            drops[i] += 1
        
        # Handle scrolling text
        active_texts = False
        for i, line in enumerate(text_lines):
            if line.strip():
                # Update position (move upward)
                scroll_positions[i][1] -= scroll_speed
                
                # Draw the text if it's on screen
                if 0 <= scroll_positions[i][1] <= height + 50:
                    active_texts = True
                    cv2.putText(img, line, (int(scroll_positions[i][0]), int(scroll_positions[i][1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 2)
                
                # Reset if text scrolls completely off screen
                if scroll_positions[i][1] < -50:
                    scroll_positions[i][1] = height + len(text_lines)*50
        
        # Occasionally make the scrolling text brighter
        if frame_idx % (fps * 5) < fps*2:  # Brighten for 2 seconds every 5
            for i, line in enumerate(text_lines):
                if line.strip() and 0 <= scroll_positions[i][1] <= height + 50:
                    cv2.putText(img, line, (int(scroll_positions[i][0]), int(scroll_positions[i][1])), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 200), 3)
        
        # Write the frame to video
        out.write(img)
    
    # Release everything
    out.release()
    print(f"Combined effect video saved as {output_file}")


def create_matrix_with_text(text_lines, target_text="AutoSuper", output_file='matrix_with_text.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect with embedded target text
    
    Args:
        text_lines: Regular text lines to display occasionally
        target_text: Text to embed in the digital rain
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize columns for the digital rain
    font_size = 16
    cols = width // font_size
    drops = [0] * cols  # Initialize drops position for each column
    
    # Total frames to generate
    total_frames = duration * fps
    
    # Prepare target text characters
    target_chars = list(target_text)
    target_len = len(target_chars)
    
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
            
            # Decide whether to show target text character based on position
            show_target_char = False
            target_char_idx = -1
            
            # Check if this position could be part of our target text
            # We'll make the target text appear diagonally across the screen
            if target_len > 0:
                # Determine row/column position in the grid
                col_pos = i
                row_pos = int(y // font_size)
                
                # Make target text appear diagonally from top-left to bottom-right
                # Target text will appear at positions where row_pos - col_pos is close to a specific offset
                offset_range = 5  # How wide the diagonal band is
                for t_offset in range(-offset_range, offset_range):
                    diag_pos = row_pos - col_pos + width//font_size//2 + t_offset
                    if 0 <= diag_pos < target_len:
                        show_target_char = True
                        target_char_idx = diag_pos
                        break
            
            if show_target_char and target_char_idx != -1:
                # Display the target character in bright white/yellow
                char = target_chars[target_char_idx]
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 255), thickness)  # Yellow
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - fade effect
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    show_target_trail = False
                    target_trail_idx = -1
                    
                    # Similar logic for trailing characters
                    if target_len > 0:
                        for t_offset in range(-offset_range, offset_range):
                            diag_pos = trail_row_pos - col_pos + width//font_size//2 + t_offset
                            if 0 <= diag_pos < target_len:
                                show_target_trail = True
                                target_trail_idx = diag_pos
                                break
                    
                    if show_target_trail and target_trail_idx != -1:
                        # Display the target character in a lighter shade
                        char = target_chars[target_trail_idx]
                        intensity = max(100, 255 - j * 15)  # Fade effect
                        cv2.putText(img, char, (x, y - j * font_size), font, font_scale, 
                                   (0, intensity, intensity), thickness)  # Yellow with fade
                    else:
                        # Fade from bright green to dark green
                        intensity = max(50, 255 - j * 20)
                        char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)
                        cv2.putText(img, char, (x, y - j * font_size), font, font_scale, 
                                   (0, intensity, 0), thickness)
            
            # Reset drop if it reaches bottom or randomly
            if y > height or random.random() > 0.975:
                drops[i] = 0
            
            # Move drop down
            drops[i] += 1
        
        # Add the text lines occasionally in brighter color
        if frame_idx % (fps * 4) < fps:  # Show text for 1 second every 4 seconds
            for idx, line in enumerate(text_lines):
                if line.strip():  # Only draw non-empty lines
                    text_y = height // 2 + idx * 40
                    cv2.putText(img, line, (width//2 - len(line)*10, text_y), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 100), 3)
        
        # Write the frame to video
        out.write(img)
    
    # Release everything
    out.release()
    print(f"Matrix effect with embedded text saved as {output_file}")


def create_matrix_with_text_patterns(text_lines, target_text="AutoSuper", output_file='matrix_with_text.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect with embedded text formed by character patterns at fixed positions
    
    Args:
        text_lines: Regular text lines to display occasionally
        target_text: Text to embed in the digital rain using character patterns
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Initialize columns for the digital rain
    font_size = 16
    cols = width // font_size
    drops = [0] * cols  # Initialize drops position for each column
    
    # Total frames to generate
    total_frames = duration * fps
    
    # Define character patterns for each letter (5x7 grid)
    char_patterns = {
        'A': [
            " 000 ",
            " 0  0",
            " 0  0", 
            " 0000",
            " 0  0",
            " 0  0",
            " 0  0"
        ],
        'B': [
            " 000 ",
            " 0  0",
            " 0  0",
            " 000 ",
            " 0  0", 
            " 0  0",
            " 000 "
        ],
        'C': [
            " 000 ",
            " 0  0",
            " 0    ",
            " 0    ",
            " 0    ",
            " 0  0",
            " 000 "
        ],
        'D': [
            " 000 ",
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 000 "
        ],
        'E': [
            " 0000",
            " 0   ",
            " 0   ",
            " 000 ",
            " 0   ",
            " 0   ",
            " 0000"
        ],
        'F': [
            " 0000",
            " 0   ",
            " 0   ",
            " 000 ",
            " 0   ",
            " 0   ",
            " 0   "
        ],
        'G': [
            " 000 ",
            " 0  0",
            " 0   ",
            " 0 00",
            " 0  0",
            " 0  0",
            " 000 "
        ],
        'H': [
            " 0  0",
            " 0  0",
            " 0  0",
            " 0000",
            " 0  0",
            " 0  0",
            " 0  0"
        ],
        'I': [
            " 000 ",
            "  0  ",
            "  0  ",
            "  0  ",
            "  0  ",
            "  0  ",
            " 000 "
        ],
        'O': [
            " 000 ",
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 000 "
        ],
        'P': [
            " 000 ",
            " 0  0",
            " 0  0",
            " 000 ",
            " 0   ",
            " 0   ",
            " 0   "
        ],
        'R': [
            " 000 ",
            " 0  0",
            " 0  0",
            " 000 ",
            " 0 0 ",
            " 0  0",
            " 0  0"
        ],
        'S': [
            " 000 ",
            " 0   ",
            " 0   ",
            " 000 ",
            "   0 ",
            "   0 ",
            " 000 "
        ],
        'T': [
            " 000 ",
            "  0  ",
            "  0  ",
            "  0  ",
            "  0  ",
            "  0  ",
            "  0  "
        ],
        'U': [
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 0  0",
            " 000 "
        ],
        'Y': [
            " 0  0",
            " 0  0",
            " 0  0",
            " 000 ",
            "  0  ",
            "  0  ",
            "  0  "
        ]
    }
    
    # Calculate position to center the text
    char_width = 5  # Pattern width
    char_height = 7  # Pattern height
    spacing = 1  # Space between characters
    total_width = len(target_text) * (char_width + spacing) - spacing
    start_col = (cols - total_width) // 2  # Center the text horizontally
    
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
            
            # Check if this position is part of our target text pattern
            is_in_pattern = False
            pattern_char = '0'
            
            # Calculate which character position in the target text this column might be part of
            for char_idx, char in enumerate(target_text.upper()):
                if char in char_patterns:
                    # Calculate the column range for this character
                    char_start_col = start_col + char_idx * (char_width + spacing)
                    char_end_col = char_start_col + char_width
                    
                    if char_start_col <= i < char_end_col:
                        # Calculate which row in the pattern this corresponds to
                        row_in_pattern = int((y // font_size) - (height // 2) // font_size)
                        
                        if 0 <= row_in_pattern < char_height:
                            # Get the pattern for this character
                            pattern = char_patterns[char]
                            # Get the specific position in the pattern
                            pattern_row = pattern[row_in_pattern]
                            pattern_col = i - char_start_col
                            
                            if 0 <= pattern_col < len(pattern_row) and pattern_row[pattern_col] == '0':
                                is_in_pattern = True
                                pattern_char = char  # Use the character itself
                                break
            
            if is_in_pattern:
                # Display the character in bright yellow
                cv2.putText(img, pattern_char, (x, y), font, font_scale, (0, 255, 255), thickness)  # Yellow
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - fade effect
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text pattern
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    is_trail_in_pattern = False
                    trail_pattern_char = '0'
                    
                    for char_idx, char_t in enumerate(target_text.upper()):
                        if char_t in char_patterns:
                            # Calculate the column range for this character
                            char_start_col = start_col + char_idx * (char_width + spacing)
                            char_end_col = char_start_col + char_width
                            
                            if char_start_col <= i < char_end_col:
                                # Calculate which row in the pattern this corresponds to
                                pattern_row_idx = trail_row_pos - (height // 2) // 1
                                
                                if 0 <= pattern_row_idx < char_height:
                                    # Get the pattern for this character
                                    pattern = char_patterns[char_t]
                                    # Get the specific position in the pattern
                                    pattern_row = pattern[pattern_row_idx]
                                    pattern_col = i - char_start_col
                                    
                                    if 0 <= pattern_col < len(pattern_row) and pattern_row[pattern_col] == '0':
                                        is_trail_in_pattern = True
                                        trail_pattern_char = char_t  # Use the character itself
                                        break
                    
                    if is_trail_in_pattern:
                        # Display the character in yellow with fade
                        intensity = max(100, 255 - j * 15)  # Fade effect
                        cv2.putText(img, trail_pattern_char, (x, y - j * font_size), font, font_scale, 
                                   (0, intensity, intensity), thickness)  # Yellow with fade
                    else:
                        # Fade from bright green to dark green
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
    print(f"Matrix effect with embedded text patterns saved as {output_file}")


def create_matrix_with_large_text(target_text="AutoSuper", output_file='matrix_large_text.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect with large embedded text formed by repeated characters
    
    Args:
        target_text: Text to embed in the digital rain using large repeated characters
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
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
            
            # Check if this position is part of our target text pattern
            is_in_pattern = False
            pattern_char = None
            
            # Calculate which character position in the target text this column might be part of
            for char_idx, char in enumerate(target_text.upper()):
                if char in char_patterns:
                    # Calculate the column range for this character
                    char_start_col = start_col + char_idx * (char_width + spacing)
                    char_end_col = char_start_col + char_width
                    
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
                                break
            
            if is_in_pattern:
                # Display the character in bright yellow
                cv2.putText(img, pattern_char, (x, y), font, font_scale, (0, 255, 255), thickness)  # Yellow
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - fade effect
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text pattern
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    is_trail_in_pattern = False
                    trail_pattern_char = None
                    
                    for char_idx, char_t in enumerate(target_text.upper()):
                        if char_t in char_patterns:
                            # Calculate the column range for this character
                            char_start_col = start_col + char_idx * (char_width + spacing)
                            char_end_col = char_start_col + char_width
                            
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
                                        break
                    
                    if is_trail_in_pattern:
                        # Display the character in yellow with fade
                        intensity = max(100, 255 - j * 15)  # Fade effect
                        cv2.putText(img, trail_pattern_char, (x, y - j * font_size), font, font_scale, 
                                   (0, intensity, intensity), thickness)  # Yellow with fade
                    else:
                        # Fade from bright green to dark green
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
    print(f"Matrix effect with large embedded text saved as {output_file}")


def create_matrix_with_flow_text(target_text="AutoSuper", output_file='matrix_flow_text.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect with large embedded text that has flowing colors
    
    Args:
        target_text: Text to embed in the digital rain using large repeated characters with flowing colors
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
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
            
            # Check if this position is part of our target text pattern
            is_in_pattern = False
            pattern_char = None
            
            # Calculate which character position in the target text this column might be part of
            for char_idx, char in enumerate(target_text.upper()):
                if char in char_patterns:
                    # Calculate the column range for this character
                    char_start_col = start_col + char_idx * (char_width + spacing)
                    char_end_col = char_start_col + char_width
                    
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
                                break
            
            if is_in_pattern:
                # Calculate color based on position and time for flowing effect
                # Use both position and frame index to create wave-like color changes
                color_shift = int(((frame_idx / fps) * 2 + i * 0.2 + y * 0.1) * 30) % 180
                # Use HSV to get a range of bright colors
                hue = color_shift
                saturation = 255  # Full saturation for vibrant colors
                value = 255       # Maximum brightness
                # Convert HSV to BGR for OpenCV
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                
                # Display the character in bright flowing color
                cv2.putText(img, pattern_char, (x, y), font, font_scale, tuple(bgr_color), thickness)
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - fade effect
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text pattern
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    is_trail_in_pattern = False
                    trail_pattern_char = None
                    
                    for char_idx, char_t in enumerate(target_text.upper()):
                        if char_t in char_patterns:
                            # Calculate the column range for this character
                            char_start_col = start_col + char_idx * (char_width + spacing)
                            char_end_col = char_start_col + char_width
                            
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
                                        break
                    
                    if is_trail_in_pattern:
                        # Calculate flowing color for trailing characters
                        color_shift = int(((frame_idx / fps) * 2 + i * 0.2 + (y-j*font_size) * 0.1) * 30) % 180
                        hue = color_shift
                        saturation = max(100, 255 - j * 15)  # Reduce saturation with trail length
                        value = max(100, 255 - j * 15)      # Reduce brightness with trail length
                        hsv_color = np.uint8([[[hue, saturation, value]]])
                        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                        
                        cv2.putText(img, trail_pattern_char, (x, y - j * font_size), font, font_scale, 
                                   tuple(bgr_color), thickness)
                    else:
                        # Fade from bright green to dark green
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
    print(f"Matrix effect with flowing color text saved as {output_file}")


def create_matrix_with_bright_text(target_text="AutoSuper", output_file='matrix_bright_text.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect with large embedded text that remains bright
    
    Args:
        target_text: Text to embed in the digital rain using large repeated characters that stay bright
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
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
            
            # Check if this position is part of our target text pattern
            is_in_pattern = False
            pattern_char = None
            char_idx = -1
            
            # Calculate which character position in the target text this column might be part of
            for idx, char in enumerate(target_text.upper()):
                if char in char_patterns:
                    # Calculate the column range for this character
                    char_start_col = start_col + idx * (char_width + spacing)
                    char_end_col = char_start_col + char_width
                    
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
            
            if is_in_pattern:
                # Calculate color based on character index and time for overall color shift
                base_hue = char_colors[char_idx]
                # Add time-based shift to make colors cycle smoothly
                time_shift = int((frame_idx / fps) * 20) % 180  # Shift colors over time
                hue = (base_hue + time_shift) % 180
                
                # Ensure full saturation and brightness for vibrant colors
                saturation = 255  
                value = 255       
                
                # Convert HSV to BGR for OpenCV
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                
                # Display the character in consistent bright color for the whole character
                cv2.putText(img, pattern_char, (x, y), font, font_scale, tuple(bgr_color), thickness)
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - keeping them bright too
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text pattern
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    is_trail_in_pattern = False
                    trail_pattern_char = None
                    trail_char_idx = -1
                    
                    for idx, char_t in enumerate(target_text.upper()):
                        if char_t in char_patterns:
                            # Calculate the column range for this character
                            char_start_col = start_col + idx * (char_width + spacing)
                            char_end_col = char_start_col + char_width
                            
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
                    
                    if is_trail_in_pattern:
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
    print(f"Matrix effect with bright text saved as {output_file}")


def create_matrix_with_consistent_color_text(target_text="AutoSuper", output_file='matrix_consistent_color_text.mp4', duration=10, fps=30):
    """
    Creates a Matrix-style digital rain effect with large embedded text that has consistent color per character
    
    Args:
        target_text: Text to embed in the digital rain using large repeated characters with consistent color per character
        output_file: Output video file name
        duration: Duration of the video in seconds
        fps: Frames per second
    """
    
    # Define video properties
    height, width = 720, 1280
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
            
            # Check if this position is part of our target text pattern
            is_in_pattern = False
            pattern_char = None
            char_idx = -1
            
            # Calculate which character position in the target text this column might be part of
            for idx, char in enumerate(target_text.upper()):
                if char in char_patterns:
                    # Calculate the column range for this character
                    char_start_col = start_col + idx * (char_width + spacing)
                    char_end_col = char_start_col + char_width
                    
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
            
            if is_in_pattern:
                # Calculate color based on character index and time for overall color shift
                base_hue = char_colors[char_idx]
                # Add time-based shift to make colors cycle smoothly
                time_shift = int((frame_idx / fps) * 20) % 180  # Shift colors over time
                hue = (base_hue + time_shift) % 180
                
                # Ensure full saturation and brightness for vibrant colors
                saturation = 255  
                value = 255       
                
                # Convert HSV to BGR for OpenCV
                hsv_color = np.uint8([[[hue, saturation, value]]])
                bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                
                # Display the character in consistent bright color for the whole character
                cv2.putText(img, pattern_char, (x, y), font, font_scale, tuple(bgr_color), thickness)
            else:
                # Get a random character - mostly '0' and '1' for binary effect, sometimes letters
                char = random.choice(['0', '1'] + [chr(i) for i in range(33, 126)]*2)  # More binary chars
                
                # Draw the character - bright green for head of trail, darker green for tail
                cv2.putText(img, char, (x, y), font, font_scale, (0, 255, 0), thickness)
            
            # Draw trailing characters - fade effect
            for j in range(1, 10):  # 10 characters in trail
                if y - j * font_size > 0:
                    # Check if this trailing position could be part of our target text pattern
                    trail_row_pos = int((y - j * font_size) // font_size)
                    
                    is_trail_in_pattern = False
                    trail_pattern_char = None
                    trail_char_idx = -1
                    
                    for idx, char_t in enumerate(target_text.upper()):
                        if char_t in char_patterns:
                            # Calculate the column range for this character
                            char_start_col = start_col + idx * (char_width + spacing)
                            char_end_col = char_start_col + char_width
                            
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
                    
                    if is_trail_in_pattern:
                        # Calculate consistent color for trailing characters
                        base_hue = char_colors[trail_char_idx]
                        time_shift = int((frame_idx / fps) * 20) % 180
                        hue = (base_hue + time_shift) % 180
                        
                        # Reduce saturation and brightness with trail length for fading effect
                        saturation = max(100, 255 - j * 15)
                        value = max(100, 255 - j * 15)
                        
                        hsv_color = np.uint8([[[hue, saturation, value]]])
                        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0][0].tolist()
                        
                        cv2.putText(img, trail_pattern_char, (x, y - j * font_size), font, font_scale, 
                                   tuple(bgr_color), thickness)
                    else:
                        # Fade from bright green to dark green
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
    print(f"Matrix effect with consistent color text saved as {output_file}")


if __name__ == "__main__":
    # Text to display
    text_lines = [
        "欢迎来到黑客帝国",
        "Welcome to the Matrix",
        "这里是自由的世界",
        "This is the free world"
    ]
    
    print("选择效果:")
    print("1. 数字雨效果")
    print("2. 滚动文字效果")
    print("3. 组合效果（数字雨+滚动文字）")
    print("4. 明亮大文本的数字雨效果（显示自定义文本）")
    
    choice = input("请输入选择 (1, 2, 3 或 4): ")
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if choice == "1":
        output_filename = f"matrix_digital_rain_{timestamp}.mp4"
        create_matrix_effect(text_lines, output_file=output_filename)
    elif choice == "2":
        output_filename = f"matrix_scroll_{timestamp}.mp4"
        scroll_text_effect(text_lines, output_file=output_filename)
    elif choice == "3":
        output_filename = f"matrix_combined_{timestamp}.mp4"
        combined_effect(text_lines, output_file=output_filename)
    elif choice == "4":
        custom_text = input("请输入要在数字雨中显示的文本 (默认为 'AutoSuper'): ").strip()
        if not custom_text:
            custom_text = "AutoSuper"
        output_filename = f"matrix_bright_text_{timestamp}.mp4"
        create_matrix_with_bright_text(target_text=custom_text, output_file=output_filename)
    else:
        print("无效选择，运行明亮大文本效果")
        custom_text = "AutoSuper"
        output_filename = f"matrix_bright_text_{timestamp}.mp4"
        create_matrix_with_bright_text(target_text=custom_text, output_file=output_filename)
