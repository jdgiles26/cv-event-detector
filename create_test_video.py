#!/usr/bin/env python3
"""
Create a test video with moving objects for CV Event Detector testing
"""

import cv2
import numpy as np
import os

def create_test_video(output_path="/tmp/test_video.mp4", duration_sec=30, fps=30):
    """Create a test video with simulated moving objects"""
    
    width, height = 1280, 720
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    num_frames = duration_sec * fps
    
    # Object states
    objects = [
        {"x": 100, "y": 200, "w": 60, "h": 100, "vx": 3, "vy": 0.5, "color": (0, 0, 255), "label": "person"},
        {"x": 400, "y": 300, "w": 80, "h": 120, "vx": -2, "vy": 1, "color": (255, 0, 0), "label": "person"},
        {"x": 800, "y": 400, "w": 150, "h": 80, "vx": 4, "vy": -0.5, "color": (0, 255, 0), "label": "car"},
        {"x": 200, "y": 500, "w": 180, "h": 90, "vx": -3, "vy": 0, "color": (255, 255, 0), "label": "truck"},
    ]
    
    for frame_num in range(num_frames):
        # Create background
        frame = np.ones((height, width, 3), dtype=np.uint8) * 50  # Dark gray
        
        # Draw some static elements (buildings, road)
        cv2.rectangle(frame, (0, 550), (width, height), (80, 80, 80), -1)  # Road
        
        # Update and draw objects
        for obj in objects:
            # Update position
            obj["x"] += obj["vx"]
            obj["y"] += obj["vy"]
            
            # Bounce off edges
            if obj["x"] < 0 or obj["x"] + obj["w"] > width:
                obj["vx"] *= -1
            if obj["y"] < 0 or obj["y"] + obj["h"] > height:
                obj["vy"] *= -1
            
            # Draw object (rectangle representing person/car)
            x, y, w, h = int(obj["x"]), int(obj["y"]), obj["w"], obj["h"]
            cv2.rectangle(frame, (x, y), (x + w, y + h), obj["color"], -1)
            
            # Add label
            cv2.putText(frame, obj["label"], (x, y - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Add some "features" to make it look like a person/car
            if obj["label"] == "person":
                # Head
                head_radius = w // 3
                cv2.circle(frame, (x + w//2, y + head_radius), head_radius, (200, 200, 200), -1)
            else:
                # Windows for cars
                cv2.rectangle(frame, (x + w//4, y + h//4), (x + 3*w//4, y + h//2), (100, 100, 150), -1)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add timestamp
        timestamp = frame_num / fps
        cv2.putText(frame, f"Time: {timestamp:.1f}s", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        writer.write(frame)
        
        if frame_num % 30 == 0:
            print(f"Generating frame {frame_num}/{num_frames}...")
    
    writer.release()
    print(f"✅ Test video created: {output_path}")
    print(f"   Duration: {duration_sec}s, FPS: {fps}, Resolution: {width}x{height}")
    return output_path

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Create a test video for CV Event Detector")
    parser.add_argument("--output", "-o", default="/tmp/test_video.mp4", help="Output video path")
    parser.add_argument("--duration", "-d", type=int, default=30, help="Duration in seconds")
    parser.add_argument("--fps", type=int, default=30, help="Frames per second")
    
    args = parser.parse_args()
    
    create_test_video(args.output, args.duration, args.fps)
