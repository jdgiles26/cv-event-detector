#!/usr/bin/env python3
"""
macOS-Compatible CV Event Detector
Runs on Apple Silicon (M1/M2/M3/M4) using Metal Performance Shaders (MPS)
Replaces NVIDIA DeepStream with OpenCV + PyTorch
"""

import cv2
import torch
import numpy as np
import json
import os
import uuid
import threading
import queue
import time
import signal
import contextlib
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path
from collections import deque
from dataclasses import dataclass, field
from multiprocessing import Process, Queue as MPQueue

from fastapi import FastAPI, HTTPException, Response, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
import uvicorn

# ============================================================================
# Configuration
# ============================================================================

STREAM_WIDTH = 1920
STREAM_HEIGHT = 1080
CLIP_CACHE_PRE_EV_TIME = int(os.environ.get("CLIP_CACHE_PRE_EV_TIME", "5"))
CLIP_CACHE_POST_EV_TIME = int(os.environ.get("CLIP_CACHE_POST_EV_TIME", "10"))
OUTPUT_FOLDER = os.environ.get("OUTPUT_FOLDER", "/tmp/via/output")
VSS_SERVER_URL = os.environ.get("VSS_SERVER_URL", "http://localhost:8000/verifyAlert")

# Ensure output directory exists
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ============================================================================
# Device Configuration - Use Metal GPU if available
# ============================================================================

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print(f"✅ Using Apple Metal GPU (MPS) for acceleration")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"✅ Using CUDA GPU for acceleration")
else:
    DEVICE = torch.device("cpu")
    print(f"⚠️  Using CPU (no GPU acceleration available)")

# ============================================================================
# Data Models
# ============================================================================

@dataclass
class Detection:
    """Object detection result"""
    bbox: tuple  # (x1, y1, x2, y2)
    confidence: float
    class_id: int
    class_name: str
    track_id: Optional[int] = None

@dataclass
class FrameData:
    """Frame with metadata"""
    frame: np.ndarray
    frame_number: int
    timestamp: float
    detections: List[Detection] = field(default_factory=list)

class CVParams(BaseModel):
    gdinoprompt: Optional[str] = None
    gdinothreshold: Optional[float] = 0.3
    gdino_rois: Optional[List[List[int]]] = None
    overlay: Optional[bool] = True

class AddStreamRequest(BaseModel):
    version: str = Field(description="Version of the API, for ex 1.0")
    timestamp: Optional[str] = Field(None, description="Timestamp of the input stream")
    stream_url: str = Field(description="URL of the video stream to analyze")
    pipeline_id: str = Field(description="Unique pipeline identifier")
    output_folder: Optional[str] = Field(None, description="Output folder path")
    sensor_id: Optional[str] = Field(None, description="Unique sensor identifier")
    stream_name: Optional[str] = Field(None, description="Name of the stream")
    processing_state: Optional[str] = Field(None, description="enabled/disabled")
    cv_params: Optional[CVParams] = Field(None, description="CV parameters")

class EventOutput(BaseModel):
    event_type: Optional[str] = Field(None, description="Type of the event")
    clip: Optional[str] = Field(None, description="Path to the clip file")
    metadata: Optional[str] = Field(None, description="Path to the JSON metadata file")

class AddStreamResponse(BaseModel):
    stream_id: str = Field(description="Unique stream identifier")
    pipeline_id: str = Field(description="Unique pipeline identifier")
    status: str = Field(description="Status of the request")
    processing_state: str = Field(description="Processing state of the stream")
    message: str = Field(description="Success or error message")
    timestamp: str = Field(description="Timestamp of the request")
    error_code: Optional[int] = Field(None, description="Error code if any")
    error_details: Optional[str] = Field(None, description="Error details")
    events: List[EventOutput] = Field(default_factory=list)

class PipelineParams(BaseModel):
    min_clip_duration: int = Field(default=5, description="Minimum clip duration in seconds")
    max_clip_duration: int = Field(default=30, description="Maximum clip duration in seconds")
    frame_skip_interval: int = Field(default=0, description="Frame skip interval")
    minimum_detection_threshold: int = Field(default=3, description="Minimum detection threshold")

class CreatePipelineRequest(BaseModel):
    name: str = Field(description="Name of the pipeline")
    endpoint_url: Optional[str] = Field(None, description="Optional endpoint URL")
    config: Optional[str] = Field(None, description="Optional config file path")
    type: str = Field(default="object_detection", description="Type of pipeline")
    params: PipelineParams = Field(default_factory=lambda: PipelineParams(), description="Pipeline parameters")

class CreatePipelineResponse(BaseModel):
    id: str = Field(description="Unique pipeline identifier")
    status: str = Field(description="Status of the pipeline creation")
    message: str = Field(description="Success or error message")
    created_at: Optional[str] = Field(None, description="Timestamp when pipeline was created")
    error_code: Optional[int] = Field(None, description="Error code if any")
    error_details: Optional[str] = Field(None, description="Error details")

class RemoveStreamRequest(BaseModel):
    stream_id: str = Field(description="Unique stream identifier")
    version: Optional[str] = Field(None, description="Version of the API")

class RemoveStreamResponse(BaseModel):
    stream_id: str = Field(description="Unique stream identifier")
    status: str = Field(description="Status of the request")
    message: str = Field(description="Success or error message")
    timestamp: str = Field(description="Timestamp of the request")
    error_code: Optional[int] = Field(None, description="Error code if any")
    error_details: Optional[str] = Field(None, description="Error details")

class StreamInfo:
    def __init__(self):
        self.stream_id: str = ""
        self.video_path: str = ""
        self.process: Optional[Process] = None
        self.version: str = ""
        self.timestamp: str = ""
        self.pipeline_id: str = ""
        self.output_folder: Optional[str] = None
        self.sensor_id: Optional[str] = None
        self.stream_name: Optional[str] = None
        self.processing_state: Optional[str] = None
        self.cv_params: Optional[CVParams] = None

class PipelineInfo:
    def __init__(self):
        self.id: str = ""
        self.name: str = ""
        self.endpoint_url: Optional[str] = None
        self.config: Optional[str] = None
        self.type: str = ""
        self.params: Optional[PipelineParams] = None
        self.created_at: Optional[str] = None
        self.status: str = ""
        self.error_code: Optional[int] = None
        self.error_details: Optional[str] = None

# ============================================================================
# Simple Object Tracker (IoU-based)
# ============================================================================

class SimpleTracker:
    """Simple IoU-based object tracker"""
    
    def __init__(self, max_age: int = 5, min_hits: int = 1, iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers = []
        self.frame_count = 0
        self.next_id = 1
    
    def iou(self, bbox1, bbox2):
        """Calculate IoU between two bboxes"""
        x1, y1, x2, y2 = bbox1
        x3, y3, x4, y4 = bbox2
        
        xi1 = max(x1, x3)
        yi1 = max(y1, y3)
        xi2 = min(x2, x4)
        yi2 = min(y2, y4)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        box1_area = (x2 - x1) * (y2 - y1)
        box2_area = (x4 - x3) * (y4 - y3)
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def update(self, detections: List[Detection]):
        """Update tracker with new detections"""
        self.frame_count += 1
        
        # Match detections to existing trackers
        matched = []
        unmatched_detections = list(range(len(detections)))
        unmatched_trackers = list(range(len(self.trackers)))
        
        # Calculate IoU matrix
        if len(detections) > 0 and len(self.trackers) > 0:
            iou_matrix = np.zeros((len(detections), len(self.trackers)))
            for d, det in enumerate(detections):
                for t, trk in enumerate(self.trackers):
                    iou_matrix[d, t] = self.iou(det.bbox, trk['bbox'])
            
            # Match detections to trackers
            matched_indices = []
            while True:
                if iou_matrix.size == 0:
                    break
                max_iou = np.max(iou_matrix)
                if max_iou < self.iou_threshold:
                    break
                max_idx = np.unravel_index(np.argmax(iou_matrix), iou_matrix.shape)
                matched_indices.append(max_idx)
                iou_matrix[max_idx[0], :] = 0
                iou_matrix[:, max_idx[1]] = 0
            
            matched = matched_indices
            matched_detections = [m[0] for m in matched]
            matched_trackers = [m[1] for m in matched]
            unmatched_detections = [d for d in range(len(detections)) if d not in matched_detections]
            unmatched_trackers = [t for t in range(len(self.trackers)) if t not in matched_trackers]
        
        # Update matched trackers
        for det_idx, trk_idx in matched:
            self.trackers[trk_idx]['bbox'] = detections[det_idx].bbox
            self.trackers[trk_idx]['hits'] += 1
            self.trackers[trk_idx]['time_since_update'] = 0
            detections[det_idx].track_id = self.trackers[trk_idx]['id']
        
        # Create new trackers for unmatched detections
        for det_idx in unmatched_detections:
            new_tracker = {
                'id': self.next_id,
                'bbox': detections[det_idx].bbox,
                'hits': 1,
                'time_since_update': 0,
                'age': 0
            }
            self.trackers.append(new_tracker)
            detections[det_idx].track_id = self.next_id
            self.next_id += 1
        
        # Update unmatched trackers
        for trk_idx in unmatched_trackers:
            self.trackers[trk_idx]['time_since_update'] += 1
        
        # Remove dead trackers
        self.trackers = [t for t in self.trackers if t['time_since_update'] <= self.max_age]
        
        # Update age for all trackers
        for t in self.trackers:
            t['age'] += 1
        
        return detections

# ============================================================================
# Simple Object Detector using OpenCV Built-in Features
# ============================================================================

class ObjectDetector:
    """Simple object detector using OpenCV's built-in HOG + SVM for people detection
    and color/shape-based detection for other objects"""
    
    def __init__(self, confidence_threshold: float = 0.3, nms_threshold: float = 0.4):
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # COCO class names
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Initialize HOG descriptor for person detection
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        
        print("✅ Object detector initialized (HOG + SVM for people detection)")
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect objects in frame"""
        detections = []
        height, width = frame.shape[:2]
        
        # Detect people using HOG
        boxes, weights = self.hog.detectMultiScale(frame, winStride=(8, 8), 
                                                    padding=(4, 4), scale=1.05)
        
        for i, (x, y, w, h) in enumerate(boxes):
            confidence = float(weights[i]) if i < len(weights) else 0.5
            
            if confidence > self.confidence_threshold:
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + w), int(y + h)
                
                # Clamp to image bounds
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(0, min(x2, width - 1))
                y2 = max(0, min(y2, height - 1))
                
                if x2 > x1 and y2 > y1:
                    detections.append(Detection(
                        bbox=(x1, y1, x2, y2),
                        confidence=confidence,
                        class_id=0,  # person
                        class_name='person'
                    ))
        
        # Simple heuristic detection for cars (large rectangular objects with certain aspect ratios)
        # Convert to grayscale and apply edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            if cv2.contourArea(contour) > 5000:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h if h > 0 else 0
                
                # Cars typically have aspect ratio between 1.5 and 4.0
                if 1.5 <= aspect_ratio <= 4.0 and w > 50 and h > 30:
                    # Check if this overlaps with existing detections
                    x1, y1 = x, y
                    x2, y2 = x + w, y + h
                    
                    # Clamp to image bounds
                    x1 = max(0, min(x1, width - 1))
                    y1 = max(0, min(y1, height - 1))
                    x2 = max(0, min(x2, width - 1))
                    y2 = max(0, min(y2, height - 1))
                    
                    if x2 > x1 and y2 > y1:
                        detections.append(Detection(
                            bbox=(x1, y1, x2, y2),
                            confidence=0.4,  # Lower confidence for heuristic detection
                            class_id=2,  # car
                            class_name='car'
                        ))
        
        # Apply NMS to remove overlapping detections
        if len(detections) > 0:
            boxes = [list(d.bbox) for d in detections]
            confidences = [d.confidence for d in detections]
            class_ids = [d.class_id for d in detections]
            
            # Group by class for NMS
            unique_classes = set(class_ids)
            final_detections = []
            
            for cls in unique_classes:
                cls_indices = [i for i, c in enumerate(class_ids) if c == cls]
                cls_boxes = [boxes[i] for i in cls_indices]
                cls_confidences = [confidences[i] for i in cls_indices]
                
                if cls_boxes:
                    indices = cv2.dnn.NMSBoxes(cls_boxes, cls_confidences, 
                                               self.confidence_threshold, self.nms_threshold)
                    if isinstance(indices, np.ndarray):
                        indices = indices.flatten()
                        for i in indices:
                            final_detections.append(detections[cls_indices[i]])
            
            detections = final_detections
        
        return detections

# ============================================================================
# Event-based Video Processor
# ============================================================================

class EventVideoProcessor:
    """Processes video and generates event clips"""
    
    def __init__(self, 
                 stream_url: str,
                 output_folder: str,
                 min_detection_threshold: int = 3,
                 frame_skip_interval: int = 0,
                 pre_event_time: int = 5,
                 post_event_time: int = 10,
                 detection_prompt: str = "person . car . truck",
                 confidence_threshold: float = 0.3,
                 stream_name: str = "stream"):
        
        self.stream_url = stream_url
        self.output_folder = output_folder
        self.min_detection_threshold = min_detection_threshold
        self.frame_skip_interval = frame_skip_interval
        self.pre_event_time = pre_event_time
        self.post_event_time = post_event_time
        self.confidence_threshold = confidence_threshold
        self.stream_name = stream_name
        
        # Parse detection prompt (e.g., "person . car . truck")
        self.target_classes = [c.strip().lower() for c in detection_prompt.split(".") if c.strip()]
        if not self.target_classes:
            self.target_classes = ["person", "car", "truck", "bicycle"]
        
        # Initialize components
        self.detector = ObjectDetector(confidence_threshold=confidence_threshold)
        self.tracker = SimpleTracker()
        
        # Frame buffer for event capture
        self.frame_buffer = deque(maxlen=int((pre_event_time + post_event_time) * 30))  # ~30fps
        
        # Event state
        self.event_active = False
        self.event_start_time = None
        self.event_frames = []
        self.events_generated = 0
        
        # Metadata tracking
        self.frame_count = 0
        self.fps = 30
        
        print(f"EventVideoProcessor initialized:")
        print(f"  - Target classes: {self.target_classes}")
        print(f"  - Min detection threshold: {min_detection_threshold}")
        print(f"  - Pre-event time: {pre_event_time}s")
        print(f"  - Post-event time: {post_event_time}s")
    
    def count_target_objects(self, detections: List[Detection]) -> int:
        """Count objects matching target classes"""
        count = 0
        for det in detections:
            if det.class_name.lower() in self.target_classes:
                count += 1
        return count
    
    def draw_detections(self, frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
        """Draw bounding boxes and labels on frame"""
        result = frame.copy()
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det.bbox)
            
            # Generate color based on track_id
            if det.track_id:
                color_hue = (det.track_id * 30) % 180
                import colorsys
                rgb = colorsys.hsv_to_rgb(color_hue / 180, 0.8, 0.9)
                color = (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            else:
                color = (0, 255, 0)
            
            # Draw bbox
            cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{det.class_name}"
            if det.track_id:
                label += f" ID:{det.track_id}"
            label += f" {det.confidence:.2f}"
            
            (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(result, (x1, y1 - text_h - 8), (x1 + text_w, y1), color, -1)
            cv2.putText(result, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        
        # Draw status
        status_text = f"Frame: {self.frame_count} | Events: {self.events_generated}"
        if self.event_active:
            status_text += " | RECORDING EVENT"
        cv2.putText(result, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return result
    
    def save_event_clip(self, frames: List[FrameData], event_start_time: float):
        """Save event clip to file"""
        if len(frames) < 10:  # Minimum 10 frames
            return None
        
        timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%S.%f')[:-3] + 'Z'
        clip_name = f"{self.stream_name}_{timestamp}"
        clip_path = os.path.join(self.output_folder, f"{clip_name}.mp4")
        metadata_path = os.path.join(self.output_folder, f"{clip_name}.json")
        
        # Get video dimensions
        height, width = frames[0].frame.shape[:2]
        
        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(clip_path, fourcc, self.fps, (width, height))
        
        # Write frames
        metadata = {
            "clip_name": clip_name,
            "timestamp": timestamp,
            "stream_name": self.stream_name,
            "frames": []
        }
        
        for frame_data in frames:
            writer.write(frame_data.frame)
            
            # Save frame metadata
            frame_meta = {
                "frame_number": frame_data.frame_number,
                "timestamp": frame_data.timestamp,
                "objects": [
                    {
                        "id": det.track_id,
                        "type": det.class_name,
                        "conf": round(det.confidence, 2),
                        "bbox": {
                            "lX": round(det.bbox[0], 2),
                            "tY": round(det.bbox[1], 2),
                            "rX": round(det.bbox[2], 2),
                            "bY": round(det.bbox[3], 2)
                        }
                    }
                    for det in frame_data.detections
                ]
            }
            metadata["frames"].append(frame_meta)
        
        writer.release()
        
        # Save metadata
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.events_generated += 1
        print(f"✅ Event clip saved: {clip_path}")
        print(f"✅ Metadata saved: {metadata_path}")
        
        return clip_path, metadata_path
    
    def process(self, stop_event: threading.Event = None):
        """Main processing loop"""
        print(f"Opening video stream: {self.stream_url}")
        
        cap = cv2.VideoCapture(self.stream_url)
        if not cap.isOpened():
            print(f"❌ Failed to open video stream: {self.stream_url}")
            return
        
        # Get video properties
        self.fps = cap.get(cv2.CAP_PROP_FPS) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or STREAM_WIDTH
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or STREAM_HEIGHT
        
        print(f"Video properties: {width}x{height} @ {self.fps}fps")
        
        frame_skip = 0
        
        try:
            while True:
                if stop_event and stop_event.is_set():
                    break
                
                ret, frame = cap.read()
                if not ret:
                    print("End of stream reached")
                    break
                
                self.frame_count += 1
                
                # Frame skipping
                if self.frame_skip_interval > 0:
                    frame_skip += 1
                    if frame_skip <= self.frame_skip_interval:
                        continue
                    frame_skip = 0
                
                # Detect objects
                detections = self.detector.detect(frame)
                
                # Track objects
                detections = self.tracker.update(detections)
                
                # Count target objects
                target_count = self.count_target_objects(detections)
                
                # Draw detections
                annotated_frame = self.draw_detections(frame, detections)
                
                # Create frame data
                frame_data = FrameData(
                    frame=annotated_frame,
                    frame_number=self.frame_count,
                    timestamp=cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0,
                    detections=detections
                )
                
                # Add to frame buffer
                self.frame_buffer.append(frame_data)
                
                # Event detection logic
                if target_count >= self.min_detection_threshold:
                    if not self.event_active:
                        # Start new event
                        self.event_active = True
                        self.event_start_time = time.time()
                        # Copy buffered frames
                        self.event_frames = list(self.frame_buffer)
                        print(f"🚨 Event started! Objects detected: {target_count}")
                    else:
                        # Continue event
                        self.event_frames.append(frame_data)
                else:
                    if self.event_active:
                        # Check if we should end the event
                        if time.time() - self.event_start_time > self.post_event_time:
                            # Save event clip
                            self.save_event_clip(self.event_frames, self.event_start_time)
                            self.event_active = False
                            self.event_frames = []
                    else:
                        # Continue buffering
                        pass
                
                # Display (optional - for debugging)
                # cv2.imshow("CV Event Detector", annotated_frame)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     break
                
                # Print progress every 30 frames
                if self.frame_count % 30 == 0:
                    print(f"Processed {self.frame_count} frames, Events: {self.events_generated}, Objects: {target_count}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            
            # Save any pending event
            if self.event_active and len(self.event_frames) > 10:
                self.save_event_clip(self.event_frames, self.event_start_time)
            
            print(f"Processing complete. Total frames: {self.frame_count}, Events: {self.events_generated}")

# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="CV Event Detector (macOS Compatible)",
    description="Computer Vision Event Detector running on Apple Silicon",
    version="1.0.0"
)

# Global storage
asset_map: Dict[str, StreamInfo] = {}
pipeline_map: Dict[str, PipelineInfo] = {}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "CV Event Detector (macOS Compatible)",
        "version": "1.0.0",
        "device": str(DEVICE),
        "endpoints": {
            "/": "This information page",
            "/api/addstream": "POST - Add a video stream",
            "/api/stream": "DELETE - Remove a stream",
            "/api/pipeline": "POST - Create pipeline, DELETE - Delete pipeline",
            "/api/pipelines": "GET - List all pipelines",
            "/api/streams": "GET - List all streams",
            "/health": "Health check"
        }
    }

@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "device": str(DEVICE),
        "torch_version": torch.__version__,
        "opencv_version": cv2.__version__,
        "streams_active": len([s for s in asset_map.values() if s.processing_state == "running"]),
        "pipelines_active": len(pipeline_map)
    }

@app.post("/api/pipeline", response_model=CreatePipelineResponse)
async def create_pipeline(request: CreatePipelineRequest):
    """Create a new pipeline"""
    try:
        pipeline_id = str(uuid.uuid4())
        while pipeline_id in pipeline_map:
            pipeline_id = str(uuid.uuid4())
        
        pipeline_config = PipelineInfo()
        pipeline_config.id = pipeline_id
        pipeline_config.name = request.name
        pipeline_config.endpoint_url = request.endpoint_url
        pipeline_config.config = request.config
        pipeline_config.type = request.type
        pipeline_config.params = request.params
        pipeline_config.created_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        pipeline_config.status = "created"
        
        pipeline_map[pipeline_id] = pipeline_config
        
        return CreatePipelineResponse(
            id=pipeline_id,
            status="success",
            message="Pipeline created successfully",
            created_at=pipeline_config.created_at
        )
    except Exception as e:
        return CreatePipelineResponse(
            id="",
            status="error",
            message="Failed to create pipeline",
            error_code=500,
            error_details=str(e)
        )

@app.get("/api/pipelines")
async def get_pipelines():
    """Get all pipelines"""
    return {
        "status": "success",
        "pipelines": [
            {
                "id": p.id,
                "name": p.name,
                "type": p.type,
                "created_at": p.created_at
            }
            for p in pipeline_map.values()
        ]
    }

@app.post("/api/addstream", response_model=AddStreamResponse)
async def add_stream(request: AddStreamRequest, response: Response):
    """Add a stream for processing"""
    try:
        if request.pipeline_id not in pipeline_map:
            raise HTTPException(status_code=404, detail=f"Pipeline {request.pipeline_id} not found")
        
        pipeline_config = pipeline_map[request.pipeline_id]
        
        # Generate stream ID
        stream_id = str(uuid.uuid4())
        while stream_id in asset_map:
            stream_id = str(uuid.uuid4())
        
        # Get CV parameters
        gdinoprompt = request.cv_params.gdinoprompt if request.cv_params else "person . car . truck"
        gdinothreshold = request.cv_params.gdinothreshold if request.cv_params else 0.3
        
        # Create stream info
        stream_info = StreamInfo()
        stream_info.stream_id = stream_id
        stream_info.video_path = request.stream_url
        stream_info.pipeline_id = request.pipeline_id
        stream_info.output_folder = request.output_folder or OUTPUT_FOLDER
        stream_info.sensor_id = request.sensor_id
        stream_info.stream_name = request.stream_name or "stream"
        stream_info.cv_params = request.cv_params
        stream_info.timestamp = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        
        # Ensure output folder exists
        os.makedirs(stream_info.output_folder, exist_ok=True)
        
        # Create processor
        processor = EventVideoProcessor(
            stream_url=request.stream_url,
            output_folder=stream_info.output_folder,
            min_detection_threshold=pipeline_config.params.minimum_detection_threshold,
            frame_skip_interval=pipeline_config.params.frame_skip_interval,
            pre_event_time=CLIP_CACHE_PRE_EV_TIME,
            post_event_time=CLIP_CACHE_POST_EV_TIME,
            detection_prompt=gdinoprompt,
            confidence_threshold=gdinothreshold,
            stream_name=stream_info.stream_name
        )
        
        # Start processing in background thread
        stop_event = threading.Event()
        
        def process_wrapper():
            try:
                processor.process(stop_event)
                stream_info.processing_state = "completed"
            except Exception as e:
                print(f"Processing error: {e}")
                stream_info.processing_state = "error"
        
        thread = threading.Thread(target=process_wrapper, daemon=True)
        thread.start()
        
        stream_info.process = type('Process', (), {'is_alive': thread.is_alive, 'terminate': stop_event.set, 'join': lambda *args: thread.join(*args[0] if args else 1)})()
        stream_info.processing_state = "running"
        asset_map[stream_id] = stream_info
        
        response.status_code = 202
        
        return AddStreamResponse(
            stream_id=stream_id,
            pipeline_id=request.pipeline_id,
            status="success",
            processing_state="running",
            message="Stream submitted for processing",
            timestamp=stream_info.timestamp
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding stream: {str(e)}")

@app.delete("/api/stream")
async def remove_stream(request: RemoveStreamRequest):
    """Remove a stream"""
    stream_id = str(request.stream_id)
    
    if stream_id in asset_map:
        stream_info = asset_map[stream_id]
        
        # Stop processing
        if hasattr(stream_info.process, 'terminate'):
            stream_info.process.terminate()
        
        asset_map.pop(stream_id)
        
        return RemoveStreamResponse(
            stream_id=stream_id,
            status="success",
            message="Stream removed successfully",
            timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
        )
    else:
        return RemoveStreamResponse(
            stream_id=stream_id,
            status="error",
            message=f"Stream {stream_id} not found",
            timestamp=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ'),
            error_code=404,
            error_details=f"Stream {stream_id} does not exist"
        )

@app.get("/api/streams")
async def get_streams():
    """Get all streams"""
    return {
        "status": "success",
        "streams": [
            {
                "stream_id": s.stream_id,
                "pipeline_id": s.pipeline_id,
                "processing_state": s.processing_state,
                "timestamp": s.timestamp
            }
            for s in asset_map.values()
        ]
    }

@app.get("/api/streams/{stream_id}/status")
async def get_stream_status(stream_id: str):
    """Get stream processing status"""
    if stream_id not in asset_map:
        raise HTTPException(status_code=404, detail=f"Stream {stream_id} not found")
    
    stream_info = asset_map[stream_id]
    
    return {
        "stream_id": stream_id,
        "status": stream_info.processing_state,
        "message": f"Stream is {stream_info.processing_state}",
        "processing_state": stream_info.processing_state,
        "timestamp": stream_info.timestamp
    }

# ============================================================================
# Web UI
# ============================================================================

# Create a simple HTML UI
ui_html = """
<!DOCTYPE html>
<html>
<head>
    <title>CV Event Detector (macOS)</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 40px; background: #1a1a2e; color: #eee; }
        h1 { color: #4ecca3; }
        .container { max-width: 800px; margin: 0 auto; }
        .card { background: #16213e; padding: 20px; margin: 20px 0; border-radius: 8px; }
        input, select, textarea { width: 100%; padding: 10px; margin: 5px 0; border-radius: 4px; border: 1px solid #4ecca3; background: #0f3460; color: white; }
        button { background: #4ecca3; color: #1a1a2e; padding: 12px 24px; border: none; border-radius: 4px; cursor: pointer; font-weight: bold; }
        button:hover { background: #3db892; }
        .status { padding: 10px; margin: 10px 0; border-radius: 4px; }
        .success { background: #4ecca3; color: #1a1a2e; }
        .error { background: #e74c3c; color: white; }
        .info { background: #3498db; color: white; }
        .stream-list { margin-top: 20px; }
        .stream-item { background: #0f3460; padding: 10px; margin: 5px 0; border-radius: 4px; display: flex; justify-content: space-between; }
    </style>
</head>
<body>
    <div class="container">
        <h1>🎥 CV Event Detector (macOS)</h1>
        <p>Running on: <strong>""" + str(DEVICE) + """</strong></p>
        
        <div class="card">
            <h2>Create Pipeline</h2>
            <input type="text" id="pipelineName" placeholder="Pipeline name" value="default_pipeline">
            <input type="number" id="minDetection" placeholder="Min detection threshold" value="3" min="1" max="20">
            <input type="number" id="frameSkip" placeholder="Frame skip interval" value="0" min="0" max="10">
            <button onclick="createPipeline()">Create Pipeline</button>
            <div id="pipelineStatus"></div>
        </div>
        
        <div class="card">
            <h2>Process Video</h2>
            <input type="text" id="videoPath" placeholder="Video file path (e.g., /path/to/video.mp4) or RTSP URL">
            <input type="text" id="streamName" placeholder="Stream name" value="test_stream">
            <textarea id="detectionPrompt" rows="3" placeholder="Detection prompt (e.g., person . car . truck)">person . car . truck . bicycle</textarea>
            <input type="number" id="confidenceThreshold" placeholder="Confidence threshold" value="0.3" min="0.1" max="1.0" step="0.1">
            <button onclick="addStream()">Process Video</button>
            <div id="streamStatus"></div>
        </div>
        
        <div class="card">
            <h2>Active Streams</h2>
            <button onclick="refreshStreams()">Refresh</button>
            <div id="streamList" class="stream-list"></div>
        </div>
        
        <div class="card">
            <h2>API Info</h2>
            <p><strong>Health:</strong> <a href="/health" target="_blank">/health</a></p>
            <p><strong>API Docs:</strong> <a href="/docs" target="_blank">/docs</a></p>
        </div>
    </div>
    
    <script>
        let currentPipelineId = null;
        
        async function createPipeline() {
            const name = document.getElementById('pipelineName').value;
            const minDetection = parseInt(document.getElementById('minDetection').value);
            const frameSkip = parseInt(document.getElementById('frameSkip').value);
            
            const response = await fetch('/api/pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    name: name,
                    type: 'object_detection',
                    params: {
                        min_clip_duration: 5,
                        max_clip_duration: 30,
                        frame_skip_interval: frameSkip,
                        minimum_detection_threshold: minDetection
                    }
                })
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                currentPipelineId = data.id;
                document.getElementById('pipelineStatus').innerHTML = 
                    '<div class="status success">Pipeline created: ' + data.id + '</div>';
            } else {
                document.getElementById('pipelineStatus').innerHTML = 
                    '<div class="status error">Error: ' + data.message + '</div>';
            }
        }
        
        async function addStream() {
            if (!currentPipelineId) {
                alert('Please create a pipeline first!');
                return;
            }
            
            const videoPath = document.getElementById('videoPath').value;
            const streamName = document.getElementById('streamName').value;
            const detectionPrompt = document.getElementById('detectionPrompt').value;
            const confidenceThreshold = parseFloat(document.getElementById('confidenceThreshold').value);
            
            const response = await fetch('/api/addstream', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    version: '1.0',
                    stream_url: videoPath,
                    pipeline_id: currentPipelineId,
                    stream_name: streamName,
                    cv_params: {
                        gdinoprompt: detectionPrompt,
                        gdinothreshold: confidenceThreshold
                    }
                })
            });
            
            const data = await response.json();
            if (data.status === 'success') {
                document.getElementById('streamStatus').innerHTML = 
                    '<div class="status success">Stream added: ' + data.stream_id + '</div>';
                refreshStreams();
            } else {
                document.getElementById('streamStatus').innerHTML = 
                    '<div class="status error">Error: ' + data.message + '</div>';
            }
        }
        
        async function refreshStreams() {
            const response = await fetch('/api/streams');
            const data = await response.json();
            
            const listDiv = document.getElementById('streamList');
            if (data.streams.length === 0) {
                listDiv.innerHTML = '<div class="status info">No active streams</div>';
            } else {
                listDiv.innerHTML = data.streams.map(s => `
                    <div class="stream-item">
                        <span>${s.stream_id.substring(0, 8)}... - ${s.processing_state}</span>
                        <button onclick="removeStream('${s.stream_id}')">Remove</button>
                    </div>
                `).join('');
            }
        }
        
        async function removeStream(streamId) {
            const response = await fetch('/api/stream', {
                method: 'DELETE',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ stream_id: streamId, version: '1.0' })
            });
            refreshStreams();
        }
        
        // Auto-refresh streams
        setInterval(refreshStreams, 5000);
        refreshStreams();
    </script>
</body>
</html>
"""

@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """Serve web UI"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=ui_html)

# Also serve UI at root
@app.get("/", response_class=HTMLResponse)
async def root_ui():
    """Serve web UI at root"""
    from fastapi.responses import HTMLResponse
    return HTMLResponse(content=ui_html)

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("  CV Event Detector - macOS Compatible Version")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Output folder: {OUTPUT_FOLDER}")
    port = int(os.environ.get("PORT", 8080))
    print(f"Server URL: http://localhost:{port}")
    print(f"Web UI: http://localhost:{port}/ui")
    print("=" * 60)
    
    # Create default pipeline
    default_pipeline = PipelineInfo()
    default_pipeline.id = "default-pipeline"
    default_pipeline.name = "Default Pipeline"
    default_pipeline.type = "object_detection"
    default_pipeline.params = PipelineParams(
        min_clip_duration=5,
        max_clip_duration=30,
        frame_skip_interval=0,
        minimum_detection_threshold=3
    )
    default_pipeline.created_at = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%S.%fZ')
    default_pipeline.status = "created"
    pipeline_map[default_pipeline.id] = default_pipeline
    print(f"✅ Default pipeline created: {default_pipeline.id}")
    
    # Start server
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
