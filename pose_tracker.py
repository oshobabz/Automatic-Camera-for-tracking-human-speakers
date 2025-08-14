import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Optional, Tuple, List, Dict

class PoseTracker:
    """
    A class for tracking human poses using MediaPipe Pose.
    Optimized for Raspberry Pi performance with reduced processing overhead.
    """
    
    def __init__(self, 
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5,
                 model_complexity=0):  # Use 0 for best performance on Pi
        """
        Initialize the pose tracker.
        
        Args:
            min_detection_confidence (float): Minimum confidence for pose detection
            min_tracking_confidence (float): Minimum confidence for pose tracking
            model_complexity (int): Model complexity (0=fastest, 2=most accurate)
        """
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize pose detection with Pi-optimized settings
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            enable_segmentation=False,  # Disabled for better performance
            smooth_landmarks=True,      # Enable smoothing to reduce jitter
            smooth_segmentation=False,  # Keep disabled
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Tracking state
        self.last_center = None
        self.tracking_active = False
        self.last_detection_time = 0
        self.detection_timeout = 2.0  # Increased timeout for more stable tracking
        
        # Performance optimization - skip frames for pose detection
        self.frame_skip_count = 0
        self.frame_skip_interval = 2  # Process every 3rd frame for pose detection
        self.last_pose_data = None
        
        # Smoothing parameters
        self.center_smoothing_factor = 0.7  # Higher = more smoothing
        self.smoothed_center = None
        self.confidence_threshold = 0.5  # Higher threshold for better quality
        
        # Key landmarks for tracking (focusing on torso for stability)
        self.key_landmarks = [
            self.mp_pose.PoseLandmark.NOSE,
            self.mp_pose.PoseLandmark.LEFT_SHOULDER,
            self.mp_pose.PoseLandmark.RIGHT_SHOULDER,
            self.mp_pose.PoseLandmark.LEFT_ELBOW,
            self.mp_pose.PoseLandmark.RIGHT_ELBOW,
            # Removed wrists and hips for more stable center calculation
        ]
        
        # Performance tracking
        self.processing_times = []
        self.last_fps_update = 0
        self.current_fps = 0
        
    def detect_pose(self, frame: np.ndarray) -> Optional[Dict]:
        """
        Detect pose in a frame with performance optimizations.
        
        Args:
            frame (np.ndarray): Input frame (BGR format)
            
        Returns:
            dict or None: Pose detection results with center point and landmarks
        """
        start_time = time.time()
        
        # Frame skipping for performance - only process every nth frame
        self.frame_skip_count += 1
        if self.frame_skip_count < self.frame_skip_interval:
            # Return smoothed version of last detection
            if self.last_pose_data and self.tracking_active:
                return self._create_smoothed_pose_data()
            return None
        
        self.frame_skip_count = 0
        
        ## Resize frame for faster processing (optional - adjust based on your needs)
        # height, width = frame.shape[:2]
        # if width > 320:  # Only resize if frame is large
        #     scale_factor = 320.0 / width
        #     new_width = int(width * scale_factor)
        #     new_height = int(height * scale_factor)
        #     resized_frame = cv2.resize(frame, (new_width, new_height))
        #     scale_back = True
        # else:
        #     resized_frame = frame
        #     scale_factor = 1.0
        #     scale_back = False
        
        # # Convert BGR to RGB for MediaPipe
        # rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.pose.process(frame)
        
        processing_time = time.time() - start_time
        self._update_performance_stats(processing_time)
        
        if results.pose_landmarks:
            # # Scale landmarks back to original frame size if needed
            # if scale_back:
            #     self._scale_landmarks(results.pose_landmarks, 1.0 / scale_factor)
            
            # Calculate center point of key landmarks
            center_x, center_y = self._calculate_pose_center(
                results.pose_landmarks, frame.shape
            )
            
            # Apply smoothing to center point
            if self.smoothed_center is None:
                self.smoothed_center = (center_x, center_y)
            else:
                # Exponential smoothing
                smooth_x = (self.smoothed_center[0] * self.center_smoothing_factor + 
                           center_x * (1 - self.center_smoothing_factor))
                smooth_y = (self.smoothed_center[1] * self.center_smoothing_factor + 
                           center_y * (1 - self.center_smoothing_factor))
                self.smoothed_center = (int(smooth_x), int(smooth_y))
            
            # Calculate confidence
            confidence = self._calculate_pose_confidence(results.pose_landmarks)
            
            # Only update tracking if confidence is high enough
            if confidence >= self.confidence_threshold:
                # Update tracking state
                self.last_center = self.smoothed_center
                self.tracking_active = True
                self.last_detection_time = time.time()
                
                pose_data = {
                    'center': self.smoothed_center,
                    'raw_center': (center_x, center_y),  # Unsmoothed center for reference
                    'landmarks': results.pose_landmarks,
                    'confidence': confidence,
                    'bbox': self._calculate_bounding_box(results.pose_landmarks, frame.shape),
                    'processing_time': processing_time
                }
                
                self.last_pose_data = pose_data
                return pose_data
            else:
                print(f"Low confidence detection: {confidence:.2f} (threshold: {self.confidence_threshold})")
        
        # Check if tracking should timeout
        if time.time() - self.last_detection_time > self.detection_timeout:
            self.tracking_active = False
            self.last_center = None
            self.smoothed_center = None
            self.last_pose_data = None
        
        return None
    
    def _scale_landmarks(self, landmarks, scale_factor):
        """Scale landmarks back to original frame size."""
        for landmark in landmarks.landmark:
            landmark.x *= scale_factor
            landmark.y *= scale_factor
    
    def _create_smoothed_pose_data(self):
        """Create pose data using smoothed/interpolated values."""
        if not self.last_pose_data:
            return None
        
        # Return a copy of last pose data with current smoothed center
        smoothed_data = self.last_pose_data.copy()
        smoothed_data['center'] = self.smoothed_center
        smoothed_data['interpolated'] = True
        return smoothed_data
    
    def _update_performance_stats(self, processing_time):
        """Update performance statistics."""
        self.processing_times.append(processing_time)
        if len(self.processing_times) > 30:
            self.processing_times.pop(0)
        
        # Update FPS every second
        current_time = time.time()
        if current_time - self.last_fps_update >= 1.0:
            if self.processing_times:
                avg_time = sum(self.processing_times) / len(self.processing_times)
                self.current_fps = 1.0 / avg_time if avg_time > 0 else 0
            self.last_fps_update = current_time
    
    def _calculate_pose_center(self, landmarks, frame_shape) -> Tuple[int, int]:
        """
        Calculate the center point of the pose based on key landmarks.
        Focuses on upper body for more stable tracking.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the frame (height, width, channels)
            
        Returns:
            tuple: (center_x, center_y) in pixel coordinates
        """
        height, width = frame_shape[:2]
        
        # Get coordinates of key landmarks with weighted importance
        weighted_points = []
        
        for landmark_id in self.key_landmarks:
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > 0.5:  # Higher visibility threshold
                x = int(landmark.x * width)
                y = int(landmark.y * height)
                
                # Weight landmarks by importance (shoulders more important than elbows)
                if landmark_id in [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                                  self.mp_pose.PoseLandmark.RIGHT_SHOULDER]:
                    weight = 3.0  # Shoulders are most important
                elif landmark_id == self.mp_pose.PoseLandmark.NOSE:
                    weight = 2.0  # Nose is also important
                else:
                    weight = 1.0  # Elbows have normal weight
                
                for _ in range(int(weight)):
                    weighted_points.append((x, y))
        
        if not weighted_points:
            # Fallback to any visible landmark
            for landmark_id in self.key_landmarks:
                landmark = landmarks.landmark[landmark_id]
                if landmark.visibility > 0.5:
                    x = int(landmark.x * width)
                    y = int(landmark.y * height)
                    weighted_points.append((x, y))
        
        if not weighted_points:
            return width // 2, height // 2  # Return center if no landmarks found
        
        # Calculate weighted center
        center_x = int(np.mean([p[0] for p in weighted_points]))
        center_y = int(np.mean([p[1] for p in weighted_points]))
        
        # Clamp to frame boundaries
        center_x = max(0, min(width - 1, center_x))
        center_y = max(0, min(height - 1, center_y))
        
        return center_x, center_y
    
    def _calculate_pose_confidence(self, landmarks) -> float:
        """
        Calculate average confidence of key landmarks with weighting.
        
        Args:
            landmarks: MediaPipe pose landmarks
            
        Returns:
            float: Weighted average confidence score
        """
        weighted_confidences = []
        
        for landmark_id in self.key_landmarks:
            landmark = landmarks.landmark[landmark_id]
            confidence = landmark.visibility
            
            # Weight by importance
            if landmark_id in [self.mp_pose.PoseLandmark.LEFT_SHOULDER, 
                              self.mp_pose.PoseLandmark.RIGHT_SHOULDER]:
                weight = 3.0
            elif landmark_id == self.mp_pose.PoseLandmark.NOSE:
                weight = 2.0
            else:
                weight = 1.0
            
            for _ in range(int(weight)):
                weighted_confidences.append(confidence)
        
        return np.mean(weighted_confidences) if weighted_confidences else 0.0
    
    def _calculate_bounding_box(self, landmarks, frame_shape) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box around the pose with more conservative padding.
        
        Args:
            landmarks: MediaPipe pose landmarks
            frame_shape: Shape of the frame
            
        Returns:
            tuple: (x, y, width, height) of bounding box
        """
        height, width = frame_shape[:2]
        
        # Get all visible landmark coordinates
        x_coords = []
        y_coords = []
        
        for landmark_id in self.key_landmarks:
            landmark = landmarks.landmark[landmark_id]
            if landmark.visibility > 0.6:  # Higher threshold for bbox
                x_coords.append(int(landmark.x * width))
                y_coords.append(int(landmark.y * height))
        
        if not x_coords or not y_coords:
            return 0, 0, 0, 0
        
        # Calculate bounding box with conservative padding
        min_x, max_x = min(x_coords), max(x_coords)
        min_y, max_y = min(y_coords), max(y_coords)
        
        # Smaller padding for more precise tracking
        padding = 15
        x = max(0, min_x - padding)
        y = max(0, min_y - padding)
        w = min(width - x, max_x - min_x + 2 * padding)
        h = min(height - y, max_y - min_y + 2 * padding)
        
        return x, y, w, h
    
    def draw_pose(self, frame: np.ndarray, pose_data: Optional[Dict]) -> np.ndarray:
        """
        Draw pose landmarks and center point on the frame with performance info.
        
        Args:
            frame (np.ndarray): Input frame
            pose_data (dict): Pose detection results
            
        Returns:
            np.ndarray: Frame with pose annotations
        """
        annotated_frame = frame.copy()
        
        if pose_data and pose_data.get('landmarks'):
            # Draw pose landmarks (simplified for performance)
            self.mp_drawing.draw_landmarks(
                annotated_frame,
                pose_data['landmarks'],
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=2),
                connection_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=2)
            )
            
            # Draw center point with different colors for smoothed vs raw
            center = pose_data['center']
            cv2.circle(annotated_frame, center, 8, (0, 255, 0), -1)
            
            # Draw raw center if available (for debugging)
            if 'raw_center' in pose_data:
                raw_center = pose_data['raw_center']
                cv2.circle(annotated_frame, raw_center, 4, (0, 0, 255), 2)
            
            cv2.putText(annotated_frame, f"Center: {center}", 
                       (center[0] + 15, center[1] - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # Draw bounding box
            bbox = pose_data['bbox']
            if bbox[2] > 0 and bbox[3] > 0:
                cv2.rectangle(annotated_frame, 
                            (bbox[0], bbox[1]), 
                            (bbox[0] + bbox[2], bbox[1] + bbox[3]), 
                            (255, 0, 0), 1)
            
            # Draw confidence and performance info
            confidence = pose_data['confidence']
            cv2.putText(annotated_frame, f"Confidence: {confidence:.2f}", 
                       (10, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Show if this is interpolated data
            if pose_data.get('interpolated'):
                cv2.putText(annotated_frame, "INTERPOLATED", 
                           (10, frame.shape[0] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            
            # Show processing FPS
            cv2.putText(annotated_frame, f"Pose FPS: {self.current_fps:.1f}", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return annotated_frame
    
    def get_tracking_offset(self, frame_shape: Tuple[int, int], pose_data: Optional[Dict]) -> Tuple[float, float]:
        """
        Calculate tracking offset from center of frame with dead zone.
        
        Args:
            frame_shape (tuple): (height, width) of the frame
            pose_data (dict or None): Pose detection results
            
        Returns:
            tuple: (x_offset, y_offset) normalized to [-1, 1] range
        """
        if not pose_data or not pose_data.get('center'):
            return 0.0, 0.0
        
        height, width = frame_shape[:2]
        center_x, center_y = pose_data['center']
        
        # Calculate offset from frame center
        frame_center_x = width // 2
        frame_center_y = height // 2
        
        # Normalize to [-1, 1] range
        x_offset = (center_x - frame_center_x) / (width / 2)
        y_offset = (center_y - frame_center_y) / (height / 2)
        
        # Apply dead zone to prevent micro-movements
        dead_zone = 0.15  # 10% dead zone
        if abs(x_offset) < dead_zone:
            x_offset = 0.0
        if abs(y_offset) < dead_zone:
            y_offset = 0.0
        
        # Clamp to [-1, 1] range
        x_offset = max(-1.0, min(1.0, x_offset))
        y_offset = max(-1.0, min(1.0, y_offset))
        
        return x_offset, y_offset
    
    def is_tracking_active(self) -> bool:
        """
        Check if pose tracking is currently active.
        
        Returns:
            bool: True if tracking is active, False otherwise
        """
        return self.tracking_active
    
    def get_performance_stats(self) -> Dict:
        """Get current performance statistics."""
        if not self.processing_times:
            return {'fps': 0, 'avg_time': 0, 'min_time': 0, 'max_time': 0}
        
        return {
            'fps': self.current_fps,
            'avg_time': sum(self.processing_times) / len(self.processing_times),
            'min_time': min(self.processing_times),
            'max_time': max(self.processing_times)
        }
    
    def reset_tracking(self):
        """Reset tracking state."""
        self.tracking_active = False
        self.last_center = None
        self.smoothed_center = None
        self.last_detection_time = 0
        self.last_pose_data = None
        self.frame_skip_count = 0
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'pose'):
            self.pose.close()
        print(f"Pose tracker cleanup complete. Final stats: {self.get_performance_stats()}")