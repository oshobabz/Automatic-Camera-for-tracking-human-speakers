import numpy as np
import cv2
import mediapipe as mp
import time
from picamera2 import Picamera2

class LipMovementDetector:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        
        # Lip landmarks for vertical movement detection
        self.outer_lip_indices = [61, 17]  # Top and bottom center points
        
        # Movement tracking
        self.prev_lip_height = None
        self.prev_timestamp = None
        self.vertical_movement_history = []
        self.history_size = 10
        self.vertical_threshold = 0.01
        
        # Detection statistics
        self.lip_gap_values = []
        self.vertical_movement_values = []
        self.detection_frames = 0
        self.total_frames = 0
        
    def start_camera(self):
        """Start the camera"""
        self.camera = Picamera2()
        self.camera_config = self.camera.create_preview_configuration(
            main={"format": 'XRGB8888', "size": (480, 360)}
        )
        self.camera.configure(self.camera_config)
        self.camera.start()
        time.sleep(0.1)
        
    def stop_camera(self):
        """Stop the camera"""
        if hasattr(self, 'camera'):
            try:
                self.camera.stop()
                self.camera.close()
            except:
                pass
            del self.camera

    
    def capture_frame(self):
        """Capture a frame from PiCamera2"""
        frame = self.camera.capture_array()
        return frame
    
    def _calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def _get_lip_points(self, face_landmarks, image_shape):
        """Extract key lip points"""
        h, w = image_shape[:2]
        points = []
        for idx in self.outer_lip_indices:
            landmark = face_landmarks.landmark[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            points.append((x, y))
        return points
    
    def _detect_vertical_movement(self, current_lip_height):
        """Detect vertical lip movement"""
        if self.prev_lip_height is None or self.prev_timestamp is None:
            self.prev_lip_height = current_lip_height
            self.prev_timestamp = time.time()
            return False, 0
        
        current_time = time.time()
        time_delta = current_time - self.prev_timestamp
        
        # Calculate vertical movement score
        vertical_gap_change = abs(current_lip_height - self.prev_lip_height)
        vertical_movement_score = vertical_gap_change / time_delta if time_delta > 0 else 0
        
        # Update movement history
        self.vertical_movement_history.append(vertical_movement_score)
        if len(self.vertical_movement_history) > self.history_size:
            self.vertical_movement_history.pop(0)
        
        # Calculate average movement
        avg_vertical_movement = sum(self.vertical_movement_history) / len(self.vertical_movement_history)
        
        # Update previous values
        self.prev_lip_height = current_lip_height
        self.prev_timestamp = current_time
        
        return avg_vertical_movement > self.vertical_threshold, vertical_movement_score
    
    def process_frame(self, frame):
        """Process frame and detect lip movement"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        annotated_frame = frame.copy()
        vertical_movement_detected = False
        vertical_movement_score = 0
        metrics = {}
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get lip points
            lip_points = self._get_lip_points(face_landmarks, frame.shape)
            
            if len(lip_points) >= 2:
                # Calculate lip height
                lip_height = self._calculate_distance(lip_points[0], lip_points[1])
                metrics = {"lip_height": lip_height}
                
                # Draw lip points
                for point in lip_points:
                    cv2.circle(annotated_frame, point, 3, (0, 255, 0), -1)
                cv2.line(annotated_frame, lip_points[0], lip_points[1], (0, 255, 255), 2)
                
                # Detect movement
                vertical_movement_detected, vertical_movement_score = self._detect_vertical_movement(lip_height)
                
                # Display info
                status_text = f"Movement: {vertical_movement_detected}, Score: {vertical_movement_score:.4f}"
                cv2.putText(annotated_frame, status_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0) if vertical_movement_detected else (255, 0, 0), 2)
                
                metrics_text = f"Lip Gap: {lip_height:.1f}px"
                cv2.putText(annotated_frame, metrics_text, (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return annotated_frame, vertical_movement_detected, metrics, vertical_movement_score
    
    def analyze_speech(self, duration=8.0, wait_for_face=True):
        """
        Analyze speech for specified duration
        
        Args:
            duration: Analysis duration in seconds
            wait_for_face: If True, timer starts only when face is detected
            
        Returns:
            bool: True if speaking detected
        """
        self.start_camera()
        
        # Reset statistics
        self.lip_gap_values = []
        self.vertical_movement_values = []
        self.detection_frames = 0
        self.total_frames = 0
        
        try:
            print(f"{'Looking for face...' if wait_for_face else f'Analyzing speech for {duration} seconds...'}")
            
            face_detected = False
            timer_start = None if wait_for_face else time.time()
            total_start = time.time()
            
            while True:
                frame = self.capture_frame()
                processed_frame, moving, metrics, movement_score = self.process_frame(frame)
                
                has_face = bool(metrics and "lip_height" in metrics)
                
                # Start timer when face detected (if waiting for face)
                if wait_for_face and has_face and not face_detected:
                    face_detected = True
                    timer_start = time.time()
                    print(f"Face detected! Analyzing speech for {duration} seconds...")
                
                # Process if timer is active
                if timer_start is not None:
                    elapsed = time.time() - timer_start
                    remaining = duration - elapsed
                    
                    # Update statistics
                    self.total_frames += 1
                    if moving:
                        self.detection_frames += 1
                    
                    if metrics and "lip_height" in metrics:
                        self.lip_gap_values.append(metrics["lip_height"])
                        self.vertical_movement_values.append(movement_score)
                    
                    # Display timer
                    cv2.putText(processed_frame, f"Time remaining: {remaining:.1f}s", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    
                    if elapsed >= duration:
                        break
                else:
                    # Waiting for face
                    cv2.putText(processed_frame, "Waiting for face...", 
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Timeout protection
                    if time.time() - total_start > 30:
                        print("Timeout waiting for face")
                        return False
                
                # Display face status
                cv2.putText(processed_frame, f"Face: {'YES' if has_face else 'NO'}", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                           (0, 255, 0) if has_face else (0, 0, 255), 2)
                
                cv2.imshow('Speech Analysis', processed_frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    return False
                
                time.sleep(0.01)
            
            # Calculate results
            if self.lip_gap_values:
                lip_gap_variation = max(self.lip_gap_values) - min(self.lip_gap_values)
                speaking_detected = 6.8 < lip_gap_variation < 27.7
                
                print(f"\nSpeech Analysis Results:")
                print(f"Frames processed: {self.total_frames}")
                print(f"Lip gap variation: {lip_gap_variation:.2f} pixels")
                print(f"Speaking detected: {speaking_detected}")
                
                return speaking_detected
            
            return False
            
        finally:
            self.stop_camera()
            cv2.destroyAllWindows()
