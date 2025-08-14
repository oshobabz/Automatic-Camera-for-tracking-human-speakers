import cv2
import time
import numpy as np
import threading
from typing import Optional, Tuple
from picamera2 import Picamera2
from pose_tracker import PoseTracker
from servo_controller import ServoController
from lip_tracker import LipMovementDetector
from vad import VAD


class CameraTrackingSystem:
    """
    Main camera tracking system that combines pose estimation with servo control.
    Automatically tracks people when detected with smooth, vibration-free movement.
    Includes background voice activity detection.
    """
    
    def __init__(self, frame_width=480, frame_height=360):
        """
        Initialize the tracking system.
        
        Args:
            frame_width (int): Camera frame width
            frame_height (int): Camera frame height
        """
        # Initialize camera
        self.camera = Picamera2()
        config = self.camera.create_preview_configuration(
            main={"size": (frame_width, frame_height), "format": "RGB888"}
        )
        self.camera.configure(config)
        
        # Initialize pose tracker
        self.pose_tracker = PoseTracker(
            min_detection_confidence=0.6,
            min_tracking_confidence=0.5,
            model_complexity=0  # Fastest model for Pi
        )
        
        # Initialize servo controller
        self.servo_controller = ServoController(17, 27)
        
        # Initialize VAD
        self.vad = VAD()
        
        # Frame settings
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # Tracking state
        self.is_tracking = False
        self.search_direction = 1
        self.last_detection_time = 0
        self.person_lost_timeout = 2.0  # 2 seconds before starting search
        self.search_active = False
        self.frames_without_detection = 0
        self.max_frames_without_detection = 15  # ~0.75 seconds at 20fps
        
        # Search timing control
        self.last_search_move_time = 0
        self.search_move_interval = 0.5  # Move search position every 0.5 seconds
        self.search_step_size = 30  # Smaller steps for smoother search
        
        # Movement smoothing
        self.last_move_time = 0
        self.min_move_interval = 0.05  # Minimum time between servo movements (50ms)
        
        # Display settings
        self.show_display = True
        
        # Performance monitoring
        self.frame_times = []
        
        # VAD background monitoring
        self.vad_thread = None
        self.vad_running = False
        self.vad_interval = 20  # Check every 30 seconds
        self.vad_duration = 10  # Listen for 5 seconds
        self.last_vad_check = 0
        self.speech_detected = True  # Start assuming speech is present
        self.no_speech_start_time = None
        self.no_speech_timeout = 5  # 3 seconds after "not speaking" message
        self.shutdown_requested = False
        self.vad_lock = threading.Lock()
        self.vad_status = "Listening..."  # Status message for display
        
    def start_search_mode(self):
        """Start search mode when person is lost."""
        if not self.search_active:
            self.search_active = True
            self.is_tracking = False
            self.last_search_move_time = time.time()
            print("Person lost - starting search mode...")
    
    def stop_search_mode(self):
        """Stop search mode when person is found."""
        if self.search_active:
            self.search_active = False
            print("Person found - stopping search mode...")
    
    def should_move_servos(self):
        """Check if enough time has passed to move servos (prevents vibration)."""
        current_time = time.time()
        if current_time - self.last_move_time >= self.min_move_interval:
            self.last_move_time = current_time
            return True
        return False
    
    def vad_background_check(self):
        """Background thread function to check for voice activity."""
        while self.vad_running:
            try:
                current_time = time.time()
                
                # Check if it's time for VAD check
                if current_time - self.last_vad_check >= self.vad_interval:
                    print(f"Starting VAD check ({self.vad_duration} seconds)...")
                    
                    with self.vad_lock:
                        self.vad_status = f"Checking speech ({self.vad_duration}s)..."
                    
                    # Perform VAD check
                    speech_detected = self.vad.detect_speech(duration=self.vad_duration)
                    
                    with self.vad_lock:
                        self.speech_detected = speech_detected
                        self.last_vad_check = current_time
                        
                        if speech_detected:
                            self.vad_status = "Speech detected"
                            self.no_speech_start_time = None
                            print("Background VAD: Speech detected")
                        else:
                            self.vad_status = "Speaker not speaking"
                            self.no_speech_start_time = current_time
                            print("Background VAD: No speech detected")
                
                # Check if we should shutdown due to no speech
                if self.no_speech_start_time is not None:
                    time_since_no_speech = current_time - self.no_speech_start_time
                    if time_since_no_speech >= self.no_speech_timeout:
                        print(f"No speech detected for {self.no_speech_timeout} seconds. Requesting shutdown...")
                        self.shutdown_requested = True
                        break
                
                # Small delay to prevent excessive CPU usage
                time.sleep(0.1)
                
            except Exception as e:
                print(f"VAD background check error: {e}")
                time.sleep(1)
    
    def start_vad_monitoring(self):
        """Start background VAD monitoring."""
        self.vad_running = True
        self.last_vad_check = time.time()
        self.vad_thread = threading.Thread(target=self.vad_background_check, daemon=True)
        self.vad_thread.start()
        print("VAD background monitoring started")
    
    def stop_vad_monitoring(self):
        """Stop background VAD monitoring."""
        self.vad_running = False
        if self.vad_thread and self.vad_thread.is_alive():
            self.vad_thread.join(timeout=1)
        print("VAD background monitoring stopped")
    
    def process_frame(self):
        """
        Process a single frame for pose detection and tracking.
        
        Returns:
            tuple: (success, frame) where success indicates if processing was successful
        """
        frame_start_time = time.time()
        
        # Capture frame
        frame = self.camera.capture_array()
        
        if frame is None:
            return False, None
        
        # Detect pose (pose_tracker expects BGR input)
        pose_data = self.pose_tracker.detect_pose(frame)
        
        current_time = time.time()
        
        # Handle tracking logic
        if pose_data and pose_data['confidence'] > 0.5:  # Person detected with good confidence
            # Stop search mode if active
            if self.search_active:
                self.stop_search_mode()
            
            # Reset frames without detection counter
            self.frames_without_detection = 0
            
            # Calculate tracking offsets
            x_offset, y_offset = self.pose_tracker.get_tracking_offset(
                (self.frame_height, self.frame_width), pose_data
            )
            
            # Move servos to track the pose only if enough time has passed
            if self.should_move_servos() and (abs(x_offset) > 0.1 or abs(y_offset) > 0.1):
                # Reduced sensitivity for smoother tracking
                self.servo_controller.move_relative(-x_offset * 0.3, -y_offset * 0.25)
            
            self.is_tracking = True
            self.last_detection_time = current_time
            
            print(f"Tracking: Center={pose_data['center']}, "
                  f"Offset=({x_offset:.2f}, {y_offset:.2f}), "
                  f"Confidence={pose_data['confidence']:.2f}")
            
        else:  # No person detected
            self.frames_without_detection += 1
            
            # Check if we should start searching
            if (self.frames_without_detection > self.max_frames_without_detection and 
                not self.search_active and 
                self.is_tracking):
                self.start_search_mode()
            
            # Perform search if active (with slower timing)
            if self.search_active:
                if current_time - self.last_search_move_time >= self.search_move_interval:
                    self.search_direction = self.servo_controller.search_pan(
                        self.search_direction, step_size=self.search_step_size
                    )
                    self.last_search_move_time = current_time
                    print(f"Search move: direction={self.search_direction}, "
                          f"pan_pos={self.servo_controller.target_pan}")
                
                self.is_tracking = False
        
        # Calculate frame processing time
        frame_time = time.time() - frame_start_time
        self.frame_times.append(frame_time)
        if len(self.frame_times) > 30:
            self.frame_times.pop(0)
        
        # Draw pose annotations if display is enabled
        if self.show_display:
            annotated_frame = self.pose_tracker.draw_pose(frame, pose_data)
            
            # Add tracking status
            if self.is_tracking:
                status_color = (0, 255, 0)
                status_text = "TRACKING"
            elif self.search_active:
                status_color = (0, 165, 255)  # Orange
                status_text = "SEARCHING"
            else:
                status_color = (128, 128, 128)  # Gray
                status_text = "IDLE"
            
            cv2.putText(annotated_frame, status_text, (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
            
           
            # Add servo positions and movement status
            moving_status = "MOVING" if self.servo_controller.is_moving() else "STABLE"
            cv2.putText(annotated_frame, 
                       f"Pan: {self.servo_controller.current_pan:.0f} Tilt: {self.servo_controller.current_tilt:.0f} ({moving_status})", 
                       (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # Add performance info
            if self.frame_times:
                avg_frame_time = sum(self.frame_times) / len(self.frame_times)
                fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                cv2.putText(annotated_frame, f"FPS: {fps:.1f}", 
                           (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            return True, annotated_frame
        else:
            return True, frame
    
    def run(self):
        """
        Main tracking loop with improved timing and smoother operation.
        """
        print("Starting Camera Tracking System...")
        print("The system will automatically track people when detected.")
        print("VAD monitoring: Every 20s for 5s duration")
        print("Press 'q' to quit, 's' to toggle search, 'c' to center camera")
        
        try:
            # Start camera
            self.camera.start()
            print("Camera started")
            
            # Center camera at start
            self.servo_controller.center_camera()
            time.sleep(2)  # Allow time for camera and servos to initialize
            
            # Start VAD monitoring
            self.start_vad_monitoring()
            
            print("System ready - looking for people to track...")
            
            frame_count = 0
            
            while True:
                loop_start_time = time.time()
                
                # Check if shutdown was requested by VAD
                if self.shutdown_requested:
                    print("Shutdown requested by VAD monitoring - no speech detected")
                    break
                
                # Process frame
                success, frame = self.process_frame()
                
                if not success:
                    print("Failed to process frame, retrying...")
                    time.sleep(0.1)
                    continue
                
                frame_count += 1
                
                # Display frame if enabled
                if self.show_display and frame is not None:
                    cv2.imshow('Camera Tracking', frame)
                    
                    # Handle keyboard input
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # Toggle search mode for testing
                        if self.search_active:
                            self.search_active = False
                            print("Search mode disabled")
                        else:
                            self.start_search_mode()
                    elif key == ord('c'):
                        # Center camera
                        self.servo_controller.center_camera()
                        self.search_active = False
                        self.is_tracking = False
                        print("Camera centered")
                    elif key == ord('r'):
                        # Reset tracking
                        self.pose_tracker.reset_tracking()
                        self.frames_without_detection = 0
                        print("Tracking reset")
                    elif key == ord('p'):
                        # Pause/resume (stop PWM to reduce vibration)
                        if hasattr(self, '_paused') and self._paused:
                            self.servo_controller.resume_pwm()
                            self._paused = False
                            print("Resumed")
                        else:
                            self.servo_controller.stop_pwm()
                            self._paused = True
                            print("Paused (PWM stopped)")
                    elif key == ord('v'):
                        # Force VAD check for testing
                        with self.vad_lock:
                            self.last_vad_check = 0  # Force immediate check
                        print("Forcing VAD check...")
                
                # Control loop timing - aim for ~20 FPS to reduce servo stress
                loop_time = time.time() - loop_start_time
                target_loop_time = 1.0 / 20.0  # 20 FPS
                if loop_time < target_loop_time:
                    time.sleep(target_loop_time - loop_time)
                
        except KeyboardInterrupt:
            print("\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up all resources."""
        print("Cleaning up resources...")
        
        # Stop VAD monitoring
        self.stop_vad_monitoring()
        
        # Stop camera
        try:
            self.camera.stop()
            self.camera.close()
            print("Camera stopped")
        except:
            pass
        
        # Center camera before shutdown
        try:
            self.servo_controller.center_camera()
            time.sleep(1.0)  # Give more time for centering
        except:
            pass
        
        # Clean up components
        self.pose_tracker.cleanup()
        self.servo_controller.cleanup()
        self.vad.close()
        
        # Close OpenCV windows
        cv2.destroyAllWindows()
        
        print("Cleanup complete")


def main():
    """Main function to detect speech with face detection timer and then run tracking system."""
    print("Starting face detection and speech analysis...")
    vad = VAD()

    try:
        while True:
            print("Listening for speech for 2 seconds...")
            if vad.detect_speech(duration=2):
                print("Speech detected! Starting camera tracking...")
                vad.close()
                break
            else:
                print("No speech detected. Waiting 2 seconds...")
                time.sleep(2)
                
        # Use streamlined lip tracker to detect speaking for 8 seconds after face is detected
        detector = LipMovementDetector()
        
        if detector.analyze_speech(duration=8.0, wait_for_face=True):
            print("Speech detected! Starting camera tracking...")
            
            # Start tracking system after speech is detected
            tracking_system = CameraTrackingSystem(
                frame_width=480,
                frame_height=360
            )
            tracking_system.run()
        else:
            print("Speaker detected not speaking. Exiting...")

    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        vad.close()


if __name__ == "__main__":
    main()
