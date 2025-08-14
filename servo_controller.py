import RPi.GPIO as GPIO
import time
import threading

class ServoController:
    """
    Controls pan and tilt servos for camera movement with smooth motion and vibration reduction.
    """
    
    def __init__(self, pan_pin=18, tilt_pin=19, 
                 pan_min=500, pan_max=2500, pan_center=1500,
                 tilt_min=500, tilt_max=2500, tilt_center=1500):
        """
        Initialize servo controller.
        
        Args:
            pan_pin (int): GPIO pin for pan servo
            tilt_pin (int): GPIO pin for tilt servo
            pan_min/max/center (int): PWM pulse widths for pan servo limits and center
            tilt_min/max/center (int): PWM pulse widths for tilt servo limits and center
        """
        self.pan_pin = pan_pin
        self.tilt_pin = tilt_pin
        
        # Servo limits (PWM pulse widths in microseconds)
        self.pan_min = pan_min
        self.pan_max = pan_max
        self.pan_center = pan_center
        self.tilt_min = tilt_min
        self.tilt_max = tilt_max
        self.tilt_center = tilt_center
        
        # Current and target positions
        self.current_pan = pan_center
        self.current_tilt = tilt_center
        self.target_pan = pan_center
        self.target_tilt = tilt_center
        
        # Movement parameters
        self.max_speed = 25  # Reduced for smoother movement
        self.dead_zone = 0.08  # Smaller dead zone for better responsiveness
        self.position_tolerance = 4  # Don't update PWM if change is less than this
        self.smooth_factor = 0.3  # For smooth movement interpolation
        
        # PWM control
        self.pwm_update_rate = 50  # Hz - how often to update PWM
        self.last_pan_pwm = None
        self.last_tilt_pwm = None
        self.pwm_active = False
        self.movement_thread = None
        self.running = True
        
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pan_pin, GPIO.OUT)
        GPIO.setup(self.tilt_pin, GPIO.OUT)
        
        # Setup PWM (50Hz for servos)
        self.pan_pwm = GPIO.PWM(self.pan_pin, 50)
        self.tilt_pwm = GPIO.PWM(self.tilt_pin, 50)
        
        # Start PWM with initial position
        initial_duty = self._pulse_to_duty_cycle(self.current_pan)
        self.pan_pwm.start(initial_duty)
        self.tilt_pwm.start(self._pulse_to_duty_cycle(self.current_tilt))
        self.last_pan_pwm = initial_duty
        self.last_tilt_pwm = self._pulse_to_duty_cycle(self.current_tilt)
        
        # Start smooth movement thread
        self.movement_thread = threading.Thread(target=self._smooth_movement_loop, daemon=True)
        self.movement_thread.start()
        
        print(f"Servo controller initialized - Pan: {self.current_pan}, Tilt: {self.current_tilt}")
        
    def _pulse_to_duty_cycle(self, pulse_width):
        """Convert pulse width (microseconds) to duty cycle percentage."""
        return (pulse_width / 20000.0) * 100.0
    
    def _smooth_movement_loop(self):
        """Background thread for smooth servo movement."""
        while self.running:
            try:
                # Calculate smooth movement towards target
                pan_diff = self.target_pan - self.current_pan
                tilt_diff = self.target_tilt - self.current_tilt
                
                # Apply smooth interpolation
                if abs(pan_diff) > self.position_tolerance:
                    self.current_pan += pan_diff * self.smooth_factor
                else:
                    self.current_pan = self.target_pan
                    
                if abs(tilt_diff) > self.position_tolerance:
                    self.current_tilt += tilt_diff * self.smooth_factor
                else:
                    self.current_tilt = self.target_tilt
                
                # Update PWM only if there's significant change
                self._update_pwm_if_needed()
                
                # Sleep to control update rate
                time.sleep(1.0 / self.pwm_update_rate)
                
            except Exception as e:
                print(f"Error in smooth movement loop: {e}")
                time.sleep(0.1)
    
    def _update_pwm_if_needed(self):
        """Update PWM only if position has changed significantly."""
        pan_duty = self._pulse_to_duty_cycle(self.current_pan)
        tilt_duty = self._pulse_to_duty_cycle(self.current_tilt)
        
        # Update pan PWM if changed significantly
        if (self.last_pan_pwm is None or 
            abs(pan_duty - self.last_pan_pwm) > 0.05):  # ~1 microsecond change
            try:
                self.pan_pwm.ChangeDutyCycle(pan_duty)
                self.last_pan_pwm = pan_duty
            except Exception as e:
                print(f"Error updating pan PWM: {e}")
        
        # Update tilt PWM if changed significantly
        if (self.last_tilt_pwm is None or 
            abs(tilt_duty - self.last_tilt_pwm) > 0.05):
            try:
                self.tilt_pwm.ChangeDutyCycle(tilt_duty)
                self.last_tilt_pwm = tilt_duty
            except Exception as e:
                print(f"Error updating tilt PWM: {e}")
    
    def move_to_position(self, pan_pos, tilt_pos):
        """
        Move servos to specific positions smoothly.
        
        Args:
            pan_pos (int): Pan position (pulse width in microseconds)
            tilt_pos (int): Tilt position (pulse width in microseconds)
        """
        # Clamp to limits
        pan_pos = max(self.pan_min, min(self.pan_max, pan_pos))
        tilt_pos = max(self.tilt_min, min(self.tilt_max, tilt_pos))
        
        # Set target positions (smooth movement thread will handle the rest)
        self.target_pan = pan_pos
        self.target_tilt = tilt_pos
    
    def move_relative(self, pan_offset, tilt_offset):
        """
        Move servos relative to current position.
        
        Args:
            pan_offset (float): Pan offset (-1.0 to 1.0)
            tilt_offset (float): Tilt offset (-1.0 to 1.0)
        """
        # Check dead zone
        if abs(pan_offset) < self.dead_zone:
            pan_offset = 0
        if abs(tilt_offset) < self.dead_zone:
            tilt_offset = 0
        
        # Only move if there's significant offset
        if abs(pan_offset) > 0 or abs(tilt_offset) > 0:
            # Calculate movement amounts (limited by max_speed)
            pan_move = int(pan_offset * self.max_speed)
            tilt_move = int(tilt_offset * self.max_speed)
            
            # Calculate new target positions
            new_pan = self.target_pan + pan_move
            new_tilt = self.target_tilt - tilt_move
            
            # Move to new positions
            self.move_to_position(new_pan, new_tilt)
    
    def center_camera(self):
        """Move camera to center position."""
        print("Centering camera...")
        self.move_to_position(self.pan_center, self.tilt_center)
    
    def search_pan(self, direction=1, step_size=40):
        """
        Perform search panning motion with slower, smoother movement.
        
        Args:
            direction (int): 1 for clockwise, -1 for counter-clockwise
            step_size (int): Size of each step in microseconds (reduced for smoother search)
        """
        new_pan = self.target_pan + (direction * step_size)
        
        # Reverse direction if at limits
        if new_pan >= self.pan_max:
            new_pan = self.pan_max
            direction = -1
        elif new_pan <= self.pan_min:
            new_pan = self.pan_min
            direction = 1
        
        self.move_to_position(new_pan, self.target_tilt)
        return direction
    
    def is_moving(self):
        """Check if servos are currently moving."""
        return (abs(self.target_pan - self.current_pan) > self.position_tolerance or
                abs(self.target_tilt - self.current_tilt) > self.position_tolerance)
    
    def stop_pwm(self):
        """Stop PWM signals to reduce vibration when not moving."""
        if self.pwm_active:
            self.pan_pwm.ChangeDutyCycle(0)
            self.tilt_pwm.ChangeDutyCycle(0)
            self.pwm_active = False
    
    def resume_pwm(self):
        """Resume PWM signals."""
        if not self.pwm_active:
            self._update_pwm_if_needed()
            self.pwm_active = True
    
    def cleanup(self):
        """Clean up GPIO resources."""
        print("Cleaning up servo controller...")
        self.running = False
        
        # Wait for movement thread to finish
        if self.movement_thread and self.movement_thread.is_alive():
            self.movement_thread.join(timeout=1.0)
        
        # Stop PWM
        try:
            self.pan_pwm.stop()
            self.tilt_pwm.stop()
        except:
            pass
        
        # Clean up GPIO
        try:
            GPIO.cleanup([self.pan_pin, self.tilt_pin])
        except:
            pass
        
        print("Servo controller cleanup complete")
