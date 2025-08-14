<!-- README.md (HTML version) -->

<h1 align="center">Automated Camera Tracking System</h1>
<p align="center">
  An AI-powered Raspberry Pi system that detects speech, confirms with lip movement, and tracks a person using pose estimation. 
  Servos move intelligently to reduce jitter and the system returns to idle after 30s of no speech.
</p>

<hr/>

<h2>Features</h2>
<ul>
  <li><strong>Speech Detection:</strong> Silero VAD via PyAudio.</li>
  <li><strong>Lip Movement Check:</strong> Visual confirmation using MediaPipe Face Mesh.</li>
  <li><strong>Pose Tracking:</strong> Tracks the person with MediaPipe Pose landmarks.</li>
  <li><strong>Smart Servo Control:</strong> Pan/tilt updates only when needed to minimize jitter.</li>
  <li><strong>Auto Idle:</strong> After 30 seconds with no speech detected.</li>
</ul>

<h2>Hardware</h2>
<ul>
  <li>Raspberry Pi 4</li>
  <li>IMX219 Camera Module</li>
  <li>Pan–Tilt Servos (e.g., SG90) + 5V servo power</li>
</ul>

<h2>Software Requirements</h2>
<ul>
  <li>Python 3.10+</li>
  <li>OpenCV, MediaPipe, NumPy, ONNXRuntime, PyAudio, RPi.GPIO</li>
</ul>

<pre><code>pip install opencv-python mediapipe numpy onnxruntime pyaudio
</code></pre>

<h2>Connect to Raspberry Pi</h2>
<ol>
  <li>Power the Raspberry Pi and wait for it to boot.</li>
  <li>Connect your laptop to the Pi’s hotspot:
    <ul>
      <li><strong>SSID:</strong> connect</li>
      <li><strong>Password:</strong> 99999888</li>
    </ul>
  </li>
  <li>Open <strong>RealVNC Viewer</strong> to access the Pi desktop (or use a conventional monitor, keyboard, and mouse).</li>
</ol>

<h2>Run the System</h2>
<ol>
  <li>Open the project folder on the Pi:
    <pre><code>cd ~/system/final_year_project</code></pre>
  </li>
  <li>Start the system:
    <pre><code>python3 main.py</code></pre>
  </li>
</ol>

<h2>How It Works</h2>
<ol>
  <li><strong>Initial VAD:</strong> Listens for speech using Silero VAD (PyAudio).</li>
  <li><strong>Lip Tracker (10s):</strong> Confirms visual speech activity.</li>
  <li><strong>Pose Tracking:</strong> Tracks the person via body landmarks.</li>
  <li><strong>Servo Updates:</strong> Moves only when the target is offset &gt; threshold.</li>
  <li><strong>Idle Mode:</strong> If no speech for 30s, system goes idle (stops tracking).</li>
</ol>

<h2>Folder Structure</h2>
<pre><code>final_year_project/
├── main.py              <!-- main controller / state machine -->
├── vad.py               <!-- Silero VAD (PyAudio) -->
├── lip_tracker.py       <!-- lip movement detection -->
├── pose_tracker.py      <!-- pose estimation + center calculation -->
├── servo_control.py     <!-- servo movement logic (jitter-aware) -->
└── README.md
</code></pre>

<h2>Stop / Troubleshooting</h2>
<ul>
  <li>Press <strong>q</strong> in the window to quit safely.</li>
  <li>If video is slow, lower resolution in <code>main.py</code> camera config.</li>
  <li>If servos jitter, ensure stable 5V supply and that PWM is disabled between moves.</li>
</ul>
