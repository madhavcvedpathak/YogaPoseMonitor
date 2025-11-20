YogaPoseMonitor is a real-time yoga posture monitoring tool developed in Python, designed to help users practice yoga more accurately and safely.

Pose Estimation: Uses a lightweight pose-tracking model (e.g., MediaPipe BlazePose) to detect 33 body keypoints.

Pose Analysis: Computes joint angles (such as elbows, shoulders, hips, knees) to assess correctness of yoga asanas.

Feedback Generation: Provides real-time (or post-session) feedback by comparing the user’s posture with ideal pose angles, highlights deviations, and logs reports.

User Interface: A simple HTML front-end (index.html) to display live video, feedback, and summary.

Reporting: Generates session-based reports (stored in a reports/ folder) and snapshots (in pictures/) for further review.

Dependencies: Managed via requirements.txt, which includes required Python libraries (e.g., OpenCV, MediaPipe, NumPy, etc.).

Use Case: Built for AyurSutra, to support self-learning yoga practitioners by giving automated posture monitoring and corrective feedback.
