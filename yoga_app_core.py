import os
import cv2
import json
import time
import threading
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, request, send_file, jsonify
from reportlab.lib.pagesizes import letter, A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import mediapipe as mp
import jwt
from functools import wraps

# --- FIREBASE IMPORTS ---
import firebase_admin
from firebase_admin import credentials, firestore, auth
# --- END FIREBASE IMPORTS ---

# Initialize Flask app
app = Flask(__name__)

# Directories
REPORTS_DIR = "reports"
LIVE_JSON_DIR = "live_json"
# os.makedirs is safe to call even if directories exist
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(LIVE_JSON_DIR, exist_ok=True)

# --- FIREBASE SETUP ---
# ⚠️ SECURITY WARNING: Netlify/Serverless environment MUST use the environment variable.
# On local testing, this will try to load the 'firebase_service_account.json' file.
try:
    CREDENTIALS_PATH = os.environ.get('FIREBASE_CREDENTIALS', 'firebase_service_account.json')
    
    if os.path.exists(CREDENTIALS_PATH) or os.environ.get('FIREBASE_CREDENTIALS'):
        # If running on Netlify/Vercel, CREDENTIALS_PATH will contain the raw JSON string.
        if CREDENTIALS_PATH.startswith('{'):
            # Load credentials from the JSON string provided by the environment variable
            import json as json_lib
            cred = credentials.Certificate(json_lib.loads(CREDENTIALS_PATH))
        else:
            # Load credentials from the file path (Local development)
            cred = credentials.Certificate(CREDENTIALS_PATH)
            
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("✅ Firebase initialized successfully.")
    else:
        # This will only happen if testing locally without the file, or if environment variable is missing
        print("⚠️ Firebase credentials not found. DB functions disabled.")
        db = None

except Exception as e:
    print(f"❌ Error initializing Firebase: {e}")
    db = None

# Global state
camera_active = False
session_active = False
pose_log = []
current_user_uid = None
current_user_display_name = "User" 
video_capture = None
pose_detector = None
last_report_path = None

# MediaPipe Pose setup
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# --- JWT DECORATOR ---
def jwt_required(f):
    """Decorator to protect routes, requiring a valid Firebase ID Token (JWT)."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global current_user_uid, current_user_display_name
        auth_header = request.headers.get('Authorization')
        if not auth_header or not auth_header.startswith('Bearer '):
            return jsonify({"status": "error", "message": "Authorization token missing or invalid"}), 401
        
        id_token = auth_header.split(' ')[1]
        
        try:
            decoded_token = auth.verify_id_token(id_token)
            current_user_uid = decoded_token['uid']
            # Get user name from the JWT payload sent by the frontend's login process
            user_name_from_request = request.get_json().get('user_name', f"User_{current_user_uid[-4:]}")
            current_user_display_name = user_name_from_request
            
        except Exception as e:
            print(f"JWT Verification Failed: {e}")
            return jsonify({"status": "error", "message": "Invalid or expired authentication token"}), 401
            
        return f(*args, **kwargs)
    return decorated_function
# --- END JWT DECORATOR ---


def calculate_angle(a, b, c):
    """Calculate angle in degrees between three 2D points (a-b-c)."""
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    
    if angle > 180.0:
        angle = 360 - angle
        
    return angle

def analyze_yoga_pose(landmarks):
    """Analyze pose and identify yoga posture (12 Poses of Surya Namaskar)"""
    try:
        # --- Extract key points ---
        get_point = lambda p: [landmarks[p.value].x, landmarks[p.value].y]
        
        left_shoulder, left_elbow, left_wrist = get_point(mp_pose.PoseLandmark.LEFT_SHOULDER), get_point(mp_pose.PoseLandmark.LEFT_ELBOW), get_point(mp_pose.PoseLandmark.LEFT_WRIST)
        left_hip, left_knee, left_ankle = get_point(mp_pose.PoseLandmark.LEFT_HIP), get_point(mp_pose.PoseLandmark.LEFT_KNEE), get_point(mp_pose.PoseLandmark.LEFT_ANKLE)
        
        right_shoulder, right_elbow, right_wrist = get_point(mp_pose.PoseLandmark.RIGHT_SHOULDER), get_point(mp_pose.PoseLandmark.RIGHT_ELBOW), get_point(mp_pose.PoseLandmark.RIGHT_WRIST)
        right_hip, right_knee, right_ankle = get_point(mp_pose.PoseLandmark.RIGHT_HIP), get_point(mp_pose.PoseLandmark.RIGHT_KNEE), get_point(mp_pose.PoseLandmark.RIGHT_ANKLE)
        
        # --- Calculate angles ---
        left_elbow_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
        right_elbow_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
        left_knee_angle = calculate_angle(left_hip, left_knee, left_ankle)
        right_knee_angle = calculate_angle(right_hip, right_knee, right_ankle)
        left_hip_angle = calculate_angle(left_shoulder, left_hip, left_knee)
        right_hip_angle = calculate_angle(right_shoulder, right_hip, right_knee)
        left_shldr_angle = calculate_angle(left_elbow, left_shoulder, left_hip)
        right_shldr_angle = calculate_angle(right_elbow, right_shoulder, right_hip)

        # Helper flags
        arms_straight = (left_elbow_angle > 160 and right_elbow_angle > 160)
        legs_straight = (left_knee_angle > 160 and right_knee_angle > 160)
        
        # --- Pose Classification Logic (Synchronized with JS Keys: Pranamasana, Dandasana) ---
        pose_name = "Unknown"
        confidence = 0.50
        
        # 1. & 12. Pranamasana (Prayer/Mountain Pose)
        if legs_straight and left_hip_angle > 160 and right_hip_angle > 160 and left_shldr_angle > 160 and right_shldr_angle > 160:
            pose_name = "Pranamasana" 
            confidence = 0.95
        
        # 2. & 11. Hasta Uttanasana (Raised Arms Pose)
        elif legs_straight and left_hip_angle > 160 and right_hip_angle > 160 and left_shldr_angle < 40 and right_shldr_angle < 40 and arms_straight:
            pose_name = "Hasta Uttanasana"
            confidence = 0.90
        
        # 3. & 10. Uttanasana (Standing Forward Bend)
        elif legs_straight and left_hip_angle < 90 and right_hip_angle < 90:
            pose_name = "Uttanasana"
            confidence = 0.85

        # 4. & 9. Ashwa Sanchalanasana (Low Lunge / Equestrian Pose)
        elif (left_knee_angle < 100 and right_knee_angle > 160 and left_hip_angle < 100 and right_hip_angle > 140) or \
             (right_knee_angle < 100 and left_knee_angle > 160 and right_hip_angle < 100 and left_hip_angle > 140):
            pose_name = "Ashwa Sanchalanasana"
            confidence = 0.88
            
        # 5. Dandasana (Plank Pose - Using the JS key Dandasana for consistency, even if detection is Phalakasana)
        elif legs_straight and abs(left_hip_angle - 180) < 20 and abs(right_hip_angle - 180) < 20 and arms_straight and abs(left_shldr_angle - 180) < 20 and abs(right_shldr_angle - 180) < 20:
            pose_name = "Dandasana" 
            confidence = 0.80

        # 6. Ashtanga Namaskara (Knees-Chest-Chin)
        elif left_hip_angle > 100 and right_hip_angle > 100 and left_knee_angle < 100 and right_knee_angle < 100:
            pose_name = "Ashtanga Namaskara"
            confidence = 0.70

        # 7. Bhujangasana (Cobra Pose)
        elif left_hip_angle > 160 and right_hip_angle > 160 and left_knee_angle > 160 and right_knee_angle > 160 and abs(left_shldr_angle - 180) < 20 and abs(right_shldr_angle - 180) < 20 and left_elbow_angle < 160 and right_elbow_angle < 160:
            pose_name = "Bhujangasana"
            confidence = 0.80
            
        # 8. Adho Mukha Svanasana (Downward-Facing Dog)
        elif left_hip_angle < 110 and right_hip_angle < 110 and legs_straight and arms_straight:
            pose_name = "Adho Mukha Svanasana"
            confidence = 0.92

        # --- Note: The 'Other Poses' detection block is kept for demonstration/robustness ---
        elif (left_knee_angle < 100 and right_knee_angle > 160) or (right_knee_angle < 100 and left_knee_angle > 160):
            if left_elbow_angle > 160 and right_elbow_angle > 160:
                pose_name = "Tree Pose (Vrksasana)"
                confidence = max(confidence, 0.85)
        # ------------------------------------------------------------------------------------
        
        return {
            'pose': pose_name,
            'confidence': round(confidence, 2),
            'left_elbow': round(left_elbow_angle, 2),
            'right_elbow': round(right_elbow_angle, 2),
            'left_knee': round(left_knee_angle, 2),
            'right_knee': round(right_knee_angle, 2),
            'left_hip': round(left_hip_angle, 2),
            'right_hip': round(right_hip_angle, 2),
            'timestamp': time.time()
        }
    except Exception as e:
        # print(f"Error analyzing pose: {e}") 
        return None

def generate_frames():
    """Generate video frames with pose detection"""
    # NOTE: This route may be unstable/slow on Netlify/Lambda due to video stream limitations.
    global camera_active, session_active, pose_log, video_capture, pose_detector
    
    video_capture = cv2.VideoCapture(0)
    pose_detector = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    
    frame_count = 0
    
    while camera_active:
        success, frame = video_capture.read()
        if not success: break
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        
        results = pose_detector.process(image)
        
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
            )
            
            if session_active:
                frame_count += 1
                if frame_count % 15 == 0:
                    pose_data = analyze_yoga_pose(results.pose_landmarks.landmark)
                    if pose_data:
                        pose_log.append(pose_data)
        
            if session_active and pose_log:
                current_pose = pose_log[-1]['pose']
                confidence = pose_log[-1]['confidence']
                cv2.putText(image, f"Pose: {current_pose}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image, f"Confidence: {confidence:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        ret, buffer = cv2.imencode('.jpg', image)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    if video_capture: video_capture.release()
    if pose_detector: pose_detector.close()

def generate_pdf_report(user_name, end_time):
    """Generate professional AyurSutra PDF report with dynamic recommendations."""
    global last_report_path
    
    timestamp = end_time.strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(REPORTS_DIR, f"{user_name}_AyurSutra_Report_{timestamp}.pdf")
    
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    AS_BLUE = colors.HexColor('#2E86AB')
    AS_PINK = colors.HexColor('#A23B72')
    
    title_style = ParagraphStyle('AyurSutraTitle', parent=styles['Heading1'], fontSize=28, textColor=AS_BLUE, spaceAfter=15, alignment=TA_CENTER)
    subtitle_style = ParagraphStyle('Subtitle', parent=styles['Heading2'], fontSize=18, textColor=AS_PINK, spaceAfter=25, alignment=TA_CENTER)
    heading_style = ParagraphStyle('ReportHeading', parent=styles['Heading2'], fontSize=16, textColor=AS_PINK, spaceAfter=10, spaceBefore=15)
    
    story.append(Paragraph("A Y U R S U T R A", title_style))
    story.append(Paragraph("Yoga Pose Monitoring & Diagnostic Report", subtitle_style))
    
    story.append(Paragraph("I. Session Metrics", heading_style))
    
    if pose_log:
        df = pd.DataFrame(pose_log)
        
        if len(pose_log) > 1:
            duration_seconds = pose_log[-1]['timestamp'] - pose_log[0]['timestamp']
            duration_min = duration_seconds / 60
        else:
            duration_min = 0.0
        
        avg_conf = df['confidence'].mean()
        
        summary_data = [
            ['Participant ID', user_name, 'Date', end_time.strftime('%B %d, %Y')],
            ['Total Duration', f"{duration_min:.2f} Minutes", 'Time', end_time.strftime('%I:%M %p')],
            ['Analyzed Frames', str(len(df)), 'Avg Confidence', f"{avg_conf:.2%}"]
        ]
        
        summary_table = Table(summary_data, colWidths=[2*inch, 1.5*inch, 1.5*inch, 1.5*inch])
        summary_table.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#F5F5F5')), ('BACKGROUND', (2, 0), (2, -1), colors.HexColor('#F5F5F5')), ('GRID', (0, 0), (-1, -1), 0.5, colors.lightgrey), ('VALIGN', (0, 0), (-1, -1), 'MIDDLE')]))
        story.append(summary_table)
        
        story.append(Paragraph("II. Pose Duration Analysis", heading_style))
        pose_counts = df['pose'].value_counts()
        
        data = [['Pose Name', 'Frames Detected', 'Total Time (Seconds)']]
        for pose, count in pose_counts.items():
            duration_sec = count * 0.5
            data.append([pose, str(count), f"{duration_sec:.1f} s"])
        
        table = Table(data, colWidths=[3.5*inch, 1.5*inch, 1.5*inch])
        table.setStyle(TableStyle([('BACKGROUND', (0, 0), (-1, 0), AS_BLUE), ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke), ('ALIGN', (0, 0), (-1, -1), 'CENTER'), ('ALIGN', (0, 1), (0, -1), 'LEFT'), ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'), ('BOTTOMPADDING', (0, 0), (-1, 0), 10), ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor('#EBF4FA')), ('GRID', (0, 0), (-1, -1), 0.5, colors.black)]))
        story.append(table)
        
        story.append(Paragraph("III. Personalized Improvement Plan", heading_style))
        
        recommendations = []
        if avg_conf < 0.80:
            recommendations.append("• Aim for brighter lighting and confirm your entire body is visible to improve tracking accuracy.")
        elif avg_conf < 0.90:
            recommendations.append("• Focus on fine-tuning your body's lines and alignment to achieve more precise posture confirmation.")
        
        if not df[df['pose'] != 'Unknown'].empty:
            pose_durations = df[df['pose'] != 'Unknown'].groupby('pose').size() * 0.5
            longest_pose = pose_durations.idxmax()
            longest_duration = pose_durations.max()
            recommendations.append(f"• **Successfully held {longest_pose}** for {longest_duration:.1f} seconds. Maintain this dedication to duration.")
            brief_poses = pose_durations[(pose_durations > 0) & (pose_durations < 5)]
            if not brief_poses.empty:
                brief_pose_name = brief_poses.idxmin()
                recommendations.append(f"• **{brief_pose_name}** was held briefly ({brief_poses.min():.1f} seconds). Practice holding fundamental poses for **15 to 30 seconds** to maximize physical benefit.")

        avg_left_hip = df['left_hip'].mean()
        avg_right_hip = df['right_hip'].mean()
        hip_diff = abs(avg_left_hip - avg_right_hip)
        if hip_diff > 10:
            recommendations.append(f"• Hips Balance Check: A notable difference of {hip_diff:.1f}° was observed in hip alignment. Work on keeping your hips level and square in standing positions.")
        
        recommendations.append("• Progression Goal: Concentrate on gaining more flexibility or depth in the postures where you spent the least amount of time.")

        if recommendations:
            list_data = [[Paragraph(rec, styles['Normal'])] for rec in recommendations]
            list_table = Table(list_data, colWidths=[6.5*inch])
            list_table.setStyle(TableStyle([('LEFTPADDING', (0, 0), (-1, -1), 0), ('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEABOVE', (0, 0), (-1, 0), 1, colors.lightgrey)]))
            story.append(list_table)
            
    else:
        story.append(Paragraph("⚠️ No pose data recorded during this session.", styles['Normal']))
    
    doc.build(story)
    print(f"✅ Report generated: {report_path}")
    last_report_path = report_path
    return report_path

# --- FIREBASE STORAGE FUNCTION ---
def save_session_to_firestore(uid, session_data):
    """Saves the session data to the Firestore database."""
    if db is None: return False
    try:
        session_ref = db.collection('users').document(uid).collection('yoga_sessions').document()
        session_ref.set(session_data)
        return True
    except Exception as e:
        print(f"❌ Failed to write to Firestore: {e}")
        return False
# --- END FIREBASE STORAGE FUNCTION ---

# Flask Routes

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Video streaming route"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/start_camera', methods=['POST'])
def start_camera():
    """Start camera (no auth required for camera stream)"""
    global camera_active
    if not camera_active:
        camera_active = True
        return jsonify({'status': 'success', 'message': 'Camera started'})
    return jsonify({'status': 'info', 'message': 'Camera already active'})

@app.route('/stop_camera', methods=['POST'])
def stop_camera():
    """Stop camera (no auth required)"""
    global camera_active, session_active
    camera_active = False
    session_active = False
    time.sleep(1)
    return jsonify({'status': 'success', 'message': 'Camera stopped'})

@app.route('/start_session', methods=['POST'])
@jwt_required 
def start_session():
    """Start yoga session (protected route)"""
    global session_active, pose_log
    
    if not camera_active:
        return jsonify({'status': 'error', 'message': 'Please start camera first'}), 400
    
    session_active = True
    pose_log = []
    
    for f in os.listdir(LIVE_JSON_DIR):
        if f.endswith('.json'): os.remove(os.path.join(LIVE_JSON_DIR, f))
    
    return jsonify({'status': 'success', 'message': f'Session started for {current_user_display_name}'})


@app.route('/end_session', methods=['POST'])
@jwt_required 
def end_session():
    """End session, calculate points (capped at 99), generate report, and store in Firestore (protected route)"""
    global session_active, current_user_uid, current_user_display_name
    
    if not session_active or not current_user_uid:
        return jsonify({'status': 'error', 'message': 'No active session or user logged in.'}), 400
    
    session_active = False
    time.sleep(2)
    session_end_time = datetime.now()
    
    points_awarded = 0
    limit_message = None
    duration_seconds = 0
    avg_conf = 0
    df = pd.DataFrame(pose_log)
    
    if pose_log:
        avg_conf = df['confidence'].mean()
        duration_seconds = pose_log[-1]['timestamp'] - pose_log[0]['timestamp'] if len(pose_log) > 1 else 0
        
        duration_min = duration_seconds / 60
        base_factor = 10 
        points_awarded = int(avg_conf * duration_min * base_factor)
        
        if points_awarded >= 100:
            points_awarded = 99
            limit_message = "Points capped at 99 to comply with limit."
    
    firestore_data = {
        'uid': current_user_uid,
        'display_name': current_user_display_name,
        'points_awarded': points_awarded,
        'session_end_time': session_end_time,
        'duration_seconds': duration_seconds,
        'average_confidence': avg_conf,
        'report_summary': df['pose'].value_counts().to_dict() if pose_log else {},
        'report_generated': True
    }
    
    storage_success = save_session_to_firestore(current_user_uid, firestore_data)
    report_path = generate_pdf_report(current_user_display_name, session_end_time)
    
    return jsonify({
        'status': 'success',
        'message': f'Session ended, {points_awarded} points awarded.',
        'report_path': report_path,
        'points_awarded': points_awarded,
        'storage_status': 'success' if storage_success else 'failure',
        'limit_message': limit_message
    })

@app.route('/download_report')
def download_report():
    """Download latest report"""
    global last_report_path
    
    if last_report_path and os.path.exists(last_report_path):
        return send_file(last_report_path, as_attachment=True)
    
    reports = sorted([f for f in os.listdir(REPORTS_DIR) if f.endswith('.pdf')])
    if reports:
        latest_report = os.path.join(REPORTS_DIR, reports[-1])
        last_report_path = latest_report
        return send_file(latest_report, as_attachment=True)
    
    return jsonify({'status': 'error', 'message': 'No reports available'})

@app.route('/session_status')
def session_status():
    """Get current session status"""
    return jsonify({
        'camera_active': camera_active,
        'session_active': session_active,
        'poses_logged': len(pose_log),
        'current_user_uid': current_user_uid,
        'current_user_display_name': current_user_display_name
    })

# --- NOTE: Removed if __name__ == '__main__': block for serverless compatibility ---