import os
import time
import json
import numpy as np
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, request, send_file, jsonify
from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER
import jwt
from functools import wraps
import requests # Need this for basic auth check

# --- SUPABASE IMPORTS ---
from supabase import create_client, Client # New Supabase Client
# --- END SUPABASE IMPORTS ---

# Initialize Flask app
app = Flask(__name__)

# Directories (for report storage)
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs("live_json", exist_ok=True)

# --- SUPABASE SETUP ---
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://ndbugxqiarkwjybgfglu.supabase.co")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5kYnVneHFpYXJrd2p5YmdmZ2x1Iiwicm9sZSI6ImFub24iLCJpYXQiOjE2Nzg0NjUwODcsImV4cCI6MTk5NDA0MTA4N30.C6d...dI") # Replace with your actual anon key
try:
    supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
    print("✅ Supabase initialized successfully.")
    db = supabase
except Exception as e:
    print(f"❌ Error initializing Supabase: {e}")
    db = None

# Global state
session_active = False
pose_log = []
current_user_uid = "default-user" # Changed to default as we rely on API Key
current_user_display_name = "User"
last_report_path = None
# --- NEW: Supabase requires a user table to exist to store the data ---
SUPABASE_TABLE_NAME = "yoga_sessions"


# --- API KEY DECORATOR (REPLACING JWT) ---
def api_key_required(f):
    """Decorator to protect routes using the Supabase Anon/Public Key as a simple check."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        global current_user_uid, current_user_display_name
        
        # We check for the API key in the Authorization header (standard practice)
        auth_header = request.headers.get('Authorization')
        
        # Simple check: Ensure the header is present and the key matches the Anon key
        if not auth_header or auth_header != f"Bearer {SUPABASE_KEY}":
            return jsonify({"status": "error", "message": "Authorization token missing or invalid"}), 401
        
        # NOTE: For real user management, you must validate a session JWT here.
        # Since we are using the public Anon Key, we will default the user:
        current_user_uid = "anonymous_supabase_user" 
        current_user_display_name = request.get_json().get('user_name', "Supabase User")
            
        return f(*args, **kwargs)
    return decorated_function
# --- END API KEY DECORATOR ---


# --- PDF GENERATION (UNCHANGED) ---
# ... (PDF generation function remains the same as in your original code) ...
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
        
        avg_conf = df['confidence'].mean() if not df.empty else 0.0
        
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
        for pose_name, count in pose_counts.items():
            duration_sec = count * 0.5 # Assuming 1 log entry every ~0.5 seconds from JS
            data.append([pose_name, str(count), f"{duration_sec:.1f} s"])
        
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
# --- END PDF GENERATION ---


# --- SUPABASE STORAGE FUNCTION ---
def save_session_to_supabase(session_data):
    """Saves the session data to the Supabase database."""
    if db is None: return False
    try:
        # Supabase client insert method
        data, count = db.table(SUPABASE_TABLE_NAME).insert(session_data).execute()
        return True
    except Exception as e:
        print(f"❌ Failed to write to Supabase: {e}")
        return False
# --- END SUPABASE STORAGE FUNCTION ---

# Flask Routes

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

# --- NEW ROUTE: Receives real-time pose data from the Frontend (JS) ---
@app.route('/log_pose', methods=['POST'])
@api_key_required 
def log_pose():
    """Receives pose data from frontend and appends to pose_log if session is active."""
    global pose_log, session_active
    
    if not session_active:
        return jsonify({'status': 'info', 'message': 'Session is not active.'}), 200

    try:
        data = request.get_json()
        if data and 'pose_data' in data:
            pose_log.extend(data['pose_data'])
            return jsonify({'status': 'success', 'message': f'Logged {len(data["pose_data"])} poses.'}), 200
        
        return jsonify({'status': 'error', 'message': 'Invalid pose data format.'}), 400
    except Exception as e:
        print(f"Error logging pose data: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to process pose data.'}), 500


@app.route('/start_session', methods=['POST'])
@api_key_required 
def start_session():
    """Start yoga session (protected route)"""
    global session_active, pose_log
    
    session_active = True
    pose_log = [] # Clear previous log
    
    return jsonify({'status': 'success', 'message': f'Session started for {current_user_display_name}'})


@app.route('/end_session', methods=['POST'])
@api_key_required 
def end_session():
    """End session, calculate points, generate report, and store in Supabase (protected route)"""
    global session_active, current_user_uid, current_user_display_name
    
    if not session_active:
        return jsonify({'status': 'error', 'message': 'No active session.'}), 400
    
    session_active = False
    time.sleep(1) # Wait for final logs
    session_end_time = datetime.now()
    
    points_awarded = 0
    limit_message = None
    duration_seconds = 0
    avg_conf = 0
    
    if pose_log:
        df = pd.DataFrame(pose_log)
        
        if len(pose_log) > 1:
            # Note: We must convert datetime to ISO string for Supabase/Postgres storage
            start_time = datetime.fromtimestamp(pose_log[0]['timestamp'])
            
            duration_seconds = pose_log[-1]['timestamp'] - pose_log[0]['timestamp']
            duration_min = duration_seconds / 60
        else:
            duration_min = 0.0
            start_time = session_end_time
        
        avg_conf = df['confidence'].mean()
        
        duration_min = max(0, duration_min) 
        base_factor = 10 
        points_awarded = int(avg_conf * duration_min * base_factor)
        
        if points_awarded >= 100:
            points_awarded = 99
            limit_message = "Points capped at 99 to comply with limit."
    
    # Data structure must match your Supabase table columns
    supabase_data = {
        'user_uid': current_user_uid,
        'display_name': current_user_display_name,
        'points_awarded': points_awarded,
        'session_end_time': session_end_time.isoformat(), # ISO format for Supabase timestamp
        'duration_seconds': duration_seconds,
        'average_confidence': float(avg_conf),
        'pose_counts': df['pose'].value_counts().to_dict() if pose_log else {},
    }
    
    # We send the data as a list of one dictionary to fit Supabase's insert method
    storage_success = save_session_to_supabase([supabase_data])
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
        'session_active': session_active,
        'poses_logged': len(pose_log),
        'current_user_uid': current_user_uid,
        'current_user_display_name': current_user_display_name
    })
