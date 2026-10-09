import streamlit as st
import pandas as pd
from datetime import datetime

from src.pipelines.voice_pipeline import process_bulk_audio
from src.database.config import supabase
from src.components.dialog_attendance_results import show_attendance_result


@st.dialog('Voice Attendance')
def voice_attendance_dialog(selected_subject_id):
    st.write('Record audio of students saying I am present. Then AI will recognize the students')

    audio_data = st.audio_input("Record classroom audio")
    threshold = st.slider('Match strictness (lower = easier to match)', 0.40, 0.90, 0.60, 0.01)

    if st.button('Analyze Audio', width='stretch', type='primary'):
        if audio_data is None:
            st.warning('Please record the audio first')
            return

        st.session_state.pop('voice_attendance_results', None)

        with st.spinner('Processing Audio data'):
            enrolled_res = supabase.table('subject_students').select("*, students(*)").eq('subject_id', selected_subject_id).execute()
            enrolled_students = enrolled_res.data

            if not enrolled_students:
                st.warning('No students enrolled in this course')
                return

            candidates_dict = {
                s['students']['student_id']: s['students']['voice_embedding']
                for s in enrolled_students if s['students'].get('voice_embedding')
            }

            if not candidates_dict:
                st.error('No enrolled students have voice profiles registered')
                return

            try:
                present, closest = process_bulk_audio(audio_data.read(), candidates_dict, threshold=threshold)
            except Exception as e:
                st.error(f'Voice processing failed: {e}')
                return

            results, attendance_to_log = [], []
            current_timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

            for node in enrolled_students:
                student = node['students']
                sid = student['student_id']
                is_present = sid in present

                results.append({
                    "Name": student['name'],
                    "ID": sid,
                    "Match score": round(closest[sid], 2) if sid in closest else "No voice profile",
                    "Status": "✅ Present" if is_present else "❌ Absent"
                })

                attendance_to_log.append({
                    'student_id': sid,
                    'subject_id': selected_subject_id,
                    'timestamp': current_timestamp,
                    'is_present': bool(is_present)
                })

            st.session_state.voice_attendance_results = (pd.DataFrame(results), attendance_to_log)

    if st.session_state.get('voice_attendance_results'):
        st.divider()
        df_results, logs = st.session_state.voice_attendance_results
        show_attendance_result(df_results, logs)