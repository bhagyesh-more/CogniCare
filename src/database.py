"""
src/database.py
Clinical data access and persistence layer using SQLite.
Manages multi-tenant hospital metadata, doctor profiles, patient clinical details,
and historical cognitive/arousal assessment trends.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
import random

from contextlib import contextmanager

DB_PATH = Path("data/cognicare.db")


def hash_password(password: str) -> str:
    """Helper to hash doctor passwords using salted SHA-256."""
    salt = "cognicare_secret_salt_2026"
    return hashlib.sha256((password + salt).encode("utf-8")).hexdigest()


class DatabaseService:
    """Provides unified database operations for CogniCare."""

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path
        # Ensure directories exist
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    @contextmanager
    def get_connection(self):
        """Create, yield, and automatically close a SQLite connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Access columns by name
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
        finally:
            conn.close()

    def init_db(self) -> None:
        """Create database tables if they do not exist and seed default records."""
        with self.get_connection() as conn:
            # 1. Hospitals Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS hospitals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL,
                registration_code TEXT UNIQUE NOT NULL,
                created_at TEXT NOT NULL
            );
            """)

            # 2. Doctors Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS doctors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hospital_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (hospital_id) REFERENCES hospitals (id) ON DELETE CASCADE
            );
            """)

            # 3. Patients Table (Doctor Managed)
            conn.execute("""
            CREATE TABLE IF NOT EXISTS patients (
                id TEXT PRIMARY KEY,
                doctor_id INTEGER NOT NULL,
                hospital_id INTEGER NOT NULL,
                name TEXT NOT NULL,
                age INTEGER NOT NULL,
                gender TEXT NOT NULL,
                dob TEXT NOT NULL,
                contact_number TEXT,
                address TEXT,
                occupation TEXT,
                industry TEXT,
                work_schedule TEXT,
                education_level TEXT,
                clinical_notes TEXT,
                registration_date TEXT NOT NULL,
                FOREIGN KEY (doctor_id) REFERENCES doctors (id) ON DELETE CASCADE,
                FOREIGN KEY (hospital_id) REFERENCES hospitals (id) ON DELETE CASCADE
            );
            """)

            # 4. Assessments Table
            conn.execute("""
            CREATE TABLE IF NOT EXISTS assessments (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                patient_id TEXT NOT NULL,
                doctor_id INTEGER NOT NULL,
                timestamp TEXT NOT NULL,
                session_id TEXT NOT NULL,
                eda_mean REAL NOT NULL,
                eda_std REAL NOT NULL,
                eda_peak_count REAL NOT NULL,
                eda_peak_amplitude REAL NOT NULL,
                heart_rate_bpm REAL NOT NULL,
                rmssd REAL NOT NULL,
                sdnn REAL NOT NULL,
                resp_rate_bpm REAL NOT NULL,
                resp_variability REAL NOT NULL,
                arousal_pred TEXT NOT NULL,
                arousal_conf REAL NOT NULL,
                arousal_tier TEXT NOT NULL,
                arousal_narrative TEXT NOT NULL,
                load_pred TEXT NOT NULL,
                load_conf REAL NOT NULL,
                load_tier TEXT NOT NULL,
                load_narrative TEXT NOT NULL,
                shap_values TEXT NOT NULL, -- JSON String
                flag_review INTEGER NOT NULL, -- 0 or 1
                FOREIGN KEY (patient_id) REFERENCES patients (id) ON DELETE CASCADE,
                FOREIGN KEY (doctor_id) REFERENCES doctors (id) ON DELETE CASCADE
            );
            """)
            conn.commit()

        # Seed initial data if database is empty
        self.seed_if_empty()

    def seed_if_empty(self) -> None:
        """Seed default hospital, doctor, patients and assessment history if empty."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM hospitals")
            if cursor.fetchone()[0] > 0:
                return  # Database is already populated

            # 1. Add Clinic
            cursor.execute(
                "INSERT INTO hospitals (name, registration_code, created_at) VALUES (?, ?, ?)",
                ("Neo-General Hospital", "NEO-GEN", datetime.now().isoformat())
            )
            hospital_id = cursor.lastrowid

            # 2. Add Doctor
            doctor_pwd = hash_password("password123")
            cursor.execute(
                "INSERT INTO doctors (hospital_id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                (hospital_id, "Dr. Evelyn Vance", "evelyn@cognicare.com", doctor_pwd, datetime.now().isoformat())
            )
            doctor_id = cursor.lastrowid

            # 3. Add Patients
            patients_data = [
                ("PAT-2026-0001", "Arthur Pendelton", 45, "Male", "1981-04-12", "+1-555-0199", "452 Cyber Way, Sector 7", "Lead DevOps Engineer", "Tech / Cybersecurity", "9 AM - 6 PM (Day)", "Bachelor of Science", "Patient complains of stress spikes during deployment cycles."),
                ("PAT-2026-0002", "Sarah Jenkins", 32, "Female", "1994-09-24", "+1-555-0245", "102 Glassmorphism Blvd", "Financial Risk Analyst", "Banking", "8 AM - 5 PM (Day)", "Master of Finance", "Monitored for cognitive fatigue in high-stress trading contexts."),
                ("PAT-2026-0003", "Devon Cole", 28, "Male", "1998-11-05", "+1-555-0371", "77 Neon District East", "Air Traffic Controller", "Aviation", "Rotating Shifts (24/7)", "Associate Degree", "Critical monitoring requirement for attention maintenance during shift handovers."),
                ("PAT-2026-0004", "Lana Croft", 37, "Female", "1989-02-18", "+1-555-0482", "30 Manor House Hill", "Field Archaeologist", "Research", "Irregular / Outdoors", "Doctorate", "Assessed for high physiological adaptability and baseline arousal triggers.")
            ]

            for pat_id, name, age, gen, dob, phone, addr, occ, ind, sched, edu, notes in patients_data:
                cursor.execute(
                    """
                    INSERT INTO patients (id, doctor_id, hospital_id, name, age, gender, dob, contact_number, address,
                                          occupation, industry, work_schedule, education_level, clinical_notes, registration_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (pat_id, doctor_id, hospital_id, name, age, gen, dob, phone, addr, occ, ind, sched, edu, notes,
                     (datetime.now() - timedelta(days=30)).isoformat())
                )

            # 4. Add Historical Assessments for Patients (to populate beautiful charts)
            now = datetime.now()
            signals = [
                {"eda_mean": 0.2, "eda_std": -0.1, "eda_peak_count": 0.5, "eda_peak_amplitude": 0.1, "heart_rate_bpm": -0.5, "rmssd": 0.2, "sdnn": 0.1, "resp_rate_bpm": -0.3, "resp_variability": 0.4, "arousal": "Low", "load": "Low", "conf_a": 0.88, "conf_c": 0.91, "narrative": "Low emotional arousal and cognitive load predicted with high confidence, supported by stable heart rate variability and reduced respiration rate."},
                {"eda_mean": 0.8, "eda_std": 0.4, "eda_peak_count": 1.2, "eda_peak_amplitude": 0.8, "heart_rate_bpm": 0.9, "rmssd": -0.7, "sdnn": -0.8, "resp_rate_bpm": 0.8, "resp_variability": -0.5, "arousal": "High", "load": "High", "conf_a": 0.93, "conf_c": 0.89, "narrative": "High emotional arousal and cognitive load predicted with high confidence, primarily driven by elevated heart rate and electrodermal response with reduced heart rate variability (RMSSD)."},
                {"eda_mean": 0.4, "eda_std": 0.1, "eda_peak_count": 0.2, "eda_peak_amplitude": 0.3, "heart_rate_bpm": 0.1, "rmssd": -0.2, "sdnn": -0.1, "resp_rate_bpm": 0.2, "resp_variability": -0.1, "arousal": "Medium", "load": "Medium", "conf_a": 0.74, "conf_c": 0.71, "narrative": "Medium cognitive load and arousal predicted with moderate confidence. Features indicate minor spikes in sweat gland activity but stable ECG levels."}
            ]

            # Arthur Pendelton History
            for i in range(12):
                date_str = (now - timedelta(days=(12-i)*2)).isoformat()
                sig = random.choice(signals)
                shap_list = [
                    {"feature": "heart_rate_bpm", "shap_value": float(sig["heart_rate_bpm"] * 0.15), "direction": "elevated" if sig["heart_rate_bpm"] > 0 else "reduced"},
                    {"feature": "eda_mean", "shap_value": float(sig["eda_mean"] * 0.1), "direction": "elevated" if sig["eda_mean"] > 0 else "reduced"},
                    {"feature": "rmssd", "shap_value": float(sig["rmssd"] * -0.08), "direction": "reduced" if sig["rmssd"] < 0 else "elevated"}
                ]
                cursor.execute(
                    """
                    INSERT INTO assessments (
                        patient_id, doctor_id, timestamp, session_id,
                        eda_mean, eda_std, eda_peak_count, eda_peak_amplitude,
                        heart_rate_bpm, rmssd, sdnn, resp_rate_bpm, resp_variability,
                        arousal_pred, arousal_conf, arousal_tier, arousal_narrative,
                        load_pred, load_conf, load_tier, load_narrative,
                        shap_values, flag_review
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "PAT-2026-0001", doctor_id, date_str, f"sess-{random.randint(1000, 9999)}",
                        sig["eda_mean"], sig["eda_std"], sig["eda_peak_count"], sig["eda_peak_amplitude"],
                        sig["heart_rate_bpm"], sig["rmssd"], sig["sdnn"], sig["resp_rate_bpm"], sig["resp_variability"],
                        sig["arousal"], sig["conf_a"], "High" if sig["conf_a"] > 0.85 else "Medium", sig["narrative"],
                        sig["load"], sig["conf_c"], "High" if sig["conf_c"] > 0.85 else "Medium", sig["narrative"],
                        json.dumps(shap_list), 0
                    )
                )

            # Sarah Jenkins History
            for i in range(8):
                date_str = (now - timedelta(days=(8-i)*3)).isoformat()
                sig = random.choice(signals)
                shap_list = [
                    {"feature": "heart_rate_bpm", "shap_value": float(sig["heart_rate_bpm"] * 0.12), "direction": "elevated" if sig["heart_rate_bpm"] > 0 else "reduced"},
                    {"feature": "eda_mean", "shap_value": float(sig["eda_mean"] * 0.08), "direction": "elevated" if sig["eda_mean"] > 0 else "reduced"},
                    {"feature": "rmssd", "shap_value": float(sig["rmssd"] * -0.06), "direction": "reduced" if sig["rmssd"] < 0 else "elevated"}
                ]
                cursor.execute(
                    """
                    INSERT INTO assessments (
                        patient_id, doctor_id, timestamp, session_id,
                        eda_mean, eda_std, eda_peak_count, eda_peak_amplitude,
                        heart_rate_bpm, rmssd, sdnn, resp_rate_bpm, resp_variability,
                        arousal_pred, arousal_conf, arousal_tier, arousal_narrative,
                        load_pred, load_conf, load_tier, load_narrative,
                        shap_values, flag_review
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        "PAT-2026-0002", doctor_id, date_str, f"sess-{random.randint(1000, 9999)}",
                        sig["eda_mean"], sig["eda_std"], sig["eda_peak_count"], sig["eda_peak_amplitude"],
                        sig["heart_rate_bpm"], sig["rmssd"], sig["sdnn"], sig["resp_rate_bpm"], sig["resp_variability"],
                        sig["arousal"], sig["conf_a"], "High" if sig["conf_a"] > 0.85 else "Medium", sig["narrative"],
                        sig["load"], sig["conf_c"], "High" if sig["conf_c"] > 0.85 else "Medium", sig["narrative"],
                        json.dumps(shap_list), 0
                    )
                )

            conn.commit()

    # ------------------------------------------------------------------
    # Hospital / Clinic Operations
    # ------------------------------------------------------------------

    def register_hospital(self, name: str, code: str) -> bool:
        """Register a new hospital entity."""
        try:
            with self.get_connection() as conn:
                conn.execute(
                    "INSERT INTO hospitals (name, registration_code, created_at) VALUES (?, ?, ?)",
                    (name, code.upper().strip(), datetime.now().isoformat())
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Code already exists

    def get_hospital_by_code(self, code: str) -> dict | None:
        """Retrieve hospital details by its unique clinic identifier code."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM hospitals WHERE registration_code = ?", (code.upper().strip(),))
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_hospital_stats(self, hospital_id: int) -> dict:
        """Get summary stats for a hospital."""
        stats = {"doctors": 0, "patients": 0, "assessments": 0}
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM doctors WHERE hospital_id = ?", (hospital_id,))
            stats["doctors"] = cursor.fetchone()[0]

            cursor = conn.execute("SELECT COUNT(*) FROM patients WHERE hospital_id = ?", (hospital_id,))
            stats["patients"] = cursor.fetchone()[0]

            cursor = conn.execute(
                """
                SELECT COUNT(*) FROM assessments a
                JOIN patients p ON a.patient_id = p.id
                WHERE p.hospital_id = ?
                """, (hospital_id,)
            )
            stats["assessments"] = cursor.fetchone()[0]
        return stats

    # ------------------------------------------------------------------
    # Doctor Operations
    # ------------------------------------------------------------------

    def register_doctor(self, hospital_id: int, name: str, email: str, password: str) -> bool:
        """Register a new doctor clinician."""
        try:
            with self.get_connection() as conn:
                hashed = hash_password(password)
                conn.execute(
                    "INSERT INTO doctors (hospital_id, name, email, password_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                    (hospital_id, name.strip(), email.lower().strip(), hashed, datetime.now().isoformat())
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Email already registered

    def authenticate_doctor(self, email: str, password: str) -> dict | None:
        """Verify doctor credentials and return user details."""
        with self.get_connection() as conn:
            hashed = hash_password(password)
            cursor = conn.execute(
                """
                SELECT d.*, h.name as hospital_name, h.registration_code as hospital_code
                FROM doctors d
                JOIN hospitals h ON d.hospital_id = h.id
                WHERE d.email = ? AND d.password_hash = ?
                """, (email.lower().strip(), hashed)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_doctors_in_hospital(self, hospital_id: int) -> list[dict]:
        """Fetch all doctors registered under a hospital."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT id, name, email, created_at FROM doctors WHERE hospital_id = ?", (hospital_id,))
            return [dict(r) for r in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Patient Operations
    # ------------------------------------------------------------------

    def create_patient(self, doctor_id: int, hospital_id: int, name: str, age: int, gender: str,
                       dob: str, contact: str, address: str, occupation: str, industry: str,
                       schedule: str, education: str, notes: str) -> str:
        """Register a new patient profile and return their generated ID."""
        date_prefix = datetime.now().strftime("%Y%m%d")
        rand_suffix = f"{random.randint(100, 999)}"
        patient_id = f"PAT-{date_prefix}-{rand_suffix}"

        with self.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO patients (id, doctor_id, hospital_id, name, age, gender, dob, contact_number, address,
                                      occupation, industry, work_schedule, education_level, clinical_notes, registration_date)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (patient_id, doctor_id, hospital_id, name.strip(), age, gender, dob, contact.strip(), address.strip(),
                 occupation.strip(), industry.strip(), schedule.strip(), education.strip(), notes.strip(), datetime.now().isoformat())
            )
            conn.commit()
        return patient_id

    def get_patient(self, patient_id: str) -> dict | None:
        """Fetch patient clinical profile by their ID."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT p.*, d.name as doctor_name
                FROM patients p
                JOIN doctors d ON p.doctor_id = d.id
                WHERE p.id = ?
                """, (patient_id,)
            )
            row = cursor.fetchone()
            return dict(row) if row else None

    def get_patients_by_hospital(self, hospital_id: int) -> list[dict]:
        """Fetch all patients registered under the clinic."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT p.*, d.name as doctor_name
                FROM patients p
                JOIN doctors d ON p.doctor_id = d.id
                WHERE p.hospital_id = ?
                ORDER BY p.registration_date DESC
                """, (hospital_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def search_patients(self, hospital_id: int, query: str) -> list[dict]:
        """Search patients by name, id or doctor name."""
        with self.get_connection() as conn:
            q = f"%{query}%"
            cursor = conn.execute(
                """
                SELECT p.*, d.name as doctor_name
                FROM patients p
                JOIN doctors d ON p.doctor_id = d.id
                WHERE p.hospital_id = ? AND (p.name LIKE ? OR p.id LIKE ? OR d.name LIKE ?)
                ORDER BY p.registration_date DESC
                """, (hospital_id, q, q, q)
            )
            return [dict(r) for r in cursor.fetchall()]

    # ------------------------------------------------------------------
    # Assessment Operations
    # ------------------------------------------------------------------

    def log_assessment(self, patient_id: str, doctor_id: int, session_id: str, inputs: dict,
                       a_pred: str, a_conf: float, a_tier: str, a_narrative: str,
                       c_pred: str, c_conf: float, c_tier: str, c_narrative: str,
                       shap_list: list[dict], flag_review: bool) -> int:
        """Log a completed clinical prediction to the history log."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO assessments (
                    patient_id, doctor_id, timestamp, session_id,
                    eda_mean, eda_std, eda_peak_count, eda_peak_amplitude,
                    heart_rate_bpm, rmssd, sdnn, resp_rate_bpm, resp_variability,
                    arousal_pred, arousal_conf, arousal_tier, arousal_narrative,
                    load_pred, load_conf, load_tier, load_narrative,
                    shap_values, flag_review
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    patient_id, doctor_id, datetime.now().isoformat(), session_id,
                    inputs["eda_mean"], inputs["eda_std"], inputs["eda_peak_count"], inputs["eda_peak_amplitude"],
                    inputs["heart_rate_bpm"], inputs["rmssd"], inputs["sdnn"], inputs["resp_rate_bpm"], inputs["resp_variability"],
                    a_pred, a_conf, a_tier, a_narrative,
                    c_pred, c_conf, c_tier, c_narrative,
                    json.dumps(shap_list), 1 if flag_review else 0
                )
            )
            conn.commit()
            return cursor.lastrowid

    def get_patient_assessments(self, patient_id: str) -> list[dict]:
        """Fetch full chronological assessment log for a patient."""
        with self.get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM assessments
                WHERE patient_id = ?
                ORDER BY timestamp DESC
                """, (patient_id,)
            )
            return [dict(r) for r in cursor.fetchall()]

    def get_assessment(self, assessment_id: int) -> dict | None:
        """Fetch a single assessment details by ID."""
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM assessments WHERE id = ?", (assessment_id,))
            row = cursor.fetchone()
            return dict(row) if row else None
