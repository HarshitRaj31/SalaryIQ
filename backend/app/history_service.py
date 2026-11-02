"""
SQLite Database History Service for Job Salary Prediction.

Provides persistent, ACID-compliant storage for prediction history:
- Thread-safe SQLite database connectivity with WAL (Write-Ahead Logging) mode
- Storing prediction requests, model predictions, and confidence margins
- Paginated retrieval, search filtering by role and location
- Granular record lookups, deletions, and history clearance
- Aggregated statistical summaries for analytics dashboards
"""

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

try:
    from app.core import config
    DEFAULT_DB_PATH = config.DB_PATH
except Exception:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.dirname(CURRENT_DIR)
    DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "data", "prediction_history.db")


class HistoryService:
    """
    SQLite-backed service managing historical salary prediction records.
    """

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        
        # Ensure database directory exists
        db_dir = os.path.dirname(os.path.abspath(self.db_path))
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir, exist_ok=True)

        self.init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """
        Creates and configures a SQLite connection with dict-like row access
        and WAL mode for high concurrency.
        """
        conn = sqlite3.connect(self.db_path, timeout=30.0)
        conn.row_factory = sqlite3.Row
        # WAL mode improves concurrent read/write performance
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self) -> None:
        """
        Initializes the SQLite schema with table and performance indexes if not existing.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    work_year INTEGER NOT NULL,
                    experience_level TEXT NOT NULL,
                    employment_type TEXT NOT NULL,
                    job_title TEXT NOT NULL,
                    employee_residence TEXT NOT NULL,
                    remote_ratio INTEGER NOT NULL,
                    company_location TEXT NOT NULL,
                    company_size TEXT NOT NULL,
                    predicted_salary_usd REAL NOT NULL,
                    formatted_salary TEXT NOT NULL,
                    salary_range_min REAL NOT NULL,
                    salary_range_max REAL NOT NULL,
                    formatted_min TEXT NOT NULL,
                    formatted_max TEXT NOT NULL,
                    confidence_margin_mae REAL NOT NULL,
                    model_version TEXT NOT NULL,
                    input_features_json TEXT,
                    prediction_result_json TEXT
                );
                """
            )

            # Performance indexes for search, filtering, and time-series ordering
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_job_title ON predictions(job_title);"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS idx_predictions_experience ON predictions(experience_level);"
            )
            conn.commit()

    def save_prediction(
        self, request_data: Dict[str, Any], prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Saves a prediction record to the SQLite database without mutating original prediction logic.

        Parameters:
            request_data: User input dictionary (job_title, experience_level, etc.)
            prediction_result: Output from prediction_service.predict()

        Returns:
            Dictionary containing the stored prediction with ID and UTC timestamp.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Sanitize and extract request input parameters
        work_year = int(request_data.get("work_year", 2023))
        experience_level = str(request_data.get("experience_level", "SE")).strip().upper()
        employment_type = str(request_data.get("employment_type", "FT")).strip().upper()
        job_title = str(request_data.get("job_title", "Data Scientist")).strip()
        employee_residence = str(request_data.get("employee_residence", "US")).strip().upper()
        remote_ratio = int(request_data.get("remote_ratio", 100))
        company_location = str(request_data.get("company_location", "US")).strip().upper()
        company_size = str(request_data.get("company_size", "M")).strip().upper()

        # Extract prediction outputs
        predicted_salary_usd = float(prediction_result.get("predicted_salary_usd", 0.0))
        formatted_salary = str(
            prediction_result.get("formatted_salary", f"${round(predicted_salary_usd):,}")
        )

        salary_range = prediction_result.get("salary_range", {})
        salary_range_min = float(salary_range.get("min", max(0.0, predicted_salary_usd - 39000.0)))
        salary_range_max = float(salary_range.get("max", predicted_salary_usd + 39000.0))
        formatted_min = str(salary_range.get("formatted_min", f"${round(salary_range_min):,}"))
        formatted_max = str(salary_range.get("formatted_max", f"${round(salary_range_max):,}"))

        confidence_margin_mae = float(prediction_result.get("confidence_margin_mae", 39067.0))
        model_version = str(prediction_result.get("model_version", "best_model.pkl"))

        input_features_json = json.dumps(request_data, ensure_ascii=False)
        prediction_result_json = json.dumps(prediction_result, ensure_ascii=False)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO predictions (
                    timestamp,
                    work_year,
                    experience_level,
                    employment_type,
                    job_title,
                    employee_residence,
                    remote_ratio,
                    company_location,
                    company_size,
                    predicted_salary_usd,
                    formatted_salary,
                    salary_range_min,
                    salary_range_max,
                    formatted_min,
                    formatted_max,
                    confidence_margin_mae,
                    model_version,
                    input_features_json,
                    prediction_result_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                """,
                (
                    timestamp,
                    work_year,
                    experience_level,
                    employment_type,
                    job_title,
                    employee_residence,
                    remote_ratio,
                    company_location,
                    company_size,
                    predicted_salary_usd,
                    formatted_salary,
                    salary_range_min,
                    salary_range_max,
                    formatted_min,
                    formatted_max,
                    confidence_margin_mae,
                    model_version,
                    input_features_json,
                    prediction_result_json,
                ),
            )
            new_id = cursor.lastrowid
            conn.commit()

        return self.get_prediction_by_id(new_id)

    def get_history(
        self,
        limit: int = 50,
        offset: int = 0,
        search: Optional[str] = None,
        experience_level: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves paginated prediction history, ordered by most recent first.

        Parameters:
            limit: Maximum records to return (default: 50, max: 200)
            offset: Pagination offset
            search: Optional substring search against job title or location
            experience_level: Optional filter by EN, MI, SE, EX
        """
        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        query = "SELECT * FROM predictions WHERE 1=1"
        params: List[Any] = []

        if search:
            query += " AND (job_title LIKE ? OR company_location LIKE ? OR employee_residence LIKE ?)"
            search_param = f"%{search.strip()}%"
            params.extend([search_param, search_param, search_param])

        if experience_level:
            query += " AND experience_level = ?"
            params.append(experience_level.strip().upper())

        query += " ORDER BY id DESC LIMIT ? OFFSET ?;"
        params.extend([limit, offset])

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [self._row_to_dict(row) for row in rows]

    def get_total_count(
        self, search: Optional[str] = None, experience_level: Optional[str] = None
    ) -> int:
        """
        Returns the total number of records matching filter criteria.
        """
        query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
        params: List[Any] = []

        if search:
            query += " AND (job_title LIKE ? OR company_location LIKE ? OR employee_residence LIKE ?)"
            search_param = f"%{search.strip()}%"
            params.extend([search_param, search_param, search_param])

        if experience_level:
            query += " AND experience_level = ?"
            params.append(experience_level.strip().upper())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return int(cursor.fetchone()[0])

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """
        Retrieves a single prediction by primary key ID.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?;", (prediction_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def delete_prediction(self, prediction_id: int) -> bool:
        """
        Deletes a specific prediction record by primary key ID.
        Returns True if a record was removed, False otherwise.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?;", (prediction_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> int:
        """
        Deletes all historical records from the predictions table.
        Returns the number of deleted records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions;")
            count = int(cursor.fetchone()[0])
            cursor.execute("DELETE FROM predictions;")
            # Reset SQLite auto-increment
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions';")
            conn.commit()
            return count

    def get_summary_stats(self) -> Dict[str, Any]:
        """
        Calculates aggregate statistics over all stored prediction history records.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Base aggregations
            cursor.execute(
                """
                SELECT
                    COUNT(*) as total_count,
                    COALESCE(AVG(predicted_salary_usd), 0.0) as avg_salary,
                    COALESCE(MIN(predicted_salary_usd), 0.0) as min_salary,
                    COALESCE(MAX(predicted_salary_usd), 0.0) as max_salary,
                    MAX(timestamp) as last_prediction_at
                FROM predictions;
                """
            )
            base_row = cursor.fetchone()

            total_count = int(base_row["total_count"])
            if total_count == 0:
                return {
                    "total_predictions": 0,
                    "average_predicted_salary": 0.0,
                    "formatted_avg_salary": "$0",
                    "min_predicted_salary": 0.0,
                    "max_predicted_salary": 0.0,
                    "most_frequent_role": None,
                    "top_experience_level": None,
                    "top_locations": [],
                    "remote_distribution": {},
                    "last_prediction_at": None,
                }

            avg_salary = round(float(base_row["avg_salary"]), 2)
            min_salary = round(float(base_row["min_salary"]), 2)
            max_salary = round(float(base_row["max_salary"]), 2)
            last_prediction_at = base_row["last_prediction_at"]

            # Most frequent job title
            cursor.execute(
                """
                SELECT job_title, COUNT(*) as cnt
                FROM predictions
                GROUP BY job_title
                ORDER BY cnt DESC
                LIMIT 1;
                """
            )
            role_row = cursor.fetchone()
            most_frequent_role = role_row["job_title"] if role_row else None

            # Most frequent experience level
            cursor.execute(
                """
                SELECT experience_level, COUNT(*) as cnt
                FROM predictions
                GROUP BY experience_level
                ORDER BY cnt DESC
                LIMIT 1;
                """
            )
            exp_row = cursor.fetchone()
            top_experience_level = exp_row["experience_level"] if exp_row else None

            # Top 5 company locations
            cursor.execute(
                """
                SELECT company_location, COUNT(*) as count
                FROM predictions
                GROUP BY company_location
                ORDER BY count DESC
                LIMIT 5;
                """
            )
            top_locations = [
                {"location": row["company_location"], "count": int(row["count"])}
                for row in cursor.fetchall()
            ]

            # Remote ratio distribution
            cursor.execute(
                """
                SELECT remote_ratio, COUNT(*) as count
                FROM predictions
                GROUP BY remote_ratio
                ORDER BY remote_ratio ASC;
                """
            )
            remote_distribution = {
                f"{row['remote_ratio']}%": int(row["count"]) for row in cursor.fetchall()
            }

            return {
                "total_predictions": total_count,
                "average_predicted_salary": avg_salary,
                "formatted_avg_salary": f"${round(avg_salary):,}",
                "min_predicted_salary": min_salary,
                "max_predicted_salary": max_salary,
                "most_frequent_role": most_frequent_role,
                "top_experience_level": top_experience_level,
                "top_locations": top_locations,
                "remote_distribution": remote_distribution,
                "last_prediction_at": last_prediction_at,
            }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """
        Transforms a database row into a structured prediction response object.
        """
        input_features = {
            "work_year": int(row["work_year"]),
            "experience_level": str(row["experience_level"]),
            "employment_type": str(row["employment_type"]),
            "job_title": str(row["job_title"]),
            "employee_residence": str(row["employee_residence"]),
            "remote_ratio": int(row["remote_ratio"]),
            "company_location": str(row["company_location"]),
            "company_size": str(row["company_size"]),
        }

        salary_range = {
            "min": round(float(row["salary_range_min"]), 2),
            "max": round(float(row["salary_range_max"]), 2),
            "formatted_min": str(row["formatted_min"]),
            "formatted_max": str(row["formatted_max"]),
        }

        return {
            "id": int(row["id"]),
            "timestamp": str(row["timestamp"]),
            "job_title": str(row["job_title"]),
            "experience_level": str(row["experience_level"]),
            "employment_type": str(row["employment_type"]),
            "company_location": str(row["company_location"]),
            "employee_residence": str(row["employee_residence"]),
            "company_size": str(row["company_size"]),
            "remote_ratio": int(row["remote_ratio"]),
            "work_year": int(row["work_year"]),
            "predicted_salary_usd": round(float(row["predicted_salary_usd"]), 2),
            "formatted_salary": str(row["formatted_salary"]),
            "salary_range": salary_range,
            "confidence_margin_mae": round(float(row["confidence_margin_mae"]), 2),
            "model_version": str(row["model_version"]),
            "input_features": input_features,
        }

    def __repr__(self) -> str:
        return f"<HistoryService db_path='{self.db_path}'>"


# Default global service instance
history_service = HistoryService()

