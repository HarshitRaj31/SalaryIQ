"""
SQLite Database History Service for Job Salary Prediction.
Supports dual-country historical records for India and the United States:
- Preserves all legacy US historical prediction records.
- Adds non-destructive schema migrations for country, currency, salary unit, and model used.
- Thread-safe SQLite database connectivity with WAL (Write-Ahead Logging) mode.
- Paginated retrieval, search filtering by role, location, experience, and country.
"""

import json
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

try:
    from app.core import config
    DEFAULT_DB_PATH = config.DB_PATH
except Exception:
    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    BACKEND_DIR = os.path.dirname(CURRENT_DIR)
    DEFAULT_DB_PATH = os.path.join(BACKEND_DIR, "data", "prediction_history.db")


class HistoryService:
    """
    SQLite-backed service managing historical salary prediction records for US and India.
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
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def init_db(self) -> None:
        """
        Initializes the SQLite schema with table and indexes,
        and applies non-destructive column migrations for dual-country support.
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Create base table if not existing
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

            # Check existing columns for non-destructive migration
            cursor.execute("PRAGMA table_info(predictions);")
            existing_cols = {row["name"] for row in cursor.fetchall()}

            # Add dual-country tracking columns if not already present
            if "country" not in existing_cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN country TEXT DEFAULT 'US';")
                logger.info("[HistoryService] Migrated schema: Added 'country' column.")

            if "currency" not in existing_cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN currency TEXT DEFAULT 'USD';")
                logger.info("[HistoryService] Migrated schema: Added 'currency' column.")

            if "salary_unit" not in existing_cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN salary_unit TEXT DEFAULT 'YEAR';")
                logger.info("[HistoryService] Migrated schema: Added 'salary_unit' column.")

            if "predicted_salary" not in existing_cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN predicted_salary REAL;")
                logger.info("[HistoryService] Migrated schema: Added 'predicted_salary' column.")

            if "model_used" not in existing_cols:
                cursor.execute("ALTER TABLE predictions ADD COLUMN model_used TEXT;")
                logger.info("[HistoryService] Migrated schema: Added 'model_used' column.")

            # Performance indexes
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_timestamp ON predictions(timestamp DESC);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_job_title ON predictions(job_title);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_experience ON predictions(experience_level);")
            cursor.execute("CREATE INDEX IF NOT EXISTS idx_predictions_country ON predictions(country);")

            # Backfill legacy rows with default US values if null
            cursor.execute(
                """
                UPDATE predictions 
                SET country = 'US', 
                    currency = 'USD', 
                    salary_unit = 'YEAR', 
                    predicted_salary = predicted_salary_usd,
                    model_used = model_version
                WHERE country IS NULL;
                """
            )
            conn.commit()

    def save_prediction(
        self, request_data: Dict[str, Any], prediction_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Saves a prediction record (US or India) to the SQLite database without mutating original logic.
        """
        timestamp = datetime.now(timezone.utc).isoformat()

        # Extract country & currency metadata
        country = str(
            prediction_result.get("country") or request_data.get("country") or "US"
        ).strip().upper()
        if country in ["INDIA", "IND"]:
            country = "IN"
        elif country in ["USA", "UNITED STATES"]:
            country = "US"

        currency = str(
            prediction_result.get("currency") or ("INR" if country == "IN" else "USD")
        )
        salary_unit = str(
            prediction_result.get("salary_unit") or ("LPA" if country == "IN" else "YEAR")
        )
        model_used = str(
            prediction_result.get("model_used")
            or ("Gradient Boosting Regressor" if country == "IN" else "Random Forest Regressor")
        )

        predicted_salary = float(
            prediction_result.get("predicted_salary")
            if prediction_result.get("predicted_salary") is not None
            else prediction_result.get("predicted_salary_usd", 0.0)
        )

        # Sanitize and extract request input parameters
        work_year = int(request_data.get("work_year", 2026 if country == "IN" else 2023))
        experience_level = str(request_data.get("experience_level", "SE")).strip().upper()
        employment_type = str(request_data.get("employment_type", "FT")).strip().upper()

        # Job role handling
        job_title = str(
            request_data.get("role_standard")
            or request_data.get("job_title")
            or "Data Scientist"
        ).strip()

        # Location handling
        if country == "IN":
            city = str(
                request_data.get("primary_city_clean")
                or request_data.get("company_location")
                or "Bangalore"
            ).strip()
            company_location = city
            employee_residence = city
        else:
            company_location = str(request_data.get("company_location", "US")).strip().upper()
            employee_residence = str(request_data.get("employee_residence", company_location)).strip().upper()

        remote_ratio = int(request_data.get("remote_ratio", 50 if country == "IN" else 100))
        company_size = str(request_data.get("company_size", "M")).strip().upper()

        # Prediction outputs
        predicted_salary_usd = float(prediction_result.get("predicted_salary_usd", predicted_salary))
        formatted_salary = str(prediction_result.get("formatted_salary", ""))

        salary_range = prediction_result.get("salary_range", {})
        salary_range_min = float(salary_range.get("min", 0.0))
        salary_range_max = float(salary_range.get("max", 0.0))
        formatted_min = str(salary_range.get("formatted_min", ""))
        formatted_max = str(salary_range.get("formatted_max", ""))

        confidence_margin_mae = float(prediction_result.get("confidence_margin_mae", 0.0))
        model_version = str(prediction_result.get("model_version", "model.pkl"))

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
                    prediction_result_json,
                    country,
                    currency,
                    salary_unit,
                    predicted_salary,
                    model_used
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
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
                    country,
                    currency,
                    salary_unit,
                    predicted_salary,
                    model_used,
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
        country: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieves paginated prediction history, ordered by most recent first.
        Supports filtering by country ('US' or 'IN'), role, experience, and search term.
        """
        limit = max(1, min(limit, 200))
        offset = max(0, offset)

        query = "SELECT * FROM predictions WHERE 1=1"
        params: List[Any] = []

        if country and isinstance(country, str):
            c = str(country).strip().upper()
            query += " AND (country = ? OR (country IS NULL AND ? = 'US'))"
            params.extend([c, c])

        if search and isinstance(search, str):
            query += " AND (job_title LIKE ? OR company_location LIKE ? OR employee_residence LIKE ?)"
            search_param = f"%{search.strip()}%"
            params.extend([search_param, search_param, search_param])

        if experience_level and isinstance(experience_level, str):
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
        self,
        search: Optional[str] = None,
        experience_level: Optional[str] = None,
        country: Optional[str] = None,
    ) -> int:
        """Returns total records matching filter criteria."""
        query = "SELECT COUNT(*) FROM predictions WHERE 1=1"
        params: List[Any] = []

        if country and isinstance(country, str):
            c = str(country).strip().upper()
            query += " AND (country = ? OR (country IS NULL AND ? = 'US'))"
            params.extend([c, c])

        if search and isinstance(search, str):
            query += " AND (job_title LIKE ? OR company_location LIKE ? OR employee_residence LIKE ?)"
            search_param = f"%{search.strip()}%"
            params.extend([search_param, search_param, search_param])

        if experience_level and isinstance(experience_level, str):
            query += " AND experience_level = ?"
            params.append(experience_level.strip().upper())

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return int(cursor.fetchone()[0])

    def get_prediction_by_id(self, prediction_id: int) -> Optional[Dict[str, Any]]:
        """Retrieves a single prediction by primary key ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM predictions WHERE id = ?;", (prediction_id,))
            row = cursor.fetchone()
            if row:
                return self._row_to_dict(row)
            return None

    def delete_prediction(self, prediction_id: int) -> bool:
        """Deletes a specific prediction record by primary key ID."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM predictions WHERE id = ?;", (prediction_id,))
            conn.commit()
            return cursor.rowcount > 0

    def clear_history(self) -> int:
        """Deletes all historical records from the predictions table."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM predictions;")
            count = int(cursor.fetchone()[0])
            cursor.execute("DELETE FROM predictions;")
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='predictions';")
            conn.commit()
            return count

    def get_summary_stats(self, country: Optional[str] = None) -> Dict[str, Any]:
        """Calculates aggregate statistics over stored prediction history records."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            country_clause = ""
            params: List[Any] = []
            if country and isinstance(country, str):
                country_clause = " WHERE (country = ? OR (country IS NULL AND ? = 'US'))"
                params.extend([country.upper(), country.upper()])

            cursor.execute(
                f"""
                SELECT
                    COUNT(*) as total_count,
                    COALESCE(AVG(predicted_salary_usd), 0.0) as avg_salary,
                    COALESCE(MIN(predicted_salary_usd), 0.0) as min_salary,
                    COALESCE(MAX(predicted_salary_usd), 0.0) as max_salary,
                    MAX(timestamp) as last_prediction_at
                FROM predictions{country_clause};
                """,
                params,
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
                    "country_breakdown": {"US": 0, "IN": 0},
                }

            avg_salary = round(float(base_row["avg_salary"]), 2)
            min_salary = round(float(base_row["min_salary"]), 2)
            max_salary = round(float(base_row["max_salary"]), 2)
            last_prediction_at = base_row["last_prediction_at"]

            # Most frequent job title
            cursor.execute(
                f"""
                SELECT job_title, COUNT(*) as cnt
                FROM predictions{country_clause}
                GROUP BY job_title
                ORDER BY cnt DESC
                LIMIT 1;
                """,
                params,
            )
            role_row = cursor.fetchone()
            most_frequent_role = role_row["job_title"] if role_row else None

            # Most frequent experience level
            cursor.execute(
                f"""
                SELECT experience_level, COUNT(*) as cnt
                FROM predictions{country_clause}
                GROUP BY experience_level
                ORDER BY cnt DESC
                LIMIT 1;
                """,
                params,
            )
            exp_row = cursor.fetchone()
            top_experience_level = exp_row["experience_level"] if exp_row else None

            # Top locations
            cursor.execute(
                f"""
                SELECT company_location, COUNT(*) as count
                FROM predictions{country_clause}
                GROUP BY company_location
                ORDER BY count DESC
                LIMIT 5;
                """,
                params,
            )
            top_locations = [
                {"location": row["company_location"], "count": int(row["count"])}
                for row in cursor.fetchall()
            ]

            # Remote distribution
            cursor.execute(
                f"""
                SELECT remote_ratio, COUNT(*) as count
                FROM predictions{country_clause}
                GROUP BY remote_ratio
                ORDER BY remote_ratio ASC;
                """,
                params,
            )
            remote_distribution = {
                f"{row['remote_ratio']}%": int(row["count"]) for row in cursor.fetchall()
            }

            # Country breakdown
            cursor.execute(
                """
                SELECT COALESCE(country, 'US') as ccode, COUNT(*) as count
                FROM predictions
                GROUP BY ccode;
                """
            )
            country_breakdown = {row["ccode"]: int(row["count"]) for row in cursor.fetchall()}

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
                "country_breakdown": country_breakdown,
            }

    def _row_to_dict(self, row: sqlite3.Row) -> Dict[str, Any]:
        """Transforms a database row into a structured prediction response object."""
        # Check if row has new migrated columns
        row_keys = row.keys()

        country = row["country"] if "country" in row_keys and row["country"] else "US"
        currency = row["currency"] if "currency" in row_keys and row["currency"] else ("INR" if country == "IN" else "USD")
        salary_unit = row["salary_unit"] if "salary_unit" in row_keys and row["salary_unit"] else ("LPA" if country == "IN" else "YEAR")
        model_used = row["model_used"] if "model_used" in row_keys and row["model_used"] else row["model_version"]
        predicted_salary = row["predicted_salary"] if "predicted_salary" in row_keys and row["predicted_salary"] is not None else float(row["predicted_salary_usd"])

        # Decode input features if available
        input_features = {}
        if "input_features_json" in row_keys and row["input_features_json"]:
            try:
                input_features = json.loads(row["input_features_json"])
            except Exception:
                pass

        if not input_features:
            input_features = {
                "country": country,
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
            "country": country,
            "currency": currency,
            "salary_unit": salary_unit,
            "model_used": str(model_used),
            "predicted_salary": round(float(predicted_salary), 2),
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
