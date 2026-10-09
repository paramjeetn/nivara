"""SQLite repository for telemetry and events."""
import sqlite3
from typing import List, Dict, Any, Optional
from pathlib import Path
from pydantic import BaseModel

class Repository:
    def __init__(self, db_path: str = "data/campus.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
        
    def init_db(self):
        with self.get_connection() as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS telemetry (
                    reading_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    asset_id TEXT,
                    reading_type TEXT,
                    power_kw REAL,
                    interval_energy_kwh REAL,
                    cumulative_energy_kwh REAL,
                    voltage_v REAL,
                    current_a REAL,
                    power_factor REAL,
                    state TEXT,
                    flow_lpm REAL,
                    temperature_c REAL,
                    setpoint_c REAL,
                    sensor_status TEXT,
                    quality_flag TEXT
                )
            ''')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_asset_time ON telemetry(asset_id, timestamp)')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_telemetry_time ON telemetry(timestamp)')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS alerts (
                    alert_id TEXT PRIMARY KEY,
                    detected_at TEXT,
                    event_start_estimate TEXT,
                    asset_id TEXT,
                    building_id TEXT,
                    zone_id TEXT,
                    anomaly_type TEXT,
                    severity TEXT,
                    actual_value REAL,
                    expected_value REAL,
                    deviation REAL,
                    evidence TEXT,
                    confidence REAL,
                    recommendation TEXT,
                    status TEXT
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS replay_state (
                    id INTEGER PRIMARY KEY CHECK (id = 1),
                    current_time TEXT,
                    status TEXT,
                    speed INTEGER
                )
            ''')

    def insert_telemetry_batch(self, records: List[Dict[str, Any]]):
        """Insert a batch of telemetry records. Ignores duplicates by reading_id."""
        if not records:
            return
            
        columns = list(records[0].keys())
        placeholders = ",".join(["?"] * len(columns))
        sql = f"INSERT OR IGNORE INTO telemetry ({','.join(columns)}) VALUES ({placeholders})"
        
        values = [[record.get(col) for col in columns] for record in records]
        
        with self.get_connection() as conn:
            conn.executemany(sql, values)

    def query_telemetry(self, asset_id: str, start_time: str, end_time: str) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(
                "SELECT * FROM telemetry WHERE asset_id = ? AND timestamp >= ? AND timestamp <= ? ORDER BY timestamp",
                (asset_id, start_time, end_time)
            )
            return [dict(row) for row in cursor.fetchall()]
            
    def insert_alert(self, alert: Dict[str, Any]):
        columns = list(alert.keys())
        placeholders = ",".join(["?"] * len(columns))
        sql = f"INSERT OR REPLACE INTO alerts ({','.join(columns)}) VALUES ({placeholders})"
        
        with self.get_connection() as conn:
            conn.execute(sql, [alert.get(col) for col in columns])
            
    def get_alerts(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            if status:
                cursor = conn.execute("SELECT * FROM alerts WHERE status = ? ORDER BY detected_at DESC", (status,))
            else:
                cursor = conn.execute("SELECT * FROM alerts ORDER BY detected_at DESC")
            return [dict(row) for row in cursor.fetchall()]
