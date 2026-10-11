"""SQLite persistence for telemetry packets accepted by the ground station."""
from __future__ import annotations

import sqlite3
import threading
from pathlib import Path
from typing import Any


SCHEMA = """
CREATE TABLE IF NOT EXISTS packets (
    id INTEGER PRIMARY KEY,
    received_at REAL NOT NULL,
    session INTEGER NOT NULL,
    sequence INTEGER NOT NULL,
    packet_size INTEGER NOT NULL,
    raw_packet BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS hazards (
    id INTEGER PRIMARY KEY,
    packet_id INTEGER NOT NULL REFERENCES packets(id),
    hazard_id INTEGER NOT NULL,
    class_id INTEGER NOT NULL,
    x REAL NOT NULL,
    y REAL NOT NULL,
    confidence REAL NOT NULL,
    sigma REAL NOT NULL,
    age_seconds REAL NOT NULL,
    detected_at REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS rover_status (
    id INTEGER PRIMARY KEY,
    packet_id INTEGER NOT NULL REFERENCES packets(id),
    x REAL,
    y REAL,
    yaw REAL,
    odometry_error REAL
);

CREATE INDEX IF NOT EXISTS idx_packets_session_sequence
    ON packets(session, sequence);
CREATE INDEX IF NOT EXISTS idx_hazards_hazard_id
    ON hazards(hazard_id);
"""


class TelemetryDatabase:
    """Thread-safe SQLite writer for reconstructed ground-station packets."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.executescript(SCHEMA)
        self._connection.commit()

    def save_packet(self, packet: dict[str, Any], raw_packet: bytes, received_at: float) -> None:
        """Save one valid DATA packet and all values decoded from it."""
        with self._lock:
            try:
                cursor = self._connection.execute(
                    """
                    INSERT INTO packets
                        (received_at, session, sequence, packet_size, raw_packet)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        received_at,
                        packet["session"],
                        packet["seq"],
                        len(raw_packet),
                        raw_packet,
                    ),
                )
                packet_id = cursor.lastrowid
                for hazard in packet["records"]:
                    self._connection.execute(
                        """
                        INSERT INTO hazards
                            (packet_id, hazard_id, class_id, x, y, confidence,
                             sigma, age_seconds, detected_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            packet_id,
                            hazard["hid"],
                            hazard["class_id"],
                            hazard["x"],
                            hazard["y"],
                            hazard["conf"],
                            hazard["sigma"],
                            hazard["age_s"],
                            received_at - hazard["age_s"],
                        ),
                    )

                status = packet["rover_status"]
                if status is not None:
                    self._connection.execute(
                        """
                        INSERT INTO rover_status
                            (packet_id, x, y, yaw, odometry_error)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (
                            packet_id,
                            status["x"],
                            status["y"],
                            status["yaw"],
                            status["error"],
                        ),
                    )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._connection.close()
