# data/init_db.py
from __future__ import annotations
import sqlite3
from pathlib import Path

DB_PATH = Path("data/supplychain.db")
DB_PATH.parent.mkdir(exist_ok=True)

def main():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
    DROP TABLE IF EXISTS inventory;
    DROP TABLE IF EXISTS shipments;
    DROP TABLE IF EXISTS orders;

    CREATE TABLE inventory (
      sku TEXT,
      dc  TEXT,
      on_hand INTEGER,
      updated_at TEXT
    );

    CREATE TABLE shipments (
      shipment_id TEXT,
      sku TEXT,
      supplier TEXT,
      eta_date TEXT,
      status TEXT,
      region TEXT
    );

    CREATE TABLE orders (
      po_id TEXT,
      sku TEXT,
      qty INTEGER,
      supplier TEXT,
      created_at TEXT,
      status TEXT
    );
    """)

    cur.executemany(
        "INSERT INTO inventory (sku, dc, on_hand, updated_at) VALUES (?,?,?,datetime('now'))",
        [
            ("SKU-100", "DC_EAST", 120, ),
            ("SKU-100", "DC_WEST", 30, ),
            ("SKU-200", "DC_EAST", 0, ),
            ("SKU-200", "DC_WEST", 15, ),
        ],
    )

    cur.executemany(
        "INSERT INTO shipments (shipment_id, sku, supplier, eta_date, status, region) VALUES (?,?,?,?,?,?)",
        [
            ("SHP-001", "SKU-200", "SupplierB", "2025-12-20", "DELAYED", "US"),
            ("SHP-002", "SKU-100", "SupplierA", "2025-12-18", "IN_TRANSIT", "US"),
        ],
    )

    cur.executemany(
        "INSERT INTO orders (po_id, sku, qty, supplier, created_at, status) VALUES (?,?,?,?,datetime('now'),?)",
        [
            ("PO-9001", "SKU-200", 200, "SupplierB", "OPEN"),
            ("PO-9002", "SKU-100", 100, "SupplierA", "OPEN"),
        ],
    )

    conn.commit()
    conn.close()
    print(f"Initialized {DB_PATH}")

if __name__ == "__main__":
    main()