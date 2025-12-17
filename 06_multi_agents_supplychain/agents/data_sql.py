# agents/data_sql.py
from __future__ import annotations
import sqlite3
from typing import Any, Dict, List, Tuple
from tabulate import tabulate

DB_PATH = "data/supplychain.db"

class DataAgent:
    def query(self, user_text: str) -> str:
        text = user_text.upper()

        if "INVENTORY" in text or "ON HAND" in text or "STOCK" in text:
            sku = _extract_sku(text)
            dc = _extract_dc(text)
            return self._inventory(sku=sku, dc=dc)

        if "SHIPMENT" in text or "ASN" in text or "ETA" in text or "DELAY" in text:
            sku = _extract_sku(text)
            return self._shipments(sku=sku)

        if "PO" in text or "PURCHASE ORDER" in text or "ORDER" in text:
            sku = _extract_sku(text)
            return self._orders(sku=sku)

        return "Data agent: I can answer about inventory, shipments, and POs. Try: 'Inventory for SKU-100 in DC East?'"

    def _inventory(self, sku: str | None, dc: str | None) -> str:
        q = "SELECT sku, dc, on_hand, updated_at FROM inventory"
        params: List[Any] = []
        clauses: List[str] = []
        if sku:
            clauses.append("sku = ?")
            params.append(sku)
        if dc:
            clauses.append("dc = ?")
            params.append(dc)
        if clauses:
            q += " WHERE " + " AND ".join(clauses)
        q += " ORDER BY sku, dc"

        rows = _fetch(q, params)
        if not rows:
            return "No inventory rows found for that filter."
        return tabulate(rows, headers=["sku", "dc", "on_hand", "updated_at"], tablefmt="github")

    def _shipments(self, sku: str | None) -> str:
        q = "SELECT shipment_id, sku, supplier, eta_date, status, region FROM shipments"
        params: List[Any] = []
        if sku:
            q += " WHERE sku = ?"
            params.append(sku)
        q += " ORDER BY eta_date"
        rows = _fetch(q, params)
        if not rows:
            return "No shipments found."
        return tabulate(rows, headers=["shipment_id", "sku", "supplier", "eta_date", "status", "region"], tablefmt="github")

    def _orders(self, sku: str | None) -> str:
        q = "SELECT po_id, sku, qty, supplier, created_at, status FROM orders"
        params: List[Any] = []
        if sku:
            q += " WHERE sku = ?"
            params.append(sku)
        q += " ORDER BY created_at DESC"
        rows = _fetch(q, params)
        if not rows:
            return "No orders found."
        return tabulate(rows, headers=["po_id", "sku", "qty", "supplier", "created_at", "status"], tablefmt="github")

def _fetch(query: str, params: List[Any]) -> List[Tuple[Any, ...]]:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(query, params)
    rows = cur.fetchall()
    conn.close()
    return rows

def _extract_sku(text: str) -> str | None:
    # naive: find token like SKU-123
    for token in text.replace("?", " ").replace(",", " ").split():
        if token.startswith("SKU-"):
            return token
    return None

def _extract_dc(text: str) -> str | None:
    if "EAST" in text:
        return "DC_EAST"
    if "WEST" in text:
        return "DC_WEST"
    return None