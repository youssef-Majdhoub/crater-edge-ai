#!/usr/bin/env python3
"""
telemetry_tx.py  --  C.R.A.T.E.R. rover-side radio (reads the REAL hazards.json from edge_detector.py)
========================================================================================================

    edge_detector.py --appends--> hazards.json --poll--> [this script] --UDP :9002--> dashboard.py
                                                           ^   (30% random loss)         |
                                                           +------ UDP :9003  ACKs -----+

Each hazard becomes one 11-byte record. A hazard stays "pending" and is re-sent every RTO_S
seconds until the ground station acknowledges it (see link_protocol.py for the why).
A tiny heartbeat packet goes out when idle so the ground can measure link quality and
declare loss-of-signal.

Run:   python telemetry_tx.py
Env:   CRATER_LOSS_RATE=0.5   simulate a worse link (default 0.30)
       CRATER_HAZARDS=path    use another hazards file (default hazards.json)
"""
from __future__ import annotations

import json
import os
import random
import re
import socket
import time

import link_protocol as lp

HAZARDS_JSON = os.environ.get("CRATER_HAZARDS", "hazards.json")
ROVER_STATUS_JSON = os.environ.get("CRATER_ROVER_STATUS", "rover_status.json")
LINK_ADDR = ("127.0.0.1", 9002)      # rover -> ground ("radio")
ACK_ADDR = ("127.0.0.1", 9003)       # ground -> rover (this script listens here)
LOSS_RATE = float(os.environ.get("CRATER_LOSS_RATE", "0.30"))   # <<< the required 30% packet loss
TICK_S = 0.1                         # main-loop period (how fast a new hazard leaves the rover)
HEARTBEAT_S = 1.0                    # idle packet interval
RTO_S = 1.0                          # first re-send of an un-acked hazard after this long ...
RTO_MAX_S = 8.0                      # ... then 2s, 4s, 8s, 8s ... (backoff: no retransmission storm on a dead link)
STATS_EVERY_S = 5.0

_rng = random.Random(os.environ.get("CRATER_SEED"))       # unset -> truly random drops
_NUM = re.compile(r"(\d+)")


def parse_entry(entry: dict, hid: int, first_seen: float) -> dict | None:
    """hazards.json entry -> internal record. 'type' is 'class_5' -> class_id 5."""
    try:
        m = _NUM.search(str(entry.get("type", "")))
        return {
            "hid": hid,
            "class_id": int(m.group(1)) if m else 255,
            "x": float(entry["x"]),
            "y": float(entry["y"]),
            "conf": float(entry.get("confidence", 0.0)),
            "sigma": float(entry.get("sigma", 0.5)),      # 1-sigma position error in metres
            "t": float(entry.get("t", first_seen)),       # detection time (epoch s)
            "last_tx": float("-inf"),
            "tries": 0,
        }
    except (KeyError, TypeError, ValueError):
        return None                                        # malformed entry: skip it


class Sender:
    def __init__(self) -> None:
        self.session = _rng.randrange(1, 256)
        self.seq = 0
        self.n_seen = 0                                    # entries of hazards.json already consumed
        self.file_sig: tuple | None = None
        self.status_sig: tuple | None = None
        self.rover_status: dict | None = None
        self.pending: dict[int, dict] = {}                 # hid -> record awaiting ACK
        self.last_pkt = 0.0
        # statistics
        self.pkts = self.dropped = self.wire_bytes = self.data_bytes = 0
        self.hazards = self.acked = self.retx = self.json_bytes = 0

    # ---- input: hazards.json -------------------------------------------------
    def poll_file(self) -> None:
        try:
            st = os.stat(HAZARDS_JSON)
        except OSError:
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self.file_sig:
            return
        try:
            with open(HAZARDS_JSON, "r") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return                                         # being written right now: retry next tick
        if not isinstance(data, list):
            return
        self.file_sig = sig

        if len(data) < self.n_seen:                        # file was reset -> new mission
            self.session = self.session % 255 + 1
            self.seq = 0
            self.pending.clear()
            self.n_seen = 0
            print(f"[tx] hazards file reset -> new session {self.session}", flush=True)

        now = time.time()
        for idx in range(self.n_seen, len(data)):
            rec = parse_entry(data[idx], idx, now)
            if rec:
                self.pending[idx] = rec
                self.hazards += 1
                self.json_bytes += len(json.dumps(data[idx]))
        self.n_seen = len(data)

    def poll_rover_status(self) -> None:
        try:
            st = os.stat(ROVER_STATUS_JSON)
        except OSError:
            self.rover_status = None
            self.status_sig = None
            return
        sig = (st.st_mtime_ns, st.st_size)
        if sig == self.status_sig:
            return
        try:
            with open(ROVER_STATUS_JSON, "r") as f:
                status = json.load(f)
            self.rover_status = {
                "x": float(status["x"]),
                "y": float(status["y"]),
                "yaw": float(status["yaw"]),
                "error": float(status["error"]),
            }
            self.status_sig = sig
        except (OSError, ValueError, KeyError, TypeError):
            return                                           # retry if written or malformed

    # ---- input: ACKs from the ground ---------------------------------------
    def handle_acks(self, sock: socket.socket) -> None:
        while True:
            try:
                raw, _ = sock.recvfrom(2048)
            except (BlockingIOError, ConnectionResetError):
                break
            ack = lp.unpack_ack(raw)
            if ack is None or ack["session"] != self.session:
                continue
            base = ack["base"]
            for hid in [h for h in self.pending if h < base]:
                del self.pending[hid]
                self.acked += 1
            for i in range(16):
                if ack["bitmap"] >> i & 1 and self.pending.pop(base + 1 + i, None) is not None:
                    self.acked += 1

    # ---- output: DATA packets ------------------------------------------------
    def maybe_send(self, radio: socket.socket) -> None:
        now = time.time()
        due = sorted(
            h for h, p in self.pending.items()
            if now - p["last_tx"] >= min(RTO_S * 2 ** max(p["tries"] - 1, 0), RTO_MAX_S)   # exponential backoff
        )[:lp.MAX_RECORDS]
        if not due and now - self.last_pkt < HEARTBEAT_S:
            return
        records = [{**self.pending[h], "age_s": now - self.pending[h]["t"]} for h in due]
        block_id = self.seq
        pkt = lp.pack_data(self.session, block_id, records, self.rover_status)
        shards = lp.encode_fec(pkt, self.session, block_id)
        self.seq = (self.seq + 1) & 0xFFFF                 # advances before transmission
        for h in due:
            if self.pending[h]["tries"] > 0:
                self.retx += 1
            self.pending[h]["tries"] += 1
            self.pending[h]["last_tx"] = now
        self.last_pkt = now

        if records:
            self.data_bytes += sum(map(len, shards))
        for shard in shards:
            self.pkts += 1
            self.wire_bytes += len(shard)
            # Simulate packet loss; RS repairs up to three missing shards per block.
            if _rng.random() < LOSS_RATE:
                self.dropped += 1
                continue
            try:
                radio.sendto(shard, LINK_ADDR)
            except OSError:
                pass

    def stats_line(self) -> str:
        loss = 100.0 * self.dropped / self.pkts if self.pkts else 0.0
        ratio = f"{self.json_bytes / self.data_bytes:.1f}x smaller than JSON" if self.data_bytes else "n/a"
        return (f"[tx] hazards={self.hazards} acked={self.acked} pending={len(self.pending)} "
                f"retransmissions={self.retx} | packets={self.pkts} dropped={self.dropped} ({loss:.0f}%) "
                f"| {ratio} (incl. retransmissions)")


def main() -> None:
    ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        ack_sock.bind(ACK_ADDR)
    except OSError as e:
        raise SystemExit(f"[tx] cannot bind ACK port {ACK_ADDR}: {e}  (is another telemetry_tx.py running?)")
    ack_sock.setblocking(False)
    radio = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    radio.setblocking(False)
    for s in (ack_sock, radio):
        if hasattr(socket, "SIO_UDP_CONNRESET"):           # Windows: ignore ICMP "port unreachable"
            s.ioctl(socket.SIO_UDP_CONNRESET, False)

    tx = Sender()
    print(f"[tx] watching {HAZARDS_JSON} -> {LINK_ADDR}, simulated loss {LOSS_RATE:.0%}, session {tx.session}",
          flush=True)
    last_stats = time.monotonic()
    while True:
        tx.poll_file()
        tx.poll_rover_status()
        tx.handle_acks(ack_sock)
        tx.maybe_send(radio)
        if time.monotonic() - last_stats >= STATS_EVERY_S:
            if tx.pkts:
                print(tx.stats_line(), flush=True)
            last_stats = time.monotonic()
        time.sleep(TICK_S)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("[tx] stopped")
