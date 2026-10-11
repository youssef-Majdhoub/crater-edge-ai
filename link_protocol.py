"""
link_protocol.py  --  C.R.A.T.E.R. radio protocol shared by the rover (telemetry_tx.py)
and Mission Control (dashboard.py). Keeping both ends in ONE file means the wire format
can never drift apart.

WHY THIS DESIGN
---------------
A hazard is detected once. If the packet carrying it is lost (30% of the time!) the ground
never learns about it. So we use selective-repeat ARQ:

  rover  --DATA(hazard records)-->  ground       (30% of packets randomly lost)
  rover  <--ACK(what I received)--  ground       (also lossy)

  * The rover keeps every hazard "pending" and re-sends it every RTO until acknowledged.
  * The ground de-duplicates by hazard id, so repeats are harmless.
  * Chance a hazard is still missing after k tries = 0.3^k  (k=5 -> 0.24%), and the ACKs
    stop the repeats, so the average cost is only ~1.4 transmissions per hazard.

PACKET LAYOUT (little-endian, no padding, CRC-8 trailer on every packet)
  DATA  header "<BBBHB"   6 B : magic, kind=1, session, seq, n_records
      rover status "<hhhH" 8 B : x_cm, y_cm, yaw_crad, odom_error_cm (sentinels = unavailable)
        record "<HBhhBBH" 11 B: hazard_id, class_id, x_cm, y_cm, conf(0-255), sigma_dm, age_cs
        crc8              1 B
      -> 1 hazard = 26 B, 10 hazards = 125 B, heartbeat (0 hazards) = 15 B
  ACK   "<BBBHH"          7 B : magic, kind=2, session, base, bitmap, + crc8 = 8 B
        base   = lowest hazard id NOT yet received (everything below it is safe)
        bitmap = bit i set  <=>  hazard id (base + 1 + i) was also received

Design details worth defending to the judges:
  * `age_cs` is "how old was this detection when the packet was SENT" (1/100 s units).
    The ground subtracts it from its own clock, so data-aging stays correct even for
    hazards that arrive late through retransmission, and no clock sync is needed.
  * `session` changes whenever the rover restarts or hazards.json is reset, so the ground
    knows to discard the previous mission instead of mixing ids.
  * Positions are int16 centimetres (range +/-327 m), uncertainty is uint8 decimetres,
    confidence is uint8. No floats on the wire.
  * Hazard ids are 16-bit and assumed not to wrap within one mission (65k hazards).
"""
from __future__ import annotations

from collections import deque
import struct

from reedsolo import RSCodec, ReedSolomonError

MAGIC = 0xC7
KIND_DATA, KIND_ACK, KIND_FEC = 1, 2, 3
HEADER_FMT = "<BBBHB"
RECORD_FMT = "<HBhhBBH"
STATUS_FMT = "<hhhH"
FEC_HEADER_FMT = "<BBBHBBH"
ACK_FMT = "<BBBHH"
HEADER_SIZE = struct.calcsize(HEADER_FMT)
RECORD_SIZE = struct.calcsize(RECORD_FMT)
STATUS_SIZE = struct.calcsize(STATUS_FMT)
FEC_HEADER_SIZE = struct.calcsize(FEC_HEADER_FMT)
ACK_SIZE = struct.calcsize(ACK_FMT) + 1
MAX_RECORDS = 50                      # 6 + 8 + 50*11 + 1 = 565 B, safely below any MTU
FEC_DATA_SHARDS = 7
FEC_PARITY_SHARDS = 3
FEC_TOTAL_SHARDS = FEC_DATA_SHARDS + FEC_PARITY_SHARDS
_RS_CODEC = RSCodec(FEC_PARITY_SHARDS)


def crc8(data: bytes, poly: int = 0x07) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def packet_kind(data: bytes) -> int | None:
    return data[1] if len(data) >= 2 and data[0] == MAGIC else None


# ----------------------------------------------------------------------------- DATA
def pack_data(session: int, seq: int, records: list[dict], rover_status: dict | None = None) -> bytes:
    """Pack hazard records and optional rover x/y/yaw/odometry-error status."""
    recs = records[:MAX_RECORDS]
    body = struct.pack(HEADER_FMT, MAGIC, KIND_DATA, session & 0xFF, seq & 0xFFFF, len(recs))
    if rover_status is None:
        body += struct.pack(STATUS_FMT, -32768, -32768, -32768, 0xFFFF)
    else:
        body += struct.pack(
            STATUS_FMT,
            _clamp(round(rover_status["x"] * 100), -32767, 32767),
            _clamp(round(rover_status["y"] * 100), -32767, 32767),
            _clamp(round(rover_status["yaw"] * 100), -32767, 32767),
            _clamp(round(rover_status["error"] * 100), 0, 65534),
        )
    for r in recs:
        body += struct.pack(
            RECORD_FMT,
            r["hid"] & 0xFFFF,
            _clamp(int(r["class_id"]), 0, 255),
            _clamp(round(r["x"] * 100), -32768, 32767),
            _clamp(round(r["y"] * 100), -32768, 32767),
            _clamp(round(r["conf"] * 255), 0, 255),
            _clamp(round(r["sigma"] * 10), 0, 255),
            _clamp(round(r["age_s"] * 100), 0, 65535),
        )
    return body + bytes([crc8(body)])


def unpack_data(data: bytes) -> dict | None:
    """Returns None for anything corrupt, truncated or not a DATA packet."""
    if len(data) < HEADER_SIZE + STATUS_SIZE + 1 or crc8(data[:-1]) != data[-1]:
        return None
    magic, kind, session, seq, n = struct.unpack_from(HEADER_FMT, data, 0)
    expected_size = HEADER_SIZE + STATUS_SIZE + n * RECORD_SIZE + 1
    if magic != MAGIC or kind != KIND_DATA or len(data) != expected_size:
        return None
    x, y, yaw, error = struct.unpack_from(STATUS_FMT, data, HEADER_SIZE)
    status = None if x == y == yaw == -32768 and error == 0xFFFF else {
        "x": None if x == -32768 else x / 100,
        "y": None if y == -32768 else y / 100,
        "yaw": None if yaw == -32768 else yaw / 100,
        "error": None if error == 0xFFFF else error / 100,
    }
    records = []
    for i in range(n):
        hid, cid, hazard_x, hazard_y, conf, sig, age = struct.unpack_from(
            RECORD_FMT, data, HEADER_SIZE + STATUS_SIZE + i * RECORD_SIZE
        )
        records.append({"hid": hid, "class_id": cid, "x": hazard_x / 100, "y": hazard_y / 100,
                        "conf": conf / 255, "sigma": sig / 10, "age_s": age / 100})
    return {"session": session, "seq": seq, "records": records, "rover_status": status}


# ----------------------------------------------------------------------------- Reed-Solomon erasure coding
def encode_fec(data: bytes, session: int, block_id: int) -> list[bytes]:
    """Split one DATA packet into 7 data + 3 byte-wise Reed-Solomon shards."""
    shard_size = (len(data) + FEC_DATA_SHARDS - 1) // FEC_DATA_SHARDS
    padded = data.ljust(shard_size * FEC_DATA_SHARDS, b"\0")
    data_shards = [
        padded[i * shard_size:(i + 1) * shard_size]
        for i in range(FEC_DATA_SHARDS)
    ]
    shards = [bytearray(shard_size) for _ in range(FEC_TOTAL_SHARDS)]

    for offset in range(shard_size):
        codeword = bytes(shard[offset] for shard in data_shards)
        encoded = _RS_CODEC.encode(codeword)
        for shard_index, value in enumerate(encoded):
            shards[shard_index][offset] = value

    packets = []
    for shard_index, shard in enumerate(shards):
        body = struct.pack(
            FEC_HEADER_FMT, MAGIC, KIND_FEC, session & 0xFF, block_id & 0xFFFF,
            shard_index, FEC_DATA_SHARDS, len(data),
        ) + bytes(shard)
        packets.append(body + bytes([crc8(body)]))
    return packets


class FECReassembler:
    """Reconstruct DATA packets after losing up to three of ten RS shard packets."""

    def __init__(self) -> None:
        self._blocks: dict[tuple[int, int], dict] = {}
        self._completed: set[tuple[int, int]] = set()
        self._completed_order: deque[tuple[int, int]] = deque()
        self.recovered_blocks = 0

    def ingest(self, packet: bytes) -> bytes | None:
        if len(packet) < FEC_HEADER_SIZE + 1 or crc8(packet[:-1]) != packet[-1]:
            return None
        magic, kind, session, block_id, shard_index, data_shards, original_len = struct.unpack_from(
            FEC_HEADER_FMT, packet, 0
        )
        if (magic != MAGIC or kind != KIND_FEC or data_shards != FEC_DATA_SHARDS
                or shard_index >= FEC_TOTAL_SHARDS):
            return None

        payload = packet[FEC_HEADER_SIZE:-1]
        key = (session, block_id)
        if key in self._completed:
            return None
        block = self._blocks.get(key)
        if block is None:
            if len(self._blocks) >= 128:
                self._blocks.pop(next(iter(self._blocks)))
            block = {"length": original_len, "shard_size": len(payload), "shards": {}}
            self._blocks[key] = block
        if block["length"] != original_len or block["shard_size"] != len(payload):
            self._blocks.pop(key, None)
            return None
        block["shards"].setdefault(shard_index, payload)
        present = block["shards"]
        if len(present) < FEC_DATA_SHARDS:
            return None

        missing = [i for i in range(FEC_TOTAL_SHARDS) if i not in present]
        data_parts = [bytearray(block["shard_size"]) for _ in range(FEC_DATA_SHARDS)]
        try:
            for offset in range(block["shard_size"]):
                codeword = bytearray(FEC_TOTAL_SHARDS)
                for index, shard in present.items():
                    codeword[index] = shard[offset]
                decoded, _, _ = _RS_CODEC.decode(codeword, erase_pos=missing)
                for index in range(FEC_DATA_SHARDS):
                    data_parts[index][offset] = decoded[index]
        except (ReedSolomonError, ValueError, IndexError):
            return None

        result = b"".join(data_parts)[:block["length"]]
        self._blocks.pop(key, None)
        self._completed.add(key)
        self._completed_order.append(key)
        if len(self._completed_order) > 256:
            self._completed.discard(self._completed_order.popleft())
        if missing:
            self.recovered_blocks += 1
        return result


# ----------------------------------------------------------------------------- ACK
def pack_ack(session: int, base: int, bitmap: int) -> bytes:
    body = struct.pack(ACK_FMT, MAGIC, KIND_ACK, session & 0xFF, base & 0xFFFF, bitmap & 0xFFFF)
    return body + bytes([crc8(body)])


def unpack_ack(data: bytes) -> dict | None:
    if len(data) != ACK_SIZE or crc8(data[:-1]) != data[-1]:
        return None
    magic, kind, session, base, bitmap = struct.unpack_from(ACK_FMT, data, 0)
    if magic != MAGIC or kind != KIND_ACK:
        return None
    return {"session": session, "base": base, "bitmap": bitmap}


# ----------------------------------------------------------------------------- GROUND SIDE
class Receiver:
    """Ground-station protocol logic: de-duplicate, track loss, build ACKs. No sockets here."""

    def __init__(self) -> None:
        self._reset(None)

    def _reset(self, session: int | None) -> None:
        self.session = session
        self.received: set[int] = set()
        self.base = 0                       # lowest hazard id not yet received
        self.hazards: dict[int, dict] = {}  # hid -> latest record
        self.last_seq: int | None = None
        self.rover_status: dict | None = None
        self.n_packets = self.n_lost = self.n_bad = self.n_dupes = 0
        self.rx_bytes = 0

    def ingest(self, data: bytes, now: float) -> bytes | None:
        """Feed one raw packet. Returns the ACK bytes to send back (or None if the packet was bad)."""
        pkt = unpack_data(data)
        if pkt is None:
            self.n_bad += 1
            return None
        if pkt["session"] != self.session:          # rover restarted / new mission
            self._reset(pkt["session"])

        if self.last_seq is not None:               # packet loss measured from sequence gaps
            gap = (pkt["seq"] - self.last_seq) & 0xFFFF
            if 1 < gap < 1000:
                self.n_lost += gap - 1
        self.last_seq = pkt["seq"]
        self.n_packets += 1
        self.rx_bytes += len(data)
        if pkt["rover_status"] is not None:
            self.rover_status = pkt["rover_status"]

        for r in pkt["records"]:
            hid = r["hid"]
            if hid in self.received:
                self.n_dupes += 1                   # a retransmission we already had
                continue
            self.received.add(hid)
            self.hazards[hid] = {**r, "t_det": now - r["age_s"], "t_rx": now}

        while self.base in self.received:
            self.base += 1
        bitmap = 0
        for i in range(16):
            if (self.base + 1 + i) in self.received:
                bitmap |= 1 << i
        return pack_ack(self.session, self.base, bitmap)
