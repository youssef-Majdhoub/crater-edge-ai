"""
dashboard.py  --  C.R.A.T.E.R. Mission Control (ground station) for the REAL telemetry link
===========================================================================================
Run with:  python -m streamlit run dashboard.py

  * A background THREAD owns the UDP socket (:9002), de-duplicates hazards, and sends ACKs back
    (:9003) - the UI never blocks.
  * The UI is a Streamlit fragment that re-renders twice a second from the shared state.
  * DATA AGING  : opacity fades with the age of the DETECTION (not of the last packet), so a hazard
                  that arrived late through a retransmission is still aged correctly.
  * UNCERTAINTY : a 1-sigma circle per hazard = sensor sigma + growth with age.
  * PERSISTENCE : every reconstructed DATA packet and its decoded values are saved to SQLite.
  * LINK HEALTH : loss measured from sequence-number gaps, duplicates discarded = proof that
                  retransmission is working, LOSS OF SIGNAL banner.

Needs streamlit >= 1.37, plotly, pandas.
"""
from __future__ import annotations

import math
import os
import random
import socket
import threading
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import link_protocol as lp
from telemetry_database import TelemetryDatabase

# =============================================================================
# CONFIG
# =============================================================================
LINK_ADDR = ("127.0.0.1", 9002)                       # data in
ACK_ADDR = ("127.0.0.1", 9003)                        # ACKs out
ACK_LOSS = float(os.environ.get("CRATER_LOSS_RATE", "0.30"))   # the return link is lossy too
REFRESH_S = 0.5
LOS_AFTER_S = 3.0                                     # no packet this long -> LOSS OF SIGNAL
DATABASE_PATH = os.environ.get("CRATER_DB", "crater_telemetry.db")

# Class names for YOUR trained model (ids 0,1,2). Anything else is shown as "class_<id>"
# (e.g. while you still use the COCO yolov8n weights).
CLASS_NAME = {0: "rock", 1: "crater", 2: "ice_deposit"}
CLASS_STYLE = {
    "rock": ("#e67e22", "square"),
    "crater": ("#b07cff", "diamond"),
    "ice_deposit": ("#2ec4e6", "star"),
}
FALLBACK_COLORS = ["#ff6b6b", "#7bd88f", "#ffd166", "#9ad1ff", "#f78fb3", "#c7f464"]


def style_for(label: str) -> tuple[str, str]:
    if label in CLASS_STYLE:
        return CLASS_STYLE[label]
    return FALLBACK_COLORS[hash(label) % len(FALLBACK_COLORS)], "circle"


# =============================================================================
# SHARED STATE + BACKGROUND LISTENER
# =============================================================================
class LinkState:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.rx = lp.Receiver()
        self.fec = lp.FECReassembler()
        self.database = TelemetryDatabase(DATABASE_PATH)
        self.last_rx: float | None = None
        self.error: str | None = None

    def ingest(self, data: bytes) -> bytes | None:
        now = time.time()
        with self.lock:
            packet = self.fec.ingest(data)
            if packet is None:
                return None
            decoded = lp.unpack_data(packet)
            if decoded is None:
                return None
            ack = self.rx.ingest(packet, now)
            if ack is not None:
                try:
                    self.database.save_packet(decoded, packet, now)
                except Exception as exc:
                    self.error = f"Cannot save telemetry to {DATABASE_PATH}: {exc}"
                    return None
                self.last_rx = now
        return ack

    def snapshot(self) -> dict:
        with self.lock:
            r = self.rx
            hazards = [{
                "id": h["hid"],
                "label": CLASS_NAME.get(h["class_id"], f"class_{h['class_id']}"),
                "x": h["x"], "y": h["y"], "sigma": h["sigma"], "conf": h["conf"],
                "t_det": h["t_det"],                  # when it was DETECTED (ground clock)
            } for h in r.hazards.values()]
            return {"hazards": hazards, "rx": r.n_packets, "lost": r.n_lost, "bad": r.n_bad,
                    "dupes": r.n_dupes, "bytes": r.rx_bytes, "last_rx": self.last_rx,
                    "rover_status": r.rover_status, "rs_recovered": self.fec.recovered_blocks,
                    "error": self.error}


def _listener(state: LinkState) -> None:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(LINK_ADDR)
    except OSError as e:
        state.error = f"Cannot bind UDP {LINK_ADDR}: {e}"
        return
    sock.settimeout(0.5)
    ack_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for s in (sock, ack_sock):
        if hasattr(socket, "SIO_UDP_CONNRESET"):      # Windows ICMP quirk
            s.ioctl(socket.SIO_UDP_CONNRESET, False)
    rng = random.Random()
    while True:
        try:
            data, _ = sock.recvfrom(65535)
        except (socket.timeout, ConnectionResetError):
            continue
        ack = state.ingest(data)
        if ack is not None and rng.random() >= ACK_LOSS:   # simulate loss on the return link too
            try:
                ack_sock.sendto(ack, ACK_ADDR)
            except OSError:
                pass


@st.cache_resource
def start_listener() -> LinkState:
    state = LinkState()
    threading.Thread(target=_listener, args=(state,), daemon=True).start()
    return state


# =============================================================================
# UI
# =============================================================================
st.set_page_config(page_title="C.R.A.T.E.R. Mission Control", layout="wide")
state = start_listener()

st.title("C.R.A.T.E.R. - Mission Control")
st.caption("Constrained Reconnaissance, Autonomy & Telemetry in Extreme Regions  |  "
           "opacity = data freshness  |  circle = 1-sigma position uncertainty (grows with age)")
st.caption(f"Telemetry history is saved in `{DATABASE_PATH}`.")

with st.sidebar:
    st.header("Display settings")
    fade_tau = st.slider("Fade time-constant (s)", 5, 300, 60)
    stale_after = st.slider("Mark STALE after (s)", 5, 600, 120)
    drift_rate = st.slider("Uncertainty growth with age (m/s)", 0.0, 0.2, 0.01, 0.005)
    show_unc = st.checkbox("Show uncertainty circles", value=True)


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{max(0.0, min(1.0, alpha)):.2f})"


def _show(fig: go.Figure) -> None:
    try:
        st.plotly_chart(fig, width="stretch", key="map")
    except Exception:                                  # older Streamlit
        st.plotly_chart(fig, use_container_width=True, key="map")


@st.fragment(run_every=REFRESH_S)
def mission_view() -> None:
    now = time.time()
    snap = state.snapshot()
    if snap["error"]:
        st.error(snap["error"])
        return
    if snap["last_rx"] is None:
        st.info(f"Waiting for the first packet on UDP {LINK_ADDR[1]} ... (start telemetry_tx.py)")
        return

    since = now - snap["last_rx"]
    total = snap["rx"] + snap["lost"]
    loss_pct = 100.0 * snap["lost"] / total if total else 0.0
    if since > LOS_AFTER_S:
        st.error(f"LOSS OF SIGNAL - last contact {since:.0f} s ago. Showing last known data.")
    c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
    c1.metric("Packets received", snap["rx"])
    c2.metric("Measured link loss", f"{loss_pct:.0f} %")
    c3.metric("Last contact", f"{since:.1f} s ago")
    c4.metric("Avg packet size", f"{snap['bytes'] / max(snap['rx'], 1):.0f} B")
    c5.metric("Hazards received", len(snap["hazards"]))
    c6.metric("Retransmits discarded", snap["dupes"])
    c7.metric("RS-recovered blocks", snap["rs_recovered"])

    rover = snap["rover_status"]
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Rover X (m)", f"{rover['x']:.2f}" if rover and rover["x"] is not None else "Unavailable")
    r2.metric("Rover Y (m)", f"{rover['y']:.2f}" if rover and rover["y"] is not None else "Unavailable")
    r3.metric("Rover yaw (rad)", f"{rover['yaw']:.2f}" if rover and rover["yaw"] is not None else "Unavailable")
    r4.metric("Odometry error (m)", f"{rover['error']:.2f}" if rover and rover["error"] is not None else "Unavailable")

    fig = go.Figure()
    shapes = []
    rows = []
    extent = max(
        (max(abs(h["x"]), abs(h["y"])) + 4 for h in snap["hazards"]),
        default=1.0,
    )
    for cls in sorted({h["label"] for h in snap["hazards"]}):
        color, symbol = style_for(cls)
        xs, ys, ops, hover = [], [], [], []
        for h in (h for h in snap["hazards"] if h["label"] == cls):
            age = now - h["t_det"]
            opac = max(0.12, math.exp(-age / fade_tau))                 # DATA AGING
            sig_eff = h["sigma"] + drift_rate * age                     # UNCERTAINTY (+ growth)
            stale = age > stale_after
            xs.append(h["x"]); ys.append(h["y"]); ops.append(opac)
            hover.append(f"{cls} #{h['id']}<br>({h['x']:.1f}, {h['y']:.1f}) m"
                         f"<br>age {age:.1f}s | sigma {sig_eff:.2f} m | conf {h['conf']:.0%}")
            if show_unc:
                shapes.append(dict(type="circle", xref="x", yref="y",
                                   x0=h["x"] - sig_eff, x1=h["x"] + sig_eff,
                                   y0=h["y"] - sig_eff, y1=h["y"] + sig_eff,
                                   fillcolor=_rgba(color, 0.18 * opac),
                                   line=dict(color=_rgba(color, 0.7 * opac), width=1,
                                             dash="dot" if stale else "solid")))
            rows.append({"ID": h["id"], "Label": cls, "X (m)": round(h["x"], 2), "Y (m)": round(h["y"], 2),
                         "Sigma (m)": round(sig_eff, 2), "Age (s)": round(age, 1),
                         "Conf": f"{h['conf']:.0%}", "Status": "STALE" if stale else "FRESH"})
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name=cls, text=hover, hoverinfo="text",
                                 marker=dict(symbol=symbol, size=14, color=color, opacity=ops,
                                             line=dict(width=1, color="white"))))

    fig.update_layout(template="plotly_dark", height=640, shapes=shapes,
                      margin=dict(l=10, r=10, t=10, b=10), legend=dict(orientation="h", y=1.04),
                      xaxis=dict(range=[-extent, extent], dtick=5, title="X (m)", gridcolor="#2a2f3a"),
                      yaxis=dict(range=[-extent, extent], dtick=5, title="Y (m)", gridcolor="#2a2f3a",
                                 scaleanchor="x", scaleratio=1))
    _show(fig)
    if rows:
        st.dataframe(pd.DataFrame(rows).sort_values("Age (s)"), hide_index=True)


mission_view()
