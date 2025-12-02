import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from dataclasses import dataclass, field
import heapq
import random

# -----------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------
st.set_page_config(
    page_title="Public Service One-Stop Simulation",
    layout="wide",
)

# -----------------------------------------------------
# CUSTOM DASHBOARD CSS
# -----------------------------------------------------
st.markdown("""
<style>
.block-container {
    padding-top: 1rem;
}

.kpi-card {
    background: #ffffff;
    border-radius: 12px;
    padding: 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    border: 1px solid #eaeaea;
    text-align: center;
}

.kpi-title {
    font-size: 0.9rem;
    color: #666;
    font-weight: 600;
}

.kpi-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #333;
    margin-top: -8px;
}

h2 {
    font-weight: 700 !important;
    margin-top: 1.2rem !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# DATA STRUCTURES
# -----------------------------------------------------
@dataclass(order=True)
class Event:
    time: float
    priority: int
    type: str
    request: any = field(compare=False)

@dataclass
class Request:
    id: int
    service_class: str
    birth: float
    service_time: float
    start_time: float = None
    end_time: float = None

# -----------------------------------------------------
# MULTI-CLASS QUEUE SIMULATOR
# -----------------------------------------------------
class MultiClassQueueSim:
    def __init__(self, params):
        self.params = params
        self.clock = 0.0
        self.event_queue = []
        self.request_counter = 0
        self.queues = {c: [] for c in params.keys()}
        self.busy_servers = {c: 0 for c in params.keys()}
        self.total_processed = {c: 0 for c in params.keys()}
        self.sla_violations = {c: 0 for c in params.keys()}
        self.stats_requests = []

    def schedule_event(self, ev_time, ev_type, req=None):
        prio = 0 if ev_type == "arrival" else 1
        heapq.heappush(self.event_queue, Event(ev_time, prio, ev_type, req))

    def run(self, until):
        for c, p in self.params.items():
            first_arr = random.expovariate(p["lambda"])
            self.schedule_event(first_arr, "arrival", Request(None, c, first_arr, None))

        while self.event_queue and self.clock <= until:
            ev = heapq.heappop(self.event_queue)
            self.clock = ev.time

            if ev.type == "arrival":
                self.handle_arrival(ev.request)
            elif ev.type == "departure":
                self.handle_departure(ev.request)

    def handle_arrival(self, req):
        c = req.service_class
        p = self.params[c]

        self.request_counter += 1
        req.id = self.request_counter
        req.birth = self.clock
        req.service_time = random.expovariate(1 / p["service_mean"])

        next_arrival = self.clock + random.expovariate(p["lambda"])
        self.schedule_event(next_arrival, "arrival", Request(None, c, next_arrival, None))

        if self.busy_servers[c] < p["servers"]:
            self.start_service(req)
        else:
            self.queues[c].append(req)

    def start_service(self, req):
        c = req.service_class
        self.busy_servers[c] += 1
        req.start_time = self.clock
        end_t = self.clock + req.service_time
        req.end_time = end_t
        self.schedule_event(end_t, "departure", req)

    def handle_departure(self, req):
        c = req.service_class
        self.busy_servers[c] -= 1

        self.total_processed[c] += 1
        self.stats_requests.append(req)

        total_time = req.end_time - req.birth
        if total_time > self.params[c]["sla_max"]:
            self.sla_violations[c] += 1

        if self.queues[c]:
            next_req = self.queues[c].pop(0)
            self.start_service(next_req)

# -----------------------------------------------------
# SIDEBAR INPUT PARAMETER
# -----------------------------------------------------
st.sidebar.header("Parameter Simulasi")

classes = ["A", "B", "C"]
lambda_vals = {}
service_mean_vals = {}
servers_vals = {}
sla_vals = {}

for cls in classes:
    st.sidebar.subheader(f"Layanan {cls}")

    lambda_vals[cls] = st.sidebar.number_input(
        f"λ_{cls} (kedatangan/jam)", 0.1, 50.0, 10.0, key=f"lambda_{cls}"
    )

    service_mean_vals[cls] = st.sidebar.number_input(
        f"mean service time {cls} (jam)", 0.01, 5.0, 0.2, key=f"service_{cls}"
    )

    servers_vals[cls] = st.sidebar.number_input(
        f"jumlah server {cls}", 1, 20, 3, key=f"servers_{cls}"
    )

    sla_vals[cls] = st.sidebar.number_input(
        f"SLA max waktu (jam)", 0.1, 10.0, 1.0, key=f"sla_{cls}"
    )

sim_time = st.sidebar.number_input(
    "Durasi Simulasi (jam)", 1.0, 24.0, 4.0, key="sim_time"
)

run_btn = st.sidebar.button("Run Simulation")

params = {
    c: {
        "lambda": lambda_vals[c],
        "service_mean": service_mean_vals[c],
        "servers": servers_vals[c],
        "sla_max": sla_vals[c]
    }
    for c in classes
}

# -----------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------
if run_btn:
    st.markdown("## 🏛️ One-Stop Public Service Dashboard")

    sim = MultiClassQueueSim(params)
    sim.run(until=sim_time)

    df = pd.DataFrame([
        {
            "id": r.id,
            "class": r.service_class,
            "arrival": r.birth,
            "start": r.start_time,
            "end": r.end_time,
            "wait": r.start_time - r.birth if r.start_time else None,
            "service": r.service_time,
            "total_time": r.end_time - r.birth if r.end_time else None
        }
        for r in sim.stats_requests if r.end_time
    ])

    # -----------------------------------------------------
    # KPI CARDS
    # -----------------------------------------------------
    st.markdown("### 📌 Key Indicators")

    avg_wait = df["wait"].mean()
    sla_compliance = 1 - (sum(sim.sla_violations.values()) / len(df))
    total_processed = len(df)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Rata-rata Waktu Tunggu</div>
            <div class='kpi-value'>{avg_wait:.2f} jam</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Kepatuhan SLA</div>
            <div class='kpi-value'>{sla_compliance*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class='kpi-card'>
            <div class='kpi-title'>Total Request Selesai</div>
            <div class='kpi-value'>{total_processed}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # -----------------------------------------------------
    # TABS
    # -----------------------------------------------------
    tab1, tab2, tab3 = st.tabs(["📋 Ringkasan Layanan", "📈 Grafik", "📄 Data Mentah"])

    # TAB 1
    with tab1:
        st.subheader("Ringkasan Metrik per Layanan")

        summary = df.groupby("class").agg({
            "wait": "mean",
            "service": "mean",
            "total_time": "mean",
        }).round(3)

        summary["processed"] = summary.index.map(lambda c: sim.total_processed[c])
        summary["SLA Violations"] = summary.index.map(lambda c: sim.sla_violations[c])

        st.dataframe(summary)

    # TAB 2
    with tab2:
        st.subheader("Distribusi Waktu Tunggu")

        chart = alt.Chart(df).mark_circle(size=60).encode(
            x=alt.X("arrival:Q", title="Waktu Kedatangan"),
            y=alt.Y("wait:Q", title="Waktu Tunggu"),
            color=alt.Color("class:N", title="Layanan"),
            tooltip=["id", "class", "wait", "total_time"]
        ).interactive()

        st.altair_chart(chart, use_container_width=True)

    # TAB 3
    with tab3:
        st.subheader("Data Request Selesai")
        st.dataframe(df)

else:
    st.info("Tekan **Run Simulation** di sidebar untuk memulai simulasi.")