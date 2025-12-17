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
    page_title="Multi-Agent Public Service Sim",
    layout="wide",
)

# -----------------------------------------------------
# CUSTOM CSS
# -----------------------------------------------------
st.markdown("""
<style>
.block-container { padding-top: 1rem; }
.kpi-card {
    background: #f8f9fa; border-radius: 10px; padding: 15px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.1); text-align: center; border: 1px solid #ddd;
}
.kpi-title { font-size: 0.85rem; color: #666; font-weight: 600; }
.kpi-value { font-size: 1.6rem; font-weight: 700; color: #333; margin-top: -5px; }
.agent-card {
    background: #eef2f5; border-left: 4px solid #4CAF50; padding: 10px; margin-bottom: 8px; border-radius: 4px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------
# AGENT DEFINITIONS (MULTI-AGENT SYSTEM)
# -----------------------------------------------------

@dataclass(order=True)
class Event:
    time: float
    priority: int
    type: str
    agent_ref: any = field(compare=False) # Bisa refer ke Visitor atau Server

class VisitorAgent:
    def __init__(self, id, service_class, birth_time):
        self.id = id
        self.service_class = service_class
        self.birth_time = birth_time
        self.start_time = None
        self.end_time = None
        self.handled_by = None  # ID Server yang melayani

class ServerAgent:
    def __init__(self, id, service_class, efficiency=1.0):
        self.id = id
        self.service_class = service_class
        self.efficiency = efficiency  # 1.0 = standard, >1.0 = cepat, <1.0 = lambat
        self.is_busy = False
        self.total_served = 0
        self.busy_time = 0.0
        self.current_visitor = None

    def start_service(self, visitor, current_time, base_service_mean):
        self.is_busy = True
        self.current_visitor = visitor
        visitor.start_time = current_time
        visitor.handled_by = self.id
        
        # Service time dipengaruhi efisiensi agen (Logika MAS: Skill agen berpengaruh)
        # Base service time (random) dibagi efficiency. Makin efisien, makin cepat.
        raw_service_time = random.expovariate(1 / base_service_mean)
        actual_service_time = raw_service_time / self.efficiency
        
        finish_time = current_time + actual_service_time
        
        # Catat statistik agen
        self.busy_time += actual_service_time
        self.total_served += 1
        
        return finish_time

    def finish_service(self):
        visitor = self.current_visitor
        self.current_visitor = None
        self.is_busy = False
        return visitor

# -----------------------------------------------------
# SIMULATION ENGINE
# -----------------------------------------------------
class MultiAgentSim:
    def __init__(self, params):
        self.params = params
        self.clock = 0.0
        self.event_queue = []
        self.visitor_counter = 0
        
        # State Sistem
        self.queues = {c: [] for c in params.keys()}
        self.completed_visitors = []
        
        # Inisialisasi Agen Server
        self.servers = []
        for c, p in params.items():
            for i in range(p["num_servers"]):
                # Simulasi skill random: efisiensi antara 0.8 (lambat) s.d. 1.2 (cepat)
                eff = random.uniform(0.8, 1.2)
                agent_id = f"{c}-{i+1}"
                self.servers.append(ServerAgent(agent_id, c, eff))

    def schedule_event(self, time, type, ref=None):
        # Priority: Arrival (0) -> Departure (1)
        prio = 0 if type == "arrival" else 1
        heapq.heappush(self.event_queue, Event(time, prio, type, ref))

    def run(self, until):
        # Schedule kedatangan pertama untuk tiap kelas
        for c, p in self.params.items():
            first_time = random.expovariate(p["lambda"])
            self.schedule_arrival(first_time, c)

        while self.event_queue and self.clock <= until:
            ev = heapq.heappop(self.event_queue)
            self.clock = ev.time

            if ev.type == "arrival":
                self.handle_arrival(ev.agent_ref) # agent_ref di sini adalah class str ('A', 'B')
            elif ev.type == "departure":
                self.handle_departure(ev.agent_ref) # agent_ref di sini adalah ServerAgent

    def schedule_arrival(self, time, service_class):
        self.schedule_event(time, "arrival", service_class)

    def handle_arrival(self, service_class):
        # 1. Create Visitor Agent
        self.visitor_counter += 1
        visitor = VisitorAgent(self.visitor_counter, service_class, self.clock)
        
        # 2. Schedule next arrival (agar loop berlanjut)
        p = self.params[service_class]
        next_time = self.clock + random.expovariate(p["lambda"])
        self.schedule_arrival(next_time, service_class)

        # 3. Cari Server Agent yang idle untuk kelas ini
        idle_server = self.find_idle_server(service_class)

        if idle_server:
            # INTERAKSI AGEN: Server melayani Visitor
            self.assign_job(idle_server, visitor)
        else:
            # Masuk antrean
            self.queues[service_class].append(visitor)

    def handle_departure(self, server_agent):
        # 1. Server menyelesaikan tugas
        visitor = server_agent.finish_service()
        visitor.end_time = self.clock
        self.completed_visitors.append(visitor)

        # 2. Server mencari tugas baru dari antrean (Proaktif)
        svc_class = server_agent.service_class
        if self.queues[svc_class]:
            next_visitor = self.queues[svc_class].pop(0)
            self.assign_job(server_agent, next_visitor)
        
        # Jika antrean kosong, server otomatis idle (sudah diset di finish_service)

    def find_idle_server(self, service_class):
        # Filter agen berdasarkan kelas dan status idle
        candidates = [s for s in self.servers if s.service_class == service_class and not s.is_busy]
        if candidates:
            # Bisa ditambahkan logika: pilih server dengan efisiensi tertinggi, atau random
            return candidates[0] 
        return None

    def assign_job(self, server, visitor):
        mean_svc = self.params[server.service_class]["service_mean"]
        finish_time = server.start_service(visitor, self.clock, mean_svc)
        self.schedule_event(finish_time, "departure", server)

# -----------------------------------------------------
# SIDEBAR INPUT
# -----------------------------------------------------
st.sidebar.header("⚙️ Parameter MAS")

classes = ["A", "B", "C"]
params = {}

for cls in classes:
    with st.sidebar.expander(f"Layanan Kelas {cls}", expanded=True):
        lmb = st.number_input(f"λ (Orang/jam) - {cls}", 1.0, 50.0, 10.0, key=f"l_{cls}")
        svc = st.number_input(f"Avg Service (Jam) - {cls}", 0.05, 2.0, 0.15, key=f"s_{cls}")
        srv = st.number_input(f"Jml Agen Server - {cls}", 1, 10, 2, key=f"n_{cls}")
        sla = st.number_input(f"Target SLA (Jam) - {cls}", 0.1, 5.0, 0.5, key=f"sla_{cls}")
        
        params[cls] = {
            "lambda": lmb,
            "service_mean": svc,
            "num_servers": srv,
            "sla_max": sla
        }

sim_duration = st.sidebar.slider("Durasi Simulasi (Jam)", 1, 48, 8)
run_btn = st.sidebar.button("🚀 Jalankan Simulasi Multi-Agen")

# -----------------------------------------------------
# MAIN DASHBOARD
# -----------------------------------------------------
if run_btn:
    st.title("🤖 Multi-Agent Service Simulation")
    
    # Run Simulation
    sim = MultiAgentSim(params)
    sim.run(until=sim_duration)
    
    # Process Data
    data = []
    sla_violations = 0
    
    for v in sim.completed_visitors:
        wait_time = v.start_time - v.birth_time
        total_time = v.end_time - v.birth_time
        sla_limit = params[v.service_class]["sla_max"]
        is_violation = total_time > sla_limit
        if is_violation: sla_violations += 1
        
        data.append({
            "Visitor ID": v.id,
            "Class": v.service_class,
            "Arrival": v.birth_time,
            "Start": v.start_time,
            "End": v.end_time,
            "Wait Time": wait_time,
            "Total Time": total_time,
            "Served By": v.handled_by,
            "SLA Violated": "Yes" if is_violation else "No"
        })
    
    df = pd.DataFrame(data)
    
    # ------------------------------------------------
    # KPI SECTION
    # ------------------------------------------------
    if not df.empty:
        col1, col2, col3, col4 = st.columns(4)
        
        avg_wait = df["Wait Time"].mean()
        sla_percent = 100 * (1 - (sla_violations / len(df)))
        utilization_rate = sum(s.busy_time for s in sim.servers) / (len(sim.servers) * sim_duration) * 100
        
        with col1:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Total Pengunjung</div><div class='kpi-value'>{len(df)}</div></div>", unsafe_allow_html=True)
        with col2:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Rata-rata Antre</div><div class='kpi-value'>{avg_wait*60:.1f} mnt</div></div>", unsafe_allow_html=True)
        with col3:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>SLA Compliance</div><div class='kpi-value'>{sla_percent:.1f}%</div></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='kpi-card'><div class='kpi-title'>Avg Server Util</div><div class='kpi-value'>{utilization_rate:.1f}%</div></div>", unsafe_allow_html=True)

        st.divider()

        # ------------------------------------------------
        # TABS DETAIL
        # ------------------------------------------------
        tab1, tab2, tab3 = st.tabs(["🕵️ Monitor Agen", "📊 Analisis Antrean", "📁 Data Log"])

        with tab1:
            st.subheader("Kinerja Individual Agen Server")
            st.caption("Setiap agen memiliki efisiensi skill yang berbeda (Random 0.8x - 1.2x).")
            
            # Siapkan data agen
            agent_data = []
            for s in sim.servers:
                util = (s.busy_time / sim_duration) * 100
                agent_data.append({
                    "ID Agen": s.id,
                    "Layanan": s.service_class,
                    "Skill (Efisiensi)": f"{s.efficiency:.2f}x",
                    "Total Dilayani": s.total_served,
                    "Jam Sibuk": f"{s.busy_time:.2f} jam",
                    "Utilisasi (%)": util
                })
            
            df_agent = pd.DataFrame(agent_data)
            
            # Chart Utilisasi Agen
            base = alt.Chart(df_agent).encode(y=alt.Y('ID Agen', sort=None))
            
            bar = base.mark_bar().encode(
                x=alt.X('Utilisasi (%):Q', title='Utilisasi Kerja (%)'),
                color=alt.Color('Layanan', legend=None),
                tooltip=['ID Agen', 'Skill (Efisiensi)', 'Total Dilayani', 'Utilisasi (%)']
            )
            
            text = base.mark_text(align='left', dx=2).encode(
                x='Utilisasi (%):Q',
                text=alt.Text('Utilisasi (%):Q', format='.1f')
            )
            
            st.altair_chart(bar + text, use_container_width=True)
            st.dataframe(df_agent, hide_index=True)

        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                st.subheader("Distribusi Waktu Tunggu")
                hist = alt.Chart(df).mark_bar().encode(
                    x=alt.X("Wait Time", bin=True, title="Waktu Tunggu (Jam)"),
                    y='count()',
                    color='Class'
                )
                st.altair_chart(hist, use_container_width=True)
            
            with col_b:
                st.subheader("Beban per Agen")
                pie = alt.Chart(df).mark_arc().encode(
                    theta='count()',
                    color='Served By',
                    tooltip=['Served By', 'count()']
                )
                st.altair_chart(pie, use_container_width=True)

        with tab3:
            st.dataframe(df)
            
    else:
        st.warning("Tidak ada pengunjung yang selesai dilayani dalam durasi ini.")
else:
    st.info("👈 Atur parameter di sidebar dan klik tombol 'Jalankan' untuk memulai.")