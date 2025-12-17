# solara_app.py
# -------------------------------------------
# Solara UI untuk visualisasi simulasi MAS
# (Updated with Advanced Visualizations)
# -------------------------------------------

import solara
import pandas as pd
import altair as alt
import random
import mesa

# ================================================================
# Model & Agent
# ================================================================
class VisitorAgent(mesa.Agent):
    def __init__(self, model, service_class, vid):
        super().__init__(model)
        self.birth_time = model.current_time
        self.service_class = service_class
        self.start_time = None
        self.end_time = None
        self.handled_by = None
        self.unique_id = vid

    def step(self):
        pass

    def advance(self):
        pass


class ServerAgent(mesa.Agent):
    def __init__(self, model, service_class, sid, efficiency):
        super().__init__(model)
        self.service_class = service_class
        self.unique_id = sid
        self.efficiency = efficiency
        self.is_busy = False
        self.current_visitor = None
        self.remaining = 0.0
        self.busy_time = 0.0  # Melacak waktu sibuk untuk utilisasi

    def step(self):
        dt = self.model.dt
        if not self.is_busy:
            q = self.model.queues[self.service_class]
            if q:
                self.start_service(q.pop(0))

        if self.is_busy:
            self.remaining -= dt
            self.busy_time += dt

    def advance(self):
        if self.is_busy and self.remaining <= 0:
            v = self.current_visitor
            v.end_time = self.model.current_time
            self.model.completed.append(v)
            self.is_busy = False
            self.current_visitor = None

    def start_service(self, visitor):
        visitor.start_time = self.model.current_time
        visitor.handled_by = self.unique_id
        mean = self.model.params[visitor.service_class]["service_mean"]
        raw = random.expovariate(1 / mean)
        self.remaining = raw / self.efficiency
        self.current_visitor = visitor
        self.is_busy = True


class PublicServiceModel(mesa.Model):
    def __init__(self, params, dt=0.05):
        super().__init__()
        self.params = params
        self.dt = dt
        self.current_time = 0.0
        self.visitor_counter = 0
        self.queues = {c: [] for c in params.keys()}
        self.completed = []

        # add servers
        for c, p in params.items():
            for i in range(p["num_servers"]):
                eff = random.uniform(0.8, 1.2)
                sid = f"{c}-{i+1}"
                self.agents.add(ServerAgent(self, c, sid, eff))

    def generate_arrivals(self):
        for cls, p in self.params.items():
            if random.random() < p["lambda"] * self.dt:
                self.visitor_counter += 1
                v = VisitorAgent(self, cls, f"V-{self.visitor_counter}")
                self.queues[cls].append(v)
                self.agents.add(v)

    def step(self):
        self.generate_arrivals()
        self.agents.do("step")
        self.agents.do("advance")
        self.current_time += self.dt


# ================================================================
# Solara UI state
# ================================================================
params = {
    "A": {"lambda": 5, "service_mean": 0.15, "num_servers": 2},
    "B": {"lambda": 3, "service_mean": 0.20, "num_servers": 2},
}

model_state = solara.reactive(None)
queue_history = solara.reactive([])

# ================================================================
# Helpers Visualisasi
# ================================================================
def get_visitor_df(model):
    if not model.completed:
        return pd.DataFrame()
    
    data = []
    for v in model.completed:
        wait_time = v.start_time - v.birth_time
        service_time = v.end_time - v.start_time
        total_time = v.end_time - v.birth_time
        data.append({
            "Visitor": v.unique_id,
            "Class": v.service_class,
            "Arrival": v.birth_time,
            "Start": v.start_time,
            "End": v.end_time,
            "Served By": v.handled_by,
            "Wait Time": wait_time,
            "Service Duration": service_time,
            "Total Time": total_time
        })
    return pd.DataFrame(data)

def get_server_utilization(model):
    servers = [a for a in model.agents if isinstance(a, ServerAgent)]
    if not servers or model.current_time == 0:
        return 0.0
    
    total_busy = sum(s.busy_time for s in servers)
    total_capacity = len(servers) * model.current_time
    return (total_busy / total_capacity) * 100

# ================================================================
# Solara Page
# ================================================================
@solara.component
def Page():
    solara.Markdown("# Solara Live Simulation (Enhanced)")

    # init model once
    if model_state.value is None:
        model_state.value = PublicServiceModel(params)

    model = model_state.value

    # ---------- Handlers ----------
    def record_history():
        queue_history.value = queue_history.value + [
            {
                "time": model.current_time,
                **{f"q_{cls}": len(model.queues[cls]) for cls in model.params},
            }
        ]

    def step_once():
        model.step()
        record_history()

    def run_steps(n):
        for _ in range(n):
            model.step()
            record_history()

    def reset_model():
        model_state.value = PublicServiceModel(params)
        queue_history.value = []

    # ---------- Control Panel ----------
    with solara.Card("Kontrol Simulasi"):
        with solara.Row(gap="10px"):
            solara.Button("Step (1x)", on_click=step_once, color="primary")
            solara.Button("Run (50x)", on_click=lambda: run_steps(50), color="primary")
            solara.Button("Reset", on_click=reset_model, color="grey", outlined=True)

    # ---------- Dashboard KPI ----------
    dfv = get_visitor_df(model)
    utilization = get_server_utilization(model)
    
    avg_wait = dfv["Wait Time"].mean() * 60 if not dfv.empty else 0
    total_served = len(dfv)
    
    solara.Markdown("## Dashboard Performa")
    with solara.Row(gap="20px", style={"margin-bottom": "20px"}):
        with solara.Card(title="Total Selesai"):
            solara.Text(f"{total_served} Pengunjung", style={"font-size": "24px", "font-weight": "bold"})
        
        with solara.Card(title="Rata-rata Tunggu"):
            # Jika > 60 menit ubah format
            fmt = f"{avg_wait:.2f} menit"
            solara.Text(fmt, style={"font-size": "24px", "font-weight": "bold", "color": "orange" if avg_wait > 30 else "green"})
            
        with solara.Card(title="Utilisasi Server"):
            solara.Text(f"{utilization:.1f}%", style={"font-size": "24px", "font-weight": "bold"})

    # ---------- Status Saat Ini ----------
    with solara.Details("Lihat Status Server Live"):
        servers = [a for a in model.agents if isinstance(a, ServerAgent)]
        with solara.GridFixed(columns=2):
            for s in servers:
                status_color = "red" if s.is_busy else "green"
                status_text = "BUSY" if s.is_busy else "IDLE"
                solara.Markdown(f"**{s.unique_id}**: <span style='color:{status_color}'>{status_text}</span>")

    # ---------- Visualisasi Tabs ----------
    with solara.lab.Tabs():
        
        # TAB 1: Queue History
        with solara.lab.Tab("Queue Heatmap"):
            if queue_history.value:
                dfq = pd.DataFrame(queue_history.value)
                dfm = dfq.melt("time", var_name="queue", value_name="len")
                
                chart_q = (
                    alt.Chart(dfm)
                    .mark_rect()
                    .encode(
                        x=alt.X("time:Q", title="Waktu Simulasi"),
                        y=alt.Y("queue:N", title="Antrean"),
                        color=alt.Color("len:Q", title="Panjang Antrean", scale=alt.Scale(scheme="inferno"))
                    )
                    .properties(height=300, title="Panjang Antrean Seiring Waktu")
                )
                solara.FigureAltair(chart_q)
            else:
                solara.Info("Jalankan simulasi untuk melihat data.")

        # TAB 2: Scatter Plot (Wait vs Arrival)
        with solara.lab.Tab("Analisis Waktu Tunggu"):
            if not dfv.empty:
                chart_scatter = (
                    alt.Chart(dfv)
                    .mark_circle(size=60)
                    .encode(
                        x=alt.X("Arrival:Q", title="Waktu Kedatangan"),
                        y=alt.Y("Wait Time:Q", title="Lama Menunggu (jam)"),
                        color="Class:N",
                        tooltip=["Visitor", "Wait Time", "Served By"]
                    )
                    .properties(height=350, title="Waktu Tunggu vs Waktu Kedatangan")
                    .interactive()
                )
                
                # Histogram Distribusi
                chart_hist = (
                    alt.Chart(dfv)
                    .mark_bar()
                    .encode(
                        x=alt.X("Wait Time:Q", bin=True, title="Lama Menunggu"),
                        y=alt.Y("count()", title="Jumlah Pengunjung"),
                        color="Class:N"
                    )
                    .properties(height=200, title="Distribusi Waktu Tunggu")
                )
                
                solara.FigureAltair(chart_scatter)
                solara.FigureAltair(chart_hist)
            else:
                solara.Info("Belum ada pengunjung selesai.")

        # TAB 3: Gantt Chart
        with solara.lab.Tab("Timeline Pelayanan (Gantt)"):
            if not dfv.empty:
                # Filter 20 pengunjung terakhir agar chart terbaca
                recent_df = dfv.tail(20)
                
                chart_gantt = (
                    alt.Chart(recent_df)
                    .mark_bar()
                    .encode(
                        x=alt.X("Start:Q", title="Mulai Layanan"),
                        x2="End:Q",
                        y=alt.Y("Served By:N", title="Server ID"),
                        color=alt.Color("Class:N"),
                        tooltip=["Visitor", "Start", "End", "Service Duration"]
                    )
                    .properties(height=350, title="Aktivitas Server (20 Pengunjung Terakhir)")
                )
                solara.FigureAltair(chart_gantt)
            else:
                solara.Info("Belum ada data pelayanan.")

        # TAB 4: Raw Data
        with solara.lab.Tab("Data Tabel"):
            if not dfv.empty:
                solara.DataFrame(dfv)
            else:
                solara.Text("Tabel kosong.")