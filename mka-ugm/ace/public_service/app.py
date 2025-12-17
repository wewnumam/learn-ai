"""
app_full.py
Mesa 3.3.0 + Streamlit single-file app with:
- SolaraViz page builder (experimental, run separately)
- Gantt chart of service intervals (Plotly)
- BatchRunner wrapper
- Queue-length heatmap (Altair)
- SQLite DB integration for persistent logs
"""

import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import sqlite3
import random
import time
import io
import json

import mesa

# For batch runner
from mesa.batchrunner import batch_run

# -------------------------
# Model + Agents
# -------------------------
class VisitorAgent(mesa.Agent):
    def __init__(self, model, service_class, vid):
        super().__init__(model)
        self.birth_time = model.current_time
        self.service_class = service_class
        self.start_time = None
        self.end_time = None
        self.handled_by = None
        self.unique_id = vid  # readable id

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
        self.busy_time = 0.0
        self.total_served = 0

    def step(self):
        dt = self.model.dt
        if not self.is_busy:
            q = self.model.queues[self.service_class]
            if q:
                visitor = q.pop(0)
                self.start_service(visitor)

        if self.is_busy:
            self.remaining -= dt
            self.busy_time += dt

    def advance(self):
        if self.is_busy and self.remaining <= 0:
            v = self.current_visitor
            v.end_time = self.model.current_time
            self.model.completed_visitors.append(v)
            # record service finish event
            self.model.log_event({
                "visitor_id": v.unique_id,
                "class": v.service_class,
                "start": v.start_time,
                "end": v.end_time,
                "served_by": v.handled_by,
            })
            self.current_visitor = None
            self.is_busy = False

    def start_service(self, visitor):
        visitor.start_time = self.model.current_time
        visitor.handled_by = self.unique_id
        mean = self.model.params[visitor.service_class]["service_mean"]
        raw = random.expovariate(1 / mean)
        actual = raw / self.efficiency
        self.remaining = actual
        self.current_visitor = visitor
        self.is_busy = True
        self.total_served += 1
        # record service start event
        self.model.log_event({
            "visitor_id": visitor.unique_id,
            "class": visitor.service_class,
            "start": visitor.start_time,
            "end": None,
            "served_by": self.unique_id,
        })


class PublicServiceModel(mesa.Model):
    def __init__(self, params, duration_hours=8, dt=0.01, run_name=None):
        super().__init__()
        self.params = params
        self.duration = duration_hours
        self.dt = dt
        self.current_time = 0.0
        self.visitor_counter = 0
        self.queues = {c: [] for c in params.keys()}
        self.completed_visitors = []
        self.run_events = []   # record service start/end events for Gantt
        self.queue_history = []  # record queue lengths over time (per step)
        self.run_name = run_name or f"run_{int(time.time())}"

        self.datacollector = mesa.DataCollector(
            model_reporters={
                "Total served": lambda m: len(m.completed_visitors)
            }
        )

        # add servers
        for c, p in params.items():
            for i in range(p["num_servers"]):
                eff = self.random.uniform(0.8, 1.2)
                sid = f"{c}-{i+1}"
                s = ServerAgent(self, c, sid, eff)
                # Mesa 3.x uses model.agents.add/remove via AgentSet wrapper
                self.agents.add(s)

    def log_event(self, ev):
        # store a shallow copy (we'll fill end later)
        self.run_events.append(dict(ev))

    def generate_arrivals(self):
        for cls, p in self.params.items():
            prob = p["lambda"] * self.dt
            if self.random.random() < prob:
                self.visitor_counter += 1
                vid = f"V-{self.visitor_counter}"
                v = VisitorAgent(self, cls, vid)
                self.queues[cls].append(v)
                self.agents.add(v)
                # record arrival as an event (arrivals are useful too)
                self.run_events.append({
                    "visitor_id": vid,
                    "class": cls,
                    "start": v.birth_time,
                    "end": None,
                    "served_by": None,
                    "event_type": "arrival"
                })

    def record_queue_lengths(self):
        snapshot = {"time": self.current_time}
        for cls in self.params.keys():
            snapshot[f"q_{cls}"] = len(self.queues[cls])
        self.queue_history.append(snapshot)

    def step(self):
        if self.current_time >= self.duration:
            return
        # arrivals
        self.generate_arrivals()
        # record queue lengths pre-step
        self.record_queue_lengths()

        # multi-stage activation using AgentSet API (Mesa 3.x)
        self.agents.do("step")
        self.agents.do("advance")

        self.current_time += self.dt

    def get_run_log_df(self):
        # Convert run_events into DataFrame
        evs = [e for e in self.run_events if e.get("event_type") != "arrival"]
        df = pd.DataFrame(evs)
        # Some rows have end==None (start record). For start-only, we'll fill end from matching later.
        return df

# -------------------------
# Helper: persist logs to sqlite
# -------------------------
DB_FILE = "runs.db"

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS runs (
        run_name TEXT PRIMARY KEY,
        params TEXT,
        duration_hours REAL,
        dt REAL,
        created_at REAL
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS visitors (
        run_name TEXT,
        visitor_id TEXT,
        class TEXT,
        arrival REAL,
        start REAL,
        end REAL,
        served_by TEXT
    )""")
    conn.commit()
    conn.close()

def save_run_to_db(model: PublicServiceModel):
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Save run meta
    c.execute("INSERT OR REPLACE INTO runs (run_name, params, duration_hours, dt, created_at) VALUES (?, ?, ?, ?, ?)",
              (model.run_name, json.dumps(model.params), model.duration, model.dt, time.time()))
    # Save all visited visitors
    for v in model.completed_visitors:
        c.execute("INSERT INTO visitors (run_name, visitor_id, class, arrival, start, end, served_by) VALUES (?, ?, ?, ?, ?, ?, ?)",
                  (model.run_name, v.unique_id, v.service_class, v.birth_time, v.start_time, v.end_time, v.handled_by))
    conn.commit()
    conn.close()

def load_runs_list():
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM runs ORDER BY created_at DESC", conn)
    conn.close()
    return df

def load_visitors_for_run(run_name):
    conn = sqlite3.connect(DB_FILE)
    df = pd.read_sql_query("SELECT * FROM visitors WHERE run_name = ?", conn, params=(run_name,))
    conn.close()
    return df

# -------------------------
# Run + Analysis helpers
# -------------------------
def run_model_and_get_artifacts(params, duration, dt=0.01, run_name=None):
    m = PublicServiceModel(params, duration, dt, run_name)
    while m.current_time < duration:
        m.step()
    # Build visitor DataFrame
    visitors = []
    for v in m.completed_visitors:
        visitors.append({
            "Visitor ID": v.unique_id,
            "Class": v.service_class,
            "Arrival": v.birth_time,
            "Start": v.start_time,
            "End": v.end_time,
            "Wait Time": (v.start_time - v.birth_time) if v.start_time else None,
            "Total Time": (v.end_time - v.birth_time) if v.end_time else None,
            "Served By": v.handled_by
        })
    df_visitors = pd.DataFrame(visitors)
    df_queue = pd.DataFrame(m.queue_history)
    # extract gantt events from run_events where start present and end present
    gantt_rows = []
    for ev in m.run_events:
        if ev.get("event_type") == "arrival":
            continue
        # For events appended, there can be both start (end None) and later end appended.
        # Our model recorded both start (end None) and later end where end filled; here entries might be duplicates.
        # We'll filter only entries that have both start and end for Gantt.
        if ev.get("start") is not None and ev.get("end") is not None:
            gantt_rows.append({
                "Task": ev.get("served_by") or ev.get("visitor_id"),
                "Visitor": ev.get("visitor_id"),
                "Class": ev.get("class"),
                "Start": ev.get("start"),
                "Finish": ev.get("end"),
                "ServedBy": ev.get("served_by")
            })
    df_gantt = pd.DataFrame(gantt_rows)
    return m, df_visitors, df_queue, df_gantt

# -------------------------
# BatchRunner wrapper
# -------------------------
def run_batch(params_grid, iterations=3, max_steps=None, dt=0.01):
    """
    params_grid: dict mapping param name to list/range (must match model __init__ signature keys)
    Our model uses 'params' as whole dict, so we'll create combinations by wrapping params grid manually.
    For convenience, the UI will call this wrapper building variations on lambda values per class.
    """
    # Use mesa.batchrunner.batch_run but adapt model constructor: it expects (params, duration_hours, dt)
    # batch_run requires model_class and param_grid as kwargs that __init__ accepts. We'll instead
    # run manually for each combination (simpler and more controlled).
    from itertools import product
    results = []
    # params_grid example: {"A_lambda": [5,10], "B_lambda": [2,4]}
    keys = list(params_grid.keys())
    combos = list(product(*[params_grid[k] for k in keys]))
    for comb in combos:
        combo_dict = dict(zip(keys, comb))
        # build param set (user of UI provides base params and only changes some lambdas)
        base = st.session_state.get("base_params_for_batch")
        # copy base
        p = {k: dict(v) for k, v in base.items()}
        # apply combo (like "A_lambda" -> p["A"]["lambda"]=value)
        for k, val in combo_dict.items():
            cls, attr = k.split("_")
            p[cls][attr] = val
        # run iterations
        for it in range(iterations):
            model, df_vis, _, _ = run_model_and_get_artifacts(p, base_duration_for_batch := base.get("_duration", 8), dt=dt)
            # compute metrics
            total = len(df_vis)
            avg_wait = df_vis["Wait Time"].mean() if not df_vis.empty else None
            sla_violations = (df_vis["Total Time"] > p[cls]["sla_max"]).sum() if not df_vis.empty else 0
            results.append({**combo_dict, "iteration": it + 1, "total": total, "avg_wait": avg_wait, "sla_violations": sla_violations})
    return pd.DataFrame(results)

# -------------------------
# Streamlit UI
# -------------------------
init_db()

st.set_page_config(page_title="MAS - Full (Solara, Gantt, Batch, Heatmap, DB)", layout="wide")
st.title("Multi-Agent Service Simulation — Full Suite")

with st.sidebar.expander("Simulation Controls", expanded=True):
    # base params
    classes = ["A", "B", "C"]
    base_params = {}
    for cls in classes:
        st.markdown(f"### Kelas {cls}")
        with st.container():
            l = st.number_input(f"λ ({cls})", 1.0, 100.0, 10.0, key=f"lambda_{cls}")
            s = st.number_input(f"Service mean (jam) ({cls})", 0.05, 2.0, 0.15, key=f"svc_{cls}")
            n = st.number_input(f"Num servers ({cls})", 1, 10, 2, key=f"num_{cls}")
            sla = st.number_input(f"SLA (jam) ({cls})", 0.1, 5.0, 0.5, key=f"sla_{cls}")
            base_params[cls] = {
                "lambda": l,
                "service_mean": s,
                "num_servers": n,
                "sla_max": sla
            }

    duration = st.slider("Duration (hours)", 1, 48, 8)
    dt = st.number_input("dt (hours per step)", 0.001, 0.1, 0.01, step=0.001)
    run_name = st.text_input("Run name (optional)", value=f"run_{int(time.time())}")

    run_button = st.button("Run Simulation")

    # Batch runner inputs
    st.markdown("---")
    st.write("Batch runner (quick setup):")
    batch_A_lambdas = st.text_input("A_lambda values (comma)", value="5,10,15")
    batch_B_lambdas = st.text_input("B_lambda values (comma)", value="2,4")
    batch_iterations = st.number_input("Iterations per combo", 1, 10, 3)
    run_batch_button = st.button("Run Batch")

    # SolaraViz switch
    run_solara = st.checkbox("Show SolaraViz page (experimental)", value=False)

    # Save/load runs
    st.markdown("---")
    if st.button("List saved runs"):
        runs_df = load_runs_list()
        st.write("Saved runs:")
        st.dataframe(runs_df)

if run_button:
    # run model
    st.info("Running simulation (this may take a moment)...")
    model, df_vis, df_queue, df_gantt = run_model_and_get_artifacts(base_params, duration, dt=dt, run_name=run_name)

    # persist to sqlite
    save_run_to_db(model)
    st.success("Run finished and saved to DB.")

    # Show KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total served", len(df_vis))
    col2.metric("Avg wait (min)", f"{(df_vis['Wait Time'].mean()*60):.2f}" if not df_vis.empty else "N/A")
    sla_violations = 0
    if not df_vis.empty:
        sla_violations = (df_vis["Total Time"] > base_params[df_vis['Class'].iloc[0]]["sla_max"]).sum()
    col3.metric("SLA compliance", f"{100*(1 - (sla_violations / len(df_vis)) if len(df_vis)>0 else 1):.1f}%")
    servers = [a for a in model.agents if isinstance(a, ServerAgent)]
    avg_util = sum(s.busy_time for s in servers) / (len(servers) * duration) * 100
    col4.metric("Avg server util (%)", f"{avg_util:.1f}")

    st.markdown("### Visitor log")
    st.dataframe(df_vis)

    # Scatter plot: Arrival vs Wait Time per Visitor
    if not df_vis.empty:
        st.markdown("### Scatter Plot — Arrival Time vs Wait Time")
        df_scatter = df_vis.copy()

        fig_scatter = px.scatter(
            df_scatter,
            x="Arrival",
            y="Wait Time",
            color="Class",
            hover_data=["Visitor ID", "Served By"],
            labels={"Arrival": "Arrival Time (hours)", "Wait Time": "Wait Time (hours)"},
            title="Wait Time vs Arrival Time"
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    else:
        st.info("No visitor data available for scatter plot.")


    # Gantt chart (Plotly) — uses df_gantt extracted from events
    if not df_gantt.empty:
        st.markdown("### Gantt chart of services")
        # Convert times to datetime for plotly convenience:
        start_ts = pd.Timestamp("2025-01-01")  # anchor date
        dfg = df_gantt.copy()
        dfg["Start_dt"] = dfg["Start"].apply(lambda x: start_ts + pd.to_timedelta(x, unit="h"))
        dfg["Finish_dt"] = dfg["Finish"].apply(lambda x: start_ts + pd.to_timedelta(x, unit="h"))
        fig = px.timeline(dfg, x_start="Start_dt", x_end="Finish_dt", y="ServedBy", color="Class", hover_data=["Visitor"])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough events to build Gantt (try increasing simulation duration or arrival rates).")

    # Heatmap of queue lengths
    if not df_queue.empty:
        st.markdown("### Queue length heatmap over time")
        dfq = df_queue.melt(id_vars=["time"], var_name="queue", value_name="length")
        chart = alt.Chart(dfq).mark_rect().encode(
            x=alt.X("time:Q", title="Time (hours)"),
            y=alt.Y("queue:N", title="Queue"),
            color=alt.Color("length:Q", title="Length")
        ).properties(height=300)
        st.altair_chart(chart, use_container_width=True)
    else:
        st.info("No queue history recorded.")

    # ... (kode sebelumnya: Metrics, Scatter Plot, Gantt, Heatmap) ...

    st.markdown("---")
    st.header("Analisis Lanjutan")

    col_a, col_b = st.columns(2)

    # 1. Histogram Distribusi Waktu Tunggu
    with col_a:
        st.subheader("Distribusi Waktu Tunggu")
        if not df_vis.empty:
            # Menggunakan Plotly untuk Histogram
            fig_hist = px.histogram(
                df_vis, 
                x="Wait Time", 
                color="Class", 
                nbins=30,
                title="Frekuensi Lama Menunggu (Jam)",
                labels={"Wait Time": "Waktu Tunggu (Jam)"},
                opacity=0.7,
                barmode="overlay"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
            
            with st.expander("Insight Histogram"):
                st.write("""
                Grafik ini menunjukkan sebaran waktu tunggu. 
                - Jika grafik condong ke kiri (banyak di angka 0), sistem sehat.
                - Jika grafik menyebar ke kanan (ekor panjang), artinya ada pengunjung yang menunggu sangat lama.
                """)
        else:
            st.info("Tidak ada data pengunjung.")

    # 2. Server Load Balance (Total Served per Server)
    with col_b:
        st.subheader("Beban Kerja Server (Load Balance)")
        if not df_vis.empty:
            # Hitung jumlah yang dilayani tiap server
            df_load = df_vis.groupby("Served By").size().reset_index(name="Total Served")
            
            fig_bar = px.bar(
                df_load,
                x="Served By",
                y="Total Served",
                color="Served By",
                title="Total Pengunjung yang Dilayani per Server",
                text="Total Served"
            )
            fig_bar.update_traces(textposition='outside')
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.info("Belum ada pengunjung yang selesai dilayani.")

    # 3. Line Chart Dinamika Antrean (Alternative to Heatmap)
    st.subheader("Dinamika Panjang Antrean (Line Chart)")
    if not df_queue.empty:
        # Transformasi data untuk Altair/Line chart
        # df_queue format: time, q_A, q_B, ...
        df_q_melt = df_queue.melt("time", var_name="Queue Class", value_name="Length")
        
        chart_line = (
            alt.Chart(df_q_melt)
            .mark_line()
            .encode(
                x=alt.X("time:Q", title="Waktu Simulasi (Jam)"),
                y=alt.Y("Length:Q", title="Jumlah Orang di Antrean"),
                color=alt.Color("Queue Class:N", title="Kelas Layanan"),
                tooltip=["time", "Queue Class", "Length"]
            )
            .properties(height=350)
            .interactive()
        )
        st.altair_chart(chart_line, use_container_width=True)
    else:
        st.info("Data antrean kosong.")

    # 4. Cumulative Flow Diagram (CFD) - Simplified
    # Membandingkan Kumulatif Kedatangan vs Kumulatif Selesai
    st.subheader("Analisis Aliran (Cumulative Flow)")
    if not df_vis.empty:
        # Buat data frame event kedatangan dan selesai
        arrivals = df_vis[["Arrival"]].copy()
        arrivals["Type"] = "Arrival"
        arrivals["Time"] = arrivals["Arrival"]
        
        departures = df_vis[["End"]].copy()
        departures["Type"] = "Departure"
        departures["Time"] = departures["End"]
        
        # Gabungkan dan urutkan
        cfd_df = pd.concat([arrivals[["Time", "Type"]], departures[["Time", "Type"]]])
        cfd_df = cfd_df.sort_values("Time")
        
        # Hitung kumulatif
        # Kita pisahkan kumulatif arrival dan departure
        cfd_data = []
        count_arr = 0
        count_dep = 0
        
        # Iterasi manual sederhana untuk membuat step chart yang akurat
        # (Alternatif: resample time series, tapi loop ini cukup cepat untuk data kecil-menengah)
        for _, row in cfd_df.iterrows():
            if row["Type"] == "Arrival":
                count_arr += 1
            else:
                count_dep += 1
            
            cfd_data.append({
                "Time": row["Time"],
                "Cumulative Arrivals": count_arr,
                "Cumulative Departures": count_dep,
                "WIP (Work In Process)": count_arr - count_dep # Orang di dalam sistem
            })
            
        df_cfd = pd.DataFrame(cfd_data)
        
        # Melt untuk plotting Arrivals vs Departures
        df_cfd_melt = df_cfd.melt("Time", value_vars=["Cumulative Arrivals", "Cumulative Departures"], var_name="Metric", value_name="Count")
        
        chart_cfd = (
            alt.Chart(df_cfd_melt)
            .mark_line(interpolate="step-after")
            .encode(
                x=alt.X("Time:Q", title="Waktu (Jam)"),
                y=alt.Y("Count:Q", title="Jumlah Kumulatif Pengunjung"),
                color=alt.Color("Metric:N", scale=alt.Scale(domain=["Cumulative Arrivals", "Cumulative Departures"], range=["blue", "green"])),
                tooltip=["Time", "Metric", "Count"]
            )
            .properties(height=400, title="Cumulative Flow Diagram (CFD)")
        )
        
        # Area chart untuk WIP (Work In Process / Antrean + Service)
        chart_wip = (
            alt.Chart(df_cfd)
            .mark_area(opacity=0.3, color="orange")
            .encode(
                x="Time:Q",
                y=alt.Y("WIP (Work In Process):Q", title="Total Orang dalam Sistem"),
                tooltip=["Time", "WIP (Work In Process)"]
            )
            .properties(height=150, title="Beban Sistem (WIP)")
        )

        st.altair_chart(chart_cfd & chart_wip, use_container_width=True)
        
        with st.expander("Cara Membaca CFD"):
            st.write("""
            **Cumulative Flow Diagram (CFD)** sangat penting untuk melihat stabilitas sistem:
            1. **Garis Biru**: Total orang yang datang.
            2. **Garis Hijau**: Total orang yang selesai dilayani.
            3. **Jarak Vertikal**: Jumlah orang yang sedang mengantre atau dilayani saat itu (WIP).
            4. **Jarak Horizontal**: Rata-rata waktu penyelesaian sistem (Lead Time).
            
            *Jika garis Biru dan Hijau semakin melebar (divergen), sistem tidak stabil (bottleneck).*
            """)

    # Download visitor CSV
    csv = df_vis.to_csv(index=False)
    st.download_button("Download visitor log CSV", csv, file_name=f"{run_name}_visitors.csv", mime="text/csv")

if run_batch_button:
    st.info("Running batch (this may take a while). Results downloadable.")
    # store base_params for the batch wrapper
    st.session_state["base_params_for_batch"] = {**base_params, "_duration": duration}
    # parse combos
    A_vals = [float(x.strip()) for x in batch_A_lambdas.split(",") if x.strip()]
    B_vals = [float(x.strip()) for x in batch_B_lambdas.split(",") if x.strip()]
    params_grid = {"A_lambda": A_vals, "B_lambda": B_vals}
    # run simple manual batch (note: simplistic; extend as needed)
    rows = []
    for A_ in A_vals:
        for B_ in B_vals:
            for it in range(batch_iterations):
                p = {k: dict(v) for k, v in base_params.items()}
                p["A"]["lambda"] = A_
                p["B"]["lambda"] = B_
                model, df_vis, _, _ = run_model_and_get_artifacts(p, duration, dt=dt)
                rows.append({
                    "A_lambda": A_,
                    "B_lambda": B_,
                    "iteration": it+1,
                    "total_served": len(df_vis),
                    "avg_wait": df_vis["Wait Time"].mean() if not df_vis.empty else None
                })
    df_batch = pd.DataFrame(rows)
    st.dataframe(df_batch)
    st.download_button("Download batch results", df_batch.to_csv(index=False), file_name="batch_results.csv", mime="text/csv")

# -------------------------
# Saved-run explorer
# -------------------------
st.sidebar.markdown("---")
st.sidebar.write("Saved runs DB")
runs_df = load_runs_list()
sel = st.sidebar.selectbox("Select run", options=["-- none --"] + runs_df["run_name"].tolist())
if sel and sel != "-- none --":
    dfv = load_visitors_for_run(sel)
    st.sidebar.write(f"Visitors for {sel}:")
    st.sidebar.dataframe(dfv)

# -------------------------
# SolaraViz builder (experimental)
# -------------------------
def build_and_show_solara(params, duration):
    """
    This function attempts to build a SolaraViz page. SolaraViz is experimental in Mesa 3.x.
    If available, it will create a page object which you can display by running this script with an appropriate runner.
    Because Streamlit is already running, the Solara page should be started separately in a Python process/REPL.
    """
    try:
        from mesa.visualization import SolaraViz, make_space_component, make_plot_component
    except Exception as e:
        st.warning("SolaraViz imports failed — your Mesa build might not include visualization. Error: " + str(e))
        return None

    def portrayal(agent):
        # simple portrayal for servers + visitors
        if isinstance(agent, ServerAgent):
            return {"color": "green", "r": 10, "text": agent.unique_id}
        else:
            return {"color": "blue", "r": 4, "text": agent.unique_id}

    model_params = {
        "params": params,
        "duration_hours": duration
    }

    page = SolaraViz(
        PublicServiceModel(params, duration_hours=duration),
        components=[
            make_space_component(portrayal_fn=portrayal),
            make_plot_component("Total served")
        ],
        model_params=model_params
    )
    st.write("SolaraViz page object created. To show it, run it in a separate Python session (not inside Streamlit).")
    return page

if run_solara:
    st.info("Attempting to build SolaraViz page (will not render inside Streamlit). See instructions in the sidebar.")
    # build but do not attempt to render here
    sv = build_and_show_solara(base_params, duration)
    if sv:
        st.success("SolaraViz page object created. Run it outside Streamlit to visualize.")

st.sidebar.markdown("""
**SolaraViz instructions**  
SolaraViz visualization is browser-based and currently experimental.  
If `build_and_show_solara` succeeded, run this file as a normal python script (not `streamlit run`), e.g.:
python -c "import app_full; app_full.build_and_show_solara(app_full_example_params, 8)
or run a small driver that calls `page` returned by the builder. Solara pages typically need a separate runner.
""")