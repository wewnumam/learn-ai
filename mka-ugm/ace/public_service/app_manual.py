"""
app_with_agent_dashboards.py
Modified from user's app.py — adds per-agent dashboards and interactive mode
Run with: streamlit run app_with_agent_dashboards.py
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
    def __init__(self, model, service_class, sid, efficiency, manual=False):
        super().__init__(model)
        self.service_class = service_class
        self.unique_id = sid
        self.efficiency = efficiency
        self.manual_mode = manual      # manual override flag
        self.is_busy = False
        self.current_visitor = None
        self.remaining = 0.0
        self.busy_time = 0.0
        self.total_served = 0

    def step(self):
        dt = self.model.dt

        # If manual mode, do not pick new visitors automatically.
        if self.manual_mode:
            if self.is_busy:
                self.remaining -= dt
                self.busy_time += dt
            return

        # Automatic behavior
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
        evs = [e for e in self.run_events if e.get("event_type") != "arrival"]
        df = pd.DataFrame(evs)
        return df

    # -------------------------
    # Manual control API
    # -------------------------
    def manual_start_service(self, server_id, visitor_id, duration):
        try:
            server = next(a for a in self.agents if isinstance(a, ServerAgent) and a.unique_id == server_id)
        except StopIteration:
            return False, "Server tidak ditemukan"

        visitor = None
        q = self.queues[server.service_class]
        for v in q:
            if v.unique_id == visitor_id:
                visitor = v
                q.remove(v)
                break

        if visitor is None:
            return False, "Visitor tidak ditemukan di antrian."

        visitor.start_time = self.current_time
        visitor.handled_by = server_id

        server.current_visitor = visitor
        server.remaining = duration
        server.is_busy = True
        server.total_served += 1

        self.log_event({
            "visitor_id": visitor.unique_id,
            "class": visitor.service_class,
            "start": visitor.start_time,
            "end": None,
            "served_by": server_id,
            "event_type": "manual_start"
        })
        return True, "Layanan manual dimulai."

    def manual_finish(self, server_id):
        try:
            server = next(a for a in self.agents if isinstance(a, ServerAgent) and a.unique_id == server_id)
        except StopIteration:
            return False, "Server tidak ditemukan"

        if not server.is_busy:
            return False, "Server tidak sedang melayani."

        v = server.current_visitor
        v.end_time = self.current_time
        self.completed_visitors.append(v)

        self.log_event({
            "visitor_id": v.unique_id,
            "class": v.service_class,
            "start": v.start_time,
            "end": v.end_time,
            "served_by": server.unique_id,
            "event_type": "manual_finish"
        })

        server.current_visitor = None
        server.is_busy = False
        return True, "Layanan diselesaikan."

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
    c.execute("INSERT OR REPLACE INTO runs (run_name, params, duration_hours, dt, created_at) VALUES (?, ?, ?, ?, ?)",
              (model.run_name, json.dumps(model.params), model.duration, model.dt, time.time()))
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
# BatchRunner wrapper (kept simple)
# -------------------------
def run_batch(params_grid, iterations=3, max_steps=None, dt=0.01):
    from itertools import product
    results = []
    keys = list(params_grid.keys())
    combos = list(product(*[params_grid[k] for k in keys]))
    for comb in combos:
        combo_dict = dict(zip(keys, comb))
        base = st.session_state.get("base_params_for_batch")
        p = {k: dict(v) for k, v in base.items()}
        for k, val in combo_dict.items():
            cls, attr = k.split("_")
            p[cls][attr] = val
        for it in range(iterations):
            model, df_vis, _, _ = run_model_and_get_artifacts(p, base_duration_for_batch := base.get("_duration", 8), dt=dt)
            total = len(df_vis)
            avg_wait = df_vis["Wait Time"].mean() if not df_vis.empty else None
            sla_violations = 0
            results.append({**combo_dict, "iteration": it + 1, "total": total, "avg_wait": avg_wait, "sla_violations": sla_violations})
    return pd.DataFrame(results)

# -------------------------
# Streamlit UI
# -------------------------
init_db()

st.set_page_config(page_title="MAS - Agent Dashboards", layout="wide")
st.title("Multi-Agent Service Simulation — Per-Agent Dashboards")

with st.sidebar.expander("Simulation Controls", expanded=True):
    classes = ["A", "B", "C"]
    base_params = {}
    for cls in classes:
        st.markdown(f"### Kelas {cls}")
        with st.container():
            l = st.number_input(f"λ ({cls})", 1.0, 100.0, 10.0, key=f"lambda_{cls}")
            s = st.number_input(f"Service mean (jam) ({cls})", 0.01, 2.0, 0.15, key=f"svc_{cls}")
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

    run_mode = st.radio("Run mode", ["To completion", "Interactive"], index=0)

    run_button = st.button("Start / Create Model")

    st.markdown("---")
    st.write("Batch runner (quick setup):")
    batch_A_lambdas = st.text_input("A_lambda values (comma)", value="5,10,15")
    batch_B_lambdas = st.text_input("B_lambda values (comma)", value="2,4")
    batch_iterations = st.number_input("Iterations per combo", 1, 10, 3)
    run_batch_button = st.button("Run Batch")

    run_solara = st.checkbox("Show SolaraViz page (experimental)", value=False)

    st.markdown("---")
    if st.button("List saved runs"):
        runs_df = load_runs_list()
        st.write("Saved runs:")
        st.dataframe(runs_df)

# We branch: if interactive mode, create model and store in session_state for step-by-step control
if run_button:
    if run_mode == "To completion":
        st.info("Running simulation to completion...")
        model, df_vis, df_queue, df_gantt = run_model_and_get_artifacts(base_params, duration, dt=dt, run_name=run_name)
        save_run_to_db(model)
        st.success("Run finished and saved to DB.")

        # Show KPIs
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total served", len(df_vis))
        col2.metric("Avg wait (min)", f"{(df_vis['Wait Time'].mean()*60):.2f}" if not df_vis.empty else "N/A")
        sla_violations = 0
        if not df_vis.empty:
            # compute SLA violations across all classes
            sla_violations = sum((df_vis['Total Time'] > base_params[row['Class']]['sla_max']).sum() for _, row in df_vis.iterrows()) if not df_vis.empty else 0
        col3.metric("SLA compliance", f"{100*(1 - (sla_violations / len(df_vis)) if len(df_vis)>0 else 1):.1f}%")
        # estimate server util
        servers = [a for a in model.agents if isinstance(a, ServerAgent)]
        avg_util = sum(s.busy_time for s in servers) / (len(servers) * duration) * 100 if servers else 0
        col4.metric("Avg server util (%)", f"{avg_util:.1f}")

        st.markdown("### Visitor log")
        st.dataframe(df_vis)

        # Gantt
        if not df_gantt.empty:
            st.markdown("### Gantt chart of services")
            start_ts = pd.Timestamp("2025-01-01")
            dfg = df_gantt.copy()
            dfg["Start_dt"] = dfg["Start"].apply(lambda x: start_ts + pd.to_timedelta(x, unit="h"))
            dfg["Finish_dt"] = dfg["Finish"].apply(lambda x: start_ts + pd.to_timedelta(x, unit="h"))
            fig = px.timeline(dfg, x_start="Start_dt", x_end="Finish_dt", y="ServedBy", color="Class", hover_data=["Visitor"])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Not enough events to build Gantt (try increasing simulation duration or arrival rates).")

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

        csv = df_vis.to_csv(index=False)
        st.download_button("Download visitor log CSV", csv, file_name=f"{run_name}_visitors.csv", mime="text/csv")

        # Show per-agent read-only summary (after-run)
        st.markdown("## Agent summaries (read-only)")
        servers = [a for a in model.agents if isinstance(a, ServerAgent)]
        for s in servers:
            st.write(f"- {s.unique_id} (class {s.service_class}): served={s.total_served}, busy_time={s.busy_time:.2f}h")

    else:
        # Interactive mode: create model and store
        st.info("Creating interactive model — use Step/Run controls and agent dashboards below.")
        model = PublicServiceModel(base_params, duration_hours=duration, dt=dt, run_name=run_name)
        st.session_state["model"] = model
        st.session_state["interactive_created_at"] = time.time()
        st.session_state["last_step_time"] = model.current_time

# Run batch if requested
if run_batch_button:
    st.info("Running batch (this may take a while). Results downloadable.")
    st.session_state["base_params_for_batch"] = {**base_params, "_duration": duration}
    A_vals = [float(x.strip()) for x in batch_A_lambdas.split(",") if x.strip()]
    B_vals = [float(x.strip()) for x in batch_B_lambdas.split(",") if x.strip()]
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
# Interactive controls and Agent dashboards
# -------------------------
if st.session_state.get("model") is not None:
    model = st.session_state["model"]

    col_left, col_right = st.columns([2,3])

    with col_left:
        st.markdown("## Simulation Controls (Interactive)")
        st.write(f"Current time: {model.current_time:.3f} / {model.duration} hours")
        step_btn = st.button("Step once")
        run_n = st.number_input("Run N steps", min_value=1, max_value=10000, value=10, step=1)
        run_n_btn = st.button(f"Run {run_n} steps")
        finish_btn = st.button("Run to completion")
        save_snapshot = st.button("Save run to DB")

        if step_btn:
            model.step()
            st.session_state["last_step_time"] = model.current_time

        if run_n_btn:
            for _ in range(run_n):
                if model.current_time >= model.duration:
                    break
                model.step()
            st.session_state["last_step_time"] = model.current_time

        if finish_btn:
            while model.current_time < model.duration:
                model.step()
            st.success("Model reached end time.")

        if save_snapshot:
            save_run_to_db(model)
            st.success("Saved interactive run to DB.")

        # Show queue overview
        st.markdown("### Queues (overview)")
        for cls in model.params.keys():
            st.write(f"Class {cls}: {len(model.queues[cls])} waiting")

    with col_right:
        st.markdown("## Agent Dashboards (Manual Control)")

        servers = [a for a in model.agents if isinstance(a, ServerAgent)]
        for s in servers:
            with st.expander(f"Server {s.unique_id} — Class {s.service_class}"):
                # ensure session state keys exist
                key_manual = f"manual_{s.unique_id}"
                if key_manual not in st.session_state:
                    st.session_state[key_manual] = s.manual_mode

                st.checkbox("Manual mode", key=key_manual, help="Jika aktif, agen tidak akan mengambil visitor otomatis")
                s.manual_mode = st.session_state[key_manual]

                st.write(f"Status: {'Busy' if s.is_busy else 'Idle'}")
                if s.is_busy and s.current_visitor is not None:
                    st.write(f"Serving: {s.current_visitor.unique_id}, remaining={s.remaining:.3f} h")
                st.write("Queue:", [v.unique_id for v in model.queues[s.service_class]])

                if s.manual_mode:
                    st.success("Manual mode aktif — Anda dapat mengontrol agen ini")
                    q_ids = [v.unique_id for v in model.queues[s.service_class]]
                    if q_ids:
                        visitor_choice = st.selectbox(f"Pilih visitor untuk {s.unique_id}", q_ids, key=f"vis_{s.unique_id}")
                    else:
                        visitor_choice = None

                    duration_manual = st.number_input(f"Durasi manual (jam) untuk {s.unique_id}", 0.01, 4.0, 0.1, key=f"dur_{s.unique_id}")

                    if st.button(f"Start service - {s.unique_id}"):
                        if visitor_choice is None:
                            st.warning("Tidak ada visitor terpilih")
                        else:
                            ok, msg = model.manual_start_service(s.unique_id, visitor_choice, duration_manual)
                            st.write(msg)

                    if st.button(f"Finish service - {s.unique_id}"):
                        ok, msg = model.manual_finish(s.unique_id)
                        st.write(msg)

    # small diagnostics / visitor log while interacting
    st.markdown("### Visitor summary (interactive)")
    visitors = []
    for v in model.completed_visitors:
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
    df_vis = pd.DataFrame(visitors)
    st.dataframe(df_vis)

# Sidebar: Saved runs explorer
st.sidebar.markdown("---")
st.sidebar.write("Saved runs DB")
runs_df = load_runs_list()
sel = st.sidebar.selectbox("Select run", options=["-- none --"] + runs_df["run_name"].tolist())
if sel and sel != "-- none --":
    dfv = load_visitors_for_run(sel)
    st.sidebar.write(f"Visitors for {sel}:")
    st.sidebar.dataframe(dfv)

# SolaraViz instructions (unchanged)
def build_and_show_solara(params, duration):
    try:
        from mesa.visualization import SolaraViz, make_space_component, make_plot_component
    except Exception as e:
        st.warning("SolaraViz imports failed — your Mesa build might not include visualization. Error: " + str(e))
        return None

    def portrayal(agent):
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
    st.write("SolaraViz page object created. To show it, run it outside Streamlit.")
    return page

if run_solara:
    st.info("Attempting to build SolaraViz page (will not render inside Streamlit). See instructions in the sidebar.")
    sv = build_and_show_solara(base_params, duration)
    if sv:
        st.success("SolaraViz page object created. Run it outside Streamlit to visualize.")

st.sidebar.markdown("""
**SolaraViz instructions**  
SolaraViz visualization is browser-based and currently experimental.  
If `build_and_show_solara` succeeded, run this file as a normal python script (not `streamlit run`).
""")
