import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import altair as alt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Sistem AI Agentic Enterprise DIY",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- DATA KASUS (HARDCODED FROM PDF) ---
# Data Produsen (Kapasitas, Stok Awal)
DATA_MANUFACTURERS = [
    {"name": "Produsen A", "capacity": 1200000, "initial": 660000},
    {"name": "Produsen B", "capacity": 1000000, "initial": 550000},
    {"name": "Produsen C", "capacity": 800000, "initial": 440000},
    {"name": "Produsen D", "capacity": 700000, "initial": 385000},
    {"name": "Produsen E", "capacity": 600000, "initial": 330000},
]

# Data Distributor (Kapasitas, Stok Awal)
DATA_DISTRIBUTORS = [
    {"name": "Distributor 1", "capacity": 500000, "initial": 300000},
    {"name": "Distributor 2", "capacity": 450000, "initial": 270000},
    {"name": "Distributor 3", "capacity": 400000, "initial": 240000},
    {"name": "Distributor 4", "capacity": 380000, "initial": 228000},
    {"name": "Distributor 5", "capacity": 360000, "initial": 216000},
    {"name": "Distributor 6", "capacity": 340000, "initial": 204000},
    {"name": "Distributor 7", "capacity": 320000, "initial": 192000},
    {"name": "Distributor 8", "capacity": 300000, "initial": 180000},
]

# Data Ritel (Kategori, Rata-rata Stok, Total Unit Real - kita akan sampling)
DATA_RETAIL_TYPES = [
    {"type": "Supermarket", "avg_stock": 2500},
    {"type": "E-commerce", "avg_stock": 3000},
    {"type": "Sekolah/Univ", "avg_stock": 2000},
    {"type": "RS & Klinik", "avg_stock": 3500},
    {"type": "Restoran", "avg_stock": 1800},
    {"type": "Hotel", "avg_stock": 4000},
    {"type": "Toko Kelontong", "avg_stock": 1500},
]

# Konstanta Biaya
COST_HOLDING_RETAIL = 100  # per unit (rupiah scaling)
COST_HOLDING_DIST = 50
COST_HOLDING_MFG = 20
COST_STOCKOUT_RETAIL = 5000
COST_STOCKOUT_DIST = 2000
COST_STOCKOUT_MFG = 1000

# --- BASE AGENT CLASS (Q-LEARNING) ---
class QLearningAgent:
    def __init__(self, name, capacity, initial_stock, role, lr=0.1, gamma=0.9, epsilon=0.2):
        self.name = name
        self.capacity = capacity
        self.stock = initial_stock
        self.role = role
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Define Actions relative to capacity to scale with Agent size
        # Actions: 0%, 5%, 10%, 20% of capacity refill/production
        self.actions = [0, int(0.05 * capacity), int(0.1 * capacity), int(0.2 * capacity)]
        
        # Q-Table: State (StockBucket 0-10, DemandLevel 0-2) -> Action Index
        self.q_table = np.zeros((11, 3, len(self.actions)))
        self.total_reward = 0
        self.last_state = None
        self.last_action_idx = None

    def get_state(self, incoming_demand):
        # Diskritisasi Stok (0-10)
        stock_idx = min(int((self.stock / self.capacity) * 10), 10)
        
        # Diskritisasi Permintaan Relative terhadap kemampuan supply (avg action)
        avg_supply_ability = sum(self.actions) / len(self.actions)
        # Avoid division by zero
        if avg_supply_ability == 0: avg_supply_ability = 1

        if incoming_demand < avg_supply_ability * 0.5: demand_lvl = 0 # Low
        elif incoming_demand < avg_supply_ability * 1.5: demand_lvl = 1 # Normal
        else: demand_lvl = 2 # High / Rush
        
        return (stock_idx, demand_lvl)

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(self.actions)))
        stock_idx, demand_lvl = state
        return np.argmax(self.q_table[stock_idx, demand_lvl])

    def learn(self, reward, next_state):
        if self.last_state is not None and self.last_action_idx is not None:
            s_stock, s_dem = self.last_state
            ns_stock, ns_dem = next_state
            a = self.last_action_idx
            old_val = self.q_table[s_stock, s_dem, a]
            next_max = np.max(self.q_table[ns_stock, ns_dem])
            new_val = old_val + self.lr * (reward + self.gamma * next_max - old_val)
            self.q_table[s_stock, s_dem, a] = new_val
        self.total_reward += reward

# --- IMPLEMENTASI SPESIFIK AGEN ---
class RetailAgent(QLearningAgent):
    def step(self, customer_demand, received_stock):
        self.stock = min(self.stock + received_stock, self.capacity)
        sales = min(self.stock, customer_demand)
        lost_sales = customer_demand - sales
        self.stock -= sales
        
        # Reward Function
        reward = -(self.stock * COST_HOLDING_RETAIL) - (lost_sales * COST_STOCKOUT_RETAIL)
        if self.stock >= self.capacity: reward -= (COST_STOCKOUT_RETAIL * 10) # Overstock penalty
        
        current_state = self.get_state(customer_demand)
        self.learn(reward, current_state)
        
        action_idx = self.choose_action(current_state)
        order_qty = self.actions[action_idx]
        
        self.last_state = current_state
        self.last_action_idx = action_idx
        return order_qty, sales, lost_sales, reward

class DistributorAgent(QLearningAgent):
    def step(self, retail_orders, received_stock):
        self.stock = min(self.stock + received_stock, self.capacity)
        fulfilled = min(self.stock, retail_orders)
        unfulfilled = retail_orders - fulfilled
        self.stock -= fulfilled
        
        reward = -(self.stock * COST_HOLDING_DIST) - (unfulfilled * COST_STOCKOUT_DIST)
        if self.stock >= self.capacity: reward -= (COST_STOCKOUT_DIST * 10)
        
        current_state = self.get_state(retail_orders)
        self.learn(reward, current_state)
        
        action_idx = self.choose_action(current_state)
        order_qty = self.actions[action_idx]
        
        self.last_state = current_state
        self.last_action_idx = action_idx
        return order_qty, fulfilled, unfulfilled, reward

class ManufacturerAgent(QLearningAgent):
    def step(self, dist_orders):
        production_qty = 0
        if self.last_action_idx is not None:
             production_qty = self.actions[self.last_action_idx]
        
        self.stock = min(self.stock + production_qty, self.capacity)
        fulfilled = min(self.stock, dist_orders)
        unfulfilled = dist_orders - fulfilled
        self.stock -= fulfilled
        
        reward = -(self.stock * COST_HOLDING_MFG) - (unfulfilled * COST_STOCKOUT_MFG)
        
        current_state = self.get_state(dist_orders)
        self.learn(reward, current_state)
        
        action_idx = self.choose_action(current_state)
        
        self.last_state = current_state
        self.last_action_idx = action_idx
        return production_qty, fulfilled, unfulfilled, reward

# --- UTILS ---
def generate_demand(day, scenario, avg_stock):
    # Base demand is proportional to avg_stock (consumption rate ~ 10-20% of stock daily)
    base = avg_stock * 0.15 
    noise = random.randint(int(-0.05*avg_stock), int(0.1*avg_stock))
    
    demand = base + noise
    
    if scenario == "Seasonal Rush (Liburan)":
        if 20 <= day <= 25 or 50 <= day <= 55: return demand * 2.5
    elif scenario == "Panic Buying (Random Rush)":
        if random.random() < 0.15: return demand * 4.0
    elif scenario == "Random (Pola Acak)":
         return base * random.uniform(0.5, 2.0)
         
    return max(0, demand)

# --- MAIN APP ---
def main():
    st.title("💧 Sistem AI Agentic Enterprise DIY")
    st.markdown("""
    **Kasus A: Pengendalian Stok Air Minum Mineral DIY**
    
    Simulasi Multi-Agent System (MAS) dengan label dan parameter sesuai dokumen kasus.
    * **Produsen**: A, B, C, D, E
    * **Distributor**: 1 s.d. 8
    * **Ritel**: Supermarket, RS, Sekolah, dll.
    """)

    # --- SESSION STATE INIT ---
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None
    if 'trained_agents' not in st.session_state:
        st.session_state.trained_agents = {}

    # Sidebar
    with st.sidebar:
        st.header("🎮 Parameter Simulasi")
        
        st.subheader("Sampel Ritel")
        n_retailers_per_type = st.slider("Jumlah Ritel per Kategori", 1, 5, 2, help="Total agen ritel = Jumlah ini x 7 Kategori")
        
        st.divider()
        n_days = st.slider("Durasi (Hari)", 30, 200, 60)
        speed = st.slider("Kecepatan (detik)", 0.0, 1.0, 0.05)
        scenario = st.selectbox("Skenario Permintaan", ["Normal", "Seasonal Rush (Liburan)", "Panic Buying (Random Rush)", "Random (Pola Acak)"])
        
        st.divider()
        st.header("🧠 Parameter AI")
        epsilon = st.slider("Eksplorasi (Epsilon)", 0.0, 1.0, 0.2)
        lr = st.number_input("Learning Rate", 0.01, 0.5, 0.1)
        
        start_btn = st.button("🚀 Jalankan Simulasi", type="primary")

    # Define Tabs
    tab_live, tab_analysis, tab_policy = st.tabs(["📡 Live Network Status", "📊 Analisis Kinerja", "🧠 Kebijakan Agen (Brain)"])

    # --- SIMULATION LOGIC ---
    if start_btn:
        # 1. INITIALIZE AGENTS BASED ON PDF DATA
        
        # Manufacturers (Fixed 5)
        manufacturers = []
        for d in DATA_MANUFACTURERS:
            manufacturers.append(ManufacturerAgent(d['name'], d['capacity'], d['initial'], "Manufacturer", lr, gamma=0.9, epsilon=epsilon))
            
        # Distributors (Fixed 8)
        distributors = []
        for d in DATA_DISTRIBUTORS:
            distributors.append(DistributorAgent(d['name'], d['capacity'], d['initial'], "Distributor", lr, gamma=0.9, epsilon=epsilon))
            
        # Retailers (Sampled based on Types)
        retailers = []
        retail_id_counter = 1
        for r_type in DATA_RETAIL_TYPES:
            for i in range(n_retailers_per_type):
                # Capacity assumed slightly higher than avg stock to allow fluctuation
                capacity = r_type['avg_stock'] * 1.5 
                initial = r_type['avg_stock']
                name = f"{r_type['type']} #{i+1}"
                retailers.append(RetailAgent(name, capacity, initial, "Retail", lr, gamma=0.9, epsilon=epsilon))
                retail_id_counter += 1

        # MAPPING
        # Retailers assigned to random Distributor
        retail_map = {r.name: random.choice(distributors) for r in retailers}
        # Distributors assigned to random Manufacturer
        dist_map = {d.name: random.choice(manufacturers) for d in distributors}
        
        shipments_to_retail = {r.name: 0 for r in retailers}
        shipments_to_dist = {d.name: 0 for d in distributors}
        data_log = []

        # UI Placeholders
        with tab_live:
            st.info("Simulasi berjalan... Mengkoordinasikan Produsen A-E, Distributor 1-8, dan Ritel DIY.")
            col_net1, col_net2, col_net3 = st.columns(3)
            with col_net1: metric_ret = st.empty()
            with col_net2: metric_dist = st.empty()
            with col_net3: metric_mfg = st.empty()
            chart_placeholder = st.empty()
            progress_bar = st.progress(0)

        # Loop
        for day in range(1, n_days + 1):
            # 1. Retail Step
            day_retail_orders = {d.name: 0 for d in distributors}
            day_retail_metrics = []
            
            for r in retailers:
                # Cari data avg_stock untuk tipe ini untuk generate demand yang sesuai
                # (Simplifikasi: kita pakai kapasitas/1.5 sebagai proxy avg stock)
                proxy_avg_stock = r.capacity / 1.5
                local_demand = generate_demand(day, scenario, proxy_avg_stock)
                
                qty_order, sales, lost, reward = r.step(local_demand, shipments_to_retail[r.name])
                
                upstream_dist = retail_map[r.name]
                day_retail_orders[upstream_dist.name] += qty_order
                
                day_retail_metrics.append({"Day": day, "Agent": r.name, "Type": "Retail", "Stock": r.stock, "Demand": local_demand, "Order": qty_order, "LostSales": lost, "Reward": reward})

            # 2. Distributor Step
            day_dist_orders = {m.name: 0 for m in manufacturers}
            day_dist_metrics = []
            next_shipments_to_retail = {r.name: 0 for r in retailers}
            
            for d in distributors:
                incoming_req = day_retail_orders[d.name]
                qty_order, fulfilled, unfulfilled, reward = d.step(incoming_req, shipments_to_dist[d.name])
                
                upstream_mfg = dist_map[d.name]
                day_dist_orders[upstream_mfg.name] += qty_order
                
                # Allocation Logic (Pro-Rata)
                fill_rate = fulfilled / incoming_req if incoming_req > 0 else 1
                connected_retailers = [r for r in retailers if retail_map[r.name] == d]
                for metric in day_retail_metrics:
                    if metric["Agent"] in [cr.name for cr in connected_retailers]:
                        next_shipments_to_retail[metric["Agent"]] = metric["Order"] * fill_rate

                day_dist_metrics.append({"Day": day, "Agent": d.name, "Type": "Distributor", "Stock": d.stock, "Demand": incoming_req, "Order": qty_order, "LostSales": unfulfilled, "Reward": reward})

            # 3. Manufacturer Step
            day_mfg_metrics = []
            next_shipments_to_dist = {d.name: 0 for d in distributors}
            
            for m in manufacturers:
                incoming_req = day_dist_orders[m.name]
                prod_qty, fulfilled, unfulfilled, reward = m.step(incoming_req)
                
                # Allocation Logic
                fill_rate = fulfilled / incoming_req if incoming_req > 0 else 1
                connected_dists = [d for d in distributors if dist_map[d.name] == m]
                for metric in day_dist_metrics:
                    if metric["Agent"] in [cd.name for cd in connected_dists]:
                        next_shipments_to_dist[metric["Agent"]] = metric["Order"] * fill_rate

                day_mfg_metrics.append({"Day": day, "Agent": m.name, "Type": "Manufacturer", "Stock": m.stock, "Demand": incoming_req, "Order": prod_qty, "LostSales": unfulfilled, "Reward": reward})

            shipments_to_retail = next_shipments_to_retail
            shipments_to_dist = next_shipments_to_dist
            data_log.extend(day_retail_metrics + day_dist_metrics + day_mfg_metrics)

            # UI Update
            if day % 2 == 0:
                avg_ret_stock = sum(r.stock for r in retailers) / len(retailers)
                avg_dist_stock = sum(d.stock for d in distributors) / len(distributors)
                avg_mfg_stock = sum(m.stock for m in manufacturers) / len(manufacturers)
                
                metric_ret.metric("Ritel (Avg Stock)", f"{int(avg_ret_stock):,}")
                metric_dist.metric("Distributor (Avg Stock)", f"{int(avg_dist_stock):,}")
                metric_mfg.metric("Produsen (Avg Stock)", f"{int(avg_mfg_stock):,}")
                
                df_curr = pd.DataFrame(data_log)
                df_agg = df_curr.groupby(['Day', 'Type'])['Stock'].mean().reset_index()
                chart = alt.Chart(df_agg).mark_line().encode(x='Day', y='Stock', color='Type').properties(height=300)
                chart_placeholder.altair_chart(chart, use_container_width=True)
            
            progress_bar.progress(day/n_days)
            time.sleep(speed)
        
        st.success("Simulasi Selesai! Data tersimpan.")
        
        st.session_state.sim_results = pd.DataFrame(data_log)
        st.session_state.trained_agents = {
            'retailers': retailers,
            'distributors': distributors,
            'manufacturers': manufacturers
        }

    # --- DISPLAY RESULTS ---
    if st.session_state.sim_results is not None:
        df_res = st.session_state.sim_results
        
        with tab_live:
            if not start_btn:
                st.info(f"Hasil Terakhir: {len(df_res['Day'].unique())} Hari Simulasi")
                df_agg = df_res.groupby(['Day', 'Type'])['Stock'].mean().reset_index()
                chart = alt.Chart(df_agg).mark_line().encode(x='Day', y='Stock', color='Type').properties(title="Rata-rata Stok Harian", height=300)
                st.altair_chart(chart, use_container_width=True)

        with tab_analysis:
            st.header("Analisis Kinerja Agen")
            view_type = st.selectbox("Pilih Tier untuk Detail", ["Retail", "Distributor", "Manufacturer"])
            df_filtered = df_res[df_res['Type'] == view_type]
            
            st.subheader(f"Level Stok: {view_type}")
            chart_stock = alt.Chart(df_filtered).mark_line().encode(
                x='Day', y='Stock', color='Agent', tooltip=['Day', 'Agent', 'Stock']
            ).interactive()
            st.altair_chart(chart_stock, use_container_width=True)
            
            st.subheader("Total Lost Sales (Permintaan Tak Terpenuhi)")
            chart_lost = alt.Chart(df_filtered).mark_bar().encode(
                x=alt.X('Agent', sort='-y'), y='sum(LostSales)', color='Agent'
            )
            st.altair_chart(chart_lost, use_container_width=True)

        with tab_policy:
            st.header("Inspeksi Kebijakan (Agent Brain)")
            saved_agents = st.session_state.trained_agents
            all_agents_list = saved_agents.get('retailers', []) + saved_agents.get('distributors', []) + saved_agents.get('manufacturers', [])
            
            if all_agents_list:
                agent_names = [a.name for a in all_agents_list]
                selected_agent_name = st.selectbox("Pilih Agen untuk Heatmap", agent_names)
                selected_agent = next((a for a in all_agents_list if a.name == selected_agent_name), None)
                
                if selected_agent:
                    q_data = []
                    for s in range(11): 
                        for d in range(3): 
                            best_action_idx = np.argmax(selected_agent.q_table[s, d])
                            action_val = selected_agent.actions[best_action_idx]
                            q_data.append({
                                "Stock Level (%)": s*10,
                                "Incoming Demand": ["Low", "Normal", "High"][d],
                                "Action (Qty)": action_val
                            })
                    
                    df_q = pd.DataFrame(q_data)
                    hm = alt.Chart(df_q).mark_rect().encode(
                        x='Incoming Demand:O',
                        y=alt.Y('Stock Level (%):O', sort='descending'),
                        color=alt.Color('Action (Qty):Q', scale=alt.Scale(scheme='viridis')),
                        tooltip=['Stock Level (%)', 'Incoming Demand', 'Action (Qty)']
                    ).properties(title=f"Policy: {selected_agent.name}", height=400)
                    st.altair_chart(hm, use_container_width=True)
                    st.caption(f"Kapasitas Agen: {selected_agent.capacity:,}. Aksi adalah jumlah pemesanan/produksi.")

if __name__ == "__main__":
    main()