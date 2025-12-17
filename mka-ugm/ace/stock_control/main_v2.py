import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import altair as alt

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Multi-Agent AI Supply Chain",
    page_icon="🏭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS CUSTOM ---
st.markdown("""
<style>
    .stMetric {
        background-color: #1e2130;
        padding: 10px;
        border-radius: 8px;
        border: 1px solid #31333F;
    }
    .metric-label { font-size: 0.8em; color: #aaa; }
    .metric-value { font-size: 1.5em; font-weight: bold; color: #fff; }
</style>
""", unsafe_allow_html=True)

# --- KONSTANTA GLOBAL ---
COST_HOLDING_RETAIL = 1
COST_HOLDING_DIST = 0.5
COST_HOLDING_MFG = 0.2
COST_STOCKOUT_RETAIL = 50
COST_STOCKOUT_DIST = 20
COST_STOCKOUT_MFG = 10

CAPACITY_RETAIL = 3000
CAPACITY_DIST = 20000
CAPACITY_MFG = 100000

ACTIONS_RETAIL = [0, 500, 1000, 2000]
ACTIONS_DIST = [0, 2000, 5000, 10000]
ACTIONS_MFG = [0, 5000, 10000, 20000]

# --- BASE AGENT CLASS (Q-LEARNING) ---
class QLearningAgent:
    def __init__(self, name, actions, capacity, lr=0.1, gamma=0.9, epsilon=0.2):
        self.name = name
        self.actions = actions
        self.capacity = capacity
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        self.q_table = np.zeros((11, 3, len(actions)))
        self.stock = capacity * 0.5
        self.total_reward = 0
        self.last_state = None
        self.last_action_idx = None

    def get_state(self, incoming_demand):
        stock_idx = min(int((self.stock / self.capacity) * 10), 10)
        avg_action = sum(self.actions)/len(self.actions)
        if incoming_demand < avg_action * 0.5: demand_lvl = 0 
        elif incoming_demand < avg_action * 1.5: demand_lvl = 1 
        else: demand_lvl = 2 
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
        
        reward = -(self.stock * COST_HOLDING_RETAIL) - (lost_sales * COST_STOCKOUT_RETAIL)
        if self.stock >= self.capacity: reward -= 500
        
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
        if self.stock >= self.capacity: reward -= 200
        
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
def generate_demand(day, scenario):
    base = 500
    noise = random.randint(-100, 150)
    if scenario == "Seasonal Rush":
        if 20 <= day <= 30 or 60 <= day <= 70: return base * 2.5 + noise
    elif scenario == "Panic Buying":
        if random.random() < 0.1: return base * 4 + noise
    return max(0, base + noise)

# --- MAIN APP ---
def main():
    st.title("🤖 Multi-Agent Supply Chain Network")
    st.markdown("Simulasi jaringan dengan **Banyak Retailer**, **Banyak Distributor**, dan **Pabrik**.")

    # --- SESSION STATE INIT ---
    if 'sim_results' not in st.session_state:
        st.session_state.sim_results = None
    if 'trained_agents' not in st.session_state:
        st.session_state.trained_agents = {}

    # Sidebar
    with st.sidebar:
        st.header("🎮 Kontrol Simulasi")
        n_retailers = st.slider("Jumlah Retailer", 1, 10, 3)
        n_distributors = st.slider("Jumlah Distributor", 1, 5, 2)
        n_manufacturers = st.slider("Jumlah Pabrik", 1, 3, 1)
        st.divider()
        n_days = st.slider("Durasi (Hari)", 30, 200, 60)
        speed = st.slider("Kecepatan (detik)", 0.0, 1.0, 0.05)
        scenario = st.selectbox("Skenario Permintaan", ["Normal", "Seasonal Rush", "Panic Buying"])
        st.divider()
        st.header("🧠 Parameter AI")
        epsilon = st.slider("Eksplorasi (Epsilon)", 0.0, 1.0, 0.2)
        lr = st.number_input("Learning Rate", 0.01, 0.5, 0.1)
        
        start_btn = st.button("Mulai Simulasi", type="primary")

    # Define Tabs (Global Scope)
    tab_live, tab_analysis, tab_policy = st.tabs(["📡 Live Network", "📊 Performance Analytics", "🧠 Agent Brains"])

    # --- SIMULATION LOGIC ---
    if start_btn:
        # Reset Data
        retailers = [RetailAgent(f"R{i+1}", ACTIONS_RETAIL, CAPACITY_RETAIL, lr, epsilon=epsilon) for i in range(n_retailers)]
        distributors = [DistributorAgent(f"D{i+1}", ACTIONS_DIST, CAPACITY_DIST, lr, epsilon=epsilon) for i in range(n_distributors)]
        manufacturers = [ManufacturerAgent(f"M{i+1}", ACTIONS_MFG, CAPACITY_MFG, lr, epsilon=epsilon) for i in range(n_manufacturers)]

        retail_map = {r.name: random.choice(distributors) for r in retailers}
        dist_map = {d.name: random.choice(manufacturers) for d in distributors}
        
        shipments_to_retail = {r.name: 0 for r in retailers}
        shipments_to_dist = {d.name: 0 for d in distributors}
        data_log = []

        # UI Placeholders for Animation
        with tab_live:
            st.info("Simulasi sedang berjalan... Harap tunggu hingga selesai untuk analisis mendalam.")
            col_net1, col_net2, col_net3 = st.columns(3)
            with col_net1: 
                st.caption("Retailers (Avg Stock)")
                metric_ret = st.empty()
            with col_net2: 
                st.caption("Distributors (Avg Stock)")
                metric_dist = st.empty()
            with col_net3: 
                st.caption("Manufacturers (Avg Stock)")
                metric_mfg = st.empty()
            chart_placeholder = st.empty()
            progress_bar = st.progress(0)

        # Loop
        for day in range(1, n_days + 1):
            # 1. Retail
            day_retail_orders = {d.name: 0 for d in distributors}
            day_retail_metrics = []
            for r in retailers:
                local_demand = generate_demand(day, scenario) * (random.uniform(0.8, 1.2))
                qty_order, sales, lost, reward = r.step(local_demand, shipments_to_retail[r.name])
                day_retail_orders[retail_map[r.name].name] += qty_order
                day_retail_metrics.append({"Day": day, "Agent": r.name, "Type": "Retail", "Stock": r.stock, "Demand": local_demand, "Order": qty_order, "LostSales": lost, "Reward": reward})

            # 2. Dist
            day_dist_orders = {m.name: 0 for m in manufacturers}
            day_dist_metrics = []
            next_shipments_to_retail = {r.name: 0 for r in retailers}
            for d in distributors:
                qty_order, fulfilled, unfulfilled, reward = d.step(day_retail_orders[d.name], shipments_to_dist[d.name])
                day_dist_orders[dist_map[d.name].name] += qty_order
                
                # Pro-rata distribution
                incoming_req = day_retail_orders[d.name]
                fill_rate = fulfilled / incoming_req if incoming_req > 0 else 1
                connected_retailers = [r for r in retailers if retail_map[r.name] == d]
                for metric in day_retail_metrics:
                    if metric["Agent"] in [cr.name for cr in connected_retailers]:
                        next_shipments_to_retail[metric["Agent"]] = metric["Order"] * fill_rate

                day_dist_metrics.append({"Day": day, "Agent": d.name, "Type": "Distributor", "Stock": d.stock, "Demand": day_retail_orders[d.name], "Order": qty_order, "LostSales": unfulfilled, "Reward": reward})

            # 3. Mfg
            day_mfg_metrics = []
            next_shipments_to_dist = {d.name: 0 for d in distributors}
            for m in manufacturers:
                prod_qty, fulfilled, unfulfilled, reward = m.step(day_dist_orders[m.name])
                
                incoming_req = day_dist_orders[m.name]
                fill_rate = fulfilled / incoming_req if incoming_req > 0 else 1
                connected_dists = [d for d in distributors if dist_map[d.name] == m]
                for metric in day_dist_metrics:
                    if metric["Agent"] in [cd.name for cd in connected_dists]:
                        next_shipments_to_dist[metric["Agent"]] = metric["Order"] * fill_rate

                day_mfg_metrics.append({"Day": day, "Agent": m.name, "Type": "Manufacturer", "Stock": m.stock, "Demand": day_dist_orders[m.name], "Order": prod_qty, "LostSales": unfulfilled, "Reward": reward})

            shipments_to_retail = next_shipments_to_retail
            shipments_to_dist = next_shipments_to_dist
            data_log.extend(day_retail_metrics + day_dist_metrics + day_mfg_metrics)

            # Animation Update
            if day % 2 == 0:
                avg_ret_stock = sum(r.stock for r in retailers) / n_retailers
                avg_dist_stock = sum(d.stock for d in distributors) / n_distributors
                avg_mfg_stock = sum(m.stock for m in manufacturers) / n_manufacturers
                metric_ret.metric("Avg Retail Stock", f"{int(avg_ret_stock)}")
                metric_dist.metric("Avg Dist Stock", f"{int(avg_dist_stock)}")
                metric_mfg.metric("Avg Mfg Stock", f"{int(avg_mfg_stock)}")
                
                df_curr = pd.DataFrame(data_log)
                df_agg = df_curr.groupby(['Day', 'Type'])['Stock'].mean().reset_index()
                chart = alt.Chart(df_agg).mark_line().encode(x='Day', y='Stock', color='Type').properties(height=300)
                chart_placeholder.altair_chart(chart, use_container_width=True)
            
            progress_bar.progress(day/n_days)
            time.sleep(speed)
        
        st.success("Simulasi Selesai! Data tersimpan di memori.")
        
        # SAVE TO SESSION STATE
        st.session_state.sim_results = pd.DataFrame(data_log)
        st.session_state.trained_agents = {
            'retailers': retailers,
            'distributors': distributors,
            'manufacturers': manufacturers
        }

    # --- POST-SIMULATION DISPLAY LOGIC ---
    # This block runs every time the script reruns (e.g. tab switch), provided data exists
    if st.session_state.sim_results is not None:
        df_res = st.session_state.sim_results
        
        # 1. Fill 'Live Network' with final summary (so it's not empty on refresh)
        with tab_live:
            if not start_btn: # Only show this if we aren't currently running the loop
                st.success(f"Menampilkan hasil simulasi terakhir ({len(df_res['Day'].unique())} Hari).")
                df_agg = df_res.groupby(['Day', 'Type'])['Stock'].mean().reset_index()
                chart = alt.Chart(df_agg).mark_line().encode(x='Day', y='Stock', color='Type').properties(title="Rata-rata Stok Harian", height=300)
                st.altair_chart(chart, use_container_width=True)

        # 2. Performance Analytics
        with tab_analysis:
            st.header("Analisis Kinerja Agen")
            view_type = st.selectbox("Pilih Tier untuk Analisis Detail", ["Retail", "Distributor", "Manufacturer"])
            df_filtered = df_res[df_res['Type'] == view_type]
            
            st.subheader(f"Dinamika Stok: {view_type} Agents")
            chart_stock = alt.Chart(df_filtered).mark_line().encode(
                x='Day', y='Stock', color='Agent', tooltip=['Day', 'Agent', 'Stock']
            ).interactive()
            st.altair_chart(chart_stock, use_container_width=True)
            
            st.subheader("Learning Performance (Avg Reward)")
            chart_rew = alt.Chart(df_filtered).mark_line().encode(
                x='Day', y='mean(Reward)', color='Agent'
            ).interactive()
            st.altair_chart(chart_rew, use_container_width=True)
            
            st.subheader("Total Lost Sales / Unfulfilled")
            chart_lost = alt.Chart(df_filtered).mark_bar().encode(
                x='Agent', y='sum(LostSales)', color='Agent'
            )
            st.altair_chart(chart_lost, use_container_width=True)

        # 3. Agent Brains (Policy Heatmaps)
        with tab_policy:
            st.header("Inspeksi Kebijakan (Agent Brain)")
            
            # Retrieve agents from session state
            saved_agents = st.session_state.trained_agents
            all_agents_list = saved_agents.get('retailers', []) + saved_agents.get('distributors', []) + saved_agents.get('manufacturers', [])
            
            if all_agents_list:
                agent_names = [a.name for a in all_agents_list]
                selected_agent_name = st.selectbox("Pilih Agen untuk melihat Q-Table Heatmap", agent_names)
                
                selected_agent = next((a for a in all_agents_list if a.name == selected_agent_name), None)
                
                if selected_agent:
                    q_data = []
                    for s in range(11): 
                        for d in range(3): 
                            best_action_idx = np.argmax(selected_agent.q_table[s, d])
                            q_data.append({
                                "Stock Level": f"{s*10}%",
                                "Incoming Demand": ["Low", "Normal", "High"][d],
                                "Action": selected_agent.actions[best_action_idx]
                            })
                    
                    df_q = pd.DataFrame(q_data)
                    hm = alt.Chart(df_q).mark_rect().encode(
                        x='Incoming Demand:O',
                        y=alt.Y('Stock Level:O', sort='descending'),
                        color=alt.Color('Action:Q', scale=alt.Scale(scheme='viridis')),
                        tooltip=['Stock Level', 'Incoming Demand', 'Action']
                    ).properties(title=f"Policy Learned by {selected_agent.name}", height=400)
                    st.altair_chart(hm, use_container_width=True)
            else:
                st.warning("Data agen tidak ditemukan. Silakan jalankan simulasi terlebih dahulu.")

if __name__ == "__main__":
    main()