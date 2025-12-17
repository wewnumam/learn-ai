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

# --- KONSTANTA GLOBAL ---
COST_HOLDING_RETAIL = 1
COST_HOLDING_DIST = 0.5
COST_HOLDING_MFG = 0.2
COST_STOCKOUT_RETAIL = 50  # Lost sale penalty tinggi
COST_STOCKOUT_DIST = 20    # Penalty gagal kirim ke ritel
COST_STOCKOUT_MFG = 10     # Penalty gagal kirim ke dist

CAPACITY_RETAIL = 3000
CAPACITY_DIST = 20000
CAPACITY_MFG = 100000

# Aksi Diskrit (Order Qty / Production Qty)
ACTIONS_RETAIL = [0, 500, 1000, 2000]
ACTIONS_DIST = [0, 2000, 5000, 10000]
ACTIONS_MFG = [0, 5000, 10000, 20000] # Production levels

# --- BASE AGENT CLASS (Q-LEARNING) ---
class QLearningAgent:
    def __init__(self, name, actions, capacity, lr=0.1, gamma=0.9, epsilon=0.2):
        self.name = name
        self.actions = actions
        self.capacity = capacity
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        
        # Q-Table: State (StockBucket, DemandLevel) -> Action
        # StockBucket: 0-10, DemandLevel: 0-2 (Low, Med, High)
        self.q_table = np.zeros((11, 3, len(actions)))
        self.stock = capacity * 0.5 # Mulai dengan stok 50%
        self.total_reward = 0
        self.last_state = None
        self.last_action_idx = None

    def get_state(self, incoming_demand):
        # Diskritisasi Stok (0-10)
        stock_idx = min(int((self.stock / self.capacity) * 10), 10)
        
        # Diskritisasi Permintaan/Order Masuk
        # Threshold ini relatif terhadap kapasitas/aksi rata-rata agent
        avg_action = sum(self.actions)/len(self.actions)
        if incoming_demand < avg_action * 0.5: demand_lvl = 0 # Low
        elif incoming_demand < avg_action * 1.5: demand_lvl = 1 # Med
        else: demand_lvl = 2 # High
        
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
            
            # Q-Learning Update
            new_val = old_val + self.lr * (reward + self.gamma * next_max - old_val)
            self.q_table[s_stock, s_dem, a] = new_val
            
        self.total_reward += reward

# --- IMPLEMENTASI SPESIFIK AGEN ---

class RetailAgent(QLearningAgent):
    def step(self, customer_demand, received_stock):
        # 1. Terima barang dari Distributor
        self.stock = min(self.stock + received_stock, self.capacity)
        
        # 2. Penuhi permintaan Customer
        sales = min(self.stock, customer_demand)
        lost_sales = customer_demand - sales
        self.stock -= sales
        
        # 3. Hitung Reward
        # Penalti stok habis sangat besar, biaya simpan kecil
        reward = -(self.stock * COST_HOLDING_RETAIL) - (lost_sales * COST_STOCKOUT_RETAIL)
        if self.stock >= self.capacity: reward -= 500 # Overstock penalty
        
        # 4. Tentukan State Baru & Belajar
        current_state = self.get_state(customer_demand)
        self.learn(reward, current_state)
        
        # 5. Ambil Aksi (Order ke Distributor untuk besok)
        action_idx = self.choose_action(current_state)
        order_qty = self.actions[action_idx]
        
        # Simpan konteks untuk learning step berikutnya
        self.last_state = current_state
        self.last_action_idx = action_idx
        
        return order_qty, sales, lost_sales, reward

class DistributorAgent(QLearningAgent):
    def step(self, retail_orders, received_stock):
        # 1. Terima barang dari Pabrik
        self.stock = min(self.stock + received_stock, self.capacity)
        
        # 2. Penuhi pesanan Retailer
        fulfilled = min(self.stock, retail_orders)
        unfulfilled = retail_orders - fulfilled
        self.stock -= fulfilled
        
        # 3. Hitung Reward
        reward = -(self.stock * COST_HOLDING_DIST) - (unfulfilled * COST_STOCKOUT_DIST)
        if self.stock >= self.capacity: reward -= 200
        
        # 4. State & Learn
        current_state = self.get_state(retail_orders)
        self.learn(reward, current_state)
        
        # 5. Action (Order ke Pabrik)
        action_idx = self.choose_action(current_state)
        order_qty = self.actions[action_idx]
        
        self.last_state = current_state
        self.last_action_idx = action_idx
        
        return order_qty, fulfilled, unfulfilled, reward

class ManufacturerAgent(QLearningAgent):
    def step(self, dist_orders):
        # 1. Produksi (Hasil dari keputusan HARI SEBELUMNYA masuk gudang hari ini)
        # Di simulasi sederhana ini, kita anggap produksi instan masuk stok di awal hari
        # atau kita gunakan last_action sebagai produksi hari ini.
        production_qty = 0
        if self.last_action_idx is not None:
             production_qty = self.actions[self.last_action_idx]
        
        self.stock = min(self.stock + production_qty, self.capacity)
        
        # 2. Penuhi pesanan Distributor
        fulfilled = min(self.stock, dist_orders)
        unfulfilled = dist_orders - fulfilled
        self.stock -= fulfilled
        
        # 3. Reward
        # Pabrik punya biaya produksi (implisit) dan smoothing cost (tidak dimodelkan kompleks disini)
        reward = -(self.stock * COST_HOLDING_MFG) - (unfulfilled * COST_STOCKOUT_MFG)
        
        # 4. State & Learn
        current_state = self.get_state(dist_orders)
        self.learn(reward, current_state)
        
        # 5. Action (Set Level Produksi untuk besok)
        action_idx = self.choose_action(current_state)
        
        self.last_state = current_state
        self.last_action_idx = action_idx
        
        return production_qty, fulfilled, unfulfilled, reward

# --- UTILS ---
def generate_demand(day, scenario):
    base = 500
    noise = random.randint(-100, 150)
    
    if scenario == "Seasonal Rush":
        if 20 <= day <= 30 or 60 <= day <= 70:
            return base * 2.5 + noise
    elif scenario == "Panic Buying":
        if random.random() < 0.1: # 10% chance spike
            return base * 4 + noise
            
    return max(0, base + noise)

# --- MAIN APP ---
def main():
    st.title("🤖 Enterprise AI Supply Chain Simulation")
    st.markdown("Simulasi koordinasi agen otonom (RL) dari Ritel ke Distributor hingga Pabrik.")

    # Sidebar
    with st.sidebar:
        st.header("🎮 Kontrol Simulasi")
        n_days = st.slider("Durasi (Hari)", 30, 200, 90)
        speed = st.slider("Kecepatan (detik)", 0.0, 1.0, 0.05)
        scenario = st.selectbox("Skenario Permintaan", ["Normal", "Seasonal Rush", "Panic Buying"])
        
        st.divider()
        st.header("🧠 Parameter AI")
        epsilon = st.slider("Eksplorasi (Epsilon)", 0.0, 1.0, 0.2, help="Seberapa sering agen mencoba hal baru vs menggunakan pengalaman.")
        lr = st.number_input("Learning Rate", 0.01, 0.5, 0.1)
        
        start_btn = st.button("Mulai Simulasi", type="primary")

    if start_btn:
        # Inisialisasi Agen
        retailer = RetailAgent("Retailer AI", ACTIONS_RETAIL, CAPACITY_RETAIL, lr=lr, epsilon=epsilon)
        distributor = DistributorAgent("Distributor AI", ACTIONS_DIST, CAPACITY_DIST, lr=lr, epsilon=epsilon)
        manufacturer = ManufacturerAgent("Factory AI", ACTIONS_MFG, CAPACITY_MFG, lr=lr, epsilon=epsilon)

        # Container Data
        data_log = []
        
        # Layout Utama
        tab_live, tab_analysis, tab_policy = st.tabs(["📡 Live Monitoring", "📊 Network Analysis", "🧠 Learned Policies"])
        
        with tab_live:
            col1, col2, col3 = st.columns(3)
            with col1: 
                st.subheader("🛒 Retailer")
                m_ret_stock = st.empty()
                m_ret_ord = st.empty()
            with col2: 
                st.subheader("🚛 Distributor")
                m_dist_stock = st.empty()
                m_dist_ord = st.empty()
            with col3: 
                st.subheader("🏭 Manufacturer")
                m_mfg_stock = st.empty()
                m_mfg_prod = st.empty()

            chart_placeholder = st.empty()
            
        # Variabel Delay Pengiriman (Lead Time 1 Hari)
        pending_retail_order = 0
        pending_dist_order = 0
        
        shipment_to_retail = 0
        shipment_to_dist = 0

        progress_bar = st.progress(0)

        for day in range(1, n_days + 1):
            # 1. Generate Demand Konsumen
            customer_demand = generate_demand(day, scenario)
            
            # 2. Langkah Retailer
            # Menerima barang yang dikirim Distributor KEMARIN (shipment_to_retail)
            ret_order_qty, sales, lost_sales, ret_reward = retailer.step(customer_demand, shipment_to_retail)
            
            # 3. Langkah Distributor
            # Menerima pesanan Retailer HARI INI (ret_order_qty)
            # Menerima barang dari Pabrik KEMARIN (shipment_to_dist)
            dist_order_qty, dist_fulfilled, dist_unfulfilled, dist_reward = distributor.step(ret_order_qty, shipment_to_dist)
            
            # Barang yang dipenuhi distributor hari ini akan sampai ke retail BESOK (Lead time 1 hari)
            # shipment_to_retail untuk iterasi selanjutnya = dist_fulfilled sekarang
            next_shipment_to_retail = dist_fulfilled 
            
            # 4. Langkah Pabrik
            # Menerima pesanan Distributor HARI INI
            prod_qty, mfg_fulfilled, mfg_unfulfilled, mfg_reward = manufacturer.step(dist_order_qty)
            
            # Barang dipenuhi pabrik hari ini sampai ke distributor BESOK
            next_shipment_to_dist = mfg_fulfilled
            
            # Update Logistik
            shipment_to_retail = next_shipment_to_retail
            shipment_to_dist = next_shipment_to_dist

            # Logging Data
            data_log.append({
                "Day": day,
                "Customer Demand": customer_demand,
                # Stocks
                "Stock Retail": retailer.stock,
                "Stock Dist": distributor.stock,
                "Stock Mfg": manufacturer.stock,
                # Decisions
                "Order Retail": ret_order_qty,
                "Order Dist": dist_order_qty,
                "Production": prod_qty,
                # Performance
                "Lost Sales Retail": lost_sales,
                "Unfulfilled Dist": dist_unfulfilled,
                "Reward Retail": ret_reward,
                "Reward Dist": dist_reward,
                "Reward Mfg": mfg_reward
            })

            # --- UPDATE UI REALTIME ---
            if day % 2 == 0 or day == 1: # Update tiap 2 frame agar smooth
                m_ret_stock.metric("Stock Lvl", f"{int(retailer.stock)}", delta=f"-{lost_sales} Missed" if lost_sales > 0 else "OK")
                m_ret_ord.metric("Last Action", f"Order {ret_order_qty}")
                
                m_dist_stock.metric("Stock Lvl", f"{int(distributor.stock)}", delta=f"-{dist_unfulfilled} Short" if dist_unfulfilled > 0 else "OK")
                m_dist_ord.metric("Last Action", f"Order {dist_order_qty}")
                
                m_mfg_stock.metric("Stock Lvl", f"{int(manufacturer.stock)}", delta=f"-{mfg_unfulfilled} Short" if mfg_unfulfilled > 0 else "OK")
                m_mfg_prod.metric("Last Action", f"Prod {prod_qty}")

                # Chart Gabungan
                df_curr = pd.DataFrame(data_log)
                base = alt.Chart(df_curr.tail(30)).encode(x='Day')
                
                line_ret = base.mark_line(color='#4CAF50').encode(y='Stock Retail', tooltip=['Day', 'Stock Retail'])
                line_dist = base.mark_line(color='#2196F3').encode(y='Stock Dist', tooltip=['Day', 'Stock Dist'])
                line_mfg = base.mark_line(color='#FF9800').encode(y='Stock Mfg', tooltip=['Day', 'Stock Mfg'])
                
                chart = alt.layer(line_ret, line_dist, line_mfg).properties(
                    title="Real-time Stock Levels (Green: Retail, Blue: Dist, Orange: Factory)",
                    height=300
                ).interactive()
                
                chart_placeholder.altair_chart(chart, use_container_width=True)
            
            progress_bar.progress(day/n_days)
            time.sleep(speed)

        st.success("Simulasi Selesai!")
        df_res = pd.DataFrame(data_log)

        # --- TAB ANALYSIS ---
        with tab_analysis:
            st.header("Analisis Kinerja Rantai Pasok")
            
            col_a1, col_a2 = st.columns(2)
            
            with col_a1:
                st.subheader("Total Lost Sales / Unfulfilled Orders")
                df_loss = df_res[['Day', 'Lost Sales Retail', 'Unfulfilled Dist']].melt('Day')
                chart_loss = alt.Chart(df_loss).mark_bar().encode(
                    x='Day',
                    y='value',
                    color='variable',
                    tooltip=['Day', 'variable', 'value']
                ).properties(height=300)
                st.altair_chart(chart_loss, use_container_width=True)
            
            with col_a2:
                st.subheader("Perkembangan Reward (Learning Curve)")
                # Rolling average reward untuk melihat tren belajar
                df_res['Roll_Reward_Retail'] = df_res['Reward Retail'].rolling(5).mean()
                df_res['Roll_Reward_Dist'] = df_res['Reward Dist'].rolling(5).mean()
                
                # FIXED: Added data types (:Q, :N) to fix Altair inference error with transform_fold
                chart_rew = alt.Chart(df_res).mark_line().transform_fold(
                    ['Roll_Reward_Retail', 'Roll_Reward_Dist'],
                    as_=['Agent', 'Avg Reward']
                ).encode(
                    x='Day:Q',
                    y='Avg Reward:Q',
                    color='Agent:N'
                ).properties(height=300)
                st.altair_chart(chart_rew, use_container_width=True)
                
            st.subheader("Dinamika Bullwhip Effect")
            st.write("Perbandingan variabilitas pesanan dari Konsumen -> Retail -> Dist -> Pabrik.")
            
            # Normalize data untuk perbandingan skala
            df_norm = df_res[['Day', 'Customer Demand', 'Order Retail', 'Order Dist', 'Production']].copy()
            chart_bullwhip = alt.Chart(df_norm.melt('Day')).mark_line(opacity=0.7).encode(
                x='Day',
                y=alt.Y('value', title='Units (Qty)'),
                color='variable'
            ).properties(height=350)
            st.altair_chart(chart_bullwhip, use_container_width=True)

        # --- TAB POLICY ---
        with tab_policy:
            st.header("Apa yang Dipelajari Agen? (Policy Heatmaps)")
            st.write("Sumbu Y: Level Stok Saat Ini | Sumbu X: Tren Permintaan Masuk | Warna: Aksi yang Dipilih")
            
            c1, c2, c3 = st.columns(3)
            
            def plot_q_heatmap(agent, title):
                q_data = []
                for s in range(11): # Stock Levels
                    for d in range(3): # Demand Levels
                        best_action_idx = np.argmax(agent.q_table[s, d])
                        q_data.append({
                            "Stock Level": f"{s*10}%",
                            "Incoming Demand": ["Low", "Normal", "High"][d],
                            "Action": agent.actions[best_action_idx]
                        })
                df_q = pd.DataFrame(q_data)
                hm = alt.Chart(df_q).mark_rect().encode(
                    x='Incoming Demand:O',
                    y=alt.Y('Stock Level:O', sort='descending'),
                    color=alt.Color('Action:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['Stock Level', 'Incoming Demand', 'Action']
                ).properties(title=title, height=300, width=300)
                return hm

            with c1: st.altair_chart(plot_q_heatmap(retailer, "Retailer Policy"), use_container_width=True)
            with c2: st.altair_chart(plot_q_heatmap(distributor, "Distributor Policy"), use_container_width=True)
            with c3: st.altair_chart(plot_q_heatmap(manufacturer, "Manufacturer Policy"), use_container_width=True)

if __name__ == "__main__":
    main()