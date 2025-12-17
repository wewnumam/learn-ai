import streamlit as st
import numpy as np
import pandas as pd
import time
import random
import altair as alt

# Konfigurasi Halaman Streamlit
st.set_page_config(
    page_title="AI Agentic Supply Chain - Q-Learning",
    page_icon="💧",
    layout="wide"
)

# --- CSS Styling untuk Tampilan yang Lebih Baik ---
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #262730;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #4CAF50;
    }
    .stockout-alert {
        color: #FF5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# --- DEFINISI KONSTANTA DAN PARAMETER ---
ACTIONS = [0, 500, 1000, 2000]  # Pilihan jumlah pemesanan (Order Quantities)
MAX_STOCK_CAPACITY = 3000       # Kapasitas Gudang Ritel
SAFETY_STOCK = 500              # Batas aman

# --- KELAS: Q-LEARNING AGENT (RETAILER) ---
class RetailAgent:
    def __init__(self, name, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.name = name
        self.stock = 1500  # Stok awal
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Q-Table: State (Stock Level, Demand Trend) -> Action (Order Qty)
        # Kita diskritisasi stok menjadi 10 level (0-3000 step 300)
        # Demand trend: 0 (Low), 1 (Normal), 2 (High)
        self.q_table = np.zeros((11, 3, len(ACTIONS)))
        self.total_reward = 0
        self.history = []

    def get_state(self, demand_trend):
        """Mengubah stok kontinu menjadi state diskrit."""
        stock_idx = min(int(self.stock / 300), 10)
        return stock_idx, demand_trend

    def choose_action(self, state):
        """Epsilon-Greedy Strategy."""
        stock_idx, trend_idx = state
        
        # Explorasi: Pilih aksi acak
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(range(len(ACTIONS)))
        
        # Eksploitasi: Pilih aksi dengan Q-value tertinggi
        return np.argmax(self.q_table[stock_idx, trend_idx])

    def update_stock(self, demand, delivered_qty):
        """Update fisik stok berdasarkan demand dan penerimaan barang."""
        self.stock += delivered_qty
        actual_sales = min(self.stock, demand)
        lost_sales = demand - actual_sales
        self.stock -= actual_sales
        
        # Cegah stok negatif (walaupun lost sales dihitung)
        self.stock = max(0, self.stock)
        
        return actual_sales, lost_sales

    def calculate_reward(self, demand, lost_sales, holding_cost_per_unit=1, stockout_penalty=50):
        """Fungsi Reward: Agen dihukum jika Stockout atau Overstock."""
        holding_cost = self.stock * holding_cost_per_unit
        penalty = lost_sales * stockout_penalty
        
        # Reward negatif (Cost minimization)
        reward = -(holding_cost + penalty)
        
        # Tambahan penalti jika melebihi kapasitas gudang
        if self.stock > MAX_STOCK_CAPACITY:
            reward -= 1000  # Hukuman berat untuk overcapacity
            
        return reward

    def learn(self, state, action, reward, next_state):
        """Q-Learning Update Rule (Bellman Equation)."""
        s_stock, s_trend = state
        ns_stock, ns_trend = next_state
        
        current_q = self.q_table[s_stock, s_trend, action]
        max_next_q = np.max(self.q_table[ns_stock, ns_trend])
        
        # Rumus Update Q
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[s_stock, s_trend, action] = new_q
        self.total_reward += reward

# --- KELAS: DISTRIBUTOR (RULE BASED) ---
class DistributorAgent:
    def __init__(self, name, initial_stock=50000):
        self.name = name
        self.stock = initial_stock
        self.max_capacity = 100000

    def fulfill_order(self, qty):
        """Memenuhi pesanan ritel jika stok cukup."""
        if self.stock >= qty:
            self.stock -= qty
            return qty
        else:
            # Partial fulfillment atau stockout di level distributor
            delivered = self.stock
            self.stock = 0
            return delivered

    def restock(self, qty=10000):
        """Distributor restock dari Pabrik (Sederhana)."""
        if self.stock < 20000:
            self.stock += qty

# --- FUNGSI UTAMA & UI STREAMLIT ---
def main():
    st.title("🤖 AI Agentic Enterprise: Simulasi Stok Cerdas")
    st.markdown("Simulasi Reinforcement Learning (Q-Learning) untuk pengendalian stok air minum mineral di D.I. Yogyakarta.")

    # --- SIDEBAR KONFIGURASI ---
    st.sidebar.header("⚙️ Konfigurasi Simulasi")
    
    simulation_speed = st.sidebar.slider("Kecepatan Simulasi (detik/hari)", 0.01, 1.0, 0.1)
    n_days = st.sidebar.slider("Durasi Simulasi (Hari)", 30, 365, 100)
    
    st.sidebar.subheader("Parameter AI Agent")
    learning_rate = st.sidebar.number_input("Learning Rate (Alpha)", 0.01, 1.0, 0.1)
    epsilon = st.sidebar.slider("Exploration Rate (Epsilon)", 0.0, 1.0, 0.2)
    
    st.sidebar.subheader("Skenario Permintaan")
    scenario_mode = st.sidebar.selectbox(
        "Pilih Pola Permintaan", 
        ["Normal", "Seasonal Rush (Liburan)", "Random Chaos"]
    )

    if st.sidebar.button("🚀 Mulai Simulasi", type="primary"):
        run_simulation(n_days, simulation_speed, learning_rate, epsilon, scenario_mode)

def get_demand(day, scenario):
    """Menghasilkan permintaan buatan berdasarkan skenario."""
    base_demand = 100
    noise = random.randint(-20, 50)
    
    if scenario == "Normal":
        # Ada rush kecil di akhir pekan (setiap kelipatan 7)
        if day % 7 == 0: return base_demand * 3 + noise
        return base_demand + noise
        
    elif scenario == "Seasonal Rush (Liburan)":
        # Rush besar di hari 30-40
        if 30 <= day <= 40: return base_demand * 5 + noise
        return base_demand + noise

    elif scenario == "Random Chaos":
        return random.randint(50, 600)
    
    return base_demand

def get_trend_index(current_demand):
    """Mengklasifikasikan trend permintaan untuk input state agent."""
    if current_demand < 100: return 0 # Low
    elif current_demand < 300: return 1 # Normal
    else: return 2 # High/Rush

def run_simulation(n_days, speed, alpha, eps, scenario):
    # Inisialisasi Agen
    retailer = RetailAgent("Agen Ritel (Supermarket)", learning_rate=alpha, epsilon=eps)
    distributor = DistributorAgent("Agen Distributor DIY")
    
    # Placeholder UI untuk update realtime
    col1, col2, col3 = st.columns(3)
    with col1:
        metric_stock = st.empty()
    with col2:
        metric_reward = st.empty()
    with col3:
        metric_action = st.empty()

    chart_placeholder = st.empty()
    log_placeholder = st.empty()
    
    history_data = []

    progress_bar = st.progress(0)
    
    # --- LOOP SIMULASI ---
    for day in range(1, n_days + 1):
        # 1. Tentukan Permintaan Hari Ini
        demand = get_demand(day, scenario)
        trend_idx = get_trend_index(demand)
        
        # 2. Agen Mengamati State Saat Ini
        state = retailer.get_state(trend_idx)
        
        # 3. Agen Memilih Aksi (Berapa banyak harus pesan?)
        action_idx = retailer.choose_action(state)
        order_qty = ACTIONS[action_idx]
        
        # 4. Eksekusi Logistik
        # Distributor memproses pesanan
        delivered_qty = distributor.fulfill_order(order_qty)
        distributor.restock() # Distributor restock sendiri jika perlu
        
        # Update stok ritel (Permintaan konsumen terjadi di sini)
        actual_sales, lost_sales = retailer.update_stock(demand, delivered_qty)
        
        # 5. Hitung Reward
        reward = retailer.calculate_reward(demand, lost_sales)
        
        # 6. Observasi State Baru & Belajar (Q-Learning Update)
        # Prediksi trend besok (sederhana: asumsikan mirip hari ini untuk state)
        next_trend = trend_idx 
        next_state = retailer.get_state(next_trend)
        
        retailer.learn(state, action_idx, reward, next_state)
        
        # --- VISUALISASI DATA ---
        history_data.append({
            "Day": day,
            "Stock Level": retailer.stock,
            "Demand": demand,
            "Order Qty": order_qty,
            "Lost Sales": lost_sales,
            "Reward": retailer.total_reward
        })
        
        df_history = pd.DataFrame(history_data)

        # Update Metrics
        with col1:
            metric_stock.metric("📦 Stok Ritel", f"{retailer.stock} Unit", f"{order_qty} Pesan")
        with col2:
            metric_reward.metric("🎯 Total Reward", f"{int(retailer.total_reward)}", f"{int(reward)} Harian")
        with col3:
            status_text = "NORMAL"
            status_color = "off"
            if lost_sales > 0: 
                status_text = "⚠️ STOCKOUT!"
                status_color = "inverse"
            elif retailer.stock > 2500:
                status_text = "⚠️ OVERSTOCK"
            metric_action.metric("Status", status_text, f"Demand: {demand}")

        # Update Charts (Menggunakan Altair untuk performa)
        if day % 2 == 0: # Update chart setiap 2 hari agar tidak terlalu berat
            base = alt.Chart(df_history.tail(50)).encode(x='Day')

            line_stock = base.mark_line(color='#4CAF50').encode(
                y=alt.Y('Stock Level', scale=alt.Scale(domain=[0, MAX_STOCK_CAPACITY+500])),
                tooltip=['Day', 'Stock Level', 'Demand']
            )
            
            line_demand = base.mark_line(color='#FF5252', strokeDash=[5,5]).encode(
                y='Demand'
            )
            
            bar_order = base.mark_bar(color='#2196F3', opacity=0.3).encode(
                y='Order Qty'
            )

            chart_layer = alt.layer(bar_order, line_stock, line_demand).properties(
                title="Dinamika Stok (Hijau), Permintaan (Merah Putus), Order (Biru)",
                height=350
            )
            
            chart_placeholder.altair_chart(chart_layer, use_container_width=True)

        # Update Log
        if lost_sales > 0:
            log_placeholder.warning(f"Day {day}: KEHABISAN STOK! Permintaan {demand}, Stok hanya {retailer.stock + actual_sales}. Lost Sales: {lost_sales}")
        elif order_qty > 0:
            log_placeholder.info(f"Day {day}: Memesan {order_qty} unit. Stok aman.")
        else:
            log_placeholder.markdown(f"*Day {day}: Tidak memesan. Menunggu penjualan.*")

        progress_bar.progress(day / n_days)
        time.sleep(speed)

    st.success("Simulasi Selesai! Agen telah mempelajari pola permintaan.")
    
    # Tampilkan Analisis Akhir
    st.subheader("📊 Analisis Pembelajaran Agen")
    st.write("Heatmap Q-Table di bawah menunjukkan keputusan yang dipelajari agen. Sumbu Y adalah level stok, Sumbu X adalah Aksi.")
    
    # Visualisasi Q-Table sederhana
    q_data = []
    for s in range(11):
        for t in range(3): # Kita ambil rata-rata trend atau salah satu trend
            best_action_idx = np.argmax(retailer.q_table[s, t])
            q_data.append({
                "Stock Level Bucket": s * 300,
                "Demand Trend": ["Low", "Normal", "High"][t],
                "Best Action": ACTIONS[best_action_idx]
            })
    
    df_q = pd.DataFrame(q_data)
    heatmap = alt.Chart(df_q).mark_rect().encode(
        x='Demand Trend:O',
        y=alt.Y('Stock Level Bucket:O', sort='descending'),
        color='Best Action:Q',
        tooltip=['Stock Level Bucket', 'Demand Trend', 'Best Action']
    ).properties(title="Kebijakan Optimal (Policy) yang Dipelajari")
    
    st.altair_chart(heatmap, use_container_width=True)

if __name__ == "__main__":
    main()