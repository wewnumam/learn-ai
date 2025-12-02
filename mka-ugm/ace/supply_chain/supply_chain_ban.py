import streamlit as st
import random
import pandas as pd
import time
from datetime import datetime, timedelta

# --- 1. Definisi Kelas Agen (Agents) ---

class RetailerAgent:
    def __init__(self, name, initial_stock):
        self.name = name
        self.inventory = initial_stock 

    def get_stock(self):
        # PENTING: Mengembalikan copy agar tidak error TypeError
        return self.inventory.copy()

    def process_order(self, tire_type, quantity):
        if tire_type not in self.inventory:
            return False, f"Tipe '{tire_type}' tidak ada."
        
        if self.inventory[tire_type] >= quantity:
            self.inventory[tire_type] -= quantity
            return True, f"Pesanan {quantity} unit '{tire_type}' BERHASIL."
        else:
            return False, f"Stok '{tire_type}' KURANG (sisa: {self.inventory[tire_type]})."

    def restock(self):
        """Menambah stok secara acak antara 20-60 unit."""
        restocked_items = []
        for tire_type in self.inventory:
            amount = random.randint(20, 60)
            self.inventory[tire_type] += amount
            restocked_items.append(f"{tire_type}: +{amount}")
        return f"{self.name} Restock ({', '.join(restocked_items)})"

class BuyerAgent:
    def __init__(self, name):
        self.name = name

    def place_random_order(self, retailers, tire_types):
        """Fungsi otomatis: Pembeli memilih retailer dan barang secara acak"""
        
        # Pilih target retailer dan ban secara acak
        target_retailer = random.choice(retailers)
        target_tire = random.choice(tire_types)
        qty = random.randint(1, 15) # Pesan antara 1 sampai 15 ban
        
        # Lakukan pemesanan
        success, msg = target_retailer.process_order(target_tire, qty)
        
        log_msg = f"{self.name} -> {target_retailer.name}: {msg}"
        return log_msg, success

class DashboardAgent:
    def __init__(self, retailer_agents):
        self.retailer_agents = retailer_agents

    def get_aggregated_stock(self):
        stock_data = []
        for retailer in self.retailer_agents:
            stock = retailer.get_stock()
            stock['Retailer'] = retailer.name
            stock_data.append(stock)
        
        if not stock_data: return pd.DataFrame()
        df = pd.DataFrame(stock_data)
        return df.set_index('Retailer').reset_index()

# --- 2. Helper Functions ---

def log_current_stock(virtual_time=None):
    """Mencatat snapshot stok ke history untuk plotting."""
    # Jika virtual_time tidak ada, gunakan waktu sekarang
    ts = virtual_time if virtual_time else datetime.now()
    
    for retailer in st.session_state.retailers:
        stock = retailer.get_stock()
        for tire_type, qty in stock.items():
            st.session_state.stock_history.append({
                'time': ts,
                'retailer': retailer.name,
                'tire': tire_type,
                'stock': qty
            })

def run_simulation_step(step_count=1):
    """
    Menjalankan satu atau beberapa langkah simulasi otomatis.
    """
    for _ in range(step_count):
        # A. AKSI PEMBELI (Selalu terjadi tiap langkah)
        # Pilih satu pembeli acak untuk beraksi
        active_buyer = random.choice(st.session_state.buyers)
        log_msg, success = active_buyer.place_random_order(
            st.session_state.retailers, 
            st.session_state.tire_types
        )
        st.session_state.logs.insert(0, f"[ORDER] {log_msg}")

        # B. AKSI RETAILER (Peluang terjadi 30% tiap langkah)
        # Cek setiap retailer, apakah mereka mau restock?
        for r in st.session_state.retailers:
            if random.random() < 0.3: # 30% kemungkinan restock
                restock_msg = r.restock()
                st.session_state.logs.insert(0, f"[RESTOCK] 🚚 {restock_msg}")

        # C. Catat Data untuk Grafik
        # Kita majukan 'waktu virtual' sedikit agar grafik terlihat bergerak maju
        last_time = st.session_state.current_virtual_time
        new_time = last_time + timedelta(minutes=30) # Maju 30 menit per langkah
        st.session_state.current_virtual_time = new_time
        
        log_current_stock(new_time)

# --- 3. Inisialisasi ---

def initialize_simulation():
    if 'initialized' not in st.session_state:
        # Setup Retailer
        r1 = RetailerAgent("R1", {'Ban_Truk': 100, 'Ban_Mobil': 200})
        r2 = RetailerAgent("R2", {'Ban_Truk': 80, 'Ban_Mobil': 250})
        st.session_state.retailers = [r1, r2]
        
        # Setup Pembeli
        st.session_state.buyers = [BuyerAgent("Andi"), BuyerAgent("Budi"), BuyerAgent("Citra"), BuyerAgent("Dewi")]
        
        # Setup Dashboard & Variabel Lain
        st.session_state.dashboard = DashboardAgent(st.session_state.retailers)
        st.session_state.tire_types = ['Ban_Truk', 'Ban_Mobil']
        st.session_state.logs = []
        st.session_state.stock_history = []
        st.session_state.current_virtual_time = datetime.now()
        
        # Catat kondisi awal
        log_current_stock(st.session_state.current_virtual_time)
        st.session_state.initialized = True

# --- 4. UI Streamlit ---

st.set_page_config(page_title="Simulasi Supply Chain Ban", layout="wide")
st.title("🤖 Simulasi Supply Chain Otomatis (Multi-Agent)")

initialize_simulation()

# Layout Kolom
col_ctrl, col_dash = st.columns([1, 2])

with col_ctrl:
    st.header("⚙️ Kontrol Simulasi")
    st.info("Klik tombol di bawah untuk membiarkan agen bekerja secara acak.")
    
    # Tombol 1 Langkah
    if st.button("▶️ Jalankan 1 Langkah Acak", use_container_width=True):
        run_simulation_step(1)
        st.success("1 Langkah selesai.")
        
    # Tombol 10 Langkah (Batch)
    if st.button("⏩ Jalankan 20 Langkah (Cepat)", use_container_width=True):
        with st.spinner("Agen sedang sibuk bertransaksi..."):
            run_simulation_step(20)
        st.success("20 Langkah simulasi selesai!")
        st.rerun()

    st.divider()
    
    st.subheader("📝 Log Aktivitas Terbaru")
    log_text = "\n".join(st.session_state.logs[:50]) # Tampilkan 50 log terakhir
    st.text_area("Log", log_text, height=400, disabled=True)

with col_dash:
    st.header("📊 Real-time Dashboard")
    
    # 1. Tabel Stok Saat Ini
    st.subheader("Stok Saat Ini")
    df_now = st.session_state.dashboard.get_aggregated_stock()
    st.dataframe(df_now, use_container_width=True)
    
    # 2. Grafik Time Series
    st.subheader("Pergerakan Stok (Time Series)")
    
    if len(st.session_state.stock_history) > 0:
        hist_df = pd.DataFrame(st.session_state.stock_history)
        
        # Buat kolom kombinasi agar legenda jelas
        hist_df['Legend'] = hist_df['retailer'] + " - " + hist_df['tire']
        
        # Plot Line Chart
        st.line_chart(
            hist_df, 
            x='time', 
            y='stock', 
            color='Legend',
            height=400
        )
    else:
        st.warning("Belum ada data. Silakan jalankan simulasi.")