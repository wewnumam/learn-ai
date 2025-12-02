import streamlit as st
import sqlite3
import pandas as pd
import time
from datetime import datetime


# ==========================================
# 1. KONFIGURASI & DATABASE
# ==========================================
st.set_page_config(page_title="Sistem Agen Cerdas", layout="wide", page_icon="🤖")


DB_FILE = 'agent_data.db'


def init_db():
    """Inisialisasi Database SQLite jika belum ada"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
   
    # Tabel Stok (Inventory)
    c.execute('''CREATE TABLE IF NOT EXISTS inventory
                 (id TEXT PRIMARY KEY, nama_barang TEXT, jumlah INTEGER, harga INTEGER)''')
   
    # Tabel Log Transaksi (Activity Log)
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (waktu TEXT, aktor TEXT, aksi TEXT, detail TEXT)''')
   
    # Seed Data (Data Awal) jika kosong
    c.execute("SELECT count(*) FROM inventory")
    if c.fetchone()[0] == 0:
        initial_data = [
            ('BAN-001', 'Ban Sedan Premium', 10, 800000),
            ('BAN-002', 'Ban SUV Offroad', 5, 1500000),
            ('BAN-003', 'Ban City Car Eco', 20, 600000)
        ]
        c.executemany("INSERT INTO inventory VALUES (?,?,?,?)", initial_data)
       
    conn.commit()
    conn.close()


def log_activity(aktor, aksi, detail):
    """Mencatat aktivitas agen ke database"""
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    waktu = datetime.now().strftime("%H:%M:%S")
    c.execute("INSERT INTO logs VALUES (?, ?, ?, ?)", (waktu, aktor, aksi, detail))
    conn.commit()
    conn.close()


# Inisialisasi DB saat aplikasi mulai
init_db()


# ==========================================
# 2. LOGIC AGENT RITEL
# ==========================================
def run_retail_agent():
    st.header("🏭 Dashboard Agent Ritel")
    st.markdown("---")
   
    # 2A. Monitoring Stok (Real-time)
    conn = sqlite3.connect(DB_FILE)
    df_stok = pd.read_sql("SELECT * FROM inventory", conn)
    conn.close()
   
    # Tampilkan Metrics
    col1, col2, col3 = st.columns(3)
    total_items = df_stok['jumlah'].sum()
    low_stock = df_stok[df_stok['jumlah'] < 5].shape[0]
   
    col1.metric("Total Stok Gudang", f"{total_items} Unit")
    col2.metric("Item Kritis (<5)", f"{low_stock} Item", delta_color="inverse")
   
    # Tampilkan Tabel
    st.dataframe(df_stok, use_container_width=True, hide_index=True)
   
    st.divider()
   
    # 2B. Aksi: Restock (Menambah Barang)
    st.subheader("📦 Restock Barang")
    with st.form("form_restock"):
        c1, c2 = st.columns([3, 1])
        item_pilih = c1.selectbox("Pilih Barang", df_stok['nama_barang'])
        jumlah_tambah = c2.number_input("Jumlah Tambahan", min_value=1, value=10)
       
        if st.form_submit_button("Tambah Stok"):
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
           
            # Update DB
            c.execute("UPDATE inventory SET jumlah = jumlah + ? WHERE nama_barang = ?",
                      (jumlah_tambah, item_pilih))
            conn.commit()
            conn.close()
           
            log_activity("Agent Ritel", "RESTOCK", f"Menambah {jumlah_tambah} unit {item_pilih}")
            st.success(f"Berhasil menambah stok {item_pilih}!")
            time.sleep(1)
            st.rerun() # Force Refresh segera


# ==========================================
# 3. LOGIC AGENT PEMBELI
# ==========================================
def run_buyer_agent():
    st.header("🛒 Dashboard Agent Pembeli")
    st.markdown("---")
   
    # 3A. Persepsi Lingkungan (Melihat Stok)
    conn = sqlite3.connect(DB_FILE)
    df_market = pd.read_sql("SELECT * FROM inventory", conn)
    conn.close()
   
    # Visualisasi Produk dalam bentuk Cards
    st.write("### Katalog Produk Tersedia")
   
    if df_market.empty:
        st.error("Tidak ada barang di pasar.")
   
    for index, row in df_market.iterrows():
        # Logic UI: Jika stok habis, disable tombol
        stok_ada = row['jumlah'] > 0
       
        with st.container(border=True):
            cols = st.columns([4, 2, 2])
           
            with cols[0]:
                st.subheader(row['nama_barang'])
                st.caption(f"ID: {row['id']}")
           
            with cols[1]:
                if stok_ada:
                    st.metric("Stok Tersedia", f"{row['jumlah']}", delta="Ready")
                else:
                    st.metric("Stok", "HABIS", delta_color="inverse")
           
            with cols[2]:
                st.write("") # Spacer
                if stok_ada:
                    # 3B. Aksi: Membeli (Mengurangi Stok)
                    if st.button(f"Beli 1", key=f"btn_{row['id']}"):
                        conn = sqlite3.connect(DB_FILE)
                        c = conn.cursor()
                       
                        # Cek ulang stok sebelum update (Concurrecy Check)
                        c.execute("SELECT jumlah FROM inventory WHERE id=?", (row['id'],))
                        curr = c.fetchone()[0]
                       
                        if curr > 0:
                            c.execute("UPDATE inventory SET jumlah = jumlah - 1 WHERE id=?", (row['id'],))
                            conn.commit()
                            log_activity("Agent Pembeli", "PEMBELIAN", f"Membeli 1 unit {row['nama_barang']}")
                            st.toast(f"✅ Berhasil membeli {row['nama_barang']}!")
                        else:
                            st.error("Gagal! Stok baru saja habis.")
                       
                        conn.close()
                        time.sleep(0.5)
                        st.rerun()
                else:
                    st.button("Stok Habis", disabled=True, key=f"d_{row['id']}")


# ==========================================
# 4. MONITORING LOGS (SHARED VIEW)
# ==========================================
def show_logs():
    st.sidebar.markdown("---")
    st.sidebar.subheader("📡 Live Activity Log")
    conn = sqlite3.connect(DB_FILE)
    # Ambil 10 log terakhir
    logs = pd.read_sql("SELECT * FROM logs ORDER BY rowid DESC LIMIT 10", conn)
    conn.close()
   
    if not logs.empty:
        for i, row in logs.iterrows():
            icon = "🏭" if row['aktor'] == "Agent Ritel" else "🛒"
            st.sidebar.caption(f"{row['waktu']} {icon} **{row['aktor']}**: {row['detail']}")
    else:
        st.sidebar.caption("Belum ada aktivitas.")


# ==========================================
# 5. MAIN CONTROLLER
# ==========================================
def main():
    # Selector Peran
    peran = st.sidebar.radio("Pilih Peran Agent:", ["Agent Ritel (Supplier)", "Agent Pembeli (Consumer)"])
   
    # Fitur Auto-Refresh (Jantung dari Real-time Agent)
    st.sidebar.markdown("---")
    auto_refresh = st.sidebar.checkbox("🔴 Live Mode (Auto-Refresh)", value=True)
   
    # Routing Halaman
    if peran == "Agent Ritel (Supplier)":
        run_retail_agent()
    else:
        run_buyer_agent()
   
    # Tampilkan Log di kedua sisi
    show_logs()
   
    # Logic Auto-Refresh
    # Ini yang membuat aplikasi terasa 'hidup'.
    # Jika dicentang, script akan run ulang setiap 2 detik untuk mengambil data DB terbaru.
    if auto_refresh:
        time.sleep(2)
        st.rerun()


if __name__ == "__main__":
    main()
