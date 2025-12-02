import sqlite3

def init_db():
    conn = sqlite3.connect('ban_system.db')
    c = conn.cursor()
    
    # Tabel Stok (Milik Service Ritel)
    c.execute('''CREATE TABLE IF NOT EXISTS stok 
                 (ritel_id TEXT, ritel_nama TEXT, tipe_ban TEXT, jumlah INTEGER)''')
    
    # Tabel Order (Milik Service Order)
    c.execute('''CREATE TABLE IF NOT EXISTS pesanan 
                 (order_id TEXT, pembeli TEXT, ritel_id TEXT, tipe_ban TEXT, status TEXT)''')
    
    # Masukkan Data Dummy Awal
    # Cek apakah kosong
    c.execute('SELECT count(*) FROM stok')
    if c.fetchone()[0] == 0:
        data_awal = [
            ('R1', 'Ritel Maju', 'Ban_Sedan', 50),
            ('R1', 'Ritel Maju', 'Ban_SUV', 20),
            ('R2', 'Ritel Abadi', 'Ban_Sedan', 10),
            ('R2', 'Ritel Abadi', 'Ban_SUV', 100),
        ]
        c.executemany('INSERT INTO stok VALUES (?,?,?,?)', data_awal)
        print("Database berhasil diinisialisasi dengan data dummy.")
    else:
        print("Database sudah ada, melewati inisialisasi.")
        
    conn.commit()
    conn.close()

if __name__ == "__main__":
    init_db()
