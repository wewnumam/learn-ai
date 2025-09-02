import random
import time

# ===================================================================
# --- KONFIGURASI DAN PARAMETER SIMULASI ---
# Anda bisa mengubah semua nilai di bawah ini untuk eksperimen
# ===================================================================

# --- Parameter Dasar Pasar ---
NAMA_ITEM = ["Ikan", "Ayam", "Telur", "Beras", "Tempe"]
HARGA_GROSIR = {
    "Ikan": 25000,
    "Ayam": 30000,
    "Telur": 20000,
    "Beras": 50000,
    "Tempe": 5000,
}
JUMLAH_PEMBELI = 5
JUMLAH_PENJUAL = 5
JEDA_LOG = 0.05 # Kecepatan log simulasi (detik), set ke 0 untuk instan

# --- Parameter Perilaku Pembeli ---
# 'harga_terbaik' (agresif)
MAKS_TAWARAN_AGRESIF = 7
# 'dapat_item' (aman)
MAKS_TAWARAN_AMAN = 5

# --- Parameter Perilaku Penjual ---
# Harga eceran awal = HARGA_GROSIR * FAKTOR_ECERAN_AWAL * (variansi)
FAKTOR_ECERAN_AWAL = 2.0
VARIANSI_HARGA_MIN = 0.95 # Variasi acak harga eceran (95%)
VARIANSI_HARGA_MAKS = 1.05 # Variasi acak harga eceran (105%)

# Strategi 'laku' (ingin cepat jual)
KESEMPATAN_MENAWAR_LAKU = 5 # Kesabaran penjual 'laku'
FAKTOR_PENURUNAN_LAKU = 0.8 # Seberapa besar penjual 'laku' menurunkan harga (80% dari selisih)

# Strategi 'untung' (ingin profit besar)
KESEMPATAN_MENAWAR_UNTUNG = 8 # Kesabaran penjual 'untung'
FAKTOR_PENURUNAN_UNTUNG = 0.5 # Seberapa besar penjual 'untung' menurunkan harga (50% dari selisih)

# --- Parameter Mekanisme Negosiasi ---
# Tawaran awal pembeli = Harga Eceran * FAKTOR_TAWARAN_AWAL_PEMBELI
FAKTOR_TAWARAN_AWAL_PEMBELI = 0.5 # (0.5 berarti setengah harga)

# Seberapa agresif pembeli menaikkan tawaran
FAKTOR_AGRESIVITAS_PEMBELI = 0.7 # (0.7 berarti menaikkan 70% dari selisih harga)

# Batas profit minimum penjual (harga jual tidak akan lebih rendah dari ini)
FAKTOR_PROFIT_MINIMUM = 1.1 # (1.1 berarti minimal untung 10%)

# ===================================================================
# --- KODE SIMULASI (Tidak perlu diubah) ---
# ===================================================================

class Pembeli:
    """Mewakili agen pembeli dengan tujuannya sendiri."""
    def __init__(self, nama, item_dicari):
        self.nama = nama
        self.item_dicari = item_dicari
        self.strategi = random.choice(['harga_terbaik', 'dapat_item'])
        self.maks_tawaran = MAKS_TAWARAN_AGRESIF if self.strategi == 'harga_terbaik' else MAKS_TAWARAN_AMAN
        self.item_didapat, self.harga_final, self.total_tawaran_final = None, 0, 0

    def __str__(self):
        return f"{self.nama} (Mencari: {self.item_dicari}, Strategi: {self.strategi})"

class Penjual:
    """Mewakili agen penjual dengan inventaris dan strateginya."""
    def __init__(self, nama):
        self.nama = nama
        self.strategi = random.choice(['laku', 'untung'])
        self.inventaris = {}
        for item, grosir in HARGA_GROSIR.items():
            faktor_harga = FAKTOR_ECERAN_AWAL * random.uniform(VARIANSI_HARGA_MIN, VARIANSI_HARGA_MAKS)
            self.inventaris[item] = {
                "harga_grosir": grosir, "harga_eceran": grosir * faktor_harga, "terjual": False
            }
        
        if self.strategi == 'laku':
            self.kesempatan_menawar = KESEMPATAN_MENAWAR_LAKU
            self.faktor_penurunan = FAKTOR_PENURUNAN_LAKU
        else: # strategi 'untung'
            self.kesempatan_menawar = KESEMPATAN_MENAWAR_UNTUNG
            self.faktor_penurunan = FAKTOR_PENURUNAN_UNTUNG
            
        self.total_keuntungan, self.transaksi = 0, []

    def __str__(self):
        return f"{self.nama} (Strategi: {self.strategi})"

class SimulasiPasar:
    def __init__(self, daftar_pembeli, daftar_penjual):
        self.pembeli, self.penjual, self.log = daftar_pembeli, daftar_penjual, []

    def _tulis_log(self, pesan, jeda=JEDA_LOG):
        print(pesan)
        self.log.append(pesan)
        if jeda > 0: time.sleep(jeda)

    def _negosiasi(self, pembeli, penjual, item):
        self._tulis_log(f"\n--- {pembeli.nama} negosiasi {item} dengan {penjual.nama} ---", jeda=0)
        info = penjual.inventaris[item]
        harga_penjual = info['harga_eceran']
        harga_terendah = info['harga_grosir'] * FAKTOR_PROFIT_MINIMUM
        tawaran_pembeli = harga_penjual * FAKTOR_TAWARAN_AWAL_PEMBELI
        
        self._tulis_log(f"Harga awal {penjual.nama} ({penjual.strategi}): Rp {harga_penjual:,.0f}")

        for i in range(1, penjual.kesempatan_menawar + 1):
            self._tulis_log(f"   [Tawaran {i}] {pembeli.nama} menawar: Rp {tawaran_pembeli:,.0f}")
            
            if tawaran_pembeli >= harga_penjual:
                self._tulis_log(f"   [DEAL!] {penjual.nama} menerima Rp {harga_penjual:,.0f}")
                return {'status': 'sukses', 'harga_akhir': harga_penjual, 'tawaran': i}

            penurunan = (harga_penjual - tawaran_pembeli) * penjual.faktor_penurunan
            harga_penjual = max(harga_penjual - penurunan, harga_terendah)
            self._tulis_log(f"   -> {penjual.nama} balas: Rp {harga_penjual:,.0f}")

            if tawaran_pembeli >= harga_penjual:
                self._tulis_log(f"   [DEAL!] {penjual.nama} menyetujui tawaran di akhir giliran.")
                return {'status': 'sukses', 'harga_akhir': harga_penjual, 'tawaran': i}

            if i >= pembeli.maks_tawaran:
                self._tulis_log(f"   [STOP] {pembeli.nama} capai batas tawaran.")
                # Tetap kembalikan harga akhir penjual meskipun gagal
                return {'status': 'gagal', 'harga_akhir': harga_penjual, 'tawaran': i}
            
            tawaran_pembeli += (harga_penjual - tawaran_pembeli) * FAKTOR_AGRESIVITAS_PEMBELI
        
        self._tulis_log(f"   [GAGAL] {penjual.nama} mundur (batas kesempatan).")
        return {'status': 'gagal', 'harga_akhir': harga_penjual, 'tawaran': penjual.kesempatan_menawar}

    def jalankan_simulasi(self):
        self._tulis_log("="*40 + "\n       MEMULAI SIMULASI PASAR CERDAS\n" + "="*40 + "\n", 0)
        for pembeli in self.pembeli:
            self._tulis_log(f"\n>>> {pembeli.nama} (Strategi: {pembeli.strategi}) mencari {pembeli.item_dicari}...", jeda=0.2)
            penjual_tersedia = [p for p in self.penjual if not p.inventaris[pembeli.item_dicari]['terjual']]
            random.shuffle(penjual_tersedia)
            
            if not penjual_tersedia:
                self._tulis_log(f"   Item {pembeli.item_dicari} sudah habis.")
                continue

            self._tulis_log(f"   Urutan penjual ditemui: {[p.nama for p in penjual_tersedia]}")
            penawaran_final = None

            if pembeli.strategi == 'harga_terbaik':
                terbaik = {'penjual': None, 'hasil': {'status': 'gagal', 'harga_akhir': float('inf'), 'tawaran': 0}}
                # Simpan siapa penjual terakhir dalam urutan
                penjual_terakhir = penjual_tersedia[-1]

                for penjual in penjual_tersedia:
                    hasil_nego = self._negosiasi(pembeli, penjual, pembeli.item_dicari)
                    if hasil_nego['harga_akhir'] < terbaik['hasil']['harga_akhir']:
                        terbaik = {'penjual': penjual, 'hasil': hasil_nego}
                        self._tulis_log(f"   [BENCHMARK BARU] Harga terbaik sementara dari {penjual.nama}: Rp {hasil_nego['harga_akhir']:,.0f}")
                
                # ==========================================================
                # LOGIKA KEPUTUSAN BARU SESUAI ATURAN ANDA
                # ==========================================================
                if terbaik['penjual'] is not None:
                    # Cek apakah penjual terbaik adalah penjual terakhir yang ditemui
                    if terbaik['penjual'] == penjual_terakhir:
                        self._tulis_log(f"   [KEPUTUSAN STRATEGIS] Penjual terakhir ({penjual_terakhir.nama}) memberikan harga terbaik. DEAL!")
                        penawaran_final = {'penjual': terbaik['penjual'], 'harga': terbaik['hasil']['harga_akhir'], 'jumlah_tawaran': terbaik['hasil']['tawaran']}
                    else:
                        self._tulis_log(f"   [KEPUTUSAN STRATEGIS] Harga terbaik datang dari {terbaik['penjual'].nama}, bukan penjual terakhir. Pembeli mundur.")
                # Jika tidak ada penawaran sama sekali (tidak mungkin terjadi jika ada penjual), maka tidak beli.

            elif pembeli.strategi == 'dapat_item':
                benchmark = None
                for penjual in penjual_tersedia:
                    hasil_nego = self._negosiasi(pembeli, penjual, pembeli.item_dicari)
                    
                    if benchmark is None:
                        benchmark = {'penjual': penjual, 'hasil': hasil_nego}
                        status_benchmark = "sukses" if hasil_nego['status'] == 'sukses' else "gagal"
                        self._tulis_log(f"   [BENCHMARK DITETAPKAN] Harga dari {penjual.nama}: Rp {hasil_nego['harga_akhir']:,.0f} (Status: {status_benchmark})")
                        continue
                    
                    if hasil_nego['harga_akhir'] < benchmark['hasil']['harga_akhir']:
                        self._tulis_log(f"   [DEAL CEPAT!] {penjual.nama} lebih baik (Rp {hasil_nego['harga_akhir']:,.0f}) dari benchmark (Rp {benchmark['hasil']['harga_akhir']:,.0f}). Langsung beli!")
                        penawaran_final = {'penjual': penjual, 'harga': hasil_nego['harga_akhir'], 'jumlah_tawaran': hasil_nego['tawaran']}
                        break

                if penawaran_final is None and benchmark is not None:
                    if benchmark['hasil']['status'] == 'sukses':
                        self._tulis_log(f"   Tidak ada yang lebih baik, {pembeli.nama} memilih penawaran benchmark awal.")
                        penawaran_final = {'penjual': benchmark['penjual'], 'harga': benchmark['hasil']['harga_akhir'], 'jumlah_tawaran': benchmark['hasil']['tawaran']}
            
            if penawaran_final:
                p_final, h_final, t_final = penawaran_final['penjual'], penawaran_final['harga'], penawaran_final['jumlah_tawaran']
                self._tulis_log(f"\n+++ KEPUTUSAN: {pembeli.nama} membeli {pembeli.item_dicari} dari {p_final.nama} seharga Rp {h_final:,.0f} +++\n", 0.2)
                pembeli.item_didapat, pembeli.harga_final, pembeli.total_tawaran_final = pembeli.item_dicari, h_final, t_final
                item_info = p_final.inventaris[pembeli.item_dicari]
                item_info['terjual'] = True
                keuntungan = h_final - item_info['harga_grosir']
                p_final.total_keuntungan += keuntungan
                p_final.transaksi.append({'item': pembeli.item_dicari, 'keuntungan': keuntungan, 'jumlah_tawaran': t_final})
            else:
                self._tulis_log(f"\n--- KEPUTUSAN: {pembeli.nama} tidak berhasil mendapatkan {pembeli.item_dicari} ---\n", 0.2)

    def tampilkan_hasil(self):
        print("\n\n" + "="*40 + "\n         HASIL AKHIR SIMULASI\n" + "="*40)
        print("\n--- LAPORAN PEMBELI ---")
        print(f"{'Nama Pembeli':<15} | {'Strategi':<15} | {'Item Dibeli':<10} | {'Harga Final':>15} | {'Jumlah Tawar':>15}")
        print("-" * 85)
        for p in self.pembeli:
            item = p.item_didapat if p.item_didapat else "Gagal"
            harga_str = f"Rp {p.harga_final:,.0f}" if p.harga_final > 0 else "-"
            print(f"{p.nama:<15} | {p.strategi:<15} | {item:<10} | {harga_str:>15} | {p.total_tawaran_final:>15}")

        print("\n--- LAPORAN PENJUAL ---")
        print(f"{'Nama Penjual':<15} | {'Strategi':<10} | {'Total Keuntungan':>18} | {'Item Terjual (Jml Tawar)'}")
        print("-" * 90)
        for s in self.penjual:
            keuntungan_str = f"Rp {s.total_keuntungan:,.0f}"
            transaksi_str = ", ".join([f"{t['item']} ({t['jumlah_tawaran']})" for t in s.transaksi]) or "Tidak ada"
            print(f"{s.nama:<15} | {s.strategi:<10} | {keuntungan_str:>18} | {transaksi_str}")


if __name__ == "__main__":
    if JUMLAH_PEMBELI > len(NAMA_ITEM):
        print(f"Error: JUMLAH_PEMBELI ({JUMLAH_PEMBELI}) tidak boleh melebihi jumlah item unik ({len(NAMA_ITEM)}).")
    else:
        para_pembeli = [Pembeli(f"Pembeli {i+1}", NAMA_ITEM[i]) for i in range(JUMLAH_PEMBELI)]
        para_penjual = [Penjual(f"Penjual {chr(65+i)}") for i in range(JUMLAH_PENJUAL)]
        print("--- DAFTAR AGEN ---")
        for p in para_pembeli: print(p)
        for p in para_penjual: print(p)
        print("-------------------\n")
        simulasi = SimulasiPasar(para_pembeli, para_penjual)
        simulasi.jalankan_simulasi()
        simulasi.tampilkan_hasil()