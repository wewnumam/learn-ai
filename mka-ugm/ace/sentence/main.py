import spacy
from googletrans import Translator
import asyncio
from pathlib import Path # -> Impor library Path untuk mengelola file

# ======================================================================
# Bagian kode di bawah ini tidak berubah (tetap sama seperti sebelumnya)
# ======================================================================
try:
    nlp = spacy.load("en_core_web_sm")
except IOError:
    print("Model 'en_core_web_sm' tidak ditemukan.")
    print("Silakan jalankan: python -m spacy download en_core_web_sm")
    exit()

translator = Translator()

def get_full_phrase(token):
    subtree = sorted([t.i for t in token.subtree])
    start = subtree[0]
    end = subtree[-1] + 1
    return token.doc[start:end].text

def get_verb_phrase(verb_token):
    verbs = []
    for child in verb_token.children:
        if child.dep_ in ("aux", "auxpass"):
            verbs.append(child.text)
    verbs.append(verb_token.text)
    return " ".join(verbs)

async def analisis_kalimat_spok(kalimat_id: str) -> dict:
    try:
        terjemahan_en = await translator.translate(kalimat_id, src='id', dest='en')
        kalimat_en = terjemahan_en.text
        print(f"   [INFO] Kalimat diterjemahkan ke Inggris: '{kalimat_en}'")
    except Exception as e:
        print(f"Error saat menerjemahkan ke Inggris: {e}")
        return {"S": "Error", "P": "Error", "O": "Error", "K": "Error"}

    doc = nlp(kalimat_en)
    komponen_en = {"S": [], "P": [], "O": [], "K": []}
    
    predikat_utama = next((token for token in doc if token.dep_ == "ROOT" and token.pos_ == "VERB"), None)

    if predikat_utama:
        komponen_en["P"].append(get_verb_phrase(predikat_utama))
        for token in predikat_utama.children:
            if token.dep_ in ["nsubj", "nsubjpass"]:
                komponen_en["S"].append(get_full_phrase(token))
            elif token.dep_ in ["dobj", "obj"]:
                komponen_en["O"].append(get_full_phrase(token))
            elif token.dep_ in ["prep", "agent", "advmod", "amod", "nmod", "obl", "pobj"]:
                komponen_en["K"].append(get_full_phrase(token))

    for key in komponen_en:
        komponen_en[key] = ", ".join(komponen_en[key])

    komponen_id = {}
    for key, value in komponen_en.items():
        if value:
            try:
                terjemahan_id = await translator.translate(value, src='en', dest='id')
                komponen_id[key] = terjemahan_id.text
            except Exception:
                komponen_id[key] = "Gagal Menerjemahkan Kembali"
        else:
            komponen_id[key] = "Tidak Terdeteksi"
            
    return komponen_id
# ======================================================================
# Akhir dari bagian kode yang tidak berubah
# ======================================================================


# ======================================================================
# --- PERUBAHAN UTAMA ADA DI FUNGSI MAIN DI BAWAH INI ---
# ======================================================================
async def main():
    # Tentukan nama file input
    nama_file = "kalimat.txt"
    file_input = Path(nama_file)

    # 1. Periksa apakah file ada
    if not file_input.is_file():
        print(f"❌ Error: File '{nama_file}' tidak ditemukan.")
        print("Pastikan file tersebut ada di folder yang sama dengan script Python.")
        return

    # 2. Baca semua baris dari file
    print(f"📖 Membaca kalimat dari file '{nama_file}'...")
    kalimat_list = file_input.read_text(encoding='utf-8').splitlines()
    print("-" * 40)

    # 3. Proses setiap baris/kalimat dalam file
    for i, kalimat in enumerate(kalimat_list):
        # Abaikan baris kosong
        if not kalimat.strip():
            continue

        print(f"Kalimat #{i+1}: '{kalimat}'")
        hasil = await analisis_kalimat_spok(kalimat)
        
        print(f"   - Subjek     : {hasil['S']}")
        print(f"   - Predikat   : {hasil['P']}")
        print(f"   - Objek      : {hasil['O']}")
        print(f"   - Keterangan : {hasil['K']}")
        print("-" * 40)


if __name__ == "__main__":
    asyncio.run(main())