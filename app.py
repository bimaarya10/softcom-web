from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz

app = Flask(__name__)

# FUNGSI LOGIKA FUZZY SUGENO
def calculate_sugeno(input_suhu, input_kelembapan):
    """
    Fungsi ini berisi semua logika Fuzzy Sugeno untuk menghitung
    kecepatan kipas berdasarkan suhu dan kelembapan.
    """
    
    # 1. DEFINISI VARIABEL DAN MEMBERSHIP FUNCTIONS (MF)
    x_suhu = np.arange(0, 41, 1)
    x_kelembapan = np.arange(0, 101, 1)

    # MF Suhu
    suhu_dingin = fuzz.trapmf(x_suhu, [0, 0, 15, 20])
    suhu_sedang = fuzz.trimf(x_suhu, [15, 20, 25])
    suhu_panas  = fuzz.trapmf(x_suhu, [20, 25, 40, 40])

    # MF Kelembapan
    kelembapan_kering = fuzz.trimf(x_kelembapan, [0, 0, 40])
    kelembapan_ideal = fuzz.trimf(x_kelembapan, [30, 50, 70])
    kelembapan_lembap = fuzz.trimf(x_kelembapan, [60, 100, 100])

    # 2. DEFINISI KONSEKUEN (OUTPUT) SUGENO (Konstan Orde 0)
    # Diubah dari MATI (0) menjadi PELAN (1000)
    OUTPUT_PELAN  = 1000
    OUTPUT_SEDANG = 2500
    OUTPUT_CEPAT  = 5000

    # 3. FUZZIFIKASI INPUT
    derajat_suhu_dingin = fuzz.interp_membership(x_suhu, suhu_dingin, input_suhu)
    derajat_suhu_sedang = fuzz.interp_membership(x_suhu, suhu_sedang, input_suhu)
    derajat_suhu_panas  = fuzz.interp_membership(x_suhu, suhu_panas, input_suhu)
    
    derajat_kel_kering = fuzz.interp_membership(x_kelembapan, kelembapan_kering, input_kelembapan)
    derajat_kel_ideal  = fuzz.interp_membership(x_kelembapan, kelembapan_ideal, input_kelembapan)
    derajat_kel_lembap = fuzz.interp_membership(x_kelembapan, kelembapan_lembap, input_kelembapan)

    # 4. APLIKASI ATURAN (Rule Evaluation) -> Mencari 'alpha' (Firing Strength)
    
    # Aturan 1: JIKA Suhu IS Dingin MAKA Pelan (1000)
    alpha_pelan = derajat_suhu_dingin

    # Aturan 2: JIKA Suhu IS Sedang ATAU Kelembapan IS Lembap MAKA Sedang (2500)
    alpha_2 = np.fmax(derajat_suhu_sedang, derajat_kel_lembap)

    # Aturan 3: JIKA Suhu IS Panas DAN Kelembapan BUKAN Kering MAKA Cepat (5000)
    derajat_kel_BUKAN_kering = 1 - derajat_kel_kering
    alpha_3 = np.fmin(derajat_suhu_panas, derajat_kel_BUKAN_kering)

    # Mengemas alpha untuk ditampilkan di web
    alphas = {
        "alpha_pelan": alpha_pelan,
        "alpha_2": alpha_2,
        "alpha_3": alpha_3
    }

    # 5. AGREASI & OUTPUT (Weighted Average)
    total_alpha = alpha_pelan + alpha_2 + alpha_3
    kecepatan_final = 0

    if total_alpha != 0:
        # Menggunakan OUTPUT_PELAN dan alpha_pelan
        pembilang = (alpha_pelan * OUTPUT_PELAN) + \
                    (alpha_2 * OUTPUT_SEDANG) + \
                    (alpha_3 * OUTPUT_CEPAT)
        
        penyebut = total_alpha
        kecepatan_final = pembilang / penyebut
    
    # Mengembalikan hasil dalam bentuk dictionary
    return {
        "kecepatan_final": kecepatan_final,
        "alphas": alphas
    }

# ROUTING
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fuzzy', methods=['GET', 'POST'])
def fuzzy():
    result = None
    input_suhu = None
    input_kelembapan = None
    
    if request.method == 'POST':
        try:
            input_suhu = float(request.form['suhu'])
            input_kelembapan = float(request.form['kelembapan'])
            
            result = calculate_sugeno(input_suhu, input_kelembapan)
            
        except (KeyError, ValueError):
            pass

    return render_template('fuzzy.html', 
                             result=result, 
                             input_suhu=input_suhu, 
                             input_kelembapan=input_kelembapan)

@app.route('/pricing')
def pricing():
    return render_template('pricing.html')

if __name__ == '__main__':
    app.run(debug=True)