from flask import Flask, render_template, request
import numpy as np
import skfuzzy as fuzz
import random
import pandas as pd 

app = Flask(__name__)

# ==========================================
# BAGIAN 1: FUZZY SUGENO (KIPAS OTOMATIS)
# ==========================================
def calculate_sugeno(input_suhu, input_kelembapan):
    """
    Menghitung kecepatan kipas (RPM) berdasarkan Suhu dan Kelembapan
    menggunakan metode Fuzzy Sugeno Orde 0.
    """
    
    # 1. DEFINISI VARIABEL & MF (MEMBERSHIP FUNCTIONS)
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

    # 2. KONSEKUEN (OUTPUT) - SUGENO ORDE 0 (KONSTAN)
    OUTPUT_PELAN  = 1000
    OUTPUT_SEDANG = 2500
    OUTPUT_CEPAT  = 5000

    # 3. FUZZIFIKASI
    derajat_suhu_dingin = fuzz.interp_membership(x_suhu, suhu_dingin, input_suhu)
    derajat_suhu_sedang = fuzz.interp_membership(x_suhu, suhu_sedang, input_suhu)
    derajat_suhu_panas  = fuzz.interp_membership(x_suhu, suhu_panas, input_suhu)
    
    derajat_kel_kering = fuzz.interp_membership(x_kelembapan, kelembapan_kering, input_kelembapan)
    derajat_kel_ideal  = fuzz.interp_membership(x_kelembapan, kelembapan_ideal, input_kelembapan)
    derajat_kel_lembap = fuzz.interp_membership(x_kelembapan, kelembapan_lembap, input_kelembapan)

    # 4. INFERENSI (ATURAN)
    # Aturan 1: IF Suhu Dingin THEN Pelan
    alpha_pelan = derajat_suhu_dingin

    # Aturan 2: IF Suhu Sedang OR Kelembapan Lembap THEN Sedang
    alpha_2 = np.fmax(derajat_suhu_sedang, derajat_kel_lembap)

    # Aturan 3: IF Suhu Panas AND Kelembapan BUKAN Kering THEN Cepat
    derajat_kel_BUKAN_kering = 1.0 - derajat_kel_kering
    alpha_3 = np.fmin(derajat_suhu_panas, derajat_kel_BUKAN_kering)

    alphas = {
        "alpha_pelan": alpha_pelan, 
        "alpha_2": alpha_2, 
        "alpha_3": alpha_3
    }

    # 5. DEFUZZIFIKASI (WEIGHTED AVERAGE)
    total_alpha = alpha_pelan + alpha_2 + alpha_3
    kecepatan_final = 0.0

    if total_alpha != 0:
        pembilang = (alpha_pelan * OUTPUT_PELAN) + \
                    (alpha_2 * OUTPUT_SEDANG) + \
                    (alpha_3 * OUTPUT_CEPAT)
        kecepatan_final = pembilang / total_alpha
    
    return {
        "kecepatan_final": kecepatan_final, 
        "alphas": alphas
    }

# ==========================================
# BAGIAN 2: GA KNAPSACK (OPTIMASI TAS)
# ==========================================
GA_ITEMS = {
    "A" : {"weight" : 7, "value" : 5},
    "B" : {"weight" : 2, "value" : 4},
    "C" : {"weight" : 1, "value" : 7},
    "D" : {"weight" : 9, "value" : 2},
}
GA_CAPACITY = 15
GA_ITEM_LIST = list(GA_ITEMS.keys())
GA_N_ITEMS = len(GA_ITEM_LIST)

def ga_decode(chromosome):
    total_weight = 0
    total_value = 0
    chosen_items = []
    for gene, name in zip(chromosome, GA_ITEM_LIST):
      if gene == 1:
          total_weight += GA_ITEMS[name]["weight"]
          total_value += GA_ITEMS[name]["value"]
          chosen_items.append(name)
    return chosen_items, total_weight, total_value

def ga_fitness(chromosome):
    _, total_weight_, total_value_ = ga_decode(chromosome)
    if total_weight_ > GA_CAPACITY: return 0
    else: return total_value_

def ga_roulette_selection(population, fitnesses):
    total_fit = sum(fitnesses)
    if total_fit == 0: return random.choice(population)
    pick = random.uniform(0, total_fit)
    current = 0
    for chrom, fit in zip(population, fitnesses):
      current += fit
      if current > pick: return chrom
    return population[-1]

def ga_crossover(p1, p2):
    point = random.randint(1, len(p1) - 1)
    return p1[:point] + p2[point:], p2[:point] + p1[point:]

def ga_mutate(chromosome, mutation_rate):
    return [1 - g if random.random() < mutation_rate else g for g in chromosome]

def run_knapsack_ga(pop_size, generations, crossover_rate, mutation_rate):
    random.seed(42)
    logs = []
    population = [[random.randint(0, 1) for _ in range(GA_N_ITEMS)] for _ in range(pop_size)]

    for gen in range(generations):
        fitnesses = [ga_fitness(ch) for ch in population]
        best_index = fitnesses.index(max(fitnesses))
        best_chrom = population[best_index]
        
        # Logging setiap generasi 
        best_items, w, v = ga_decode(best_chrom)
        logs.append(f"Gen {gen+1}: Value Terbaik = {v} (Berat: {w})")

        new_population = []
        new_population.append(best_chrom)

        while len(new_population) < pop_size:
            parent1 = ga_roulette_selection(population, fitnesses)
            parent2 = ga_roulette_selection(population, fitnesses)
            if random.random() < crossover_rate:
                child1, child2 = ga_crossover(parent1, parent2)
            else :
                child1, child2 = parent1[:], parent2[:]
            child1 = ga_mutate(child1, mutation_rate)
            child2 = ga_mutate(child2, mutation_rate)
            new_population.extend([child1, child2])
        population = new_population[:pop_size]

    fitnesses = [ga_fitness(ch) for ch in population]
    best_index = fitnesses.index(max(fitnesses))
    best_chrom = population[best_index]
    best_items, w, v = ga_decode(best_chrom)
    
    return logs, {
        "chromosome": best_chrom,
        "chosen_items": best_items,
        "weight": w,
        "value": v,
        "fitness": ga_fitness(best_chrom),
        "capacity": GA_CAPACITY
    }

# ==========================================
# BAGIAN 3: GA TRAVELING SALESPERSON (TSP)
# ==========================================

def get_tsp_data():
    """
    Mengambil data kota dan matriks jarak.
    Menggunakan data dari CSV: '3.b. TSP - AG.xlsx - Sheet1.csv'
    """
    
    cities = ['A', 'B', 'C', 'D', 'E']

    matrix_data = [
        [0, 7, 5, 9, 9], # A
        [7, 0, 7, 2, 8], # B
        [5, 7, 0, 4, 3], # C
        [9, 2, 4, 0, 6], # D
        [9, 8, 3, 6, 0]  # E
    ]
    
    dist_matrix = np.array(matrix_data, dtype=float)

    return cities, dist_matrix

def tsp_route_distance(route, dist_matrix):
    d = sum(dist_matrix[route[i], route[(i+1)%len(route)]] for i in range(len(route)))
    return d

def tsp_create_individual(n):
    ind = list(range(n))
    random.shuffle(ind)
    return ind

def tsp_ordered_crossover(p1, p2):
    # Ordered Crossover
    size = len(p1)
    a, b = sorted(random.sample(range(size), 2))
    child = [-1] * size
    child[a:b+1] = p1[a:b+1]

    p2_idx = 0
    for i in range(size):
        if child[i] == -1:
            while p2[p2_idx] in child:
                p2_idx += 1
            child[i] = p2[p2_idx]
    return child

def tsp_swap_mutation(ind):
    # Swap Mutation
    a, b = random.sample(range(len(ind)), 2)
    ind[a], ind[b] = ind[b], ind[a]
    return ind

def run_tsp_ga(pop_size, generations, crossover_rate, mutation_rate, tournament_k=5, elite_size=1):
    cities, dist_matrix = get_tsp_data()
    n_cities = len(cities)
    
    # Helper Selection
    def tournament_selection(pop):
        k = random.sample(pop, min(tournament_k, len(pop)))
        return min(k, key=lambda ind: tsp_route_distance(ind, dist_matrix))

    # 1. Inisialisasi
    pop = [tsp_create_individual(n_cities) for _ in range(pop_size)]
    
    # Cari best awal
    best_ind = min(pop, key=lambda ind: tsp_route_distance(ind, dist_matrix))
    best_dist = tsp_route_distance(best_ind, dist_matrix)
    
    logs = []
    logs.append(f"Gen 0: Best Distance = {best_dist:.4f}")
    
    # 2. Loop Generasi
    for g in range(generations):
        pop = sorted(pop, key=lambda ind: tsp_route_distance(ind, dist_matrix))
        
        if tsp_route_distance(pop[0], dist_matrix) < best_dist:
            best_ind = pop[0][:]
            best_dist = tsp_route_distance(best_ind, dist_matrix)
        
        new_pop = []
        new_pop.extend(pop[:elite_size]) 
        
        while len(new_pop) < pop_size:
            p1 = tournament_selection(pop)
            p2 = tournament_selection(pop)
            
            if random.random() < crossover_rate:
                child = tsp_ordered_crossover(p1, p2)
            else:
                child = p1[:]
            
            if random.random() < mutation_rate:
                child = tsp_swap_mutation(child)
                
            new_pop.append(child)
            
        pop = new_pop
        
        # Log setiap 50 generasi
        if (g + 1) % 50 == 0:
             logs.append(f"Gen {g+1}: Best Distance = {best_dist:.4f}")

    # Hasil Akhir
    best_route_names = [cities[i] for i in best_ind]
    
    return logs, {
        "best_route_indices": best_ind,
        "best_route_names": best_route_names,
        "best_distance": best_dist,
        "cities": cities
    }

# ==========================================
# ROUTING FLASK
# ==========================================

@app.route('/')
def home():
    return render_template('index.html')

# RUTE FUZZY SUGENO
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
        except: pass
    return render_template('fuzzy.html', result=result, input_suhu=input_suhu, input_kelembapan=input_kelembapan)

# RUTE GA KNAPSACK
@app.route('/mutation', methods=['GET', 'POST'])
def mutation_page():
    form_data = {'pop_size': 10, 'generations': 10, 'crossover_rate': 0.8, 'mutation_rate': 0.1}
    logs = None; final_result = None; error = None

    if request.method == 'POST':
        try:
            pop_size = int(request.form['pop_size'])
            generations = int(request.form['generations'])
            crossover_rate = float(request.form['crossover_rate'])
            mutation_rate = float(request.form['mutation_rate'])
            form_data = request.form
            
            logs, final_result = run_knapsack_ga(pop_size, generations, crossover_rate, mutation_rate)
        except Exception as e: error = str(e)

    return render_template('mutation.html', form_data=form_data, logs=logs, final_result=final_result, error=error)

# RUTE GA TSP
@app.route('/tsp', methods=['GET', 'POST'])
def tsp_page():
    form_data = {
        'pop_size': 100, 
        'generations': 500, 
        'crossover_rate': 0.9,
        'mutation_rate': 0.2
    }
    logs = None
    final_result = None
    error = None

    if request.method == 'POST':
        try:
            pop_size = int(request.form['pop_size'])
            generations = int(request.form['generations'])
            crossover_rate = float(request.form['crossover_rate'])
            mutation_rate = float(request.form['mutation_rate'])
            
            form_data = {
                'pop_size': pop_size,
                'generations': generations,
                'crossover_rate': crossover_rate,
                'mutation_rate': mutation_rate
            }

            logs, final_result = run_tsp_ga(pop_size, generations, crossover_rate, mutation_rate)
            
        except Exception as e:
            error = f"Terjadi kesalahan: {e}"

    return render_template('tsp.html', form_data=form_data, logs=logs, final_result=final_result, error=error)

if __name__ == '__main__':
    app.run(debug=True)