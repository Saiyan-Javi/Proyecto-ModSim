import random
import numpy as np
import math
import streamlit as st
import uuid
from copy import deepcopy
from typing import List, Dict, Any, Tuple
import io
import wave
import time

# ==============================================================================
# 1. SETUP DE DATOS Y CONSTANTES (Fase 1 y 2)
# ==============================================================================

# Diccionario de escalas musicales y sus acordes correspondientes (GA1)
scales_chords = {
    "C mayor": ["C", "Dm", "Em", "F", "G", "Am", "Bdim"],
    "G mayor": ["G", "Am", "Bm", "C", "D", "Em", "F#dim"],
    "D mayor": ["D", "Em", "F#m", "G", "A", "Bm", "C#dim"],
    "A mayor": ["A", "Bm", "C#m", "D", "E", "F#m", "G#dim"],
    "E mayor": ["E", "F#m", "G#m", "A", "B", "C#m", "D#dim"],
    "B mayor": ["B", "C#m", "D#m", "E", "F#", "G#m", "A#dim"],
    "F mayor": ["F", "Gm", "Am", "Bb", "C", "Dm", "Edim"],
    "Bb mayor": ["Bb", "Cm", "Dm", "Eb", "F", "Gm", "Adim"],
    "Eb mayor": ["Eb", "Fm", "Gm", "Ab", "Bb", "Cm", "Ddim"],
    "Ab mayor": ["Ab", "Bbm", "Cm", "Db", "Eb", "Fm", "Gdim"],
    "Db mayor": ["Db", "Ebm", "Fm", "Gb", "Ab", "Bbm", "Cdim"],
    "Gb mayor": ["Gb", "Abm", "Bbm", "Cb", "Db", "Ebm", "Fdim"],
    "Cb mayor": ["Cb", "Dbm", "Ebm", "Fb", "Gb", "Abm", "Bbdim"],

    "A menor": ["Am", "Bdim", "C", "Dm", "Em", "F", "G"],
    "E menor": ["Em", "F#dim", "G", "Am", "Bm", "C", "D"],
    "B menor": ["Bm", "C#dim", "D", "Em", "F#m", "G", "A"],
    "F# menor": ["F#m", "G#dim", "A", "Bm", "C#m", "D", "E"],
    "C# menor": ["C#m", "D#dim", "E", "F#m", "G#m", "A", "B"],
    "G# menor": ["G#m", "A#dim", "B", "C#m", "D#m", "E", "F#"],
    "D# menor": ["D#m", "E#dim", "F#", "G#m", "A#m", "B", "C#"],
    "A# menor": ["A#m", "B#dim", "C#", "D#m", "E#m", "F#", "G#"],

    "D menor": ["Dm", "Edim", "F", "Gm", "Am", "Bb", "C"],
    "G menor": ["Gm", "Adim", "Bb", "Cm", "Dm", "Eb", "F"],
    "C menor": ["Cm", "Ddim", "Eb", "Fm", "Gm", "Ab", "Bb"],
    "F menor": ["Fm", "Gdim", "Ab", "Bbm", "Cm", "Db", "Eb"],
    "Bb menor": ["Bbm", "Cdim", "Db", "Ebm", "Fm", "Gb", "Ab"],
    "Eb menor": ["Ebm", "Fdim", "Gb", "Abm", "Bbm", "Cb", "Db"],
    "Ab menor": ["Abm", "Bbdim", "Cb", "Dbm", "Ebm", "Fb", "Gb"]
}

# Matrices de transición de acordes (GA1)
M = np.array([[0,0,0,0,1,0,0],[1,0,0,0,1,0,0],[0,0,0,1,0,1,0],[1,0,0,0,1,0,0],[1,0,0,0,0,0,0],[0,1,0,1,0,0,0],[1,0,0,0,0,0,0]])
m = np.array([[0,0,0,0,1,0,0],[0,0,0,0,1,0,0],[0,0,0,1,0,1,0],[1,0,0,0,1,0,0],[1,0,0,0,0,0,0],[0,1,0,0,1,0,0],[1,0,0,0,0,0,0]])


# Parámetros del estado de ánimo (GA1)
estado_tonalidad = {'alegre': 'mayor', 'triste': 'menor', 'miedo': 'menor', 'calma': 'mayor',
                   'épico': 'mayor', 'funk': 'mayor', 'relajado': 'mayor', 'vals': 'mayor'}

estado_bpm = {'alegre': [130, 200], 'triste': [40, 75], 'miedo': [120, 200], 'calma': [60, 90],
              'épico': [100, 160], 'funk': [90, 120], 'relajado': [50, 80], 'vals': [130, 180]}

estado_compases ={'alegre': ['4/4'], 'triste': ['4/4', '3/4', '6/8'], 'miedo': ['4/4', '5/4', '7/8'], 'calma': ['4/4', '3/4'],
                  'épico': ['4/4'], 'funk': ['4/4'], 'relajado': ['4/4', '6/8'], 'vals': ['3/4']}

compases = ['3/4', '4/4', '5/4', '6/4', '6/8', '7/8', '12/8']


# Mapeo de Acciones (GA2) para visualización
ACTION_MAP = {
    0: '➖', # Silencio
    1: '🔵', # Tocar (Normal)
    2: '🟢', # Tocar (Suave/Ghost)
    3: '🟡', # Tocar (Fuerte)
    4: '🟠', # Tocar (Acento)
    5: '🔴'  # Tocar (Acento Fuerte)
}

# Parámetros para la generación de audio
SAMPLE_RATE = 44100
NOTE_FREQUENCY = 440 # Frecuencia base para el 'click' o 'pluck'

# ==============================================================================
# 2. FUNCIONES DE UTILIDAD DE AUDIO
# ==============================================================================

def generate_audio_data(pattern_genes: List[int], tempo: int, steps_per_bar: int) -> bytes:
    """Genera una simple secuencia de audio WAV para el patrón rítmico."""
    
    # Calcular la duración de un paso (en segundos)
    # Asumimos que la progresión es de 4 compases (4 acordes)
    beat_duration = 60.0 / tempo # Duración de una negra (4/4)
    # Duración del paso, relativa a la resolución rítmica
    subdivision_per_beat = steps_per_bar / 4 if steps_per_bar >= 4 else steps_per_bar
    if subdivision_per_beat == 0: subdivision_per_beat = 1 
    step_duration = beat_duration / subdivision_per_beat
    
    total_samples = int(SAMPLE_RATE * len(pattern_genes) * step_duration)
    
    audio_data = np.zeros(total_samples, dtype=np.float32)
    current_sample = 0
    
    for gene in pattern_genes:
        samples_in_step = int(SAMPLE_RATE * step_duration)
        t = np.linspace(0, step_duration, samples_in_step, False)
        
        amplitude = 0
        if gene > 0:
            # Diferentes amplitudes para las acciones (1-5)
            amplitude = 0.1 + (gene / 5.0) * 0.4
            
        # Generar onda seno para el paso
        step_waveform = amplitude * np.sin(2 * np.pi * NOTE_FREQUENCY * t)
        
        # Aplicar un simple envolvente de decaimiento para un sonido de 'pluck'
        decay_samples = min(samples_in_step, int(0.05 * SAMPLE_RATE))
        envelope = np.ones(samples_in_step)
        envelope[:decay_samples] = np.linspace(1, 0.1, decay_samples)
        envelope[decay_samples:] = 0.1 # Pequeño sostenido para evitar clicks
        
        step_waveform *= envelope
        
        end_sample = min(current_sample + samples_in_step, total_samples)
        audio_data[current_sample:end_sample] = step_waveform[:(end_sample - current_sample)]
        current_sample += samples_in_step

    # Convertir a 16-bit PCM
    audio_data_int16 = (audio_data * 32767).astype(np.int16)
    
    # Escribir en formato WAV
    buffer = io.BytesIO()
    with wave.open(buffer, 'wb') as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(2)  # 16 bits
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_data_int16.tobytes())
        
    return buffer.getvalue()


def play_pattern(pattern_genes: List[int], tempo: int, steps_per_bar: int):
    """Genera y muestra el widget de audio de Streamlit."""
    audio_bytes = generate_audio_data(pattern_genes, tempo, steps_per_bar)
    # Eliminamos el argumento 'key' de st.audio para compatibilidad
    st.audio(audio_bytes, format='audio/wav', start_time=0)


# ==============================================================================
# 3. FUNCIONES DEL ALGORITMO GENÉTICO 1 (COMPOSITOR)
# ==============================================================================

def random_chord(diccionario: dict = scales_chords) -> str:
    tonalidad = random.choice(list(diccionario.keys()))
    acorde = random.choice(scales_chords[tonalidad])
    return acorde

def generar_progresion(size: int, diccionario: dict = scales_chords) -> List[str]:
    return [random_chord() for _ in range(size)]

def find_key(progresion: List[str], diccionario: dict = scales_chords) -> str:
    value_max = -1
    tonalidad_optima = "No determinada"

    for key, chords in diccionario.items():
        value = sum(1 for chord in progresion if chord in chords)

        if value > value_max:
            value_max = value
            tonalidad_optima = key
    
    return tonalidad_optima

def fitness_function(progresion: List[str], diccionario: dict = scales_chords) -> int:
    longitud = len(progresion)
    score = 0
    tonalidad = find_key(progresion)
    
    try:
        escala = diccionario[tonalidad]
    except KeyError:
        return 0
        
    for i in range(longitud):
        acorde_actual = progresion[i]
        acorde_siguiente = progresion[(i+1) % longitud]
        
        if (acorde_actual in escala) and (acorde_siguiente in escala):
            score += 1
            
        try:
            index = escala.index(acorde_actual)
            next_index = escala.index(acorde_siguiente)
            
            if "mayor" in tonalidad:
                score += M[index][next_index]
            else:
                score += m[index][next_index]
                
        except (ValueError, IndexError):
            continue
            
    return score

def crossover_function(prog1: List[str], prog2: List[str]) -> List[str]:
    return prog1[:2] + prog2[2:]

def mutation_function(progresion: List[str], mutation_rate: float) -> List[str]:
    progresion_copy = progresion[:] 
    longitud = len(progresion_copy)
    for i in range(longitud):
        if random.random() < mutation_rate:
            progresion_copy[i] = random_chord()
    return progresion_copy

def calculate_new_fitness(prog_dict: Dict[str, Any],
                         estado_animo_tonalidad: str,
                         lim_inf: int,
                         lim_sup: int,
                         compases_tonalidad: List[str]):
  
  prog = prog_dict['prog']
  tempo = prog_dict['tempo']
  compass = prog_dict['compass']
  
  score = fitness_function(prog) 
  
  tonalidad_encontrada = find_key(prog)
  if estado_animo_tonalidad in tonalidad_encontrada:
    score += 5 
  
  if compass in compases_tonalidad:
    score += 2
  
  if tempo >= lim_inf and tempo <= lim_sup:
    score += 2
    
  return score

def generate_new_prog(estado_animo_tonalidad: str,
                      lim_inf: int,
                      lim_sup: int,
                      compases_tonalidad: List[str],
                      size:int = 4) -> Dict[str, Any]:
  
  prog = generar_progresion(size)
  compass = random.choice(compases)
  tempo = random.randint(0, 200)

  prog_dict = {
      'fitness': 0,
      'prog': prog,
      'tempo': tempo,
      'compass': compass
  }
  
  score = calculate_new_fitness(prog_dict, estado_animo_tonalidad, lim_inf, lim_sup, compases_tonalidad)
  prog_dict['fitness'] = score

  return prog_dict

def new_crossover_function(prog_dict_1: Dict[str, Any], prog_dict_2: Dict[str, Any]) -> Dict[str, Any]:
  
  prog_result = crossover_function(prog_dict_1['prog'], prog_dict_2['prog']) 
  
  tempo_1 = prog_dict_1['tempo']
  tempo_2 = prog_dict_2['tempo']
  tempo = random.randint(min(tempo_1, tempo_2), max(tempo_1, tempo_2))
  
  compass = random.choice([prog_dict_1['compass'], prog_dict_2['compass']])
  
  return {
      'fitness': 0, 
      'prog': prog_result,
      'tempo': tempo,
      'compass': compass
  }

def new_mutation_function(prog_dict: Dict[str, Any],
                          mutation_rate_1: float,
                          mutation_rate_2: float,
                          mutation_rate_3: float) -> Dict[str, Any]:
  
  mutated_dict = deepcopy(prog_dict) 
  
  mutated_dict['prog'] = mutation_function(mutated_dict['prog'], mutation_rate_1)
  
  if random.random() < mutation_rate_2:
    mutated_dict['tempo'] = random.randint(0, 200)
    
  if random.random() < mutation_rate_3:
    mutated_dict['compass'] = random.choice(compases)
    
  return mutated_dict

def new_genetic_algorithm_4music(pop_size: int,
                             iterations:int,
                             prog_size:int,
                             selection_rate: float,
                             mutation_rate_1: float,
                             mutation_rate_2: float,
                             mutation_rate_3: float,
                             estado_animo: str) -> Dict[str, Any]:

  mood = estado_animo.lower()
  estado_animo_tonalidad = estado_tonalidad[mood]
  lim_inf = estado_bpm[mood][0]
  lim_sup = estado_bpm[mood][1]
  compases_tonalidad = estado_compases[mood]

  sup_index = math.ceil(pop_size * selection_rate)

  population: List[Dict[str, Any]] = []
  for _ in range(pop_size):
    new_prog_dict = generate_new_prog(estado_animo_tonalidad, lim_inf, lim_sup, compases_tonalidad, prog_size)
    population.append(new_prog_dict)

  count = 0
  
  while count < iterations:
    population = sorted(population, key = lambda x: x['fitness'], reverse = True)[:sup_index]
    
    new_population = population[:] 
    while len(new_population) < pop_size:
      parent1 = random.choice(population) 
      parent2 = random.choice(population) 
      
      child_prog_dict_unscored = new_crossover_function(parent1, parent2) 
      
      child_score = calculate_new_fitness(child_prog_dict_unscored, estado_animo_tonalidad, lim_inf, lim_sup, compases_tonalidad)
      child_prog_dict_unscored['fitness'] = child_score
      new_population.append(child_prog_dict_unscored)

    population = new_population

    for i in range(len(population)):
      current_prog_dict = population[i] 
      
      mutated_prog_dict = new_mutation_function(current_prog_dict, mutation_rate_1, mutation_rate_2, mutation_rate_3) 
      
      mutated_score = calculate_new_fitness(mutated_prog_dict, estado_animo_tonalidad, lim_inf, lim_sup, compases_tonalidad)
      mutated_prog_dict['fitness'] = mutated_score
      
      population[i] = mutated_prog_dict

    count += 1
    
  return sorted(population, key = lambda x: x['fitness'], reverse = True)[0]

def run_progression_ga(mood: str) -> Dict[str, Any]:
    POP_SIZE = 500
    ITERATIONS = 50
    SELECTION_RATE = 0.2
    MUTATION_RATE = 0.05
    PROG_SIZE = 4 

    st.toast(f"Ejecutando GA1 para generar progresión '{mood}'...", icon="🎼")

    result_dict = new_genetic_algorithm_4music(
        pop_size=POP_SIZE, 
        iterations=ITERATIONS, 
        prog_size=PROG_SIZE, 
        selection_rate=SELECTION_RATE, 
        mutation_rate_1=MUTATION_RATE, 
        mutation_rate_2=MUTATION_RATE, 
        mutation_rate_3=MUTATION_RATE, 
        estado_animo=mood
    )
    
    return {
        'progression': result_dict['prog'],
        'time_signature': result_dict['compass'],
        'tempo': result_dict['tempo'],
        'mood': mood
    }

# ==============================================================================
# 4. FUNCIONES DEL ALGORITMO GENÉTICO 2 (INTÉRPRETE IGA)
# ==============================================================================

def get_steps_per_bar(time_signature: str, resolution: str) -> int:
    try:
        numerador, denominador = map(int, time_signature.split('/'))
        subdivisions = {'negra': 1, 'corchea': 2, 'semicorchea': 4}
        sub_factor = subdivisions.get(resolution, 2)
        
        if denominador == 4:
            return numerador * sub_factor
        elif denominador == 8:
            return (numerador // 3) * (sub_factor * 3) if numerador % 3 == 0 else numerador * sub_factor
        else:
            return numerador * sub_factor
            
    except Exception:
        return 16 

def calculate_total_chromosome_length(progression: list, time_signature: str, resolution: str) -> Tuple[int, int]:
    num_bars = len(progression)
    steps_per_bar = get_steps_per_bar(time_signature, resolution)
    total_length = num_bars * steps_per_bar
    return total_length, steps_per_bar

def _create_individual(chromosome_length: int) -> dict:
    genes = [random.randint(0, 5) for _ in range(chromosome_length)]
    return {
        "id": str(uuid.uuid4()),
        "genes": genes,
        "fitness": 0.0,
        "lock_mask": [False] * chromosome_length
    }

def _run_selection(population: list) -> Tuple[dict, dict]:
    p1 = max(random.sample(population, k=3), key=lambda p: p['fitness'])
    p2 = max(random.sample(population, k=3), key=lambda p: p['fitness'])
    return p1, p2

def _run_crossover(parent1: dict, parent2: dict) -> dict:
    genes1 = parent1['genes']
    genes2 = parent2['genes']
    crossover_point = random.randint(1, len(genes1) - 1)
    
    child_genes = genes1[:crossover_point] + genes2[crossover_point:]
    
    return {
        "id": str(uuid.uuid4()),
        "genes": child_genes,
        "fitness": 0.0,
        "lock_mask": [False] * len(child_genes)
    }

def _run_mutation_ga2(individual: dict, mutation_rate: float):
    genes = individual['genes']
    lock_mask = individual['lock_mask']
    for i in range(len(genes)):
        if not lock_mask[i] and random.random() < mutation_rate:
            genes[i] = random.randint(0, 5) 
    individual['genes'] = genes
    return individual

def setup_evolution(progression: list, time_signature: str, resolution: str,
                    pop_size: int, mood: str, tempo: int) -> dict:
    total_length, steps_per_bar = calculate_total_chromosome_length(
        progression, time_signature, resolution
    )
    initial_population = [_create_individual(total_length) for _ in range(pop_size)]
    
    initial_population[0]['genes'] = [1 if i % steps_per_bar == 0 else 0 for i in range(total_length)] 

    return {
        "progression": progression,
        "time_signature": time_signature,
        "resolution": resolution,
        "pop_size": pop_size,
        "total_length": total_length,
        "steps_per_bar": steps_per_bar,
        "population": initial_population,
        "generation": 0,
        "mood": mood,
        "tempo": tempo
    }

def evolve_next_generation(current_state: dict, ratings: dict, mutation_rate=0.05) -> dict:
    pop_copy = deepcopy(current_state['population'])
    
    for p in pop_copy:
        p['fitness'] = ratings.get(p['id'], 0.0)

    best_pattern = max(pop_copy, key=lambda p: p['fitness'])
    new_population = [deepcopy(best_pattern)]
    
    while len(new_population) < current_state['pop_size']:
        parent1, parent2 = _run_selection(pop_copy)
        child = _run_crossover(parent1, parent2)
        
        child = _run_mutation_ga2(child, mutation_rate)
        
        new_population.append(child)

    new_state = deepcopy(current_state)
    new_state['population'] = new_population
    new_state['generation'] = current_state['generation'] + 1
    return new_state

# Función para el modo automático de GA2
def run_full_auto_ga2(initial_state: dict, generations: int) -> dict:
    """Ejecuta GA2 automáticamente simulando calificaciones aleatorias."""
    
    current_state = deepcopy(initial_state)
    pop_size = current_state['pop_size']
    
    # Simple rate of increase for simulated fitness
    base_rating = 3.0 
    
    for gen in range(generations):
        # 1. Simular Calificaciones (asignar fitness aleatorio centrado en un promedio)
        # Esto simula que el "gusto" se mantiene, y los mejores evolucionan.
        ratings = {p['id']: base_rating + random.uniform(-1.0, 1.5) for p in current_state['population']}
        
        # 2. Evolucionar
        current_state = evolve_next_generation(current_state, ratings)
        current_state['generation'] = gen + 1
        
    # Devolver el mejor patrón de la última generación
    best_pattern = max(current_state['population'], key=lambda p: p['fitness'])
    return best_pattern

# --- Funciones de edición (Modo Pro) ---

def modify_gene_lock(pattern_id: str, gene_index: int):
    if 'evolution_state' in st.session_state:
        for p in st.session_state.evolution_state['population']:
            if p['id'] == pattern_id:
                p['lock_mask'][gene_index] = not p['lock_mask'][gene_index]
                break
        st.rerun()

def modify_gene_edit(pattern_id: str, gene_index: int, new_value: int):
    if 'evolution_state' in st.session_state:
        for p in st.session_state.evolution_state['population']:
            if p['id'] == pattern_id:
                p['genes'][gene_index] = new_value
                p['fitness'] = 10.0
                p['lock_mask'][gene_index] = True
                break
        st.rerun()

# ==============================================================================
# 5. INTERFAZ STREAMLIT
# ==============================================================================

def render_setup_ui():
    """Muestra la pantalla de configuración inicial (GA1 y GA2 Setup)."""
    
    st.markdown("<h2 style='color: #4B0082;'>🎵 1. Generar Composición (GA1: Compositor)</h2>", unsafe_allow_html=True)
    st.markdown("El compositor crea una progresión, tempo y compás coherentes con el estado de ánimo.")

    # --- PASO 1: EJECUTAR EL COMPOSITOR (GA1) ---
    mood_options = ['Alegre', 'Triste', 'Miedo', 'Calma', 'Épico', 'Funk', 'Relajado', 'Vals']
    mood = st.selectbox("Elige un Estado de Ánimo", [m.capitalize() for m in mood_options])
    
    if st.button("Generar Composición (GA1)", type="primary"):
        try:
            with st.spinner(f"El GA Compositor está generando música para el mood: {mood}..."):
                ga1_result = run_progression_ga(mood)
                st.session_state.ga1_result = ga1_result
                st.success(f"Composición Generada: {', '.join(ga1_result['progression'])} en {ga1_result['time_signature']} a {ga1_result['tempo']} BPM.")
                st.rerun()
        except KeyError:
            st.error(f"Error: El estado de ánimo '{mood}' no tiene parámetros musicales configurados. Por favor, revisa la configuración.")
        except Exception as e:
            st.error(f"Ocurrió un error inesperado al ejecutar el GA1: {e}")

    # --- PASO 2: CONFIGURAR EL INTÉRPRETE (GA2) ---
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='color: #1E90FF;'>🎶 2. Configurar Intérprete (GA2: Patrones Rítmicos)</h2>", unsafe_allow_html=True)
    st.markdown("El intérprete genera patrones rítmicos que evolucionan según tu calificación, o de forma automática.")

    # Valores por defecto
    progression_default = "C, G, Am, F"
    compas_options = compases
    compas_index = compas_options.index('4/4') if '4/4' in compas_options else 0
    tempo_default = 120
    mood_ga2 = mood

    if 'ga1_result' in st.session_state:
        ga1_data = st.session_state.ga1_result
        progression_default = ", ".join(ga1_data['progression'])
        
        if ga1_data['time_signature'] in compas_options:
             compas_index = compas_options.index(ga1_data['time_signature'])
        else:
            compas_options.insert(0, ga1_data['time_signature'])
            compas_index = 0
             
        tempo_default = ga1_data['tempo']
        mood_ga2 = ga1_data.get('mood', mood)
        st.info(f"Usando resultados del Compositor: **Mood: {mood_ga2}** | **Tempo: {tempo_default}** | **Compás: {ga1_data['time_signature']}**")


    # GA2 Configuration Widgets
    progression_str = st.text_input(
        "Progresión de Acordes (separados por coma, un acorde por compás):",
        progression_default
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        time_signature = st.selectbox(
            "Compás:",
            compas_options,
            index=compas_index
        )
    with col2:
        tempo = st.number_input(
            "Tempo (BPM):",
            30, 240, tempo_default
        )
    with col3:
        resolution = st.selectbox("Resolución Rítmica (Granularidad):", ['negra', 'corchea', 'semicorchea'], index=1, help="Corchea = 2 pasos/pulso. Semicorchea = 4 pasos/pulso.")

    pop_size = st.slider("Tamaño de Población (Nº de patrones rítmicos):", 5, 20, 10)
    
    # --- Opciones de Ejecución ---
    st.subheader("Modo de Ejecución GA2")
    col_mode1, col_mode2 = st.columns(2)
    
    with col_mode1:
        # Botón para iniciar el modo Interactivo (IGA)
        if st.button("🚀 Iniciar Evolución Interactiva (IGA)", type="secondary", use_container_width=True):
            progression_list = [chord.strip() for chord in progression_str.split(',') if chord.strip()]
            if not progression_list: st.error("Ingresa un acorde."); return
            initial_state = setup_evolution(progression_list, time_signature, resolution, pop_size, mood_ga2, tempo)
            st.session_state.evolution_state = initial_state
            if 'ga1_result' in st.session_state: del st.session_state.ga1_result
            st.rerun()

    with col_mode2:
        # Botón para iniciar el modo Automático (GA)
        auto_generations = st.number_input("Generaciones Automáticas", 5, 50, 15, step=5)
        if st.button("🤖 Ejecución Automática (GA)", type="secondary", use_container_width=True):
            progression_list = [chord.strip() for chord in progression_str.split(',') if chord.strip()]
            if not progression_list: st.error("Ingresa un acorde."); return
            
            with st.spinner(f"El GA Intérprete está evolucionando automáticamente por {auto_generations} generaciones..."):
                initial_state = setup_evolution(progression_list, time_signature, resolution, pop_size, mood_ga2, tempo)
                final_pattern = run_full_auto_ga2(initial_state, auto_generations)
                st.session_state.final_auto_result = {
                    'pattern': final_pattern,
                    'config': {'tempo': tempo, 'steps_per_bar': initial_state['steps_per_bar'], 'progression': progression_list}
                }
            st.success("¡Evolución automática completada!")
            if 'ga1_result' in st.session_state: del st.session_state.ga1_result
            st.rerun()
            

    # Mostrar Resultado Automático si existe
    if 'final_auto_result' in st.session_state:
        final_data = st.session_state.final_auto_result
        st.markdown("---")
        st.markdown("<h3 style='color: #28a745;'>✅ Resultado Automático Final</h3>", unsafe_allow_html=True)
        st.markdown(f"**Fitness Final:** {final_data['pattern']['fitness']:.2f}")
        
        # Visualización del patrón final
        final_genes_str = "".join([ACTION_MAP.get(gene, '?') for gene in final_data['pattern']['genes']])
        st.code(final_genes_str, language="text")
        
        # Botón de Reproducción
        col_play, _ = st.columns([0.2, 0.8])
        with col_play:
            if st.button("🔊 Reproducir Resultado Final", key='auto_play_btn'):
                with st.spinner("Generando audio..."):
                    # Llamada corregida sin el argumento 'key'
                    play_pattern(final_data['pattern']['genes'], final_data['config']['tempo'], final_data['config']['steps_per_bar'])

def render_evolution_ui():
    """Muestra la interfaz principal del IGA (calificación y evolución)."""

    state = st.session_state.evolution_state
    
    # --- Sidebar (Information and Controls) ---
    st.sidebar.markdown(f"## 🧬 Generación **{state['generation']}**", unsafe_allow_html=True)
    st.sidebar.markdown(f"**Mood:** {state['mood']}")
    st.sidebar.markdown(f"**Prog.:** {', '.join(state['progression'])}")
    st.sidebar.markdown(f"**Pasos/Compás:** {state['steps_per_bar']}")

    best_pattern = max(state['population'], key=lambda p: p['fitness'])
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Mejor Patrón Actual (Elitismo)")
    st.sidebar.markdown(f"**Fitness:** {best_pattern['fitness']:.2f}")
    
    # Rhythmic pattern visualization for the best individual
    best_genes_str = "".join([ACTION_MAP.get(gene, '?') for gene in best_pattern['genes']])
    st.sidebar.code(best_genes_str, language="text")

    if st.sidebar.button("Volver a Configuración"):
        keys_to_del = list(st.session_state.keys())
        for key in keys_to_del:
            del st.session_state[key]
        st.rerun()

    # --- Main UI (Pattern Rating) ---
    st.header(f"🎶 Califica los Patrones del Intérprete (IGA) - Generación {state['generation']}")
    st.caption("Usa el slider para calificar. El patrón con mayor calificación se convierte en el 'Elitismo' (sobrevive y se reproduce).")
    
    ratings_from_ui = {}
    
    num_bars = len(state['progression'])
    steps_per_bar = state['steps_per_bar']

    for i, pattern in enumerate(state['population']):
        st.divider()
        col_title, col_play_btn = st.columns([0.8, 0.2])
        with col_title:
            st.markdown(f"### Patrón {i + 1} (Fitness Anterior: {pattern['fitness']:.2f})")
        
        with col_play_btn:
            if st.button("🔊 Reproducir", key=f"play_btn_{pattern['id']}"):
                with st.spinner("Generando audio..."):
                    # Llamada corregida sin el argumento 'key'
                    play_pattern(pattern['genes'], state['tempo'], steps_per_bar)

        # Rhythmic Pattern Visualization per Bar
        cols = st.columns(num_bars)
        for bar_idx in range(num_bars):
            with cols[bar_idx]:
                st.markdown(f"**Acorde:** `{state['progression'][bar_idx]}`")
                start = bar_idx * steps_per_bar
                end = (bar_idx + 1) * steps_per_bar
                bar_genes = pattern['genes'][start:end]
                
                symbols = "".join([ACTION_MAP.get(gene, '?') for gene in bar_genes])
                st.code(symbols, language="text")
                
        # --- Rating Widget ---
        rating = st.slider("Calificación (1=Malo, 5=Excelente)", 1, 5, 3, key=f"rating_{pattern['id']}")
        ratings_from_ui[pattern['id']] = float(rating) 

        # --- "Pro Mode" Tools (Edit and Lock) ---
        with st.expander("🛠️ Herramientas de Edición (Modo Pro)"):
            st.markdown("Permite bloquear (preservar) genes específicos o editarlos manualmente.")
            
            sc1, sc2, sc3 = st.columns([1, 1, 1])
            with sc1:
                gene_idx = st.number_input("Índice del Gen (Global):", 0, state['total_length'] - 1, 0, key=f"pro_idx_{pattern['id']}")
            with sc2:
                new_val = st.number_input("Nuevo Valor del Gen (0-5):", 0, 5, 1, key=f"pro_val_{pattern['id']}")
            
            with sc3:
                 current_state = "Bloqueado 🔒" if pattern['lock_mask'][gene_idx] else "Libre 🔓"
                 st.info(f"Estado Gen {gene_idx}: {current_state}")

            b_col1, b_col2 = st.columns(2)
            with b_col1:
                if st.button("✏️ Editar y Bloquear Gen", key=f"edit_btn_{pattern['id']}"):
                    modify_gene_edit(pattern['id'], gene_idx, new_val)
            with b_col2:
                if st.button("🔒 Bloquear/Desbloquear", key=f"lock_btn_{pattern['id']}"):
                    modify_gene_lock(pattern['id'], gene_idx)

    st.divider()

    # Evolution Button
    if st.button("🧬 Evolucionar a la Siguiente Generación", type="primary", use_container_width=True):
        with st.spinner("Evolucionando patrones rítmicos..."):
            new_state = evolve_next_generation(state, ratings_from_ui)
            st.session_state.evolution_state = new_state
            st.rerun()

# --- MAIN APP LOGIC ---
if __name__ == "__main__":
    
    st.markdown("<h1 style='text-align: center; color: #4B0082;'>Sistema de Composición e Interpretación Musical 🎶</h1>", unsafe_allow_html=True)
    st.caption("Fase 1 (Compositor): Crea la base armónica. | Fase 2 (Intérprete IGA): Genera patrones rítmicos con tu guía o automáticamente.")
    
    if 'evolution_state' not in st.session_state:
        render_setup_ui()
    else:
        render_evolution_ui()