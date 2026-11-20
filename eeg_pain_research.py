import os
os.environ["PYTHONWARNINGS"] = "ignore::UserWarning"

import re
import csv
import struct
import zlib
import warnings
import numpy as np
import scipy.io as sio
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. CONFIGURACIÓN GENERAL Y BANDAS DE FRECUENCIA
# ==============================================================================
DIRECTORIO = 'C:\\Datos_Control\\'
NUM_SUJETOS = 51
FS = 500  # Frecuencia de muestreo (500 Hz)

BANDAS = [
    ('Delta', 1.5, 4.0),
    ('Theta', 4.1, 8.0),
    ('Alpha', 8.1, 12.0),
    ('Beta1', 12.1, 20.0),
    ('Beta2', 20.1, 31.0),
    ('Gamma', 31.1, 60.0)
]

# Índices de los 20 canales somatosensoriales (base-0 para Python)
CANALES_SOMATO_IDX = [c - 1 for c in [13, 14, 47, 48, 5, 6, 33, 34, 18, 29, 30, 55, 56, 27, 28, 41, 42, 23, 24, 61]]

# ==============================================================================
# 2. DECODIFICADOR DE EVENTOS1 (MCOS STRING) DESDE .MAT
# ==============================================================================
def extraer_eventos1_mcos(ruta_archivo, n_eventos):
    """
    Extrae las etiquetas exactas de 'Eventos1' desde el bloque UTF-16LE del .mat.
    Los marcadores S, L y T tienen exactamente 1 dígito, seguidos de comentarios numéricos o texto.
    """
    with open(ruta_archivo, 'rb') as f:
        data = f.read()

    subsys_offset = struct.unpack('<Q', data[116:124])[0]
    dtype, nbytes = struct.unpack('<II', data[subsys_offset : subsys_offset + 8])
    bloque = data[subsys_offset + 8 : subsys_offset + 8 + nbytes]
    if dtype == 15:
        bloque = zlib.decompress(bloque)

    inicio_utf16 = bloque.find('boundary'.encode('utf-16le'))
    if inicio_utf16 == -1:
        return None

    fin_utf16 = bloque.find('New Segment'.encode('utf-16le'), inicio_utf16)
    if fin_utf16 == -1:
        fin_utf16 = len(bloque)

    cadena_continua = bloque[inicio_utf16:fin_utf16].decode('utf-16le', errors='ignore')
    etiquetas = re.findall(r'boundary|Comment|[SLT]\s+\d|\d+', cadena_continua)

    if len(etiquetas) == n_eventos:
        return etiquetas

    print(f"  [Aviso] Se extrajeron {len(etiquetas)} etiquetas vs {n_eventos} esperadas.")
    return None

# ==============================================================================
# 3. LIMPIEZA DE EVENTOS Y EXTRACCIÓN DE CARACTERÍSTICAS
# ==============================================================================
def limpiar_eventos_por_intensidad(eventos1_str, eventos_num, mark_s, mark_l, val_l):
    n_eventos = len(eventos1_str)
    estado = np.zeros((n_eventos, 2), dtype=float)
    
    for i, ev in enumerate(eventos1_str):
        if mark_s in ev: estado[i, 0] += 1.0
        if 'S  9' in ev: estado[i, 0] += 9.0
        if 'S  8' in ev: estado[i, 0] += 8.0
        if 'T  3' in ev: estado[i, 0] += 3.0
        if mark_l in ev: estado[i, 0] += val_l
        if 'Comment' in ev or ev.strip().isdigit():
            estado[i, 1] = 100.0

    for i in range(n_eventos):
        if estado[i, 0] == 1.0:
            limite_4 = min(i + 5, n_eventos)
            if 9.0 in estado[i:limite_4, 0] or 8.0 in estado[i:limite_4, 0]:
                estado[i, 0] = 0.0
                continue
            
            limite_2 = min(i + 3, n_eventos)
            for j in range(i, limite_2):
                if estado[j, 1] == 100.0:
                    estado[j, 1] = 0.0
                    break
            
            for j in range(i, limite_4):
                if estado[j, 0] == 3.0:
                    estado[j, 0] *= 10.0
                    break
            
            limite_5 = min(i + 6, n_eventos)
            for j in range(i, limite_5):
                if estado[j, 1] == 100.0:
                    estado[j, 0] = eventos_num[j, 7] + 100.0
                    break
            
            for j in range(i, limite_5):
                if np.isclose(estado[j, 0], val_l):
                    estado[j, 0] *= 10.0
                    break

    pos_S = np.where(estado[:, 0] == 1.0)[0]
    pos_T3 = np.where(estado[:, 0] == 30.0)[0]
    pos_L = np.where(np.isclose(estado[:, 0], val_l * 10.0))[0]
    pos_Cm = np.where(estado[:, 0] >= 100.0)[0]

    min_len = min(len(pos_S), len(pos_T3), len(pos_L))
    if min_len == 0:
        return np.empty((0, 5))
    
    # Restamos 1 para convertir índices de muestra de base-1 (MATLAB) a base-0 (Python)
    t_aviso1 = eventos_num[pos_S[:min_len], 1] - 1
    t_aviso2 = eventos_num[pos_L[:min_len], 1] - 1
    t_final  = eventos_num[pos_T3[:min_len], 1] - 1
    comentarios = eventos_num[pos_Cm[:min_len], 7] if len(pos_Cm) >= min_len else np.zeros(min_len)
    t_coment = eventos_num[pos_Cm[:min_len], 1] - 1 if len(pos_Cm) >= min_len else np.zeros(min_len)

    return np.column_stack((t_aviso1, t_aviso2, t_final, comentarios, t_coment))

def calcular_bandpower_ventana(ventana_1d, fs=500):
    freqs, psd = welch(ventana_1d, fs=fs, nperseg=len(ventana_1d))
    idx_all = np.logical_and(freqs >= 1.5, freqs <= 60.0)
    allb = np.trapezoid(psd[idx_all], freqs[idx_all]) + 1e-12
    
    abs_powers, rel_powers = [], []
    for _, f_low, f_high in BANDAS:
        idx = np.logical_and(freqs >= f_low, freqs <= f_high)
        bp = np.trapezoid(psd[idx], freqs[idx])
        abs_powers.append(bp)
        rel_powers.append((bp / allb) * 100.0)
        
    return np.array(abs_powers), np.array(rel_powers)

def extraer_datos_ensayos(datos, data_all, fs=500):
    pb_dolor_abs, pb_dolor_rel = [], []
    pb_nopain_abs, pb_nopain_rel = [], []
    erd_ers_lista = []
    indices_validos = []

    num_canales = min(62, datos.shape[1] - 1)

    for i in range(len(data_all)):
        t_aviso1 = int(data_all[i, 0])
        t_laser = int(data_all[i, 1])

        inicio_erp = t_laser - FS
        fin_post = t_laser + FS + 1
        fin_np = t_aviso1 + FS + 1

        if inicio_erp < 0 or fin_post > datos.shape[0] or t_aviso1 < 0 or fin_np > datos.shape[0]:
            continue

        indices_validos.append(i)
        vec_d_abs, vec_d_rel, vec_np_abs, vec_np_rel, vec_erd = [], [], [], [], []

        for ch in range(1, num_canales + 1):
            sig_dolor = datos[t_laser:fin_post, ch]
            d_abs, d_rel = calcular_bandpower_ventana(sig_dolor, fs=fs)

            sig_pre = datos[inicio_erp:t_laser+1, ch]
            pre_abs, _ = calcular_bandpower_ventana(sig_pre, fs=fs)

            sig_np = datos[t_aviso1:fin_np, ch]
            np_abs, np_rel = calcular_bandpower_ventana(sig_np, fs=fs)

            erd = ((d_abs - pre_abs) / (pre_abs + 1e-12)) * 100.0

            vec_d_abs.extend(d_abs)
            vec_d_rel.extend(d_rel)
            vec_np_abs.extend(np_abs)
            vec_np_rel.extend(np_rel)
            vec_erd.extend(erd)

        pb_dolor_abs.append(vec_d_abs)
        pb_dolor_rel.append(vec_d_rel)
        pb_nopain_abs.append(vec_np_abs)
        pb_nopain_rel.append(vec_np_rel)
        erd_ers_lista.append(vec_erd)

    return {
        'dolor_abs': np.array(pb_dolor_abs),
        'dolor_rel': np.array(pb_dolor_rel),
        'nopain_abs': np.array(pb_nopain_abs),
        'nopain_rel': np.array(pb_nopain_rel),
        'erd_ers': np.array(erd_ers_lista),
        'data_all_valida': data_all[indices_validos] if indices_validos else np.empty((0, 5))
    }

def filtrar_canales_somato(matriz_X):
    cols = []
    for ch_idx in CANALES_SOMATO_IDX:
        inicio_col = ch_idx * 6
        cols.extend(range(inicio_col, inicio_col + 6))
    return matriz_X[:, cols]

def obtener_modelos():
    return {
        'KNN': KNeighborsClassifier(n_neighbors=5, metric='cosine'),
        'SVMLineal': SVC(kernel='linear'),
        'RandomForest': RandomForestClassifier(n_estimators=100, min_samples_leaf=10, n_jobs=1, random_state=42),
        'Perceptron': MLPClassifier(hidden_layer_sizes=(20, 10), activation='relu', max_iter=500, random_state=42)
    }

# ==============================================================================
# 4. EVALUACIÓN INTRA-SUJETO (K-FOLD ESTRATIFICADO)
# ==============================================================================
def evaluar_clasificadores_intra(X, y, nombre_experimento, sujeto, n_l1, n_l2, n_l3, registros_csv):
    if len(np.unique(y)) < 2 or len(y) < 10:
        return

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    modelos = obtener_modelos()
    accs = {m: [] for m in modelos}

    for train_idx, test_idx in skf.split(X, y):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])
        
        for nombre, clf in modelos.items():
            clf.fit(X_train, y[train_idx])
            y_pred = clf.predict(X_test)
            accs[nombre].append(np.mean(y_pred == y[test_idx]) * 100.0)
            
    promedios = {m: np.mean(accs[m]) for m in modelos}
    res_str = " | ".join([f"{m}: {promedios[m]:.1f}%" for m in modelos])
    print(f"  Sujeto {sujeto:02d} [{nombre_experimento:<26}] -> {res_str}")

    registros_csv.append({
        'Sujeto': f'Control{sujeto:02d}',
        'Ensayos_L1': n_l1,
        'Ensayos_L2': n_l2,
        'Ensayos_L3': n_l3,
        'Experimento': nombre_experimento,
        'KNN': round(promedios['KNN'], 2),
        'SVMLineal': round(promedios['SVMLineal'], 2),
        'RandomForest': round(promedios['RandomForest'], 2),
        'Perceptron': round(promedios['Perceptron'], 2)
    })

# ==============================================================================
# 5. EVALUACIÓN INTER-SUJETO (POTENCIA RELATIVA: 70/30 Y LEAVE-ONE-OUT)
# ==============================================================================
def evaluar_inter_sujeto_relativa(X, y, grupos, nombre_experimento, registros_inter):
    modelos = obtener_modelos()
    
    # A. Validación 70% sujetos entrenamiento / 30% sujetos prueba
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, grupos))
    
    scaler = StandardScaler()
    X_train_70 = scaler.fit_transform(X[train_idx])
    X_test_30 = scaler.transform(X[test_idx])
    
    suj_train = len(np.unique(grupos[train_idx]))
    suj_test = len(np.unique(grupos[test_idx]))
    
    accs_70_30 = {}
    for nombre, clf in modelos.items():
        clf.fit(X_train_70, y[train_idx])
        y_pred = clf.predict(X_test_30)
        accs_70_30[nombre] = np.mean(y_pred == y[test_idx]) * 100.0

    str_70_30 = " | ".join([f"{m}: {accs_70_30[m]:.1f}%" for m in modelos])
    print(f"\n  [INTER-SUJETO 70/30 ({suj_train} train / {suj_test} test)] {nombre_experimento}")
    print(f"    -> {str_70_30}")

    registros_inter.append({
        'Validacion': f'HoldOut_70_30 ({suj_train}v{suj_test} suj)',
        'Experimento': nombre_experimento,
        'KNN_Media': round(accs_70_30['KNN'], 2), 'KNN_DE': 0.0,
        'SVM_Media': round(accs_70_30['SVMLineal'], 2), 'SVM_DE': 0.0,
        'RF_Media': round(accs_70_30['RandomForest'], 2), 'RF_DE': 0.0,
        'MLP_Media': round(accs_70_30['Perceptron'], 2), 'MLP_DE': 0.0
    })

    # B. Validación Leave-One-Subject-Out (LOSO: 50 sujetos vs 1 sujeto)
    logo = LeaveOneGroupOut()
    accs_loso = {m: [] for m in modelos}
    
    for tr_idx, te_idx in logo.split(X, y, grupos):
        sc = StandardScaler()
        X_tr = sc.fit_transform(X[tr_idx])
        X_te = sc.transform(X[te_idx])
        for nombre, clf in modelos.items():
            clf.fit(X_tr, y[tr_idx])
            accs_loso[nombre].append(np.mean(clf.predict(X_te) == y[te_idx]) * 100.0)

    str_loso = " | ".join([f"{m}: {np.mean(accs_loso[m]):.1f}% (±{np.std(accs_loso[m]):.1f}%)" for m in modelos])
    print(f"  [INTER-SUJETO LOSO (Leave-One-Subject-Out)] {nombre_experimento}")
    print(f"    -> {str_loso}")

    registros_inter.append({
        'Validacion': 'Leave-One-Subject-Out (LOSO)',
        'Experimento': nombre_experimento,
        'KNN_Media': round(np.mean(accs_loso['KNN']), 2), 'KNN_DE': round(np.std(accs_loso['KNN']), 2),
        'SVM_Media': round(np.mean(accs_loso['SVMLineal']), 2), 'SVM_DE': round(np.std(accs_loso['SVMLineal']), 2),
        'RF_Media': round(np.mean(accs_loso['RandomForest']), 2), 'RF_DE': round(np.std(accs_loso['RandomForest']), 2),
        'MLP_Media': round(np.mean(accs_loso['Perceptron']), 2), 'MLP_DE': round(np.std(accs_loso['Perceptron']), 2)
    })

# ==============================================================================
# 6. BUCLE PRINCIPAL (51 SUJETOS) Y EXPORTACIÓN DE ARCHIVOS CSV
# ==============================================================================
print("Iniciando procesamiento de los 51 sujetos en VS Code...\n")

registros_intra = []
global_X_rel_pain_np, global_y_pain_np, global_grp_pain_np = [], [], []
global_X_rel_1y3, global_y_1y3, global_grp_1y3 = [], [], []
global_X_rel_3niv, global_y_3niv, global_grp_3niv = [], [], []

for sujeto in range(1, NUM_SUJETOS + 1):
    archivo = os.path.join(DIRECTORIO, f'Control{sujeto}.mat')
    if not os.path.exists(archivo):
        continue
        
    mat = sio.loadmat(archivo, simplify_cells=True)
    if 'datos' not in mat or 'Eventos' not in mat:
        continue

    datos = mat['datos']
    eventos = mat['Eventos']
    
    eventos1 = extraer_eventos1_mcos(archivo, len(eventos))
    if eventos1 is None:
        print(f"Sujeto {sujeto:02d}: No se pudo decodificar Eventos1.")
        continue

    d_L1 = limpiar_eventos_por_intensidad(eventos1, eventos, 'S  1', 'L  1', 1.1)
    d_L2 = limpiar_eventos_por_intensidad(eventos1, eventos, 'S  2', 'L  1', 1.2)
    d_L3 = limpiar_eventos_por_intensidad(eventos1, eventos, 'S  3', 'L  1', 1.3)

    res1 = extraer_datos_ensayos(datos, d_L1, fs=FS)
    res2 = extraer_datos_ensayos(datos, d_L2, fs=FS)
    res3 = extraer_datos_ensayos(datos, d_L3, fs=FS)

    n_l1, n_l2, n_l3 = len(res1['dolor_abs']), len(res2['dolor_abs']), len(res3['dolor_abs'])
    if n_l1 == 0 or n_l2 == 0 or n_l3 == 0:
        print(f"Sujeto {sujeto:02d}: Ensayos insuficientes tras filtrado.")
        continue

    X_abs_3 = np.vstack((res1['dolor_abs'], res2['dolor_abs'], res3['dolor_abs']))
    X_rel_3 = np.vstack((res1['dolor_rel'], res2['dolor_rel'], res3['dolor_rel']))
    X_erd_3 = np.vstack((res1['erd_ers'], res2['erd_ers'], res3['erd_ers']))
    
    X_nopain_abs = np.vstack((res1['nopain_abs'], res2['nopain_abs'], res3['nopain_abs']))
    X_nopain_rel = np.vstack((res1['nopain_rel'], res2['nopain_rel'], res3['nopain_rel']))

    tiempos_total = np.concatenate((
        res1['data_all_valida'][:, 2] - res1['data_all_valida'][:, 1],
        res2['data_all_valida'][:, 2] - res2['data_all_valida'][:, 1],
        res3['data_all_valida'][:, 2] - res3['data_all_valida'][:, 1]
    ))
    p33, p66 = np.percentile(tiempos_total, [33, 66])
    y_time_3 = np.where(tiempos_total <= p33, 3, np.where(tiempos_total <= p66, 2, 1))

    # Matrices Pain vs No-Pain
    X_p_np_abs = np.vstack((X_abs_3, X_nopain_abs))
    X_p_np_rel = np.vstack((X_rel_3, X_nopain_rel))
    y_p_np = np.concatenate((np.ones(len(X_abs_3)), np.zeros(len(X_nopain_abs))))

    # Matrices Low vs High (clases 1 y 3)
    mask_1y3 = (y_time_3 != 2)
    X_1y3_abs = X_abs_3[mask_1y3]
    X_1y3_rel = X_rel_3[mask_1y3]
    X_1y3_erd = X_erd_3[mask_1y3]
    y_1y3 = y_time_3[mask_1y3]

    # Acumular Potencia Relativa para análisis Inter-Sujeto
    global_X_rel_pain_np.append(X_p_np_rel)
    global_y_pain_np.append(y_p_np)
    global_grp_pain_np.append(np.full(len(y_p_np), sujeto))

    global_X_rel_1y3.append(X_1y3_rel)
    global_y_1y3.append(y_1y3)
    global_grp_1y3.append(np.full(len(y_1y3), sujeto))

    global_X_rel_3niv.append(X_rel_3)
    global_y_3niv.append(y_time_3)
    global_grp_3niv.append(np.full(len(y_time_3), sujeto))

    print(f"--- Sujeto {sujeto:02d} (Ensayos válidos: L1={n_l1}, L2={n_l2}, L3={n_l3}) ---")
    evaluar_clasificadores_intra(X_p_np_abs, y_p_np, "Pain vs No-Pain Abs (1s)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(X_p_np_rel, y_p_np, "Pain vs No-Pain Rel (1s)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(X_1y3_abs, y_1y3, "Low vs High Abs (1s)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(X_1y3_rel, y_1y3, "Low vs High Rel (1s)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(X_1y3_erd, y_1y3, "Low vs High ERD/ERS (%)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(filtrar_canales_somato(X_1y3_rel), y_1y3, "Low vs High Rel (20ch)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(filtrar_canales_somato(X_1y3_erd), y_1y3, "Low vs High ERD (20ch)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(X_rel_3, y_time_3, "3 Niveles Rel (62ch)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(X_erd_3, y_time_3, "3 Niveles ERD/ERS (62ch)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(filtrar_canales_somato(X_rel_3), y_time_3, "3 Niveles Rel (20ch)", sujeto, n_l1, n_l2, n_l3, registros_intra)
    evaluar_clasificadores_intra(filtrar_canales_somato(X_erd_3), y_time_3, "3 Niveles ERD/ERS (20ch)", sujeto, n_l1, n_l2, n_l3, registros_intra)

# Exportar resultados intra-sujeto
ruta_csv_intra = os.path.join(DIRECTORIO, 'resultados_intra_sujeto_51.csv')
with open(ruta_csv_intra, 'w', newline='', encoding='utf-8') as f:
    campos = ['Sujeto', 'Ensayos_L1', 'Ensayos_L2', 'Ensayos_L3', 'Experimento', 'KNN', 'SVMLineal', 'RandomForest', 'Perceptron']
    writer = csv.DictWriter(f, fieldnames=campos)
    writer.writeheader()
    writer.writerows(registros_intra)

print("\n" + "="*80)
print("RESUMEN ESTADÍSTICO GLOBAL INTRA-SUJETO (MEDIA ± DESVIACIÓN ESTÁNDAR)")
print("="*80)

experimentos_unicos = list(dict.fromkeys(r['Experimento'] for r in registros_intra))
resumen_filas = []

for exp in experimentos_unicos:
    filas_exp = [r for r in registros_intra if r['Experimento'] == exp]
    fila_res = {'Experimento': exp, 'N_Sujetos': len(filas_exp)}
    linea_print = f"{exp:<26} (N={len(filas_exp)}) -> "
    
    for m in ['KNN', 'SVMLineal', 'RandomForest', 'Perceptron']:
        vals = [r[m] for r in filas_exp]
        media, de = np.mean(vals), np.std(vals)
        fila_res[f'{m}_Media'] = round(media, 2)
        fila_res[f'{m}_DE'] = round(de, 2)
        linea_print += f"{m}: {media:.1f}% (±{de:.1f}%) | "
    
    print(linea_print[:-3])
    resumen_filas.append(fila_res)

ruta_csv_resumen = os.path.join(DIRECTORIO, 'resumen_estadistico_51.csv')
with open(ruta_csv_resumen, 'w', newline='', encoding='utf-8') as f:
    campos_dict = ['Experimento', 'N_Sujetos', 'KNN_Media', 'KNN_DE', 'SVMLineal_Media', 'SVMLineal_DE', 'RandomForest_Media', 'RandomForest_DE', 'Perceptron_Media', 'Perceptron_DE']
    writer = csv.DictWriter(f, fieldnames=campos_dict)
    writer.writeheader()
    writer.writerows(resumen_filas)

# Evaluación Inter-Sujeto (Solo Potencia Relativa)
print("\n" + "="*80)
print("VALIDACIÓN ENTRE SUJETOS (CROSS-SUBJECT) - POTENCIA RELATIVA PURA (1s)")
print("="*80)

registros_inter = []

X_all_p_np = np.vstack(global_X_rel_pain_np)
y_all_p_np = np.concatenate(global_y_pain_np)
grp_all_p_np = np.concatenate(global_grp_pain_np)
evaluar_inter_sujeto_relativa(X_all_p_np, y_all_p_np, grp_all_p_np, "Pain vs No-Pain Rel (62ch)", registros_inter)

X_all_1y3 = np.vstack(global_X_rel_1y3)
y_all_1y3 = np.concatenate(global_y_1y3)
grp_all_1y3 = np.concatenate(global_grp_1y3)
evaluar_inter_sujeto_relativa(X_all_1y3, y_all_1y3, grp_all_1y3, "Low vs High Rel (62ch)", registros_inter)
evaluar_inter_sujeto_relativa(filtrar_canales_somato(X_all_1y3), y_all_1y3, grp_all_1y3, "Low vs High Rel (20ch Somato)", registros_inter)

X_all_3niv = np.vstack(global_X_rel_3niv)
y_all_3niv = np.concatenate(global_y_3niv)
grp_all_3niv = np.concatenate(global_grp_3niv)
evaluar_inter_sujeto_relativa(X_all_3niv, y_all_3niv, grp_all_3niv, "3 Niveles Rel (62ch)", registros_inter)
evaluar_inter_sujeto_relativa(filtrar_canales_somato(X_all_3niv), y_all_3niv, grp_all_3niv, "3 Niveles Rel (20ch Somato)", registros_inter)

ruta_csv_inter = os.path.join(DIRECTORIO, 'resultados_inter_sujeto_LOSO_70_30.csv')
with open(ruta_csv_inter, 'w', newline='', encoding='utf-8') as f:
    campos_inter = ['Validacion', 'Experimento', 'KNN_Media', 'KNN_DE', 'SVM_Media', 'SVM_DE', 'RF_Media', 'RF_DE', 'MLP_Media', 'MLP_DE']
    writer = csv.DictWriter(f, fieldnames=campos_inter)
    writer.writeheader()
    writer.writerows(registros_inter)

print("\n¡Proceso completado! Se han guardado 3 archivos CSV en C:\\Datos_Control\\:")
print(f"  1. {ruta_csv_intra}")
print(f"  2. {ruta_csv_resumen}")
print(f"  3. {ruta_csv_inter}")