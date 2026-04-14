import numpy as np
from scipy.signal import correlate, stft, butter, filtfilt

# ============================================================
# 0. ADQUISICIÓN  Y PREPROCESAMIENTO DE DATOS
# ============================================================

def adquirir_senales():
    """
    Simula la captura de señales de sensores acústicos (p) e inducción (e).
    
    Returns:
        tuple: (p1, p2, e1, e2, fs, d)
               p: Presión, e: Inducción, fs: Frecuencia muestreo, d: Distancia micros.
    """
    fs = 44100
    dur = 0.2
    t = np.linspace(0, dur, int(fs*dur), endpoint=False)
    f0 = 500

    # Simulación de señales con ruido
    p1 = 0.5 * np.sin(2*np.pi*f0*t) + 0.01*np.random.randn(len(t))
    p2 = 0.5 * np.sin(2*np.pi*f0*(t - 0.0001)) + 0.01*np.random.randn(len(t))
    e1 = 0.1 * np.sin(2*np.pi*f0*t) + 0.01*np.random.randn(len(t))
    e2 = 0.1 * np.sin(2*np.pi*f0*t) + 0.01*np.random.randn(len(t))

    d = 0.1 # Distancia entre sensores en metros
    return p1, p2, e1, e2, fs, d


def filtro_dron(signal, fs):
    b, a = butter(4, [80/(fs/2), 600/(fs/2)], btype='band')
    return filtfilt(b, a, signal)


# ============================================================
# 1. FRECUENCIA DOMINANTE (OPTIMIZADA PARA DRONES)
# ============================================================

def frecuencia_dominante_dron(signal, fs, nperseg=1024):
    f, t, Zxx = stft(signal, fs=fs, nperseg=nperseg)
    magnitudes = np.abs(Zxx)

    # Solo frecuencias típicas de hélices
    mask = (f >= 80) & (f <= 600)
    f = f[mask]
    magnitudes = magnitudes[mask, :]

    idx = np.argmax(magnitudes, axis=0)
    return f[idx], t


# ============================================================
# 2. DIRECCIÓN (AZIMUT)
# ============================================================


def direccion_azimut(p1, p2, fs, d, c=343):
    """Calcula el ángulo de llegada (DoA) mediante correlación cruzada."""
    corr = correlate(p1, p2, mode='full')
    lag = np.argmax(corr) - (len(p1) - 1)
    delta_t = lag / fs
    arg = np.clip((c * delta_t) / d, -1, 1)
    theta = np.arcsin(arg)
    return theta # en radianes


# ============================================================
# 3. ALTITUD (RELACIÓN INDUCCIÓN / PRESIÓN)
# ============================================================

def altitud_relativa(p1, p2, e1, e2):
    P = np.abs(p1) + np.abs(p2)
    E = np.abs(e1) + np.abs(e2)
    R = E / (P + 1e-9)

    # Normalización robusta
    R_norm = (R - np.percentile(R, 5)) / (np.percentile(R, 95) - np.percentile(R, 5) + 1e-9)
    R_norm = np.clip(R_norm, 0, 1)

    # Mapear a 0–45° (drones rara vez llegan desde arriba del sensor)
    return np.mean(R_norm) * (np.pi / 4)


# ============================================================
# 4. ESTABILIDAD DE FRECUENCIA (RPM ESTABLE)
# ============================================================

def estabilidad_frecuencia(f_dom_resampled, ventana_muestras=200, delta_f=5):
    """Calcula la estabilidad S(t) basada en la varianza de la frecuencia."""
    S = np.zeros_like(f_dom_resampled)
    for i in range(len(f_dom_resampled)):
        ini = max(0, i - ventana_muestras)
        sub = f_dom_resampled[ini:i+1]
        sigma = np.std(sub)
        S[i] = np.exp(-(sigma*2) / (delta_f*2))
    return S


# ============================================================
# 5. MASA EFECTIVA (OPTIMIZADA PARA DRONES)
# ============================================================

def masa_efectiva_dron(p, f_dom_resampled, S, kappa=0.8, n=1.3):
    """
    Drones más grandes → frecuencias más bajas → mayor masa efectiva.
    """
    E = p**2
    m_bruta = kappa * E / (f_dom_resampled**n + 1e-9)
    return S * m_bruta


# ============================================================
# 6. CONSTRUIR D(t) Y T(t)
# ============================================================

def construir_D_T_dron(p1, p2, e1, e2, fs, d):
    # Filtrar señales para aislar hélices
    p1_f = filtro_dron(p1, fs)
    p2_f = filtro_dron(p2, fs)

    p_sum = p1_f + p2_f

    # Frecuencia dominante
    f_dom, t_f = frecuencia_dominante_dron(p_sum, fs)
    t_signal = np.linspace(0, len(p_sum)/fs, len(p_sum))
    t_f_norm = np.linspace(0, len(p_sum)/fs, len(f_dom))
    f_dom_resampled = np.interp(t_signal, t_f_norm, f_dom)

    # Estabilidad de RPM
    S = estabilidad_frecuencia(f_dom_resampled)

    # Masa efectiva
    m_eff = masa_efectiva_dron(p_sum, f_dom_resampled, S)

    # Dirección y altitud
    theta = direccion_azimut(p1_f, p2_f, fs, d)
    alpha = altitud_relativa(p1_f, p2_f, e1, e2)

    # D(t) = magnitud angular combinada
    D_const = np.sqrt(theta**2 + alpha**2)
    D_t = np.full_like(m_eff, D_const)

    # T(t) = masa efectiva
    T_t = m_eff

    return D_t, T_t, theta, alpha, m_eff


# ============================================================
# 7. ECUACIÓN DIFERENCIAL UNIFICADA (ESTABLE)
# ============================================================

def integrar_ecuacion_unificada_estable( D_t, T_t, F_xt ):

    K_D=1.0
    K_T=1.0
    L=1.0
    N=200
    PASOS=2000
    C=343
    CFL=0.9
    LAMBDA_DAMP=0.05

    dx = L / (N - 1)
    x = np.linspace(0, L, N)
    dt = CFL * dx / C

    Phi_prev = np.zeros(N)
    Phi = np.zeros(N)
    Phi_t = np.zeros(N)

    historia = []

    for n in range(PASOS):
        t = n * dt
        idx_t = min(n, len(D_t)-1)
        gamma = K_D * D_t[idx_t] + K_T * T_t[idx_t]

        F = F_xt(x, t)

        d2Phi_dx2 = np.zeros(N)
        d2Phi_dx2[1:-1] = (Phi[2:] - 2*Phi[1:-1] + Phi[:-2]) / dx**2

        Phi_next = (
            2*Phi - Phi_prev
            + dt**2 * (C**2 * d2Phi_dx2 - gamma * Phi + F)
            - LAMBDA_DAMP * dt * Phi_t
        )

        Phi_t = (Phi_next - Phi_prev) / (2*dt)

        Phi_next[0] = Phi_next[-1] = 0.0

        Phi_prev, Phi = Phi, Phi_next.copy()
        historia.append(Phi.copy())

    return x, np.array(historia), dt


# ============================================================
# 8. PIPELINE COMPLETO PARA DRONES
# ============================================================

def pipeline_dron(p1, p2, e1, e2, fs, d):
    D_t, T_t, theta, alpha, m_eff = construir_D_T_dron(p1, p2, e1, e2, fs, d)

    N_samples = len(p1)
    t_señal = np.linspace(0, N_samples / fs, N_samples)
    p_mean = (p1 + p2) / 2

    def F_xt(x, t):
        amp = np.interp(t, t_señal, p_mean)
        return amp * np.sin(np.pi * x)

    x, Phi_hist, dt = integrar_ecuacion_unificada_estable(D_t, T_t, F_xt)

    # --- Features extraídas de Phi_hist ---

    # 1. Energía temporal en cada punto espacial
    energia_espacial = np.sum(Phi_hist**2, axis=0)  # shape (200,)

    # 2. Centro de masa energético → dónde se concentra la perturbación
    energia_total = np.sum(energia_espacial) + 1e-12
    x_centroide = np.sum(x * energia_espacial) / energia_total  # escalar ∈ [0, 1]

    # 3. Asimetría espacial → refinamiento de theta
    # Si la energía se concentra en x<0.5, la fuente está más cerca del mic 1
    asimetria = x_centroide - 0.5  # ∈ [-0.5, 0.5]
    # Corrección pequeña sobre theta inicial
    delta_theta = asimetria * np.pi  # mapea [-0.5, 0.5] → [-π/2, π/2]
    theta_refinado = theta + 0.1 * delta_theta  # factor 0.1: no sobreescribir

    # 4. Velocidad de propagación efectiva → refinamiento de alpha
    # Comparamos energía en primera mitad vs segunda mitad temporal
    mitad = Phi_hist.shape[0] // 2
    E_primera = np.sum(Phi_hist[:mitad, :]**2)
    E_segunda = np.sum(Phi_hist[mitad:, :]**2)
    ratio_temporal = E_segunda / (E_primera + 1e-12)
    # Si ratio > 1: la energía crece (dron se acerca), < 1: se aleja
    delta_alpha = (ratio_temporal - 1.0) * 0.05
    alpha_refinado = alpha + delta_alpha

    # 5. Masa efectiva refinada desde amplitud máxima de Phi
    amp_max = np.max(np.abs(Phi_hist))  # escalar
    m_eff_refinada = np.mean(m_eff) * (1.0 + 0.05 * amp_max)

    return {
        # Estimaciones originales
        "direccion_rad":        theta,
        "altitud_rad":          alpha,
        "masa_efectiva_media":  float(np.mean(m_eff)),
        # Estimaciones refinadas por Phi_hist
        "direccion_refinada_rad":       float(theta_refinado),
        "altitud_refinada_rad":         float(alpha_refinado),
        "masa_efectiva_refinada":       float(m_eff_refinada),
        # Diagnóstico
        "x_centroide":          float(x_centroide),
        "ratio_temporal":       float(ratio_temporal),
        "asimetria":            float(asimetria),
        # Raw
        "x":                    x,
        "Phi_hist":             Phi_hist,
        "dt":                   dt,
    }