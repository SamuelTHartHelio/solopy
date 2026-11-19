import numpy as np

### normalized PSD. Note: The normalization is dt*PSD/len(signal).
def PSD(signal, dt):

    # Generate the Autocorrelation Function
    autocorr = np.correlate(signal, signal, "same")

    # Calculate the FFT
    fft = np.fft.fft(autocorr)
    fft_freq = np.fft.fftfreq(len(autocorr), d=dt)
    idx = np.argsort(fft_freq)

    # Return
    return(fft_freq[idx], dt*np.abs(fft[idx]/len(signal)))

### normalized PSD. Note: The normalization is dt*PSD/len(signal).
def cross_PSD(signal1, signal2, dt):

    # Generate the Autocorrelation Function
    autocorr = np.correlate(signal1, signal2, "same")

    # Calculate the FFT
    fft = np.fft.fft(autocorr)
    fft_freq = np.fft.fftfreq(len(autocorr), d=dt)
    idx = np.argsort(fft_freq)

    # Return
    return(fft_freq[idx], dt*np.abs(fft[idx]/len(signal1)))

def _nextpow2(x):
    return np.ceil(np.log2(np.abs(x)))

def _Morlet_wavelet(t, OMEGA_0=6.0):
    wavelet = (np.pi**(-0.25))*(np.exp(1j*OMEGA_0*t) - np.exp(-0.5*OMEGA_0**2))*np.exp(-0.5*(t**2))
    return wavelet

def Wavelet(Br, deltaT, M, OMEGA_0=6.0, C=1.06):
    N = len(Br)
    temp = _nextpow2(2**(_nextpow2(N) + 1))
    L    = int(np.power(2,temp))
    So   = 4.0*deltaT

    temp2 = (np.array(list(range(-int(np.floor(N/2.0)+1), int(np.floor(N/2.0))))))*deltaT
    time_gn = temp2[2:]

    k_max_wavelet = 0.5
    S_max = k_max_wavelet*N*deltaT

    S_m = np.logspace(np.log10(So), np.log10(S_max), M)
    nu  = (1.0 + 1.0/(2.0*OMEGA_0**2))*OMEGA_0/(2.0*np.pi*S_m)

    P_nu = 4.0*np.pi/(C * OMEGA_0 * N)

    ### Initialize values
    gn     = np.zeros((int(M), int(L)), dtype=complex)
    FK_Br  = np.zeros((int(M), int(L)), dtype=complex)
    F_s_Br = np.zeros((int(M), int(L)), dtype=complex) 

    FK_Br = np.fft.fft(Br, L)
    for i in range(0,M):
        gn[i,:] = np.fft.fft(np.conj(_Morlet_wavelet(-time_gn/S_m[i])), L)
        F_s_Br[i,:] = (deltaT*S_m[i]**(-0.5))*np.fft.ifft(gn[i,:]*FK_Br)

    F_final = F_s_Br[:,int(np.floor(N/2.0)):int(np.floor(N*3./2.-1.))]

    power_r = (np.abs(F_final))**2
    power_all = P_nu*(power_r)
    global_ws = np.sum(power_all, axis=1)

    return [nu, global_ws/2., power_all, time_gn, F_final, S_m]

def wavepower_at_freq(nu, power_all, f_target):
    """
    DISCLAIMER: WRITTEN BY CHATGPT - MODIFIED BY S. Hart.
    Extract wavelet power at a time-dependent frequency.
    
    nu : (M,) array - Wavelet center frequencies
    power_all : (M, N) array - Wavelet power spectrum
    f_target : (N,) array - frequency at each time 

    Returns -- P_out : (N,) array - Wavelet power evaluated at f_target(t)
    """
    log_nu = np.log10(nu)
    interp_P = np.full(power_all.shape[1], np.nan)
    for j in range(power_all.shape[1]):
        f = f_target[j]
        log_f = np.log10(f)
        log_P = np.log10(power_all[:, j])
        if (f <= nu.min()) or (f >= nu.max()):
            interp_P[j] == np.nan
            continue
        idx = np.where(log_nu > log_f)[0][-1]
        interp_log_P = (log_P[idx]+log_P[idx+1])/2
        interp_P[j] = 10**interp_log_P
    return interp_P

def turbulence_analysis(time, b_vec, v_vec=None, den=None, Nwave=50):
    mu_0 = np.pi*4e-7
    mp = 1.6726 * 10**(-27)
    # Time resolution that will be used in the wavelet transform.
    dt = np.nanmean(np.diff(time)/np.timedelta64(1,'s'))

    if b_vec.shape[0] == 4: b_vec = b_vec.T 
    if v_vec.shape[0] == 4: v_vec = v_vec.T
    
    bmag = np.linalg.norm(b_vec, axis=1)

    wfreq, waveBx, powerBx, time_gn, Bx_cwt, _ = Wavelet(b_vec[:,0], dt, Nwave)
    _,     waveBy, powerBy, _,       By_cwt, _ = Wavelet(b_vec[:,1], dt, Nwave)
    _,     waveBz, powerBz, _,       Bz_cwt, _ = Wavelet(b_vec[:,2], dt, Nwave)

    Eb = np.abs(Bx_cwt)**2 + np.abs(By_cwt)**2 + np.abs(Bz_cwt)**2
    
    wavelet_dict = {'frequency' : wfreq,'time' : time,'bx_cwt' : Bx_cwt,
                    'by_cwt' : By_cwt, 'bz_cwt' : Bz_cwt, 'trace_b_cwt' : Eb}
    
    if np.any(v_vec) != None and np.any(den) != None:
        vmag = np.linalg.norm(v_vec, axis=1)

        # Get the angle between the field and flow directions
        # This is important for understanding the turbulence we are looking at!
        theta_bv = np.degrees(np.arccos(np.einsum('ij,ij->i', b_vec, v_vec)/(bmag * vmag)))
        avg_den = np.nanmean(den) 
        va_vec = (b_vec*10**(-9))/np.sqrt(mu_0*(avg_den*1e6)*mp)/1000. #Alfven Velocity
        
        zplus, zminus = v_vec + va_vec, v_vec - va_vec #Elsasser Variables

        br_sign = np.sign(np.nanmean(b_vec[:,0]))   # note: this is assuming rtn coordinates!

        if br_sign > 0:
            wplus = zplus
            wminus = zminus
        else:
            wplus = zminus
            wminus = zplus

        wfreq, waveBx, powerBx, time_gn, Vax_cwt, _ = Wavelet(va_vec[:,0], dt, Nwave)
        _,     waveBy, powerBy, _, Vay_cwt, _ = Wavelet(va_vec[:,1], dt, Nwave)
        _,     waveBz, powerBz, _, Vaz_cwt, _ = Wavelet(va_vec[:,2], dt, Nwave)

        _, waveVx, powerVx, _, Vx_cwt, _ = Wavelet(v_vec[:,0], dt, Nwave)
        _, waveVy, powerVy, _, Vy_cwt, _ = Wavelet(v_vec[:,1], dt, Nwave)
        _, waveVz, powerVz, _, Vz_cwt, _ = Wavelet(v_vec[:,2], dt, Nwave)

        _, waveZpx, powerZpx, _, Zpx_cwt, _ = Wavelet(wplus[:,0], dt, Nwave)
        _, waveZpy, powerZpy, _, Zpy_cwt, _ = Wavelet(wplus[:,1], dt, Nwave)
        _, waveZpz, powerZpz, _, Zpz_cwt, _ = Wavelet(wplus[:,2], dt, Nwave)

        _, waveZmx, powerZmx, _, Zmx_cwt, _ = Wavelet(wminus[:,0], dt, Nwave)
        _, waveZmy, powerZmy, _, Zmy_cwt, _ = Wavelet(wminus[:,1], dt, Nwave)
        _, waveZmz, powerZmz, _, Zmz_cwt, _ = Wavelet(wminus[:,2], dt, Nwave)

        Ev = np.abs(Vx_cwt)**2 + np.abs(Vy_cwt)**2 + np.abs(Vz_cwt)**2
        Eva = np.abs(Vax_cwt)**2 + np.abs(Vay_cwt)**2 + np.abs(Vaz_cwt)**2

        Ep = np.abs(Zpx_cwt)**2 + np.abs(Zpy_cwt)**2 + np.abs(Zpz_cwt)**2
        Em = np.abs(Zmx_cwt)**2 + np.abs(Zmy_cwt)**2 + np.abs(Zmz_cwt)**2

        sigma_c, sigma_r = (Ep - Em)/(Ep + Em), (Ev - Eva)/(Ev + Eva)

        wavelet_dict = wavelet_dict.copy()
        wavelet_dict.update({'vax_cwt' : Vax_cwt,'vay_cwt' : Vay_cwt,'vaz_cwt' : Vaz_cwt,
                             'vx_cwt' : Vx_cwt,'vy_cwt' : Vy_cwt,'vz_cwt' : Vz_cwt,
                             'zpx_cwt' : Zpx_cwt,'zpy_cwt' : Zpy_cwt,'zpz_cwt' : Zpz_cwt,
                             'zmx_cwt' : Zmx_cwt,'zmy_cwt' : Zmy_cwt,'zmz_cwt' : Zmz_cwt,
                             'trace_v_cwt' : Ev,'trace_va_cwt' : Eva,'trace_zp_cwt' : Ep,
                             'trace_zm_cwt' : Em,'sigma_c' : sigma_c,'sigma_r' : sigma_r})
    return wavelet_dict