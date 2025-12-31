import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import pdist, squareform

def higuchi_fd(time_series, k_max=10):
    """Higuchi fractal dimension."""
    if len(time_series) < 10:
        return 1.0  # Default neutral value for very short segments
    N = len(time_series)
    L = []
    x = np.arange(1, min(k_max + 1,N//2))
    for k in x:
        Lk = []
        for m in range(k):
            idx = np.arange(m, N - k, k)
            if len(idx) < 2:
                continue
            diffs = np.abs(time_series[idx + k] - time_series[idx])
            Lk.append(np.sum(diffs) * (N - 1) / ((len(diffs)) * k**2))
        if len(Lk) == 0:
            return 1.0
        L.append(np.mean(Lk))
    if len(L) < 2:
        return 1.0
    return np.polyfit(np.log(x), np.log(L), 1)[0]  # Slope

def rosenstein_lyapunov(time_series, embed_dim=3, tau=1, mean_period=10):
    """Approximate Lyapunov exponent (Rosenstein method)."""
    # Phase space reconstruction
    N = len(time_series)
    if N < embed_dim * tau + 10:
        return 0.0  # Neutral value for short/empty
    # psr = np.array([time_series[i:N - (embed_dim-1)*tau:tau] for i in range(embed_dim)])
    # psr = psr.T
    
    # # Nearest neighbors
    # dist = squareform(pdist(psr))
    # nn = np.argmin(dist + np.eye(N - (embed_dim-1)*tau) * 1e10, axis=1)
    
    # # Divergence
    # div = np.mean(np.log(np.abs(psr[1:] - psr[nn[1:]]).mean(axis=1) + 1e-10))
    # return div / mean_period

    try:
        # Phase space reconstruction with fixed length
        rows = []
        max_start = N - (embed_dim - 1) * tau
        for i in range(embed_dim):
            row = time_series[i * tau : max_start + i * tau : tau]
            rows.append(row)
        
        # Pad shorter rows to match length
        max_len = max(len(r) for r in rows)
        psr = np.array([np.pad(r, (0, max_len - len(r)), constant_values=0) for r in rows]).T
        
        if psr.shape[0] < 2:
            return 0.0
        
        # Nearest neighbors (avoid self)
        dist = squareform(pdist(psr))
        np.fill_diagonal(dist, np.inf)
        nn = np.argmin(dist, axis=1)
        
        # Divergence
        div = []
        for i in range(len(psr) - mean_period):
            if nn[i] < len(psr):
                d = np.linalg.norm(psr[i + mean_period] - psr[nn[i] + mean_period])
                if d > 1e-10:
                    div.append(np.log(d))
        return np.mean(div) / mean_period if div else 0.0
    except:
        return 0.0

def sample_entropy(time_series, m=2, r=0.2):
    """Sample entropy."""
    # std = np.std(time_series)
    # return -np.log(entropy(time_series) / std) if std > 0 else 0

    if len(time_series) < m + 2:
        return 0.0
    std = np.std(time_series)
    if std == 0:
        return 0.0
    r = r * std
    
    def _phi(m):
        x = np.array([time_series[i:i+m] for i in range(len(time_series) - m + 1)])
        C = np.sum(np.max(np.abs(x[:, None] - x[None, :]), axis=2) <= r, axis=1)
        return np.sum(C) / len(C)
    
    try:
        return -np.log(_phi(m+1) / _phi(m)) if _phi(m) > 0 else 0.0
    except:
        return 0.0

def extract_chaos_features(phase_audio):
    """Extract 16D features per phase: 3 core + derivations."""
    # if len(phase_audio) < 10:
    #     return np.zeros(16)
    
    # # Core metrics
    # fd = higuchi_fd(phase_audio)
    # le = rosenstein_lyapunov(phase_audio)
    # ent = sample_entropy(phase_audio)
    
    # # Derivations (multi-scale, stats): e.g., windowed versions
    # windows = np.array_split(phase_audio, 4)  # 4 sub-windows
    # fds = [higuchi_fd(w) for w in windows]
    # les = [rosenstein_lyapunov(w) for w in windows]
    # ents = [sample_entropy(w) for w in windows]
    
    # features = np.array([fd, le, ent, 
    #                      np.mean(fds), np.std(fds), np.min(fds), np.max(fds),
    #                      np.mean(les), np.std(les), np.min(les), np.max(les),
    #                      np.mean(ents), np.std(ents), np.min(ents), np.max(ents)])
    # return features


    phase_audio = np.asarray(phase_audio).flatten()
    if len(phase_audio) == 0:
        return np.zeros(16)  # All zero for empty phase
    
    if len(phase_audio) < 10:
        # Very short → minimal variation
        base = np.array([1.0, 0.0, 0.0])  # FD≈1, LE=0, Entropy=0
    else:
        fd = higuchi_fd(phase_audio)
        le = rosenstein_lyapunov(phase_audio)
        ent = sample_entropy(phase_audio)
        base = np.array([fd, le, ent])
    
    # Multi-scale statistics (4 windows)
    if len(phase_audio) < 4:
        windows = [phase_audio] * 4
    else:
        splits = np.array_split(phase_audio, 4)
        windows = [s for s in splits if len(s) > 0] or [phase_audio]
    
    fds = [higuchi_fd(w) for w in windows]
    les = [rosenstein_lyapunov(w) for w in windows]
    ents = [sample_entropy(w) for w in windows]
    
    # Pad if fewer than 4 windows
    while len(fds) < 4:
        fds.append(1.0)
        les.append(0.0)
        ents.append(0.0)
    
    features = np.concatenate([
        base,
        [np.mean(fds), np.std(fds), np.min(fds), np.max(fds)],
        [np.mean(les), np.std(les), np.min(les), np.max(les)],
        [np.mean(ents), np.std(ents), np.min(ents), np.max(ents)]
    ])
    
    return features

def get_cpce_vector(phases):
    """Combine to 64D vector."""
    # vector = []
    # for phase in ['inspiration', 'compression', 'expulsion', 'glottis']:
    #     vector.append(extract_chaos_features(phases[phase]))
    # return np.concatenate(vector)

    vector = []
    for phase_name in ['inspiration', 'compression', 'expulsion', 'glottis']:
        phase = phases.get(phase_name, np.array([]))
        vector.append(extract_chaos_features(phase))
    return np.concatenate(vector)