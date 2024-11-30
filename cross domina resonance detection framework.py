import numpy as np
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq

class ResonanceAnalyzer:
    def __init__(self, sampling_rate=1000):
        self.sampling_rate = sampling_rate
        self.resonance_patterns = {}
        
    def analyze_wave_pattern(self, data, domain_name, threshold=0.1):
        """Analyze wave patterns across different physical domains"""
        # Time domain analysis
        t = np.arange(len(data)) / self.sampling_rate
        peaks, properties = find_peaks(data, height=threshold)
        
        # Frequency domain analysis
        yf = fft(data)
        xf = fftfreq(len(data), 1/self.sampling_rate)
        
        # Calculate resonance signature
        resonance_signature = {
            'peak_intervals': np.diff(peaks),
            'peak_heights': properties['peak_heights'],
            'dominant_frequencies': xf[np.argsort(np.abs(yf))[-3:]],
            'energy_distribution': np.abs(yf)**2
        }
        
        self.resonance_patterns[domain_name] = resonance_signature
        return resonance_signature
    
    def compare_domains(self, domain1, domain2):
        """Compare resonance patterns between different domains"""
        if domain1 not in self.resonance_patterns or domain2 not in self.resonance_patterns:
            raise ValueError("Domain not analyzed yet")
            
        pattern1 = self.resonance_patterns[domain1]
        pattern2 = self.resonance_patterns[domain2]
        
        # Calculate correlation coefficients
        peak_interval_corr = np.corrcoef(pattern1['peak_intervals'], pattern2['peak_intervals'])[0,1]
        frequency_overlap = len(set(pattern1['dominant_frequencies']) & 
                              set(pattern2['dominant_frequencies']))
        
        # Energy distribution similarity
        energy_similarity = np.corrcoef(pattern1['energy_distribution'], 
                                      pattern2['energy_distribution'])[0,1]
        
        return {
            'peak_interval_correlation': peak_interval_corr,
            'frequency_overlap': frequency_overlap,
            'energy_similarity': energy_similarity
        }

    def detect_cross_domain_resonance(self, threshold=0.8):
        """Detect significant cross-domain resonance patterns"""
        domains = list(self.resonance_patterns.keys())
        resonance_map = {}
        
        for i in range(len(domains)):
            for j in range(i+1, len(domains)):
                comparison = self.compare_domains(domains[i], domains[j])
                if (comparison['peak_interval_correlation'] > threshold or
                    comparison['energy_similarity'] > threshold):
                    resonance_map[(domains[i], domains[j])] = comparison
                    
        return resonance_map

def experimental_protocol():
    """Define experimental protocol for cross-domain testing"""
    protocols = {
        'fluid_dynamics': {
            'setup': 'Circular wave tank with controlled wave generator',
            'measurements': ['Surface height', 'Wave velocity', 'Energy dissipation'],
            'sampling_rate': 1000,  # Hz
            'duration': 60  # seconds
        },
        'acoustic_waves': {
            'setup': 'Cylindrical resonator with precise frequency control',
            'measurements': ['Pressure amplitude', 'Phase', 'Modal patterns'],
            'sampling_rate': 44100,  # Hz
            'duration': 30  # seconds
        },
        'electromagnetic': {
            'setup': 'Controlled EM cavity with variable frequency source',
            'measurements': ['Field strength', 'Standing wave patterns', 'Resonant modes'],
            'sampling_rate': 2000,  # Hz
            'duration': 45  # seconds
        },
        'quantum_oscillations': {
            'setup': 'Trapped ion system with laser cooling',
            'measurements': ['Energy levels', 'Transition frequencies', 'Coherence times'],
            'sampling_rate': 5000,  # Hz
            'duration': 20  # seconds
        }
    }
    return protocols