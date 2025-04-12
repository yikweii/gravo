import numpy as np
import warnings

class FIR_filter:
    def __init__(self, coefficients):
        self.weights = coefficients.astype(np.float32)  # Ensure float32 precision
        self.buffer = np.zeros_like(coefficients)
        
    def filter(self, x):
        # Update buffer
        self.buffer = np.roll(self.buffer, 1)
        self.buffer[0] = x
        
        # Safe dot product with clipping
        try:
            output = np.clip(np.dot(self.weights, self.buffer), -1e6, 1e6)
            if not np.isfinite(output):
                raise ValueError
            return output
        except:
            return 0.0  # Fallback value

    def lms(self, error, mu):
        # Stable weight update with safeguards
        update = np.clip(mu * error * self.buffer, -1.0, 1.0)
        self.weights = np.clip(self.weights + update, -1e4, 1e4)
