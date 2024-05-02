import numpy as np

class GustOfWind:
    def __init__(self, mean_force=0.0, max_force=1.0, gust_duration=10, gust_interval=50):
        self.mean_force = mean_force
        self.max_force = max_force
        self.gust_duration = gust_duration
        self.gust_interval = gust_interval
        self.last_gust_start = -gust_interval
        self.current_gust_end = 0

    def get_wind_force(self, t):
        if t >= self.current_gust_end:
            self.last_gust_start = t
            self.current_gust_end = t + self.gust_interval + np.random.randint(0, self.gust_duration)
            self.wind_force = self.mean_force
            self.gust_direction = np.random.choice([-1, 1])  # Randomly choose the direction of the gust

        if t >= self.last_gust_start and t < self.current_gust_end:
            gust_strength = np.random.uniform(0, self.max_force)
            self.wind_force = self.mean_force + self.gust_direction * gust_strength

        return self.wind_force