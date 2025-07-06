import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import h5py
from scipy.spatial.distance import cdist
from scipy.integrate import odeint
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import warnings
warnings.filterwarnings('ignore')

# Try to use optimized libraries if available
try:
    import rebound
    REBOUND_AVAILABLE = True
except ImportError:
    REBOUND_AVAILABLE = False
    print("REBOUND not available - using custom N-body implementation")

try:
    from astropy import units as u
    from astropy.cosmology import FlatLambdaCDM
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    print("Astropy not available - using simplified cosmology")

@dataclass
class SimulationParameters:
    """Configuration parameters for galaxy simulation"""
    # Galaxy properties
    galaxy_mass: float = 1e12  # Solar masses
    galaxy_radius: float = 10.0  # kpc
    galaxy_type: str = "spiral"  # spiral, elliptical, irregular
    
    # Simulation parameters
    n_particles: int = 50000  # Reduced for M1 Pro memory constraints
    n_gas_particles: int = 20000
    n_star_particles: int = 25000
    n_dm_particles: int = 5000
    
    # Temporal parameters
    simulation_time: float = 1e9  # years
    base_timestep: float = 1e5  # years
    adaptive_timestep: bool = True
    supernova_timestep: float = 1e-3  # years (for high resolution)
    
    # Cosmological parameters
    h0: float = 70.0  # km/s/Mpc
    omega_m: float = 0.3
    omega_lambda: float = 0.7
    
    # Physics switches
    include_gravity: bool = True
    include_sph: bool = True
    include_supernovae: bool = True
    include_star_formation: bool = True
    include_agn_feedback: bool = False  # Disabled for performance
    
    # Output parameters
    output_interval: float = 1e6  # years
    high_res_output: bool = True
    
class AdaptiveTimestepping:
    """Manages adaptive timestepping for high temporal resolution"""
    
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.current_dt = params.base_timestep
        self.min_dt = params.supernova_timestep
        self.max_dt = params.base_timestep * 10
        self.events_active = []
        
    def update_timestep(self, particles, events):
        """Update timestep based on current system state"""
        # Check for active supernovae or other rapid events
        rapid_events = [e for e in events if e.is_active()]
        
        if rapid_events:
            # Use high temporal resolution during supernovae
            self.current_dt = self.min_dt
        else:
            # Dynamically adjust based on system evolution rate
            if not particles:
                self.current_dt = self.max_dt
                return self.current_dt

            velocities = np.array([p.velocity for p in particles])
            positions = np.array([p.position for p in particles])
            
            max_velocity = np.max(np.sqrt(np.sum(velocities**2, axis=1))) if velocities.size > 0 else 0

            if len(particles) > 1:
                min_distance = np.min(cdist(positions, positions) + np.eye(len(particles)) * 1e10)
            else:
                min_distance = 0
            
            # CFL condition for stability
            cfl_dt = 0.1 * min_distance / max_velocity if max_velocity > 0 else self.max_dt
            self.current_dt = min(cfl_dt, self.max_dt)
            
        return self.current_dt

class Particle:
    """Enhanced particle class with astrophysical properties"""
    
    def __init__(self, mass, position, velocity, particle_type='star'):
        self.mass = mass  # Solar masses
        self.position = np.array(position, dtype=float)  # kpc
        self.velocity = np.array(velocity, dtype=float)  # km/s
        self.particle_type = particle_type  # 'star', 'gas', 'dark_matter'
        
        # Stellar properties
        self.age = 0.0  # years
        self.metallicity = 0.02  # Z_sun
        self.stellar_mass = mass if particle_type == 'star' else 0.0
        
        # Gas properties
        self.temperature = 1e4 if particle_type == 'gas' else 0.0  # K
        self.density = 1e-24 if particle_type == 'gas' else 0.0  # g/cm^3
        self.sph_smoothing = 0.1  # kpc
        
        # Evolution tracking
        self.is_supernova_candidate = False
        self.supernova_timer = 0.0
        
    def update_stellar_evolution(self, dt):
        """Update stellar evolution including supernova checks"""
        if self.particle_type == 'star':
            self.age += dt
            
            # Simple stellar evolution - massive stars go supernova
            if self.stellar_mass > 8.0:  # Solar masses
                stellar_lifetime = 1e10 * (self.stellar_mass / 1.0)**(-2.5)  # years
                if self.age > stellar_lifetime and not self.is_supernova_candidate:
                    self.is_supernova_candidate = True
                    self.supernova_timer = 0.0

class SupernovaEvent:
    """High-resolution supernova event handler"""
    
    def __init__(self, particle, simulation_time):
        self.particle = particle
        self.start_time = simulation_time
        self.duration = 1000.0  # years
        self.phase = 'progenitor'  # progenitor, explosion, remnant
        self.energy_released = 1e51  # erg
        self.ejecta_mass = particle.stellar_mass * 0.8  # Solar masses
        self.remnant_mass = particle.stellar_mass * 0.2
        
    def is_active(self):
        """Check if supernova is in active phase"""
        return self.phase in ['explosion', 'early_remnant']
        
    def update(self, current_time, dt):
        """Update supernova evolution with high temporal resolution"""
        elapsed = current_time - self.start_time
        
        if elapsed < 0.1:  # First 0.1 years - core collapse
            self.phase = 'explosion'
            # Model core collapse and explosion
            self.explosive_energy_injection(dt)
            
        elif elapsed < 10.0:  # Next 10 years - shock expansion
            self.phase = 'early_remnant'
            self.shock_propagation(dt)
            
        elif elapsed < self.duration:  # Long-term evolution
            self.phase = 'remnant'
            self.remnant_evolution(dt)
            
        else:
            self.phase = 'complete'
            
    def explosive_energy_injection(self, dt):
        """Model energy injection during explosion phase"""
        # Simplified energy injection model
        energy_rate = self.energy_released / (0.1 * 365.25 * 24 * 3600)  # erg/s
        return energy_rate * dt
        
    def shock_propagation(self, dt):
        """Model shock wave propagation through ISM"""
        # Simplified Sedov-Taylor blast wave
        shock_velocity = (self.energy_released / (4 * np.pi * self.particle.density))**(1/5)
        shock_radius = shock_velocity * dt
        return shock_radius
        
    def remnant_evolution(self, dt):
        """Model supernova remnant evolution"""
        # Create compact remnant (NS or BH)
        if self.particle.stellar_mass > 25.0:
            remnant_type = 'black_hole'
        else:
            remnant_type = 'neutron_star'
        return remnant_type

class SPHGasDynamics:
    """Smoothed Particle Hydrodynamics implementation"""
    
    def __init__(self, particles, smoothing_length=0.1):
        self.particles = particles
        self.smoothing_length = smoothing_length
        self.gamma = 5.0/3.0  # Adiabatic index
        
    def kernel_function(self, r, h):
        """Wendland C2 kernel function"""
        q = r / h
        if q <= 2.0:
            return (7.0 / (4.0 * np.pi * h**3)) * (1 - q/2.0)**4 * (2*q + 1)
        else:
            return 0.0
            
    def compute_density(self, particle_positions):
        """Compute SPH density for gas particles"""
        densities = np.zeros(len(particle_positions))
        
        for i, pos_i in enumerate(particle_positions):
            for j, pos_j in enumerate(particle_positions):
                if i != j:
                    r = np.linalg.norm(pos_i - pos_j)
                    densities[i] += self.particles[j].mass * self.kernel_function(r, self.smoothing_length)
                    
        return densities
        
    def compute_pressure_force(self, particle_positions, densities):
        """Compute pressure forces using SPH"""
        forces = np.zeros_like(particle_positions)
        
        for i, pos_i in enumerate(particle_positions):
            for j, pos_j in enumerate(particle_positions):
                if i != j:
                    r_vec = pos_i - pos_j
                    r = np.linalg.norm(r_vec)
                    if r > 0:
                        # Pressure from ideal gas law
                        P_i = densities[i] * self.particles[i].temperature * 8.314e7 / 2.0  # Pressure
                        P_j = densities[j] * self.particles[j].temperature * 8.314e7 / 2.0
                        
                        # SPH pressure force
                        force_mag = -self.particles[j].mass * (P_i/densities[i]**2 + P_j/densities[j]**2)
                        forces[i] += force_mag * r_vec / r
                        
        return forces

class GalaxySimulation:
    """Main galaxy simulation class with high temporal resolution"""
    
    def __init__(self, params: SimulationParameters):
        self.params = params
        self.particles = []
        self.time = 0.0
        self.timestep_manager = AdaptiveTimestepping(params)
        self.supernova_events = []
        self.sph_solver = None
        
        # Output data storage
        self.output_data = {
            'time': [],
            'positions': [],
            'velocities': [],
            'star_formation_rate': [],
            'supernova_rate': [],
            'metallicity': []
        }
        
        # Performance tracking
        self.performance_log = {
            'step_times': [],
            'memory_usage': [],
            'supernova_count': 0
        }
        
    def initialize_galaxy(self):
        """Initialize galaxy with realistic structure"""
        print("Initializing galaxy...")
        
        # Initialize dark matter halo (NFW profile)
        self._initialize_dark_matter_halo()
        
        # Initialize stellar disk
        self._initialize_stellar_disk()
        
        # Initialize gas disk
        self._initialize_gas_disk()
        
        # Initialize central black hole
        self._initialize_central_black_hole()
        
        # Initialize SPH solver
        gas_particles = [p for p in self.particles if p.particle_type == 'gas']
        if gas_particles:
            self.sph_solver = SPHGasDynamics(gas_particles)
            
        print(f"Galaxy initialized with {len(self.particles)} particles")
        
    def _initialize_dark_matter_halo(self):
        """Initialize NFW dark matter halo"""
        n_dm = self.params.n_dm_particles
        
        # NFW profile parameters
        r_s = self.params.galaxy_radius / 10.0  # Scale radius
        rho_0 = self.params.galaxy_mass / (4 * np.pi * r_s**3)  # Characteristic density
        
        for i in range(n_dm):
            # Sample radius from NFW profile
            r = self._sample_nfw_radius(r_s, self.params.galaxy_radius)
            
            # Random spherical coordinates
            theta = np.random.uniform(0, np.pi)
            phi = np.random.uniform(0, 2*np.pi)
            
            # Convert to Cartesian
            x = r * np.sin(theta) * np.cos(phi)
            y = r * np.sin(theta) * np.sin(phi)
            z = r * np.cos(theta)
            
            # Velocity from virial theorem
            v_circ = np.sqrt(4.3e-6 * self.params.galaxy_mass / r)  # km/s
            vx = v_circ * np.random.normal(0, 0.3)
            vy = v_circ * np.random.normal(0, 0.3)
            vz = v_circ * np.random.normal(0, 0.1)
            
            mass = self.params.galaxy_mass * 0.85 / n_dm  # 85% dark matter
            particle = Particle(mass, [x, y, z], [vx, vy, vz], 'dark_matter')
            self.particles.append(particle)
            
    def _initialize_stellar_disk(self):
        """Initialize stellar disk with exponential profile"""
        n_stars = self.params.n_star_particles
        
        # Exponential disk parameters
        r_d = self.params.galaxy_radius / 3.0  # Scale radius
        z_0 = 0.3  # Scale height (kpc)
        
        for i in range(n_stars):
            # Sample radius from exponential profile
            r = self._sample_exponential_radius(r_d, self.params.galaxy_radius)
            
            # Random azimuthal angle
            phi = np.random.uniform(0, 2*np.pi)
            
            # Height from sech^2 profile
            z = z_0 * np.random.laplace(0, 1)
            
            # Convert to Cartesian
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            # Circular velocity with dispersion
            v_circ = np.sqrt(4.3e-6 * self.params.galaxy_mass / r)
            v_phi = v_circ * (1 + np.random.normal(0, 0.1))
            
            vx = -v_phi * np.sin(phi) + np.random.normal(0, 30)
            vy = v_phi * np.cos(phi) + np.random.normal(0, 30)
            vz = np.random.normal(0, 10)
            
            # Stellar mass from IMF
            stellar_mass = self._sample_stellar_mass()
            
            particle = Particle(stellar_mass, [x, y, z], [vx, vy, vz], 'star')
            particle.age = np.random.uniform(0, 1e10)  # Random age
            self.particles.append(particle)
            
    def _initialize_gas_disk(self):
        """Initialize gas disk"""
        n_gas = self.params.n_gas_particles
        
        for i in range(n_gas):
            # Similar to stellar disk but more extended
            r = self._sample_exponential_radius(self.params.galaxy_radius / 2.0, self.params.galaxy_radius)
            phi = np.random.uniform(0, 2*np.pi)
            z = 0.1 * np.random.laplace(0, 1)  # Thinner than stellar disk
            
            x = r * np.cos(phi)
            y = r * np.sin(phi)
            
            # Circular velocity
            v_circ = np.sqrt(4.3e-6 * self.params.galaxy_mass / r)
            vx = -v_circ * np.sin(phi) + np.random.normal(0, 20)
            vy = v_circ * np.cos(phi) + np.random.normal(0, 20)
            vz = np.random.normal(0, 5)
            
            mass = self.params.galaxy_mass * 0.1 / n_gas  # 10% gas
            particle = Particle(mass, [x, y, z], [vx, vy, vz], 'gas')
            particle.temperature = 1e4  # Warm medium
            self.particles.append(particle)
            
    def _initialize_central_black_hole(self):
        """Initialize central supermassive black hole"""
        bh_mass = self.params.galaxy_mass * 0.001  # M_BH ~ 0.1% M_galaxy
        particle = Particle(bh_mass, [0, 0, 0], [0, 0, 0], 'black_hole')
        self.particles.append(particle)
        
    def _sample_nfw_radius(self, r_s, r_max):
        """Sample radius from NFW profile"""
        # Simplified inverse transform sampling
        u = np.random.uniform(0, 1)
        return r_s * u / (1 - u) if u < 0.99 else r_max
        
    def _sample_exponential_radius(self, r_d, r_max):
        """Sample radius from exponential profile"""
        u = np.random.uniform(0, 1)
        return -r_d * np.log(1 - u)
        
    def _sample_stellar_mass(self):
        """Sample stellar mass from Salpeter IMF"""
        # Simplified power-law IMF
        u = np.random.uniform(0, 1)
        return 0.1 * (1 / u)**(1/2.35)  # Salpeter slope
        
    def compute_gravitational_forces(self):
        """Compute gravitational forces between particles"""
        positions = np.array([p.position for p in self.particles])
        masses = np.array([p.mass for p in self.particles])
        forces = np.zeros_like(positions)
        
        # Gravitational constant in simulation units
        G = 4.3e-6  # kpc^3 / (M_sun * year^2)
        
        # Compute pairwise forces (N^2 algorithm - simplified for demo)
        for i in range(len(self.particles)):
            for j in range(i+1, len(self.particles)):
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                
                if r > 0.01:  # Softening length
                    force_mag = G * masses[i] * masses[j] / r**2
                    force_dir = r_vec / r
                    
                    forces[i] += force_mag * force_dir
                    forces[j] -= force_mag * force_dir
                    
        return forces
        
    def update_stellar_evolution(self, dt):
        """Update stellar evolution and trigger supernovae"""
        new_supernovae = []
        
        for particle in self.particles:
            if particle.particle_type == 'star':
                particle.update_stellar_evolution(dt)
                
                if particle.is_supernova_candidate and particle.supernova_timer == 0:
                    # Trigger supernova
                    sn_event = SupernovaEvent(particle, self.time)
                    new_supernovae.append(sn_event)
                    particle.supernova_timer = 1.0
                    
        self.supernova_events.extend(new_supernovae)
        return len(new_supernovae)
        
    def update_supernovae(self, dt):
        """Update all active supernova events with high temporal resolution"""
        active_supernovae = []
        
        for sn_event in self.supernova_events:
            if sn_event.phase != 'complete':
                sn_event.update(self.time, dt)
                
                if sn_event.is_active():
                    active_supernovae.append(sn_event)
                    
        return len(active_supernovae)
        
    def step(self):
        """Single simulation step with adaptive timestepping"""
        step_start = time.time()
        
        # Update timestep based on current conditions
        dt = self.timestep_manager.update_timestep(self.particles, self.supernova_events)
        
        # Compute forces
        if self.params.include_gravity:
            grav_forces = self.compute_gravitational_forces()
        else:
            grav_forces = np.zeros((len(self.particles), 3))
            
        # Update stellar evolution
        if self.params.include_supernovae:
            new_sn_count = self.update_stellar_evolution(dt)
            active_sn_count = self.update_supernovae(dt)
            
            if new_sn_count > 0:
                print(f"Time: {self.time:.2e} years - New supernovae: {new_sn_count}")
                self.performance_log['supernova_count'] += new_sn_count
        
        # SPH gas dynamics
        if self.params.include_sph and self.sph_solver:
            gas_positions = np.array([p.position for p in self.particles if p.particle_type == 'gas'])
            if len(gas_positions) > 0:
                densities = self.sph_solver.compute_density(gas_positions)
                pressure_forces = self.sph_solver.compute_pressure_force(gas_positions, densities)
                
                # Apply pressure forces to gas particles
                gas_idx = 0
                for i, particle in enumerate(self.particles):
                    if particle.particle_type == 'gas':
                        grav_forces[i] += pressure_forces[gas_idx]
                        gas_idx += 1
        
        # Integrate equations of motion (Leapfrog integration)
        for i, particle in enumerate(self.particles):
            acceleration = grav_forces[i] / particle.mass
            particle.velocity += acceleration * dt
            particle.position += particle.velocity * dt
            
        # Update simulation time
        self.time += dt
        
        # Performance logging
        step_time = time.time() - step_start
        self.performance_log['step_times'].append(step_time)
        
        return dt
        
    def run_simulation(self):
        """Run the complete galaxy simulation"""
        print("Starting galaxy simulation...")
        print(f"Target time: {self.params.simulation_time:.2e} years")
        print(f"Particles: {len(self.particles)}")
        
        # Main simulation loop
        last_output_time = 0
        step_count = 0
        
        while self.time < self.params.simulation_time:
            dt = self.step()
            step_count += 1
            
            # Output data at regular intervals
            if self.time - last_output_time >= self.params.output_interval:
                self.save_output()
                last_output_time = self.time
                
                # Progress report
                progress = 100 * self.time / self.params.simulation_time
                avg_step_time = np.mean(self.performance_log['step_times'][-100:])
                print(f"Progress: {progress:.1f}% | Time: {self.time:.2e} years | "
                      f"dt: {dt:.2e} years | Step time: {avg_step_time:.3f}s")
                
            # Memory management for long simulations
            if step_count % 10000 == 0:
                # Clean up completed supernova events
                self.supernova_events = [sn for sn in self.supernova_events if sn.phase != 'complete']
                
        print("Simulation completed!")
        self.generate_final_report()
        
    def save_output(self):
        """Save simulation output data"""
        positions = np.array([p.position for p in self.particles])
        velocities = np.array([p.velocity for p in self.particles])
        
        self.output_data['time'].append(self.time)
        self.output_data['positions'].append(positions.copy())
        self.output_data['velocities'].append(velocities.copy())
        
        # Calculate derived quantities
        star_particles = [p for p in self.particles if p.particle_type == 'star']
        gas_particles = [p for p in self.particles if p.particle_type == 'gas']
        
        # Star formation rate (simplified)
        sfr = len([p for p in star_particles if p.age < 1e8]) / 1e8  # M_sun/year
        self.output_data['star_formation_rate'].append(sfr)
        
        # Supernova rate
        recent_sn = len([sn for sn in self.supernova_events if (self.time - sn.start_time) < 1e6])
        sn_rate = recent_sn / 1e6  # per year
        self.output_data['supernova_rate'].append(sn_rate)
        
        # Average metallicity
        avg_metallicity = np.mean([p.metallicity for p in star_particles + gas_particles])
        self.output_data['metallicity'].append(avg_metallicity)
        
    def generate_visualizations(self):
        """Generate comprehensive visualizations"""
        print("Generating visualizations...")
        
        # 1. Galaxy structure plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Current positions
        positions = np.array([p.position for p in self.particles])
        particle_types = [p.particle_type for p in self.particles]
        
        # Face-on view
        ax = axes[0, 0]
        for ptype, color in [('star', 'yellow'), ('gas', 'blue'), ('dark_matter', 'gray')]:
            mask = np.array(particle_types) == ptype
            if np.any(mask):
                ax.scatter(positions[mask, 0], positions[mask, 1], 
                          c=color, s=0.5, alpha=0.6, label=ptype)
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Y (kpc)')
        ax.set_title('Galaxy Face-on View')
        ax.legend()
        ax.set_aspect('equal')
        
        # Edge-on view
        ax = axes[0, 1]
        for ptype, color in [('star', 'yellow'), ('gas', 'blue'), ('dark_matter', 'gray')]:
            mask = np.array(particle_types) == ptype
            if np.any(mask):
                ax.scatter(positions[mask, 0], positions[mask, 2], 
                          c=color, s=0.5, alpha=0.6, label=ptype)
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Z (kpc)')
        ax.set_title('Galaxy Edge-on View')
        ax.legend()
        ax.set_aspect('equal')
        
        # Evolution plots
        if len(self.output_data['time']) > 1:
            times = np.array(self.output_data['time']) / 1e6  # Myr
            
            # Star formation history
            ax = axes[1, 0]
            ax.plot(times, self.output_data['star_formation_rate'], 'b-', linewidth=2)
            ax.set_xlabel('Time (Myr)')
            ax.set_ylabel('Star Formation Rate (M☉/yr)')
            ax.set_title('Star Formation History')
            ax.grid(True, alpha=0.3)
            
            # Supernova rate
            ax = axes[1, 1]
            ax.plot(times, self.output_data['supernova_rate'], 'r-', linewidth=2)
            ax.set_xlabel('Time (Myr)')
            ax.set_ylabel('Supernova Rate (yr⁻¹)')
            ax.set_title('Supernova Rate Evolution')
            ax.grid(True, alpha=0.3)
            
        plt.tight_layout()
        plt.savefig('visualizations/galaxy_simulation_overview.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. Supernova evolution plot
        if self.supernova_events:
            self.plot_supernova_evolution()
            
    def plot_supernova_evolution(self):
        """Plot detailed supernova evolution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Supernova timeline
        ax = axes[0, 0]
        sn_times = [sn.start_time / 1e6 for sn in self.supernova_events]  # Myr
        sn_masses = [sn.particle.stellar_mass for sn in self.supernova_events]
        
        ax.scatter(sn_times, sn_masses, c='red', s=50, alpha=0.7)
        ax.set_xlabel('Time (Myr)')
        ax.set_ylabel('Progenitor Mass (M☉)')
        ax.set_title('Supernova Timeline')
        ax.grid(True, alpha=0.3)
        
        # Supernova spatial distribution
        ax = axes[0, 1]
        sn_positions = np.array([sn.particle.position for sn in self.supernova_events])
        if len(sn_positions) > 0:
            ax.scatter(sn_positions[:, 0], sn_positions[:, 1], 
                      c='red', s=100, alpha=0.7, marker='*')
        ax.set_xlabel('X (kpc)')
        ax.set_ylabel('Y (kpc)')
        ax.set_title('Supernova Spatial Distribution')
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        
        # Energy injection timeline
        ax = axes[1, 0]
        if self.supernova_events:
            # Calculate cumulative energy injection
            sn_times_sorted = sorted([(sn.start_time / 1e6, sn.energy_released) for sn in self.supernova_events])
            times, energies = zip(*sn_times_sorted)
            cumulative_energy = np.cumsum(energies) / 1e51  # in units of 10^51 erg
            
            ax.plot(times, cumulative_energy, 'r-', linewidth=2)
            ax.set_xlabel('Time (Myr)')
            ax.set_ylabel('Cumulative Energy (10⁵¹ erg)')
            ax.set_title('Supernova Energy Injection')
            ax.grid(True, alpha=0.3)
        
        # High-resolution supernova evolution example
        ax = axes[1, 1]
        if self.supernova_events:
            # Show detailed evolution of first supernova
            sn = self.supernova_events[0]
            time_points = np.linspace(0, 100, 1000)  # First 100 years
            phases = []
            
            for t in time_points:
                if t < 0.1:
                    phases.append('Explosion')
                elif t < 10:
                    phases.append('Shock Expansion')
                elif t < 100:
                    phases.append('Remnant')
                else:
                    phases.append('Complete')
            
            # Color code by phase
            colors = {'Explosion': 'red', 'Shock Expansion': 'orange', 
                     'Remnant': 'yellow', 'Complete': 'gray'}
            
            for phase in colors:
                mask = np.array(phases) == phase
                if np.any(mask):
                    ax.scatter(time_points[mask], np.ones(np.sum(mask)), 
                              c=colors[phase], label=phase, s=20)
            
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Supernova Phase')
            ax.set_title('High-Resolution SN Evolution')
            ax.legend()
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/supernova_evolution_detailed.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def save_data_to_hdf5(self, filename='data/galaxy_simulation_data.h5'):
        """Save simulation data to HDF5 format"""
        print(f"Saving simulation data to {filename}...")
        
        with h5py.File(filename, 'w') as f:
            # Simulation parameters
            param_group = f.create_group('parameters')
            param_group.attrs['galaxy_mass'] = self.params.galaxy_mass
            param_group.attrs['galaxy_radius'] = self.params.galaxy_radius
            param_group.attrs['simulation_time'] = self.params.simulation_time
            param_group.attrs['n_particles'] = len(self.particles)
            
            # Time series data
            time_group = f.create_group('time_series')
            time_group.create_dataset('time', data=self.output_data['time'])
            time_group.create_dataset('star_formation_rate', data=self.output_data['star_formation_rate'])
            time_group.create_dataset('supernova_rate', data=self.output_data['supernova_rate'])
            time_group.create_dataset('metallicity', data=self.output_data['metallicity'])
            
            # Particle data (final state)
            particle_group = f.create_group('particles')
            positions = np.array([p.position for p in self.particles])
            velocities = np.array([p.velocity for p in self.particles])
            masses = np.array([p.mass for p in self.particles])
            types = np.array([p.particle_type for p in self.particles], dtype='S10')
            
            particle_group.create_dataset('positions', data=positions)
            particle_group.create_dataset('velocities', data=velocities)
            particle_group.create_dataset('masses', data=masses)
            particle_group.create_dataset('types', data=types)
            
            # Supernova events
            if self.supernova_events:
                sn_group = f.create_group('supernovae')
                sn_times = [sn.start_time for sn in self.supernova_events]
                sn_masses = [sn.particle.stellar_mass for sn in self.supernova_events]
                sn_positions = np.array([sn.particle.position for sn in self.supernova_events])
                
                sn_group.create_dataset('times', data=sn_times)
                sn_group.create_dataset('progenitor_masses', data=sn_masses)
                sn_group.create_dataset('positions', data=sn_positions)
                
        print("Data saved successfully!")
        
    def generate_final_report(self):
        """Generate comprehensive simulation report"""
        print("\n" + "="*80)
        print("GALAXY SIMULATION FINAL REPORT")
        print("="*80)
        
        # Basic statistics
        print(f"\n📊 SIMULATION OVERVIEW:")
        print(f"  • Total simulation time: {self.time:.2e} years")
        print(f"  • Total particles: {len(self.particles)}")
        print(f"  • Final galaxy mass: {sum(p.mass for p in self.particles):.2e} M☉")
        
        # Particle breakdown
        particle_counts = {}
        for p in self.particles:
            particle_counts[p.particle_type] = particle_counts.get(p.particle_type, 0) + 1
            
        print(f"\n🔬 PARTICLE COMPOSITION:")
        for ptype, count in particle_counts.items():
            percentage = 100 * count / len(self.particles)
            print(f"  • {ptype.capitalize()}: {count:,} ({percentage:.1f}%)")
        
        # Supernova statistics
        print(f"\n💥 SUPERNOVA EVENTS:")
        print(f"  • Total supernovae: {len(self.supernova_events)}")
        print(f"  • Final supernova rate: {self.output_data['supernova_rate'][-1]:.2e} yr⁻¹")
        
        if self.supernova_events:
            sn_masses = [sn.particle.stellar_mass for sn in self.supernova_events]
            print(f"  • Average progenitor mass: {np.mean(sn_masses):.1f} M☉")
            print(f"  • Mass range: {np.min(sn_masses):.1f} - {np.max(sn_masses):.1f} M☉")
            
            # Energy injection
            total_energy = sum(sn.energy_released for sn in self.supernova_events)
            print(f"  • Total energy injected: {total_energy:.2e} erg")
        
        # Evolution metrics
        print(f"\n📈 GALAXY EVOLUTION:")
        if len(self.output_data['star_formation_rate']) > 1:
            initial_sfr = self.output_data['star_formation_rate'][0]
            final_sfr = self.output_data['star_formation_rate'][-1]
            print(f"  • Initial SFR: {initial_sfr:.2e} M☉/yr")
            print(f"  • Final SFR: {final_sfr:.2e} M☉/yr")
            
        if len(self.output_data['metallicity']) > 1:
            initial_z = self.output_data['metallicity'][0]
            final_z = self.output_data['metallicity'][-1]
            print(f"  • Initial metallicity: {initial_z:.3f}")
            print(f"  • Final metallicity: {final_z:.3f}")
            print(f"  • Enrichment factor: {final_z/initial_z:.2f}")
        
        # Performance metrics
        print(f"\n⚡ PERFORMANCE METRICS:")
        if self.performance_log['step_times']:
            avg_step_time = np.mean(self.performance_log['step_times'])
            total_steps = len(self.performance_log['step_times'])
            print(f"  • Total simulation steps: {total_steps:,}")
            print(f"  • Average step time: {avg_step_time:.3f} seconds")
            print(f"  • Total computation time: {sum(self.performance_log['step_times']):.1f} seconds")
        
        # Adaptive timestepping effectiveness
        if hasattr(self.timestep_manager, 'current_dt'):
            print(f"  • Final timestep: {self.timestep_manager.current_dt:.2e} years")
            print(f"  • Minimum timestep used: {self.timestep_manager.min_dt:.2e} years")
        
        # Observational comparisons
        print(f"\n🔭 OBSERVATIONAL COMPARISONS:")
        
        # Milky Way comparison
        if len(self.supernova_events) > 0:
            sn_rate_comparison = len(self.supernova_events) / (self.time / 1e2)  # per century
            print(f"  • Supernova rate: {sn_rate_comparison:.2f} per century")
            print(f"    (Milky Way observed: ~2-3 per century)")
        
        # Tully-Fisher relation check
        stellar_mass = sum(p.mass for p in self.particles if p.particle_type == 'star')
        print(f"  • Stellar mass: {stellar_mass:.2e} M☉")
        print(f"    (Milky Way: ~6×10¹⁰ M☉)")
        
        print(f"\n📁 OUTPUT FILES:")
        print(f"  • visualizations/galaxy_simulation_overview.png")
        print(f"  • visualizations/supernova_evolution_detailed.png")
        print(f"  • data/galaxy_simulation_data.h5")
        
        print("\n" + "="*80)
        print("SIMULATION COMPLETED SUCCESSFULLY!")
        print("="*80)

class OptimizedGalaxySimulation(GalaxySimulation):
    """Optimized version for M1 Pro performance"""
    
    def __init__(self, params: SimulationParameters):
        super().__init__(params)
        
        # M1 Pro specific optimizations
        self.use_multiprocessing = True
        self.n_cores = min(8, mp.cpu_count())  # M1 Pro has 8 cores
        
        # Reduce particle count for better performance
        if params.n_particles > 10000:
            print(f"Reducing particle count from {params.n_particles} to 10000 for M1 Pro optimization")
            scale_factor = 10000 / params.n_particles
            params.n_particles = 10000
            params.n_star_particles = int(params.n_star_particles * scale_factor)
            params.n_gas_particles = int(params.n_gas_particles * scale_factor)
            params.n_dm_particles = int(params.n_dm_particles * scale_factor)
    
    def compute_gravitational_forces_parallel(self):
        """Parallel computation of gravitational forces"""
        positions = np.array([p.position for p in self.particles])
        masses = np.array([p.mass for p in self.particles])
        
        # Split particles into chunks for parallel processing
        chunk_size = len(self.particles) // self.n_cores
        chunks = [range(i, min(i + chunk_size, len(self.particles))) 
                 for i in range(0, len(self.particles), chunk_size)]
        
        def compute_force_chunk(chunk_indices):
            """Compute forces for a chunk of particles"""
            forces = np.zeros((len(chunk_indices), 3))
            G = 4.3e-6  # Gravitational constant
            
            for i, particle_idx in enumerate(chunk_indices):
                for j in range(len(self.particles)):
                    if particle_idx != j:
                        r_vec = positions[j] - positions[particle_idx]
                        r = np.linalg.norm(r_vec)
                        
                        if r > 0.01:  # Softening
                            force_mag = G * masses[particle_idx] * masses[j] / r**2
                            forces[i] += force_mag * r_vec / r
            
            return forces
        
        # Parallel computation
        with ProcessPoolExecutor(max_workers=self.n_cores) as executor:
            force_chunks = list(executor.map(compute_force_chunk, chunks))
        
        # Combine results
        all_forces = np.vstack(force_chunks)
        return all_forces
    
    def step(self):
        """Optimized simulation step for M1 Pro"""
        step_start = time.time()
        
        # Use parallel gravity computation
        if self.params.include_gravity and self.use_multiprocessing:
            try:
                grav_forces = self.compute_gravitational_forces_parallel()
            except:
                # Fall back to serial computation if parallel fails
                grav_forces = self.compute_gravitational_forces()
        else:
            grav_forces = self.compute_gravitational_forces()
        
        # Continue with standard step procedure
        dt = self.timestep_manager.update_timestep(self.particles, self.supernova_events)
        
        # Update stellar evolution and supernovae
        if self.params.include_supernovae:
            new_sn_count = self.update_stellar_evolution(dt)
            active_sn_count = self.update_supernovae(dt)
            
            if new_sn_count > 0:
                print(f"Time: {self.time:.2e} years - New supernovae: {new_sn_count}")
                self.performance_log['supernova_count'] += new_sn_count
        
        # Integrate motion
        for i, particle in enumerate(self.particles):
            acceleration = grav_forces[i] / particle.mass
            particle.velocity += acceleration * dt
            particle.position += particle.velocity * dt
        
        self.time += dt
        
        # Performance tracking
        step_time = time.time() - step_start
        self.performance_log['step_times'].append(step_time)
        
        return dt

def create_demo_simulation():
    """Create a demonstration simulation optimized for M1 Pro"""
    
    # Optimized parameters for M1 Pro
    params = SimulationParameters(
        galaxy_mass=1e11,  # Slightly smaller galaxy
        galaxy_radius=8.0,  # kpc
        galaxy_type="spiral",
        
        # Reduced particle counts for performance
        n_particles=1000, # Actual value: 8000
        n_star_particles=500, # Actual value: 4000
        n_gas_particles=375, # Actual value: 3000
        n_dm_particles=125, # Actual value: 1000
        
        # Shorter simulation for demo
        simulation_time=1e7,  # 10 Myr (Actual value: 1e8)
        base_timestep=1e4,  # 10,000 years
        supernova_timestep=1e-2,  # 0.01 years for high resolution
        
        # Output settings
        output_interval=1e6,  # 1 Myr
        high_res_output=True,
        
        # Enable key physics
        include_gravity=True,
        include_sph=False,  # Disabled for performance
        include_supernovae=True,
        include_star_formation=True,
        include_agn_feedback=False
    )
    
    return OptimizedGalaxySimulation(params)

def run_galaxy_simulation():
    """Main function to run the complete galaxy simulation"""
    
    print("🌌 ADVANCED GALAXY SIMULATION")
    print("Optimized for Apple M1 Pro with High Temporal Resolution")
    print("="*60)
    
    # Create and initialize simulation
    sim = create_demo_simulation()
    
    print("\n🚀 Initializing simulation...")
    sim.initialize_galaxy()
    
    # Run simulation
    print("\n⚡ Starting simulation run...")
    start_time = time.time()
    
    try:
        sim.run_simulation()
        
        # Generate outputs
        print("\n📊 Generating visualizations...")
        sim.generate_visualizations()
        
        print("\n💾 Saving data...")
        sim.save_data_to_hdf5()
        
        total_time = time.time() - start_time
        print(f"\n✅ Simulation completed in {total_time:.1f} seconds")
        
        return sim
        
    except KeyboardInterrupt:
        print("\n⚠️  Simulation interrupted by user")
        return sim
    except Exception as e:
        print(f"\n❌ Simulation failed: {e}")
        return None

# Additional utility functions for analysis
def analyze_supernova_timing():
    """Analyze supernova timing and high temporal resolution"""
    
    print("\n🔬 SUPERNOVA TEMPORAL RESOLUTION ANALYSIS")
    print("="*50)
    
    # Create a focused supernova event for analysis
    class DetailedSupernovaAnalysis:
        def __init__(self):
            self.time_resolution = 0.001  # 0.001 years = ~8.8 hours
            self.total_duration = 100  # years
            self.time_points = np.arange(0, self.total_duration, self.time_resolution)
            
        def analyze_core_collapse_phase(self):
            """Detailed analysis of core collapse (first 0.1 years)"""
            collapse_times = self.time_points[self.time_points <= 0.1]
            
            # Core collapse dynamics
            core_radius = 10 * np.exp(-collapse_times / 0.01)  # km
            core_density = 1e14 * np.exp(collapse_times / 0.01)  # g/cm³
            
            return collapse_times, core_radius, core_density
            
        def analyze_explosion_phase(self):
            """Detailed analysis of explosion phase (0.1 to 10 years)"""
            explosion_times = self.time_points[(self.time_points > 0.1) & (self.time_points <= 10)]
            
            # Shock wave expansion
            shock_radius = 1000 * (explosion_times - 0.1)**0.4  # km
            shock_velocity = 0.4 * 1000 * (explosion_times - 0.1)**(-0.6)  # km/s
            
            return explosion_times, shock_radius, shock_velocity
            
        def plot_detailed_evolution(self):
            """Plot high temporal resolution supernova evolution"""
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Core collapse
            collapse_times, core_radius, core_density = self.analyze_core_collapse_phase()
            
            ax = axes[0, 0]
            ax.plot(collapse_times * 365.25 * 24, core_radius, 'r-', linewidth=2)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Core Radius (km)')
            ax.set_title('Core Collapse (High Resolution)')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            ax = axes[0, 1]
            ax.plot(collapse_times * 365.25 * 24, core_density, 'b-', linewidth=2)
            ax.set_xlabel('Time (days)')
            ax.set_ylabel('Core Density (g/cm³)')
            ax.set_title('Core Density Evolution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            # Explosion phase
            explosion_times, shock_radius, shock_velocity = self.analyze_explosion_phase()
            
            ax = axes[1, 0]
            ax.plot(explosion_times, shock_radius, 'g-', linewidth=2)
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Shock Radius (km)')
            ax.set_title('Shock Wave Expansion')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            ax = axes[1, 1]
            ax.plot(explosion_times, shock_velocity, 'orange', linewidth=2)
            ax.set_xlabel('Time (years)')
            ax.set_ylabel('Shock Velocity (km/s)')
            ax.set_title('Shock Velocity Evolution')
            ax.set_yscale('log')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('visualizations/supernova_high_resolution_analysis.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("✅ High-resolution supernova analysis complete!")
            print("📁 Saved: visualizations/supernova_high_resolution_analysis.png")
    
    # Run detailed analysis
    sn_analysis = DetailedSupernovaAnalysis()
    sn_analysis.plot_detailed_evolution()
    
    return sn_analysis

# Performance benchmarking
def benchmark_simulation_performance():
    """Benchmark simulation performance on M1 Pro"""
    
    print("\n🏃 PERFORMANCE BENCHMARKING")
    print("="*40)
    
    # Test different particle counts
    particle_counts = [1000, 2000, 5000, 8000, 10000]
    performance_results = []
    
    for n_particles in particle_counts:
        print(f"\n📊 Testing with {n_particles} particles...")
        
        # Create test parameters
        test_params = SimulationParameters(
            n_particles=n_particles,
            n_star_particles=int(n_particles * 0.5),
            n_gas_particles=int(n_particles * 0.375),
            n_dm_particles=int(n_particles * 0.125),
            simulation_time=1e6,  # Short test
            base_timestep=1e4
        )
        
        # Create test simulation
        test_sim = OptimizedGalaxySimulation(test_params)
        test_sim.initialize_galaxy()
        
        # Benchmark 10 steps
        start_time = time.time()
        for _ in range(10):
            test_sim.step()
        end_time = time.time()
        
        avg_step_time = (end_time - start_time) / 10
        performance_results.append((n_particles, avg_step_time))
        
        print(f"   Average step time: {avg_step_time:.3f} seconds")
    
    # Plot performance results
    fig, ax = plt.subplots(figsize=(10, 6))
    
    particles, step_times = zip(*performance_results)
    ax.plot(particles, step_times, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Particles')
    ax.set_ylabel('Average Step Time (seconds)')
    ax.set_title('M1 Pro Performance Scaling')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('visualizations/m1_pro_performance_benchmark.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ Performance benchmarking complete!")
    print("📁 Saved: visualizations/m1_pro_performance_benchmark.png")
    
    return performance_results

if __name__ == "__main__":
    # Run the complete simulation
    print("🌟 Starting Advanced Galaxy Simulation with High Temporal Resolution")
    
    # Main simulation
    simulation = run_galaxy_simulation()
    
    if simulation:
        # Additional analyses
        print("\n🔬 Running detailed supernova analysis...")
        analyze_supernova_timing()
        
        print("\n🏃 Running performance benchmarks...")
        benchmark_simulation_performance()
        
        print("\n🎉 All analyses complete!")
        print("Check the generated PNG files and HDF5 data for results.")
    else:
        print("❌ Simulation failed to complete")
