# Advanced Galaxy Simulation

This project provides a high-performance, high-temporal-resolution simulation of galaxy dynamics, optimized for modern multi-core processors like the Apple M1 Pro. It models the gravitational and hydrodynamic interactions of stars, gas, and dark matter, with a focus on capturing high-frequency astrophysical events like supernovae in detail.

## Features

- **N-body Simulation**: Simulates the gravitational interactions of thousands of particles representing stars, gas, and dark matter.
- **Smoothed Particle Hydrodynamics (SPH)**: Models gas dynamics, including pressure forces and shocks.
- **Stellar Evolution**: Includes a simplified model of stellar evolution, where massive stars can go supernova.
- **High-Temporal-Resolution Supernovae**: Implements an adaptive timestepping scheme to resolve the fine-grained details of supernova explosions.
- **Optimized for Performance**: Utilizes multiprocessing to parallelize force calculations, significantly speeding up the simulation.
- **Modular and Extensible**: The code is organized into classes for different physical components, making it easy to extend with new physics or models.
- **Comprehensive Output**: Generates detailed HDF5 data files and visualizations of the galaxy's evolution, star formation history, and supernova events.

## Project Structure

```
Galaxi-Simulation/
├── data/                     # Output data files (HDF5)
├── visualizations/           # Output plots and animations
├── src/
│   └── galaxy_simulation.py  # Core simulation code
├── .gitignore                # Git ignore file
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── run.py                    # Main script to run the simulation
```

## Getting Started

### Prerequisites

- Python 3.8+
- The dependencies listed in `requirements.txt`

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/Galaxi-Simulation.git
    cd Galaxi-Simulation
    ```

2.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Simulation

To run the simulation with the default parameters, simply execute the `run.py` script:

```bash
python run.py
```

The simulation will start, and you will see progress updates in the console. The output data and visualizations will be saved in the `data/` and `visualizations/` directories, respectively.

## Customization

You can customize the simulation by modifying the parameters in the `create_demo_simulation` function within `src/galaxy_simulation.py`. This includes changing the number of particles, the simulation time, and the physical models to include.

## Contributing

Contributions are welcome! If you have ideas for new features or improvements, please open an issue or submit a pull request.
