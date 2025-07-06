import sys
import os

# Add the 'src' directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from galaxy_simulation import run_galaxy_simulation, analyze_supernova_timing, benchmark_simulation_performance

def main():
    """
    Main function to run the complete galaxy simulation and analyses.
    """
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

if __name__ == "__main__":
    main()
