from pathlib import Path
from pyngs.core import NGSpiceInstance


def measure_fcut_for_list(netlist_path: Path, rc_list: list[tuple[float, float]]):
    """Run AC analysis for a list of (R, C) values and print the cutoff frequency."""
    # Create ngspice instance
    inst = NGSpiceInstance()

    # Load the netlist once
    inst.load(netlist_path)

    # First run just to check everything is OK
    inst.run()
    print("Available parameters:", inst.list_parameters())
    print("Available measures:", inst.list_measures())
    print()

    results = []

    # Loop over (R, C) pairs
    for R_value, C_value in rc_list:
        print(f"Running for R={R_value} Ω, C={C_value} F...")

        # Set parameters defined in the netlist (.param Rval, .param Cval)
        inst.set_parameter("Rval", R_value)
        inst.set_parameter("Cval", C_value)

        # Run simulation with new parameters
        inst.run()

        # Get cutoff frequency from .meas fcut
        fcut = inst.get_measure("fcut")
        print(f"  -> fcut = {fcut} Hz\n")

        results.append((R_value, C_value, fcut))

    # Stop ngspice instance
    inst.stop()

    return results


if __name__ == "__main__":
    # Path to your RC netlist (must contain .param Rval, .param Cval and .meas fcut)
    netlist_path = Path("rc.cir")

    # Example list of (R, C) values in SI units
    rc_values = [
        (1e3, 1e-6),   # 1 kΩ, 1 µF
        (1e4, 1e-6),   # 10 kΩ, 1 µF
        (1e3, 1e-7),   # 1 kΩ, 0.1 µF
        (5e3, 2e-6),   # 5 kΩ, 2 µF
    ]

    results = measure_fcut_for_list(netlist_path, rc_values)

    print("Summary:")
    for R, C, fcut in results:
        print(f"R={R} Ω, C={C} F -> fcut={fcut} Hz")

