import argparse
import numpy as np

from physics import rk4


def main():
    parser = argparse.ArgumentParser(description="Asteroid simulation runner")
    parser.add_argument(
        "--mode",
        choices=["kepler", "gravity", "dataset"],
        default="kepler",
        help="Which calculation to perform",
    )
    parser.add_argument("--samples", type=int, default=500, help="Number of samples when generating a dataset")
    args = parser.parse_args()

    if args.mode == "kepler":
        rk4.run_kepler_simulation()
    elif args.mode == "gravity":
        rk4.run_gravity_simulation()
    elif args.mode == "dataset":
        data = rk4.generate_kepler_dataset(args.samples)
        print("Dataset shape:", data.shape)
        # optionally save
        np.savetxt("dataset.csv", data, delimiter=",")
        print("Written dataset.csv")


if __name__ == "__main__":
    main()
