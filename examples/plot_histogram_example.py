"""Render a histogram for a fixed probability distribution."""

from qiskit.exceptions import MissingOptionalLibraryError
from qiskit.visualization import plot_histogram


def main() -> None:
    probs = {"00": 0.48, "11": 0.52}

    try:
        plot_histogram(probs, title="Bell distribution")
    except (MissingOptionalLibraryError, ImportError) as exc:
        print("Matplotlib not installed:", exc)


if __name__ == "__main__":
    main()
