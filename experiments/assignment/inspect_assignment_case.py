from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum"
)

REFERENCE_DIR = (
    PROJECT_ROOT
    / "data"
    / "assignment"
    / "tum_reference"
)


def describe_npz(path):
    print()
    print("=" * 70)
    print(path)
    print("=" * 70)

    with np.load(
        path,
        allow_pickle=True,
    ) as data:

        for key in data.files:

            value = data[key]

            print(
                f"{key:<35} "
                f"shape={value.shape!s:<18} "
                f"dtype={value.dtype}"
            )

            if value.ndim == 0:
                try:
                    print(
                        f"    value = {value.item()}"
                    )
                except Exception:
                    pass


def main():

    cases = sorted(
        DATA_DIR.glob("tum_*.npz")
    )

    if not cases:
        raise FileNotFoundError(
            f"No TUM Assignment cases found in:\n"
            f"{DATA_DIR}"
        )

    case_file = cases[0]

    describe_npz(
        case_file
    )

    # Try to infer case_id from filename.
    case_id = case_file.stem

    reference_file = (
        REFERENCE_DIR
        / f"{case_id}_gt.npz"
    )

    if reference_file.exists():

        describe_npz(
            reference_file
        )

    else:

        print()
        print(
            "Reference sidecar not found at:"
        )

        print(
            reference_file
        )


if __name__ == "__main__":
    main()