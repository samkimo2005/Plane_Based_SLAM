
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path, PureWindowsPath

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


CAMERAS = ("ir_left", "ir_right", "rgb")

PAIRS = (
    ("left_to_right", "ir_right"),
    ("left_to_rgb", "rgb"),
)


def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8-sig")
    )


def choose(paths, label):
    paths = sorted(
        set(
            Path(p).resolve()
            for p in paths
            if Path(p).is_file()
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"Cannot find {label}. "
            f"Please provide its path explicitly."
        )

    if len(paths) == 1:
        return paths[0]

    for i, p in enumerate(paths, 1):
        print(f"{i}: {p}")

    index = int(input(f"Select {label}: "))
    return paths[index - 1]


def rms(a, b):
    delta = (
        np.asarray(a).reshape(-1, 2)
        - np.asarray(b).reshape(-1, 2)
    )

    return float(
        np.sqrt(
            np.mean(np.sum(delta * delta, axis=1))
        )
    )


def summarise(values):
    return {
        "overall_rms_px": float(
            np.sqrt(
                np.mean(
                    np.square(list(values.values()))
                )
            )
        ),
        "maximum_image_rms_px": float(
            max(values.values())
        ),
        "per_image_rms_px": values,
    }


def camera_models(intrinsics, factory):
    results = {
        "calibrated": {},
        "factory": {},
    }

    for camera in CAMERAS:
        c = intrinsics["cameras"][camera]

        results["calibrated"][camera] = (
            np.asarray(
                c["camera_matrix"],
                dtype=np.float64,
            ),
            np.asarray(
                c["distortion_k1_k2_p1_p2_k3"],
                dtype=np.float64,
            ),
        )

        f = factory["stream_intrinsics"][camera]

        model = str(
            f["distortion_model"]
        ).lower()

        D = np.asarray(
            f["distortion_coefficients"],
            dtype=np.float64,
        )

        # OpenCV projectPoints uses forward
        # Brown-Conrady distortion.
        #
        # The saved D435i inverse Brown-Conrady
        # coefficients are zero, so projection
        # with zero distortion is valid.
        #
        # Nonzero inverse coefficients require
        # a different projection implementation.

        if (
            "inverse_brown_conrady" in model
            and np.any(np.abs(D) > 1e-12)
        ):
            raise ValueError(
                f"{camera}: Nonzero inverse "
                "Brown-Conrady coefficients "
                "require a different projector."
            )

        if not any(
            x in model
            for x in ("brown_conrady", "none")
        ):
            raise ValueError(
                f"{camera}: Unsupported factory "
                f"distortion model: {model}"
            )

        K = np.array(
            [
                [f["fx"], 0, f["cx"]],
                [0, f["fy"], f["cy"]],
                [0, 0, 1],
            ],
            dtype=np.float64,
        )

        results["factory"][camera] = (K, D)

    return results


def detect_corners(session, board, factory):
    pattern = tuple(board["inner_corners"])

    count = factory["planned_sets"]["validation"]

    paths = {
        cam: sorted(
            (session / "validation" / cam).glob("*.png")
        )
        for cam in CAMERAS
    }

    names = [
        p.name for p in paths["ir_left"]
    ]

    if (
        len(names) != count
        or any(
            [p.name for p in paths[c]] != names
            for c in CAMERAS
        )
    ):
        raise ValueError(
            "Validation image count or filenames "
            "differ across camera streams."
        )

    corners = {
        cam: {} for cam in CAMERAS
    }

    for cam in CAMERAS:
        f = factory["stream_intrinsics"][cam]

        for path in paths[cam]:
            gray = cv.imdecode(
                np.fromfile(
                    str(path),
                    np.uint8,
                ),
                cv.IMREAD_GRAYSCALE,
            )

            if (
                gray is None
                or gray.shape[1::-1]
                != (f["width"], f["height"])
            ):
                raise ValueError(
                    f"Unreadable image or "
                    f"wrong image size: {path}"
                )

            found, pts = cv.findChessboardCornersSB(
                gray,
                pattern,
                flags=cv.CALIB_CB_ACCURACY,
            )

            if (
                not found
                or len(pts)
                != pattern[0] * pattern[1]
            ):
                raise ValueError(
                    f"Checkerboard detection failed: {path}"
                )

            corners[cam][path.name] = (
                pts.reshape(-1, 2).astype(np.float64)
            )

    # Check corner ordering across views.

    for name in names:
        a = corners["ir_left"][name].reshape(
            pattern[1],
            pattern[0],
            2,
        )

        for cam in ("ir_right", "rgb"):
            b = corners[cam][name].reshape(
                pattern[1],
                pattern[0],
                2,
            )

            for axis in (0, 1):
                va = np.diff(
                    a, axis=axis
                ).reshape(-1, 2).mean(axis=0)

                vb = np.diff(
                    b, axis=axis
                ).reshape(-1, 2).mean(axis=0)

                if np.dot(va, vb) <= 0:
                    raise ValueError(
                        f"Corner ordering differs: "
                        f"{name} / {cam}"
                    )

    return names, corners


def project(points, rvec, tvec, camera):
    K, D = camera

    return cv.projectPoints(
        points,
        rvec,
        tvec,
        K,
        D,
    )[0].reshape(-1, 2)


def pair_pose(rvec_left, tvec_left, pair):
    R_board, _ = cv.Rodrigues(rvec_left)

    R = np.asarray(
        pair["rotation_matrix"],
        dtype=np.float64,
    )

    t = np.asarray(
        pair["translation_m"],
        dtype=np.float64,
    ).reshape(3, 1)

    convention = pair.get(
        "convention",
        "p_target = R @ p_source + t",
    )

    if convention != (
        "p_target = R @ p_source + t"
    ):
        raise ValueError(
            "Unexpected extrinsics "
            "coordinate convention."
        )

    rvec_target, _ = cv.Rodrigues(
        R @ R_board
    )

    tvec_target = (
        R @ tvec_left + t
    )

    return rvec_target, tvec_target


def evaluate(
    points,
    names,
    corners,
    models,
    pair,
    target,
):
    source = "ir_left"

    per_source = {}
    per_transfer = {}
    per_joint = {}
    per_joint_left = {}
    per_joint_target = {}

    for name in names:
        obs_left = corners[source][name]
        obs_target = corners[target][name]

        # Estimate checkerboard pose using
        # the source camera's own intrinsics.

        ok, rv0, tv0 = cv.solvePnP(
            points,
            obs_left,
            *models[source],
        )

        if not ok:
            raise RuntimeError(
                f"solvePnP failed: {name}"
            )

        per_source[name] = rms(
            project(
                points,
                rv0,
                tv0,
                models[source],
            ),
            obs_left,
        )

        # Directional transfer validation.
        # Camera parameters stay fixed.

        rv_target, tv_target = pair_pose(
            rv0,
            tv0,
            pair,
        )

        per_transfer[name] = rms(
            project(
                points,
                rv_target,
                tv_target,
                models[target],
            ),
            obs_target,
        )

        # Joint validation:
        # optimise ONLY checkerboard pose.
        # Both camera models and the
        # inter-camera transformation stay fixed.

        def projections(x):
            rv = x[:3].reshape(3, 1)
            tv = x[3:].reshape(3, 1)

            rv2, tv2 = pair_pose(
                rv,
                tv,
                pair,
            )

            predicted_left = project(
                points,
                rv,
                tv,
                models[source],
            )

            predicted_target = project(
                points,
                rv2,
                tv2,
                models[target],
            )

            return (
                predicted_left,
                predicted_target,
            )

        def residual(x):
            left, right = projections(x)

            return np.r_[
                (left - obs_left).ravel(),
                (right - obs_target).ravel(),
            ]

        x0 = np.r_[
            rv0.ravel(),
            tv0.ravel(),
        ]

        fit = least_squares(
            residual,
            x0,
            method="lm",
            max_nfev=300,
            ftol=1e-11,
            xtol=1e-11,
            gtol=1e-11,
        )

        if not fit.success:
            raise RuntimeError(
                f"Joint optimisation failed: "
                f"{target}/{name}: {fit.message}"
            )

        left, right = projections(fit.x)

        e1 = rms(left, obs_left)
        e2 = rms(right, obs_target)

        per_joint_left[name] = e1
        per_joint_target[name] = e2

        # Both views have the same corner count.

        per_joint[name] = float(
            np.sqrt(
                (e1 ** 2 + e2 ** 2) / 2
            )
        )

    return {
        "left_ir_pnp": summarise(per_source),

        "transfer": summarise(per_transfer),

        "joint": {
            **summarise(per_joint),

            "per_image_source_rms_px":
                per_joint_left,

            "per_image_target_rms_px":
                per_joint_target,

            "success_count": len(names),

            "method":
                "fixed_intrinsics_fixed_extrinsics_"
                "joint_board_pose_only",
        },
    }


def main():
    parser = argparse.ArgumentParser()

    for key in (
        "session",
        "intrinsics",
        "extrinsics",
        "out",
    ):
        parser.add_argument(
            "--" + key,
            type=Path,
        )

    args = parser.parse_args()

    script = Path(__file__).resolve()

    roots = list(
        dict.fromkeys(
            (
                Path.cwd().resolve(),
                script.parent,
                *script.parents,
            )
        )
    )

    project = next(
        (
            p for p in roots
            if (
                p / "data" / "camera_calibration"
            ).is_dir()
        ),
        None,
    )

    if project is None and args.session is None:
        raise FileNotFoundError(
            "Run from the PlaneBasedSLAM "
            "project root or specify --session."
        )

    if args.session:
        session = args.session.resolve()

        if project is None:
            project = session.parent.parent.parent

    else:
        factory_path = choose(
            (
                project
                / "data"
                / "camera_calibration"
            ).glob(
                "session_*/factory_parameters.json"
            ),
            "session",
        )

        session = factory_path.parent

    results_dir = (
        project
        / "experiments"
        / "calibration"
        / "results"
    )

    if args.extrinsics:
        extr_path = args.extrinsics.resolve()
    else:
        extr_path = choose(
            results_dir.glob("*/extrinsics.json"),
            "extrinsics",
        )

    extrinsics = load_json(extr_path)

    recorded = PureWindowsPath(
        extrinsics["intrinsics_file"]
    )

    if args.intrinsics:
        intr_path = args.intrinsics.resolve()
    else:
        intr_path = (
            results_dir
            / recorded.parent.name
            / recorded.name
        )

    if not intr_path.is_file():
        raise FileNotFoundError(
            f"Expected matching intrinsics: {intr_path}. "
            "Supply --intrinsics explicitly."
        )

    intrinsics = load_json(intr_path)

    factory = load_json(
        session / "factory_parameters.json"
    )

    # Verify that all files belong
    # to the same calibration experiment.

    if (
        intrinsics["board"] != factory["board"]
        or extrinsics["board"] != factory["board"]
    ):
        raise ValueError(
            "Checkerboard settings differ."
        )

    if (
        intr_path.parent.name,
        intr_path.name,
    ) != (
        recorded.parent.name,
        recorded.name,
    ):
        raise ValueError(
            "Chosen intrinsics do not match "
            "the extrinsics training source."
        )

    if Path(
        extrinsics["session"].replace("\\", "/")
    ).name != session.name:
        raise ValueError(
            "Extrinsics belong to "
            "a different session."
        )

    if not extrinsics.get("intrinsics_fixed"):
        raise ValueError(
            "Expected fixed-intrinsics "
            "stereo training."
        )

    for pair_name, target in PAIRS:
        pair = extrinsics["pairs"][pair_name]

        if (
            pair["source"] != "ir_left"
            or pair["target"] != target
        ):
            raise ValueError(
                f"Unexpected calibrated "
                f"camera pair: {pair_name}"
            )

    for cam in CAMERAS:
        f = factory["stream_intrinsics"][cam]

        if (
            tuple(
                intrinsics["cameras"][cam][
                    "image_size_wh"
                ]
            )
            != (f["width"], f["height"])
        ):
            raise ValueError(
                f"Image size mismatch: {cam}"
            )

    pattern = tuple(
        factory["board"]["inner_corners"]
    )

    square = float(
        factory["board"]["square_size_m"]
    )

    points = np.zeros(
        (
            pattern[0] * pattern[1],
            3,
        ),
        dtype=np.float64,
    )

    points[:, :2] = (
        np.mgrid[
            0:pattern[0],
            0:pattern[1],
        ].T.reshape(-1, 2) * square
    )

    names, corners = detect_corners(
        session,
        factory["board"],
        factory,
    )

    models = camera_models(
        intrinsics,
        factory,
    )

    results = {
        "session": str(session),

        "intrinsics_file": str(intr_path),

        "extrinsics_file": str(extr_path),

        "validation_image_count": len(names),

        "comparison": {},

        "note":
            "Factory intrinsics + factory extrinsics "
            "versus calibrated intrinsics + "
            "calibrated extrinsics. Joint optimisation "
            "changes checkerboard poses only.",
    }

    # Use the correct intrinsic model
    # for EACH extrinsic model.

    for label in ("calibrated", "factory"):
        results["comparison"][label] = {
            "pairs": {}
        }

        for pair_name, target in PAIRS:
            if label == "calibrated":
                pair = extrinsics["pairs"][pair_name]
            else:
                pair = factory["extrinsics"][pair_name]

            # Target is supplied explicitly.
            # Factory JSON does not require
            # a "target" metadata field.

            results["comparison"][label][
                "pairs"
            ][pair_name] = evaluate(
                points,
                names,
                corners,
                models[label],
                pair,
                target,
            )

    # Display results.

    for label, report in results[
        "comparison"
    ].items():
        print(
            f"\n{label.upper()} intrinsics "
            f"+ {label.upper()} extrinsics"
        )

        for pair_name, data in report[
            "pairs"
        ].items():
            transfer = data[
                "transfer"
            ]["overall_rms_px"]

            joint = data[
                "joint"
            ]["overall_rms_px"]

            print(
                f"{pair_name}: "
                f"Transfer {transfer:.6f} px; "
                f"Joint {joint:.6f} px"
            )

    # Save results without overwriting
    # previous calibration runs.

    if args.out:
        out = args.out.resolve()
    else:
        out = (
            results_dir
            / (
                "factory_comparison_"
                + datetime.now().strftime(
                    "%Y%m%d_%H%M%S_%f"
                )
            )
        )

    if out.exists() and any(out.iterdir()):
        raise ValueError(
            f"Output directory is not empty: {out}"
        )

    out.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        out / "factory_comparison.json"
    ).write_text(
        json.dumps(
            results,
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )

    # Save per-frame Left IR-RGB comparison.

    csv_path = (
        out / "left_to_rgb_factory_comparison.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                "frame_id",
                "calibrated_transfer_px",
                "factory_transfer_px",
                "calibrated_joint_px",
                "factory_joint_px",
            ]
        )

        for name in names:
            cal = results["comparison"][
                "calibrated"
            ]["pairs"]["left_to_rgb"]

            fac = results["comparison"][
                "factory"
            ]["pairs"]["left_to_rgb"]

            writer.writerow(
                [
                    Path(name).stem,

                    cal["transfer"][
                        "per_image_rms_px"
                    ][name],

                    fac["transfer"][
                        "per_image_rms_px"
                    ][name],

                    cal["joint"][
                        "per_image_rms_px"
                    ][name],

                    fac["joint"][
                        "per_image_rms_px"
                    ][name],
                ]
            )

    print(
        f"\nSaved results to: {out.resolve()}"
    )


if __name__ == "__main__":
    main()
