"""Convert data to target camera model."""

import argparse
import pprint
import shutil
import subprocess
import tempfile
import typing as T
from pathlib import Path

import cv2
import numpy as np
import tomli
from fisheye_model import unproject_fisheye
from utils import create_grid

SCRIPT_PATH = Path(__file__).parent.resolve()


def get_image_dimensions(camera: dict[str, T.Any]) -> tuple[int, int]:
    match camera.get("dimensions", dict()):
        case {"width": width, "height": height}:
            return width, height
        case _:
            raise ValueError(f"Incorrect specification of dimensions: {pprint.pformat(camera)}")


def get_camera_matrix(camera: dict[str, T.Any]) -> np.ndarray:
    match camera.get("camera_matrix", dict()):
        case {"fx": fx, "fy": fy, "cx": cx, "cy": cy}:
            return np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        case _:
            raise ValueError(f"Incorrect specification of camera matrix: {pprint.pformat(camera)}")


def create_fisheye_remap_table(camera: dict[str, T.Any]) -> np.ndarray:
    """Create remap table for the kannala-brant fisheye model."""
    width, height = get_image_dimensions(camera)
    K = get_camera_matrix(camera)

    match camera.get("distortion_coefficients", dict()):
        case {"k1": k1, "k2": k2, "k3": k3, "k4": k4}:
            coeffs = np.array([k1, k2, k3, k4])
        case _:
            raise ValueError(
                f"Incorrect specification of distortion parameters: {pprint.pformat(camera)}"
            )

    # construct table
    p_native = create_grid(width=width, height=height)
    return unproject_fisheye(p_native, K=K, coeffs=coeffs).reshape([height, width, 3])


def create_brown_conrady_remap_table(camera: dict[str, T.Any]) -> np.ndarray:
    """Create remap table for the OpenCV/brown-conrady model."""
    width, height = get_image_dimensions(camera)
    K = get_camera_matrix(camera)

    match camera.get("distortion_coefficients", dict()):
        case {"k1": k1, "k2": k2, "k3": k3, "p1": p1, "p2": p2}:
            coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float64)
        case _:
            raise ValueError(
                f"Incorrect specification of distortion parameters: {pprint.pformat(camera)}"
            )

    p_native = create_grid(width=width, height=height)
    p_undistorted = cv2.undistortPoints(p_native.astype(np.float64), K.astype(np.float64), coeffs)
    p_undistorted = np.squeeze(p_undistorted, axis=1)
    p_undistorted_unit_depth = np.concatenate(
        [p_undistorted, np.ones_like(p_undistorted[:, 1:])], axis=1
    )
    v_cam = p_undistorted_unit_depth / np.linalg.norm(
        p_undistorted_unit_depth, axis=1, keepdims=True
    )
    return v_cam.reshape([height, width, 3])


def create_equirectangular_remap_table(camera: dict[str, T.Any]) -> np.ndarray:
    width, height = get_image_dimensions(camera)
    p_native = create_grid(width=width, height=height)

    x_normalized = p_native[:, 0] / (width - 1)
    y_normalized = p_native[:, 1] / height

    theta = np.pi * y_normalized
    phi = 2 * np.pi * x_normalized

    # https://www.pbr-book.org/4ed/Cameras_and_Film/Spherical_Camera
    v_cam = np.stack(
        [
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta),
        ],
        axis=-1,
    )
    R = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, -1.0], [-1.0, 0.0, 0.0]])
    v_cam = R @ v_cam.reshape(-1, 3, 1)
    v_cam = v_cam.reshape(-1, 3)

    return v_cam.reshape([height, width, 3])


def create_remap_table(camera: dict[str, T.Any]) -> np.ndarray:
    """Create remap table for the provided camera."""
    if "model" not in camera:
        raise KeyError(f"Camera lacks a model specifier: {pprint.pformat(camera)}")

    model = camera["model"]
    if model == "fisheye":
        return create_fisheye_remap_table(camera)
    elif model == "brown-conrady":
        return create_brown_conrady_remap_table(camera)
    elif model == "equirectangular":
        return create_equirectangular_remap_table(camera)
    else:
        raise KeyError(f"Invalid camera model: {model}")


def main(args: argparse.Namespace):
    if args.bin is None:
        args.bin = SCRIPT_PATH.parent / "build" / "cubemap_converter.exe"

    input_path = Path(args.input).absolute()
    output_path = Path(args.output).absolute()

    if args.clean and output_path.exists():
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    # determine how many images there should be:
    gt_poses = np.genfromtxt(input_path / "ground_truth_imu_pose.csv", delimiter=",", skip_header=1)
    print(f"Dataset contains {len(gt_poses)} time-steps to process...")

    # load the config file
    with open(args.config, "rb") as handle:
        config = tomli.load(handle)

    cameras = config.get("cameras", [])
    if not len(cameras):
        print(f"Specified config [{args.config}] had no cameras in it:\n{pprint.pformat(config)}")
        exit(1)

    temp_dir = Path(tempfile.mkdtemp(prefix="cubemap_converter_"))

    for index, description in enumerate(cameras):
        remap_table = create_remap_table(camera=description)

        # Write out the table somewhere
        remap_table_path = temp_dir / f"camera_{index:02}.raw"
        remap_table.astype(np.float32).tofile(remap_table_path)

        # Create a command to convert
        command = [
            str(args.bin),
            "--input-path",
            str(input_path),
            "--output-path",
            str(output_path),
            "--width",
            str(remap_table.shape[1]),
            "--height",
            str(remap_table.shape[0]),
            "--camera-index",
            str(index),
            "--num-images",
            str(len(gt_poses)),
            "--remap-table",
            str(remap_table_path),
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.check_call(command)

    # Copy CSV and TOML files:
    for src_path in [*input_path.glob("*.csv"), *input_path.glob("*.toml")]:
        if src_path.name == "intrinsics.csv":
            continue
        dest_path = output_path / src_path.name
        print(f"Copying: {str(src_path)} -> {str(dest_path)}")
        shutil.copy(src_path, dest_path)

    if (input_path / "lidar").exists():
        shutil.copytree(input_path / "lidar", output_path / "lidar")

    # Copy the calibration
    dest_path = output_path / "intrinsics.toml"
    print(f"Copying: {args.config} -> {str(dest_path)}")
    shutil.copy(args.config, dest_path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-b", "--bin", type=str, help="Path to cubemap converter binary.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to configuration file."
    )
    parser.add_argument(
        "-i", "--input", type=str, default=None, required=True, help="Input directory."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        required=True,
        help="Output directory.",
    )
    parser.add_argument("--clean", action="store_true", help="Clean the destination if it exists.")
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
