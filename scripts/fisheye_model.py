"""Kannala-Brandt fisheye camera model."""
import numpy as np
import typing as T

from utils import create_grid


def fisheye_distortion(
    theta: np.ndarray, coeffs: np.ndarray
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Evaluate the Kannala-Brandt fisheye distortion curve (theta -> radius)."""
    assert (
        len(theta.shape) == 2 and theta.shape[-1] == 1
    ), f"theta should be n x 1, got: {theta.shape}"
    assert coeffs.shape == (4,), f"coeffs should be 4-elements, got: {coeffs.shape}"

    poly_terms_even = np.concatenate(
        [
            np.ones_like(theta),
            np.power(theta, 2),
            np.power(theta, 4),
            np.power(theta, 6),
            np.power(theta, 8),
        ],
        axis=-1,
    )

    # broadcast multiply over columns:
    poly_terms_odd = poly_terms_even * theta

    # Compute image plane radius:
    coeffs_extended = np.concatenate([[1.0], coeffs])
    r = poly_terms_odd @ coeffs_extended.reshape([5, 1])

    # Compute derivative wrt theta:
    r_D_theta = poly_terms_even @ (
        coeffs_extended * np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    ).reshape([5, 1])

    return r, r_D_theta


def fisheye_invert_distortion(
    r: np.ndarray, coeffs: np.ndarray, max_iters: int = 10
) -> T.Tuple[np.ndarray, np.ndarray]:
    """Invert fisheye_distortion_poly with gauss newton solver."""
    assert max_iters > 0, f"max_iters = {max_iters}"
    assert len(r.shape) == 2 and r.shape[-1] == 1, f"r should be n x 1, got: {r.shape}"
    assert coeffs.shape == (4,), f"coeffs should be 4-elements, got: {coeffs.shape}"

    theta = np.copy(r)
    for _ in range(0, max_iters):
        r_predicted, r_D_theta = fisheye_distortion(theta=theta, coeffs=coeffs)
        error = r_predicted - r
        theta -= error / r_D_theta

    return theta, error


def get_fisheye_coefficients() -> T.Tuple[np.ndarray, T.List[float]]:
    """Some representative testing coefficients."""
    coefficients = np.array(
        [
            # Stereograhic
            [0.08327424, 0.00852979, 0.00063325, 0.00017048],
            # Equidistant
            [
                0.0,
                0.0,
                0.0,
                0.0,
            ],
            # Equisolid
            [-4.16666666e-02, 5.20833056e-04, -3.09988000e-06, 1.06158708e-08],
            # Orthogonal
            [-1.66666587e-01, 8.33305775e-03, -1.98095183e-04, 2.60641004e-06],
        ]
    )

    # Maximum image plane radius of each of the models above:
    max_radius = [2.0, np.pi / 2, np.sqrt(2.0), 1.0]
    return coefficients, max_radius


def test_fisheye_distortion_curve():
    """Test radial distortion curve."""
    coefficients, max_radius = get_fisheye_coefficients()
    for coeffs, max_r in zip(coefficients, max_radius):
        r = np.linspace(0.0, max_r, 100).reshape([-1, 1])
        # radii -> theta
        theta, error = fisheye_invert_distortion(r=r, coeffs=coeffs)
        # theta -> radii
        r_recomputed, r_D_theta = fisheye_distortion(theta=theta, coeffs=coeffs)

        # should have zero round-trip error
        np.testing.assert_allclose(error, np.zeros_like(error), rtol=0.0, atol=1.0e-6)
        np.testing.assert_allclose(r_recomputed, r, rtol=0.0, atol=1.0e-6)

        # check derivative:
        dx = 0.001
        r_D_theta_numerical = (
            fisheye_distortion(theta=theta + dx, coeffs=coeffs)[0]
            - fisheye_distortion(theta=theta - dx, coeffs=coeffs)[0]
        ) / (dx * 2)
        np.testing.assert_allclose(
            r_D_theta_numerical, r_D_theta, rtol=0.0, atol=1.0e-6
        )


def project_fisheye(p_cam: np.ndarray, K: np.ndarray, coeffs: np.ndarray):
    """Forward distortion model for fisheye."""
    _, xyz = p_cam.shape
    assert xyz == 3, f"p_cam should be N x 3, got: {p_cam.shape}"
    assert K.shape == (3, 3), f"K should be 3x3, got: {K.shape}"
    assert coeffs.shape == (4,), f"coeffs should be 4-elements, got: {coeffs.shape}"

    # Angle wrt the optical axis:
    theta = np.arccos(p_cam[:, 2:])

    # Angle in the image plane:
    phi = np.arctan2(p_cam[:, 1:2], p_cam[:, 0:1])

    # Distort theta:
    r, _ = fisheye_distortion(theta, coeffs)

    # Compute image plane location:
    p_img = np.concatenate([np.cos(phi), np.sin(phi)], axis=-1) * r
    p_img_homogenous = np.concatenate([p_img, np.ones_like(phi)], axis=-1)

    # Multiply by camera matrix, then convert back to (N, 3) shape.
    p_native = (K @ p_img_homogenous.transpose()).transpose()
    return p_native[:, (0, 1)]


def unproject_fisheye(
    p_native: np.ndarray, K: np.ndarray, coeffs: np.ndarray
) -> np.ndarray:
    """Inverse distortion model for fisheye."""
    assert (
        len(p_native.shape) == 2 and p_native.shape[-1] == 2
    ), f"p_native should be N x 2, got: {p_native.shape}"
    assert K.shape == (3, 3), f"K should be 3x3, got: {K.shape}"

    # Convert pixel coordinates to the image plane:
    p_native_homogenous = np.concatenate(
        [p_native, np.ones_like(p_native[:, -1:])], axis=-1
    )
    p_img_homogenous = (np.linalg.inv(K) @ p_native_homogenous.transpose()).transpose()

    # Compute radius and angle:
    r = np.linalg.norm(p_img_homogenous[:, 0:2], axis=1, keepdims=True)
    phi = np.arctan2(p_img_homogenous[:, 1:2], p_img_homogenous[:, 0:1])

    # Undistort to compute theta via newton's method:
    theta = np.copy(r)
    for _ in range(0, 10):
        r_predicted, r_D_theta = fisheye_distortion(theta=theta, coeffs=coeffs)
        error = r_predicted - r
        theta -= error / r_D_theta

    # TODO: Check the error here.
    # Compute unit vector
    return np.concatenate(
        [np.cos(phi) * np.sin(theta), np.sin(phi) * np.sin(theta), np.cos(theta)],
        axis=-1,
    )


def test_fisheye_model():
    """Test that the forward and reverse fisheye model agree."""
    coefficients, max_radius = get_fisheye_coefficients()

    # Image dims (width by height)
    image_dims = np.array([320, 240])
    principal_point = (image_dims - 1.0) / 2.0

    # Create grid of pixel coordinates
    pixel_coords = create_grid(*image_dims)

    # Test all the models:
    for coeffs, max_r in zip(coefficients, max_radius):
        # Pick a suitable focal length that satisfies our maximum radius.
        f = np.linalg.norm(image_dims) * 0.5 / max_r * 1.05
        K = np.array(
            [
                [f, 0.0, principal_point[0]],
                [0.0, f, principal_point[1]],
                [0.0, 0.0, 1.0],
            ]
        )

        p_cam = unproject_fisheye(p_native=pixel_coords, K=K, coeffs=coeffs)
        p_native = project_fisheye(p_cam=p_cam, K=K, coeffs=coeffs)
        np.testing.assert_allclose(p_native, pixel_coords, rtol=0.0, atol=1.0e-6)
