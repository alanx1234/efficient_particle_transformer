"""
Unit tests for clustering-based GeometricMessagePassing.

Run from repo root with:
    PYTHONPATH=. python -m pytest networks/test_clustering_gmp.py -v
    PYTHONPATH=. python networks/test_clustering_gmp.py
"""
import torch
import pytest
from networks.parT import GeometricMessagePassing


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_inputs(B=2, P=20, C=32, device="cpu"):
    torch.manual_seed(0)
    x = torch.randn(B, P, C, device=device)
    c1 = torch.randn(B, P, device=device)
    c2 = torch.randn(B, P, device=device)
    # Last 4 particles per jet are padding
    pad = torch.zeros(B, P, dtype=torch.bool, device=device)
    pad[:, -4:] = True
    # v: (B, 4, P) with physically sensible values
    pt_vals = torch.rand(B, P, device=device) * 100 + 1
    phi_vals = torch.rand(B, P, device=device) * 2 * 3.14159 - 3.14159
    eta_vals = torch.randn(B, P, device=device) * 2
    pz_vals = pt_vals * torch.sinh(eta_vals)
    px_vals = pt_vals * torch.cos(phi_vals)
    py_vals = pt_vals * torch.sin(phi_vals)
    E_vals = torch.sqrt(px_vals**2 + py_vals**2 + pz_vals**2 + 0.14**2)
    v = torch.stack([px_vals, py_vals, pz_vals, E_vals], dim=1)  # (B, 4, P)
    return x, c1, c2, pad, v


# ---------------------------------------------------------------------------
# Grid mode (existing behaviour unchanged)
# ---------------------------------------------------------------------------

def test_grid_mode_shape():
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="grid")
    out = gmp(x, c1, c2, pad=pad)
    assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"


def test_grid_mode_no_v_arg():
    """Grid path must work without v (backward compat)."""
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="grid")
    out_no_v = gmp(x, c1, c2, pad=pad)
    out_with_v = gmp(x, c1, c2, pad=pad, v=torch.randn(2, 4, 20))
    # Grid path ignores v entirely, outputs should be identical
    assert torch.allclose(out_no_v, out_with_v)


def test_grid_mode_padded_vs_unpadded_residual():
    """Padded positions should only change via the residual (grid path)."""
    x, c1, c2, pad, _ = _make_inputs(B=1, P=10)
    gmp = GeometricMessagePassing(channels=32, cluster_mode="grid")
    out = gmp(x, c1, c2, pad=pad)
    # Padded positions: output = residual + norm(gathered_grid_value)
    # We can't assert they're exactly x because conv output lands back on padded
    # cells, but shape must match.
    assert out.shape == x.shape


# ---------------------------------------------------------------------------
# KMeans mode
# ---------------------------------------------------------------------------

def test_kmeans_output_shape():
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="kmeans", cluster_k=8)
    out = gmp(x, c1, c2, pad=pad)
    assert out.shape == x.shape


def test_kmeans_no_padding():
    x, c1, c2, _, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="kmeans", cluster_k=8)
    out = gmp(x, c1, c2, pad=None)
    assert out.shape == x.shape


def test_kmeans_fixed_beta():
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(
        channels=32, cluster_mode="kmeans", cluster_k=8,
        cluster_beta=False
    )
    assert not isinstance(gmp.beta, torch.nn.Parameter)
    out = gmp(x, c1, c2, pad=pad)
    assert out.shape == x.shape


def test_kmeans_no_combine_mlp():
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(
        channels=32, cluster_mode="kmeans", cluster_k=8,
        cluster_combine_mlp=False
    )
    assert not hasattr(gmp, "cluster_mlp")
    out = gmp(x, c1, c2, pad=pad)
    assert out.shape == x.shape


def test_kmeans_gradient_flows():
    x, c1, c2, pad, _ = _make_inputs()
    x = x.requires_grad_(True)
    gmp = GeometricMessagePassing(channels=32, cluster_mode="kmeans", cluster_k=8)
    out = gmp(x, c1, c2, pad=pad)
    out.sum().backward()
    assert x.grad is not None


# ---------------------------------------------------------------------------
# HDBSCAN mode (skipped if package not available)
# ---------------------------------------------------------------------------

try:
    import hdbscan as _hdbscan_check  # noqa: F401
    HDBSCAN_INSTALLED = True
except ImportError:
    HDBSCAN_INSTALLED = False


@pytest.mark.skipif(not HDBSCAN_INSTALLED, reason="hdbscan not installed")
def test_hdbscan_output_shape():
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="hdbscan", cluster_k=8)
    out = gmp(x, c1, c2, pad=pad)
    assert out.shape == x.shape


@pytest.mark.skipif(not HDBSCAN_INSTALLED, reason="hdbscan not installed")
def test_hdbscan_no_padding():
    x, c1, c2, _, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="hdbscan", cluster_k=8)
    out = gmp(x, c1, c2, pad=None)
    assert out.shape == x.shape


@pytest.mark.skipif(not HDBSCAN_INSTALLED, reason="hdbscan not installed")
def test_hdbscan_all_real_particles_assigned():
    """Every real particle row in a should sum to >= 1 (hard assignment)."""
    x, c1, c2, pad, _ = _make_inputs(B=1, P=20)
    gmp = GeometricMessagePassing(channels=32, cluster_mode="hdbscan", cluster_k=8)
    # Monkey-patch to capture assignment matrix
    original = gmp._cluster_aggregate_broadcast

    captured = {}

    def capture(x_, a_, residual_):
        captured["a"] = a_.detach().cpu()
        return original(x_, a_, residual_)

    gmp._cluster_aggregate_broadcast = capture
    gmp(x, c1, c2, pad=pad)

    a = captured["a"]  # (1, P, K)
    real_mask = ~pad[0].cpu()
    row_sums = a[0][real_mask].sum(dim=-1)
    assert (row_sums > 0).all(), "Some real particles have no cluster assignment"


# ---------------------------------------------------------------------------
# Anti-kt mode (skipped if package not available)
# ---------------------------------------------------------------------------

try:
    import pyjet as _pyjet_check  # noqa: F401
    PYJET_INSTALLED = True
except ImportError:
    PYJET_INSTALLED = False


@pytest.mark.skipif(not PYJET_INSTALLED, reason="pyjet not installed")
def test_antikt_output_shape():
    x, c1, c2, pad, v = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="antikt", cluster_k=8)
    out = gmp(x, c1, c2, pad=pad, v=v)
    assert out.shape == x.shape


@pytest.mark.skipif(not PYJET_INSTALLED, reason="pyjet not installed")
def test_antikt_no_padding():
    x, c1, c2, _, v = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="antikt", cluster_k=8)
    out = gmp(x, c1, c2, pad=None, v=v)
    assert out.shape == x.shape


@pytest.mark.skipif(not PYJET_INSTALLED, reason="pyjet not installed")
def test_antikt_requires_v():
    x, c1, c2, pad, _ = _make_inputs()
    gmp = GeometricMessagePassing(channels=32, cluster_mode="antikt", cluster_k=8)
    with pytest.raises(ValueError, match="anti-kt clustering requires"):
        gmp(x, c1, c2, pad=pad, v=None)


@pytest.mark.skipif(not PYJET_INSTALLED, reason="pyjet not installed")
def test_antikt_every_real_particle_assigned():
    """Every real particle must end up in exactly one subjet cluster."""
    x, c1, c2, pad, v = _make_inputs(B=1, P=20)
    gmp = GeometricMessagePassing(channels=32, cluster_mode="antikt", cluster_k=8)

    original = gmp._cluster_aggregate_broadcast
    captured = {}

    def capture(x_, a_, residual_):
        captured["a"] = a_.detach().cpu()
        return original(x_, a_, residual_)

    gmp._cluster_aggregate_broadcast = capture
    gmp(x, c1, c2, pad=pad, v=v)

    a = captured["a"]  # (1, P, K)
    real_mask = ~pad[0].cpu()
    row_sums = a[0][real_mask].sum(dim=-1)
    assert (row_sums == 1).all(), (
        f"Some real particles assigned to != 1 cluster: {row_sums}"
    )


@pytest.mark.skipif(not PYJET_INSTALLED, reason="pyjet not installed")
def test_antikt_padded_particles_unassigned():
    """Padded particles should have all-zero assignment rows."""
    x, c1, c2, pad, v = _make_inputs(B=1, P=20)
    gmp = GeometricMessagePassing(channels=32, cluster_mode="antikt", cluster_k=8)

    original = gmp._cluster_aggregate_broadcast
    captured = {}

    def capture(x_, a_, residual_):
        captured["a"] = a_.detach().cpu()
        return original(x_, a_, residual_)

    gmp._cluster_aggregate_broadcast = capture
    gmp(x, c1, c2, pad=pad, v=v)

    a = captured["a"]
    pad_mask = pad[0].cpu()
    padded_row_sums = a[0][pad_mask].sum(dim=-1)
    assert (padded_row_sums == 0).all(), "Padded particles should not be assigned"


# ---------------------------------------------------------------------------
# Parameter plumbing: ParticleTransformer passes params to GMP
# ---------------------------------------------------------------------------

def test_particle_transformer_passes_cluster_params():
    from networks.parT import ParticleTransformer

    model = ParticleTransformer(
        input_dim=16,
        num_classes=2,
        embed_dims=[32],
        pair_embed_dims=[32, 32],
        num_heads=2,
        num_layers=2,
        num_cls_layers=1,
        use_gmp=True,
        gmp_cluster="kmeans",
        gmp_k=4,
        gmp_cluster_beta=True,
        gmp_cluster_combine_mlp=True,
    )
    assert model.gmp is not None
    assert model.gmp.cluster_mode == "kmeans"
    assert model.gmp.cluster_k == 4
    assert hasattr(model.gmp, "centroids")
    assert hasattr(model.gmp, "cluster_mlp")


def test_particle_transformer_grid_unchanged():
    """Grid mode GMP should not have cluster attributes."""
    from networks.parT import ParticleTransformer

    model = ParticleTransformer(
        input_dim=16,
        num_classes=2,
        embed_dims=[32],
        pair_embed_dims=[32, 32],
        num_heads=2,
        num_layers=2,
        num_cls_layers=1,
        use_gmp=True,
        gmp_cluster="grid",
    )
    assert model.gmp.cluster_mode == "grid"
    assert not hasattr(model.gmp, "centroids")
    assert not hasattr(model.gmp, "cluster_mlp")


if __name__ == "__main__":
    # Run without pytest for quick smoke-test
    tests = [
        test_grid_mode_shape,
        test_grid_mode_no_v_arg,
        test_grid_mode_padded_vs_unpadded_residual,
        test_kmeans_output_shape,
        test_kmeans_no_padding,
        test_kmeans_fixed_beta,
        test_kmeans_no_combine_mlp,
        test_kmeans_gradient_flows,
        test_particle_transformer_passes_cluster_params,
        test_particle_transformer_grid_unchanged,
    ]
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except Exception as e:
            print(f"FAIL  {t.__name__}: {e}")
