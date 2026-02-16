import math
import torch
import torch.nn.functional as F

# Import ParT (adjust path/name if needed)
from networks.parT import ParticleTransformerTagger, unwrap_phi_per_jet


def make_physical_p4(N, P, device, mass=0.139, p_scale=1.0):
    """
    Returns v: (N,4,P) with [px,py,pz,E] and E = sqrt(p^2 + m^2).
    This guarantees E >= |p| and avoids NaNs in rapidity.
    """
    px = torch.randn(N, P, device=device) * p_scale
    py = torch.randn(N, P, device=device) * p_scale
    pz = torch.randn(N, P, device=device) * p_scale
    m2 = (mass * mass)
    E  = torch.sqrt(px*px + py*py + pz*pz + m2)
    v = torch.stack([px, py, pz, E], dim=1)  # (N,4,P)
    return v

def make_mask(N, P, min_real=10):
    """mask: (N,1,P) True=real, False=padded"""
    mask = torch.zeros(N, 1, P, dtype=torch.bool)
    for i in range(N):
        n_real = torch.randint(low=min_real, high=P + 1, size=(1,)).item()
        mask[i, 0, :n_real] = True
    return mask


@torch.no_grad()
def assert_all_finite(t, name="tensor"):
    if not torch.isfinite(t).all():
        bad = (~torch.isfinite(t)).sum().item()
        raise RuntimeError(f"{name} has {bad} non-finite entries (nan/inf).")


@torch.no_grad()
def perturb_only_padded(x, mask, noise_scale=1.0):
    """
    x: (N,C,P)
    mask: (N,1,P) True=real False=padded
    returns x' where ONLY padded entries are randomized.
    """
    x2 = x.clone()
    pad = (~mask).expand_as(x2)
    noise = torch.randn_like(x2) * noise_scale
    x2 = x2.masked_scatter(pad, noise[pad])
    return x2


@torch.no_grad()
def padding_invariance_test(model, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask, trials=5):
    """
    Run same batch, but scramble ONLY padded tokens.
    Output should barely change if padding is handled correctly.
    """
    model.eval()
    base = model(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask).detach()

    max_diffs = []
    mean_diffs = []
    for _ in range(trials):
        # perturb PF padded region
        pf_x2 = perturb_only_padded(pf_x, pf_mask, noise_scale=5.0)
        pf_v2 = perturb_only_padded(pf_v, pf_mask, noise_scale=5.0)
        out_pf = model(pf_x2, pf_v2, pf_mask, sv_x, sv_v, sv_mask).detach()

        # perturb SV padded region
        sv_x2 = perturb_only_padded(sv_x, sv_mask, noise_scale=5.0)
        sv_v2 = perturb_only_padded(sv_v, sv_mask, noise_scale=5.0)
        out_sv = model(pf_x, pf_v, pf_mask, sv_x2, sv_v2, sv_mask).detach()

        d_pf = (out_pf - base).abs()
        d_sv = (out_sv - base).abs()

        max_diffs.append(max(d_pf.max().item(), d_sv.max().item()))
        mean_diffs.append(0.5 * (d_pf.mean().item() + d_sv.mean().item()))

    print("\n[Padding invariance test]")
    print(f"  max |Δlogit| over trials: min={min(max_diffs):.4g}, median={sorted(max_diffs)[len(max_diffs)//2]:.4g}, max={max(max_diffs):.4g}")
    print(f"  mean|Δlogit| over trials: min={min(mean_diffs):.4g}, median={sorted(mean_diffs)[len(mean_diffs)//2]:.4g}, max={max(mean_diffs):.4g}")
    print("  (Goal: very close to 0. If these are large, padding is leaking.)")


@torch.no_grad()
def gmp_effect_test(model_gmp_off, model_gmp_on, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask):
    """
    Same batch, compare logits with GMP off vs on.
    They should differ (otherwise GMP path isn't affecting forward).
    """
    model_gmp_off.eval()
    model_gmp_on.eval()
    out_off = model_gmp_off(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask).detach()
    out_on  = model_gmp_on(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask).detach()

    diff = (out_on - out_off).abs()
    print("\n[GMP effect test]")
    print(f"  mean|Δlogit| = {diff.mean().item():.6g}")
    print(f"  max |Δlogit| = {diff.max().item():.6g}")
    print("  (Goal: not ~0. If ~0, GMP may be bypassed / disabled.)")


def overfit_tiny_batch_test(model, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask, num_classes, steps=30, lr=3e-3):
    """
    Train on the SAME tiny batch for a few steps.
    Loss should generally go down.
    """
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)

    target = torch.randint(0, num_classes, (pf_x.size(0),), device=pf_x.device)

    losses = []
    for s in range(steps):
        opt.zero_grad(set_to_none=True)
        out = model(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask)
        loss = F.cross_entropy(out, target)
        loss.backward()
        opt.step()
        losses.append(loss.detach().item())

    print("\n[Overfit tiny batch test]")
    print(f"  loss start: {losses[0]:.6g}")
    print(f"  loss end  : {losses[-1]:.6g}")
    print("  (Goal: end < start by a noticeable amount.)")


def build_part_tagger(
    *,
    pf_input_dim,
    sv_input_dim,
    num_classes,
    device,
    use_gmp,
    gmp_grid=0.05,
):
    # Keep it small for a smoke test
    return ParticleTransformerTagger(
        pf_input_dim=pf_input_dim,
        sv_input_dim=sv_input_dim,
        num_classes=num_classes,
        num_heads=4,
        num_layers=2,
        num_cls_layers=1,
        embed_dims=[32, 64, 32],
        pair_embed_dims=[16, 16, 16],
        fc_params=[(64, 0.0)],
        trim=False,
        for_inference=False,
        use_amp=False,

        # your added knobs in ParticleTransformer
        use_gmp=use_gmp,
        gmp_kernel=3,
        gmp_grid=gmp_grid,
        gmp_reduce="sum",
    ).to(device)


@torch.no_grad()
def phi_seam_grid_index_test(device="cpu", grid_size=0.05):
    """
    Demonstrate the seam problem for grid binning and why unwrap fixes it.

    Two particles:
      phi1 ~ +pi - eps
      phi2 ~ -pi + eps
    Physically neighbors, but numerically far apart unless you unwrap.
    """
    B, P = 1, 2
    eps = 0.01

    # same eta so only phi matters
    eta = torch.zeros(B, P, device=device)
    phi_raw = torch.tensor([[math.pi - eps, -math.pi + eps]], device=device)

    pad = torch.zeros(B, P, dtype=torch.bool, device=device)  # all real

    def grid_bins(phi_in, use_unwrap: bool):
        phi = phi_in
        if use_unwrap:
            phi = unwrap_phi_per_jet(phi, pad=pad)

        # mimic your GMP shift+bin logic (only in phi dimension)
        phi_min = phi.min(dim=1, keepdim=True).values
        phi_shift = phi - phi_min
        grid_phi = (phi_shift / grid_size).floor().to(torch.long)
        return grid_phi, phi, phi_shift

    grid_no, phi_no, shift_no = grid_bins(phi_raw, use_unwrap=False)
    grid_yes, phi_yes, shift_yes = grid_bins(phi_raw, use_unwrap=True)

    print("\n[Phi seam test: bin indices]")
    print(f"  grid_size = {grid_size}")
    print(f"  raw phi         = {phi_raw.cpu().numpy()}")
    print(f"  NO unwrap:")
    print(f"    phi used       = {phi_no.cpu().numpy()}")
    print(f"    phi shift      = {shift_no.cpu().numpy()}")
    print(f"    grid_phi bins  = {grid_no.cpu().numpy()}  (often far apart)")
    print(f"  WITH unwrap:")
    print(f"    phi used       = {phi_yes.cpu().numpy()}")
    print(f"    phi shift      = {shift_yes.cpu().numpy()}")
    print(f"    grid_phi bins  = {grid_yes.cpu().numpy()}  (should be same/nearby)")

    # a simple “pass” check: bins should be close after unwrap
    dist = (grid_yes[0, 0] - grid_yes[0, 1]).abs().item()
    print(f"  bin distance (unwrap) = {dist}")
    print("  (Goal: small, like 0 or 1. If huge, unwrap/bucketing logic is broken.)")
@torch.no_grad()
def random_stress_test(model_on, model_off, device, iters=100, check_every=10):
    torch.manual_seed(123)
    N = 4
    P_pf = 128
    P_sv = 32
    pf_input_dim = 17
    sv_input_dim = 17
    num_classes = 10

    for t in range(1, iters + 1):
        pf_x = torch.randn(N, pf_input_dim, P_pf, device=device)
        pf_v = make_physical_p4(N, P_pf, device=device, mass=0.139, p_scale=1.0)
        pf_mask = make_mask(N, P_pf, min_real=5).to(device)

        sv_x = torch.randn(N, sv_input_dim, P_sv, device=device)
        sv_v = make_physical_p4(N, P_sv, device=device, mass=0.139, p_scale=1.0)
        sv_mask = make_mask(N, P_sv, min_real=2).to(device)

        # zero padded tokens
        pf_x = pf_x.masked_fill(~pf_mask.expand_as(pf_x), 0.0)
        pf_v = pf_v.masked_fill(~pf_mask.expand_as(pf_v), 0.0)
        sv_x = sv_x.masked_fill(~sv_mask.expand_as(sv_x), 0.0)
        sv_v = sv_v.masked_fill(~sv_mask.expand_as(sv_v), 0.0)

        out = model_on(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask)
        if not torch.isfinite(out).all():
            raise RuntimeError(f"[stress] NaN/Inf at iter {t}")

        # occasionally: padding invariance + gmp effect
        if t % check_every == 0:
            padding_invariance_test(model_on, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask, trials=2)
            gmp_effect_test(model_off, model_on, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask)

        if t % 20 == 0:
            print(f"[stress] iter {t}/{iters}: OK")

    print(f"[stress] finished {iters} iters: ✅ no NaNs, padding stable")


@torch.no_grad()
def seam_adversarial_test(model_on, device, eps=1e-3):
    """
    Build a jet where many particles sit near +/-pi.
    This is the real seam stress case for your GMP grid.
    """
    N = 2
    P = 128
    C = 32  # must match embed_dim used inside your model; this test just hits v/phi logic though

    # Make physical p4 but force phi near +/-pi by setting px,py
    # phi = atan2(py, px)
    px = torch.ones(N, P, device=device)
    py = torch.zeros(N, P, device=device)
    # half +pi-eps, half -pi+eps achieved by flipping signs
    px[:, :P//2] = -1.0
    py[:, :P//2] =  eps
    px[:, P//2:] = -1.0
    py[:, P//2:] = -eps

    pz = torch.randn(N, P, device=device) * 0.5
    m2 = 0.139**2
    E  = torch.sqrt(px*px + py*py + pz*pz + m2)
    v = torch.stack([px, py, pz, E], dim=1)  # (N,4,P)

    # dummy x/mask (all real)
    pf_x = torch.randn(N, 17, P, device=device)
    pf_mask = torch.ones(N, 1, P, dtype=torch.bool, device=device)

    # no SV branch for simplicity: give minimal SV with all padded
    sv_x = torch.zeros(N, 17, 1, device=device)
    sv_v = make_physical_p4(N, 1, device=device)
    sv_mask = torch.zeros(N, 1, 1, dtype=torch.bool, device=device)

    out = model_on(pf_x, v, pf_mask, sv_x, sv_v, sv_mask)
    assert_all_finite(out, "seam adversarial out")
    print("[seam adversarial] ✅ forward finite")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    N = 2
    P_pf = 64
    P_sv = 16
    pf_input_dim = 17
    sv_input_dim = 17
    num_classes = 10

    model_on = build_part_tagger(
        pf_input_dim=pf_input_dim,
        sv_input_dim=sv_input_dim,
        num_classes=num_classes,
        device=device,
        use_gmp=True,
        gmp_grid=0.05,
    )
    model_off = build_part_tagger(
        pf_input_dim=pf_input_dim,
        sv_input_dim=sv_input_dim,
        num_classes=num_classes,
        device=device,
        use_gmp=False,
        gmp_grid=0.05,
    )

    # dummy inputs
    pf_x = torch.randn(N, pf_input_dim, P_pf, device=device)
    pf_v = make_physical_p4(N, P_pf, device=device, mass=0.139, p_scale=1.0)
    pf_mask = make_mask(N, P_pf).to(device)

    sv_v = make_physical_p4(N, P_sv, device=device, mass=0.139, p_scale=1.0)
    sv_x = torch.randn(N, sv_input_dim, P_sv, device=device)
    sv_mask = make_mask(N, P_sv, min_real=2).to(device)

    # make padded explicitly zero (good baseline)
    pf_x = pf_x.masked_fill(~pf_mask.expand_as(pf_x), 0.0)
    pf_v = pf_v.masked_fill(~pf_mask.expand_as(pf_v), 0.0)
    sv_x = sv_x.masked_fill(~sv_mask.expand_as(sv_x), 0.0)
    sv_v = sv_v.masked_fill(~sv_mask.expand_as(sv_v), 0.0)

    # basic forward + backward sanity
    model_on.train()
    out = model_on(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask)
    print("output shape:", tuple(out.shape))
    assert_all_finite(out, "out")

    target = torch.randint(0, num_classes, (N,), device=device)
    loss = F.cross_entropy(out, target)
    loss.backward()
    print("loss:", loss.detach().item())
    print("grad check: ✅ OK")

    # --- A) padding invariance ---
    padding_invariance_test(model_on, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask, trials=5)

    # --- B) GMP effect ---
    gmp_effect_test(model_off, model_on, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask)

    # --- C) overfit tiny batch ---
    overfit_tiny_batch_test(model_on, pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask, num_classes, steps=30, lr=3e-3)

    # --- D) explicit -pi/+pi seam demo for grid binning ---
    phi_seam_grid_index_test(device=device, grid_size=0.05)

    random_stress_test(model_on, model_off, device, iters=100, check_every=10)
    seam_adversarial_test(model_on, device)


if __name__ == "__main__":
    main()