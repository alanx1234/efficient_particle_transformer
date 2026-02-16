import torch

from EfficientParticleTransformer import EfficientParticleTransformer

def make_mask(N, P, min_real=10):
    """
    Returns mask shaped (N, 1, P) with 1 = real, 0 = padded
    """
    mask = torch.zeros(N, 1, P, dtype=torch.bool)
    for i in range(N):
        n_real = torch.randint(low=min_real, high=P + 1, size=(1,)).item()
        mask[i, 0, :n_real] = True
    return mask

@torch.no_grad()
def assert_padded_outputs_finite(output, name="output"):
    if not torch.isfinite(output).all():
        bad = (~torch.isfinite(output)).sum().item()
        raise RuntimeError(f"{name} has {bad} non-finite entries (nan/inf).")

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)

    # ---- dummy shapes (match your expected inputs) ----
    N = 2          # batch size
    P_pf = 64      # PF max particles
    P_sv = 16      # SV max particles
    pf_input_dim = 17
    sv_input_dim = 17
    num_classes = 10

    # ---- build model ----
    model = EfficientParticleTransformerTagger(
        pf_input_dim=pf_input_dim,
        sv_input_dim=sv_input_dim,
        num_classes=num_classes,
        num_heads=8,
        num_layers=2,        # small for smoke test
        num_cls_layers=1,
        embed_dims=[32, 64, 32],
        fc_params=[(64, 0.0)],
        trim=False,          # keep padding behavior visible
        use_amp=False,
        use_gmp=True,
        gmp_kernel=3,
        gmp_grid=0.05,
    ).to(device)

    model.train()

    # ---- dummy inputs ----
    # pf_x: (N, C, P_pf), pf_v: (N, 4, P_pf), pf_mask: (N, 1, P_pf)
    pf_x = torch.randn(N, pf_input_dim, P_pf, device=device)
    pf_v = torch.randn(N, 4, P_pf, device=device)
    pf_mask = make_mask(N, P_pf).to(device)

    sv_x = torch.randn(N, sv_input_dim, P_sv, device=device)
    sv_v = torch.randn(N, 4, P_sv, device=device)
    sv_mask = make_mask(N, P_sv, min_real=2).to(device)

    # Force padded tokens to be obviously "fake" to catch leakage
    pf_x = pf_x.masked_fill(~pf_mask.expand_as(pf_x), 0.0)
    pf_v = pf_v.masked_fill(~pf_mask.expand_as(pf_v), 0.0)
    sv_x = sv_x.masked_fill(~sv_mask.expand_as(sv_x), 0.0)
    sv_v = sv_v.masked_fill(~sv_mask.expand_as(sv_v), 0.0)

    # ---- forward ----
    out = model(pf_x, pf_v, pf_mask, sv_x, sv_v, sv_mask)
    print("output shape:", tuple(out.shape))  # expect (N, num_classes)

    assert_padded_outputs_finite(out, "out")

    # ---- backward test ----
    target = torch.randint(0, num_classes, (N,), device=device)
    loss = torch.nn.functional.cross_entropy(out, target)
    loss.backward()

    # quick grad sanity
    grads_ok = True
    for n, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            print("⚠️ no grad:", n)
            grads_ok = False
            break
        if p.grad is not None and not torch.isfinite(p.grad).all():
            print("❌ non-finite grad:", n)
            grads_ok = False
            break

    print("loss:", float(loss))
    print("grad check:", "✅ OK" if grads_ok else "❌ issue")

if __name__ == "__main__":
    main()