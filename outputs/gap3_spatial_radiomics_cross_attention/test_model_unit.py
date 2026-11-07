import sys
import os
import torch

sys.path.append(os.path.abspath("."))

from outputs.gap3_spatial_radiomics_cross_attention.model import SpatialRadiomicsSwinCoralModel
from src.models.coral import coral_loss

def run_unit_tests():
    print("=" * 60)
    print("  RUNNING UNIT TESTS FOR SPATIAL RADIOMICS CROSS-ATTENTION MODEL")
    print("=" * 60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"[1] Testing device placement: {device}")
    
    model = SpatialRadiomicsSwinCoralModel(
        num_classes=5,
        d_vis=768,
        d_rad=32,
        d_model=768,
        num_heads=8,
        dropout=0.1,
        pretrained_backbone=False
    ).to(device)
    
    B = 4
    dummy_imgs = torch.randn(B, 3, 224, 224, device=device)
    dummy_rads = torch.randn(B, 9, 32, device=device)
    dummy_labels = torch.tensor([0, 1, 3, 4], dtype=torch.long, device=device)
    
    print("\n[2] Testing Forward Pass...")
    logits = model(dummy_imgs, dummy_rads)
    print(f"  Logits shape: {logits.shape} (Expected: ({B}, 4))")
    assert logits.shape == (B, 4), f"Expected ({B}, 4), got {logits.shape}"
    
    print("\n[3] Testing Forward with Attention...")
    logits_att, attn = model.forward_with_attention(dummy_imgs, dummy_rads)
    print(f"  Attention weights shape: {attn.shape} (Expected: ({B}, 49, 9))")
    assert attn.shape == (B, 49, 9), f"Expected ({B}, 49, 9), got {attn.shape}"
    model.eval()
    with torch.no_grad():
        _, attn_eval = model.forward_with_attention(dummy_imgs, dummy_rads)
    attn_sums = attn_eval.sum(dim=-1)
    print(f"  Attention sum per query (eval mode): {attn_sums[0, 0].item():.4f} (Expected: 1.0000)")
    assert torch.allclose(attn_sums, torch.ones_like(attn_sums), atol=1e-4)
    model.train()
    
    print("\n[4] Testing Backward Pass and Gradient Flow...")
    loss = coral_loss(logits, dummy_labels, num_classes=5)
    print(f"  CORAL Loss value: {loss.item():.4f}")
    loss.backward()
    
    grad_proj_q = model.cross_attention.proj_q.weight.grad
    grad_proj_k = model.cross_attention.proj_k.weight.grad
    grad_proj_v = model.cross_attention.proj_v.weight.grad
    grad_mha = model.cross_attention.mha.in_proj_weight.grad if model.cross_attention.mha.in_proj_weight is not None else None
    grad_coral = model.coral_head.weight.grad
    
    print(f"  grad proj_q: {'OK' if grad_proj_q is not None and torch.norm(grad_proj_q) > 0 else 'MISSING'}")
    print(f"  grad proj_k: {'OK' if grad_proj_k is not None and torch.norm(grad_proj_k) > 0 else 'MISSING'}")
    print(f"  grad proj_v: {'OK' if grad_proj_v is not None and torch.norm(grad_proj_v) > 0 else 'MISSING'}")
    print(f"  grad coral_head: {'OK' if grad_coral is not None and torch.norm(grad_coral) > 0 else 'MISSING'}")
    
    assert grad_proj_q is not None and torch.norm(grad_proj_q) > 0
    assert grad_proj_k is not None and torch.norm(grad_proj_k) > 0
    assert grad_proj_v is not None and torch.norm(grad_proj_v) > 0
    assert grad_coral is not None and torch.norm(grad_coral) > 0
    
    print("\n[5] Testing Inference Predict Pipeline...")
    model.eval()
    with torch.no_grad():
        out = model.predict(dummy_imgs, dummy_rads)
    print(f"  Predicted grades: {out['predicted_grade'].cpu().tolist()}")
    print(f"  Class probabilities shape: {out['class_probabilities'].shape}")
    print(f"  Rank probabilities shape: {out['rank_probabilities'].shape}")
    assert out['predicted_grade'].shape == (B,)
    assert out['class_probabilities'].shape == (B, 5)
    assert out['rank_probabilities'].shape == (B, 4)
    
    print("\n" + "=" * 60)
    print("  ALL UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 60)

if __name__ == "__main__":
    run_unit_tests()
