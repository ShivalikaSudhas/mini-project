import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
import torch
import torch.nn.functional as F
import cv2
from torchvision import transforms
from visual_model import VisualForensicModel, get_default_visual_model
from preprocessing import load_split_dataloaders

def compute_error_level_metric(image_path):
    # Simple baseline: variance of Laplacian (focus) + file size in bytes
    img = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    size = os.path.getsize(image_path)
    return lap_var + size * 1e-6  # scale size down

def baseline_error_level(test_loader):
    rows = []
    scores = []
    labels = []
    for batch in test_loader:
        paths = batch["image_path"]
        labs = batch["label"].numpy()
        for p, l in zip(paths, labs):
            metric = compute_error_level_metric(p)
            if metric is None:
                continue
            rows.append({"image_path": p, "label": int(l), "metric": metric})
            scores.append(metric)
            labels.append(int(l))
    auc = roc_auc_score(labels, scores)
    df = pd.DataFrame(rows)
    out_path = os.path.join("diagnostics", "error_level_baseline.csv")
    os.makedirs("diagnostics", exist_ok=True)
    df.to_csv(out_path, index=False)
    # also write a tiny summary JSON
    with open(os.path.join("diagnostics", "error_level_summary.json"), "w") as f:
        json.dump({"auc": auc, "n_samples": len(scores)}, f, indent=2)
    print("Baseline error‑level AUC:", auc)

def blank_tamper_and_evaluate(model, test_loader, mask_key="tamper_mask_path"):
    # This function assumes the dataset has a column "tamper_mask_path" that points to a binary mask image.
    # If not present we simply blank the central 64x64 region as a fallback.
    rows = []
    scores = []
    labels = []
    for batch in test_loader:
        imgs = batch["image"].to(model.device)
        labs = batch["label"].numpy()
        # retrieve mask paths if available
        mask_paths = batch.get(mask_key, None)
        if mask_paths is not None:
            # blank mask region
            for i, mp in enumerate(mask_paths):
                if os.path.exists(mp):
                    mask = cv2.imread(mp, cv2.IMREAD_GRAYSCALE)
                    mask = (mask > 0).astype(np.uint8)
                    img_np = imgs[i].cpu().numpy().transpose(1,2,0)
                    img_np[mask==1] = 0.5  # neutral gray
                    imgs[i] = torch.from_numpy(img_np.transpose(2,0,1)).to(imgs.device)
        else:
            # fallback: blank centre 64x64
            B, C, H, W = imgs.shape
            cx, cy = W//2, H//2
            half = 32
            imgs[:, :, cy-half:cy+half, cx-half:cx+half] = 0.5
        with torch.no_grad():
            out = model(imgs)
            prob = out["proba"][:, 1].cpu().numpy()
        for p, l in zip(prob, labs):
            rows.append({"prob_forged": float(p), "label": int(l)})
            scores.append(p)
            labels.append(l)
    auc = roc_auc_score(labels, scores)
    df = pd.DataFrame(rows)
    out_path = os.path.join("diagnostics", "blanked_region_metrics.json")
    with open(out_path, "w") as f:
        json.dump({"auc": auc, "rows": rows}, f, indent=2)
    print("Blanked‑region AUC:", auc)

def grad_cam_example(model, loader, out_dir="diagnostics/grad_cam"):
    import torchvision.transforms as T
    from torch.nn import ReLU
    os.makedirs(out_dir, exist_ok=True)
    # Grab first few batches (2 genuine, 2 forged)
    count = 0
    for batch in loader:
        imgs = batch["image"].to(model.device)
        paths = batch["image_path"]
        labels = batch["label"].numpy()
        # Get features from last conv block
        model.eval()
        # Hook to capture activation
        activations = []
        def hook_fn(module, inp, out):
            activations.append(out.detach())
        handle = model.backbone.register_forward_hook(hook_fn)
        with torch.no_grad():
            _ = model(imgs)
        handle.remove()
        act = activations[0]  # shape B, C, H, W
        # Compute weights by global average pooling
        weights = torch.mean(act, dim=(2,3), keepdim=True)  # B, C, 1, 1
        cam = torch.sum(weights * act, dim=1)  # B, H, W
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = cam.cpu().numpy()
        for i in range(imgs.size(0)):
            img_np = imgs[i].cpu().numpy().transpose(1,2,0)
            img_np = (img_np * np.array([0.229,0.224,0.225]) + np.array([0.485,0.456,0.406])
            img_np = np.clip(img_np, 0, 1)
            heatmap = cv2.applyColorMap(np.uint8(255*cam[i]), cv2.COLORMAP_JET)
            overlay = 0.4*heatmap/255.0 + 0.6*img_np
            overlay = np.clip(overlay,0,1)
            out_path = os.path.join(out_dir, os.path.basename(paths[i]) + "_gradcam.png")
            cv2.imwrite(out_path, np.uint8(255*overlay[..., ::-1]))
            count += 1
            if count >= 8:
                return

if __name__ == "__main__":
    # Load test split
    loaders = load_split_dataloaders("D:/visual/data/metadata/split_in_domain.csv", batch_size=16, num_workers=0)
    test_loader = loaders["test"]
    # Load visual model checkpoint
    model = get_default_visual_model().to(torch.device('cpu'))
    model.eval()
    # Baseline error‑level metric
    baseline_error_level(test_loader)
    # Blank‑region evaluation
    blank_tamper_and_evaluate(model, test_loader)
    # Grad‑CAM samples (use a small subset of test loader)
    grad_cam_example(model, test_loader)
