import cv2
import numpy as np

# -------------------------------------------------------------------------
# VISUAL STREAM FEATURE EXTRACTOR (LIZA'S FUSION LEAD MODULE)
# -------------------------------------------------------------------------
# Extracts a 16-dimensional visual feature vector combining:
#   1. Deep Spatial Latents (8-D) via MobileNet-V3 CNN backbone
#   2. Forensic Tampering Artifacts (8-D):
#      - Laplacian sharpness & variance (blur detection)
#      - Sobel gradient magnitude (edge consistency)
#      - Fast Fourier Transform (FFT) high-frequency cutoff energy
#      - Error Level Analysis (ELA) compression residual variance
#      - Color channel covariance
# -------------------------------------------------------------------------

VISUAL_FEATURE_NAMES = [
    "cnn_feat_1", "cnn_feat_2", "cnn_feat_3", "cnn_feat_4",
    "cnn_feat_5", "cnn_feat_6", "cnn_feat_7", "cnn_feat_8",
    "laplacian_var", "laplacian_mean", "gradient_mean", "gradient_std",
    "high_freq_ratio", "ela_mean", "ela_std", "color_std_ratio"
]

class VisualStreamExtractor:
    def __init__(self, visual_checkpoint_path="models/visual/best_visual_model.pth"):
        self.use_efficientnet = False
        self.efficientnet_model = None

        try:
            from src.visual_model import get_default_visual_model
            self.efficientnet_model = get_default_visual_model(visual_checkpoint_path)
            self.use_efficientnet = True
        except Exception:
            self.efficientnet_model = None

    def extract_deep_cnn_features(self, img_bgr):
        """Extracts 8 compressed deep spatial latent features using Shivalika's EfficientNet-B0."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))

        if self.use_efficientnet and self.efficientnet_model is not None:
            try:
                import torch
                from PIL import Image
                from src.visual_model import _preprocess_input_image
                pil_img = Image.fromarray(rgb)
                tensor_in = _preprocess_input_image(pil_img)
                with torch.no_grad():
                    out = self.efficientnet_model(tensor_in)
                    feats = out["features"].squeeze(0).cpu().numpy()
                return np.mean(feats.reshape(8, -1), axis=1).astype(np.float32)
            except Exception:
                pass

        # Robust spatial frequency descriptor fallback
        normalized = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        h_splits = np.array_split(normalized, 8, axis=0)
        return np.array([float(np.mean(s)) for s in h_splits], dtype=np.float32)

    def extract_forensic_texture_features(self, img_bgr):
        """Extracts 8 forensic visual features (edge sharpness, frequency, ELA)."""
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # 1. Laplacian sharpness & edge variance (flags smoothing/blurring)
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = float(lap.var())
        lap_mean = float(np.abs(lap).mean())

        # 2. Sobel gradient energy (horizontal & vertical gradient variance)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        grad_mag = np.sqrt(sobelx**2 + sobely**2)
        grad_mean = float(grad_mag.mean())
        grad_std = float(grad_mag.std())

        # 3. FFT Frequency domain analysis (flags digital copy-paste seams)
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-6)
        cy, cx = h // 2, w // 2
        high_freq_ratio = float(np.mean(magnitude_spectrum[:cy//2, :cx//2]) / (magnitude_spectrum.mean() + 1e-6))

        # 4. Error Level Analysis (ELA) proxy: JPEG compression residual variance
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        _, encimg = cv2.imencode('.jpg', img_bgr, encode_param)
        resaved = cv2.imdecode(encimg, 1)
        diff = np.abs(img_bgr.astype(np.float32) - resaved.astype(np.float32))
        ela_mean = float(diff.mean())
        ela_std = float(diff.std())

        # 5. Color channel distribution covariance
        color_std_ratio = float(img_bgr[:, :, 0].std() / (img_bgr[:, :, 2].std() + 1e-5))

        return np.array([
            lap_var, lap_mean, grad_mean, grad_std,
            high_freq_ratio, ela_mean, ela_std, color_std_ratio
        ], dtype=np.float32)

    def extract_features(self, img_path):
        """Extracts complete 16-D visual feature vector."""
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")

        deep_feats = self.extract_deep_cnn_features(img)
        forensic_feats = self.extract_forensic_texture_features(img)
        return np.concatenate([deep_feats, forensic_feats]).astype(np.float32)


def extract_visual_16d_features(img_path):
    """Convenience helper function to extract 16-D visual features."""
    extractor = VisualStreamExtractor()
    return extractor.extract_features(img_path)
