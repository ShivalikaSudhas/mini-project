import cv2
import numpy as np
import paddle
import paddle.vision.models as models

# -------------------------------------------------------------------------
# VISUAL STREAM FEATURE EXTRACTOR
# -------------------------------------------------------------------------
# Extracts a 16-dimensional visual feature vector combining:
#   1. Deep Spatial Latents (8-D) via CNN backbone
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
    def __init__(self):
        self.backbone = models.mobilenet_v3_small(pretrained=False)
        self.backbone.eval()
        
    def extract_deep_cnn_features(self, img_bgr):
        """Extracts 8 compressed deep spatial latent features."""
        rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (224, 224))
        normalized = (resized / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor_in = paddle.to_tensor(normalized.transpose(2, 0, 1), dtype='float32').unsqueeze(0)
        
        with paddle.no_grad():
            preds = self.backbone(tensor_in).numpy().flatten()
            
        pooled = np.mean(preds.reshape(8, -1), axis=1)
        return pooled

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
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image at {img_path}")
            
        deep_feats = self.extract_deep_cnn_features(img)
        forensic_feats = self.extract_forensic_texture_features(img)
        return np.concatenate([deep_feats, forensic_feats])
