import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np
import torch

# ---------- paths ----------
HERE = os.path.dirname(os.path.abspath(__file__))
VSVIG_DIR = os.path.join(HERE, "VSViG")
sys.path.insert(0, VSVIG_DIR)
sys.path.insert(0, os.path.join(HERE, "lwopenpose"))
from models.with_mobilenet import PoseEstimationWithMobileNet  # noqa: E402
from modules.keypoints import extract_keypoints, group_keypoints  # noqa: E402

# ---------- settings ----------
POSE_WEIGHTS = os.path.join(VSVIG_DIR, "pose.pth")
POSE_HEIGHT = 256       # frames are shrunk to this height for the pose model
KPT_CONF = 0.5
VSVIG_THRESHOLD = 0.5   # VSViG seizure probability needed
VSVIG_EVERY = 10        # run VSViG every N frames (it looks at the last 30)
ALERT_SEC = 5.0         # a seizure type must last this long
CLIP = 30               # VSViG clip length
MOTION_WINDOW = 1.0     # seconds of movement the score looks at
MOTION_FULL = 4.0       # body speed (torso-lengths/sec) that counts as a score of 1.0

DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


# ---------- pose model ----------
# VSViG's own pose model: Lightweight OpenPose fine-tuned on patients in bed.
# Much more accurate than COCO-trained YOLO on people lying down or under blankets.
# Its 18 points: nose, neck, R sho, R elb, R wri, L sho, L elb, L wri,
# R hip, R knee, R ank, L hip, L knee, L ank, R eye, L eye, R ear, L ear
BONES = [(1, 2), (1, 5), (2, 3), (3, 4), (5, 6), (6, 7), (1, 8), (8, 9), (9, 10),
         (1, 11), (11, 12), (12, 13), (1, 0), (0, 14), (14, 16), (0, 15), (15, 17)]

# OpenPose -> VSViG's 15 points:
# nose, L eye, R eye, R sho, R elb, R wri, L sho, L elb, L wri, R hip, R knee, R ank, L hip, L knee, L ank
TO_VSVIG = [0, 15, 14, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
# OpenPose -> COCO-17 order, used by the rule checks below
TO_COCO = [0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10]

pose_net = PoseEstimationWithMobileNet()
pose_net.load_state_dict(torch.load(POSE_WEIGHTS, map_location="cpu")["state_dict"])
pose_net = pose_net.eval().to(DEVICE)


@torch.no_grad()
def detect_pose(frame):
    """frame: BGR image. Returns the most confident person's (18, 3) keypoints
    (x, y, conf) in frame pixels, conf = 0 for joints not found. None if nobody is found."""
    stride, upsample = 8, 4
    scale = POSE_HEIGHT / frame.shape[0]
    img = cv2.resize(frame, (0, 0), fx=scale, fy=scale).astype(np.float32)
    img = (img - 128) / 256
    pad_w = math.ceil(img.shape[1] / stride) * stride - img.shape[1]
    img = np.pad(img, ((0, 0), (0, pad_w), (0, 0)))

    x = torch.from_numpy(img).permute(2, 0, 1)[None].to(DEVICE)
    heatmaps, pafs = pose_net(x)[-2:]
    heatmaps = cv2.resize(heatmaps[0].permute(1, 2, 0).cpu().numpy(), (0, 0), fx=upsample, fy=upsample,
                          interpolation=cv2.INTER_CUBIC)
    pafs = cv2.resize(pafs[0].permute(1, 2, 0).cpu().numpy(), (0, 0), fx=upsample, fy=upsample,
                      interpolation=cv2.INTER_CUBIC)

    by_type, total = [], 0
    for i in range(18):
        total += extract_keypoints(heatmaps[:, :, i], by_type, total)
    people, all_kp = group_keypoints(by_type, pafs)
    if len(people) == 0:
        return None
    all_kp[:, :2] = all_kp[:, :2] * stride / upsample / scale

    best = max(people, key=lambda p: p[18])     # the patient: the most complete, confident skeleton
    kp = np.zeros((18, 3))
    for i in range(18):
        if best[i] != -1:
            kp[i] = all_kp[int(best[i]), :3]
    return kp


def draw_skeleton(frame, kp, color=(0, 200, 0)):
    """Draw the 18-keypoint skeleton onto frame (in place)."""
    if kp is None:
        return frame
    pts = kp[:, :2].astype(int)
    seen = kp[:, 2] > 0
    for a, b in BONES:
        if seen[a] and seen[b]:
            cv2.line(frame, tuple(pts[a]), tuple(pts[b]), color, 3)
    for (x, y), ok in zip(pts, seen):
        if ok:
            cv2.circle(frame, (x, y), 4, (255, 255, 255), -1)
    return frame


# ---------- VSViG model ----------
def load_vsvig():
    order = torch.load(f"{VSVIG_DIR}/dy_point_order.pt", map_location="cpu", weights_only=False)
    real_load = torch.load
    torch.load = lambda *a, **k: order          # repo has a placeholder path inside VSViG_base
    from VSViG import VSViG_base
    model = VSViG_base()
    torch.load = real_load

    ckpt = torch.load(f"{VSVIG_DIR}/VSViG-base.pth", map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if not isinstance(ckpt, dict):              # whole model was saved
        return ckpt.eval()
    ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=False)
    return model.eval()


vsvig = load_vsvig()
KERNEL = cv2.getGaussianKernel(128, 128 * 0.3)
KERNEL = KERNEL @ KERNEL.T
KERNEL = (KERNEL - KERNEL.min()) / (KERNEL.max() - KERNEL.min())
KERNEL = np.repeat(KERNEL[:, :, None], 3, axis=2)


def extract_patches(frame, kpts15):
    """Same as the repo's extract_patches.py: 128x128 gaussian-weighted crop around
    each keypoint on a 1920x1080 frame, resized to 32x32. Returns (15, 3, 32, 32)."""
    img = np.pad(frame, ((128, 128), (128, 128), (0, 0))).astype(float)
    out = np.zeros((15, 3, 32, 32), dtype=np.float32)
    for i, (x, y) in enumerate(kpts15.astype(int)):
        crop = img[y + 64:y + 192, x + 64:x + 192] * KERNEL
        crop = 255 * (crop - crop.min()) / (crop.max() - crop.min() + 1e-6)
        out[i] = cv2.resize(crop, (32, 32)).transpose(2, 0, 1)
    return out


@torch.no_grad()
def vsvig_probability(patch_buf, kpt_buf):
    x = torch.from_numpy(np.stack(patch_buf))[None]          # 1, 30, 15, 3, 32, 32
    k = torch.from_numpy(np.stack(kpt_buf)).float()[None]     # 1, 30, 15, 3
    return float(vsvig(x, k))


# ---------- rule checks for each seizure type ----------
def body_frame(k):
    """Keypoints relative to torso center, in torso-lengths."""
    xy = k[:, :2].astype(float)
    xy[k[:, 2] < KPT_CONF] = np.nan
    sh, hp = np.nanmean(xy[[5, 6]], 0), np.nanmean(xy[[11, 12]], 0)
    if np.isnan(sh).any() or np.isnan(hp).any():
        return None, None, None
    size = np.linalg.norm(sh - hp)
    if size < 1:
        return None, None, None
    return (xy - (sh + hp) / 2) / size, hp, size


def rhythmic(series, fps):
    """True if a movement is rhythmic jerking at 2-6 Hz."""
    s = series[~np.isnan(series)]
    if len(s) < 0.7 * len(series) or len(s) < 2 * fps:
        return False
    s = s - np.polyval(np.polyfit(np.arange(len(s)), s, 1), np.arange(len(s)))
    if s.std() < 0.03:
        return False
    p = np.abs(np.fft.rfft(s * np.hanning(len(s)))) ** 2
    f = np.fft.rfftfreq(len(s), 1 / fps)
    band = (f >= 2) & (f <= 6)
    return p[band].sum() / (p[f >= 0.5].sum() + 1e-9) > 0.45


LEFT, RIGHT = [7, 9, 13, 15], [8, 10, 14, 16]   # L/R elbow, wrist, knee, ankle


def classify(hist, fps):
    """hist: list of (body_xy, hip_px, size) for the last 4 s. Returns set of seizure types."""
    pts = np.stack([h[0] for h in hist])
    types = set()

    # 1) Tonic-clonic / clonic: rhythmic jerking in limbs on BOTH sides
    # 2) Focal motor: rhythmic jerking on ONE side only
    left = any(rhythmic(pts[:, i, a], fps) for i in LEFT for a in (0, 1))
    right = any(rhythmic(pts[:, i, a], fps) for i in RIGHT for a in (0, 1))
    if left and right:
        types.add("tonic-clonic")
    elif left or right:
        types.add("focal motor")

    # 3) Myoclonic: sudden single jerks (fast spikes) without steady rhythm
    speed = np.nanmax(np.linalg.norm(np.diff(pts[:, LEFT + RIGHT], axis=0), axis=2), axis=1) * fps
    spikes = np.sum(speed > 8)
    if spikes >= 3 and not (left or right):
        types.add("myoclonic")

    # 4) Tonic: arms stiff and extended away from body, almost no movement
    arm_out = np.nanmean(np.linalg.norm(pts[:, [9, 10]] - pts[:, [5, 6]], axis=2)) > 0.9
    still = np.nanmedian(speed) < 0.3
    if arm_out and still:
        types.add("tonic")

    # 5) Atonic: sudden collapse - hips drop more than one torso-length within 1 s
    hips = np.array([h[1][1] / h[2] for h in hist])
    n = int(fps)
    if len(hips) > n and np.nanmax(hips[n:] - hips[:-n]) > 1.0:
        types.add("atonic / fall")

    return types


# ---------- per-frame detector (used by the app and by run_camera) ----------
class SeizureDetector:
    """Feed it frames one at a time. process() returns (annotated_frame, seizure, prob)."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.hist = deque()
        self.patch_buf = deque(maxlen=CLIP)
        self.kpt_buf = deque(maxlen=CLIP)
        self.type_start = {}
        self.prob = 0.0            # VSViG seizure probability
        self.motion_score = 0.0    # 0..1, how much the body is moving
        self.score = 0.0           # combined score shown in the app
        self.prev_body = None
        self.prev_t = None
        self.last_kp = None        # last known position of each joint, for frames where it's hidden
        self.speeds = deque()      # (time, speed) over the last MOTION_WINDOW seconds
        self.n = 0
        self.start = time.time()
        self.seizure = False
        self.types = []

    def _update_motion(self, body, t):
        """Body speed in torso-lengths/sec from the change in keypoints since last frame."""
        if body is not None and self.prev_body is not None and t > self.prev_t:
            d = np.linalg.norm(body - self.prev_body, axis=1)
            if np.isfinite(d).sum() >= 4:
                self.speeds.append((t, float(np.nanmean(d)) / (t - self.prev_t)))
        if body is not None:
            self.prev_body, self.prev_t = body, t
        while self.speeds and t - self.speeds[0][0] > MOTION_WINDOW:
            self.speeds.popleft()
        if self.speeds:
            motion = float(np.median([s for _, s in self.speeds]))
            self.motion_score = min(1.0, motion / MOTION_FULL)
        else:
            self.motion_score *= 0.9   # no person / no motion: fade out
        self.score = max(self.prob, self.motion_score)

    def _fill_missing(self, kp):
        """Joints the pose model can't see keep their last known position (confidence stays 0)."""
        kp = kp.copy()
        missing = kp[:, 2] <= 0
        if self.last_kp is not None:
            kp[missing, :2] = self.last_kp[missing, :2]
        elif (~missing).any():
            kp[missing, :2] = kp[~missing, :2].mean(axis=0)
        self.last_kp = kp
        return kp

    def process(self, frame, t=None):
        """frame: BGR image. t: time in seconds (video time); defaults to wall-clock."""
        if t is None:
            t = time.time() - self.start
        self.n += 1
        fps = max(self.n / max(t, 1e-6), 1)

        kp = detect_pose(frame)
        drawn = draw_skeleton(frame.copy(), kp)
        found = set()

        if kp is not None:
            # VSViG input
            h, w = frame.shape[:2]
            big = cv2.resize(frame, (1920, 1080))
            k15 = self._fill_missing(kp)[TO_VSVIG]
            k15[:, 0] *= 1920 / w
            k15[:, 1] *= 1080 / h
            self.patch_buf.append(extract_patches(big, k15[:, :2]))
            self.kpt_buf.append(k15)   # (x, y, conf), pixel coords on 1920x1080
            if len(self.patch_buf) == CLIP and self.n % VSVIG_EVERY == 0:
                self.prob = vsvig_probability(self.patch_buf, self.kpt_buf)
            if self.prob >= VSVIG_THRESHOLD:
                found.add("VSViG model")

            # rule checks
            body, hip, size = body_frame(kp[TO_COCO])
            self._update_motion(body, t)
            if body is not None:
                self.hist.append((t, body, hip, size))
                while t - self.hist[0][0] > 4.0:
                    self.hist.popleft()
                if t - self.hist[0][0] > 3.5:
                    found |= classify([x[1:] for x in self.hist], fps)
        else:
            self._update_motion(None, t)

        # a type must last ALERT_SEC (falls count right away)
        for name in list(self.type_start):
            if name not in found:
                self.type_start.pop(name)
        for name in found:
            self.type_start.setdefault(name, t)
        confirmed = [s for s, t0 in self.type_start.items()
                     if t - t0 >= ALERT_SEC or s == "atonic / fall"]

        if confirmed and not self.seizure:
            self.seizure = True
            self.types = confirmed
            print(f"Seizure detected! type: {', '.join(confirmed)}")

        return drawn, self.seizure, self.score


# ---------- main loop ----------
def run_camera(camera=0):
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        print("Could not open camera.")
        return

    detector = SeizureDetector()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        drawn, _, _ = detector.process(frame)
        cv2.imshow("Seizure monitor (q = quit, r = reset)", drawn)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord("r"):
            detector.reset()
            print("Reset: seizure = 0")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print(f"Using {DEVICE}")
    run_camera(0)
