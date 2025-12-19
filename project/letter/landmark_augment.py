

import numpy as np
import math
import random

def split_lr(x):
    # x shape (84,)
    left = x[:42].reshape(-1,2)  # (21,2)
    right = x[42:].reshape(-1,2)
    return left, right

def flatten_lr(left, right):
    return np.concatenate([left.reshape(-1), right.reshape(-1)], axis=0)

class RandomJitter:
    def __init__(self, sigma=0.01):
        self.sigma = sigma
    def __call__(self, x):
        noise = np.random.normal(0, self.sigma, size=x.shape).astype(np.float32)
        return x + noise

class RandomScaleRotate:
    def __init__(self, scale_range=(0.97,1.03), rot_deg=5):
        self.scale_low, self.scale_high = scale_range
        self.rot_deg = rot_deg
    def apply_to_points(self, pts, scale, theta):
        # pts: (21,2) normalized
        # center at mean
        center = pts.mean(axis=0, keepdims=True)
        pts_c = pts - center
        # scale
        pts_s = pts_c * scale
        # rotate
        th = math.radians(theta)
        R = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]], dtype=np.float32)
        pts_r = (pts_s @ R.T) + center
        return pts_r
    def __call__(self, x):
        left, right = split_lr(x)
        s = random.uniform(self.scale_low, self.scale_high)
        theta = random.uniform(-self.rot_deg, self.rot_deg)
        # apply same transform to both hands for realism (so relative pos preserved)
        left_t = self.apply_to_points(left, s, theta)
        right_t = self.apply_to_points(right, s, theta)
        return flatten_lr(left_t, right_t)

class RandomFlipLR:
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, x):
        if random.random() > self.p:
            return x
        left, right = split_lr(x)
        # swap left/right and mirror x coordinates
        left_new = right.copy()
        right_new = left.copy()
        # flip x: x' = 1 - x (assuming normalized coordinates). Keep y the same.
        left_new[:,0] = 1.0 - left_new[:,0]
        right_new[:,0] = 1.0 - right_new[:,0]
        return flatten_lr(left_new, right_new)

class AddGlobalTranslation:
    def __init__(self, tx_range=(-0.03,0.03), ty_range=(-0.03,0.03)):
        self.tx_range = tx_range
        self.ty_range = ty_range
    def __call__(self, x):
        tx = random.uniform(*self.tx_range)
        ty = random.uniform(*self.ty_range)
        left, right = split_lr(x)
        left_t = left + np.array([tx, ty], dtype=np.float32)
        right_t = right + np.array([tx, ty], dtype=np.float32)
        return flatten_lr(left_t, right_t)

class Compose:
    def __init__(self, ops):
        self.ops = ops
    def __call__(self, x):
        for o in self.ops:
            x = o(x)
        return x

# small convenience: realistic default augment pipeline
def default_augment():
    return Compose([
        RandomJitter(sigma=0.01),
        RandomScaleRotate(scale_range=(0.97,1.03), rot_deg=6),
        AddGlobalTranslation(tx_range=(-0.02,0.02), ty_range=(-0.02,0.02)),
        #RandomFlipLR(p=0.0)  # enable if you want handedness invariance
    ])
