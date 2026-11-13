"""Temporal discretisation shared by training and live inference.

The learned model is trained on ETH/UCY, which is annotated at 2.5 Hz. Live tracks are
resampled onto the same 0.4 s grid before being passed to the model, so the network sees
the same time step it was trained on regardless of camera frame rate.
"""

DT = 0.4  # seconds between model time steps
OBS_LEN = 8  # observed steps (2.8 s span, 8 positions)
PRED_LEN = 13  # predicted steps (5.2 s) - covers the 5 s UI horizon
MAX_NEIGHBORS = 8
NEIGHBOR_RADIUS = 5.0  # meters (in training units)
NUM_MODES = 5
MIN_OBS = 2  # minimum observed positions for any forecast
