import pytest

from motionpredict.schemas import Detection
from motionpredict.tracking import ByteTracker, iou_matrix
import numpy as np


def box(cx, cy, w=40, h=100, conf=0.9):
    return Detection(bbox=[cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], confidence=conf)


def test_iou():
    a = np.array([[0, 0, 10, 10]], float)
    assert iou_matrix(a, a)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(a, np.array([[20, 20, 30, 30]], float))[0, 0] == 0.0


def test_ids_stable_through_crossing():
    tr = ByteTracker(high_thresh=0.5, track_buffer_s=1.0)
    ids_left = set()
    for i in range(40):
        t = i * 0.05
        a, b = 100 + 8 * i, 420 - 8 * i  # walk towards each other and cross
        tr.update([box(a, 200), box(b, 210)], t)
        trk = sorted([x for x in tr.tracks if x.state == "confirmed"], key=lambda x: x.kf.box[0])
        if i == 5:
            ids_left = {trk[0].track_id}
    final = {x.track_id: x.kf.box[0] for x in tr.tracks}
    assert len(final) == 2
    # the track that started on the left must now be on the right
    (left_id,) = ids_left
    assert final[left_id] == max(final.values())


def test_occlusion_recovery_and_low_score_extension():
    tr = ByteTracker(high_thresh=0.5, track_buffer_s=1.0)
    for i in range(10):
        tr.update([box(100 + 5 * i, 200)], i * 0.1)
    tid = tr.tracks[0].track_id
    for i in range(10, 14):  # fully missing for 0.4 s
        tr.update([], i * 0.1)
    assert tr.tracks[0].state == "lost"
    tr.update([box(100 + 5 * 14, 200)], 1.4)
    assert tr.tracks[0].track_id == tid and tr.tracks[0].state == "confirmed"
    tr.update([box(100 + 5 * 15, 200, conf=0.2)], 1.5)  # low score only extends existing tracks
    assert tr.tracks[0].matched_this_frame and len(tr.tracks) == 1


def test_lost_tracks_removed_and_reset_clears_ids():
    tr = ByteTracker(track_buffer_s=0.5)
    for i in range(5):
        tr.update([box(100, 200)], i * 0.1)
    for i in range(5, 15):
        tr.update([], i * 0.1)
    assert tr.tracks == []
    tr.reset()
    tr.update([box(100, 200)], 0)
    assert tr.tracks[0].track_id == 1


def test_low_confidence_does_not_spawn_tracks():
    tr = ByteTracker(high_thresh=0.5, new_track_thresh=0.6)
    tr.update([box(100, 100, conf=0.3), box(300, 100, conf=0.55)], 0)
    assert tr.tracks == []


def test_reset_keep_ids_never_reuses_ids():
    tr = ByteTracker()
    tr.update([box(100, 200)], 0.0)
    first = tr.tracks[0].track_id
    tr.reset(keep_ids=True)
    tr.update([box(100, 200)], 0.0)
    assert tr.tracks[0].track_id > first


def test_time_going_backwards_drops_old_tracks():
    tr = ByteTracker(track_buffer_s=1.0)
    for i in range(10):
        tr.update([box(100 + 5 * i, 200)], 5.0 + i * 0.1)
    old = {t.track_id for t in tr.tracks}
    tr.update([box(400, 200)], 0.0)  # looped video
    assert all(t.track_id not in old for t in tr.tracks)
    assert all(0.0 - t.last_t >= 0 for t in tr.tracks)
