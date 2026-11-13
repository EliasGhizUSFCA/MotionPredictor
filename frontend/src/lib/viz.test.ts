import assert from "node:assert/strict";
import { test } from "node:test";
import { bandFromSigma, interpAt, inv3, nearestForecast, partialErrors, probLabel, trackPosAt } from "./viz.ts";

test("interpAt interpolates and refuses to extrapolate", () => {
  assert.deepEqual(interpAt([0, 1], [[0, 0], [2, 4]], 0.5), [1, 2]);
  assert.equal(interpAt([0, 1], [[0, 0], [2, 4]], 1.5), null);
});

test("partialErrors only scores revealed time", () => {
  const t = [0, 1, 2];
  const pred: [number, number][] = [[0, 0], [1, 0], [2, 0]];
  const act: [number, number][] = [[0, 0], [1, 1], [2, 2]];
  assert.deepEqual(partialErrors(t, pred, t, act, 1), { ade: 1, fde: 1, n: 1 });
  assert.equal(partialErrors(t, pred, t, act, 2)!.fde, 2);
  assert.equal(partialErrors(t, pred, t, act, 0), null);
});

test("band is symmetric and widens with sigma", () => {
  const { left, right } = bandFromSigma([[0, 0], [1, 0], [2, 0]], [[0.1, 0.1, 0], [0.5, 0.5, 0], [1, 1, 0]], 2);
  assert.ok(Math.abs(left[2][1] - 2) < 1e-9 && Math.abs(right[2][1] + 2) < 1e-9);
  assert.ok(left[1][1] < left[2][1]);
});

test("probability labels never imply calibration that was not validated", () => {
  assert.equal(probLabel(0.8, "calibrated_probability"), "80%");
  assert.match(probLabel(0.8, "uncalibrated_weight"), /uncal/);
  assert.equal(probLabel(null, "none"), "no probability");
});

test("horizon keys match the backend's str(float) format", async () => {
  const { hk } = await import("./viz.ts");
  assert.equal(hk(3), "3.0");
  assert.equal(hk(0.5), "0.5");
});

test("partialErrors holds the last actual sample within the backend tolerance (np.interp clamp)", () => {
  const predT = [0, 1, 2, 3];
  const pred: [number, number][] = [[0, 0], [1, 0], [2, 0], [3, 0]];
  const actT = [0, 1, 2, 2.9];
  const act: [number, number][] = [[0, 0], [1, 0], [2, 0], [2.9, 0]];
  // ground truth ends at 2.9 s; at full reveal (3 s) the FDE uses the clamped last point, like scoring.py
  assert.ok(Math.abs(partialErrors(predT, pred, actT, act, 3)!.fde - 0.1) < 1e-9);
  assert.equal(partialErrors(predT, pred, actT, act, 3)!.n, 3);
  // beyond the tolerance nothing is invented
  assert.equal(partialErrors(predT, pred, [0, 1, 2.5], [[0, 0], [1, 0], [2.5, 0]], 3)!.n, 2);
});

test("inv3 inverts a homography", () => {
  const m = [[2, 0, 1], [0, 3, 2], [0.1, 0, 1]];
  const r = inv3(m)!;
  for (let i = 0; i < 3; i++)
    for (let j = 0; j < 3; j++) {
      const v = m[i][0] * r[0][j] + m[i][1] * r[1][j] + m[i][2] * r[2][j];
      assert.ok(Math.abs(v - (i === j ? 1 : 0)) < 1e-9);
    }
  assert.equal(inv3([[1, 2, 3], [2, 4, 6], [0, 0, 1]]), null);
});

test("trackPosAt interpolates and refuses gaps", () => {
  const obs = [[0, 0, 0], [1, 10, 0], [5, 50, 0]];
  assert.deepEqual(trackPosAt(obs, 0.5), [5, 0]);
  assert.equal(trackPosAt(obs, 3), null);
  assert.equal(trackPosAt(obs, 7), null);
});

test("nearestForecast prefers the latest forecast at or before t", () => {
  const ps = [{ t: 1 }, { t: 2 }, { t: 3 }];
  assert.equal(nearestForecast(ps, 2.7)!.t, 2);
  assert.equal(nearestForecast(ps, 0.2)!.t, 1);
  assert.equal(nearestForecast([], 1), undefined);
});
