import assert from "node:assert/strict";
import { test } from "node:test";
import { bandFromSigma, interpAt, partialErrors, probLabel } from "./viz.ts";

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
