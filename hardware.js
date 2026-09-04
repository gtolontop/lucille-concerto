// What one rack can do, and the teachers we run on it.
// The derivation, its anchor and its sources: docs/throughput.md

const HARDWARE = (() => {
"use strict";

// One NVL72 Rubin rack serving a sparse MoE teacher in NVFP4, with MTP
// speculative decoding, in a throughput regime — this is a dataset build,
// there is no latency SLA to respect.
//
//   anchor   one GB300 NVL72 at its *throughput-max* point, not its
//            interactive one: ~120 k tok/s aggregate on a 671B-A37B in FP4
//            with a single MTP head. That is ~1.7 k tok/s per GPU, i.e. well
//            under 1 % of the rack's dense FP4 flops — decode is bound by
//            memory traffic, not by arithmetic, so this is a serving point,
//            not a silicon limit.
//   rubin    ~3.3x that rack (FP4 dense flops + HBM4 bandwidth)
//   active   decode cost per token scales with *active* parameters
//   mtp      E[accepted tokens per step] = (1 - a^(k+1)) / (1 - a)
//   batch    no interactivity target at all buys another ~2x: batch is pushed
//            until the KV cache, not the latency budget, says stop
//
// The compute roofline (2 FLOP per active parameter per token against
// 3.6 EF FP4 at 25 % MFU) is applied as a hard ceiling, so the estimate can
// never claim more than the silicon can do.
const ANCHOR_TOKS = 120e3;
const ANCHOR_ACTIVE = 37e9;
const ANCHOR_SPEC = 1.8;
const RUBIN_GAIN = 3.3;
const BATCH_GAIN = 2.0;
const RACK_FLOPS_FP4 = 3.6e18;
const MFU = 0.25;

const TEACHERS = [
  // concerto-6 is the fictional teacher this floor is built around.
  { id: "concerto-6", name: "concerto-6", label: "CONCERTO 6",
    total: 1.6e12, active: 55e9, mtp: 4, accept: 0.62 },
  // GLM-5.3 rides the GLM-5.2 backbone: 744B total, 40B active per token,
  // 256 experts (8 routed + 1 shared), one MTP layer for speculative decode.
  { id: "glm-5.3", name: "glm-5.3", label: "GLM 5.3",
    total: 744e9, active: 40e9, mtp: 1, accept: 0.72 },
  // GLM-5.3-Flash, the small sibling: 320B total, 18B active.
  { id: "glm-5.3-flash", name: "glm-5.3-flash", label: "GLM 5.3 FLASH",
    total: 320e9, active: 18e9, mtp: 1, accept: 0.75 },
];

const RACKS = [
  { id: 1, name: "1" }, { id: 2, name: "2" }, { id: 3, name: "3" },
  { id: 4, name: "4" }, { id: 5, name: "5" }, { id: 16, name: "max" },
];

function specGain(m) {
  return (1 - Math.pow(m.accept, m.mtp + 1)) / (1 - m.accept);
}

function rackCapacity(m) {
  const est = ANCHOR_TOKS * RUBIN_GAIN * BATCH_GAIN
    * (ANCHOR_ACTIVE / m.active) * (specGain(m) / ANCHOR_SPEC);
  const roofline = RACK_FLOPS_FP4 * MFU / (2 * m.active);
  return Math.min(est, roofline);
}

return { TEACHERS, RACKS, specGain, rackCapacity };
})();
