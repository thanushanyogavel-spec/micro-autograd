/** Rough throughput numbers so the README can quote real figures. */
import { tensor, matmul, tanh, sum, randn } from "../src/tensor.js";

function time(label: string, fn: () => void) {
  fn();                                   // warm up
  const t0 = process.hrtime.bigint();
  fn();
  const ms = Number(process.hrtime.bigint() - t0) / 1e6;
  console.log(`  ${label.padEnd(34)} ${ms.toFixed(1)} ms`);
}

const a = randn([256, 256], 1, 1);
const b = randn([256, 256], 1, 2);
const x = randn([100000], 1, 3);

time("matmul 256x256 x 256x256 (fwd)", () => matmul(a, b));
time("matmul 256x256 x 256x256 (fwd+bwd)", () => {
  const c = matmul(a, b);
  sum(c).backward();
});
time("tanh over 100k elements (fwd+bwd)", () => {
  const y = tanh(x);
  sum(y).backward();
});
