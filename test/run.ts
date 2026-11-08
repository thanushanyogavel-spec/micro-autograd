/**
 * Test suite. The important part is the numerical gradient check: every op's
 * analytic gradient is compared against central finite differences. If the
 * calculus is wrong anywhere, these fail.
 */
import {
  Tensor, add, sub, mul, matmul, addBias, relu, tanh, mean, sum, scale,
  tensor, randn, zeros,
} from "../src/tensor.js";
import { Linear, Sequential, Tanh as TanhLayer, mse, crossEntropy } from "../src/nn.js";
import { SGD } from "../src/optim.js";

let passed = 0, failed = 0;
const results: string[] = [];

function ok(name: string, cond: boolean, detail = "") {
  if (cond) { passed++; results.push(`  PASS  ${name}`); }
  else { failed++; results.push(`  FAIL  ${name}${detail ? "  -> " + detail : ""}`); }
}

function close(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol * Math.max(1, Math.abs(a), Math.abs(b));
}

/** central-difference gradient check for a scalar-valued function */
function gradCheck(name: string, build: (x: Tensor) => Tensor, values: number[], shape: number[], h = 1e-5) {
  const x = new Tensor(Float64Array.from(values), shape, true);
  const loss = build(x);
  loss.backward();

  let worst = 0;
  for (let i = 0; i < values.length; i++) {
    const orig = x.data[i];
    x.data[i] = orig + h;
    const fp = build(x).data[0];
    x.data[i] = orig - h;
    const fm = build(x).data[0];
    x.data[i] = orig;

    const numeric = (fp - fm) / (2 * h);
    const analytic = x.grad[i];
    const denom = Math.max(1, Math.abs(numeric), Math.abs(analytic));
    worst = Math.max(worst, Math.abs(numeric - analytic) / denom);
  }
  ok(name, worst < 1e-4, `max relative error ${worst.toExponential(2)}`);
}

/* ── forward correctness ─────────────────────────────────────────────── */
{
  const a = tensor([1, 2, 3]);
  const b = tensor([4, 5, 6]);
  ok("add", add(a, b).data.join(",") === "5,7,9");
  ok("sub", sub(a, b).data.join(",") === "-3,-3,-3");
  ok("mul", mul(a, b).data.join(",") === "4,10,18");
  ok("sum", sum(a).data[0] === 6);
  ok("mean", mean(a).data[0] === 2);
  ok("scale", scale(a, 2).data.join(",") === "2,4,6");

  const m = tensor([[1, 2], [3, 4]]);
  const n = tensor([[5, 6], [7, 8]]);
  const p = matmul(m, n);
  ok("matmul shape", p.shape.join(",") === "2,2");
  ok("matmul values", p.data.join(",") === "19,22,43,50");

  const biased = addBias(m, tensor([10, 20]));
  ok("addBias", biased.data.join(",") === "11,22,13,24");

  ok("relu", relu(tensor([-1, 0, 2])).data.join(",") === "0,0,2");
  ok("tanh", close(tanh(tensor([0])).data[0], 0));
}

/* ── gradient checks ─────────────────────────────────────────────────── */
gradCheck("grad: add", (x) => sum(add(x, tensor([1, 2, 3]))), [0.5, -1, 2], [3]);
gradCheck("grad: mul", (x) => sum(mul(x, tensor([2, -3, 0.5]))), [0.5, -1, 2], [3]);
gradCheck("grad: scale", (x) => sum(scale(x, 3)), [1, 2, 3], [3]);
gradCheck("grad: tanh", (x) => sum(tanh(x)), [0.3, -0.8, 1.4], [3]);
gradCheck("grad: relu", (x) => sum(relu(x)), [0.3, -0.8, 1.4], [3]);
gradCheck("grad: sum", (x) => sum(x), [1, 2, 3], [3]);
gradCheck("grad: mean", (x) => mean(x), [1, 2, 3], [3]);

gradCheck(
  "grad: matmul",
  (x) => sum(matmul(x, tensor([[1, 2], [3, 4]]))),
  [0.5, -1, 2, 0.25], [2, 2]
);
gradCheck(
  "grad: addBias",
  (x) => sum(addBias(x, tensor([1, -1]))),
  [0.5, -1, 2, 0.25], [2, 2]
);
gradCheck(
  "grad: chain (mul -> tanh -> sum)",
  (x) => sum(tanh(mul(x, x))),
  [0.3, -0.8], [2]
);
gradCheck(
  "grad: mse loss",
  (x) => mse(x, tensor([[1, 0], [0, 1]])),
  [0.2, 0.9, -0.4, 1.3], [2, 2]
);
gradCheck(
  "grad: cross entropy",
  (x) => crossEntropy(x, [0, 1]),
  [0.2, -0.9, 1.3, 0.4], [2, 2]
);

/* ── graph behaviour ─────────────────────────────────────────────────── */
{
  // gradients accumulate across two backward passes
  const x = tensor([2], true);
  const y = mul(x, x);
  y.backward();
  const first = x.grad[0];
  y.backward();
  ok("grad accumulates", close(x.grad[0], first * 2, 1e-9), `${x.grad[0]} vs ${first * 2}`);

  // zeroGrad resets
  x.zeroGrad();
  ok("zeroGrad", x.grad[0] === 0);

  // a node used twice in the graph receives both contributions
  const z = tensor([3], true);
  const shared = mul(z, z);
  const out = add(shared, shared);          // d/dz (2z^2) = 4z = 12
  out.backward();
  ok("shared node gradient", close(z.grad[0], 12, 1e-9), String(z.grad[0]));
}

/* ── a real network actually learns ──────────────────────────────────── */
{
  const net = new Sequential(new Linear(2, 8, 1), new TanhLayer(), new Linear(8, 1, 2));
  const opt = new SGD(net.parameters(), 0.3, 0.9);
  const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
  const Y = tensor([[0], [1], [1], [0]]);

  let last = Infinity;
  for (let epoch = 0; epoch < 3000; epoch++) {
    const loss = mse(net.forward(X), Y);
    opt.zeroGrad(net.parameters());
    loss.backward();
    opt.step(net.parameters());
    last = loss.data[0];
  }
  ok("MLP trains XOR below 0.01 loss", last < 0.01, `final loss ${last.toFixed(5)}`);

  const pred = net.forward(X);
  const rounded = Array.from(pred.data).map((v) => (v > 0.5 ? 1 : 0)).join(",");
  ok("XOR predictions correct", rounded === "0,1,1,0", rounded);
}

console.log(results.join("\n"));
console.log(`\n${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
