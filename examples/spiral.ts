/**
 * Two interleaved spirals — the same benchmark used in CS231n. A network with
 * no hidden layer cannot separate these; with a hidden layer it can.
 */
import { tensor, Tensor } from "../src/tensor.js";
import { Linear, Sequential, ReLU, crossEntropy } from "../src/nn.js";
import { SGD } from "../src/optim.js";

function makeSpirals(pointsPerClass = 150, noise = 0.15, seed = 42): { X: Tensor; labels: number[] } {
  let s = seed;
  const rnd = () => { s = (s * 1664525 + 1013904223) % 4294967296; return s / 4294967296; };
  const xs: number[] = [], labels: number[] = [];
  for (let cls = 0; cls < 2; cls++) {
    for (let i = 0; i < pointsPerClass; i++) {
      const r = i / pointsPerClass;
      const t = cls * Math.PI + r * 3.5 * Math.PI + (rnd() - 0.5) * noise;
      xs.push(r * Math.sin(t), r * Math.cos(t));
      labels.push(cls);
    }
  }
  return { X: tensor(xs).reshape([pointsPerClass * 2, 2]), labels };
}

const { X, labels } = makeSpirals();
const net = new Sequential(
  new Linear(2, 64, 7),
  new ReLU(),
  new Linear(64, 64, 8),
  new ReLU(),
  new Linear(64, 2, 9)
);
const opt = new SGD(net.parameters(), 0.15, 0.9);

for (let epoch = 1; epoch <= 500; epoch++) {
  const loss = crossEntropy(net.forward(X), labels);
  opt.zeroGrad(net.parameters());
  loss.backward();
  opt.step(net.parameters());
  if (epoch % 50 === 0) console.log(`epoch ${String(epoch).padStart(3)}   loss ${loss.data[0].toFixed(4)}`);
}

// accuracy
const logits = net.forward(X);
let correct = 0;
for (let i = 0; i < labels.length; i++) {
  const a = logits.data[i * 2], b = logits.data[i * 2 + 1];
  const pred = a > b ? 0 : 1;
  if (pred === labels[i]) correct++;
}
console.log(`\naccuracy: ${(100 * correct / labels.length).toFixed(1)}%  (${correct}/${labels.length})`);
