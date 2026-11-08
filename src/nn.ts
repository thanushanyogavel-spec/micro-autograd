import { Tensor, addBias, matmul, mul, mean, relu, sub, tanh, randn, zeros, sum } from "./tensor.js";

export interface Module {
  forward(x: Tensor): Tensor;
  parameters(): Tensor[];
}

export class Linear implements Module {
  W: Tensor;
  b: Tensor;
  constructor(inDim: number, outDim: number, seed?: number) {
    const std = Math.sqrt(2 / inDim);           // He initialisation
    this.W = randn([inDim, outDim], std, seed);
    this.b = zeros([outDim]);
  }
  forward(x: Tensor): Tensor { return addBias(matmul(x, this.W), this.b); }
  parameters(): Tensor[] { return [this.W, this.b]; }
}

export class ReLU implements Module {
  forward(x: Tensor): Tensor { return relu(x); }
  parameters(): Tensor[] { return []; }
}

export class Tanh implements Module {
  forward(x: Tensor): Tensor { return tanh(x); }
  parameters(): Tensor[] { return []; }
}

export class Sequential implements Module {
  private layers: Module[];
  constructor(...layers: Module[]) { this.layers = layers; }
  forward(x: Tensor): Tensor { return this.layers.reduce((acc, l) => l.forward(acc), x); }
  parameters(): Tensor[] { return this.layers.flatMap((l) => l.parameters()); }
}

/** mean squared error between two identically shaped tensors */
export function mse(pred: Tensor, target: Tensor): Tensor {
  return mean(mul(sub(pred, target), sub(pred, target)));
}

/** softmax cross-entropy over a batch of (m, classes) logits */
export function crossEntropy(logits: Tensor, labels: number[]): Tensor {
  const [m, c] = logits.shape;
  const probs = new Float64Array(m * c);
  for (let i = 0; i < m; i++) {
    let mx = -Infinity;
    for (let j = 0; j < c; j++) mx = Math.max(mx, logits.data[i * c + j]);
    let s = 0;
    for (let j = 0; j < c; j++) { probs[i * c + j] = Math.exp(logits.data[i * c + j] - mx); s += probs[i * c + j]; }
    for (let j = 0; j < c; j++) probs[i * c + j] /= s;
  }
  const nll = new Float64Array(m);
  for (let i = 0; i < m; i++) nll[i] = -Math.log(Math.max(probs[i * c + labels[i]], 1e-12));

  const t = new Tensor(new Float64Array([0]), [1], logits.requiresGrad);
  let loss = 0;
  for (let i = 0; i < m; i++) loss += nll[i];
  t.data[0] = loss / m;
  t._prev = [logits];
  t._backward = () => {
    if (!logits.requiresGrad) return;
    const g = t.grad[0] / m;
    for (let i = 0; i < m; i++)
      for (let j = 0; j < c; j++)
        logits.grad[i * c + j] += g * (probs[i * c + j] - (j === labels[i] ? 1 : 0));
  };
  return t;
}

export { sum };
