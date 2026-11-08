import { Tensor } from "./tensor.js";

export interface Optimizer {
  step(params: Tensor[]): void;
  zeroGrad(params: Tensor[]): void;
}

/** stochastic gradient descent with optional momentum */
export class SGD implements Optimizer {
  lr: number;
  momentum: number;
  private velocity = new WeakMap<Tensor, Float64Array>();

  constructor(params: Tensor[], lr = 0.1, momentum = 0.9) {
    this.lr = lr;
    this.momentum = momentum;
    for (const p of params) this.velocity.set(p, new Float64Array(p.size));
  }

  zeroGrad(params: Tensor[]): void { for (const p of params) p.zeroGrad(); }

  step(params: Tensor[]): void {
    for (const p of params) {
      let v = this.velocity.get(p);
      if (!v) { v = new Float64Array(p.size); this.velocity.set(p, v); }
      for (let i = 0; i < p.size; i++) {
        v[i] = this.momentum * v[i] + p.grad[i];
        p.data[i] -= this.lr * v[i];
      }
    }
  }
}
