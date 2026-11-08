/**
 * A reverse-mode automatic differentiation engine over dense float tensors.
 * No dependencies. Every op records how to push gradients back to its inputs,
 * and backward() walks the graph in topological order.
 */

export type Shape = number[];

let _id = 0;

export class Tensor {
  readonly id = _id++;
  data: Float64Array;
  grad: Float64Array;
  shape: Shape;
  requiresGrad: boolean;
  _backward: () => void = () => {};
  _prev: Tensor[] = [];

  constructor(data: ArrayLike<number>, shape?: Shape, requiresGrad = false) {
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
    this.shape = shape ?? [this.data.length];
    this.grad = new Float64Array(this.data.length);
    this.requiresGrad = requiresGrad;
  }

  get size(): number { return this.data.length; }

  /** return a view of the same data with a new shape */
  reshape(shape: Shape): Tensor {
    const n = numel(shape);
    if (n !== this.size) throw new Error(`reshape: cannot fit ${this.size} elements into [${shape}]`);
    const out = new Tensor(this.data, shape, this.requiresGrad);
    out._prev = [this];
    out._backward = () => {
      if (this.requiresGrad) for (let i = 0; i < this.size; i++) this.grad[i] += out.grad[i];
    };
    return out;
  }

  /** zero the accumulated gradient */
  zeroGrad(): void { this.grad.fill(0); }

  /** run backprop from this tensor (must be a scalar) */
  backward(): void {
    const topo: Tensor[] = [];
    const seen = new Set<Tensor>();
    const build = (t: Tensor) => {
      if (seen.has(t)) return;
      seen.add(t);
      for (const p of t._prev) build(p);
      topo.push(t);
    };
    build(this);
    this.grad.fill(0);
    this.grad[0] = 1;
    for (let i = topo.length - 1; i >= 0; i--) topo[i]._backward();
  }
}

/* ── helpers ─────────────────────────────────────────────────────────── */

const numel = (s: Shape) => s.reduce((a, b) => a * b, 1);

function sameShape(a: Tensor, b: Tensor): void {
  if (a.shape.length !== b.shape.length || a.shape.some((v, i) => v !== b.shape[i]))
    throw new Error(`shape mismatch: [${a.shape}] vs [${b.shape}]`);
}

function make(data: Float64Array, shape: Shape, prev: Tensor[], requiresGrad: boolean): Tensor {
  const t = new Tensor(data, shape, requiresGrad);
  t._prev = prev;
  return t;
}

const needsGrad = (ts: Tensor[]) => ts.some((t) => t.requiresGrad);

/* ── elementwise ops ─────────────────────────────────────────────────── */

export function add(a: Tensor, b: Tensor): Tensor {
  sameShape(a, b);
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] + b.data[i];
  const t = make(out, a.shape, [a, b], needsGrad([a, b]));
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[i];
    if (b.requiresGrad) for (let i = 0; i < b.size; i++) b.grad[i] += t.grad[i];
  };
  return t;
}

export function sub(a: Tensor, b: Tensor): Tensor {
  sameShape(a, b);
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] - b.data[i];
  const t = make(out, a.shape, [a, b], needsGrad([a, b]));
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[i];
    if (b.requiresGrad) for (let i = 0; i < b.size; i++) b.grad[i] -= t.grad[i];
  };
  return t;
}

export function mul(a: Tensor, b: Tensor): Tensor {
  sameShape(a, b);
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] * b.data[i];
  const t = make(out, a.shape, [a, b], needsGrad([a, b]));
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[i] * b.data[i];
    if (b.requiresGrad) for (let i = 0; i < b.size; i++) b.grad[i] += t.grad[i] * a.data[i];
  };
  return t;
}

export function scale(a: Tensor, k: number): Tensor {
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] * k;
  const t = make(out, a.shape, [a], a.requiresGrad);
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[i] * k;
  };
  return t;
}

export function relu(a: Tensor): Tensor {
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = a.data[i] > 0 ? a.data[i] : 0;
  const t = make(out, a.shape, [a], a.requiresGrad);
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) if (a.data[i] > 0) a.grad[i] += t.grad[i];
  };
  return t;
}

export function tanh(a: Tensor): Tensor {
  const out = new Float64Array(a.size);
  for (let i = 0; i < a.size; i++) out[i] = Math.tanh(a.data[i]);
  const t = make(out, a.shape, [a], a.requiresGrad);
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[i] * (1 - out[i] * out[i]);
  };
  return t;
}

/* ── reductions ──────────────────────────────────────────────────────── */

export function sum(a: Tensor): Tensor {
  let s = 0;
  for (let i = 0; i < a.size; i++) s += a.data[i];
  const t = make(new Float64Array([s]), [1], [a], a.requiresGrad);
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[0];
  };
  return t;
}

export function mean(a: Tensor): Tensor {
  return scale(sum(a), 1 / a.size);
}

/* ── linear algebra ──────────────────────────────────────────────────── */

/** (m,k) @ (k,n) -> (m,n) */
export function matmul(a: Tensor, b: Tensor): Tensor {
  const [m, k] = a.shape;
  const [k2, n] = b.shape;
  if (k !== k2) throw new Error(`matmul: inner dims ${k} != ${k2}`);
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let p = 0; p < k; p++) {
      const av = a.data[i * k + p];
      if (av === 0) continue;
      for (let j = 0; j < n; j++) out[i * n + j] += av * b.data[p * n + j];
    }
  }
  const t = make(out, [m, n], [a, b], needsGrad([a, b]));
  t._backward = () => {
    if (a.requiresGrad) {
      for (let i = 0; i < m; i++)
        for (let p = 0; p < k; p++) {
          let acc = 0;
          for (let j = 0; j < n; j++) acc += t.grad[i * n + j] * b.data[p * n + j];
          a.grad[i * k + p] += acc;
        }
    }
    if (b.requiresGrad) {
      for (let p = 0; p < k; p++)
        for (let j = 0; j < n; j++) {
          let acc = 0;
          for (let i = 0; i < m; i++) acc += a.data[i * k + p] * t.grad[i * n + j];
          b.grad[p * n + j] += acc;
        }
    }
  };
  return t;
}

/** add a (n,) bias to every row of an (m,n) tensor */
export function addBias(a: Tensor, bias: Tensor): Tensor {
  const [m, n] = a.shape;
  if (bias.size !== n) throw new Error(`addBias: bias ${bias.size} != cols ${n}`);
  const out = new Float64Array(a.size);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out[i * n + j] = a.data[i * n + j] + bias.data[j];
  const t = make(out, a.shape, [a, bias], needsGrad([a, bias]));
  t._backward = () => {
    if (a.requiresGrad) for (let i = 0; i < a.size; i++) a.grad[i] += t.grad[i];
    if (bias.requiresGrad)
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) bias.grad[j] += t.grad[i * n + j];
  };
  return t;
}

/* ── constructors ────────────────────────────────────────────────────── */

export function tensor(data: number[] | number[][], requiresGrad = false): Tensor {
  if (Array.isArray(data[0])) {
    const rows = data as number[][];
    const m = rows.length, n = rows[0].length;
    const flat = new Float64Array(m * n);
    for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) flat[i * n + j] = rows[i][j];
    return new Tensor(flat, [m, n], requiresGrad);
  }
  return new Tensor(data as number[], [(data as number[]).length], requiresGrad);
}

/** gaussian init, deterministic when a seed is supplied */
export function randn(shape: Shape, std = 1, seed?: number): Tensor {
  const n = numel(shape);
  const out = new Float64Array(n);
  let s = seed ?? Math.floor(Math.random() * 1e9);
  const rnd = () => {
    s = (s * 1664525 + 1013904223) % 4294967296;
    return s / 4294967296;
  };
  for (let i = 0; i < n; i++) {
    const u = Math.max(rnd(), 1e-12), v = rnd();
    out[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
  }
  return new Tensor(out, shape, true);
}

export function zeros(shape: Shape, requiresGrad = true): Tensor {
  return new Tensor(new Float64Array(numel(shape)), shape, requiresGrad);
}
