# micro-autograd

A reverse-mode automatic differentiation engine and neural network library written from
scratch in TypeScript. **Zero runtime dependencies.** No PyTorch, no TensorFlow, no wrappers
around someone else's math — the calculus and the backprop are implemented here.

Built to understand what frameworks actually do, and to prove it with numerical gradient
checks rather than assertions.

```bash
npm install
npm test        # 28 checks, including finite-difference gradient verification
npm run xor     # train a small MLP on XOR
npm run spiral  # two interleaved spirals, 99.7% accuracy
npm run bench   # throughput numbers
```

## Why

Calling `loss.backward()` is easy. Knowing that it is correct is a different problem. Every
operation in this library is verified against central finite differences: the analytic
gradient is compared to `(f(x+h) - f(x-h)) / 2h` for every input element, and the test fails
if the relative error exceeds `1e-4`. If a derivative is wrong anywhere, CI goes red.

## Results

| Task | Result |
|---|---|
| XOR (2→8→1 MLP, tanh) | final loss `0.000000`, predictions exact |
| Two spirals (2→64→64→2, ReLU, cross-entropy) | **99.7%** accuracy (299/300) |
| `matmul` 256×256 × 256×256, forward | ~143 ms |
| `matmul` 256×256 × 256×256, forward + backward | ~429 ms |
| `tanh` over 100k elements, forward + backward | ~10 ms |

Measured on a single Node 20 process, no GPU, no SIMD. It is not fast — it is correct and
readable, which is the point.

## How it works

A `Tensor` holds a flat `Float64Array` plus a shape, a gradient buffer of the same size, and
a closure that knows how to push gradients to its inputs.

```ts
export class Tensor {
  data: Float64Array;
  grad: Float64Array;
  shape: Shape;
  _backward: () => void;
  _prev: Tensor[];
}
```

Each operation builds a new tensor and records how to route the gradient backwards. For
example, matrix multiply `C = A @ B` registers:

```
dA = dC @ Bᵀ      dB = Aᵀ @ dC
```

`backward()` does a topological sort of the graph, seeds the output gradient with `1`, then
walks the nodes in reverse calling each `_backward`. Gradients **accumulate** with `+=`, so a
tensor used twice in the graph correctly receives both contributions — there is a test for
exactly that.

## API

```ts
import { tensor, matmul, tanh, sum } from "micro-autograd";

const x = tensor([[1, 2], [3, 4]], true);   // requiresGrad
const y = sum(tanh(matmul(x, tensor([[1, 0], [0, 1]]))));
y.backward();
console.log(x.grad);                        // dL/dx
```

**Ops** — `add`, `sub`, `mul`, `scale`, `matmul`, `addBias`, `relu`, `tanh`, `sum`, `mean`
**Layers** — `Linear` (He init), `ReLU`, `Tanh`, `Sequential`
**Losses** — `mse`, `crossEntropy` (softmax + NLL, fused for stability)
**Optimiser** — `SGD` with momentum
**Constructors** — `tensor`, `randn` (seeded), `zeros`

## Design notes

**Why flat arrays and not nested arrays.** A tensor with shape `[64, 64]` is one contiguous
buffer. Indexing is `i * cols + j`, which keeps the backward passes cache-friendly and makes
the shape bookkeeping explicit instead of implicit in array nesting.

**Why fused cross-entropy.** Computing softmax then taking a log of it loses precision when
logits are large. The implementation subtracts the row maximum before exponentiating, and
derives the gradient directly as `(softmax - onehot) / batch`, which is both stable and
faster than composing separate ops.

**Why a seeded RNG.** `randn` takes an optional seed, so the spiral example and the tests
produce identical numbers on every machine. Reproducible runs are worth more than true
randomness here.

## Limitations, honestly

- **No GPU, no SIMD, no WASM.** Everything is scalar JavaScript.
- **No broadcasting** beyond `addBias`. Shapes must match for elementwise ops.
- **No convolutions, attention or autograd over control flow.**
- **No batching optimiser** — it is plain SGD with momentum, no Adam.
- **`Float64`, not `Float32`.** Slower, but gradients stay precise enough for the
  finite-difference checks to be meaningful.

This is a learning artefact and a correctness demonstration. For real work, use a real
framework.

## License

MIT
