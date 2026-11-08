/** Train a small MLP on XOR — the classic non-linearly-separable test. */
import { tensor } from "../src/tensor.js";
import { Linear, Sequential, Tanh, mse } from "../src/nn.js";
import { SGD } from "../src/optim.js";

const X = tensor([[0, 0], [0, 1], [1, 0], [1, 1]]);
const Y = tensor([[0], [1], [1], [0]]);

const net = new Sequential(new Linear(2, 8, 1), new Tanh(), new Linear(8, 1, 2));
const opt = new SGD(net.parameters(), 0.3, 0.9);

for (let epoch = 1; epoch <= 2000; epoch++) {
  const loss = mse(net.forward(X), Y);
  opt.zeroGrad(net.parameters());
  loss.backward();
  opt.step(net.parameters());
  if (epoch % 200 === 0 || epoch === 1) {
    console.log(`epoch ${String(epoch).padStart(4)}   loss ${loss.data[0].toFixed(6)}`);
  }
}

console.log("\ninput   target   prediction");
const pred = net.forward(X);
for (let i = 0; i < 4; i++) {
  console.log(`  ${X.data[i * 2]},${X.data[i * 2 + 1]}      ${Y.data[i]}       ${pred.data[i].toFixed(4)}`);
}
