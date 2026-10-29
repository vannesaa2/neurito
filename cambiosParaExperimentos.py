import numpy as np

ACTIONS = ("COMER", "DORMIR", "JUGAR")

def policy(X):
    return np.argmax(np.column_stack((X[:, 0], 1 - X[:, 1], X[:, 2])), axis=1)

def relu(v):
    return np.maximum(0, v)

def softmax(v):
    e = np.exp(v - np.max(v, axis=1, keepdims=True))
    return e / np.sum(e, axis=1, keepdims=True)

class Net:
    def __init__(self, hidden=6, seed=42):
        rs = np.random.RandomState(seed)
        self.X = rs.rand(3000, 3)
        self.y = policy(self.X)
        self.W1 = rs.randn(3, hidden) * 0.5
        self.b1 = np.zeros((1, hidden))
        self.W2 = rs.randn(hidden, 3) * 0.5
        self.b2 = np.zeros((1, 3))
        self.rs = rs
        self.losses = []

    def predict(self, d):
        return softmax(relu(d @ self.W1 + self.b1) @ self.W2 + self.b2)

    def train(self, epochs=1200, lr=0.1):
        X, t = self.X, self.y
        n = len(X)
        for _ in range(epochs):
            z1 = X @ self.W1 + self.b1
            h = relu(z1)
            p = softmax(h @ self.W2 + self.b2)
            loss = -np.mean(np.log(p[np.arange(n), t] + 1e-10))
            self.losses.append(loss)
            d2 = p.copy()
            d2[np.arange(n), t] -= 1
            d2 /= n
            gW2 = h.T @ d2
            gb2 = d2.sum(0, keepdims=True)
            d1 = (d2 @ self.W2.T) * (z1 > 0)
            gW1 = X.T @ d1
            gb1 = d1.sum(0, keepdims=True)
            self.W1 -= lr * gW1
            self.b1 -= lr * gb1
            self.W2 -= lr * gW2
            self.b2 -= lr * gb2
        return self

    def acc(self, d, y):
        return float(np.mean(np.argmax(self.predict(d), axis=1) == y))

def fresh_test(net):
    t = net.rs.rand(1000, 3)
    return t, policy(t)

print("=== BASE (6 neuronas, 1200 epocas, lr 0.1, seed 42) ===")
base = Net().train()
Xt, yt = fresh_test(base)
print(f"loss final {base.losses[-1]:.4f} | train acc {base.acc(base.X, base.y):.2%} | test acc {base.acc(Xt, yt):.2%}")
print("distribucion de clases (train):", np.bincount(base.y) / len(base.y))

print("\n=== EXP 1: dos necesidades fijas, varia la tercera ===")

def sweep(fixed, idx, label):
    prev = None
    rows = []
    for v in np.round(np.arange(0, 1.0001, 0.01), 2):
        n = list(fixed)
        n[idx] = v
        a = int(np.argmax(base.predict(np.array([n]))[0]))
        t = int(policy(np.array([n]))[0])
        if a != prev:
            rows.append((v, ACTIONS[a], ACTIONS[t]))
            prev = a
    print(f"-- {label}, fijas {fixed}")
    for v, a, t in rows:
        print(f"   desde {label}={v:.2f}: red={a} (politica real={t})")


sweep([0.0, 0.5, 0.3], 0, "hambre")     
sweep([0.3, 0.0, 0.3], 1, "energia")     
sweep([0.3, 0.5, 0.0], 2, "aburrimiento")
sweep([0.2, 0.9, 0.0], 2, "aburrimiento")  

print("\n=== EXP 2: prioridades casi iguales ===")
casos = [
    [0.50, 0.50, 0.50], [0.52, 0.50, 0.48], [0.55, 0.50, 0.45],
    [0.60, 0.50, 0.40], [0.80, 0.50, 0.20], [0.51, 0.49, 0.50],
    [0.70, 0.30, 0.70], [0.90, 0.10, 0.10],
]
for c in casos:
    p = base.predict(np.array([c]))[0]
    pr = [c[0], 1 - c[1], c[2]]
    print(f"hambre={c[0]:.2f} energia={c[1]:.2f} aburr={c[2]:.2f} | prioridades={np.round(pr,2)} | softmax={np.round(p,3)} | red={ACTIONS[int(np.argmax(p))]} real={ACTIONS[int(np.argmax(pr))]}")

print("\n=== EXP 3: neuronas ocultas (media de 5 semillas) ===")
for h in (3, 6, 10, 20):
    L, A, T = [], [], []
    for s in range(5):
        n = Net(h, s).train()
        a, b = fresh_test(n)
        L.append(n.losses[-1]); A.append(n.acc(n.X, n.y)); T.append(n.acc(a, b))
    n42 = Net(h, 42).train(); a, b = fresh_test(n42)
    print(f"h={h:2d} | seed42: loss {n42.losses[-1]:.4f} train {n42.acc(n42.X,n42.y):.2%} test {n42.acc(a,b):.2%} | media5: loss {np.mean(L):.4f} (+-{np.std(L):.4f}) test {np.mean(T):.2%} (+-{np.std(T):.2%}) | params={3*h+h+h*3+3}")

print("\n=== EXP 4a: epocas (lr 0.1) ===")
for ep in (50, 200, 600, 1200, 3000, 10000):
    n = Net().train(ep); a, b = fresh_test(n)
    print(f"epocas={ep:5d} | loss {n.losses[-1]:.4f} | train {n.acc(n.X,n.y):.2%} | test {n.acc(a,b):.2%}")

print("\n=== EXP 4b: tasa de aprendizaje (1200 epocas) ===")
for lr in (0.001, 0.01, 0.1, 0.5, 1, 2, 5, 20, 50):
    n = Net().train(1200, lr); a, b = fresh_test(n)
    L = np.array(n.losses)
    osc = int(np.sum(np.diff(L) > 1e-9))
    print(f"lr={lr:6} | loss {L[-1]:.4f} (min {L.min():.4f}) | subidas de loss {osc} | train {n.acc(n.X,n.y):.2%} | test {n.acc(a,b):.2%}")

print("\n=== EXP 6: precision enganosa ===")
yp = np.argmax(base.predict(Xt), axis=1)
print("matriz de confusion (filas real, cols pred):\n", np.array([[np.sum((yt == i) & (yp == j)) for j in range(3)] for i in range(3)]))
pr = np.column_stack((Xt[:, 0], 1 - Xt[:, 1], Xt[:, 2]))
s = np.sort(pr, axis=1)
margin = s[:, 2] - s[:, 1]
for lo, hi in ((0, 0.02), (0.02, 0.05), (0.05, 0.15), (0.15, 1)):
    m = (margin >= lo) & (margin < hi)
    print(f"margen entre las 2 prioridades mayores [{lo},{hi}): n={m.sum():4d} acc={np.mean(yp[m]==yt[m]):.2%}")
# distribucion distinta a la de entrenamiento
Xs = np.random.RandomState(7).rand(1000, 3) * 0.3 + np.array([0.35, 0.35, 0.35])
print(f"acc en estados cercanos al centro (rango 0.35-0.65): {base.acc(Xs, policy(Xs)):.2%}")
Xo = np.random.RandomState(8).rand(1000, 3) * 3 - 1  # fuera de [0,1]
print(f"acc con entradas fuera de rango [-1,2]: {base.acc(Xo, policy(Xo)):.2%}")
# la politica real tiene su 'estado calmo' que la red no ve
calm = (Xt[:, 0] < 0.4) & (Xt[:, 1] > 0.6) & (Xt[:, 2] < 0.4)
print(f"fraccion de test en region 'CALMA' (regla de pantalla): {calm.mean():.2%}; la red nunca puede responder CALMA")
print("clase mayoritaria como baseline:", f"{np.max(np.bincount(yt))/len(yt):.2%}")
