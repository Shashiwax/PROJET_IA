import pandas as pd
from inverter_env import InverterEnv

env = InverterEnv(max_steps=10, random_reset=False, timeout_s=5.0)

rows = []

obs, info = env.reset()
m = info["metrics"]
rows.append({"t": 0, "kind": "reset", **m, "wn": obs[0], "wp": obs[1]})

for t in range(1, 11):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    m = info["metrics"]
    rows.append(
        {
            "t": t,
            "kind": "step",
            "reward": r,
            "terminated": term,
            "truncated": trunc,
            "wn_action": float(a[0]),
            "wp_action": float(a[1]),
            "wn_used": float(m.get("wn_chk", float("nan"))),
            "wp_used": float(m.get("wp_chk", float("nan"))),
            **m,
        }
    )
    if term or trunc:
        break

env.close()

df = pd.DataFrame(rows)
print(df)
df.to_csv("rollout_metrics.csv", index=False)
print("\nSaved: rollout_metrics.csv")
