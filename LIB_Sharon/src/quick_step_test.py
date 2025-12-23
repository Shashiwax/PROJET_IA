from inverter_env import InverterEnv

env = InverterEnv(max_steps=10, random_reset=False)
obs, info = env.reset()
print("RESET ok")

for k in range(10):
    a = env.action_space.sample()
    obs, r, term, trunc, info = env.step(a)
    print(k+1, "reward", r, "term", term, "trunc", trunc)
    if term or trunc:
        break

env.close()
