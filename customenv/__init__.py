from gym.envs.registration import register

register(
    id="MyEnv-v001",
    entry_point="customenv.myenv:MyEnv"
)
