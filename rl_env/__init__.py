from gymnasium.envs.registration import register

# Register the environment so that rl_baseline3_zoo can find and train
# with this environment
register(
    id="SuikaEnv-v0",
    entry_point="rl_env.suika_env:SuikaEnv", 
)

# This env is registered so I can see how the training loop behaves 
# so that I can tweak the code for better training
register(
    id="SuikaEnv-v0-test-setup",
    entry_point="rl_env.suika_env:SuikaEnv",
)