from gymnasium.envs.registration import register

# Added kwargs to the register functions because sometimes after running `pip install -e .` in `SuikaReinforcement/`
# training would crash in rl-baseline3-zoo due to the observation space dimensions being mismatched when the first eval happens
# this would happen if I toggled spatial_features or clustering on in the training conf yaml sometimes and registering these
# static values for those settings seemed to help sometimes but it's unclear when the issue happens or doesn't

# Register the environment so that rl_baseline3_zoo can find and train
# with this environment
register(
    id="SuikaEnv-v0",
    entry_point="rl_env.suika_env:SuikaEnv",
    kwargs={"spatial_features": False, "clustering": False}
)

# This env is registered so I can see how the training loop behaves 
# so that I can tweak the code for better training
register(
    id="SuikaEnv-v0-test-setup",
    entry_point="rl_env.suika_env:SuikaEnv",
    # kwargs={"spatial_features": False, "clustering": False}
)