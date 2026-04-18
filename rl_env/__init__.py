from gymnasium.envs.registration import register

# This registers your env so Gymnasium knows it exists
register(
    id="SuikaEnv-v0",
    # Path format: module_name.file_name:ClassName
    entry_point="rl_env.suika_env:SuikaEnv", 
)