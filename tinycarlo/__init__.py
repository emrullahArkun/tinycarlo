from gymnasium.envs.registration import register

register(id="tinycarlo-v2", entry_point="tinycarlo.env:TinyCarloEnv")
register(id="tinycarlo-realworld-v2", entry_point="tinycarlo.real_world.env:TinyCarloRealWorldEnv")
