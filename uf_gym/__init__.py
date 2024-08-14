import os

from gymnasium.envs.registration import register

ENV_IDS = []

for task in ["Reach", "Slide", "Push", "PickAndPlace", "Stack", "Flip"]:
    for reward_type in ["sparse", "dense"]:
        for control_type in ["ee", "joints"]:
            reward_suffix = "Dense" if reward_type == "dense" else ""
            control_suffix = "Joints" if control_type == "joints" else ""

            env_id = f"XArm6{task}{control_suffix}{reward_suffix}-v3"

            register(
                id=env_id,
                entry_point=f"uf_gym.envs:XArm6{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100 if task == "Stack" else 50,
            )

            ENV_IDS.append(env_id)

            env_id = f"XArm7{task}{control_suffix}{reward_suffix}-v3"

            register(
                id=env_id,
                entry_point=f"uf_gym.envs:XArm7{task}Env",
                kwargs={"reward_type": reward_type, "control_type": control_type},
                max_episode_steps=100 if task == "Stack" else 50,
            )

            ENV_IDS.append(env_id)
