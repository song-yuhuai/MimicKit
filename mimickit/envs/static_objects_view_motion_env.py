import envs.view_motion_env as view_motion_env   # adjust name if different
import engines.engine as engine
import numpy as np

class StaticObjectsViewMotionEnv(view_motion_env.ViewMotionEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize):
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize)
        return

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        self._build_static_object(env_id, config)
        return

    def _build_static_object(self, env_id, env_config):
        objs_config = env_config.get("objects", [])
        if len(objs_config) == 0:
            return

        color = np.array([0.3, 0.3, 0.3])

        for i, obj_config in enumerate(objs_config):
            asset_file = obj_config["file"]
            pos = np.array(obj_config["pos"], dtype=np.float32)
            rot = np.array(obj_config.get("rot", [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

            obj_name = f"static_object{i}"
            self._engine.create_obj(
                env_id=env_id,
                obj_type=engine.ObjType.rigid,
                asset_file=asset_file,
                name=obj_name,
                start_pos=pos,
                start_rot=rot,
                fix_root=True,
                color=color,
            )
        return
