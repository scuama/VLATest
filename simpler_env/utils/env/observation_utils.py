def get_image_from_maniskill2_obs_dict(env, obs, camera_name=None):
    # obtain image from observation dictionary returned by ManiSkill2 environment
    if camera_name is None:
        # Handle potential wrapper - try direct access first, fallback to unwrapped
        if hasattr(env, 'robot_uid'):
            robot_uid = env.robot_uid
        else:
            robot_uid = env.unwrapped.robot_uid
            
        if "google_robot" in robot_uid:
            camera_name = "overhead_camera"
        elif "widowx" in robot_uid:
            camera_name = "3rd_view_camera"
        else:
            raise NotImplementedError()
    return obs["image"][camera_name]["rgb"]
