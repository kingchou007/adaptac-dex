import torch
import MinkowskiEngine as ME


def preprocess_data(data, args, device):
    cloud_coords = data["input_coords_list"]
    cloud_feats = data["input_feats_list"]
    action_data = data["action_normalized"]

    if args.enable_tactile:
        tactile_data = data["tactile"]
        tactile_tensor = torch.stack(tactile_data).to(device)
    else:
        tactile_tensor = None

    if args.num_force_prediction > 0:
        net_force = data["net_force"]
        net_force = torch.stack(net_force).to(device)
    else:
        net_force = None

    if args.obs_net_force:
        obs_net_force = data["obs_net_force"]
        obs_net_force = torch.stack(obs_net_force).to(device)
    else:
        obs_net_force = None

    cloud_feats, cloud_coords, action_data = (
        cloud_feats.to(device),
        cloud_coords.to(device),
        action_data.to(device),
    )
    cloud_data = ME.SparseTensor(cloud_feats, cloud_coords)
    return cloud_data, action_data, tactile_tensor, net_force, obs_net_force
