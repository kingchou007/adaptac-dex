import torch
import torch.nn.functional as F
from torch import nn

from adaptac.model.tactile.tactile_utils import data_to_gnn_batch, create_activation
from adaptac.model.tactile.gnn.gat import MAEGATConv


class TactileMLP(nn.Module):
    """
    Simple multi-layer perceptron with ELU activation.
    Reference: https://arxiv.org/abs/2404.16823
    """

    def __init__(self, units, input_size):
        super().__init__()
        layers = []
        for output_size in units:
            layers.append(nn.Linear(input_size, output_size))
            layers.append(nn.ELU())
            input_size = output_size
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class MAEGATNet(nn.Module):
    """
    Multi-head Attention Enhanced Graph Attention Network.
    Reference: https://arxiv.org/pdf/2409.17549
    """
    
    def __init__(
        self,
        input_channel=6,
        num_hidden=64,
        output_channel=32,
        num_layers=3,
        nhead=4,
        nhead_out=4,
        activation="prelu",
        feat_drop=0.2,
        attn_drop=0.1,
        negative_slope=0.2,
        residual=False,
        norm=nn.Identity,
        concat_out=True,
        encoding=True,
        pretrained=True,
        output_net="mlp",
        edge_type="four",
    ):
        super().__init__()
        self.output_channel = output_channel * nhead_out if concat_out else output_channel
        self.num_layers = num_layers
        self.edge_type = edge_type
        self.feat_drop = feat_drop
        
        activation_fn = create_activation(activation)
        last_activation = activation_fn if encoding else None
        last_residual = encoding and residual
        last_norm = norm if encoding else None

        self.gat_layers = nn.ModuleList()
        
        if num_layers == 1:
            self.gat_layers.append(
                MAEGATConv(
                    input_channel,
                    output_channel,
                    nhead_out,
                    concat=concat_out,
                    negative_slope=negative_slope,
                    dropout=attn_drop,
                    residual=last_residual,
                    norm=last_norm,
                )
            )
        else:
            self.gat_layers.append(
                MAEGATConv(
                    input_channel,
                    num_hidden,
                    nhead,
                    concat=concat_out,
                    negative_slope=negative_slope,
                    dropout=attn_drop,
                    activation=activation_fn,
                    residual=residual,
                    norm=norm,
                )
            )
            
            hidden_input_dim = num_hidden * nhead if concat_out else num_hidden
            for _ in range(1, num_layers - 1):
                self.gat_layers.append(
                    MAEGATConv(
                        hidden_input_dim,
                        num_hidden,
                        nhead,
                        concat=concat_out,
                        negative_slope=negative_slope,
                        dropout=attn_drop,
                        activation=activation_fn,
                        residual=residual,
                        norm=norm,
                    )
                )
            
            self.gat_layers.append(
                MAEGATConv(
                    hidden_input_dim,
                    output_channel,
                    nhead_out,
                    concat=concat_out,
                    negative_slope=negative_slope,
                    dropout=attn_drop,
                    activation=last_activation,
                    residual=last_residual,
                    norm=last_norm,
                )
            )

        self.head = nn.Identity()

    def forward(self, data, ori_edge_index=None, return_hidden=False):
        """Forward pass through the network."""
        if isinstance(data, torch.Tensor) and ori_edge_index is None:
            batch_data, num_batch, num_nodes, _ = data_to_gnn_batch(data, self.edge_type)
            x, edge_index = batch_data.x, batch_data.edge_index
        else:
            x = data
            edge_index = ori_edge_index
            num_batch = None
            num_nodes = None

        h = x
        hidden_list = []
        for layer in self.gat_layers:
            h = F.dropout(h, p=self.feat_drop, training=self.training)
            h = layer(h, edge_index)
            hidden_list.append(h)

        output = self.head(h)

        if return_hidden:
            return output, hidden_list
        
        if ori_edge_index is None:
            return output.reshape(num_batch, num_nodes, self.output_channel)
        return output


class MAEGAT(nn.Module):
    """Wrapper for MAEGATNet with pretrained model loading support."""

    def __init__(
        self,
        input_channel=12,
        num_hidden=32,
        output_channel=32,
        num_layers=3,
        nhead=4,
        nhead_out=4,
        activation="prelu",
        feat_drop=0.0,
        attn_drop=0.0,
        negative_slope=0.2,
        residual=False,
        norm=nn.Identity,
        concat_out=True,
        encoding=True,
        pretrained=True,
        output_net="mlp",
        edge_type="four",
        pretrain_ckpt_path=None,
        device=None,
    ):
        super().__init__()
        if encoding:
            assert output_channel == num_hidden, "For encoding, output_channel must equal num_hidden"

        self.nets = MAEGATNet(
            input_channel=input_channel,
            num_hidden=num_hidden,
            output_channel=output_channel,
            num_layers=num_layers,
            nhead=nhead,
            nhead_out=nhead_out,
            activation=activation,
            feat_drop=feat_drop,
            attn_drop=attn_drop,
            negative_slope=negative_slope,
            residual=residual,
            norm=norm,
            concat_out=concat_out,
            encoding=encoding,
            pretrained=pretrained,
            output_net=output_net,
            edge_type=edge_type,
        )
        
        if pretrained and encoding:
            if pretrain_ckpt_path is None:
                raise ValueError("pretrain_ckpt_path must be provided when pretrained=True")
            self.nets.load_state_dict(
                torch.load(pretrain_ckpt_path, map_location="cpu")
            )

    def output_shape(self, input_shape):
        """Compute output shape from input shape."""
        return [input_shape[0], self.nets.output_channel]

    def forward(self, data, ori_edge_index=None, return_hidden=False):
        return self.nets(data, ori_edge_index, return_hidden)


class DimReducer(nn.Module):
    """Linear dimension reduction layer."""

    def __init__(self, input_dim=120 * 128, output_dim=512):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)