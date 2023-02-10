from .edcoder import PreModel


def build_model(args):
    num_heads = args.num_heads
    num_out_heads = args.num_out_heads
    num_hidden = args.num_hidden
    num_layers = args.num_layers
    attn_drop = args.attn_drop
    in_drop = args.dropout
    norm = args.norm
    negative_slope = args.negative_slope
    encoder_type = args.encoder
    decoder_type = args.decoder
    mask_node_rate = args.mask_node_rate
    mask_feature_rate = args.mask_feature_rate

    activation = args.activation
    loss_fn = args.loss_fn
    concat_hidden = args.concat_hidden
    num_features = args.num_features
    latent_dim = args.latent_dim
    pe = args.pe
    drop_node_rate = args.drop_node_rate
    objective = args.objective
    standardscale = args.standardscale
    cat_pe = args.cat_pe
    pe_aug = args.pe_aug
    aggr = args.aggr
    
    model = PreModel(
        in_dim=num_features,
        num_hidden=num_hidden,
        num_layers=num_layers,
        nhead=num_heads,
        nhead_out=num_out_heads,
        activation=activation,
        feat_drop=in_drop,
        attn_drop=attn_drop,
        negative_slope=negative_slope,
        encoder_type=encoder_type,
        decoder_type=decoder_type,
        mask_node_rate=mask_node_rate,
        mask_feature_rate=mask_feature_rate,
        norm=norm,
        loss_fn=loss_fn,
        concat_hidden=concat_hidden,
        latent_dim=latent_dim,
        pe = pe,
        drop_node_rate = drop_node_rate,
        objective = objective,
        standardscale = standardscale,
        cat_pe = cat_pe,
        pe_aug = pe_aug,
        aggr = aggr,
    )
    return model
