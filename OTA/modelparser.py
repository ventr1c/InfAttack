def model_parser(model, dataset):
    if model=='GCN':
        model_parms = {'fastmode' : False,
                       'seed' : 42,
                       'epochs' : 200,
                       'lr' : 0.01,
                       'weight_decay' : 5e-4,
                       'hidden' : 16,
                       'dropout' : 0.5
                       }
        return model_parms


    elif model=='SGC':
        model_parms = {'fastmode': False,
                       'seed': 42,
                       'epochs': 200,
                       'lr': 0.2,
                       'cora_weight_decay': 1.656e-5,
                       'citeseer_weight_decay': 2.3545587e-5,
                       'pubmed_weight_decay': 7.4039e-5,
                       'hidden': 0,
                       'dropout': 0,
                       'degree': 2
                       }
        return model_parms

    else:
        raise NotImplementedError('model: {} is not impletmented'.format(model))