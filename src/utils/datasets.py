import numpy as np
import torch
import torch.utils.data as data_utils




def get_fromdataframe(df, batch_size, standardize=True, seed=123):
    """
    :param df: dataframe index: (int), columns: (time (list) | values (list) | apoe_all1 (int) | apoe_all2 (int))
    :param batch_size: batch size for data generator
    :param standardize: boolean for standardization (based on train statistics only)
    :param nobs: minimal number of visits per subject (default:3)
    :param seed: random seed for reproducibility
    :return: data_loaders and mean|variance standardizations values (if standardize=True, else None)
    """

    np.random.seed(seed)
    ratio_train, ratio_val = .7, .1
    assert 1 - (ratio_train + ratio_val) >= .05
    format_ = 'online'
    assert format_ in ['discretized', 'online']
    assert format_ == 'online', "only online version for now (due to very few visits)"

    # Randomization of dataframe (safety check)
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    max_lenseq = np.max([len(row[0]) for _, row in df.iterrows()])

    # Sequence-wise generation
    points = []
    maskers = []
    times = []
    labels = []
    nb_bins = max_lenseq

    for _, row in df.iterrows():
        # DATA GENERATION
        rowt, rowy, rowapoe_all1, rowapoe_all2 = row[0], row[1], row[2], row[3]
        reference_time = torch.from_numpy(np.array(rowt + [rowt[-1] ] *(nb_bins -len(rowt))))
        masker = torch.from_numpy(np.array([1 ] *len(rowt) + [0 ] *(nb_bins -len(rowt))).astype(int))
        fns = torch.from_numpy(np.array(rowy + [rowy[-1] ] *(nb_bins -len(rowy))))
        label = torch.from_numpy(np.array([rowapoe_all1, rowapoe_all2]))

        # GET AS MASKERS
        times.append(reference_time)
        points.append(fns)
        maskers.append(masker)
        labels.append(label)

    times = torch.stack(times).type(torch.float)
    points = torch.stack(points).type(torch.float)
    maskers = torch.stack(maskers).type(torch.float)  # | torch.int16
    labels = torch.stack(labels).type(torch.float)  # | torch.int16

    n_get = len(points)
    nb_train = int(n_get * ratio_train)
    nb_val = int(n_get * ratio_val)
    nb_test = n_get - nb_train - nb_val
    print('Train : {:d} | Validation {:d} | Test {:d}'.format(nb_train, nb_val, nb_test))

    # Dataset splits
    index_train = torch.linspace(0, nb_train - 1, nb_train).type(torch.long)
    index_val = torch.linspace(0, nb_val - 1, nb_val).type(torch.long)
    index_test = torch.linspace(0, nb_test - 1, nb_test).type(torch.long)
    maskers_train = maskers[:nb_train]
    maskers_val = maskers[nb_train + 1:nb_train + nb_val + 1]
    maskers_test = maskers[-nb_test:]
    times_train = times[:nb_train]
    times_val = times[nb_train + 1:nb_train + nb_val + 1]
    times_test = times[-nb_test:]
    x_train = points[:nb_train]
    x_val = points[nb_train + 1:nb_train + nb_val + 1]
    x_test = points[-nb_test:]
    labels_train = labels[:nb_train]
    labels_val = labels[nb_train + 1:nb_train + nb_val + 1]
    labels_test = labels[-nb_test:]

    # Standard normalization of data :
    if standardize:
        x_mean, x_std = torch.mean(x_train), torch.std(x_train)
        x_train = (x_train - x_mean) / x_std
        x_val = (x_val - x_mean) / x_std
        x_test = (x_test - x_mean) / x_std
        x_all = (points - x_mean) / x_std

        times_mean, times_std = torch.mean(times_train), torch.std(times_train)
        times_train = (times_train - times_mean) / times_std
        times_val = (times_val - times_mean) / times_std
        times_test = (times_test - times_mean) / times_std
        times_all = (times - times_mean) / times_std
    else:
        x_mean, x_std = None, None
        times_mean, times_std = None, None
        times_all = times
        x_all = points

    # pytorch custom data loader
    train = data_utils.TensorDataset(index_train, maskers_train, times_train, x_train, labels_train)
    train_loader = data_utils.DataLoader(train, batch_size=batch_size, shuffle=True)

    validation = data_utils.TensorDataset(index_val, maskers_val, times_val, x_val, labels_val)
    val_loader = data_utils.DataLoader(validation, batch_size=batch_size, shuffle=True)

    test = data_utils.TensorDataset(index_test, maskers_test, times_test, x_test, labels_test)
    test_loader = data_utils.DataLoader(test, batch_size=batch_size, shuffle=True)

    index_all = torch.linspace(0, n_get -1, n_get).type(torch.long)
    # print(index_train.shape, index_all.shape)
    all_data = data_utils.TensorDataset(index_all, maskers, times_all, x_all, labels)
    all_data_loader = data_utils.DataLoader(all_data, batch_size=batch_size, shuffle=True)

    return train_loader, val_loader, test_loader, all_data_loader, (times_mean, times_std), (x_mean, x_std)