def plot_tensorboard_csv(csv_file={}, smooth_weight=0, title="tensorboard-plot", ylabel="loss", steps_or_runtime='steps', is_save=0):
    fig = plt.figure()
    plt.title(title)
    plt.ylabel(ylabel)
    for file_type in list(csv_file.keys()):
        file = pd.read_csv(csv_file[file_type])
        if steps_or_runtime == 'steps':
            x = file['Step']
            plt.xlabel('epoch')
        else:
            x = list(file['Wall time'])
            x = [i-x[0] for i in x]
            plt.xlabel('runtime/s')
        scalar = file['Value']
        last = scalar[0]
        smoothed_value = []
        for point in scalar:
            smoothed_val = last * smooth_weight + (1 - smooth_weight) * point
            smoothed_value.append(smoothed_val)
            last = smoothed_val
        plt.plot(x, smoothed_value, label=file_type)
        if is_save:
            plt.savefig(title+'.png')
            
    plt.legend()
