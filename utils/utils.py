def save_confusion_matrix(cm, path2save, ordered_names):
    import seaborn as sns
    sns.set(font_scale=5)
    
    template_path = os.path.join(path2save, "{}.svg")
    #PRECISION
    cmn = cm.astype('float') / cm.sum(axis=-1)[:, np.newaxis]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    g = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20})
    #g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path_precision = template_path.format("precision")
    plt.savefig(path_precision, format="svg")

    #RECALL
    cmn = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :]
    cmn[np.isnan(cmn) | np.isinf(cmn)] = 0
    fig, ax = plt.subplots(figsize=(31, 31))
    g = sns.heatmap(cmn, annot=True, fmt='.2f', xticklabels=ordered_names, yticklabels=ordered_names, annot_kws={"size": 20})
    #g.set_xticklabels(g.get_xticklabels(), rotation = 35, fontsize = 20)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    path_recall = template_path.format("recall")
    plt.savefig(path_recall, format="svg")