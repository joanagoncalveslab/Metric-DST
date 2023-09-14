import argparse
import platform
import os, sys, random, gc
project_path = '.'
path2this = os.path.dirname(os.path.abspath(__file__)).split('/')
for i, folder in enumerate(path2this):
    if folder.lower() == 'dssl4sl':
        project_path = '/'.join(path2this[:i + 1])
sys.path.insert(0, project_path)

import pandas as pd
import numpy as np
#import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
FIG_SIZE = [4.3,2]
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['figure.figsize'] = FIG_SIZE
plt.rc('font', size=6)
plt.rc('axes', titlesize=6)
plt.rc('axes', labelsize=6)
plt.rc('xtick', labelsize=5)
plt.rc('ytick', labelsize=5)
plt.rc('legend', fontsize=5.5)


from src import config

#RAW_METHOD_NAMES = ['supervised', 'true_semi_supervised', 'balanced_semi_supervised', 'diversity']
#FIXED_METHOD_NAMES = ['Supervised (ML+KNN)', 'Self-Training (ST)', 'Balanced ST (BST)', 'Diverse BST (DBST)']
SCORES = {'test_loss': 'Loss', 'accuracy': 'Accuracy', 'f1_score': 'F1 Score', 'average_precision': 'AUPRC', 'auroc': 'AUROC'}
RAW_METHOD_NAMES = ['supervised_none', 'supervised_random', 'supervised', 'true_semi_supervised', 'diversity'] #'supervised_none', 'supervised_random', 
FIXED_METHOD_NAMES = ['No Bias - Supervised (ML+KNN)', 'Random - Supervised (ML+KNN)','Bias - Supervised (ML+KNN)', 'Bias - Vanilla Self-training (Metric-ST)', 'Bias - Diversity-guided ST (Metric-DST)'] #'No Bias - Supervised (ML+KNN)', 'Random - Supervised (ML+KNN)', 
#RAW_METHOD_NAMES, FIXED_METHOD_NAMES = RAW_METHOD_NAMES[2:], FIXED_METHOD_NAMES[2:]
ALL_DATASETS = ['fire', 'pistachio', 'pumpkin', 'raisin', 'rice', 'spam', 'adult', 'breast_cancer']#fire
DIMS = [16, 32, 64, 128]#, 256, 512, 1024]
N_CLUSTERS = [2, 3]
RIS = {1.0: 100, 0.8: 80}
BIAS_SIZES = [30, 50, 100, 200]
CONVS = {'all': 'All'}
D_SYMBOL0 = r'$\Delta_0$'
D_SYMBOL1 = r'$\Delta_1$'
for adataset in ALL_DATASETS:
    CONVS[adataset] = adataset.replace('_', ' ').capitalize()
for bias_size in BIAS_SIZES:
    CONVS[f'hierarchyy_0.9_{float(bias_size)}'] = f'Hierarchy (0.9, {bias_size*2})'
    CONVS[f'hierarchyy_0.9_{float(bias_size)}_title'] = f'Hierarchy (0.9) Bias - {bias_size*2} Selections'
    CONVS[f'concept_1000|{bias_size}|200_2|2'] = f'Concept Shift ({bias_size})'
    CONVS[f'delta_{bias_size}|2|0,0|0,0'] = f'{bias_size*2} Samples\n{D_SYMBOL0}: (0,0)\n{D_SYMBOL1}: (0,0)'
    CONVS[f'delta_{bias_size}|2|1,0.5|0,0'] = f'{bias_size*2} Samples\n{D_SYMBOL0}: (1,0.5)\n{D_SYMBOL1}: (0,0)'
    CONVS[f'delta_{bias_size}|2|0,0|0,0_title'] = f'Moon Dataset - '
    #delta_100|2|0,0|0,0
for dim in DIMS:
    for n_c in N_CLUSTERS:
        for ri_f, ri_p in RIS.items():
            CONVS[f'custom_ns2000_nc{n_c}_nf_ri{ri_f}_rs0_fy0_cs1.5'] = f'{n_c} clusters with {ri_p}% Informative Features'
            CONVS[f'custom_ns2000_nc{n_c}_nf_ri_rs0_fy0_cs1.5'] = f'2 Classes, each class with {n_c} clusters'
            CONVS[f'custom_ns2000_nc{n_c}_nf{dim}_ri{ri_f}_rs0_fy0_cs1.5'] = f'{dim} Dimension\n{ri_p}% Informative'
            CONVS[f'moon_2000'] = f'Moon'

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    mainOutputFolder = config.RESULT_DIR / 'bias'
else:
    webDriveFolder = "/home/nfs/ytepeli/python_projects/msc-thesis-2122-mathijs-de-wolf/data"
    mainOutputFolder = config.RESULT_DIR / 'bias'

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run metric learning pipeline"
    )
    parser.add_argument("--batch-size", "-bs", type=int, default=64)
    parser.add_argument('--dataset', "-d", default='all')
    parser.add_argument("--output-file", "-of", type=str, default=None)
    parser.add_argument("--experiment_id", "-eid", type=int, default=-1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--lr", type=float, default=0.01)

    parser.add_argument("--knn", type=int, default=10)
    parser.add_argument("--conf", "-c", type=float, default=0.9)
    parser.add_argument("--num-pseudolabels", type=int, default=-2)

    parser.add_argument("--retrain", "-rt", action='store_true')
    parser.add_argument("--early_stop_pseudolabeling", "-esp", action='store_true')
    parser.add_argument("--single-fold", action='store_true')
    parser.add_argument("--model", "-m", choices=['supervised', 'balanced_semi_supervised', 'true_semi_supervised', 'diversity', 'all'], default='all')
    parser.add_argument("--bias", "-b", default='hierarchyy_0.9')
    parser.add_argument("--balance", "-bal", default='none')
    parser.add_argument("--output_d_folder", "-odf", default=None)
    parser.add_argument("--output_b_folder", "-obf", default=None)
    
    return parser

def add_median_labels(ax, lineno=4, fmt='.2f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[lineno:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center', color='black', fontsize=3)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=0.2, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def get_results(bias_name, dataset_name, experiment_file):
    res_pd_list = []
    res_folder = mainOutputFolder / bias_name / dataset_name / f'{experiment_file}'
    print(res_folder)
    bias_name = bias_name.split('/')[-1]
    for raw_m_name, fix_m_name in zip(RAW_METHOD_NAMES, FIXED_METHOD_NAMES):
        try:
            tmp_res_loc = res_folder / f'{raw_m_name}_test_performance.csv'
            tmp_res = pd.read_csv(tmp_res_loc)
            tmp_res['Bias'] = CONVS[bias_name]
            tmp_res['Dataset'] = CONVS[dataset_name] 
            tmp_res['Bias-Dataset'] = CONVS[bias_name]+' --- '+CONVS[dataset_name]   
            tmp_res['Experiment Id'] = experiment_file
            tmp_res['Method'] = fix_m_name
            res_pd_list.append(tmp_res)
        except Exception as e:
            print(f'{fix_m_name} for {CONVS[bias_name]}, {CONVS[dataset_name]} not run due to {e}')
    full_res = pd.concat(res_pd_list, ignore_index = True)
    return full_res


def draw_boxplot(res, outputFolder, title_col, y_axis_col, x_axis_col):
    """Visualize the res (pandas df) data with the score suggested and saves it in loc.
    Args:
        loc (str or Path): Location for saving the figure
        res (pd.DataFrame): The data to visualize
        score (str): The name of the score.
    """
    #print(res)
    plt.clf()
    #sns.set_style("ticks")
    #sns.set_context("paper")
    plt.rc('text', color='black')
    plt.rc('font', size=6)
    plt.rc('axes', titlesize=6)
    plt.rc('axes', labelsize=6)
    plt.rc('axes', linewidth=0.5)
    plt.rc('axes', labelpad=1.1)
    plt.rc('xtick', labelsize=5)
    plt.rc('ytick', labelsize=5)
    plt.rc('xtick.major', pad=1)
    plt.rc('ytick.major', pad=1)
    plt.rc('legend', fontsize=5.5)
    PROPS = {
        'boxprops': {'edgecolor': 'black', 'linewidth': 0.3},#'facecolor': 'none'
        'medianprops': {'color': 'white', 'linewidth':0.3},
        'meanprops': {"marker":"o",
                       "markerfacecolor":"white", 
                       "markeredgecolor":"black",
                      "markersize":1.2,
                      "markeredgewidth":0.1},
        'whiskerprops': {'color': 'black', 'linewidth': 0.3},
        'capprops': {'color': 'black', 'linewidth': 0.3}
    }
    #palette=["#D6604D", "#B2182B", "#5AAE61", "#1B7837", "#9970AB", "#762A83"]#"PRGn"
    palette=['#882E72', '#1965B0', '#4EB265', '#EE8026', '#DC050C']
    if len(np.unique(res['Method'].values))<5:
        palette = palette[5-len(np.unique(res['Method'].values)):]
    g = sns.boxplot(data=res, x=x_axis_col, y=f"{y_axis_col}", hue="Method", palette=palette, width=0.5, hue_order=FIXED_METHOD_NAMES, whis=20, showmeans=True, **PROPS)
    g.legend_.set_title(None)
    #sns.swarmplot(data=res, x="Cancer", y=f"{score}", hue="Method", palette=palette, hue_order=hue_order)
    # plt.legend()#loc='upper left')
    sns.despine(offset=0)
    add_median_labels(g, 2)
    add_median_labels(g, 3)
    add_median_labels(g, 4)
    plt.title(title_col, fontsize = 6.5)
    # for scoret, scored in res.items():
    #     scored.boxplot(grid=False, label=scoret)
    for d_id in range(res[x_axis_col].nunique()-1):
        plt.axvline(d_id+0.5, ls='--', color='black', lw=0.3)
    if 'loss' in y_axis_col:
        plt.ylim((0, 1.0))
    else:
        plt.ylim((0.2, 1.0))
    plt.tick_params(axis ='both', width=0.5)
    plt.ylabel(SCORES[y_axis_col])
    for out_fmt in ['png', 'pdf']:
        out_loc = outputFolder / f'{y_axis_col}_report.{out_fmt}'
        plt.savefig(out_loc, dpi=300, bbox_inches='tight')

def prepare_main():
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    
    experiment_args = {'bs': args.batch_size, 'rs':args.seed, 'lr': args.lr, 'knn':args.knn, 'c':args.conf, 
                       'kb': args.num_pseudolabels, 'bal': args.balance}
    if args.retrain:
        experiment_args['rt']=''
    if args.early_stop_pseudolabeling:
        experiment_args['esp']=''
    
    experiment_str = 'experiment-' + "_".join([f'{exp_k}{exp_v}' for exp_k, exp_v in experiment_args.items()])+'-0'
    if args.output_d_folder is None:
        output_d_folder = args.dataset
    else:
        output_d_folder = args.output_d_folder
    if args.output_b_folder is None:
        output_b_folder = args.bias
    else:
        output_b_folder = args.output_b_folder
    outputFolder = mainOutputFolder / output_b_folder / output_d_folder / experiment_str / 'result_vis'
    config.ensure_dir(outputFolder)
    config.safe_create_dir(outputFolder)
    
    return args, experiment_str, outputFolder


def direct_plot(args, experiment_str, outputFolder, title, x_axis_col):
    res_datasets = []
    dataset_list = ALL_DATASETS if args.dataset == 'all' else args.dataset.split(';')
    bias_folder = '/'.join(args.bias.split('/')[:-1])
    bias_folder = bias_folder if bias_folder=='' else bias_folder +'/'
    bias_list =  args.bias.split('/')[-1].split(';')
    for dataset in dataset_list:
        for bias in bias_list:
            print(f'folder: {bias_folder}, bias: {bias}, dataset: {dataset}')
            res_datasets.append(get_results(f'{bias_folder}{bias}', dataset, experiment_str))

    final_res = pd.concat(res_datasets, ignore_index=True)
    if title=='bias':
        title_col = CONVS[f'{bias_list[0]}_title']
    elif title=='dataset':
        title_col = CONVS[args.dataset]
    #x_axis_col = 'Bias' #title_col, y_axis_col, x_axis_col
    for score in ['test_loss', 'accuracy','f1_score','average_precision','auroc']:
        draw_boxplot(final_res, outputFolder, title_col, score, x_axis_col)
        
def plot_yasin_dims(args, experiment_str, outputFolder):
    target_x_col = 'Dimension'
    res_datasets = []
    dataset_list = ALL_DATASETS if args.dataset == 'all' else args.dataset.split(';')
    bias_folder = '/'.join(args.bias.split('/')[:-1])
    bias_folder = bias_folder if bias_folder=='' else bias_folder +'/'
    bias =  args.bias.split('/')[-1]
    for dim in DIMS:
        dataset = args.dataset.replace('nf', f'nf{dim}')
        print(f'folder: {bias_folder}, bias: {bias}, dataset: {dataset}')
        one_res = get_results(f'{bias_folder}{bias}', dataset, experiment_str)
        one_res[target_x_col] = str(dim)
        res_datasets.append(one_res)

    final_res = pd.concat(res_datasets, ignore_index=True)
    print(final_res.head())
    for score in ['test_loss', 'accuracy','f1_score','average_precision','auroc']:
        draw_boxplot(final_res, outputFolder, title_col=CONVS[args.dataset], y_axis_col = score, x_axis_col=target_x_col)
        
def plot_yasin_dims_ris(args, experiment_str, outputFolder):
    target_x_col = 'Dimension-Informative Features(%)'
    res_datasets = []
    dataset_list = ALL_DATASETS if args.dataset == 'all' else args.dataset.split(';')
    bias_folder = '/'.join(args.bias.split('/')[:-1])
    bias_folder = bias_folder if bias_folder=='' else bias_folder +'/'
    bias =  args.bias.split('/')[-1]
    for dim in DIMS:
        dataset_tmp = args.dataset.replace('nf', f'nf{dim}')
        for ri_v, ri_p in RIS.items():
            dataset = dataset_tmp.replace('ri', f'ri{ri_v}')
            print(f'folder: {bias_folder}, bias: {bias}, dataset: {dataset}')
            one_res = get_results(f'{bias_folder}{bias}', dataset, experiment_str)
            one_res[target_x_col] = f'{dim} - {ri_p}%'
            res_datasets.append(one_res)

    final_res = pd.concat(res_datasets, ignore_index=True)
    print(final_res.head())
    for score in SCORES.keys():
        draw_boxplot(final_res, outputFolder, title_col=CONVS[f'{bias}_title'], y_axis_col = score, x_axis_col=target_x_col)

if __name__ == '__main__':
    args, experiment_str, outputFolder = prepare_main()
    #plot_yasin_dims(args, experiment_str, outputFolder)
    if 'custom_' in args.dataset:
        plot_yasin_dims_ris(args, experiment_str, outputFolder)
    if 'moon' in args.dataset:
        direct_plot(args, experiment_str, outputFolder, 'dataset', 'Bias')
    if 'all' in args.dataset:
        direct_plot(args, experiment_str, outputFolder, 'bias', 'Dataset')