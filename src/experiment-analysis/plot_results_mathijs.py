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
import pathlib
FIG_SIZE = [4.1,2]
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
RAW_METHOD_NAMES, FIXED_METHOD_NAMES = RAW_METHOD_NAMES[2:], FIXED_METHOD_NAMES[2:]
ALL_CANCERS = ['BRCA', 'CESC', 'LUAD', 'OV', 'SKCM']
ALL_DATASETS = {}
ALL_DATASETS['Single Cancer'] = {'BRCA': (90, 20), 'OV': (85, 20), 'CESC': (90, 10), 'SKCM': (90, 10), 'LUAD': (90, 10)}
ALL_DATASETS['Double Holdout'] = {'BRCA': (85, 6), 'OV': (80, 6), 'CESC': (75, 6), 'SKCM': (75, 20), 'LUAD': (90, 10)}
ALL_DATASETS['Cross Dataset'] = [('isle-dsl', 'BRCA', 6, 90), ('dsl-isle', 'BRCA', 6, 90), ('isle-dsl', 'LUAD', 10, 80), ('dsl-isle', 'LUAD', 10, 85), ('exp2sl-dsl', 'LUAD', 6, 85), ('dsl-exp2sl', 'LUAD', 10, 80)]

CONVS = {'output/experiments-pipeline-what-settings': 'Single Cancer', 'Single Cancer': 'output/experiments-pipeline-what-settings',
         'output/experiments-double-holdout': 'Double Holdout', 'Double Holdout': 'output/experiments-double-holdout',
         'multiple_experiments/output': 'Cross Dataset', 'Cross Dataset': 'multiple_experiments/output'}
D_SYMBOL0 = r'$\Delta_0$'
D_SYMBOL1 = r'$\Delta_1$'
#delta_100|2|0,0|0,0

if platform.system() == 'Windows':
    webDriveFolder = "W:/staff-umbrella/JGMasters/2122-mathijs-de-wolf/feature_sets/"
    mainOutputFolder = config.RESULT_DIR / 'mathijs'
    mainInputFolder = pathlib.Path('/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf')
else:
    webDriveFolder = "/home/nfs/ytepeli/python_projects/msc-thesis-2122-mathijs-de-wolf/data"
    mainOutputFolder = config.RESULT_DIR / 'mathijs'
    mainInputFolder = pathlib.Path('/tudelft.net/staff-umbrella/JGMasters/2122-mathijs-de-wolf')

def init_argparse() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        usage="%(prog)s [OPTION] [FILE]...",
        description="Run metric learning pipeline"
    )
    parser.add_argument("--model", "-m", choices=['supervised', 'balanced_semi_supervised', 'true_semi_supervised', 'diversity', 'all'], default='all')
    parser.add_argument("--sl_exp_type", "-slet", choices=['single_cancer', 'double_holdout', 'cross_dataset'], default='single_cancer')
    parser.add_argument("--output_d_folder", "-odf", default=None)
    parser.add_argument("--output_b_folder", "-obf", default=None)
    
    return parser

def get_results(bias_name = 'experiments-pipeline-what-settings', cancer='BRCA', experiment_file=None):
    res_pd_list = []
    #f"experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}
    res_folder = mainInputFolder / bias_name / f'{experiment_file}'
    print(res_folder)
    for raw_m_name, fix_m_name in zip(RAW_METHOD_NAMES, FIXED_METHOD_NAMES):
        try:
            tmp_res_loc = res_folder / f'{raw_m_name}_test_performance.csv'
            tmp_res = pd.read_csv(tmp_res_loc)
            tmp_res['Bias'] = CONVS[bias_name]
            tmp_res['Cancer'] = cancer
            tmp_res['Bias-Cancer'] = CONVS[bias_name]+' --- '+cancer
            tmp_res['Experiment Id'] = experiment_file
            tmp_res['Method'] = fix_m_name
            res_pd_list.append(tmp_res)
        except Exception as e:
            print(f'{fix_m_name} for {CONVS[bias_name]}, {cancer} not run due to {e}')
    full_res = pd.concat(res_pd_list, ignore_index = True)
    return full_res

def add_median_labels(ax, lineno=4, fmt='.2f'):
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if type(c).__name__ == 'PathPatch']
    lines_per_box = int(len(lines) / len(boxes))
    for median in lines[lineno:len(lines):lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if (median.get_xdata()[1] - median.get_xdata()[0]) == 0 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center', color='black', fontsize=2.8)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=0.2, foreground=median.get_color()),
            path_effects.Normal(),
        ])

def draw_boxplot(res, outputFolder, title_col, y_axis_col, x_axis_col, report=False):
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
    if report:
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
        plt.ylim((0.3, 1.0))
    plt.tick_params(axis ='both', width=0.5)
    plt.ylabel(SCORES[y_axis_col])
    for out_fmt in ['png', 'pdf']:
        out_loc = outputFolder / f'{y_axis_col}2.{out_fmt}'
        if report:
            out_loc = outputFolder / f'{y_axis_col}_report.{out_fmt}'
        plt.savefig(out_loc, dpi=300, bbox_inches='tight')

def prepare_main():
    parser = init_argparse()
    args = parser.parse_args()
    print(args)
    
    if args.output_b_folder is None:
        output_b_folder = args.sl_exp_type
    else:
        output_b_folder = args.output_b_folder
    outputFolder = mainOutputFolder / output_b_folder / 'knn-10' / 'result_vis'
    config.ensure_dir(outputFolder)
    config.safe_create_dir(outputFolder)
    
    return args, 'knn-10', outputFolder


def plot_cancers(x_axis_col = 'Cancer'):
    args, experiment_str, outputFolder = prepare_main()
    res_datasets = []
    bias = args.sl_exp_type.replace('_', ' ').title()
    for cancer in ALL_CANCERS:
        conf, pseudolabel = ALL_DATASETS[bias][cancer]
        experiment_str = f'experiment-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}'
        if 'Holdout' in bias:
            experiment_str = f'experiment-double-holdout-divergence-pipeline-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}'
        elif 'Cross' in bias:
            experiment_str = f'experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}'
        print(f'folder: {bias}, cancer: {cancer}, experiment: {experiment_str}')
        res_datasets.append(get_results(f'{CONVS[bias]}', cancer, experiment_str))

    final_res = pd.concat(res_datasets, ignore_index=True)
    title_col = bias
    #x_axis_col = 'Bias' #title_col, y_axis_col, x_axis_col
    print(final_res)
    for score in ['test_loss', 'accuracy','f1_score','average_precision','auroc']:
        draw_boxplot(final_res, outputFolder, title_col, score, x_axis_col, report=True)
        draw_boxplot(final_res, outputFolder, title_col, score, x_axis_col)
        
def plot_cross_cancers(x_axis_col = 'Cancer'):
    args, experiment_str, outputFolder = prepare_main()
    res_datasets = []
    bias = args.sl_exp_type.replace('_', ' ').title()
    for exp, cancer, pseudolabel, conf in ALL_DATASETS[bias]:
        experiment_str = f'experiment-{exp}-{cancer}-knn-10-conf-{conf}-pseudolabels-{pseudolabel}'
        print(f'folder: {bias}, cancer: {cancer}, experiment: {experiment_str}')
        res_tmp = get_results(f'{CONVS[bias]}', cancer, experiment_str)
        res_tmp['Cancer-Exp'] = f'{cancer}\n{exp}'
        res_datasets.append(res_tmp)

    final_res = pd.concat(res_datasets, ignore_index=True)
    title_col = bias
    #x_axis_col = 'Bias' #title_col, y_axis_col, x_axis_col
    print(final_res)
    for score in ['test_loss', 'accuracy','f1_score','average_precision','auroc']:
        draw_boxplot(final_res, outputFolder, title_col, score, x_axis_col, report=True)
        draw_boxplot(final_res, outputFolder, title_col, score, x_axis_col)

if __name__ == '__main__':
    #plot_yasin_dims()
    #plot_yasin_dims_ris()
    plot_cancers('Cancer')
    #plot_cross_cancers('Cancer-Exp')