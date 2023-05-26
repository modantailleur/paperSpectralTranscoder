import os
import torch
import yaml
import yamlloader
import numpy as np
import matplotlib.pyplot as plt
import math
import utils.bands_transform as bt
from prettytable import PrettyTable
import matplotlib as mpl

class SettingsLoader(yaml.SafeLoader):
    def __init__(self, stream):
        self._root = os.path.split(stream.name)[0]
        super(SettingsLoader, self).__init__(stream)
    
    def include(self, node):
        filename = os.path.join(self._root, self.construct_scalar(node))
        with open(filename, 'r') as f:
            #return yaml.load(f, YAMLLoader)
            return yaml.load(f, yamlloader)
SettingsLoader.add_constructor('!include', SettingsLoader.include)

def load_settings(file_path):
    with file_path.open('r') as f:
        return yaml.load(f, Loader=SettingsLoader)
    
def tukey_window(M, alpha=0.2):
    """Return a Tukey window, also known as a tapered cosine window, and an 
    energy correction value to make sure to preserve energy.
    Window and energy correction calculated according to:
    https://github.com/nicolas-f/noisesensor/blob/master/core/src/acoustic_indicators.c#L150

    Parameters
    ----------
    M : int
        Number of points in the output window. 
    alpha : float, optional
        Shape parameter of the Tukey window, representing the fraction of the
        window inside the cosine tapered region.
        If zero, the Tukey window is equivalent to a rectangular window.
        If one, the Tukey window is equivalent to a Hann window.

    Returns
    -------
    window : ndarray
        The window, with the maximum value normalized to 1.
    energy_correction : float
        The energy_correction used to compensate the loss of energy due to
        the windowing
    """
    #nicolas' calculation
    index_begin_flat = int((alpha / 2) * M)
    index_end_flat = int(M - index_begin_flat)
    energy_correction = 0
    window = np.zeros(M)
    
    for i in range(index_begin_flat):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - alpha / 2))))
        energy_correction += window_value * window_value
        window[i]=window_value
    
    energy_correction += (index_end_flat - index_begin_flat) #window*window=1
    for i in range(index_begin_flat, index_end_flat):
        window[i] = 1
    
    for i in range(index_end_flat, M):
        window_value = (0.5 * (1 + math.cos(2 * math.pi / alpha * ((i / M) - 1 + alpha / 2))))
        energy_correction += window_value * window_value
        window[i] = window_value
    
    energy_correction = 1 / math.sqrt(energy_correction / M)
    
    return(window, energy_correction)

def get_transforms(sr=32000, flen=4096, hlen=4000, classifier='YamNet', device=torch.device("cpu"), tho_freq=True, tho_time=True, mel_template=None):
    
    if mel_template is None:
        tho_tr = bt.ThirdOctaveTransform(sr=sr, flen=flen, hlen=hlen)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=tho_tr.flen, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=tho_tr.flen, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=tho_tr.flen, hlen=tho_tr.hlen)
    else:
        tho_tr = bt.NewThirdOctaveTransform(32000, 1024, 320, 64, mel_template=mel_template, tho_freq=tho_freq, tho_time=tho_time)
        if classifier == 'PANN':
            mels_tr = bt.PANNMelsTransform(flen_tho=4096, device=device)
        if classifier == 'YamNet':
            mels_tr = bt.YamNetMelsTransform(flen_tho=4096, device=device)
        if classifier == 'default':
            mels_tr = bt.DefaultMelsTransform(sr=tho_tr.sr, flen=4096, hlen=4000)
    return(tho_tr, mels_tr)   

#count the number of parameters of a model
def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: 
            continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def plot_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False):
    if vmin==None:
        vmin = torch.min(x_m)
    if vmax==None:
        vmax = torch.max(x_m)
    exthmin = 1
    exthmax = len(x_m)
    extlmin = 0
    extlmax = 1

    plt.figure(figsize=(8, 5))
    if diff:
        plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
                vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Power differences (dB)', rotation=90, labelpad=15)
    else:

        plt.imshow(x_m, extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
                vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        cbar = plt.colorbar()
        cbar.ax.set_ylabel('Power (dB)', rotation=90, labelpad=15)

    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel(ylabel)
    if save:
        plt.savefig('figures_spectrograms/'+name)
    plt.show()

def plot_multi_spectro(x_m, fs, title='title', vmin=None, vmax=None, diff=False, name='default', ylabel='Mel bin', save=False):
    if vmin==None:
        vmin = torch.min(x_m)
    if vmax==None:
        vmax = torch.max(x_m)
    exthmin = 1
    exthmax = len(x_m[0])
    extlmin = 0
    extlmax = 1

    mpl.rcParams['font.family'] = 'Times New Roman'
    mpl.rcParams['font.size'] = 20
    #mpl.use("pgf")
    # mpl.rcParams.update({
    #     "pgf.texsystem": "pdflatex",
    #     'font.family': 'Times New Roman',
    #     'text.usetex': True,
    #     'pgf.rcfonts': False,
    # })

    #fig, axs = plt.subplots(1, 4, figsize=(20, 5), sharey=True, gridspec_kw={'width_ratios': [1, 1, 1, 1]})
    fig, axs = plt.subplots(ncols=len(x_m), sharey=True, figsize=(len(x_m)*8, 5))
    #fig.subplots_adjust(wspace=1)

    for i, ax in enumerate(axs):
        if i == 0:
            ylabel_ = ylabel
        else:
            ylabel_ = ''
        if diff:
            im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='seismic',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')
        else:
            im = ax.imshow(x_m[i], extent=[extlmin,extlmax,exthmin,exthmax], cmap='inferno',
                    vmin=vmin, vmax=vmax, origin='lower', aspect='auto')

        ax.set_title(title[i])
        #ax.set_xlabel('Time (s)')
        ax.set_ylabel(ylabel_)

    fig.text(0.5, 0.1, 'Time (s)', ha='center', va='center')
    
    #cbar_ax = fig.add_axes([0.06, 0.15, 0.01, 0.7])
    cbar_ax = fig.add_axes([0.97, 0.15, 0.01, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax, label='Power (dB)')
    cbar.ax.yaxis.set_label_position('left')
    cbar.ax.yaxis.set_ticks_position('left')

    axs[0].set_ylabel(ylabel)
    #fig.tight_layout()
    #fig.tight_layout(rect=[0.1, 0.05, 1, 1], pad=2)
    fig.tight_layout(rect=[0, 0.05, 0.92, 1], pad=2)
    #fig.savefig('fig_spectro' + name + '.pdf', bbox_inches='tight', dpi=fig.dpi)
    if save:
        plt.savefig('fig_spectro' + name + '.pdf', dpi=fig.dpi, bbox_inches='tight')
    plt.show()

def sort_labels_by_score(scores, labels, top=-1):
    # create a list of tuples where each tuple contains a score and its corresponding label
    score_label_tuples = list(zip(scores, labels))
    # sort the tuples based on the score in descending order
    sorted_tuples = sorted(score_label_tuples, reverse=True)
    # extract the sorted labels from the sorted tuples
    sorted_labels = [t[1] for t in sorted_tuples]
    sorted_scores = [t[0] for t in sorted_tuples]
    
    # create a list of 1s and 0s indicating if the score is in the top 10 or not
    top_scores = sorted_scores[:top]
    top_labels = sorted_labels[:top]

    if top >= 1:
        in_top = [1 if label in top_labels else 0 for label in labels]
    else:
        in_top = None
    
    return sorted_scores, sorted_labels, in_top

# #MT: added
# class ChunkManager():
#     def __init__(self, dataset_name, model_name, model_batch_path, batch_type_name, batch_lim=1000):
#         self.dataset_name = dataset_name
#         self.model_name = model_name
#         self.model_batch_path = model_batch_path
#         self.batch_type_name = batch_type_name
#         self.current_batch_id = 0
#         self.batch_lim = batch_lim
#         self.total_folder_name = self.batch_type_name + '_' + self.model_name + '_' + self.dataset_name
#         self.total_path = self.model_batch_path / self.total_folder_name
#         if not os.path.exists(self.total_path):
#             os.makedirs(self.total_path)
#         else:
#             print(f'WARNING: everything will be deleted in path: {self.total_path}')
#             self._delete_everything_in_folder(self.total_path)
        
#     def save_chunk(self, batch, forced=False):
#         if len(batch) == 0:
#             return(batch)
#         if len(batch) >= self.batch_lim or forced == True:
#             file_path = self.total_path / (self.total_folder_name + '_' + str(self.current_batch_id) + '.npy')
#             np.save(file_path, batch)
#             print(f'save made in: {file_path}')
#             self.current_batch_id+=1
#             return([])
#         else:
#             return(batch)
        
#     def open_chunks(self):
#         stacked_batch = np.array([])
#         for root, dirs, files in os.walk(self.total_path):
            
#             #sort files
#             files_splitted = [re.split(r'[_.]', file) for file in files]
#             file_indices = [int(file[-2]) for file in files_splitted]
#             file_indices_sorted = file_indices.copy()
#             file_indices_sorted.sort()
#             file_new_indices = [file_indices.index(ind) for ind in file_indices_sorted]
#             files_sorted = [files[i] for i in file_new_indices]
            

#             for file in files_sorted:
#                 cur_batch = np.load(self.total_path / file, allow_pickle=True)
#                 stacked_batch = np.concatenate((stacked_batch, cur_batch))

#         return(stacked_batch)
            
#     def _delete_everything_in_folder(self, folder):
#         for filename in os.listdir(folder):
#             file_path = os.path.join(folder, filename)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     send2trash(file_path)
#                     #shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete %s. Reason: %s' % (file_path, e))