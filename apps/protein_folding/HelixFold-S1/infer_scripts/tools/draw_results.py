from typing import List, Union
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def add_chain_to_token_offset_legend(ax, chain_idx_map, add_dividing_line=True, verbose=True):
    """Add the chain offset(token offset) legend to the figure.

        Args:
            ax: matplotlib.axes.Axes, the axis to add the legend to.
            chain_idx_map: dict, the mapping for chain name to the start and end indices.
                such as: {1-1: (0, 1), 1-2: (2, 5), 2-1: (6, 10), 2-2: (11, 15), ...}
            add_dividing_line: bool, whether to add the dividing line between chains in ax.
            verbose: bool, whether to print the chain name, start, and end indices.
    """
    legend_elements = []
    for idx, (chain_name, (start, end)) in enumerate(chain_idx_map.items()):
        start = start + 1
        end = end + 1
        if verbose:
            print(f'chain_name: {chain_name}, start: {start}, end: {end}')
        
        if idx > 0 and add_dividing_line:
            ax.axhline(start - 0.5, color='red', linewidth=1, linestyle='--')  
            ax.axvline(start - 0.5, color='red', linewidth=1, linestyle='--')  
        
        legend_elements.append(plt.Line2D([0], [0], color='none', 
                                        label=f'{chain_name}: {start}-{end}'))

    ax.legend(handles=legend_elements, 
             loc='upper center', 
             bbox_to_anchor=(0.5, -0.12),  
             ncol=min(len(legend_elements), 3),
             frameon=True, 
             framealpha=0.9, 
             edgecolor='gray',
             fancybox=True,  
             shadow=True,  
             fontsize=10,  
             labelspacing=0.5, 
             columnspacing=1.0,
             handletextpad=0.5,
             borderpad=0.5, 
             title='Token offset for entity (entity id: range)', 
             title_fontsize=11)


def draw_interface_heatmap(matrix: Union[np.ndarray, List[List[float]]], 
                           chain_idx_map: dict,
                           path: str, figure_size=(8, 6), title='', verbose=True):
    """Draw the interface heatmap.

        Args:
            matrix: np.ndarray, the interface heatmap.
            chain_idx_map: dict, the chain ids map: start and end indices.
            path: str, the path to save the figure.
            figure_size: tuple, the figure size.
            title: str, the title of the figure.
    """
    if type(matrix) == list:
        matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=figure_size)
    N = matrix.shape[0]
    
    im = ax.imshow(matrix, cmap='Blues', aspect='auto', interpolation='nearest', 
                    origin='upper', extent=[0.5, N + 0.5, N + 0.5, 0.5])
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)

    add_chain_to_token_offset_legend(ax, chain_idx_map, verbose=verbose)

    axis_ticks_interval = max(1, len(matrix) // 10)  
    y_ticks = x_ticks = list(range(1, len(matrix) + 1, axis_ticks_interval)) 

    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    ax.set_xlim(0.5, N + 0.5)
    ax.set_ylim(N + 0.5, 0.5)  

    ax.set_title(title)
    ax.set_xlabel('Token Index')
    ax.set_ylabel('Token Index')

    plt.savefig(path, format='png', bbox_inches='tight', dpi=300)
    plt.close(fig)


def draw_pae(pae_matrix: list, chain_idx_map: dict, path: str, figure_size=(8, 6), verbose=True):
    """Draw the PAE heatmap.

        Args:
            pae_matrix: list, the PAE matrix.
            chain_idx_map: dict, the chain ids map: start and end indices.
            path: str, the path to save the figure.
            figure_size: tuple, the figure size.
    """
    if len(pae_matrix) == 0 or len(pae_matrix[0]) != len(pae_matrix):
                raise ValueError('PAE matrix is not square')

    fig, ax = plt.subplots(figsize=figure_size)
    plt.rcParams.update({'font.size': 9}) 
    pae_matrix = np.minimum(pae_matrix, 50)

    plt.imshow(pae_matrix, cmap='Greens_r', interpolation='nearest', 
                extent=[1, len(pae_matrix), len(pae_matrix), 1])
    
    add_chain_to_token_offset_legend(ax, chain_idx_map, add_dividing_line=False, verbose=verbose)

    aspect_ratio = figure_size[1] / figure_size[0] 
    shrink_value = 0.72 * aspect_ratio  
    cbar = plt.colorbar(shrink=shrink_value)
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: int(x)))

    plt.title('Predicted Aligned Error (Ångströms)')
    plt.xlabel('Scored Token')
    plt.ylabel('Aligned Token')
    plt.tight_layout(pad=2.0)

    axis_ticks_interval = max(1, len(pae_matrix) // 10)  
    y_ticks = x_ticks = list(range(1, len(pae_matrix) + 1, axis_ticks_interval)) 
    ax.set_xticks(x_ticks)
    ax.set_yticks(y_ticks)

    plt.savefig(path, format='svg', bbox_inches='tight', dpi=100)
    plt.close(fig)


def test_png():
    import glob, os, json, sys
    import numpy as np
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    def _get_chain_indices(chain_ids: Union[np.ndarray, list]) -> dict:
        if isinstance(chain_ids, list):
            chain_ids = np.array(chain_ids)
        
        chain_starts_ends = {}
        unique_chains = np.unique(chain_ids) # chains are numbered 1-1, 1-2, 2-1, 2-2, ...
        for chain in unique_chains:
            positions = np.where(chain_ids == chain)[0]
            chain_starts_ends[chain] = (positions[0], positions[-1])

        return chain_starts_ends
    
    root_path = './test_data/interface_png_debug/'
    interface_json_path = glob.glob(os.path.join(root_path, '*', '*.json'))
    print('interface_json_path: ', len(interface_json_path))

    for json_path in interface_json_path:
        interface_json = json.load(open(json_path, 'r'))
        interface_prob = interface_json['token_pair_interface_probs']
        chain_idx_map = _get_chain_indices(interface_json['token_chain_ids'])
        draw_interface_heatmap(interface_prob, chain_idx_map, 
                               json_path.replace('.json', '.png'), 
                               title="Predicted Interface Probability")
        draw_pae(interface_prob, chain_idx_map, json_path.replace('.json', '_pae.svg'))


if __name__ == '__main__':
     test_png()