import numpy as np
import matplotlib as mpl
from typing import Literal
import matplotlib.pyplot as plt

from typing import Any




def sphere_points(radius:float,
                  center:np.ndarray|tuple=(0.0,0.0,0.0),
                  vertical_points:int=10,
                  horizontal_points:int=20
                 ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Returns a series of pints in a sphere format

    **Arguments**:
    - `radius` : radius of sphere
    - `center` : coordinates of the sphere's center
    - `n_points` : how many point should the mesh have

    **Returns**: `x_coordinates`, `y_coordinates`, `z_coordinates`
    '''
    u, v = np.mgrid[0:2*np.pi:complex(0, horizontal_points), 0:np.pi:complex(0, vertical_points)]
    x = radius*np.cos(u)*np.sin(v) + center[0]
    y = radius*np.sin(u)*np.sin(v) + center[1]
    z = radius*np.cos(v) + center[2]
    return x, y, z


def circumference_points(radius:float,
                         center:np.ndarray|tuple=(0.0,0.0,0.0),
                         n_points:int=10,
                        ) -> tuple[np.ndarray,np.ndarray,np.ndarray]:
    '''
    Returns a series of pints in a sphere format

    **Arguments**:
    - `radius` : radius of sphere
    - `center` : coordinates of the sphere's center
    - `n_points` : how many point should the mesh have

    **Returns**: `x_coordinates`, `y_coordinates`, `z_coordinates`
    '''
    u = np.linspace(0.0, 2*np.pi, n_points)
    x = radius*np.cos(u) + center[0]
    y = radius*np.sin(u) + center[1]
    return x, y


def plot_roc_curve(actual:list[int], predicted:list[float], title:str="ROC", show_plot:bool=False, save_plot_path:str|None=None, verbose:bool=False) -> None:
    '''
    `actual` must contain the real class with values in `{0,1}'.
    
    `predicted` must contain **probabilities** not labels!! 
    '''
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, thresholds = roc_curve(actual, predicted)
    print(fpr, tpr)
    exit()
    roc_auc = auc(fpr, tpr)

    plt.figure()
    lw = 2
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=lw,
        label="ROC curve (area = %0.2f)" % roc_auc,
    )
    plt.plot([0, 1], [0, 1], color="navy", linestyle="--", label="Random chance")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if show_plot:
        plt.show()
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=200)
    plt.clf()




def compare_sequences(real:np.ndarray, fake:np.ndarray,
                      real_label:str="Real sequence", fake_label:str="Fake Sequence",
                      show_graph:bool=False, save_img:bool=False,
                      img_idx:int=0, img_name:str="plot", folder_path:str=None):
    '''
    Plots two graphs with the two sequences.

    Arguments:
        - `real`: the first sequence with dimension [seq_len, data_dim]
        - `fake`: the second sequence with dimension [seq_len, data_dim]
        - `show_graph`: whether to display the graph or not
        - `save_img`: whether to save the image of the graph or not
        - `img_idx`: the id of the graph that will be used to name the file
        - `img_name`: the file name of the graph that will be used to name the file
        - `folder_path`: path to the folder where to save the image

    Returns:
        - numpy matrix with the pixel values for the image
    '''
    mpl.use('Agg')
    fig, (ax0, ax1) = plt.subplots(2, 1, layout='constrained')
    ax0.set_xlabel('Time-Steps')

    for i in range(real.shape[1]):
        ax0.plot(real.cpu()[:,i])
    ax0.set_ylabel(real_label)

    for i in range(fake.shape[1]):
        ax1.plot(fake.cpu()[:,i])
    ax1.set_ylabel(fake_label)

    if show_graph:
        plt.show()
    if save_img:
        plt.savefig(f"{folder_path}{img_name}-{img_idx}.png")


    # return picture as array
    canvas = fig.canvas
    canvas.draw()  # Draw the canvas, cache the renderer
    plt.clf()

    image_flat = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')  # (H * W * 3,)
    # NOTE: reversed converts (W, H) from get_width_height to (H, W)
    return image_flat.reshape(*reversed(canvas.get_width_height()), 3)  # (H, W, 3)
    



def plot_processes(samples:np.ndarray, labels:list[str]|None=None,
                 img_name:str="plot",
                 folder_path:str|None=None,
                 show_plot:bool=False,
                 title:str=None,
                 dpi:int=100,
                ) -> None:
    '''
    Plots all the dimensions of the generated dataset.
    '''
    assert folder_path is not None or show_plot
    for i in range(samples.shape[1]):
        if labels is not None:
            plt.plot(samples[:,i], label=labels[i])
        else:
            plt.plot(samples[:,i])
    # giving a title to my graph 
    if labels is not None:
        plt.legend()
    if title is not None:
        plt.title(title)
    plt.grid()
    # function to show the plot 
    if folder_path is not None:
        plt.savefig(f"{folder_path}{img_name}.png", dpi=dpi)
    if show_plot:
        plt.show()
    plt.clf()




def label_distribution_plot(DATA:list[str],
                            str_to_id:dict[str,int]|None=None,
                            save_plot_path:str|None=None,
                            title:str|None=None,
                            x_label:str|None=None,
                            y_label:str|None=None,
                            show_plot:bool=False,
                            verbose:bool=False,
                           ) -> None:
    '''
    - `DATA`: list of classifications to count for the histogram
    - `str_to_id`: if the classes are strings provide the mapping from `str` to `int`, the `int` translation will also be used to order the bins on the *x* axis
    - `save_plot_path`: if provided it will save the *.png* of the plot in the given path
    - `show_plot`: will trigger the `plt.show()` function
    - `verbose`: whether to print status messager or not
    - `title`: plot title
    - `y_label`: label on the *y* axis
    - `x_label`: label on the *x* axis
    '''
    from utils.utils import invert_dict

    n_samples:int = len(DATA)

    # if mapping is present, convert in order to enforce label ordering on the x axis
    if str_to_id:
        int_label_data = [str_to_id[i] for i in DATA]
    else:
        int_label_data = DATA
    
    # HISTOGRAM
    labels, counts = np.unique(int_label_data, return_counts=True)
    plt.bar(labels, [c/n_samples*100 for c in counts], align='center')

    if verbose:
        int_to_str = invert_dict(str_to_id)
        print("Label distribution is:")
        for l in labels:
            print(f"\t{int_to_str[l]}: {int(counts[l])} ({round(counts[l]/n_samples*100,2)}%)")

    # center labels on bins
    if str_to_id:
        plt.xticks(ticks=list(str_to_id.values()), labels=list(str_to_id.keys()))
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    plt.grid(axis='y', color="black", linestyle='--', linewidth=0.5)
    plt.yticks([i for i in range(0, 101, 10)], labels=[f"{i}%" for i in range(0, 101, 10)])
    if save_plot_path:
        plt.savefig(save_plot_path, dpi=200)
        if verbose:
            print(f"Label distribution plot saved to '{save_plot_path}' successfully.")
    if show_plot:
        plt.show()
    plt.clf()




def corr_heatmap(correlation:np.ndarray,
                 show_pic:bool=True,
                 pic_name:str="correlation-heatmap",
                 save_pic:bool=True,
                 pic_folder:str=None
                 ) -> None:
    '''
    Saves a picture of the correlation matrix as a heatmap.
    '''
    import seaborn as sns
    plt.figure()
    sns.heatmap(correlation,
                cmap='RdBu',
                annot=False,
                vmin=-1,
                vmax=1
               )
    if save_pic:
        plt.savefig(f"{pic_folder}{pic_name}.png",dpi=300)
    if show_pic:
        plt.show()




def PCA_visualization(nominal_data:np.ndarray,
                      anomalous_data:np.ndarray,
                      label_1:str="Normal",
                      label_2:str="Anomalous",
                      title:str="Distribution comparison",
                      show_plot:bool=False,
                      save_plot:bool=True,
                      folder_path:str=None,
                      img_name:str="pca-visual",
                      verbose:bool=False
                      ) -> None:
    """
    Using PCA for generated and original data visualization
     on both the original and synthetic datasets (flattening the temporal dimension).
     This visualizes how closely the distribution of generated samples
     resembles that of the original in 2-dimensional space

    Arguments:
    - `nominal_data`: original data ( num_sequences, data_dim )
    - `anomalous_data`: generated synthetic data ( num_sequences, data_dim )
    - `show_plot`: display the plot
    - `save_plot`: save the .png of the plot
    - `folder_path`: where to save the file
    """  
    from sklearn.decomposition import PCA
    if show_plot or save_plot:
        # Data preprocessing
        N1 = nominal_data.shape[0]
        N2 = anomalous_data.shape[0]
        p = nominal_data.shape[1]
        assert(anomalous_data.shape[1] == p)

        prep_data = nominal_data.reshape((N1,p))
        prep_data_hat = anomalous_data.reshape((N2,p))
        
        # Visualization parameter        
        # PCA Analysis
        pca = PCA(n_components=2)
        pca.fit(prep_data)
        pca_results = pca.transform(prep_data)
        pca_hat_results = pca.transform(prep_data_hat)

        # Plotting
        blue = ["blue" for i in range(N1)]
        red = ["red" for i in range(N2)]
        f, ax = plt.subplots(1)    
        plt.scatter(pca_results[:,0], pca_results[:,1],
                    c=blue, alpha = 0.25, label = label_1)
        plt.scatter(pca_hat_results[:,0], pca_hat_results[:,1], 
                    c=red, alpha = 0.25, label = label_2)

        ax.legend()  
        plt.title(title)
        if save_plot:
            plt.savefig(f"{folder_path}{img_name}.png", dpi=300)
        if show_plot:
            plt.show()
        plt.clf()
    else:
        raise ValueError(f"either show_plot ({show_plot}) or save_plot ({save_plot}) must be true")
    



def show_summary_statistics(actual:np.ndarray, 
                            predicted:np.ndarray,
                            model_name:str='model',
                            labels:list[int]="auto",
                            normalize:Literal['true', 'pred', 'all']='true',
                            title:str='Confusion Matrix',
                            use_round:bool=False,
                            save_pic:bool=True,
                            pic_folder:str=None,
                            show_plot:bool=True,
                            verbose:bool=True,
                            get_all_stats:bool=False,
                            ) -> np.ndarray | tuple[np.ndarray, float, float, float]:
    '''
    Computes and displays confusion matrix.

    If `get_all_stats=True` returns the tuple `(confusion_matrix, f1_score, precision, recall)`, otherise it returns only the confusion matrix
    '''
    import seaborn as sns
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score
    from sklearn.metrics import f1_score
    from sklearn.metrics import recall_score
    cm = confusion_matrix(y_true=actual, y_pred=predicted, normalize=normalize)
    if use_round:
        cm = np.round(cm, 4)
    norm = plt.Normalize(0,100)
    if save_pic or show_plot:
        sns.heatmap(cm * 100, 
                    annot=True,
                    fmt='g', 
                    xticklabels=labels,
                    yticklabels=labels,
                    norm=norm
                    )
        plt.xlabel('Prediction',fontsize=13)
        plt.ylabel('Actual',fontsize=13)
        plt.title(title,fontsize=17)

    if save_pic:
        plt.savefig(f"{pic_folder}{model_name}_confusion.png", dpi=200)
        plt.clf()
    if show_plot:
        plt.show()
        plt.clf()

    f1_val = f1_score(actual, predicted, average=None)
    precision_val = precision_score(actual, predicted, average=None)
    recall = recall_score(actual, predicted, average=None)
    if verbose:
        print("Precision: ", precision_val)
        print("Recall:    ", recall)
        print("F1 score:  ", f1_val)
    if get_all_stats:
        return cm, f1_val, precision_val, recall
    return cm




def confront_univariate_plots(main_series:np.ndarray,
                              other_series:np.ndarray,
                              plot_img:str,
                              main_label:str="Actual",
                              other_label:str="Predicted",
                              title:str="Predictions vs Actual"
                             ) -> None:
    '''
    Plots the model's predictions against the actual `y_test` values.

    **Parameters**:
    - `main_series` : the main plot, it will be shown as a continuous thick line
    - `other_series` : the other plot to be confronted with the `main_series`
    - `plot_img` : where to save the picture
    - `title` : title of the plot
    - `main_label` : the name for the `main_series`
    - `other_label` : the name for the `other_series`
    '''
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(main_series, label=main_label, color='black', linewidth=2)
    plt.plot(other_series, label=other_label, color='red', linestyle='--')
    steps = np.arange(0, len(main_series))
    plt.fill_between(steps, main_series, other_series,
                     where=None,       # or a boolean array if you only want some segments
                     interpolate=True, # helps when lines cross
                     color='red',
                     alpha=0.3,
                     label="Error",
                    )
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_img)
    plt.clf()




def confront_multivariate_plots(main_series,
                                other_series,
                                plot_img:str,
                                labels:list[str]|None=None,
                                main_label:str="Actual",
                                other_label:str="Predicted",
                                title:str="Predictions vs Actual",
                               ) -> None:
    '''
    Plots the model's predictions against the actual `y_test` values.

    **Parameters**:
    - `main_series` : the main plot, it will be shown as a continuous thick line
    - `other_series` : the other plot to be confronted with the `main_series`
    - `plot_img` : where to save the picture
    - `title` : title of the plot
    - `main_label` : the name for the `main_series`
    - `other_label` : the name for the `other_series`
    '''
    # Ensure 2D shape
    if main_series.ndim == 1:
        main_series = main_series.reshape(-1, 1)
        other_series = other_series.reshape(-1, 1)

    n_steps, n_dims = main_series.shape
    steps = np.arange(n_steps)

    fig, axes = plt.subplots(
        n_dims,
        1,
        figsize=(10, 4 * n_dims),
        sharex=True
    )

    if n_dims == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        ax.plot(main_series[:, i], label=main_label, color='black', linewidth=2)
        ax.plot(other_series[:, i], label=other_label, color='red', linestyle='--')

        ax.fill_between(
            steps,
            main_series[:, i],
            other_series[:, i],
            interpolate=True,
            color='red',
            alpha=0.3,
            label="Error"
        )
        label = labels[i] if labels is not None else f"Dim {i}"
        ax.set_ylabel(label)
        ax.legend()
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Time Step")
    fig.suptitle(title, fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(plot_img)
    plt.close()



