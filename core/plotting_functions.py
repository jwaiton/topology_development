'''
PLOTTING FUNCTIONS GO HERE
'''

import numpy as np
import matplotlib.pyplot as plt

def raw_plotter(q, evt, MC = False, pitch = 15.55, show = True, title = None):
    '''
    just plots the hits, nothing smart
    '''


    # MC plotter
    if MC:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        axes[0].plot(q.x, q.y)
        axes[0].set_xlabel('X (mm)');
        axes[0].set_ylabel('Y (mm)');

        axes[1].plot(q.x, q.z)
        axes[1].set_xlabel('X (mm)');
        axes[1].set_ylabel('Z (mm)');

        axes[2].plot(q.y, q.z)
        axes[2].set_xlabel('Y (mm)');
        axes[2].set_ylabel('Z (mm)');
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        xx = np.arange(q.X.min(), q.X.max() + pitch, pitch)
        yy = np.arange(q.Y.min(), q.Y.max() + pitch, pitch)
        zz = np.sort(q.Z.unique())

        axes[0].hist2d(q.X, q.Y, bins=[xx, yy], weights=q.Q, cmin=0.0001);
        axes[0].set_xlabel('X (mm)');
        axes[0].set_ylabel('Y (mm)');

        axes[1].hist2d(q.X, q.Z, bins=[xx, zz], weights=q.Q, cmin=0.0001);
        axes[1].set_xlabel('X (mm)');
        axes[1].set_ylabel('Z (mm)');


        axes[2].hist2d(q.Y, q.Z, bins=[yy, zz], weights=q.Q, cmin=0.0001);
        axes[2].set_xlabel('Y (mm)');
        axes[2].set_ylabel('Z (mm)');
        if title is None:
            fig.suptitle(f"event: {evt}")
        if show:
            plt.show(fig)


def plot_MC_prt_info(df, prt_info, evt, show = True):
    '''
    Plots a given event's hits with particle information
    '''
    df = df[df.event_id == evt]
    df_MC = prt_info[prt_info.event_id == evt]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))


    for p_id, df in df.groupby('particle_id'):
        # associate the p_id back to its original from the MC_df
        particle = df_MC[df_MC.particle_id == p_id]
        if not particle.empty:
            particle_name = particle.iloc[0].particle_name
            if particle_name not in plt.gca().get_legend_handles_labels()[1]:
                
                axes[0].plot(df.x, df.y, label=f'{particle_name}')
                axes[1].plot(df.x, df.z, label=f'{particle_name}')
                axes[2].plot(df.y, df.z, label=f'{particle_name}')
            else:
                # reuse color of the already-plotted particle
                ax = plt.gca()
                handles, labels = ax.get_legend_handles_labels()
                try:
                    idx = labels.index(particle_name)
                    color = handles[idx].get_color()
                except ValueError:
                    color = None
                plt.plot(df.x, df.y, color=color)
                plt.plot(df.x, df.z, color=color)
                plt.plot(df.y, df.z, color=color)
        else:
            plt.plot(df.x, df.y, label=f'Particle ID: {p_id}, Name: Unknown')

    axes[0].set_xlabel('X (mm)');
    axes[0].set_ylabel('Y (mm)');
    
    axes[1].set_xlabel('X (mm)');
    axes[1].set_ylabel('Z (mm)');

    axes[2].set_xlabel('Y (mm)');
    axes[2].set_ylabel('Z (mm)');    
    axes[0].legend()
    if show:
        plt.show()



def plot_MC_over_hits(reco, MC, prt_info, evt, blob_info = None, show = True, pitch = 15.55, title = None):
    '''
    plot MC true over the hits

    df --> reco hits
    MC --> MC hits
    prt_info --> particle info
    blob_info --> tuple of dataframes with reco and MC
    '''

    q = reco[reco.event == evt]
    MC = MC[MC.event_id == evt]
    part_info = prt_info[prt_info.event_id == evt]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # plot reco hits first
    xx = np.arange(q.X.min(), q.X.max() + pitch, pitch)
    yy = np.arange(q.Y.min(), q.Y.max() + pitch, pitch)
    zz = np.sort(q.Z.unique())

    axes[0].hist2d(q.X, q.Y, bins=[xx, yy], weights=q.Q, cmin=0.0001);
    axes[0].set_xlabel('X (mm)');
    axes[0].set_ylabel('Y (mm)');

    axes[1].hist2d(q.X, q.Z, bins=[xx, zz], weights=q.Q, cmin=0.0001);
    axes[1].set_xlabel('X (mm)');
    axes[1].set_ylabel('Z (mm)');


    axes[2].hist2d(q.Y, q.Z, bins=[yy, zz], weights=q.Q, cmin=0.0001);
    axes[2].set_xlabel('Y (mm)');
    axes[2].set_ylabel('Z (mm)');

    for p_id, df in MC.groupby('particle_id'):
       # associate the p_id back to its original from the MC_df
       particle = part_info[part_info.particle_id == p_id]
       if not particle.empty:
           particle_name = particle.iloc[0].particle_name
           if particle_name not in plt.gca().get_legend_handles_labels()[1]:
               
               axes[0].plot(df.x, df.y, label=f'{particle_name}')
               axes[1].plot(df.x, df.z, label=f'{particle_name}')
               axes[2].plot(df.y, df.z, label=f'{particle_name}')
           else:
               # reuse color of the already-plotted particle
               ax = plt.gca()
               handles, labels = ax.get_legend_handles_labels()
               try:
                   idx = labels.index(particle_name)
                   color = handles[idx].get_color()
               except ValueError:
                   color = None
               plt.plot(df.x, df.y, color=color)
               plt.plot(df.x, df.z, color=color)
               plt.plot(df.y, df.z, color=color)
       else:
           plt.plot(df.x, df.y, label=f'Particle ID: {p_id}, Name: Unknown')


    # blob info
    if blob_info is not None:
        reco_blob_info = blob_info[0]
        MC_blob_info = blob_info[1]
        axes[0].scatter(reco_blob_info.blob1_x, reco_blob_info.blob1_y, label = 'reco b1', color = 'b', s = 400, marker='x', linewidths=2)
        axes[0].scatter(reco_blob_info.blob2_x, reco_blob_info.blob2_y, label = 'reco b2', color = 'r', s = 200, marker='x', linewidths=2)
        axes[0].scatter(MC_blob_info.blob1_x, MC_blob_info.blob1_y, label = f'MC b1', color = 'cyan', s = 400)
        axes[0].scatter(MC_blob_info.blob2_x, MC_blob_info.blob2_y, label = 'MC b2', color = 'magenta', s = 200)

        axes[1].scatter(reco_blob_info.blob1_x, reco_blob_info.blob1_z, label = 'reco b1', color = 'b', s = 400, marker='x', linewidths=2)
        axes[1].scatter(reco_blob_info.blob2_x, reco_blob_info.blob2_z, label = 'reco b2', color = 'r', s = 200, marker='x', linewidths=2)
        axes[1].scatter(MC_blob_info.blob1_x, MC_blob_info.blob1_z, color = 'cyan', s = 400)
        axes[1].scatter(MC_blob_info.blob2_x, MC_blob_info.blob2_z, color = 'magenta', s = 200)

        axes[2].scatter(reco_blob_info.blob1_y, reco_blob_info.blob1_z, label='reco b1', color='b', s=400, marker='x', linewidths=2)
        axes[2].scatter(reco_blob_info.blob2_y, reco_blob_info.blob2_z, label='reco b2', color='r', s=200, marker='x', linewidths=2)
        axes[2].scatter(MC_blob_info.blob1_y, MC_blob_info.blob1_z, color = 'cyan', s = 400)
        axes[2].scatter(MC_blob_info.blob2_y, MC_blob_info.blob2_z, color = 'magenta', s = 200)

    axes[0].set_xlabel('X (mm)');
    axes[0].set_ylabel('Y (mm)');
    
    axes[1].set_xlabel('X (mm)');
    axes[1].set_ylabel('Z (mm)');

    axes[2].set_xlabel('Y (mm)');
    axes[2].set_ylabel('Z (mm)');    


    axes[0].legend()


    if title is None:
        fig.suptitle(f'evt: {evt}')
    else:
        fig.suptitle(f'{title}')


    if show:
        plt.show()

