import matplotlib.pyplot as plt
from celluloid import Camera
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from precision import np_float_precision, torch_float_precision
from case import Case
import torch
import time
import scipy
import numpy as np


def figure_to_data(fig):
    """
    A function to convert matplotlib fig to numpy array
    Return a numpy array.
    Args:
        fig: The figure to convert.
    """
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
    X = buf.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    X = np.transpose(X, (2, 0, 1))
    return X

def compute_norm(u, case, xx, yy, zz=None, refine_coef=0., r=1):
        if case==Case.circle:            
            dx = r * refine_coef
            if zz == None:
                norm_u = np.linalg.norm(u.reshape(xx.shape+(2,))[xx**2+yy**2<r**2-2*dx], axis=1)
            else:
                norm_u = np.linalg.norm(u.reshape(xx.shape+(2,))[xx**2+yy**2+zz**2<r**2-2*dx], axis=1)
        else: # if boundary_dict["case"]==Case.rectangle: 
            norm_u = np.linalg.norm(u, axis=1)

        return norm_u

def scale_u(u, case, xx, yy, zz=None, vector_func=True, tol=1e-1, r=1):    
    if case==Case.circle:        
        if zz == None:
            scale = (xx**2+yy**2<r**2 + tol) * 1.
        else:
            scale = (xx**2+yy**2+zz**2<r**2 + tol) * 1.
        if vector_func:
            if zz == None:
                scale = scale.reshape(u.shape[0],1)
                scale = np.repeat(scale, 2, axis=1)
            else:
                scale = scale.reshape(u.shape[0],1)
                scale = np.repeat(scale, 3, axis=1)
            return scale * u + (1e-2)*(1-scale)*u
        return scale * u

    return u


def make_meshgrid(x_bounds, y_bounds, Nx=50, Ny=None, include_boundary=True):
    if Nx < 1:
        raise RuntimeError("In function make_meshgrid Nx should be greater than 1.")
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds
    if Ny is None:
        Ny = Nx
    dx = (x_max - x_min) / Nx
    dy = (y_max - y_min) / Ny
    #x_min, x_max = min(x_min,-bound[0]) - h1, max(x_max, bound[0]) + h1
    #y_min, y_max =  min(y_min,-bound[1]) - h2, max(y_max, bound[1]) + h2
    if include_boundary:
        xx, yy = np.meshgrid(np.linspace(x_min, x_max, Nx, endpoint=True),
                             np.linspace(y_min, y_max, Ny, endpoint=True))
        
        # xx_bound = np.logical_or(xx == -1., xx==1.)
        # yy_bound = np.logical_or(yy == -1., yy==1.)        
        # corners = np.logical_and(xx_bound, yy_bound)
        # xx = xx[~corners]
        # yy = yy[~corners]
        
    else:
        xx, yy = np.meshgrid(np.linspace(x_min+dx, x_max, Nx),
                             np.linspace(y_min+dy, y_max, Ny))
    mesh = np_float_precision(np.c_[xx.ravel(), yy.ravel()])
    return mesh, xx, yy


def make_meshgrid2(bounds, Nx=[50], include_boundary=True):    
    if not isinstance(Nx, list):
        Nx = [Nx]*len(bounds)
    if include_boundary:
        lsp = [np.linspace(*bi, Nx[i], endpoint=True) for i, bi in enumerate(bounds)]
    else:
        bounds2 = np.array(bounds)
        bounds2[:,0] += (bounds2[:,1]-bounds2[:,0])/(np.array(Nx))
        lsp = [np.linspace(*bi, Nx[i], endpoint=True) for i, bi in enumerate(list(bounds2))]        
    x_y = np.meshgrid(*lsp)
    if len(x_y) == 2:            
        mesh = np_float_precision(np.c_[x_y[0].ravel(), x_y[1].ravel()])
        return mesh, x_y[0], x_y[1]
    else:
        mesh = np_float_precision(np.c_[x_y[0].ravel(), x_y[1].ravel(), x_y[2].ravel()])
        return mesh, x_y[0], x_y[1], x_y[2]

def mesh_from_trajectories(x_trajectories, y_trajectories, z_trajectories=None, global_bound=0, tol=0., Nx=18):
    x_min, x_max = (min(x_trajectories.min(),  -global_bound) - tol, 
                    max(x_trajectories.max(), global_bound) + tol)
    y_min, y_max = (min(y_trajectories.min(), -global_bound) - tol, 
                    max(y_trajectories.max(), global_bound) + tol)
    if z_trajectories is not None:
        z_min, z_max = (min(y_trajectories.min(), -global_bound) - tol, 
                        max(y_trajectories.max(), global_bound) + tol)
        return make_meshgrid2([[x_min, x_max], [y_min, y_max], [z_min, z_max]], Nx=Nx)
    else:
        return make_meshgrid2([[x_min, x_max], [y_min, y_max]], Nx=Nx)    
    
    

def animate_particles(V, mesh_x_y, trajectories, Y, dt,
                      problem_type="classification"):
    """
    Save plots of the velocity field.
    Args:
            V: the velocity field
            mesh_x_y: The list of coordinates [x, y] associated to the plot.
            trajectories: the trajectories.
            Y: Y the labels (or the target points for regression).
            dt: the time step.
            problem_type: the type of problem (e.g. classification ,regression, normalizing flows).
    """

    # Plot the trajectories.
    figures = []
    t = 0
    
    fig = plt.figure()
    camera = Camera(fig)
    
    for i in range(len(V)):
        X = trajectories[i]
        Vi = V[i]
        if Case.classification == problem_type:
            plot_velocity_field(Vi, mesh_x_y, [X, Y], t,
                                problem_type=problem_type)
        else:
            plot_velocity_field(Vi, mesh_x_y, [X, Y], t,
                                problem_type=problem_type)
        t += dt
        camera.snap()

    animation = camera.animate(interval=100)    
    plt.close()
    return animation


def plot_function(x, y, f, callback, logger, name='', constraint_out_dom = False, boundary_dict=None, contour=False):
    if constraint_out_dom:
        f = scale_u(f, boundary_dict["case"], x, y, vector_func=False, tol=-5e-2, r=boundary_dict["radius"])
    fig, ax=plt.subplots()
    cm = plt.cm.RdBu
    if not contour:
        p = ax.pcolormesh(x, y, f, cmap=plt.cm.bwr,shading='gouraud')#,vmin=abs(Z).min(),vmax=abs(Z).max())
    else:
        p = ax.contourf(x, y, f)
    cb = fig.colorbar(p, ax=ax)
    plt.close()
    callback.add_tensorboard_figure(logger, fig, name)


def plot_scatter(x, y, f, callback, logger, name=''):
    fig = plt.figure()
    plt.scatter(x, y, c=f, alpha=0.3, cmap=plt.cm.bwr)
    plt.colorbar()
    plt.close()
    callback.add_tensorboard_figure(logger, fig, name=name)


def plot_velocity_field(V, mesh_x_y, X_Y, plots_points=True,
                        normalize=True, problem_type="classification", colorbar_range=None, change_color=False, plot_norm=True):
    if V.shape[1] == 2:
        return plot_velocity_field_2D(V, mesh_x_y, X_Y, plots_points,
                        normalize, problem_type, colorbar_range, change_color, plot_norm)
    else:
        return plot_velocity_field_3D(V, mesh_x_y, X_Y, plots_points,
                        normalize, problem_type, colorbar_range, change_color, plot_norm)


def plot_velocity_field_2D(V, mesh_x_y, X_Y, plots_points=True,
                        normalize=True, problem_type="classification", colorbar_range=None, change_color=False, plot_norm=True):
    """
    Plot of the velocity field.
    Args:
            V: the velocity field
            mesh_x_y: The list of coordinates [x, y] associated to the plot.
            X_Y: The list [X, Y] wich will be used during the plots: X are the
            data and Y the labels (or the target points for regression).
            normalize: True if the velocity field is normalized on the plot.
            plots_points: True if one wants to plot the points X_Y on the
            velocity field.
            problem_type: the type of the problem.
    """    

    xx, yy = mesh_x_y
    Vx = V[:, 0].reshape(xx.shape)#.detach().numpy().reshape(xx.shape)
    Vy = V[:, 1].reshape(yy.shape)#.detach().numpy().reshape(yy.shape)
    #fig = plt.figure()
    V_norm = np.sqrt(Vx**2 + Vy**2)

    if normalize:
        Vx = Vx / (V_norm+1e-5)
        Vy = Vy / (V_norm+1e-5)

    if plot_norm:
        if colorbar_range is None:
            cf = plt.contourf(xx, yy, V_norm, cmap="YlGn")
        else:
            vmin, vmax = colorbar_range
            levels = np.linspace(vmin, vmax+(vmax-vmin)/20., 7)
            cf = plt.contourf(xx, yy, V_norm, cmap="YlGn",  levels=levels, extend='both')#, vmin=vmin, vmax=vmax +(vmax-vmin)/20.)
        plt.colorbar().remove()
        plt.colorbar(cf)
    else:
        plt.contourf(xx, yy, 3*np.ones(xx.shape), cmap="YlGn", extend='both')#, vmin=vmin, vmax=vmax +(vmax-vmin)/20.)

    
        # plt.streamplot(xx, yy, Vx, Vy, color="black")
        # plt.quiver(xx, yy, Vx, Vy)
    qk = plt.quiver(xx, yy, Vx, Vy)#, units='width')
    
    # scale_units="inches", scale=4, width=0.005, pivot="mid")
    # #, V_norm, cmap="autumn", scale_units="inches",
    # scale=5, width=0.015, pivot="mid")
    if plots_points:
        plot_points(X_Y, problem_type, change_color)
        
def plot_velocity_field_3D(V, mesh_x_y, X_Y, plots_points=True,
                        normalize=True, problem_type="classification", colorbar_range=None, change_color=False, plot_norm=True):
    """
    Plot of the velocity field.
    Args:
            V: the velocity field
            mesh_x_y: The list of coordinates [x, y] associated to the plot.
            X_Y: The list [X, Y] wich will be used during the plots: X are the
            data and Y the labels (or the target points for regression).
            normalize: True if the velocity field is normalized on the plot.
            plots_points: True if one wants to plot the points X_Y on the
            velocity field.
            problem_type: the type of the problem.
    """    

    xx, yy, zz = mesh_x_y
    Vx = V[:, 0].reshape(xx.shape)
    Vy = V[:, 1].reshape(yy.shape)
    Vz = V[:, 2].reshape(zz.shape)
    
    V_norm = np.sqrt(Vx**2 + Vy**2 + Vz**2)
    """
    if normalize:
        Vx = Vx / (V_norm+1e-5)
        Vy = Vy / (V_norm+1e-5)
        Vy = Vz / (V_norm+1e-5)
    
    if plot_norm:
        if colorbar_range is None:
            cf = plt.contour3D(xx, yy, zz, V_norm, cmap="YlGn")
        else:
            vmin, vmax = colorbar_range
            levels = np.linspace(vmin, vmax+(vmax-vmin)/20., 7)
            cf = plt.contour3D(xx, yy, zz, V_norm, cmap="YlGn",  levels=levels, extend='both')#, vmin=vmin, vmax=vmax +(vmax-vmin)/20.)
        plt.colorbar().remove()
        plt.colorbar(cf)
    else:
        plt.contour3D(xx, yy, zz, 3*np.ones(xx.shape), cmap="YlGn", extend='both')#, vmin=vmin, vmax=vmax +(vmax-vmin)/20.)
    """
    
        # plt.streamplot(xx, yy, Vx, Vy, color="black")
        # plt.quiver(xx, yy, Vx, Vy)    
    #plt.figure().gca(projection='3d')
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # Repeat for each body line and two head lines
    V_norm = V_norm#np.concatenate((V_norm.ravel(), np.repeat(V_norm.ravel(), 2)))
    # Colormap
    c = plt.cm.YlGn(V_norm.reshape(-1))
    q = ax.quiver(xx, yy, zz, Vx, Vy, Vz, length=0.1, normalize=True)#, colors=c, units='width')
    #q.set_array(np.linspace(0,V_norm.max(),10))
    #fig.colorbar(q)
    
    
    #if plots_points:
    #    plot_points(X_Y, problem_type, change_color)


def plot_points(X_Y, problem_type, change_color=False):
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    X, Y = X_Y        
    if Case.classification == problem_type:# or Case.normalizing_flows == problem_type:
        Y = Y.reshape(-1)
        if change_color:
            add_color = 2
        else:
            add_color = 0
        plt.scatter(X[:, 0], X[:, 1], c=Y+add_color, cmap=cm_bright,
                    edgecolors='k', s=10, linewidths=0.5)
    else:
        col = np.full(X.shape[0], 0).reshape(-1,1)
        
        if change_color:
            color = 'green'
        else:
            color = 'red'

        plt.scatter(X[:, 0], X[:, 1], c=color, cmap=cm_bright,
                    edgecolors='k', s=10, linewidths=0.5)
        if  Y is not None:
            col += 1
            if change_color:
                color = 'yellow'
            else:
                color = 'blue'
            plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=cm_bright,
                        edgecolors='k', s=10, linewidths=0.3)

def save_velocity_field(callback, logger, V, mesh_x_y, normalize=False, colorbar_range=None,
                        boundary_dict={"plot": False}, name=''):
    
    
    # ax = fig.gca(projection='3d')  
    plot_velocity_field(V, mesh_x_y, [None, None], plots_points=False,
                              normalize=normalize, colorbar_range=colorbar_range)
    fig = plt.gcf()
    #fig = plt.figure()
    if boundary_dict["plot"]:
        if boundary_dict["case"] == Case.circle:
            radius = boundary_dict["radius"]
            circle = plt.Circle((0, 0), radius, color='b', fill=False)
            plt.gca().add_patch(circle)
            
        elif boundary_dict["case"] == Case.rectangle:
            pass
            
            # bounds = boundary_dict["bounds"] 
            # x_min, x_max, y_min, y_max = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1] 
            # rectangle = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
            #                          color='b', fill=False)
            # plt.gca().add_patch(rectangle)
            
        else:
            raise RuntimeError("Unknown boundary domain.")                            
    
    # fig = plt.figure()
    # ax = fig.gca(projection='3d')

    
     
    # plt.show()
    callback.add_tensorboard_figure(logger, fig, name)
    plt.close()

def save_velocity_fields(V, mesh_x_y, callback, logger, trajectories_label = [None, None], trajectories_label_test = [None, None],
                         problem_type="classification", name='', boundary_dict={"plot":False}, normalize=True, constraint_out_dom=False,
                         plot_norm=True):
    """
    Save plots of the velocity field.
    Args:
            V: the velocity field
            mesh_x_y: The list of coordinates [x, y] associated to the plot.
            trajectories: the trajectories.
            Y: Y the labels (or the target points for regression).
            dt: the time step.
            callback: CallBack object used to save the plot.
            pl_module: the pytorch lightning module.
            problem_type: the type of problem (e.g. classification ,regression, normalizing flows).
            name: a string to name the plots.
    """
    trajectories, Y = trajectories_label
    trajectories_test, Y_test = trajectories_label_test
    # Plot the velocity field only (no data).
    #save_velocity_field(callback, logger, V[0], mesh_x_y, boundary_dict=boundary_dict, name=name+"initial velocity field")
    """
    fig = plt.figure()
    plot_velocity_field(V[0], mesh_x_y, [None, None], plots_points=False,
                              normalize=False, problem_type=problem_type)
    plt.close()
    callback.add_tensorboard_figure(logger, fig, "velocity field")
    """
    # Plot the trajectories.
    figures = []

    norm_v_max = 0.
    if len(mesh_x_y)==3:
        zz = mesh_x_y[2]
    else:
        zz = None
    for i in range(len(V)):
        Vi = V[i]
        norm_v = compute_norm(Vi, boundary_dict["case"], mesh_x_y[0], mesh_x_y[1], zz=zz, r=boundary_dict["radius"])
        norm_v_max = max(norm_v.max(), norm_v_max)
    
    for i in range(len(V)):        
        if trajectories != None:
            X = trajectories[i]
            plots_points = True
        else:
            X = None
            plots_points = False
        Vi = V[i]
        if constraint_out_dom:
            v = scale_u(Vi, boundary_dict["case"], mesh_x_y[0], mesh_x_y[1], zz=zz, r=boundary_dict["radius"])
        else:
            v = Vi
        
        plot_velocity_field(v, mesh_x_y, [X, Y],
                            problem_type=problem_type,
                            colorbar_range=[0., norm_v_max], 
                            plots_points=plots_points, normalize=normalize,
                            plot_norm=plot_norm)
        fig = plt.gcf()
        if trajectories_test != None:
            X_test = trajectories_test[i]
            plot_points([X_test, Y_test], problem_type=problem_type, change_color=True)
        #figure, ax = plt.subplots()
        
        if boundary_dict["plot"]:
            if boundary_dict["case"] == Case.circle:
                radius = boundary_dict["radius"]
                circle = plt.Circle((0, 0), radius, color='b', fill=False)
                plt.gca().add_patch(circle)
            elif boundary_dict["case"] == Case.rectangle:
                pass
                """
                bounds = boundary_dict["bounds"]
                x_min, x_max, y_min, y_max = bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]
                rectangle = plt.Rectangle((x_min, y_min), x_max-x_min, y_max-y_min,
                                          color='b', fill=False)
                plt.gca().add_patch(rectangle)
                """
            else:
                raise RuntimeError("Unknown boundary domain.")

        figures.append(figure_to_data(fig))
        plt.close()
    
    callback.add_tensorboard_video(logger, figures,
                                   name + " Particles motion")
    

def plot_boundary_decision(X_Y, pl_module, callback, name, global_bound=0, Nx=18, refine=False):
    """
    Save plot of the boundary decisions when the test case is two spirals.
    Args:
            X_Y: the list [X, Y] wich will be used during the plots: X are the
            data and Y the labels.
            pl_module: the pytorch lightning module.
            callback: CallBack object used to save the plot.
            name: a string to name the plot.
    """
    X, Y = X_Y
    mesh, xx, yy = mesh_from_trajectories(X[:, 0], X[:, 1], global_bound=global_bound, Nx=Nx, tol=0.2)
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])

    Z = pl_module.predict(torch.tensor(mesh, dtype= torch_float_precision))
    Z = Z.detach().numpy().reshape(xx.shape)
    fig = plt.figure()
    if refine:
        xx_copy = scipy.ndimage.zoom(xx, 3)
        yy_copy = scipy.ndimage.zoom(yy, 3)
        Z_copy = scipy.ndimage.zoom(Z, 3)
        plt.contourf(xx_copy, yy_copy, np.round(Z_copy), cmap=cm, alpha=0.8)
    else:
        plt.contourf(xx, yy, Z, cmap=cm, alpha=0.8)
    
    Y = Y.reshape(-1)
    plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=cm_bright,
                edgecolors='k')
    plt.close()
    callback.add_tensorboard_figure(pl_module.get_logger(), fig, name)


def save_trajectories(trajectories, Y, T, callback, logger, name="Trajectories 2d projection"):
    """
    Save 2d projection of the trajecories.
    Args:
            trajectories: the trajectories
            Y: labels (or target points for regression)
            T: The discrete times of the trajectories
            problem_type: the type of problem (e.g. classification ,regression, normalizing flows).
            callback: CallBack object used to save the plot.
            pl_module: the pytorch lightning module.
            name: a string to name the plots.
    """
    
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax = Axes3D(fig)
    Tn = np.array(T).reshape(-1, 1)
    T_np = np.array([Ti.repeat(trajectories[0].shape[0]) for Ti in Tn])
    T_np = np.swapaxes(T_np, 0, 1)
    traj = np.swapaxes(np.array(trajectories), 0, 1)

    for i in range(traj.shape[0]):
        current_traj = traj[i,:,:]
        if Y[i] ==0:
            color = 'red'
        else:
            color = 'blue'
        ax.scatter(current_traj[0, 0], T_np[i,0], current_traj[0, 1], alpha=1, c=color)
        ax.scatter(current_traj[-1, 0], T_np[i,-1], current_traj[-1, 1], alpha=0.1, c=color)
        ax.plot(current_traj[1:-1, 0], T_np[i,1:-1], current_traj[1:-1, 1], alpha=0.1, c=color)

    callback.add_tensorboard_figure(logger, fig, name)