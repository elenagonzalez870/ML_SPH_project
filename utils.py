import numpy as np
from autograd import grad, hessian
import autograd.numpy as np
import matplotlib.pyplot as plt
from sklearn.inspection import DecisionBoundaryDisplay

def gradient_descent(g, x, y, w, n, alpha):

    gradient = grad(g)
    
    cost_history = [g(w,x,y)]
    weight_history = [w]

    for k in range(1,n):
        g_prime = gradient(w,x,y)
        w = w - alpha * g_prime
        
        cost_history.append(g(w,x,y))
        weight_history.append(w)  
    return cost_history, weight_history

def model(x,w):
    a = w[0] + np.dot(x.T, w[1:])
    return a.T

def standard_normalize(x):

    # Calculate mean and standard deviation accross rows (N)
    # x needs to have shape (points, N)
    nth_mean = np.nanmean(x, axis=1)[:,np.newaxis]
    nth_std  = np.nanstd(x, axis=1)[:,np.newaxis]

    x_normalized = (x-nth_mean)/nth_std
   
    # Check if there's any nan values (I got NaN values for the car dataset)
    nan_index = np.argwhere(np.isnan(x_normalized) == True)

    #Replace the nan's by 0
    for index in nan_index:
        x_normalized[index[0], index[1]] = 0

    return x_normalized, nth_mean, nth_std

def multiclass_perceptron(w,x,y):
    # pre-compute predictions on all points
    all_evals = model(x,w)

    # compute maximum cross data points 
    a = np.max(all_evals, axis = 0)

    # compute cost in compact form using numpy broadcasting 
    b = all_evals[y.astype(int).flatten(), np.arange(np.size(y))]

    cost = np.sum(a - b)

    # add regularizer 
    lam = 10**(-5)
    cost = cost + lam * np.linalg.norm(w[1:,:], 'fro') **2

    # return the average 
    return cost / float(np.size(y))

def missclassification_history(w, x, y):

    models = model(x, w)
    y_pred = np.argmax(models, axis = 0)
    miss_classifications = y_pred[y_pred != y[0]]

    return len(miss_classifications)

def perceptron_boundary_plotter(x_data, y_data, model, w_best, nth_mean, nth_std ):

    #--- Plot data ---#

    # Define conditions and corresponding colors
    conditions = [y_data == 0., y_data == 1., y_data == 2.]
    choices = ['orchid', 'orange', 'blue']

    # Apply mapping
    colors = np.select(conditions, choices, default = 'gray').flatten()

    f, ax = plt.subplots()

    plt.scatter(np.log10(x_data[0,:]),np.log10(x_data[1,:]),  c=colors, s = 15)
    plt.ylabel("log10(V_rel) [km/s]")
    plt.xlabel("log10(b) [RSUN]")

    #--- Plot predictions ---#

    # Create a mesh grid
    x_min, x_max = np.log10(x_data[0,:].min()) -0.1 , np.log10(x_data[0,:].max()) + 0.2
    y_min, y_max = np.log10(x_data[1,:].min()) - 1, np.log10(x_data[1,:].max()) + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                        np.arange(y_min, y_max, 0.1))


    # Predict classes for each point in the grid
    grid_points = np.vstack([xx.ravel(), yy.ravel()])

    # grid_points = standard_normalize(grid_points)
    grid_points_normalized = (grid_points-nth_mean)/nth_std

    Z = np.array(model(grid_points_normalized, w_best))
    Z = np.argmax(Z, axis = 0)
    Z = Z.reshape(xx.shape)  # Reshape to match grid shape

    # Plot decision boundaries (contour lines)
    colors = ['orchid', 'orange', 'blue']

    # Plot decision regions
    plt.contourf(xx, yy, Z, levels = [-0.5, 0.5, 1.5, 2.5], alpha=0.2,colors = colors)

    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='orchid', markerfacecolor='orchid', label='Disrupted', markersize=5),
        plt.Line2D([0], [0], marker='o', color='orange', markerfacecolor='orange', label='Merged', markersize=5),
        plt.Line2D([0], [0], marker='o', color='blue', markerfacecolor='blue', label='Unmerged', markersize=5)
    ]
    plt.legend(handles=legend_elements, title="Classes")
    
    return f

def svc_gridsearch_plotter(x_data, y_data, svc, svc_scaler,  X_test, y_test):

    #--- Plot predictions ---#
    
    # Create a mesh grid
    x_min, x_max = x_data[:,0].min() -1, x_data[:,0].max()+1
    y_min, y_max = x_data[:,1].min() - 0.5, x_data[:,1].max() +0.5

    xx, yy = np.meshgrid(np.arange(x_min,  x_max, 0.01),
                        np.arange(y_min,  y_max, 0.01))

    #transform the data
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid_scaled = svc_scaler.transform(X_grid)

    # Predict classes for each point in the grid
    best_model = svc.best_estimator_
    Z = best_model.predict(X_grid_scaled)
    Z = Z.reshape(xx.shape)


    class_colors = ['orchid', 'orange', 'blue']
    display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=Z, xlabel = "log10(b) [RSUN]", ylabel ="log10(V_rel) [km/s]")
    display.plot(cmap=plt.cm.colors.ListedColormap(class_colors), alpha = 0.5)  # Apply custom colors


    #--- Overplot the data ---#

    # Define conditions and corresponding colors
    # conditions = [y_data == 0., y_data == 1., y_data == 2.]
    conditions = [y_test== 0., y_test == 1., y_test == 2.]
    choices = ['orchid', 'orange', 'blue']

    # Apply mapping
    colors = np.select(conditions, choices, default = 'gray').flatten()

    # plt.scatter(x_data[:,0],x_data[:,1],  c=colors, s = 15,  edgecolor="black")
    plt.scatter(X_test[:,0],X_test[:,1],  c=colors, s = 15,  edgecolor="black")

    #--- Plot settings ---#

    plt.ylabel("log10(V_rel [km/s])")
    plt.xlabel("log10(b [RSUN])")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    return display


def svr_gridsearch_plotter_2d(x_data_encoded, x_data, y_data, svc, svc_scaler, svr, svr_scaler, encoder):

    #--- Create meshgrid ---#

    # Create a mesh grid
    x_min, x_max = x_data_encoded[:,0].min() - 1, x_data_encoded[:,0].max() + 1
    y_min, y_max = x_data_encoded[:,1].min() - 0.5, x_data_encoded[:,1].max() + 0.5
    z_min, z_max = y_data.min(), y_data.max()

    xx, yy = np.meshgrid(np.arange(x_min,  x_max, 0.01),
                        np.arange(y_min,  y_max, 0.01))

    X_grid = np.c_[xx.ravel(), yy.ravel()]


    #--- Send data to get classifications ---#

    #Transform with SVC Scaler to pass to the trained SVC
    X_grid_scaled = svc_scaler.transform(X_grid)

    best_model_svc = svc.best_estimator_
    zz = best_model_svc.predict(X_grid_scaled)
    zz = zz[:, np.newaxis]

    #--- Make Predictions ---#

    #Encode the classification results to send into trained SVR
    zz_encoded = encoder.fit_transform(zz).toarray()
    zz = zz_encoded.reshape(*xx.shape, 3)

    X_grid = np.c_[xx.ravel(), yy.ravel(), zz[:,:,0].ravel(), zz[:,:,1].ravel(), zz[:,:,2].ravel()]
    X_grid_scaled = svr_scaler.transform(X_grid)

    # Predict classes for each point in the grid
    best_model_svr = svr.best_estimator_
    Z = best_model_svr.predict(X_grid_scaled)
    Z = Z.reshape(xx.shape)

    #---Create a 3D plot---#

    display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=Z, xlabel = "log10(b) [RSUN]", ylabel ="log10(V_rel) [km/s]")
    display.plot(cmap='viridis', alpha = 0.5)  # Apply custom colors


    #--- Overplot the data ---#
    plt.scatter(x_data[:,0], x_data[:,1], c = y_data, cmap = 'viridis', edgecolor = 'black')


    #--- Plot settings ---#
    plt.ylabel("log10(V_rel [km/s])")
    plt.xlabel("log10(b [RSUN])")

    return display

def data_3d_plotter(x_data, y_data):
    # Create a 3D plot
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the surface
    scatter = ax.scatter(x_data[:,0], x_data[:,1], y_data, c =  x_data[:,2])

    # Add labels
    ax.set_xlabel('log10(b) [RSUN]')
    ax.set_ylabel('log10(V_rel) [km/s]')
    ax.set_zlabel('Fractional Mass Loss ', labelpad = 0, rotation=90)

    cbar = fig.colorbar(scatter)
    cbar.set_label('Class Label', rotation=270, labelpad=15)
    cbar.set_ticks([0, 1, 2])
    fig.tight_layout()

    return fig

def knn_gridsearch_plotter(x_data, y_data, neigh, knn_scaler,  X_test, y_test):

    #--- Plot predictions ---#
    
    # Create a mesh grid
    x_min, x_max = x_data[:,0].min() -1, x_data[:,0].max()+1
    y_min, y_max = x_data[:,1].min() - 0.5, x_data[:,1].max() +0.5

    xx, yy = np.meshgrid(np.arange(x_min,  x_max, 0.01),
                        np.arange(y_min,  y_max, 0.01))

    #transform the data
    X_grid = np.c_[xx.ravel(), yy.ravel()]
    X_grid_scaled = knn_scaler.transform(X_grid)

    # Predict classes for each point in the grid
    Z = neigh.predict(X_grid_scaled)
    Z = Z.reshape(xx.shape)


    class_colors = ['orchid', 'orange', 'blue']
    display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=Z, xlabel = "log10(b) [RSUN]", ylabel ="log10(V_rel) [km/s]")
    display.plot(cmap=plt.cm.colors.ListedColormap(class_colors), alpha = 0.5)  # Apply custom colors


    #--- Overplot the data ---#

    # Define conditions and corresponding colors
    conditions = [y_test== 0., y_test == 1., y_test == 2.]
    choices = ['orchid', 'orange', 'blue']

    # Apply mapping
    colors = np.select(conditions, choices, default = 'gray').flatten()

    plt.scatter(X_test[:,0],X_test[:,1],  c=colors, s = 15,  edgecolor="black")

    #--- Plot settings ---#

    plt.ylabel("log10(V_rel [km/s])")
    plt.xlabel("log10(b [RSUN])")

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    return display

def N_knn_gridsearch_plotter(X_train, y_train, X_test, y_test, neigh, knn_scaler,feature_idx=(0, 1), fixed_values={}, labels = []):
    import matplotlib.colors as mcolors
    
    # Step 1: Filter train and test data based on fixed_values
    
    train_mask = np.all(np.array([np.isclose(X_train[:, k], v, rtol=0.09) for k, v in fixed_values.items()]), axis=0)
    test_mask = np.all(np.array([np.isclose(X_test[:, k], v, rtol=0.09) for k, v in fixed_values.items()]), axis=0)


    X_train_filtered, y_train_filtered = X_train[train_mask], y_train[train_mask]
    X_test_filtered, y_test_filtered = X_test[test_mask], y_test[test_mask]

    #--- Plot predictions ---#
    
    # Create a mesh grid
    # Step 2: Create a meshgrid for the chosen two features
    x_min, x_max = X_train[:, feature_idx[0]].min() - 0.1, X_train[:, feature_idx[0]].max() + 0.1
    y_min, y_max = X_train[:, feature_idx[1]].min() - 0.1, X_train[:, feature_idx[1]].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Step 3: Construct full-dimensional input space for predictions
    X_grid = np.zeros((xx.ravel().shape[0], X_train.shape[1]))
    X_grid[:, feature_idx[0]] = xx.ravel()
    X_grid[:, feature_idx[1]] = yy.ravel()

    # Assign fixed values to other dimensions
    for k, v in fixed_values.items():
        X_grid[:, k] = v  

    #transform the data
    X_grid_scaled = knn_scaler.transform(X_grid)
 
    #Predict labels for meshgrid
    Z = neigh.predict(X_grid_scaled)
    Z = Z.reshape(xx.shape)

    # Normalize color mapping to match the scatter points and contour
    cmap = plt.cm.coolwarm  # Set the same colormap
    norm = mcolors.Normalize(vmin=Z.min(), vmax=Z.max())  # Normalize colors

    # Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, norm=norm)

    #--- Overplot the data ---#

    plt.scatter(X_train_filtered[:, feature_idx[0]], X_train_filtered[:, feature_idx[1]],  c=y_train_filtered,  cmap='coolwarm',  edgecolor="k", marker='o', label = 'Train' )
    plt.scatter(X_test_filtered[:, feature_idx[0]], X_test_filtered[:, feature_idx[1]],  c=y_test_filtered, cmap='coolwarm', edgecolor="k" , marker='x', label = 'Test')

    #--- Plot settings ---#

    plt.title(f"KNN: Decision Boundary Mass1 = {labels[2]} Msun , Mass2 = {labels[3]}Msun")
    plt.xlabel(f" {labels[0]}")
    plt.ylabel(f" {labels[1]}")
    plt.legend()
    plt.show()

def svr_gridsearch_plotter(x_data_encoded, x_data, y_data, svc, svc_scaler, svr, svr_scaler, encoder):
    from itertools import combinations
   
    feature_pairs = list(combinations(range(x_data.shape[1]), 2))  # All 2D feature pairs

    for f1, f2 in feature_pairs:
        plt.figure(figsize=(6, 5))

        #--- Create meshgrid ---#

        # Create a mesh grid
        x_min, x_max = x_data_encoded[:,f1].min() - 1, x_data_encoded[:,f1].max() + 1
        y_min, y_max = x_data_encoded[:,f2].min() - 0.5, x_data_encoded[:,f2].max() + 0.5
        z_min, z_max = y_data.min(), y_data.max()

        xx, yy = np.meshgrid(np.arange(x_min,  x_max, 0.01),
                            np.arange(y_min,  y_max, 0.01))

        # Create 4D input grid (fixed values for remaining features)
        X_grid = np.zeros((xx.size, x_data.shape[1]))
        X_grid[:, f1] = xx.ravel()
        X_grid[:, f2] = yy.ravel()


        #--- Send data to get classifications ---#

        #Transform with SVC Scaler to pass to the trained SVC
        X_grid_scaled = svc_scaler.transform(X_grid)

        best_model_svc = svc.best_estimator_
        zz = best_model_svc.predict(X_grid_scaled)
        zz = zz[:, np.newaxis]

        #--- Make Predictions ---#

        #Encode the classification results to send into trained SVR
        zz_encoded = encoder.fit_transform(zz).toarray()
        zz = zz_encoded.reshape(*xx.shape, 3)

        X_grid = np.c_[xx.ravel(), yy.ravel(), zz[:,:,0].ravel(), zz[:,:,1].ravel(), zz[:,:,2].ravel()]
        X_grid_scaled = svr_scaler.transform(X_grid)

        # Predict classes for each point in the grid
        best_model_svr = svr.best_estimator_
        Z = best_model_svr.predict(X_grid_scaled)
        Z = Z.reshape(xx.shape)

        #---Create a 3D plot---#

        display = DecisionBoundaryDisplay(xx0=xx, xx1=yy, response=Z, xlabel = "log10(b) [RSUN]", ylabel ="log10(V_rel) [km/s]")
        display.plot(cmap='viridis', alpha = 0.5)  # Apply custom colors


        #--- Overplot the data ---#
        plt.scatter(x_data[:,0], x_data[:,1], c = y_data, cmap = 'viridis', edgecolor = 'black')


        #--- Plot settings ---#
        plt.ylabel("log10(V_rel [km/s])")
        plt.xlabel("log10(b [RSUN])")

    return display

def plot_4d_decision_boundary(model, X_train, y_train, X_test, y_test, feature_idx=(0, 1), fixed_values={}, labels = []):
    """
    Plots the decision boundary for a PyTorch model using a 2D slice of a higher-dimensional space.
    
    Parameters:
    - model: Trained PyTorch model.
    -X, y:  data and labels.
    - feature_idx: Tuple (i, j) specifying which two features to plot.
    - fixed_values: Dictionary {feature_index: value} for fixing other dimensions.
    labels = list of strings 
    """

    # Step 1: Filter train and test data based on fixed_values
    
    train_mask= np.all([np.isclose(X_train[:, k], v, rtol = 0.09) for k, v in fixed_values.items()], axis=0)
    test_mask= np.all([np.isclose(X_test[:, k], v, rtol = 0.09) for k, v in fixed_values.items()], axis=0) 

    X_train_filtered, y_train_filtered = X_train[train_mask], y_train[train_mask]
    X_test_filtered, y_test_filtered = X_test[test_mask], y_test[test_mask]

    # Step 2: Create a meshgrid for the chosen two features
    x_min, x_max = X_train[:, feature_idx[0]].min() - 0.1, X_train[:, feature_idx[0]].max() + 0.1
    y_min, y_max = X_train[:, feature_idx[1]].min() - 0.1, X_train[:, feature_idx[1]].max() + 0.1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))

    # Step 3: Construct full-dimensional input space for predictions
    X_vis = torch.zeros((xx.ravel().shape[0], X_train.shape[1]), dtype=torch.float32)
    X_vis[:, feature_idx[0]] = torch.tensor(xx.ravel(), dtype=torch.float32)
    X_vis[:, feature_idx[1]] = torch.tensor(yy.ravel(), dtype=torch.float32)

    # Assign fixed values to other dimensions
    for k, v in fixed_values.items():
        X_vis[:, k] = v

    # Step 4: Predict labels for meshgrid
    model.eval()
    with torch.no_grad():
        Z = model(X_vis).argmax(dim=1).numpy()
    Z = Z.reshape(xx.shape)

    # Normalize color mapping to match the scatter points and contour
    cmap = plt.cm.coolwarm  # Set the same colormap
    norm = mcolors.Normalize(vmin=y_train.min(), vmax=y_train.max())  # Normalize colors

    # Step 5: Plot decision boundary
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap, norm=norm)
    plt.scatter(X_train_filtered[:, feature_idx[0]], X_train_filtered[:, feature_idx[1]], c=y_train_filtered, cmap='coolwarm', edgecolor="k", marker='o', label = 'Train')
    plt.scatter(X_test_filtered[:, feature_idx[0]], X_test_filtered[:, feature_idx[1]], c=y_test_filtered, cmap='coolwarm', edgecolor="k", marker='x', label = 'Test')
    plt.xlabel(f" {labels[0]}")
    plt.ylabel(f" {labels[1]}")
    plt.legend()
    plt.title(f"NN: Decision Boundary Mass1 = {labels[2]} Msun , Mass2 = {labels[3]}Msun")
    plt.show()

def gen_confusion_matrix(y_test, y_pred, title, ax=None):
    from sklearn.metrics import confusion_matrix
    import matplotlib.pyplot as plt
    import matplotlib
    import numpy as np
    
    matplotlib.rcParams['text.usetex'] = True
    
    # Compute confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    # Plot the heatmap using imshow
    im = ax.imshow(cm_norm, cmap="Blues", aspect='auto', vmin=0, vmax=1)
    
    # Manually add text annotations with LaTeX
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            percent = cm_norm[i, j] * 100
            text = rf"${percent:.1f}\%$" + "\n" + rf"$({cm[i,j]}/{cm[i].sum()})$"
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, text, ha="center", va="center", 
                   fontsize=16, color=color)
    
    # Set tick labels with LaTeX
    unique_labels = np.unique(y_test)
    ax.set_xticks(np.arange(len(unique_labels)))
    ax.set_yticks(np.arange(len(unique_labels)))
    ax.set_xticklabels([rf"${label}$" for label in unique_labels], fontsize=18)
    ax.set_yticklabels([rf"${label}$" for label in unique_labels], fontsize=18)
    
    ax.set_xlabel(r"$\rm Predicted$", fontsize=20)
    ax.set_ylabel(r"$\rm True$", fontsize=20)
    ax.set_title(title, fontsize=22)
    
    return ax