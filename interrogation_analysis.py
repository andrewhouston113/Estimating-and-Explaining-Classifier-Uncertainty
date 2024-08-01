import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import scipy.stats as stats
from AITIA.uncertainty_system import MetaUncertaintyEstimator
import random
import pandas as pd
import seaborn as sns

class UncertaintyExplainer:
    """
    UncertaintyExplainer is a class for explaining model uncertainty using SHAP values.

    Parameters:
    - uncertainty_system: An instance of the uncertainty model to be explained.
    - M: Number of Monte Carlo samples for SHAP value estimation.

    Methods:
    - explain(self, X_train, x, feature_names=[], level='meta'):
        Explains model uncertainty using SHAP values and visualizes the explanations.

    Parameters:
    - X_train: Training data used to explain the model.
    - x: Data point for which uncertainty is to be explained.
    - feature_names: List of feature names. If not provided, default names are used.
    - level: 'meta' for explaining meta-uncertainty, 'instance' for explaining instance-level uncertainty.

    Attributes:
    - uncertainty_system: The uncertainty model to be explained.
    - M: Number of Monte Carlo samples for SHAP value estimation.

    Example:
    # Create an instance of UncertaintyExplainer
    explainer = UncertaintyExplainer(uncertainty_system=my_uncertainty_model, M=100)

    # Explain model uncertainty for a data point
    explainer.explain(X_train=data_train, x=data_point, feature_names=['Feature1', 'Feature2'], level='meta')
    """
    def __init__(self, uncertainty_system = None, M=100):
        self.uncertainty_system = uncertainty_system
        self.M = M
    
    def explain(self, X_train, x, feature_names=[], level='meta'):
        """
        Explains model uncertainty using SHAP values and visualizes the explanations.

        Parameters:
        - X_train: Training data used to explain the model.
        - x: Data point for which uncertainty is to be explained.
        - feature_names: List of feature names. If not provided, default names are used.
        - level: 'meta' for explaining meta-uncertainty, 'instance' for explaining instance-level uncertainty.
        """
        # If feature names are not provided, use default names like 'Feature 0', 'Feature 1', etc.
        if not feature_names:
            for i in range(X_train.shape[1]):
                feature_names.append(f'Feature {i}')

        # If explaining meta-uncertainty, calculate heuristics and use a MetaUncertaintyEstimator
        if level == 'meta':
            heuristics = self.uncertainty_system.heuristics_calculator.calculate(x)
            meta_uncertainty_system = MetaUncertaintyEstimator(self.uncertainty_system)
            shap_values = self._explain_features(meta_uncertainty_system, self.uncertainty_system.heuristics, heuristics)
        else:
            # If explaining instance-level uncertainty, use the provided uncertainty_system directly
            shap_values = self._explain_features(self.uncertainty_system, X_train, x)
        
        # Check if there are multiple data points for visualization
        if x.shape[0] > 1:
            if level == 'meta':
                # Visualize meta-uncertainty using a beeswarm plot
                self.beeswarm_plot(heuristics, shap_values, feature_names=['KDN','DS','DCD','OL','CLOL','HD','EC'])
                #self.bar_plot(shap_values, feature_names=['KDN','DS','DCD','OL','CLOL','HD','EC'])
            else:
                # Visualize instance-level uncertainty using a beeswarm plot
                self.beeswarm_plot(x, shap_values, feature_names=feature_names)
                #self.bar_plot(shap_values, feature_names=feature_names)

        else:
            if level == 'meta':
                # Visualize meta-uncertainty for a single data point using a force plot
                self.force_plot(heuristics[0], shap_values[0], np.mean(self.uncertainty_system.predict(X_train)), feature_names=['KDN','DS','DCD','OL','CLOL','HD','EC'])
            else:
                # Visualize instance-level uncertainty for a single data point using a force plot
                self.force_plot(x[0], shap_values[0], np.mean(self.uncertainty_system.predict(X_train)), feature_names=feature_names)


    def _explain_features(self, uncertainty_system, X_train, x):
        """
        Calculate Shapley values for all features for all instances in x.

        Parameters:
        - f: Uncertainty Explanation object.
        - X_train (np.ndarray): Training dataset used to sample instances.
        - x (np.ndarray): Instance(s) for which the Shapley values are calculated.
        - M (int): Number of iterations to estimate the Shapley values (default is 100).

        Returns:
        - list of lists: Shapley values for all features for each instance in x.
        """
        shapley_values_for_all_instances = []
        n_features = x.shape[1]

        # Iterate over instances in x
        for instance in tqdm(x, desc="Explaining Instances", unit="instance"):
            shapley_values_for_instance = []

            # Get the number of features in the instance

            # Calculate Shapley value for each feature
            for j in range(n_features):
                shapley_value = self._calculate_shapley_value(X_train, uncertainty_system, instance.reshape(1,-1), j)
                shapley_values_for_instance.append(shapley_value)

            shapley_values_for_all_instances.append(shapley_values_for_instance)

        return np.array(shapley_values_for_all_instances)

    def _calculate_shapley_value(self, X_train, f, x, j):
        """
        Calculate the Shapley value for a specific feature j using a random subset of other features.

        Parameters:
        - X_train (np.ndarray): Training dataset used to sample instances.
        - f: Uncertainty Explaination object
        - x (np.ndarray): Instance(s) for which the Shapley value is calculated.
        - j (int): Index of the feature for which the Shapley value is calculated.
        - M (int): Number of iterations to estimate the Shapley value (default is 100).

        Returns:
        - float: Shapley value for the specified feature.
        """
        # Check if x has the correct shape
        if x.shape[0] != 1:
            raise ValueError("Input instance x must have shape (1, n).")

        # Get the number of features in the instance x
        n_features = x.shape[1]
        
        # Initialize an empty list to store marginal contributions
        marginal_contributions = []
        
        # Create a list of feature indices, excluding the feature of interest (j)
        feature_idxs = list(range(n_features))
        feature_idxs.remove(j)

        # Perform M iterations to estimate Shapley value
        for _ in range(self.M):
            # Sample a random index to get a random instance from X_train
            random_idx = random.randint(0, len(X_train) - 1)
            z = X_train[random_idx]
            
            # Randomly select a subset of features for the positive side of the Shapley value
            x_idx = random.sample(feature_idxs, min(max(int(0.2 * n_features), random.choice(feature_idxs)), int(0.8 * n_features)))

            # Determine the complement set for the negative side of the Shapley value
            z_idx = [idx for idx in feature_idxs if idx not in x_idx]

            # Construct two new instances by modifying the features
            x_plus_j = np.array([x[0, i] if i in x_idx + [j] else z[i] for i in range(n_features)])
            x_minus_j = np.array([z[i] if i in z_idx + [j] else x[0, i] for i in range(n_features)])

            # Calculate the marginal contribution for the current iteration
            marginal_contribution = f.predict(x_plus_j.reshape(1, -1))[0] - \
                                    f.predict(x_minus_j.reshape(1, -1))[0]
            
            # Append the marginal contribution to the list
            marginal_contributions.append(marginal_contribution)

        # Calculate the average Shapley value over all iterations
        phi_j_x = sum(marginal_contributions) / len(marginal_contributions)

        return phi_j_x    
    
    def beeswarm_plot(self, X_train, X_shap, feature_names):
        """
        Create a horizontal bee swarm plot to visualize the Shapley values of features
        alongside their scaled and normalized training set values.

        Parameters:
        - X_train (numpy.ndarray): The training set data with features.
        - X_shap (numpy.ndarray): The Shapley values corresponding to each feature in the training set.
        - feature_names (list): A list of feature names.

        Returns:
        None
        """
        # Create a DataFrame for the data
        data = pd.DataFrame(X_train, columns=feature_names)
        for col in data.columns:
            data[col] =  stats.zscore(data[col])

        for col in data.columns:
            data[col] = (data[col] - data[col].min()) / (data[col].max() - data[col].min())


        # Create a DataFrame for the Shapley values
        shap_df = pd.DataFrame(X_shap, columns=feature_names)

        # Calculate mean absolute Shapley values for each feature
        mean_abs_shap_values = shap_df.abs().mean()

        # Sort feature names based on mean absolute Shapley values
        sorted_feature_names = mean_abs_shap_values.sort_values(ascending=False).index

        # Melt both DataFrames to long format
        data_melted = pd.melt(data, var_name='feature', value_name='feature_value')
        shap_melted = pd.melt(shap_df, var_name='feature', value_name='shap_value')

        # Combine the feature values and Shapley values
        combined_data = pd.concat([data_melted, shap_melted['shap_value']], axis=1)

        # Increase the space between y-ticks
        plt.figure(figsize=(12, 1.5 * len(feature_names)))

        # Create a horizontal bee swarm plot with no legend, using the sorted feature order
        ax = sns.swarmplot(
            x='shap_value', y='feature', hue='feature_value',
            data=combined_data, palette=sns.color_palette("coolwarm", as_cmap=True),
            order=sorted_feature_names, orient='h', size=8, legend=False
        )

        # Add a vertical line at x=0
        ax.axvline(x=0, color='black', linestyle='--', linewidth=1)

        # Add a vertical color bar to the right
        sm = plt.cm.ScalarMappable(cmap=sns.color_palette("coolwarm", as_cmap=True))
        sm.set_clim(vmin=0, vmax=1)
        cbar = plt.colorbar(sm, ax=ax, orientation='vertical', pad=0.05)
        cbar.set_label('Feature Value')

        # Customize color bar ticks and labels
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(['Low', 'High'])

        plt.xlabel('SHAP value (impact on model output)')
        ax.set_ylabel('') 
        plt.show()
    
    def force_plot(self, features, shap_values, base_value, feature_names):
        upper_bounds = None
        lower_bounds = None
        num_features = len(shap_values)
        row_height = 0.75
        rng = range(num_features - 1, -1, -1)
        order = np.argsort(-np.abs(shap_values))
        pos_lefts = []
        pos_inds = []
        pos_widths = []
        pos_low = []
        pos_high = []
        neg_lefts = []
        neg_inds = []
        neg_widths = []
        neg_low = []
        neg_high = []
        loc = base_value + shap_values.sum()
        yticklabels = ["" for i in range(num_features + 1)]

        plt.gcf().set_size_inches(10, num_features * row_height + 2.25)

        if num_features == len(shap_values):
            num_individual = num_features
        else:
            num_individual = num_features - 1

        for i in range(num_individual):
            sval = shap_values[order[i]]
            loc -= sval
            if sval >= 0:
                pos_inds.append(rng[i])
                pos_widths.append(sval)
                if lower_bounds is not None:
                    pos_low.append(lower_bounds[order[i]])
                    pos_high.append(upper_bounds[order[i]])
                pos_lefts.append(loc)
            else:
                neg_inds.append(rng[i])
                neg_widths.append(sval)
                if lower_bounds is not None:
                    neg_low.append(lower_bounds[order[i]])
                    neg_high.append(upper_bounds[order[i]])
                neg_lefts.append(loc)
            if num_individual != num_features or i + 4 < num_individual:
                plt.plot([loc, loc], [rng[i] - 1 - 0.4, rng[i] + 0.4],
                        color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
            if features is None:
                yticklabels[rng[i]] = feature_names[order[i]]
            else:
                yticklabels[rng[i]] = feature_names[order[i]] + " = " + str(np.round(features[order[i]],3))
            
        if num_features < len(shap_values):
            yticklabels[0] = "%d other features" % (len(shap_values) - num_features + 1)
            remaining_impact = base_value - loc
            if remaining_impact < 0:
                pos_inds.append(0)
                pos_widths.append(-remaining_impact)
                pos_lefts.append(loc + remaining_impact)
            else:
                neg_inds.append(0)
                neg_widths.append(-remaining_impact)
                neg_lefts.append(loc + remaining_impact)
        
        points = pos_lefts + list(np.array(pos_lefts) + np.array(pos_widths)) + neg_lefts + \
            list(np.array(neg_lefts) + np.array(neg_widths))
        dataw = np.max(points) - np.min(points)

        label_padding = np.array([0.1*dataw if w < 1 else 0 for w in pos_widths])
        plt.barh(pos_inds, np.array(pos_widths) + label_padding + 0.02*dataw,
                left=np.array(pos_lefts) - 0.01*dataw, color='#5876e2', alpha=0)
        label_padding = np.array([-0.1*dataw if -w < 1 else 0 for w in neg_widths])
        plt.barh(neg_inds, np.array(neg_widths) + label_padding - 0.02*dataw,
                left=np.array(neg_lefts) + 0.01*dataw, color='#d1493e', alpha=0)

        head_length = 0.08
        bar_width = 0.8
        xlen = plt.xlim()[1] - plt.xlim()[0]
        fig = plt.gcf()
        ax = plt.gca()
        bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
        width = bbox.width
        bbox_to_xscale = xlen/width
        hl_scaled = bbox_to_xscale * head_length
        renderer = fig.canvas.get_renderer()

        for i in range(len(pos_inds)):
            dist = pos_widths[i]
            arrow_obj = plt.arrow(
                pos_lefts[i], pos_inds[i], max(dist-hl_scaled, 0.000001), 0,
                head_length=min(dist, hl_scaled),
                color='#5876e2', width=bar_width,
                head_width=bar_width
            )

            if pos_low is not None and i < len(pos_low):
                plt.errorbar(
                    pos_lefts[i] + pos_widths[i], pos_inds[i],
                    xerr=np.array([[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]),
                    ecolor='#5876e2'
                )

            txt_obj = plt.text(
                pos_lefts[i] + 0.5*dist, pos_inds[i], "+"+str(np.round(pos_widths[i],4)),
                horizontalalignment='center', verticalalignment='center', color="white",
                fontsize=14  # Increased fontsize
            )
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()

                txt_obj = plt.text(
                    pos_lefts[i] + (5/72)*bbox_to_xscale + dist, pos_inds[i], "+"+str(np.round(pos_widths[i],4)),
                    horizontalalignment='left', verticalalignment='center', color='#5876e2',
                    fontsize=14  # Increased fontsize
                )

        for i in range(len(neg_inds)):
            dist = neg_widths[i]

            arrow_obj = plt.arrow(
                neg_lefts[i], neg_inds[i], -max(-dist-hl_scaled, 0.000001), 0,
                head_length=min(-dist, hl_scaled),
                color='#d1493e', width=bar_width,
                head_width=bar_width
            )

            if neg_low is not None and i < len(neg_low):
                plt.errorbar(
                    neg_lefts[i] + neg_widths[i], neg_inds[i],
                    xerr=np.array([[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]),
                    ecolor='#d1493e'
                )

            txt_obj = plt.text(
                neg_lefts[i] + 0.5*dist, neg_inds[i], str(np.round(neg_widths[i],4)),
                horizontalalignment='center', verticalalignment='center', color="white",
                fontsize=14  # Increased fontsize
            )
            text_bbox = txt_obj.get_window_extent(renderer=renderer)
            arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

            if text_bbox.width > arrow_bbox.width:
                txt_obj.remove()

                txt_obj = plt.text(
                    neg_lefts[i] - (5/72)*bbox_to_xscale + dist, neg_inds[i], str(np.round(neg_widths[i],4)),
                    horizontalalignment='right', verticalalignment='center', color='#d1493e',
                    fontsize=14  # Increased fontsize
                )
            
        plt.yticks(list(range(num_features)), yticklabels[:-1], fontsize=14)  # Increased fontsize

        for i in range(num_features):
            plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        plt.axvline(base_value, 0, 1/num_features, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)
        fx = base_value + shap_values.sum()
        plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

        plt.gca().xaxis.set_ticks_position('bottom')
        plt.gca().yaxis.set_ticks_position('none')
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        ax.tick_params(labelsize=14)  # Increased fontsize

        xmin, xmax = ax.get_xlim()
        ax2 = ax.twiny()
        ax2.set_xlim(xmin, xmax)
        ax2.set_xticks([base_value+1e-8])
        ax2.set_xticklabels([f"\nBase Value $=$ {str(np.round(base_value,4))}"], fontsize=14, ha="center")  # Increased fontsize

        ax3 = ax2.twiny()
        ax3.set_xlim(xmin, xmax)

        ax3.set_xticks([
            base_value + shap_values.sum() + 1e-8,
        ])
        ax3.set_xticklabels([f"\nUncertainty $=$ {str(np.round(fx,4))}"], fontsize=14, ha="center")  # Increased fontsize
        tick_labels = ax3.xaxis.get_majorticklabels()
        tick_labels[0].set_transform(tick_labels[0].get_transform(
        ) + matplotlib.transforms.ScaledTranslation(-10/72., 0, fig.dpi_scale_trans))

        tick_labels = ax2.xaxis.get_majorticklabels()
        tick_labels[0].set_transform(tick_labels[0].get_transform(
        ) + matplotlib.transforms.ScaledTranslation(-20/72., 0, fig.dpi_scale_trans))