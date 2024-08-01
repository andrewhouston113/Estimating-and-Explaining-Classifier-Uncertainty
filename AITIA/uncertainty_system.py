import numpy as np
from sklearn.model_selection import StratifiedKFold
from AITIA.heuristics import HeursiticsCalculator
import skfuzzy as fuzz
from sklearn import metrics
from skopt import gp_minimize
from skopt.callbacks import EarlyStopper
from tqdm import tqdm
from AITIA.syboid import SyBoid
from AITIA.complexity_measures import F1, N1
from AITIA.utils import generate_points_around_x

class UncertaintyEstimator:
    """
    A class for estimating uncertainty in a machine learning model's predictions.

    Attributes:
    model (object): The machine learning model for which uncertainty is to be estimated.
    method (str): The uncertainty estimation method to be applied, such as 'Bootstrap' or 'MonteCarlo'.
    n_samples (int): The number of samples to be drawn when estimating uncertainty.

    Methods:
    - fit(X, y): Fit the model to the training data.
    - predict(X): Generate predictions using the fitted model.
    - estimate_uncertainty(X): Estimate uncertainty for input data using the chosen method.

    Examples Usage:
    >>> model = LogisticRegression(random_state=0)
    >>> uc = UncertaintyEstimator(model=model)
    >>> uc.fit(X_train,y_train)
    >>> y_pred, misclassification_risk = uc.predict(X_test)
    """

    def __init__(self, model, cv = 5, knowledge_base = None, verbose=False, kneighbors_n=5, max_depth=4, balanced=False, rdos_neighbors=5, categorical_idx=[]):
        self.model = model
        self.cv = cv
        self.knowledge_base = knowledge_base
        self.verbose = verbose
        self.heuristics_calculator = None
        self.X = None
        self.y = None
        self.categorical_idx = categorical_idx
        self.kneighbors_n=kneighbors_n, 
        self.max_depth=max_depth, 
        self.balanced=balanced, 
        self.rdos_neighbors=rdos_neighbors
        np.int = np.int64

    
    def fit(self, X, y):
        """
        Fit the clustering-based model and initialize its components.

        Parameters:
        X (array-like): Input data for training.
        y (array-like): Target labels for training.

        Returns:
        None
        """
        # Store the input data and labels for later use.
        self.X = X
        self.y = y
        
        # Tune the clustering system to determine the number of clusters and their weights.
        n_clusters, weights = self._tune_clustering_system()
        self.weights = weights
        
        # Create a Stratified K-Fold cross-validator for model evaluation.
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)
        
        # Initialize empty arrays for heuristics and misclassifications.
        heuristics = np.array([]).reshape(0, 7)
        misclassifications = []
        
        # Iterate over cross-validation folds.
        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index, :], X[test_index, :]
            y_train, y_test = y[train_index], y[test_index]
            
            # Fit the model on the training data and make predictions.
            clf = self.model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            misclassifications += list(y_test != y_pred)
            
            # Calculate heuristics for this fold.
            hc = HeursiticsCalculator(kneighbors_n=self.kneighbors_n, max_depth=self.max_depth, balanced=self.balanced, rdos_neighbors=self.rdos_neighbors, categorical_idx=self.categorical_idx)
            hc.fit(X_train, y_train)
            heuristics = np.vstack([heuristics, hc.calculate(X_test)])        

        # If a knowledge base is available, append its heuristics and misclassifications.
        if self.knowledge_base is not None:
            heuristics = np.append(heuristics, self.knowledge_base['heuristics'], axis=0)
            misclassifications = np.append(misclassifications, self.knowledge_base['misclassifications'], axis=0)
        
        self.heuristics = heuristics
        
        ## Weight heuristics by the determined weights.
        heuristics = heuristics * weights

        ## Train the fuzzy clustering system using the weighted heuristics.
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                        np.transpose(heuristics), n_clusters, 2, error=0.005, maxiter=100, init=None)
        self.cntr = cntr
                    
        # Assign data points to clusters based on their membership.
        cluster_membership = np.argmax(u, axis=0)
        self.Cluster_IH_mean = [np.nanmean(heuristics[np.where(cluster_membership == group)[0]]) for group in range(n_clusters)]

        # Fit the model on the full dataset and create a heuristic calculator for future use.
        self.model.fit(X, y)
        self.heuristics_calculator = HeursiticsCalculator(kneighbors_n=self.kneighbors_n, max_depth=self.max_depth, balanced=self.balanced, rdos_neighbors=self.rdos_neighbors, categorical_idx=self.categorical_idx)
        self.heuristics_calculator.fit(X, y)

        # Clear the stored input data and labels.
        self.X = None
        self.y = None

    def predict(self, X, return_predictions=False):
        """
        Generate predictions and assess misclassification risk for input data.

        Parameters:
        X (array-like): Input data for prediction.

        Returns:
        y_pred (array-like): Predicted labels generated by the fitted model.
        y_prob (array-like): Predicted probabilities generated by the fitted model.
        misclassifications_risk (array-like): Risk assessment of misclassifications for the input data.
        """

        # Calculate heuristics for the input data using the heuristic calculator.
        heuristics = self.heuristics_calculator.calculate(X)

        ## Weight heuristics by the determined weights.
        heuristics = heuristics * self.weights

        # Predict cluster memberships and other information using the fuzzy clustering system.
        # This step is necessary for risk assessment.
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                np.transpose(heuristics), self.cntr, 2, error=0.005, maxiter=100, init=None)

        # Calculate misclassifications risk based on cluster memberships and cluster-specific heuristics.
        misclassifications_risk = self._weighted_average(u, self.Cluster_IH_mean)

        # Make predictions using the fitted model on the input data X.
        if return_predictions:
            y_pred = self.model.predict(X)
            y_prob = self.model.predict_proba(X)
            return y_pred, y_prob, misclassifications_risk
        else:
            return misclassifications_risk
    
    def generate_knowledge_base(self, n_datasets, X, y, max_distance=0.4, pop_size=40, n_gen=10, n_splits=5):
        """
        Generate a knowledge base by creating synthetic datasets and evaluating a model on them.

        Parameters:
        - model (object): The machine learning model to evaluate on synthetic datasets.
        - n_datasets (int): The number of synthetic datasets to generate.
        - X (array-like): The feature matrix of the original dataset.
        - y (array-like): The target labels of the original dataset.
        - max_distance (float, optional): Maximum distance for generating synthetic datasets around the original dataset.
        Defaults to 0.4.
        - pop_size (int, optional): Population size for the genetic algorithm used to generate synthetic datasets.
        Defaults to 40.
        - n_gen (int, optional): Number of generations for the genetic algorithm.
        Defaults to 10.
        - n_splits (int, optional): Number of folds for stratified k-fold cross-validation.
        Defaults to 5.

        Returns:
        - knowledge_base (dict): A dictionary containing the generated knowledge base, including heuristics and misclassifications.
        """

        # Calculate F1 and N1 scores for the original dataset.
        f1_score = F1(X, y)
        n1_score = N1(X, y)

        # Generate synthetic datasets around the original dataset.
        generated_points = generate_points_around_x(f1_score, n1_score, n_datasets, max_distance)

        # Initialize empty arrays for heuristics and misclassifications.
        heuristics = np.array([]).reshape(0, 7)
        misclassifications = []

        # Iterate over generated synthetic datasets.
        for i, point in tqdm(enumerate(generated_points), desc="Generating synthetic datasets", total=len(generated_points)):

            # Create a synthetic dataset using SyBoid.
            syboid = SyBoid(F1_Score=point[0], 
                            N1_Score=point[1], 
                            X=X, 
                            y=y, 
                            Mimic_Classes=True, 
                            Mimic_DataTypes=False,
                            Mimic_Dataset=False)

            # Generate synthetic data using a genetic algorithm.
            syboid.Generate_Data(pop_size=pop_size, n_gen=n_gen)

            # Retrieve the best synthetic dataset from the generated populations.
            X_, y_ = syboid.return_best_dataset()

            # Create a Stratified K-Fold cross-validator for model evaluation.
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
            
            # Iterate over cross-validation folds.
            for train_index, test_index in skf.split(X, y):
                X_train, X_test = X[train_index, :], X[test_index, :]
                y_train, y_test = y[train_index], y[test_index]
                
                # Fit the model on the training data and make predictions.
                clf = self.model.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                misclassifications += list(y_test != y_pred)
                
                # Calculate heuristics for this fold.
                hc = HeursiticsCalculator(kneighbors_n=self.kneighbors_n, max_depth=self.max_depth, balanced=self.balanced, rdos_neighbors=self.rdos_neighbors, categorical_idx=self.categorical_idx)
                hc.fit(X_train, y_train)
                heuristics = np.vstack([heuristics, hc.calculate(X_test)])
        
        # Create a knowledge base dictionary.
        knowledge_base = {'heuristics': heuristics,
                        'misclassifications': np.array(misclassifications)}
        
        self.knowledge_base = knowledge_base


    def _tune_clustering_system(self):
        """
        Perform hyperparameter tuning to determine the optimal cluster count and heuristic weights.

        Returns:
        n_clusters (int): The determined number of clusters for the clustering system.
        weights (list): A list of heuristic weights for risk assessment.
        """

        # Initialize an early stopping mechanism for optimization.
        early_stopping = RepeatedMinStopper(n_best=10)

        # Perform Bayesian optimization to find the optimal cluster count and weights for heuristics.
        tuning_res = gp_minimize(self._heuristic_weightings,  # Function to minimize.
                                [(2, 5), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0), (0.0, 1.0)],  # Bounds on each dimension of x.
                                acq_func="gp_hedge",  # Acquisition function for optimization.
                                n_calls=100,  # Number of evaluations of the objective function.
                                n_random_starts=5,  # Number of random initialization points.
                                random_state=42,  # Random seed for reproducibility.
                                callback=early_stopping,  # Early stopping callback.
                                x0=[3, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],  # Initial guess for optimization.
                                verbose=self.verbose)  # Verbosity level for optimization process.

        # Extract the optimal cluster count and heuristic weights from the optimization result.
        n_clusters = tuning_res.x[0]
        weights = tuning_res.x[1:]

        # Return the determined cluster count and weights.
        return n_clusters, weights

        
    def _heuristic_weightings(self, x):
        """
        Calculate heuristic weightings and assess their impact on model performance.

        Parameters:
        x (list): A list containing cluster count and heuristic weights to be evaluated.

        Returns:
        float: The negative mean of optimization scores, indicating the quality of the heuristic weightings.
        """

        # Extract the cluster count and heuristic weights from the input parameters 'x'.
        n_clusters = x[0]
        weights = x[1:]

        # Create a Stratified K-Fold cross-validator for model evaluation.
        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=42)

        # Initialize empty arrays for heuristics and misclassifications.
        heuristics = np.array([]).reshape(0, 7)
        misclassifications = []

        # Iterate over cross-validation folds for training and testing.
        for train_index, test_index in skf.split(self.X, self.y):
            X_train, X_test = self.X[train_index, :], self.X[test_index, :]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # Fit the model on the training data and make predictions.
            clf = self.model.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            misclassifications += list(y_test != y_pred)

            # Calculate heuristics for this fold using HeuristicsCalculator.
            hc = HeursiticsCalculator(kneighbors_n=self.kneighbors_n, max_depth=self.max_depth, balanced=self.balanced, rdos_neighbors=self.rdos_neighbors, categorical_idx=self.categorical_idx)
            hc.fit(X_train, y_train)
            heuristics = np.vstack([heuristics, hc.calculate(X_test)])

        # Initialize an array to store optimization scores for each fold.
        score = []

        # Iterate over cross-validation folds again to compute optimization scores.
        for train_index, test_index in skf.split(self.X, self.y):
            train_heuristics, test_heuristics = heuristics[train_index, :], heuristics[test_index, :]
            train_misclassifications, test_misclassifcations = np.array(misclassifications)[train_index], np.array(misclassifications)[test_index]

            # If a knowledge base is available, append its heuristics and misclassifications to the training data.
            if self.knowledge_base != None:
                train_heuristics = np.append(train_heuristics, self.knowledge_base['heuristics'], axis=0)
                train_misclassifications = np.append(train_misclassifications, self.knowledge_base['misclassifications'], axis=0)

            # Weight the heuristics by the given weights.
            train_heuristics = train_heuristics * weights
            test_heuristics = test_heuristics * weights

            # Train a fuzzy clustering system using the weighted training heuristics.
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(
                np.transpose(train_heuristics), n_clusters, 2, error=0.005, maxiter=100, init=None)

            # Calculate cluster-specific means for training data.
            cluster_membership = np.argmax(u, axis=0)
            Cluster_IH_mean = [np.nanmean(train_heuristics[np.where(cluster_membership == group)[0]]) for group in range(n_clusters)]

            # Estimate complexity of the test set using the trained clustering system.
            u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                np.transpose(test_heuristics), cntr, 2, error=0.005, maxiter=100, init=None)

            # Calculate misclassification risk for the test set and compute the optimization score.
            misclassifications_risk = self._weighted_average(u, Cluster_IH_mean)
            precision, recall, _ = metrics.precision_recall_curve(test_misclassifcations, misclassifications_risk)
            score.append(metrics.auc(recall, precision))

        # Return the negative mean of optimization scores, as we aim to minimize this value.
        return -np.mean(score)


    def _weighted_average(self, u, Cluster_IH_mean):
        """
        Calculate the weighted average of cluster-specific means based on cluster memberships.

        Parameters:
        u (numpy.ndarray): Cluster membership values.
        Cluster_IH_mean (list): Cluster-specific means of heuristics.

        Returns:
        numpy.ndarray: An array of defuzzified values representing weighted averages of cluster-specific means.
        """
        
        # Initialize an empty list to store defuzzified values.
        defuzzified = []

        # Iterate over cluster memberships to compute weighted averages.
        for i in range(u.shape[1]):
            # Calculate the weighted average by summing the products of cluster membership and cluster-specific means,
            # then dividing by the sum of cluster memberships. Handle NaN values gracefully.
            defuzzified.append(np.nansum((u[:, i] * Cluster_IH_mean)) / np.nansum(u[:, i]))

        # Convert the list of defuzzified values to a NumPy array and return it.
        return np.array(defuzzified)

class RepeatedMinStopper(EarlyStopper):
    """
    Stop the optimization when there is no improvement in the minimum
    achieved function evaluation after `n_best` iterations.

    Attributes:
    n_best (int): Number of iterations with no improvement to trigger stopping.
    count (int): Counter for consecutive non-improvements.
    minimum (float): The minimum value initialized with positive infinity.
    """

    def __init__(self, n_best=50):
        super().__init__()  # Call the constructor of the parent class (EarlyStopper)
        self.n_best = n_best  # Number of iterations with no improvement to trigger stopping
        self.count = 0  # Initialize the counter for consecutive non-improvements
        self.minimum = float('inf')  # Initialize with positive infinity as the minimum value

    def _criterion(self, result):
        """
        Check the criterion for stopping the optimization.

        Parameters:
        result (OptimizeResult): Result object containing optimization information.

        Returns:
        bool: True if stopping criteria are met, indicating that optimization should stop.
        """
        if result.fun < self.minimum:
            self.minimum = result.fun
            # Reset the counter when there's an improvement
            self.count = 0
        else:
            # Increment the counter for consecutive non-improvements
            self.count += 1

        # Return True if the counter exceeds or equals the defined `n_best` value, indicating stopping.
        return self.count >= self.n_best

class MetaUncertaintyEstimator:
    """
    MetaUncertaintyEstimator class for estimating uncertainty using heuristics and fuzzy clustering.

    Parameters:
    - f: A fitted uncertainty explanation object with weights and cluster centers.

    Methods:
    - predict(x): Predicts the misclassifications risk based on the provided instance and the meta-model's properties.
    """
    def __init__(self, f):
        self.f = f

    def predict(self, x):
        """
        Generate predictions and assess misclassification risk for input data.

        Parameters:
        X (array-like): Input data for prediction.

        Returns:
        misclassifications_risk (array-like): Risk assessment of misclassifications for the input data.
        """
        ## Weight heuristics by the determined weights.
        x = x * self.f.weights

        # Predict cluster memberships and other information using the fuzzy clustering system.
        # This step is necessary for risk assessment.
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(
                np.transpose(x), self.f.cntr, 2, error=0.005, maxiter=100, init=None)

        # Calculate misclassifications risk based on cluster memberships and cluster-specific heuristics.
        misclassifications_risk = self.f._weighted_average(u, self.f.Cluster_IH_mean)
        return misclassifications_risk