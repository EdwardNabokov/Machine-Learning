class KMeans:
    def __init__(self, k_clusters, 
                       epochs=20, 
                       dist='euclidean'):
        """
        Base constructor for KMeans.
        
        Parameters
        ----------
        k_clusters : int 
            A number of clusters.
            
        epochs : int (default 20)
            Quantity of iterations.
        
        dist : str (default euclidean)
            The metric of calculating distance between points.
        
        Variables
        ---------        
        J : list
            Loss history for every centroid (cluster)
            
        """
        
        self.k = k_clusters
        self.epochs = epochs 
        self.method_dist = dict(euclidean=self.euclidean_distance, 
                                manhattan=self.manhattan_distance)[dist]
        self.J = []
    
    def manhattan_distance(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        return sum(abs(point_a - point_b))
    
    def euclidean_distance(self, point_a: np.ndarray, point_b: np.ndarray) -> float:
        return np.sqrt(sum((point_a - point_b)**2))
    
    def loss_function(self, X, labels, centroids):
        """
        Calculate loss function for each centroid.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        labels : numpy.ndarray, shape (n_samples, 1)
            Labeled data-points according to the centroids.
        
        centroids : list
            The list of clusters' centroids.
        
        """
        
        # temp list to keep all loss of centroids together
        # then add up to total list self.J
        l = []
        for num_cluster in range(self.k):
            c = np.where(labels == num_cluster)[0]
            l.append(1/X.shape[0] * sum(sum((X[c] - centroids[num_cluster])**2)))
            
        self.J.append(np.array(l))
    
    def _randomize_centroids(self, X):
        """
        Randomize positions of clusters' centroids.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        Returns
        -------
        cluster_centroids : list
            The list that consists of centroid of each cluster.
        
        """
        
        cluster_centroids = []
        for i in range(self.k):
            
            # randomize row from data-points
            r = np.random.randint(0, X.shape[0])
            x_, y_ = X[r]
                
            # add centroid coordinates to the total list of centroids    
            cluster_centroids.append(np.array([x_, y_]))
            
        return cluster_centroids
    
    def calculate_centroid(self, X, labels):
        """
        Calculate positions of clusters' centroids.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        labels : numpy.ndarray, shape (n_samples, 1)
            Labeled data-points according to the centroids.
        
        Returns
        -------
        new_centroids : list
            The updated list with positions of centroids.
        
        """
        
        new_centroids = []
        for number_of_cluster in range(self.k):
            
            # take args, where data-point belongs to cluster 
            c = np.where(labels == number_of_cluster)[0]
            if not c.any():
                logging.warning('There is no data belonging to one of the centroids.\n\
                                Please decrease number of centroids.')
            else:
                new_centroids.append(sum(X[c]) / X[c].shape[0])
                
        return new_centroids
    
    def label_data(self, X, centroids):
        """
        Label data-points, according to the centroids.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        centroids : list
            The list of clusters' centroids.
        
        Returns
        -------
        labels : numpy.ndarray
            An array of labels of data-points.
        
        """
        
        # initialize empty np.array just as a shape
        labels = np.empty([Xtrain.shape[0], 1])    
        
        # go through every data-point and assign to a specific cluster
        for i, point in enumerate(Xtrain):
            cluster_num = np.argmin([self.method_dist(x, point) for x in centroids])
            labels[i] = cluster_num
            
        return labels
    
    def fit(self, X):
        """
        Train to break down on clusters. Calculate centroids and label data-points.
        
        Parameters
        ----------
        X : numpy.ndarray, shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and
            n_features is the number of features.
        
        """
        
        # randomize centroids for the first time
        centroids = self._randomize_centroids(X)
        
        
        # label data with a particular cluster's centroid 
        labels = self.label_data(X, centroids)
        
        # add loss to the history
        self.loss_function(Xtrain, labels, centroids)        
        
        # training
        for i in range(self.epochs):
            
            # re-calculate position of clusters' centroids
            centroids = self.calculate_centroid(X, labels)
            
            # label data with a particular cluster's centroid again
            labels = self.label_data(X, centroids)
            
            # add loss to the history
            self.loss_function(Xtrain, labels, centroids)  
            
            # print out current status
            print('Epoch: {0}, J for centroids: {1}'.format(i+1, centroids))

        self.Xtrain = X
        self.labels = labels
        self.centroids = centroids
    
    def show_plot(self):
        """Plot the clustered data-point with their centroids in two-dimensional space."""
        
        for i in range(self.k):
            cluster = np.where(self.labels == i)[0]
            plt.scatter(self.Xtrain[:, 0][cluster], self.Xtrain[:, 1][cluster], label='Cluster-{0}'.format(i+1))
            plt.scatter(self.centroids[i][0], self.centroids[i][1], marker='+', c='black', s=180)
        plt.title('Summary')
        plt.xlabel('X1')
        plt.ylabel('X2')
        plt.legend()
        plt.show()
        
    def show_history(self):
        """Plot history of loss function."""

        for i in range(self.k):
            plt.plot(np.arange(0, self.J[:][i].shape[0]), self.J[:][i], label='Cluster-{0}'.format(i+1))
        
        plt.title('Loss history')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
