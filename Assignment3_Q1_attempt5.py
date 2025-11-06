"""
EECE5644 Fall 2024 - Assignment 3 - Question 1
MLP Classification - FIXED VERSION

Author: Shreyas Kapanaiah Mahesh
Date: November 2025

Fixes:
- Reduced max_iter to prevent overfitting
- Added momentum to optimizer
- Better early stopping parameters
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, train_test_split
import time
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)

# ============================================================================
# DATA GENERATION
# ============================================================================

class GaussianDataGenerator:
    """Generate 4-class Gaussian data in 3D"""
    
    def __init__(self):
        self.num_classes = 4
        self.priors = np.array([0.25, 0.25, 0.25, 0.25])
        
        self.means = [
            np.array([0.0, 0.0, 0.0]),
            np.array([3.0, 0.5, 0.0]),
            np.array([0.5, 3.0, 0.5]),
            np.array([2.5, 2.5, 2.5])
        ]
        
        self.covariances = [
            np.array([[1.5, 0.3, 0.2], [0.3, 1.5, 0.3], [0.2, 0.3, 1.5]]),
            np.array([[1.2, -0.2, 0.1], [-0.2, 1.8, 0.2], [0.1, 0.2, 1.2]]),
            np.array([[1.8, 0.4, -0.1], [0.4, 1.3, 0.2], [-0.1, 0.2, 1.6]]),
            np.array([[1.4, 0.2, 0.3], [0.2, 1.4, 0.1], [0.3, 0.1, 1.4]])
        ]
        
        self.distributions = [
            multivariate_normal(mean=m, cov=c) 
            for m, c in zip(self.means, self.covariances)
        ]
    
    def generate_data(self, n_samples):
        samples_per_class = np.random.multinomial(n_samples, self.priors)
        X, y = [], []
        
        for class_idx in range(self.num_classes):
            n = samples_per_class[class_idx]
            if n > 0:
                samples = self.distributions[class_idx].rvs(size=n)
                if n == 1:
                    samples = samples.reshape(1, -1)
                X.append(samples)
                y.append(np.full(n, class_idx))
        
        X = np.vstack(X)
        y = np.hstack(y)
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]
    
    def compute_class_posteriors(self, X):
        n_samples = X.shape[0]
        posteriors = np.zeros((n_samples, self.num_classes))
        
        for i in range(self.num_classes):
            posteriors[:, i] = self.distributions[i].pdf(X) * self.priors[i]
        
        posteriors = posteriors / posteriors.sum(axis=1, keepdims=True)
        return posteriors
    
    def map_classify(self, X):
        posteriors = self.compute_class_posteriors(X)
        return np.argmax(posteriors, axis=1)


# ============================================================================
# MODEL SELECTION
# ============================================================================

def select_best_perceptrons(X_train, y_train, perceptron_options):
    """Smart model selection with FIXED parameters"""
    n_samples = len(X_train)
    
    if n_samples <= 1000:
        print(f"  Using 10-fold CV (N={n_samples})")
        cv_results = {}
        
        for p in perceptron_options:
            clf = MLPClassifier(
                hidden_layer_sizes=(p,),
                activation='relu',
                solver='adam',
                max_iter=200,  # FIXED: Reduced from 500
                early_stopping=True,
                validation_fraction=0.1,  # FIXED: Smaller validation
                n_iter_no_change=10,  # FIXED: Stricter early stopping
                random_state=42,
                learning_rate_init=0.01,  # FIXED: Higher learning rate
                alpha=0.0001  # L2 regularization
            )
            
            scores = cross_val_score(clf, X_train, y_train, cv=10, 
                                    scoring='accuracy', n_jobs=-1)
            avg_error = 1 - scores.mean()
            cv_results[p] = avg_error
            print(f"    P={p}: error={avg_error:.4f}")
        
        best_p = min(cv_results, key=cv_results.get)
        
    else:
        print(f"  Using validation split (N={n_samples})")
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        cv_results = {}
        
        for p in perceptron_options:
            clf = MLPClassifier(
                hidden_layer_sizes=(p,),
                activation='relu',
                solver='adam',
                max_iter=300,
                early_stopping=True,
                validation_fraction=0.1,
                n_iter_no_change=15,
                random_state=42,
                learning_rate_init=0.01,
                alpha=0.0001
            )
            
            clf.fit(X_tr, y_tr)
            val_error = 1 - clf.score(X_val, y_val)
            cv_results[p] = val_error
            print(f"    P={p}: error={val_error:.4f}")
        
        best_p = min(cv_results, key=cv_results.get)
    
    print(f"  → Selected: P={best_p}\n")
    return best_p


def train_final_model(X_train, y_train, hidden_dim):
    """Train final model with multiple inits"""
    print(f"  Training final model (P={hidden_dim})...", end=' ')
    start = time.time()
    
    best_model = None
    best_score = 0
    
    for seed in range(5):
        clf = MLPClassifier(
            hidden_layer_sizes=(hidden_dim,),
            activation='relu',
            solver='adam',
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=seed,
            learning_rate_init=0.01,
            alpha=0.0001
        )
        
        clf.fit(X_train, y_train)
        score = clf.score(X_train, y_train)
        
        if score > best_score:
            best_score = score
            best_model = clf
    
    elapsed = time.time() - start
    print(f"Done ({elapsed:.1f}s)")
    
    return best_model


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run complete experiment"""
    
    print("="*70)
    print("QUESTION 1: MLP CLASSIFICATION (FIXED)")
    print("="*70)
    print()
    
    start_time = time.time()
    
    data_gen = GaussianDataGenerator()
    
    print("[1] Generating datasets...")
    train_sizes = [100, 500, 1000, 5000, 10000]
    test_size = 100000
    
    X_test, y_test = data_gen.generate_data(test_size)
    print(f"  Test set: {test_size} samples")
    
    train_datasets = {}
    for n in train_sizes:
        X_train, y_train = data_gen.generate_data(n)
        train_datasets[n] = (X_train, y_train)
        print(f"  Training set: {n} samples")
    
    print("\n[2] Evaluating theoretically optimal MAP classifier...")
    optimal_predictions = data_gen.map_classify(X_test)
    optimal_error = np.mean(optimal_predictions != y_test)
    print(f"  Optimal error: {optimal_error:.4f} ({optimal_error*100:.2f}%)")
    
    print("\n[3] Model selection and training...\n")
    
    perceptron_options = [5, 10, 20, 30]
    
    results = {
        'train_sizes': [],
        'best_perceptrons': [],
        'test_errors': [],
        'optimal_error': optimal_error
    }
    
    for n in train_sizes:
        print(f"Training size: N={n}")
        print("-"*50)
        
        X_train, y_train = train_datasets[n]
        
        best_p = select_best_perceptrons(X_train, y_train, perceptron_options)
        final_model = train_final_model(X_train, y_train, best_p)
        test_error = 1 - final_model.score(X_test, y_test)
        
        print(f"  Test error: {test_error:.4f} ({test_error*100:.2f}%)\n")
        
        results['train_sizes'].append(n)
        results['best_perceptrons'].append(best_p)
        results['test_errors'].append(test_error)
    
    print("[4] Generating visualizations...")
    plot_results(results)
    
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✓ Question 1 complete in {total_time/60:.1f} minutes")
    print(f"{'='*70}")
    
    return results


def plot_results(results):
    """Create visualizations"""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax1 = axes[0]
    ax1.semilogx(results['train_sizes'], results['test_errors'], 
                 'bo-', linewidth=2.5, markersize=10, label='MLP')
    ax1.axhline(y=results['optimal_error'], color='red', linestyle='--', 
                linewidth=2, label='Optimal')
    ax1.set_xlabel('Training Samples', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Test Error', fontsize=12, fontweight='bold')
    ax1.set_title('MLP Performance vs Training Size', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    for x, y in zip(results['train_sizes'], results['test_errors']):
        ax1.annotate(f'{y*100:.1f}%', (x, y), xytext=(0,10), 
                    textcoords="offset points", ha='center', fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    ax2.semilogx(results['train_sizes'], results['best_perceptrons'], 
                 'gs-', linewidth=2.5, markersize=10)
    ax2.set_xlabel('Training Samples', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Selected Perceptrons', fontsize=12, fontweight='bold')
    ax2.set_title('Model Complexity by CV', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    for x, y in zip(results['train_sizes'], results['best_perceptrons']):
        ax2.annotate(f'{y}', (x, y), xytext=(0,10), 
                    textcoords="offset points", ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('question1_results.png', dpi=300, bbox_inches='tight')
    print("  Saved: question1_results.png")
    plt.show()
    
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"{'Train N':<12} {'Best P':<10} {'Test Error':<15} {'Error %':<12}")
    print("-"*70)
    for n, p, err in zip(results['train_sizes'], results['best_perceptrons'], results['test_errors']):
        print(f"{n:<12} {p:<10} {err:<15.4f} {err*100:<12.2f}")
    print("-"*70)
    print(f"{'Optimal':<12} {'-':<10} {results['optimal_error']:<15.4f} {results['optimal_error']*100:<12.2f}")
    print("="*70)


if __name__ == "__main__":
    results = run_experiment()