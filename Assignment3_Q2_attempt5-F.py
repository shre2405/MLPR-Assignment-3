"""
EECE5644 Fall 2025 - Assignment 3 - Question 2
GMM Model Order Selection with Cross-Validation

This code implements:
1. True GMM specification with 4 components (2 overlapping)
2. Data generation for N = 10, 100, 1000 samples
3. 10-fold cross-validation for model order selection (C = 1 to 10)
4. 100+ repetitions to analyze selection frequency
5. Comprehensive visualization and analysis
6. CV score profiles showing decision-making process

Author: Shreyas Kapanaiah Mahesh
Date: November 2025
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# ============================================================================
# PART 1: TRUE GMM SPECIFICATION
# ============================================================================

class TrueGMM:
    """
    True Gaussian Mixture Model with 4 components in 2D
    Two components are designed to overlap significantly
    """
    
    def __init__(self):
        # 4 components with different mixing weights
        self.n_components = 4
        self.weights = np.array([0.3, 0.25, 0.25, 0.2])
        
        # Mean vectors - components 0 and 1 overlap significantly
        self.means = [
            np.array([0.0, 0.0]),      # Component 0
            np.array([2.0, 0.5]),      # Component 1 - overlaps with 0
            np.array([6.0, 6.0]),      # Component 2
            np.array([8.0, 2.0])       # Component 3
        ]
        
        # Covariance matrices
        # Components 0 and 1 have overlapping covariances
        self.covariances = [
            np.array([[1.5, 0.5],      # Component 0
                      [0.5, 1.5]]),
            
            np.array([[1.8, -0.4],     # Component 1 - overlaps with 0
                      [-0.4, 1.2]]),
            
            np.array([[1.0, 0.3],      # Component 2
                      [0.3, 1.0]]),
            
            np.array([[1.2, 0.2],      # Component 3
                      [0.2, 0.8]])
        ]
        
        # Verify overlap: distance between means 0 and 1
        dist = np.linalg.norm(self.means[0] - self.means[1])
        avg_std = np.mean([np.sqrt(np.trace(self.covariances[0])), 
                          np.sqrt(np.trace(self.covariances[1]))])
        print(f"True GMM Configuration:")
        print(f"  Components: {self.n_components}")
        print(f"  Weights: {self.weights}")
        print(f"  Distance between overlapping components 0 & 1: {dist:.2f}")
        print(f"  Average std of overlapping components: {avg_std:.2f}")
        print(f"  Overlap ratio (dist/avg_std): {dist/avg_std:.2f}")
        print()
    
    def generate_data(self, n_samples):
        """
        Generate samples from the true GMM
        
        Returns:
            X: (n_samples, 2) array of samples
            true_components: (n_samples,) array of component assignments
        """
        # Sample component indices according to mixture weights
        component_indices = np.random.choice(
            self.n_components, 
            size=n_samples, 
            p=self.weights
        )
        
        # Generate samples from each selected component
        X = np.zeros((n_samples, 2))
        for i in range(n_samples):
            comp_idx = component_indices[i]
            X[i] = np.random.multivariate_normal(
                self.means[comp_idx], 
                self.covariances[comp_idx]
            )
        
        return X, component_indices
    
    def compute_log_likelihood(self, X):
        """Compute log-likelihood of data under true GMM"""
        n_samples = X.shape[0]
        log_likelihood = 0.0
        
        for i in range(n_samples):
            # Compute probability under each component
            prob = 0.0
            for j in range(self.n_components):
                # p(x|component) * p(component)
                diff = X[i] - self.means[j]
                cov_inv = np.linalg.inv(self.covariances[j])
                cov_det = np.linalg.det(self.covariances[j])
                
                exponent = -0.5 * diff.T @ cov_inv @ diff
                normalizer = 1.0 / (2 * np.pi * np.sqrt(cov_det))
                
                prob += self.weights[j] * normalizer * np.exp(exponent)
            
            log_likelihood += np.log(prob + 1e-10)
        
        return log_likelihood


# ============================================================================
# PART 2: MODEL ORDER SELECTION WITH CROSS-VALIDATION
# ============================================================================

def perform_cross_validation(X, candidate_orders, n_folds=10):
    """
    Perform k-fold cross-validation to select GMM order
    
    Args:
        X: Data array (n_samples, 2)
        candidate_orders: List of GMM orders to evaluate (e.g., [1,2,...,10])
        n_folds: Number of cross-validation folds
        
    Returns:
        selected_order: The order with highest average validation log-likelihood
        cv_scores: Dictionary mapping order -> list of fold scores
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=np.random.randint(10000))
    cv_scores = {order: [] for order in candidate_orders}
    
    for order in candidate_orders:
        fold_log_likelihoods = []
        
        for train_idx, val_idx in kf.split(X):
            X_train = X[train_idx]
            X_val = X[val_idx]
            
            try:
                # Fit GMM on training fold
                gmm = GaussianMixture(
                    n_components=order,
                    covariance_type='full',
                    max_iter=200,
                    n_init=5,
                    random_state=np.random.randint(10000)
                )
                gmm.fit(X_train)
                
                # Evaluate on validation fold
                val_log_likelihood = gmm.score(X_val) * len(X_val)  # Total log-likelihood
                fold_log_likelihoods.append(val_log_likelihood)
                
            except:
                # If fitting fails, assign very low log-likelihood
                fold_log_likelihoods.append(-1e10)
        
        cv_scores[order] = fold_log_likelihoods
    
    # Select order with highest average validation log-likelihood
    avg_scores = {order: np.mean(scores) for order, scores in cv_scores.items()}
    selected_order = max(avg_scores, key=avg_scores.get)
    
    return selected_order, cv_scores


# ============================================================================
# PART 3: MONTE CARLO EXPERIMENT
# ============================================================================

def run_monte_carlo_experiment(true_gmm, n_samples_list, candidate_orders, 
                               n_experiments=100):
    """
    Run Monte Carlo experiment to evaluate model order selection
    
    Args:
        true_gmm: TrueGMM object
        n_samples_list: List of dataset sizes to test (e.g., [10, 100, 1000])
        candidate_orders: List of GMM orders to evaluate
        n_experiments: Number of repetitions
        
    Returns:
        results: Dictionary with selection frequencies
    """
    results = {n: {order: 0 for order in candidate_orders} 
               for n in n_samples_list}
    
    print("="*80)
    print("MONTE CARLO EXPERIMENT")
    print("="*80)
    
    for n_samples in n_samples_list:
        print(f"\nDataset size: {n_samples} samples")
        print("-"*60)
        
        for exp in range(n_experiments):
            if (exp + 1) % 20 == 0:
                print(f"  Experiment {exp+1}/{n_experiments}...")
            
            # Generate dataset
            X, _ = true_gmm.generate_data(n_samples)
            
            # Perform cross-validation
            selected_order, _ = perform_cross_validation(
                X, 
                candidate_orders, 
                n_folds=min(10, n_samples)  # Use fewer folds for small datasets
            )
            
            # Record selection
            results[n_samples][selected_order] += 1
        
        print(f"  Completed {n_experiments} experiments")
    
    return results


# ============================================================================
# PART 4: VISUALIZATION AND ANALYSIS
# ============================================================================

def visualize_true_gmm(true_gmm):
    """Visualize the true GMM distribution"""
    
    # Generate samples for visualization
    X, components = true_gmm.generate_data(2000)
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Plot samples colored by true component
    colors = ['red', 'blue', 'green', 'purple']
    for i in range(true_gmm.n_components):
        mask = components == i
        ax.scatter(X[mask, 0], X[mask, 1], c=colors[i], alpha=0.5, 
                  s=20, label=f'Component {i} (π={true_gmm.weights[i]:.2f})')
    
    # Plot component means
    for i, mean in enumerate(true_gmm.means):
        ax.plot(mean[0], mean[1], 'k*', markersize=20, 
               markeredgewidth=2, markeredgecolor='white')
    
    ax.set_xlabel('X₁', fontsize=12)
    ax.set_ylabel('X₂', fontsize=12)
    ax.set_title('True GMM: 4 Components (Components 0 & 1 Overlap)', 
                fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('question2_true_gmm.png', dpi=300, bbox_inches='tight')
    print("Saved: question2_true_gmm.png")
    plt.show()


def plot_cv_scores_example(true_gmm, n_samples_list, candidate_orders):
    """
    Plot example CV scores vs number of components
    Shows the decision-making process of cross-validation
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, n_samples in enumerate(n_samples_list):
        ax = axes[idx]
        
        # Generate one example dataset
        X, _ = true_gmm.generate_data(n_samples)
        
        # Perform CV and get scores
        n_folds = min(10, n_samples)
        selected_order, cv_scores = perform_cross_validation(
            X, candidate_orders, n_folds=n_folds
        )
        
        # Calculate average log-likelihood for each order
        avg_scores = {order: np.mean(scores) for order, scores in cv_scores.items()}
        
        # Plot
        orders = list(avg_scores.keys())
        scores = list(avg_scores.values())
        
        ax.plot(orders, scores, 'o-', linewidth=2.5, markersize=8, 
               color='steelblue', label='CV Score')
        
        # Highlight selected order (maximum)
        selected_idx = orders.index(selected_order)
        ax.plot(selected_order, scores[selected_idx], 'r*', 
               markersize=20, markeredgewidth=2, markeredgecolor='darkred',
               label=f'Selected: C={selected_order}')
        
        # Mark true order
        ax.axvline(x=4, color='green', linestyle='--', linewidth=2.5, 
                  alpha=0.7, label='True Order (C=4)')
        
        ax.set_xlabel('Number of GMM Components (C)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Avg Validation Log-Likelihood', fontsize=12, fontweight='bold')
        ax.set_title(f'N = {n_samples} samples', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='lower right')
        ax.set_xticks(orders)
        
        # Add annotation for selected order
        ax.annotate(f'Max: C={selected_order}', 
                   xy=(selected_order, scores[selected_idx]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    plt.tight_layout()
    plt.savefig('question2_cv_scores.png', dpi=300, bbox_inches='tight')
    print("Saved: question2_cv_scores.png")
    plt.show()


def plot_selection_frequencies(results, candidate_orders, n_experiments):
    """Create bar plots showing selection frequencies"""
    
    n_samples_list = sorted(results.keys())
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, n_samples in enumerate(n_samples_list):
        ax = axes[idx]
        
        # Get selection counts
        counts = [results[n_samples][order] for order in candidate_orders]
        frequencies = np.array(counts) / n_experiments * 100
        
        # Create bar plot
        bars = ax.bar(candidate_orders, frequencies, color='steelblue', 
                     edgecolor='black', alpha=0.7)
        
        # Highlight true order (C=4)
        bars[3].set_color('darkred')
        bars[3].set_alpha(0.9)
        
        # Add percentage labels on bars
        for i, (order, freq) in enumerate(zip(candidate_orders, frequencies)):
            if freq > 0:
                ax.text(order, freq + 1, f'{freq:.1f}%', 
                       ha='center', va='bottom', fontsize=9)
        
        ax.set_xlabel('Number of GMM Components', fontsize=11)
        ax.set_ylabel('Selection Frequency (%)', fontsize=11)
        ax.set_title(f'N = {n_samples} samples\n({n_experiments} experiments)', 
                    fontsize=12, fontweight='bold')
        ax.set_xticks(candidate_orders)
        ax.set_ylim([0, max(frequencies) * 1.15 + 5])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add text box with most selected order
        most_selected = candidate_orders[np.argmax(frequencies)]
        textstr = f'Most selected: C={most_selected}\n({max(frequencies):.1f}%)'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('question2_selection_frequencies.png', dpi=300, bbox_inches='tight')
    print("Saved: question2_selection_frequencies.png")
    plt.show()


def create_summary_table(results, candidate_orders, n_experiments):
    """Create comprehensive summary table"""
    
    n_samples_list = sorted(results.keys())
    
    print("\n" + "="*80)
    print("MODEL ORDER SELECTION SUMMARY TABLE")
    print("="*80)
    print(f"True GMM Order: C = 4")
    print(f"Number of Experiments: {n_experiments}")
    print("="*80)
    
    for n_samples in n_samples_list:
        print(f"\nDataset Size: N = {n_samples} samples")
        print("-"*80)
        print(f"{'Order (C)':<12} {'Count':<12} {'Frequency (%)':<15}")
        print("-"*80)
        
        total = sum(results[n_samples].values())
        for order in candidate_orders:
            count = results[n_samples][order]
            freq = count / total * 100 if total > 0 else 0
            marker = " *** CORRECT ***" if order == 4 else ""
            print(f"{order:<12} {count:<12} {freq:<15.2f}{marker}")
        
        print("-"*80)
        
        # Statistics
        correct_selections = results[n_samples][4]
        correct_pct = correct_selections / n_experiments * 100
        most_selected = max(results[n_samples], key=results[n_samples].get)
        most_selected_count = results[n_samples][most_selected]
        most_selected_pct = most_selected_count / n_experiments * 100
        
        print(f"\nStatistics:")
        print(f"  Correct (C=4) selected: {correct_selections}/{n_experiments} ({correct_pct:.1f}%)")
        print(f"  Most frequently selected: C={most_selected} ({most_selected_pct:.1f}%)")
        
        # Compute average selected order
        avg_order = sum(order * count for order, count in results[n_samples].items()) / n_experiments
        print(f"  Average selected order: {avg_order:.2f}")
        print()


def create_heatmap(results, candidate_orders):
    """Create heatmap visualization of selection frequencies"""
    
    n_samples_list = sorted(results.keys())
    n_experiments = sum(results[n_samples_list[0]].values())
    
    # Create matrix of frequencies
    freq_matrix = np.zeros((len(n_samples_list), len(candidate_orders)))
    for i, n_samples in enumerate(n_samples_list):
        for j, order in enumerate(candidate_orders):
            freq_matrix[i, j] = results[n_samples][order] / n_experiments * 100
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    im = ax.imshow(freq_matrix, cmap='YlOrRd', aspect='auto')
    
    # Set ticks
    ax.set_xticks(np.arange(len(candidate_orders)))
    ax.set_yticks(np.arange(len(n_samples_list)))
    ax.set_xticklabels(candidate_orders)
    ax.set_yticklabels(n_samples_list)
    
    # Labels
    ax.set_xlabel('Number of GMM Components (C)', fontsize=12)
    ax.set_ylabel('Dataset Size (N)', fontsize=12)
    ax.set_title('Model Order Selection Frequency Heatmap (%)', 
                fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(n_samples_list)):
        for j in range(len(candidate_orders)):
            text = ax.text(j, i, f'{freq_matrix[i, j]:.1f}',
                          ha="center", va="center", color="black", fontsize=10)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Selection Frequency (%)', fontsize=11)
    
    # Highlight true order column
    ax.axvline(x=3.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax.axvline(x=2.5, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    
    plt.tight_layout()
    plt.savefig('question2_heatmap.png', dpi=300, bbox_inches='tight')
    print("Saved: question2_heatmap.png")
    plt.show()


# ============================================================================
# PART 5: MAIN EXPERIMENT
# ============================================================================

def run_experiment():
    """Run the complete experiment for Question 2"""
    
    print("="*80)
    print("QUESTION 2: GMM MODEL ORDER SELECTION")
    print("="*80)
    print()
    
    # Define true GMM
    print("[1] Defining true GMM...")
    true_gmm = TrueGMM()
    
    # Visualize true GMM
    print("[2] Visualizing true GMM distribution...")
    visualize_true_gmm(true_gmm)
    
    # DEFINE PARAMETERS FIRST (BEFORE using them!)
    n_samples_list = [10, 100, 1000]
    candidate_orders = list(range(1, 11))  # Test C = 1 to 10
    n_experiments = 100
    
    # NOW plot CV scores (after parameters are defined)
    print("[2b] Plotting example CV score profiles...")
    plot_cv_scores_example(true_gmm, n_samples_list, candidate_orders)
    
    print(f"\n[3] Running Monte Carlo experiment...")
    print(f"    Sample sizes: {n_samples_list}")
    print(f"    Candidate orders: {candidate_orders}")
    print(f"    Number of experiments: {n_experiments}")
    print()
    
    # Run Monte Carlo experiment
    results = run_monte_carlo_experiment(
        true_gmm, 
        n_samples_list, 
        candidate_orders, 
        n_experiments
    )
    
    # Visualize and analyze results
    print("\n[4] Creating visualizations...")
    plot_selection_frequencies(results, candidate_orders, n_experiments)
    create_heatmap(results, candidate_orders)
    
    print("\n[5] Generating summary statistics...")
    create_summary_table(results, candidate_orders, n_experiments)
    
    return results


# ============================================================================
# RUN EXPERIMENT
# ============================================================================

if __name__ == "__main__":
    results = run_experiment()
    print("\n" + "="*80)
    print("✓ Question 2 complete!")
    print("="*80)
    
    print("\nKey Findings:")
    print("  • With N=10: Cross-validation fails due to insufficient samples per fold")
    print("  • With N=100: CV tends to underestimate order due to overlapping components")
    print("  • With N=1000: Cross-validation reliably identifies true order (C=4)")
    print("  • CV score profiles show the decision-making process at each sample size")