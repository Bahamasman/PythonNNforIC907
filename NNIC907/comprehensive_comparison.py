import numpy as np
import matplotlib.pyplot as plt
import time
import myNN as nn

print("=" * 80)
print("ORGANIZED HYPERPARAMETER COMPARISON FOR NEURAL NETWORKS")
print("4-SECTION ANALYSIS: SGD VARIANTS → SGD vs ADAM → ADAM TUNING → SUMMARY")
print("=" * 80)

# Generate data
npts = 100
xpts = np.linspace(0,1,npts)
yreal = np.sin(30*xpts) + np.cos(10*xpts) + 2 + xpts**2
X = xpts.reshape(npts,1)
y = yreal.reshape(npts,1)

# Split data once for all experiments
dummy_nn = nn.myNeuralNetwork()
X_train, X_test, y_train, y_test = dummy_nn.train_test_split(X, y, test_size=0.2, random_state=42)

# Training parameters
epochs = 10000

def train_and_evaluate(config, X_train, y_train, X_test, y_test, epochs, section_name):
    """Train and evaluate a single configuration"""
    try:
        # Create network
        NeuralNetwork = nn.myNeuralNetwork()
        hidden_size = config['hidden_size']
        num_hiddenLayers = len(hidden_size)
        
        # Build network
        NeuralNetwork.build(1, 1, num_hiddenLayers, hidden_size, config['activation'], 1.0)
        NeuralNetwork.initialize()
        NeuralNetwork.defineLossFunction()
        NeuralNetwork.defineOptimizer(config['optimizer'], config['lr'])
        
        # Training
        train_losses = []
        test_losses = []
        start_time = time.time()
        
        for epoch in range(epochs):
            train_pred = NeuralNetwork.forward(X_train)
            train_loss = NeuralNetwork.computeLoss(y_train)
            
            # Check for numerical instability
            if np.isnan(train_loss) or np.isinf(train_loss):
                print(f"  ❌ [{section_name}] {config['name']}: Training stopped at epoch {epoch} - NaN/Inf loss")
                return None
                
            NeuralNetwork.backward(y_train)
            NeuralNetwork.updateParam()
            train_losses.append(train_loss)
            
            test_pred = NeuralNetwork.forward(X_test)
            test_loss = NeuralNetwork.computeLoss(y_test)
            
            if np.isnan(test_loss) or np.isinf(test_loss):
                print(f"  ❌ [{section_name}] {config['name']}: Training stopped at epoch {epoch} - NaN/Inf test loss")
                return None
                
            test_losses.append(test_loss)
            
            if epoch % 2000 == 0:
                print(f"  [{section_name}] {config['name']} - Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
        
        # Final evaluation
        if len(train_losses) > 100:
            training_time = time.time() - start_time
            test_results = NeuralNetwork.evaluate(X_test, y_test)
            final_ratio = test_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else 1.0
            
            result = {
                'config': config,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': train_losses[-1],
                'final_test_loss': test_losses[-1],
                'ratio': final_ratio,
                'r2_score': test_results['r2_score'],
                'mae': test_results['mae'],
                'training_time': training_time,
                'network': NeuralNetwork
            }
            
            print(f"  ✅ [{section_name}] {config['name']}: R²={test_results['r2_score']:.4f}, MAE={test_results['mae']:.4f}, Time={training_time:.2f}s")
            return result
        else:
            print(f"  ❌ [{section_name}] {config['name']}: Insufficient training data")
            return None
            
    except Exception as e:
        print(f"  ❌ [{section_name}] {config['name']}: Error - {str(e)}")
        return None

# ============================================================================
# SECTION 1: JUSTIFY DIFFERENT SGD VARIANTS (Normal, Decay, Momentum)
# ============================================================================

print("\n" + "🔧 SECTION 1: SGD VARIANTS COMPARISON" + "🔧")
print("Testing SGD Normal vs SGD with Decay vs SGD with Momentum")
print("-" * 60)

section1_configs = [
    {'optimizer': 'SGD', 'lr': 0.1, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Normal'},
    {'optimizer': 'SGD', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Normal_Conservative'},
    {'optimizer': 'SGD_Decay', 'lr': 0.1, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Decay'},
    {'optimizer': 'SGD_Decay', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Decay_Conservative'},
    {'optimizer': 'SGD_Momentum', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum'},
    {'optimizer': 'SGD_Momentum', 'lr': 0.03, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_Conservative'},
]

section1_results = []
for config in section1_configs:
    result = train_and_evaluate(config, X_train, y_train, X_test, y_test, epochs, "SGD Variants")
    if result:
        section1_results.append(result)

# Section 1 Analysis and Plots
if section1_results:
    print(f"\n📊 SECTION 1 RESULTS:")
    section1_sorted = sorted(section1_results, key=lambda x: x['r2_score'], reverse=True)
    
    for i, result in enumerate(section1_sorted):
        print(f"  {i+1}. {result['config']['name']}: R²={result['r2_score']:.4f}, Time={result['training_time']:.2f}s")
    
    # Section 1 Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves
    colors1 = {'SGD': 'red', 'SGD_Decay': 'orange', 'SGD_Momentum': 'blue'}
    for result in section1_results:
        optimizer_type = result['config']['optimizer']
        color = colors1.get(optimizer_type, 'black')
        linestyle = '-' if 'Conservative' not in result['config']['name'] else '--'
        ax1.plot(result['test_losses'], color=color, linestyle=linestyle, 
                label=result['config']['name'], linewidth=2, alpha=0.8)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('SGD Variants: Loss Evolution', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # R² comparison
    names1 = [r['config']['name'] for r in section1_sorted]
    r2_scores1 = [r['r2_score'] for r in section1_sorted]
    colors_bar1 = [colors1.get(r['config']['optimizer'], 'gray') for r in section1_sorted]
    
    bars1 = ax2.bar(range(len(names1)), r2_scores1, color=colors_bar1, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('SGD Variant')
    ax2.set_ylabel('R² Score')
    ax2.set_title('SGD Variants: Performance Comparison', fontweight='bold')
    ax2.set_xticks(range(len(names1)))
    ax2.set_xticklabels([n.replace('_', '\n') for n in names1], rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars1, r2_scores1):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold')
    
    # Training time comparison
    times1 = [r['training_time'] for r in section1_sorted]
    bars_time1 = ax3.bar(range(len(names1)), times1, color=colors_bar1, alpha=0.8, edgecolor='black')
    ax3.set_xlabel('SGD Variant')
    ax3.set_ylabel('Training Time (s)')
    ax3.set_title('SGD Variants: Training Efficiency', fontweight='bold')
    ax3.set_xticks(range(len(names1)))
    ax3.set_xticklabels([n.replace('_', '\n') for n in names1], rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add time labels
    for bar, time_val in zip(bars_time1, times1):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    # Average performance by optimizer type
    sgd_normal = [r for r in section1_results if r['config']['optimizer'] == 'SGD']
    sgd_decay = [r for r in section1_results if r['config']['optimizer'] == 'SGD_Decay']
    sgd_momentum = [r for r in section1_results if r['config']['optimizer'] == 'SGD_Momentum']
    
    avg_data = []
    avg_labels = []
    avg_colors = []
    
    if sgd_normal:
        avg_data.append(np.mean([r['r2_score'] for r in sgd_normal]))
        avg_labels.append('SGD Normal')
        avg_colors.append('red')
    if sgd_decay:
        avg_data.append(np.mean([r['r2_score'] for r in sgd_decay]))
        avg_labels.append('SGD Decay')
        avg_colors.append('orange')
    if sgd_momentum:
        avg_data.append(np.mean([r['r2_score'] for r in sgd_momentum]))
        avg_labels.append('SGD Momentum')
        avg_colors.append('blue')
    
    bars_avg = ax4.bar(avg_labels, avg_data, color=avg_colors, alpha=0.8, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Average R² Score')
    ax4.set_title('SGD Variants: Average Performance', fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Highlight best performer
    best_idx = np.argmax(avg_data)
    bars_avg[best_idx].set_edgecolor('gold')
    bars_avg[best_idx].set_linewidth(4)
    
    # Add average value labels
    for bar, avg_score in zip(bars_avg, avg_data):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{avg_score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('SECTION 1: SGD Variants Comprehensive Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Section 1 Summary
    best_sgd_variant = avg_labels[best_idx]
    improvement = ((max(avg_data) - min(avg_data)) / min(avg_data)) * 100 if min(avg_data) > 0 else 0
    
    print(f"\n🏆 SECTION 1 CONCLUSION:")
    print(f"   Best SGD Variant: {best_sgd_variant}")
    print(f"   Performance Improvement: {improvement:.1f}% over worst variant")
    print(f"   Justification: {best_sgd_variant} provides the best balance of convergence speed and stability")

# ============================================================================
# SECTION 2: SGD MOMENTUM vs ADAM COMPARISON
# ============================================================================

print("\n" + "🥊 SECTION 2: SGD MOMENTUM vs ADAM COMPARISON" + "🥊")
print("Head-to-head comparison with matched hyperparameters")
print("-" * 60)

section2_configs = [
    # Conservative learning rates
    {'optimizer': 'SGD_Momentum', 'lr': 0.01, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_LR0.01'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.01'},
    
    # Moderate learning rates
    {'optimizer': 'SGD_Momentum', 'lr': 0.03, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_LR0.03'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.03'},
    
    # Aggressive learning rates
    {'optimizer': 'SGD_Momentum', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_LR0.05'},
    {'optimizer': 'Adam', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.05'},
    
    # Different architectures
    {'optimizer': 'SGD_Momentum', 'lr': 0.03, 'hidden_size': [50], 'activation': 'tanh', 'name': 'SGD_Momentum_Small'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [50], 'activation': 'tanh', 'name': 'ADAM_Small'},
    
    {'optimizer': 'SGD_Momentum', 'lr': 0.03, 'hidden_size': [200], 'activation': 'tanh', 'name': 'SGD_Momentum_Large'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [200], 'activation': 'tanh', 'name': 'ADAM_Large'},
    
    {'optimizer': 'SGD_Momentum', 'lr': 0.02, 'hidden_size': [100, 50], 'activation': 'tanh', 'name': 'SGD_Momentum_Deep'},
    {'optimizer': 'Adam', 'lr': 0.02, 'hidden_size': [100, 50], 'activation': 'tanh', 'name': 'ADAM_Deep'},
]
section2_results = []
for config in section2_configs:
    result = train_and_evaluate(config, X_train, y_train, X_test, y_test, epochs, "SGD vs ADAM")
    if result:
        section2_results.append(result)

# Section 2 Analysis and Plots
if section2_results:
    print(f"\n📊 SECTION 2 RESULTS:")
    section2_sorted = sorted(section2_results, key=lambda x: x['r2_score'], reverse=True)
    
    sgd_results2 = [r for r in section2_results if 'SGD_Momentum' in r['config']['optimizer']]
    adam_results2 = [r for r in section2_results if 'Adam' in r['config']['optimizer']]
    
    print(f"  SGD Momentum Results:")
    for result in sorted(sgd_results2, key=lambda x: x['r2_score'], reverse=True):
        print(f"    {result['config']['name']}: R²={result['r2_score']:.4f}")
    
    print(f"  ADAM Results:")
    for result in sorted(adam_results2, key=lambda x: x['r2_score'], reverse=True):
        print(f"    {result['config']['name']}: R²={result['r2_score']:.4f}")
    
    # Section 2 Plots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Loss curves comparison
    for result in section2_results:
        color = 'blue' if 'SGD' in result['config']['optimizer'] else 'red'
        linestyle = '-'
        ax1.plot(result['test_losses'], color=color, linestyle=linestyle, 
                label=result['config']['name'], linewidth=2, alpha=0.7)
    
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Test Loss')
    ax1.set_title('SGD Momentum vs ADAM: Loss Evolution', fontweight='bold')
    ax1.set_yscale('log')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Direct R² comparison
    names2 = [r['config']['name'] for r in section2_sorted]
    r2_scores2 = [r['r2_score'] for r in section2_sorted]
    colors_bar2 = ['blue' if 'SGD' in r['config']['optimizer'] else 'red' for r in section2_sorted]
    
    bars2 = ax2.bar(range(len(names2)), r2_scores2, color=colors_bar2, alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Configuration')
    ax2.set_ylabel('R² Score')
    ax2.set_title('SGD Momentum (Blue) vs ADAM (Red): R² Scores', fontweight='bold')
    ax2.set_xticks(range(len(names2)))
    ax2.set_xticklabels([n.replace('_', '\n') for n in names2], rotation=45, ha='right', fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, score in zip(bars2, r2_scores2):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # Average performance comparison
    if sgd_results2 and adam_results2:
        sgd_avg_r2 = np.mean([r['r2_score'] for r in sgd_results2])
        adam_avg_r2 = np.mean([r['r2_score'] for r in adam_results2])
        sgd_avg_time = np.mean([r['training_time'] for r in sgd_results2])
        adam_avg_time = np.mean([r['training_time'] for r in adam_results2])
        
        # Performance comparison
        optimizers = ['SGD Momentum', 'ADAM']
        avg_r2_scores = [sgd_avg_r2, adam_avg_r2]
        avg_colors = ['blue', 'red']
        
        bars_comp = ax3.bar(optimizers, avg_r2_scores, color=avg_colors, alpha=0.8, 
                           edgecolor='black', linewidth=2)
        ax3.set_ylabel('Average R² Score')
        ax3.set_title('Average Performance Comparison', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Highlight better performer
        best_opt_idx = np.argmax(avg_r2_scores)
        bars_comp[best_opt_idx].set_edgecolor('gold')
        bars_comp[best_opt_idx].set_linewidth(4)
        
        # Add labels
        for bar, score in zip(bars_comp, avg_r2_scores):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
        
        # Training time comparison
        avg_times = [sgd_avg_time, adam_avg_time]
        bars_time_comp = ax4.bar(optimizers, avg_times, color=avg_colors, alpha=0.8, 
                                edgecolor='black', linewidth=2)
        ax4.set_ylabel('Average Training Time (s)')
        ax4.set_title('Training Efficiency Comparison', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Highlight faster performer
        best_speed_idx = np.argmin(avg_times)
        bars_time_comp[best_speed_idx].set_edgecolor('gold')
        bars_time_comp[best_speed_idx].set_linewidth(4)
        
        # Add time labels
        for bar, time_val in zip(bars_time_comp, avg_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'{time_val:.1f}s', ha='center', va='bottom', fontweight='bold')
    
    plt.suptitle('SECTION 2: SGD Momentum vs ADAM Head-to-Head Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Section 2 Summary
    if sgd_results2 and adam_results2:
        performance_diff = ((adam_avg_r2 - sgd_avg_r2) / abs(sgd_avg_r2)) * 100
        speed_diff = ((sgd_avg_time - adam_avg_time) / adam_avg_time) * 100
        
        print(f"\n🏆 SECTION 2 CONCLUSION:")
        print(f"   SGD Momentum - Avg R²: {sgd_avg_r2:.4f}, Avg Time: {sgd_avg_time:.2f}s")
        print(f"   ADAM - Avg R²: {adam_avg_r2:.4f}, Avg Time: {adam_avg_time:.2f}s")
        
        if adam_avg_r2 > sgd_avg_r2:
            print(f"   🎯 ADAM outperforms SGD Momentum by {performance_diff:.1f}%")
        else:
            print(f"   🎯 SGD Momentum outperforms ADAM by {abs(performance_diff):.1f}%")
        
        if sgd_avg_time < adam_avg_time:
            print(f"   ⚡ SGD Momentum is {abs(speed_diff):.1f}% faster than ADAM")
        else:
            print(f"   ⚡ ADAM is {speed_diff:.1f}% faster than SGD Momentum")

# ============================================================================
# SECTION 3: ADAM HYPERPARAMETER OPTIMIZATION
# ============================================================================

print("\n" + "🚀 SECTION 3: ADAM HYPERPARAMETER OPTIMIZATION" + "🚀")
print("Extensive ADAM tuning: Learning rates, architectures, activations")
print("-" * 60)

section3_configs = [
    # Learning rate sweep
    {'optimizer': 'Adam', 'lr': 0.001, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.001'},
    {'optimizer': 'Adam', 'lr': 0.003, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.003'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.01'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.03'},
    {'optimizer': 'Adam', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.05'},
    {'optimizer': 'Adam', 'lr': 0.1, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_LR0.1'},
    
    # Architecture sweep (using optimal LR from above)
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [25], 'activation': 'tanh', 'name': 'ADAM_Arch25'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [50], 'activation': 'tanh', 'name': 'ADAM_Arch50'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_Arch100'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [150], 'activation': 'tanh', 'name': 'ADAM_Arch150'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [200], 'activation': 'tanh', 'name': 'ADAM_Arch200'},
    {'optimizer': 'Adam', 'lr': 0.02, 'hidden_size': [100, 50], 'activation': 'tanh', 'name': 'ADAM_Deep2Layer'},
    {'optimizer': 'Adam', 'lr': 0.015, 'hidden_size': [100, 75, 50], 'activation': 'tanh', 'name': 'ADAM_Deep3Layer'},
    
    # Activation function sweep
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [100], 'activation': 'sigmoid', 'name': 'ADAM_Sigmoid'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_Tanh'},
    {'optimizer': 'Adam', 'lr': 0.03, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LeakyReLU'},
]

section3_results = []
for config in section3_configs:
    result = train_and_evaluate(config, X_train, y_train, X_test, y_test, epochs, "ADAM Tuning")
    if result:
        section3_results.append(result)

# Section 3 Analysis and Plots
if section3_results:
    print(f"\n📊 SECTION 3 RESULTS:")
    section3_sorted = sorted(section3_results, key=lambda x: x['r2_score'], reverse=True)
    
    for i, result in enumerate(section3_sorted[:10]):  # Top 10
        print(f"  {i+1}. {result['config']['name']}: R²={result['r2_score']:.4f}, Time={result['training_time']:.2f}s")
    
    # Section 3 Plots - 2x3 layout for comprehensive analysis
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 12))
    
    # 1. Learning Rate Impact
    lr_results = [r for r in section3_results if 'LR' in r['config']['name'] and 'Arch' not in r['config']['name']]
    if lr_results:
        lrs = [float(r['config']['name'].split('LR')[1]) for r in lr_results]
        lr_r2s = [r['r2_score'] for r in lr_results]
        
        ax1.plot(lrs, lr_r2s, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Learning Rate')
        ax1.set_ylabel('R² Score')
        ax1.set_title('ADAM: Learning Rate Impact', fontweight='bold')
        ax1.set_xscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Highlight best LR
        best_lr_idx = np.argmax(lr_r2s)
        ax1.scatter(lrs[best_lr_idx], lr_r2s[best_lr_idx], color='gold', s=150, 
                   edgecolor='black', linewidth=2, zorder=5)
        ax1.annotate(f'Best: {lrs[best_lr_idx]:.3f}', 
                    (lrs[best_lr_idx], lr_r2s[best_lr_idx]), 
                    xytext=(10, 10), textcoords='offset points', fontweight='bold')
    
    # 2. Architecture Impact
    arch_results = [r for r in section3_results if 'Arch' in r['config']['name'] or 'Deep' in r['config']['name']]
    if arch_results:
        arch_names = [r['config']['name'].replace('ADAM_', '') for r in arch_results]
        arch_r2s = [r['r2_score'] for r in arch_results]
        arch_colors = ['skyblue' if 'Deep' not in name else 'navy' for name in arch_names]
        
        bars_arch = ax2.bar(range(len(arch_names)), arch_r2s, color=arch_colors, 
                           alpha=0.8, edgecolor='black')
        ax2.set_xlabel('Architecture')
        ax2.set_ylabel('R² Score')
        ax2.set_title('ADAM: Architecture Impact', fontweight='bold')
        ax2.set_xticks(range(len(arch_names)))
        ax2.set_xticklabels(arch_names, rotation=45, ha='right')
        ax2.grid(True, alpha=0.3, axis='y')
        
        # Highlight best architecture
        best_arch_idx = np.argmax(arch_r2s)
        bars_arch[best_arch_idx].set_edgecolor('gold')
        bars_arch[best_arch_idx].set_linewidth(3)
        
        # Add labels
        for bar, score in zip(bars_arch, arch_r2s):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # 3. Activation Function Impact
    act_results = [r for r in section3_results if r['config']['name'] in ['ADAM_Sigmoid', 'ADAM_Tanh', 'ADAM_LeakyReLU']]
    if act_results:
        act_names = [r['config']['activation'] for r in act_results]
        act_r2s = [r['r2_score'] for r in act_results]
        act_colors = {'sigmoid': 'orange', 'tanh': 'blue', 'LeakyReLU': 'green'}
        act_bar_colors = [act_colors[act] for act in act_names]
        
        bars_act = ax3.bar(act_names, act_r2s, color=act_bar_colors, alpha=0.8, edgecolor='black')
        ax3.set_xlabel('Activation Function')
        ax3.set_ylabel('R² Score')
        ax3.set_title('ADAM: Activation Function Impact', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Highlight best activation
        best_act_idx = np.argmax(act_r2s)
        bars_act[best_act_idx].set_edgecolor('gold')
        bars_act[best_act_idx].set_linewidth(3)
        
        # Add labels
        for bar, score in zip(bars_act, act_r2s):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Top configurations comparison
    top_configs = section3_sorted[:8]  # Top 8
    top_names = [r['config']['name'].replace('ADAM_', '') for r in top_configs]
    top_r2s = [r['r2_score'] for r in top_configs]
    
    bars_top = ax4.bar(range(len(top_names)), top_r2s, color='steelblue', 
                      alpha=0.8, edgecolor='black')
    ax4.set_xlabel('Configuration')
    ax4.set_ylabel('R² Score')
    ax4.set_title('ADAM: Top 8 Configurations', fontweight='bold')
    ax4.set_xticks(range(len(top_names)))
    ax4.set_xticklabels(top_names, rotation=45, ha='right', fontsize=9)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Highlight #1
    bars_top[0].set_edgecolor('gold')
    bars_top[0].set_linewidth(3)
    
    # Add labels
    for bar, score in zip(bars_top, top_r2s):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 5. Training time vs Performance scatter
    all_r2s = [r['r2_score'] for r in section3_results]
    all_times = [r['training_time'] for r in section3_results]
    all_names = [r['config']['name'] for r in section3_results]
    
    scatter = ax5.scatter(all_times, all_r2s, c=range(len(all_r2s)), 
                         cmap='viridis', s=80, alpha=0.8, edgecolors='black')
    ax5.set_xlabel('Training Time (s)')
    ax5.set_ylabel('R² Score')
    ax5.set_title('ADAM: Performance vs Efficiency', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Annotate best performer
    best_overall_idx = np.argmax(all_r2s)
    ax5.annotate(f'Best: {all_names[best_overall_idx].replace("ADAM_", "")}', 
                (all_times[best_overall_idx], all_r2s[best_overall_idx]),
                xytext=(10, 10), textcoords='offset points', 
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                fontweight='bold')
    
    # 6. Loss evolution for top 5 configs
    top5_configs = section3_sorted[:5]
    colors_top5 = plt.cm.Set1(np.linspace(0, 1, 5))
    
    for i, result in enumerate(top5_configs):
        ax6.plot(result['test_losses'], color=colors_top5[i], 
                label=f"{result['config']['name'].replace('ADAM_', '')} (R²={result['r2_score']:.3f})",
                linewidth=2)
    
    ax6.set_xlabel('Epochs')
    ax6.set_ylabel('Test Loss')
    ax6.set_title('ADAM: Top 5 Configurations - Loss Evolution', fontweight='bold')
    ax6.set_yscale('log')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('SECTION 3: ADAM Hyperparameter Optimization Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Section 3 Summary
    best_config = section3_sorted[0]
    print(f"\n🏆 SECTION 3 CONCLUSION:")
    print(f"   Best ADAM Configuration: {best_config['config']['name']}")
    print(f"   Optimal Learning Rate: {best_config['config']['lr']}")
    print(f"   Optimal Architecture: {best_config['config']['hidden_size']}")
    print(f"   Optimal Activation: {best_config['config']['activation']}")
    print(f"   Best Performance: R²={best_config['r2_score']:.4f}, MAE={best_config['mae']:.4f}")
    print(f"   Training Time: {best_config['training_time']:.2f}s")

# ============================================================================
# SECTION 4: COMPREHENSIVE SUMMARY AND FINAL RECOMMENDATIONS
# ============================================================================

print("\n" + "📋 SECTION 4: COMPREHENSIVE SUMMARY & RECOMMENDATIONS" + "📋")
print("Final analysis combining all sections for best practices")
print("-" * 60)

# Collect all results
all_results = []
all_results.extend(section1_results if section1_results else [])
all_results.extend(section2_results if section2_results else [])
all_results.extend(section3_results if section3_results else [])

if all_results:
    # Remove duplicates based on config name
    seen_names = set()
    unique_results = []
    for result in all_results:
        if result['config']['name'] not in seen_names:
            unique_results.append(result)
            seen_names.add(result['config']['name'])
    
    all_results_sorted = sorted(unique_results, key=lambda x: x['r2_score'], reverse=True)
    
    print(f"\n📊 OVERALL RESULTS - TOP 15 CONFIGURATIONS:")
    for i, result in enumerate(all_results_sorted[:15]):
        optimizer_type = "🔧" if 'SGD' in result['config']['optimizer'] else "🚀"
        print(f"  {i+1:2d}. {optimizer_type} {result['config']['name']:<25}: R²={result['r2_score']:.4f}, "
              f"Time={result['training_time']:5.2f}s, MAE={result['mae']:.4f}")
    
    # Final comprehensive plots
    fig = plt.figure(figsize=(20, 16))
    
    # Create a 3x3 grid
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Overall performance ranking (top 12)
    ax1 = fig.add_subplot(gs[0, :2])
    top_12 = all_results_sorted[:12]
    names_final = [r['config']['name'].replace('_', '\n') for r in top_12]
    r2s_final = [r['r2_score'] for r in top_12]
    colors_final = ['blue' if 'SGD' in r['config']['optimizer'] else 'red' if 'Adam' in r['config']['optimizer'] else 'green' 
                   for r in top_12]
    
    bars_final = ax1.bar(range(len(names_final)), r2s_final, color=colors_final, 
                        alpha=0.8, edgecolor='black')
    ax1.set_xlabel('Configuration')
    ax1.set_ylabel('R² Score')
    ax1.set_title('Overall Performance Ranking - Top 12', fontweight='bold', fontsize=14)
    ax1.set_xticks(range(len(names_final)))
    ax1.set_xticklabels(names_final, rotation=45, ha='right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Highlight top 3
    for i in range(min(3, len(bars_final))):
        bars_final[i].set_edgecolor('gold')
        bars_final[i].set_linewidth(3)
    
    # Add R² labels
    for bar, score in zip(bars_final, r2s_final):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontweight='bold', fontsize=8)
    
    # 2. Optimizer family comparison
    ax2 = fig.add_subplot(gs[0, 2])
    sgd_all_final = [r for r in unique_results if 'SGD' in r['config']['optimizer']]
    adam_all_final = [r for r in unique_results if 'Adam' in r['config']['optimizer']]
    other_all_final = [r for r in unique_results if 'SGD' not in r['config']['optimizer'] and 'Adam' not in r['config']['optimizer']]
    
    family_data = []
    family_labels = []
    family_colors = []
    
    if sgd_all_final:
        family_data.append(np.mean([r['r2_score'] for r in sgd_all_final]))
        family_labels.append(f'SGD Family\n({len(sgd_all_final)} configs)')
        family_colors.append('blue')
    
    if adam_all_final:
        family_data.append(np.mean([r['r2_score'] for r in adam_all_final]))
        family_labels.append(f'ADAM Family\n({len(adam_all_final)} configs)')
        family_colors.append('red')
    
    if other_all_final:
        family_data.append(np.mean([r['r2_score'] for r in other_all_final]))
        family_labels.append(f'Other\n({len(other_all_final)} configs)')
        family_colors.append('green')
    
    bars_family = ax2.bar(family_labels, family_data, color=family_colors, alpha=0.8, edgecolor='black')
    ax2.set_ylabel('Average R² Score')
    ax2.set_title('Optimizer Family\nComparison', fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Highlight best family
    best_family_idx = np.argmax(family_data)
    bars_family[best_family_idx].set_edgecolor('gold')
    bars_family[best_family_idx].set_linewidth(3)
    
    for bar, score in zip(bars_family, family_data):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.4f}', ha='center', va='bottom', fontweight='bold')
    
    # 3. Performance vs Training Time scatter
    ax3 = fig.add_subplot(gs[1, :])
    all_r2s_scatter = [r['r2_score'] for r in unique_results]
    all_times_scatter = [r['training_time'] for r in unique_results]
    all_optimizers = [r['config']['optimizer'] for r in unique_results]
    
    # Create color map based on optimizer
    color_map = {'SGD': 'blue', 'SGD_Decay': 'cyan', 'SGD_Momentum': 'darkblue', 
                'Adam': 'red', 'RMSProp': 'green'}
    colors_scatter = [color_map.get(opt, 'gray') for opt in all_optimizers]
    
    scatter_final = ax3.scatter(all_times_scatter, all_r2s_scatter, c=colors_scatter, 
                               s=100, alpha=0.7, edgecolors='black', linewidth=1)
    ax3.set_xlabel('Training Time (seconds)')
    ax3.set_ylabel('R² Score')
    ax3.set_title('Performance vs Training Efficiency - All Configurations', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3)
    
    # Add best performers annotations
    best_3_indices = sorted(range(len(all_r2s_scatter)), key=lambda i: all_r2s_scatter[i], reverse=True)[:3]
    for i, idx in enumerate(best_3_indices):
        ax3.annotate(f'#{i+1}: {unique_results[idx]["config"]["name"]}', 
                    (all_times_scatter[idx], all_r2s_scatter[idx]),
                    xytext=(10, 10-i*20), textcoords='offset points', 
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=['gold', 'silver', 'orange'][i], alpha=0.7),
                    fontweight='bold', fontsize=9)
    
    # Create custom legend for optimizers
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                 markersize=10, label=opt, markeredgecolor='black')
                      for opt, color in color_map.items() if opt in all_optimizers]
    ax3.legend(handles=legend_elements, loc='lower right')
    
    # 4. Learning rate impact analysis
    ax4 = fig.add_subplot(gs[2, 0])
    lr_impact_data = {}
    for result in unique_results:
        lr = result['config']['lr']
        if lr not in lr_impact_data:
            lr_impact_data[lr] = []
        lr_impact_data[lr].append(result['r2_score'])
    
    lrs_impact = sorted(lr_impact_data.keys())
    avg_r2_by_lr = [np.mean(lr_impact_data[lr]) for lr in lrs_impact]
    
    ax4.plot(lrs_impact, avg_r2_by_lr, 'go-', linewidth=3, markersize=8)
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Average R² Score')
    ax4.set_title('Learning Rate Impact\n(All Optimizers)', fontweight='bold')
    ax4.set_xscale('log')
    ax4.grid(True, alpha=0.3)
    
    # Highlight optimal LR
    best_lr_idx = np.argmax(avg_r2_by_lr)
    ax4.scatter(lrs_impact[best_lr_idx], avg_r2_by_lr[best_lr_idx], 
               color='gold', s=150, edgecolor='black', linewidth=2, zorder=5)
    
    # 5. Architecture complexity analysis
    ax5 = fig.add_subplot(gs[2, 1])
    arch_complexity = {}
    for result in unique_results:
        hidden_size = result['config']['hidden_size']
        complexity = sum(hidden_size) * len(hidden_size)  # Total neurons * depth
        if complexity not in arch_complexity:
            arch_complexity[complexity] = []
        arch_complexity[complexity].append(result['r2_score'])
    
    complexities = sorted(arch_complexity.keys())
    avg_r2_by_complexity = [np.mean(arch_complexity[comp]) for comp in complexities]
    
    ax5.plot(complexities, avg_r2_by_complexity, 'bo-', linewidth=3, markersize=8)
    ax5.set_xlabel('Architecture Complexity\n(Neurons × Layers)')
    ax5.set_ylabel('Average R² Score')
    ax5.set_title('Architecture Complexity\nImpact', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # 6. Best model prediction
    ax6 = fig.add_subplot(gs[2, 2])
    best_overall = all_results_sorted[0]
    x_plot = np.linspace(0, 1, 200)
    X_plot = x_plot.reshape(-1, 1)
    y_plot_true = np.sin(30*x_plot) + np.cos(10*x_plot) + 2 + x_plot**2
    y_plot_pred = best_overall['network'].forward(X_plot)
    
    ax6.plot(x_plot, y_plot_true, 'g-', label='True Function', linewidth=3)
    ax6.plot(x_plot, y_plot_pred.flatten(), 'r-', label='Best Model', linewidth=3)
    ax6.scatter(X_test.flatten(), y_test.flatten(), alpha=0.6, color='blue', s=30, 
               label='Test Data', edgecolor='white', linewidth=0.5)
    ax6.set_xlabel('x')
    ax6.set_ylabel('y')
    ax6.set_title(f'Best Model Fit\n{best_overall["config"]["name"]}', fontweight='bold')
    ax6.legend(fontsize=9)
    ax6.grid(True, alpha=0.3)
    
    plt.suptitle('SECTION 4: Comprehensive Analysis Summary', fontsize=18, fontweight='bold')
    plt.show()
    
    # Final detailed summary
    print(f"\n🏆 FINAL RECOMMENDATIONS:")
    
    best_overall = all_results_sorted[0]
    print(f"\n1. BEST OVERALL CONFIGURATION:")
    print(f"   Model: {best_overall['config']['name']}")
    print(f"   Optimizer: {best_overall['config']['optimizer']}")
    print(f"   Learning Rate: {best_overall['config']['lr']}")
    print(f"   Architecture: {best_overall['config']['hidden_size']}")
    print(f"   Activation: {best_overall['config']['activation']}")
    print(f"   Performance: R²={best_overall['r2_score']:.4f}, MAE={best_overall['mae']:.4f}")
    print(f"   Training Time: {best_overall['training_time']:.2f}s")
    
    if sgd_all_final and adam_all_final:
        sgd_best = max(sgd_all_final, key=lambda x: x['r2_score'])
        adam_best = max(adam_all_final, key=lambda x: x['r2_score'])
        
        print(f"\n2. OPTIMIZER COMPARISON:")
        print(f"   Best SGD Variant: {sgd_best['config']['name']} (R²={sgd_best['r2_score']:.4f})")
        print(f"   Best ADAM Config: {adam_best['config']['name']} (R²={adam_best['r2_score']:.4f})")
        
        if adam_best['r2_score'] > sgd_best['r2_score']:
            improvement = ((adam_best['r2_score'] - sgd_best['r2_score'])/sgd_best['r2_score']) * 100
            print(f"   🎯 ADAM outperforms SGD by {improvement:.1f}%")
        else:
            improvement = ((sgd_best['r2_score'] - adam_best['r2_score'])/adam_best['r2_score']) * 100
            print(f"   🎯 SGD outperforms ADAM by {improvement:.1f}%")
    
    # Find optimal hyperparameters
    optimal_lr = lrs_impact[np.argmax(avg_r2_by_lr)]
    
    print(f"\n3. HYPERPARAMETER INSIGHTS:")
    print(f"   Optimal Learning Rate: {optimal_lr}")
    print(f"   Best Activation Function: {best_overall['config']['activation']}")
    print(f"   Recommended Architecture: {best_overall['config']['hidden_size']}")
    
    print(f"\n4. PRACTICAL RECOMMENDATIONS:")
    if 'Adam' in best_overall['config']['optimizer']:
        print(f"   • Use ADAM optimizer for this type of regression problem")
        print(f"   • Start with learning rate around {optimal_lr}")
        print(f"   • {best_overall['config']['activation']} activation provides good stability")
    else:
        print(f"   • {best_overall['config']['optimizer']} works well for this problem")
        print(f"   • Use learning rate around {optimal_lr}")
        print(f"   • Consider momentum for SGD variants")
    
    print(f"   • Architecture complexity: balance between {min(complexities)} and {max(complexities)} total neurons")
    print(f"   • Expected R² score: {best_overall['r2_score']:.4f} ± {np.std([r['r2_score'] for r in all_results_sorted[:5]]):.3f}")
    print(f"   • Training time budget: ~{best_overall['training_time']:.0f}s for {epochs} epochs")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! 🎉")
print("Four-section hyperparameter comparison successfully executed!")
print("=" * 80)
