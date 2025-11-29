import numpy as np
import matplotlib.pyplot as plt
import time
import myNN as nn

print("=" * 80)
print("ADAM OPTIMIZER: HYPERPARAMETER SENSITIVITY STUDY")
print("Comprehensive Analysis of Learning Rates, Architectures & Activations")
print("=" * 80)

# Generate data - more complex function
npts = 100
xpts = np.linspace(0,1,npts)
def y_func(x):
    return np.sin(30*x) + np.cos(10*x) + x**2
yreal = y_func(xpts)
X = xpts.reshape(npts,1)
y = yreal.reshape(npts,1)

# Split data once for all experiments
dummy_nn = nn.myNeuralNetwork()
X_train, X_test, y_train, y_test = dummy_nn.train_test_split(X, y, test_size=0.2, random_state=42)

# Training parameters
epochs = 15001  # Reduced to see clearer differences between configurations

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
            
            if epoch % 5000 == 0:
                print(f"  [{section_name}] {config['name']} - Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
        
        # Final evaluation
        if len(train_losses) > 100:
            training_time = time.time() - start_time
            test_results = NeuralNetwork.evaluate(X_test, y_test)
            final_ratio = test_losses[-1] / train_losses[-1] if train_losses[-1] > 0 else 1.0
            
            # Overfitting detection metrics
            # Check if test loss is increasing in last 20% of training
            last_20_percent = int(len(test_losses) * 0.2)
            test_loss_trend = test_losses[-1] - test_losses[-last_20_percent]
            is_overfitting = test_loss_trend > 0 and final_ratio > 1.1  # Test loss increasing & ratio > 1.1
            
            # Gap between train and test loss
            loss_gap = test_losses[-1] - train_losses[-1]
            
            result = {
                'config': config,
                'train_losses': train_losses,
                'test_losses': test_losses,
                'final_train_loss': train_losses[-1],
                'final_test_loss': test_losses[-1],
                'ratio': final_ratio,
                'loss_gap': loss_gap,
                'is_overfitting': is_overfitting,
                'r2_score': test_results['r2_score'],
                'mae': test_results['mae'],
                'training_time': training_time,
                'network': NeuralNetwork
            }
            
            overfit_indicator = "⚠️ OVERFIT" if is_overfitting else "✅"
            print(f"  {overfit_indicator} [{section_name}] {config['name']}: R²={test_results['r2_score']:.4f}, "
                  f"Test/Train={final_ratio:.3f}, MAE={test_results['mae']:.4f}, Time={training_time:.2f}s")
            return result
        else:
            print(f"  ❌ [{section_name}] {config['name']}: Insufficient training data")
            return None
            
    except Exception as e:
        print(f"  ❌ [{section_name}] {config['name']}: Error - {str(e)}")
        return None

# ============================================================================
# ADAM HYPERPARAMETER SENSITIVITY STUDY
# ============================================================================

print("\n" + "🚀 ADAM HYPERPARAMETER SENSITIVITY ANALYSIS" + "🚀")
print("Comprehensive study: Learning rates, architectures, activations")
print("-" * 60)

adam_configs = [
    # Learning rate sweep - wider range with extremes
    {'optimizer': 'Adam', 'lr': 0.0001, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LR0.0001'},
    {'optimizer': 'Adam', 'lr': 0.001, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LR0.001'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LR0.01'},
    {'optimizer': 'Adam', 'lr': 0.05, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LR0.05'},
    {'optimizer': 'Adam', 'lr': 0.2, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LR0.2'},
    
    # Architecture sweep - extreme sizes
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [10], 'activation': 'LeakyReLU', 'name': 'ADAM_Arch10'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [50], 'activation': 'LeakyReLU', 'name': 'ADAM_Arch50'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_Arch100'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [300], 'activation': 'LeakyReLU', 'name': 'ADAM_Arch300'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [500], 'activation': 'LeakyReLU', 'name': 'ADAM_Arch500'},
    
    # Deep networks - various depths
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100, 50], 'activation': 'LeakyReLU', 'name': 'ADAM_Deep2'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100, 75, 50, 25], 'activation': 'LeakyReLU', 'name': 'ADAM_Deep4'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100, 80, 60, 40, 20], 'activation': 'LeakyReLU', 'name': 'ADAM_Deep5'},
    
    # Activation function sweep
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'sigmoid', 'name': 'ADAM_Sigmoid'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_Tanh'},
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LeakyReLU'},
]

adam_results = []
for config in adam_configs:
    result = train_and_evaluate(config, X_train, y_train, X_test, y_test, epochs, "ADAM Study")
    if result:
        adam_results.append(result)

# ADAM Analysis and Plots
if adam_results:
    print(f"\n📊 ADAM SENSITIVITY STUDY RESULTS:")
    adam_sorted = sorted(adam_results, key=lambda x: x['r2_score'], reverse=True)
    
    # Count overfitting cases
    overfit_count = sum(1 for r in adam_results if r.get('is_overfitting', False))
    print(f"\n⚠️  Overfitting detected in {overfit_count}/{len(adam_results)} configurations\n")
    
    for i, result in enumerate(adam_sorted[:10]):  # Top 10
        overfit_flag = "⚠️" if result.get('is_overfitting', False) else "  "
        print(f"  {overfit_flag}{i+1}. {result['config']['name']:<30}: R²={result['r2_score']:.4f}, "
              f"Test/Train={result['ratio']:.3f}, Time={result['training_time']:.2f}s")
    
    # Plot 1: Learning Rate Impact
    lr_results = [r for r in adam_results if 'LR' in r['config']['name'] and 'Arch' not in r['config']['name']]
    if lr_results:
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        lrs = [r['config']['lr'] for r in lr_results]
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
        plt.tight_layout()
        plt.show()
    
    # Plot 2: Architecture Impact
    arch_results = [r for r in adam_results if 'Arch' in r['config']['name'] or 'Deep' in r['config']['name']]
    if arch_results:
        fig2, ax2 = plt.subplots(figsize=(12, 6))
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
        plt.tight_layout()
        plt.show()
    
    # Plot 3: Activation Function Impact
    act_results = [r for r in adam_results if r['config']['name'] in ['ADAM_Sigmoid', 'ADAM_Tanh', 'ADAM_LeakyReLU']]
    if act_results:
        fig3, ax3 = plt.subplots(figsize=(8, 6))
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
        plt.tight_layout()
        plt.show()
    
    # Plot 4: Overfitting Analysis
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    config_names_short = [r['config']['name'].replace('ADAM_', '') for r in adam_sorted]
    test_train_ratios = [r['ratio'] for r in adam_sorted]
    # Use the same overfitting archetypes used elsewhere: 
    # - red: flagged as overfitting by the detection logic (`is_overfitting`)
    # - orange: warning when ratio is > 1.05 (mild gap)
    # - green: good (ratio close to 1)
    colors_overfit = [
        'red' if (r.get('is_overfitting', False) or r['ratio'] > 1.05) else 'green'
        for r in adam_sorted
    ]
    
    bars_overfit = ax4.barh(range(len(config_names_short)), test_train_ratios, 
                            color=colors_overfit, alpha=0.7, edgecolor='black')
    ax4.axvline(x=1.1, color='darkred', linestyle='--', linewidth=2, label='Overfit Threshold (ratio=1.1)')
    ax4.set_xlabel('Test Loss / Train Loss Ratio')
    ax4.set_ylabel('Configuration')
    ax4.set_title('Overfitting Detection', fontweight='bold')
    ax4.set_yticks(range(len(config_names_short)))
    ax4.set_yticklabels(config_names_short, fontsize=8)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, axis='x')
    
    # Add ratio labels
    for i, (bar, ratio) in enumerate(zip(bars_overfit, test_train_ratios)):
        width = bar.get_width()
        ax4.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                f'{ratio:.3f}', ha='left', va='center', fontweight='bold', fontsize=7)
    plt.tight_layout()
    plt.show()
    
    # Plot 5: Performance vs Efficiency
    # Use the same sorted order and color mapping as the overfitting plot so archetypes match
    fig5, ax5 = plt.subplots(figsize=(12, 6))
    all_r2s = [r['r2_score'] for r in adam_sorted]
    all_times = [r['training_time'] for r in adam_sorted]
    all_names = [r['config']['name'].replace('ADAM_', '') for r in adam_sorted]
    colors_scatter = colors_overfit
    
    # Plot points with numbers
    for i, (time, r2, color) in enumerate(zip(all_times, all_r2s, colors_scatter)):
        ax5.scatter(time, r2, c=color, s=150, alpha=0.6, edgecolors='black', linewidth=1.5)
        ax5.text(time, r2, str(i+1), ha='center', va='center', fontsize=8, fontweight='bold', color='white')
    
    ax5.set_xlabel('Training Time (s)')
    ax5.set_ylabel('R² Score')
    ax5.set_title('Performance vs Efficiency', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Create legend mapping numbers to names
    legend_text = "Configuration Key:\n" + "\n".join([f"{i+1}. {name}" for i, name in enumerate(all_names)])
    ax5.text(1.02, 0.5, legend_text, transform=ax5.transAxes, fontsize=7,
            verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    plt.tight_layout()
    plt.show()
    
    # Plot 6: Best vs Worst Configurations
    fig6, ax6 = plt.subplots(figsize=(12, 6))
    top3_configs = adam_sorted[:3]
    worst2_configs = adam_sorted[-2:]
    colors_top = ['green', 'lime', 'darkgreen']
    colors_worst = ['red', 'darkred']
    
    # Plot best 3 configurations
    for i, result in enumerate(top3_configs):
        ax6.plot(result['test_losses'], color=colors_top[i], linestyle='-', 
                linewidth=2.5, label=f"#{i+1}: {result['config']['name'].replace('ADAM_', '')} (R²={result['r2_score']:.3f})",
                alpha=0.8)
    
    # Plot worst 2 configurations
    for i, result in enumerate(worst2_configs):
        rank = len(adam_sorted) - len(worst2_configs) + i + 1
        ax6.plot(result['test_losses'], color=colors_worst[i], linestyle='--', 
                linewidth=2, label=f"#{rank}: {result['config']['name'].replace('ADAM_', '')} (R²={result['r2_score']:.3f})",
                alpha=0.7)
    
    ax6.set_xlabel('Epochs')
    ax6.set_ylabel('Test Loss')
    ax6.set_title('Best 3 vs Worst 2 Configurations (Test Loss)', fontweight='bold')
    ax6.set_yscale('log')
    ax6.legend(fontsize=8, loc='best')
    ax6.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # ADAM Study Summary
    best_config = adam_sorted[0]
    print(f"\n🏆 ADAM SENSITIVITY STUDY CONCLUSION:")
    print(f"   Best ADAM Configuration: {best_config['config']['name']}")
    print(f"   Optimal Learning Rate: {best_config['config']['lr']}")
    print(f"   Optimal Architecture: {best_config['config']['hidden_size']}")
    print(f"   Optimal Activation: {best_config['config']['activation']}")
    print(f"   Best Performance: R²={best_config['r2_score']:.4f}, MAE={best_config['mae']:.4f}")
    print(f"   Test/Train Ratio: {best_config['ratio']:.3f} {'⚠️ (Overfitting!)' if best_config.get('is_overfitting', False) else '✅ (Good generalization)'}")
    print(f"   Training Time: {best_config['training_time']:.2f}s")
    
    # Overfitting insights
    print(f"\n📊 OVERFITTING ANALYSIS:")
    print(f"   Configurations with overfitting: {overfit_count}/{len(adam_results)}")
    if overfit_count > 0:
        overfit_configs = [r for r in adam_results if r.get('is_overfitting', False)]
        print(f"   Most overfitted config: {max(overfit_configs, key=lambda x: x['ratio'])['config']['name']} (ratio={max(overfit_configs, key=lambda x: x['ratio'])['ratio']:.3f})")
    best_generalization = min(adam_results, key=lambda x: abs(x['ratio'] - 1.0))
    print(f"   Best generalization: {best_generalization['config']['name']} (ratio={best_generalization['ratio']:.3f})")

# ============================================================================
# FINAL SUMMARY AND RECOMMENDATIONS
# ============================================================================

if adam_results:
    print("\n" + "="  * 80)
    print("📋 FINAL SUMMARY & RECOMMENDATIONS")
    print("=" * 80)
    
    adam_sorted_final = sorted(adam_results, key=lambda x: x['r2_score'], reverse=True)
    best_overall = adam_sorted_final[0]
    
    print(f"\n📊 OVERALL RESULTS - TOP 10 CONFIGURATIONS:")
    for i, result in enumerate(adam_sorted_final[:10]):
        print(f"  {i+1:2d}. {result['config']['name']:<25}: R²={result['r2_score']:.4f}, "
              f"Time={result['training_time']:5.2f}s, MAE={result['mae']:.4f}")
    
    # Learning rate analysis
    lr_impact_data = {}
    for result in adam_results:
        lr = result['config']['lr']
        if lr not in lr_impact_data:
            lr_impact_data[lr] = []
        lr_impact_data[lr].append(result['r2_score'])
    
    lrs_impact = sorted(lr_impact_data.keys())
    avg_r2_by_lr = [np.mean(lr_impact_data[lr]) for lr in lrs_impact]
    optimal_lr = lrs_impact[np.argmax(avg_r2_by_lr)]
    
    print(f"\n🏆 BEST OVERALL CONFIGURATION:")
    print(f"   Model: {best_overall['config']['name']}")
    print(f"   Optimizer: {best_overall['config']['optimizer']}")
    print(f"   Learning Rate: {best_overall['config']['lr']}")
    print(f"   Architecture: {best_overall['config']['hidden_size']}")
    print(f"   Activation: {best_overall['config']['activation']}")
    print(f"   Performance: R²={best_overall['r2_score']:.4f}, MAE={best_overall['mae']:.4f}")
    print(f"   Training Time: {best_overall['training_time']:.2f}s")
    
    print(f"\n💡 HYPERPARAMETER INSIGHTS:")
    print(f"   Optimal Learning Rate: {optimal_lr} (averaged across architectures)")
    print(f"   Best Activation Function: {best_overall['config']['activation']}")
    print(f"   Recommended Architecture: {best_overall['config']['hidden_size']}")
    
    print(f"\n📝 PRACTICAL RECOMMENDATIONS:")
    print(f"   • Use ADAM optimizer for this type of regression problem")
    print(f"   • Start with learning rate around {optimal_lr}")
    print(f"   • {best_overall['config']['activation']} activation provides good stability")
    print(f"   • Expected R² score: {best_overall['r2_score']:.4f} ± {np.std([r['r2_score'] for r in adam_sorted_final[:5]]):.3f}")
    print(f"   • Training time budget: ~{best_overall['training_time']:.0f}s for {epochs} epochs")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE! 🎉")
print("ADAM hyperparameter sensitivity study successfully executed!")
print("=" * 80)