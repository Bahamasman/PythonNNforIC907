import numpy as np
import matplotlib.pyplot as plt
import time
import myNN as nn

# Generate data
npts = 100
xpts = np.linspace(0,1,npts)
yreal = np.sin(30*xpts) + np.cos(10*xpts) + 2 + xpts**2
X = xpts.reshape(npts,1)
y = yreal.reshape(npts,1)

# Improved configurations for fairer comparison
configs = [
    # ADAM variants
    {'optimizer': 'Adam', 'lr': 0.01, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_standard', 'epochs': 8000},
    {'optimizer': 'Adam', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_higher_lr', 'epochs': 8000},
    {'optimizer': 'Adam', 'lr': 0.1, 'hidden_size': [100], 'activation': 'tanh', 'name': 'ADAM_aggressive', 'epochs': 8000},
    
    # SGD Momentum variants with more training
    {'optimizer': 'SGD_Momentum', 'lr': 0.05, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_standard', 'epochs': 12000},
    {'optimizer': 'SGD_Momentum', 'lr': 0.1, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_aggressive', 'epochs': 12000},
    {'optimizer': 'SGD_Momentum', 'lr': 0.15, 'hidden_size': [100], 'activation': 'tanh', 'name': 'SGD_Momentum_very_aggressive', 'epochs': 12000},
    
    # Architecture tests
    {'optimizer': 'Adam', 'lr': 0.05, 'hidden_size': [50], 'activation': 'tanh', 'name': 'ADAM_small_arch', 'epochs': 8000},
    {'optimizer': 'SGD_Momentum', 'lr': 0.1, 'hidden_size': [50], 'activation': 'tanh', 'name': 'SGD_Momentum_small_arch', 'epochs': 12000},
    
    # LeakyReLU comparison
    {'optimizer': 'Adam', 'lr': 0.05, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'ADAM_LeakyReLU', 'epochs': 8000},
    {'optimizer': 'SGD_Momentum', 'lr': 0.1, 'hidden_size': [100], 'activation': 'LeakyReLU', 'name': 'SGD_Momentum_LeakyReLU', 'epochs': 12000},
]

results = []

print("=" * 90)
print("COMPREHENSIVE SGD vs ADAM COMPARISON - OPTIMIZED FOR FAIRNESS")
print("=" * 90)

# Split data once
dummy_nn = nn.myNeuralNetwork()
X_train, X_test, y_train, y_test = dummy_nn.train_test_split(X, y, test_size=0.2, random_state=42)

for i, config in enumerate(configs):
    print(f"\n[{i+1}/{len(configs)}] Testing: {config['name']} ({config['epochs']} epochs)")
    
    try:
        # Create network
        NeuralNetwork = nn.myNeuralNetwork()
        hidden_size = config['hidden_size']
        num_hiddenLayers = len(hidden_size)
        epochs = config['epochs']
        
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
            NeuralNetwork.backward(y_train)
            NeuralNetwork.updateParam()
            
            train_losses.append(train_loss)
            
            test_pred = NeuralNetwork.forward(X_test)
            test_loss = NeuralNetwork.computeLoss(y_test)
            test_losses.append(test_loss)
            
            if epoch % 2000 == 0:
                print(f"  Epoch {epoch}: Train Loss = {train_loss:.6f}, Test Loss = {test_loss:.6f}")
        
        training_time = time.time() - start_time
        
        # Evaluate
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
        results.append(result)
        
        print(f"  ✅ Final: R²={test_results['r2_score']:.4f}, MAE={test_results['mae']:.4f}, Time={training_time:.1f}s")
    
    except Exception as e:
        print(f"  ❌ Error: {str(e)}")

# Comprehensive analysis
print("\n" + "=" * 90)
print("DETAILED ANALYSIS")
print("=" * 90)

if results:
    results_sorted = sorted(results, key=lambda x: x['r2_score'], reverse=True)
    
    print(f"{'Rank':<4} {'Configuration':<30} {'R²':<8} {'MAE':<8} {'Time':<8} {'Epochs':<8}")
    print("-" * 85)
    for i, result in enumerate(results_sorted):
        epochs = result['config']['epochs']
        print(f"{i+1:<4} {result['config']['name']:<30} {result['r2_score']:<8.4f} {result['mae']:<8.4f} {result['training_time']:<8.1f} {epochs:<8}")

    # Optimizer-specific analysis
    sgd_results = [r for r in results if 'SGD' in r['config']['optimizer']]
    adam_results = [r for r in results if 'Adam' in r['config']['optimizer']]
    
    if sgd_results and adam_results:
        print(f"\n" + "="*60)
        print("OPTIMIZER PERFORMANCE BREAKDOWN")
        print("="*60)
        
        # Best from each optimizer
        best_sgd = max(sgd_results, key=lambda x: x['r2_score'])
        best_adam = max(adam_results, key=lambda x: x['r2_score'])
        
        print(f"\n🏆 CHAMPION MODELS:")
        print(f"  🥇 Overall Best: {results_sorted[0]['config']['name']}")
        print(f"     R²={results_sorted[0]['r2_score']:.4f}, Time={results_sorted[0]['training_time']:.1f}s")
        print(f"  🥈 SGD Best: {best_sgd['config']['name']}")
        print(f"     R²={best_sgd['r2_score']:.4f}, Time={best_sgd['training_time']:.1f}s")
        print(f"  🥉 ADAM Best: {best_adam['config']['name']}")
        print(f"     R²={best_adam['r2_score']:.4f}, Time={best_adam['training_time']:.1f}s")
        
        # Performance gap analysis
        performance_gap = ((best_sgd['r2_score'] / best_adam['r2_score']) - 1) * 100
        
        print(f"\n📊 PERFORMANCE COMPARISON:")
        print(f"  SGD vs ADAM performance gap: {performance_gap:+.1f}%")
        
        # Speed analysis
        sgd_time_per_epoch = best_sgd['training_time'] / best_sgd['config']['epochs']
        adam_time_per_epoch = best_adam['training_time'] / best_adam['config']['epochs']
        speed_advantage = ((adam_time_per_epoch / sgd_time_per_epoch) - 1) * 100
        
        print(f"  SGD speed advantage: {speed_advantage:+.1f}% (per epoch)")
        
        # Final verdict
        print(f"\n🎯 VERDICT:")
        if performance_gap > -5:
            print(f"  ✅ SUCCESS! SGD with momentum is competitive with ADAM")
            print(f"  💡 SGD achieves {100 + performance_gap:.1f}% of ADAM's performance")
        elif performance_gap > -15:
            print(f"  🟡 GOOD PROGRESS! SGD significantly improved")
            print(f"  💡 SGD achieves {100 + performance_gap:.1f}% of ADAM's performance")
        else:
            print(f"  ❌ ADAM still dominates with {-performance_gap:.1f}% better performance")
        
        if speed_advantage > 0:
            print(f"  ⚡ Bonus: SGD is {speed_advantage:.1f}% faster per epoch!")

    # Create comparison plot
    print(f"\n📈 Generating performance visualization...")
    
    plt.figure(figsize=(15, 10))
    
    # Subplot 1: R² scores comparison
    plt.subplot(2, 2, 1)
    names = [r['config']['name'] for r in results_sorted]
    r2_scores = [r['r2_score'] for r in results_sorted]
    colors = ['steelblue' if 'ADAM' in name else 'coral' for name in names]
    
    bars = plt.bar(range(len(names)), r2_scores, color=colors, alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('R² Score')
    plt.title('Model Performance Comparison', fontweight='bold')
    plt.xticks(range(len(names)), [name.replace('_', '\\n') for name in names], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Subplot 2: Training time comparison
    plt.subplot(2, 2, 2)
    times = [r['training_time'] for r in results_sorted]
    bars2 = plt.bar(range(len(names)), times, color=colors, alpha=0.8)
    plt.xlabel('Models')
    plt.ylabel('Training Time (s)')
    plt.title('Training Efficiency', fontweight='bold')
    plt.xticks(range(len(names)), [name.replace('_', '\\n') for name in names], rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Subplot 3: Loss curves for best models
    plt.subplot(2, 2, 3)
    if sgd_results and adam_results:
        plt.plot(best_adam['test_losses'], 'b-', label=f"Best ADAM: {best_adam['config']['name']}", linewidth=2)
        plt.plot(best_sgd['test_losses'], 'r-', label=f"Best SGD: {best_sgd['config']['name']}", linewidth=2)
        plt.xlabel('Epochs')
        plt.ylabel('Test Loss')
        plt.title('Learning Curves: Best Models', fontweight='bold')
        plt.yscale('log')
        plt.legend()
        plt.grid(alpha=0.3)
    
    # Subplot 4: Summary statistics
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    if sgd_results and adam_results:
        summary_text = f"""COMPARISON SUMMARY

🏆 Best Overall: {results_sorted[0]['config']['name']}
   R² = {results_sorted[0]['r2_score']:.4f}

🥈 Best SGD: {best_sgd['config']['name']}
   R² = {best_sgd['r2_score']:.4f}
   Training Time: {best_sgd['training_time']:.1f}s

🥉 Best ADAM: {best_adam['config']['name']}
   R² = {best_adam['r2_score']:.4f}
   Training Time: {best_adam['training_time']:.1f}s

📊 Performance Gap: {performance_gap:+.1f}%

⚡ Speed per Epoch:
   SGD: {sgd_time_per_epoch:.3f}s
   ADAM: {adam_time_per_epoch:.3f}s
   Advantage: {speed_advantage:+.1f}%

💡 Recommendation: {'SGD' if performance_gap > -5 else 'ADAM'}"""

        plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, 
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8),
                verticalalignment='top', fontsize=10, fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig('sgd_vs_adam_comprehensive.png', dpi=300, bbox_inches='tight')
    plt.show()

else:
    print("❌ No successful results to analyze!")
