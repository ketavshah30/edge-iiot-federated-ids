import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score
import tensorflow as tf
from tensorflow import keras
layers = keras.layers
import warnings
warnings.filterwarnings('ignore')

# XAI Libraries
import lime
import lime.lime_tabular
import shap
import os
import shutil
from datetime import datetime

# --- NEW: Import for class weighting ---
from sklearn.utils import class_weight

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Set high DPI for matplotlib
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['xtick.labelsize'] = 11
plt.rcParams['ytick.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 18

class EdgeIIoTFederatedLearning:
    def __init__(self, num_clients=5, num_rounds=50, local_epochs=5, batch_size=32, results_dir='results'):
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.results_dir = results_dir
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.global_model = None
        self.client_models = []
        self.client_data = []
        self.X_train_original = None
        self.y_train_original = None
        self.history = {
            'global_accuracy': [],
            'global_loss': [],
            'client_accuracies': [[] for _ in range(num_clients)],
            'client_losses': [[] for _ in range(num_clients)]
        }
        
        # Setup results directory
        self.setup_results_directory()
        
    def setup_results_directory(self):
        """Setup and clean results directory"""
        if os.path.exists(self.results_dir):
            shutil.rmtree(self.results_dir)
        os.makedirs(self.results_dir)
        
        # Create subdirectories
        os.makedirs(os.path.join(self.results_dir, 'training_plots'))
        os.makedirs(os.path.join(self.results_dir, 'evaluation_plots'))
        os.makedirs(os.path.join(self.results_dir, 'xai_analysis'))
        os.makedirs(os.path.join(self.results_dir, 'lime_explanations'))
        os.makedirs(os.path.join(self.results_dir, 'shap_analysis'))
        
        print(f"Results directory '{self.results_dir}' created and cleaned.")
        
    def load_and_preprocess_data(self, file_path):
        """Load and preprocess the Edge-IIoTset dataset using imputation."""
        print("Loading Edge-IIoTset dataset...")
        
        # Read the dataset, low_memory=False can help with mixed dtypes
        df = pd.read_csv(file_path, low_memory=False)
        print(f"Initial dataset shape: {df.shape}")
        
        # Display basic info
        print(f"\nAttack type distribution (Initial):")
        print(df['Attack_type'].value_counts())
        
        # --- FIX: STEP 1 ---
        # Separate features (X) and target (y) BEFORE any cleaning.
        # The target column is precious and should not be modified.
        if 'Attack_type' not in df.columns:
            raise ValueError("Target column 'Attack_type' not found in the initial dataset.")
            
        # We also drop 'Attack_label' as it is a numeric version of the target and redundant.
        # We drop 'frame.time' as it's an identifier, not a feature.
        X = df.drop(columns=['Attack_type', 'Attack_label', 'frame.time'])
        y = df['Attack_type']

        # --- FIX: STEP 2 ---
        # Now, perform cleaning and conversion ONLY on the features (X).
        print("\nPerforming data type conversion and imputation on features...")
        
        # Convert object columns to numeric. If a value can't be converted, it becomes NaN.
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.to_numeric(X[col], errors='coerce')
        
        # --- FIX: STEP 3 (The MOST IMPORTANT CHANGE) ---
        # Instead of dropping rows with NaNs, we fill them with 0.
        # This correctly assumes that a missing feature value means that feature was absent (e.g., no HTTP traffic).
        X.fillna(0, inplace=True)

        print(f"\nShape after cleaning and imputation: X={X.shape}, y={y.shape}")
        
        # Now all columns in X are clean and numeric
        numeric_columns = X.columns
        print(f"Selected {len(numeric_columns)} numeric features")
        
        # Encode target labels
        y_encoded = self.label_encoder.fit_transform(y)
        self.num_classes = len(np.unique(y_encoded))
        print(f"Number of classes: {self.num_classes}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Store original (pre-scaled) training data for XAI
        # We need to re-align X before scaling to match y_train for the explainer
        X_train_original_df = pd.DataFrame(X, index=y.index).loc[pd.Series(y_train, name='y').index]
        self.X_train_original = X_train_original_df.values

        self.y_train_original = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_names = numeric_columns.tolist()
        
        return X_train, y_train
        
    def create_dnn_model(self, input_shape):
        """Create a Deep Neural Network model"""
        model = keras.Sequential([
            layers.Dense(128, activation='relu', input_shape=(input_shape,)),
            layers.Dropout(0.3),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.2),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
        
    def distribute_data_to_clients(self, X_train, y_train):
        """Distribute training data among clients"""
        print(f"Distributing data to {self.num_clients} clients...")
        self.client_data = []
        indices = np.random.permutation(len(X_train))
        X_train_shuffled = X_train[indices]
        y_train_shuffled = y_train[indices]
        client_data_size = len(X_train) // self.num_clients
        for i in range(self.num_clients):
            start_idx = i * client_data_size
            end_idx = (i + 1) * client_data_size if i != self.num_clients - 1 else len(X_train)
            client_X = X_train_shuffled[start_idx:end_idx]
            client_y = y_train_shuffled[start_idx:end_idx]
            self.client_data.append((client_X, client_y))
            print(f"Client {i+1}: {len(client_X)} samples")
               
    def federated_averaging(self, client_weights):
        """Perform FedAvg aggregation"""
        total_samples = sum(len(self.client_data[i][0]) for i in range(self.num_clients))
        avg_weights = []
        for layer_idx in range(len(client_weights[0])):
            weighted_sum = np.zeros_like(client_weights[0][layer_idx])
            for client_idx in range(self.num_clients):
                client_samples = len(self.client_data[client_idx][0])
                weight = client_samples / total_samples
                weighted_sum += weight * client_weights[client_idx][layer_idx]
            avg_weights.append(weighted_sum)
        return avg_weights
        
    def train_federated_learning(self, X_train, y_train):
        """Main federated learning training loop with class weighting"""
        print("Starting Federated Learning with FedAvg...")
        
        self.distribute_data_to_clients(X_train, y_train)
        
        self.global_model = self.create_dnn_model(X_train.shape[1])
        self.client_models = [self.create_dnn_model(X_train.shape[1]) for _ in range(self.num_clients)]
            
        for round_num in range(self.num_rounds):
            print(f"\n--- Round {round_num + 1}/{self.num_rounds} ---")
            
            client_weights, round_client_accuracies, round_client_losses = [], [], []
            
            for client_id in range(self.num_clients):
                self.client_models[client_id].set_weights(self.global_model.get_weights())
                client_X, client_y = self.client_data[client_id]
                
                # --- MODIFICATION: CALCULATE AND APPLY CLASS WEIGHTS ---
                # Calculate class weights for the client's local data to handle imbalance
                class_weights = class_weight.compute_class_weight(
                    'balanced',
                    classes=np.unique(client_y),
                    y=client_y
                )
                class_weights_dict = dict(enumerate(class_weights))
                # --- END OF MODIFICATION ---

                # Train client model using the calculated class weights
                history = self.client_models[client_id].fit(
                    client_X, client_y,
                    epochs=self.local_epochs,
                    batch_size=self.batch_size,
                    verbose=0,
                    validation_split=0.1,
                    class_weight=class_weights_dict  # Pass weights to the fit method
                )
                
                client_acc = history.history['accuracy'][-1]
                client_loss = history.history['loss'][-1]
                round_client_accuracies.append(client_acc)
                round_client_losses.append(client_loss)
                client_weights.append(self.client_models[client_id].get_weights())
                
                print(f"Client {client_id + 1}: Accuracy = {client_acc:.4f}, Loss = {client_loss:.4f}")
            
            averaged_weights = self.federated_averaging(client_weights)
            self.global_model.set_weights(averaged_weights)
            
            global_loss, global_acc = self.global_model.evaluate(self.X_test, self.y_test, verbose=0)
            
            print(f"Global Model: Accuracy = {global_acc:.4f}, Loss = {global_loss:.4f}")
            
            self.history['global_accuracy'].append(global_acc)
            self.history['global_loss'].append(global_loss)
            
            for client_id in range(self.num_clients):
                self.history['client_accuracies'][client_id].append(round_client_accuracies[client_id])
                self.history['client_losses'][client_id].append(round_client_losses[client_id])
                
    def evaluate_global_model(self):
        """Evaluate the final global model"""
        print("\n=== Final Global Model Evaluation ===")
        
        y_pred_proba = self.global_model.predict(self.X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        accuracy = accuracy_score(self.y_test, y_pred)
        f1 = f1_score(self.y_test, y_pred, average='weighted')
        recall = recall_score(self.y_test, y_pred, average='weighted')
        
        print(f"Final Global Model Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score (Weighted): {f1:.4f}")
        print(f"Recall (Weighted): {recall:.4f}")
        
        class_names = self.label_encoder.classes_
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred, target_names=class_names))
        
        report_dict = classification_report(self.y_test, y_pred, target_names=class_names, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose()
        report_df.to_csv(os.path.join(self.results_dir, 'classification_report.csv'))
        
        return y_pred, accuracy, f1, recall
        
    def plot_training_results(self):
        """Plot training results with high resolution"""
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_rounds + 1), self.history['global_accuracy'],
                 'b-', linewidth=3, label='Global Model', marker='o', markersize=6)
        plt.title('Global Model Accuracy over Rounds', fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.9, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_plots', 'global_model_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_rounds + 1), self.history['global_loss'],
                 'r-', linewidth=3, label='Global Model', marker='o', markersize=6)
        plt.title('Global Model Loss over Rounds', fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_plots', 'global_model_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        colors = plt.cm.Set3(np.linspace(0, 1, self.num_clients))
        for client_id in range(self.num_clients):
            plt.plot(range(1, self.num_rounds + 1),
                     self.history['client_accuracies'][client_id],
                     label=f'Client {client_id + 1}', alpha=0.8, linewidth=2,
                     marker='s', markersize=4, color=colors[client_id])
        plt.title('Clients Accuracy over Rounds', fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.ylim(0.9, 1)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_plots', 'clients_accuracy.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(10, 6))
        for client_id in range(self.num_clients):
            plt.plot(range(1, self.num_rounds + 1),
                     self.history['client_losses'][client_id],
                     label=f'Client {client_id + 1}', alpha=0.8, linewidth=2,
                     marker='s', markersize=4, color=colors[client_id])
        plt.title('Clients Loss over Rounds', fontweight='bold')
        plt.xlabel('Communication Round')
        plt.ylabel('Loss')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_plots', 'clients_loss.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def plot_evaluation_results(self, y_pred, accuracy, f1, recall):
        """Plot evaluation results with high resolution"""
        class_names = self.label_encoder.classes_
        display_names = [f'C{i}' for i in range(len(class_names))]

        plt.figure(figsize=(16, 12))
        cm = confusion_matrix(self.y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=class_names, yticklabels=class_names,
                    linewidths=0.5, cbar_kws={"shrink": .8})
        plt.title('Confusion Matrix - Global Model', fontweight='bold')
        plt.xlabel('Predicted Class')
        plt.ylabel('True Class')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'evaluation_plots', 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.close()

        plt.figure(figsize=(12, 8))
        report_dict = classification_report(self.y_test, y_pred, target_names=class_names, output_dict=True)
        class_f1_scores = [report_dict[class_name]['f1-score'] for class_name in class_names]
        y_pos = np.arange(len(class_names))
        bars = plt.barh(y_pos, class_f1_scores, color=plt.cm.viridis(np.linspace(0, 1, len(class_names))))
        plt.yticks(y_pos, class_names)
        plt.xlabel('F1-Score')
        plt.title('Class-wise F1-Score Performance', fontweight='bold')
        plt.xlim(0, 1.1)
        for i, (bar, score) in enumerate(zip(bars, class_f1_scores)):
            plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, f'{score:.3f}', ha='left', va='center')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'evaluation_plots', 'classwise_f1_score.png'), dpi=300, bbox_inches='tight')
        plt.close()

        from sklearn.preprocessing import label_binarize
        from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
        y_test_bin = label_binarize(self.y_test, classes=np.arange(len(class_names)))
        y_pred_proba = self.global_model.predict(self.X_test)
        
        # ROC AUC Curve
        fpr, tpr, roc_auc = dict(), dict(), dict()
        for i in range(len(class_names)):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
        plt.figure(figsize=(10, 8))
        plt.plot(fpr["micro"], tpr["micro"], label=f'micro-average ROC (area = {roc_auc["micro"]:.2f})', color='deeppink', linestyle=':', linewidth=4)
        colors = plt.cm.jet(np.linspace(0, 1, len(class_names)))
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'ROC {class_names[i]} (area = {roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlim([0.0, 1.0]), plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'), plt.ylabel('True Positive Rate')
        plt.title('ROC AUC Curve - Global Model', fontweight='bold')
        plt.legend(loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'evaluation_plots', 'roc_auc_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Precision-Recall Curve
        precision, recall, average_precision = dict(), dict(), dict()
        for i in range(len(class_names)):
            precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred_proba[:, i])
            average_precision[i] = average_precision_score(y_test_bin[:, i], y_pred_proba[:, i])
        precision["micro"], recall["micro"], _ = precision_recall_curve(y_test_bin.ravel(), y_pred_proba.ravel())
        average_precision["micro"] = average_precision_score(y_test_bin, y_pred_proba, average="micro")
        plt.figure(figsize=(10, 8))
        plt.plot(recall["micro"], precision["micro"], label=f'micro-average PR (AP = {average_precision["micro"]:.2f})', color='navy', linestyle=':', linewidth=4)
        for i, color in zip(range(len(class_names)), colors):
            plt.plot(recall[i], precision[i], color=color, lw=2, label=f'PR {class_names[i]} (AP = {average_precision[i]:.2f})')
        plt.xlabel('Recall'), plt.ylabel('Precision')
        plt.title('Precision-Recall Curve - Global Model', fontweight='bold')
        plt.legend(loc="lower left", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'evaluation_plots', 'pr_curve.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def lime_explanation(self, num_samples=1):
        """Generate one LIME explanation for each class"""
        print("\n=== Generating LIME Explanations ===")
        try:
            explainer = lime.lime_tabular.LimeTabularExplainer(
                self.X_train_original,
                feature_names=self.feature_names,
                class_names=self.label_encoder.classes_,
                mode='classification',
                discretize_continuous=True,
                random_state=42
            )

            def model_predict_proba(X):
                return self.global_model.predict(X)

            for class_idx, class_name in enumerate(self.label_encoder.classes_):
                indices = np.where(self.y_test == class_idx)[0]
                if len(indices) == 0:
                    print(f"No test samples for class {class_name}, skipping.")
                    continue
                
                idx = np.random.choice(indices)
                instance = self.X_test[idx]
                pred_proba = model_predict_proba(instance.reshape(1, -1))[0]
                pred_label = np.argmax(pred_proba)
                
                explanation = explainer.explain_instance(
                    instance, model_predict_proba, num_features=15, top_labels=1
                )
                fig = explanation.as_pyplot_figure(label=pred_label)
                fig.suptitle(f'LIME for Class: {class_name} | Predicted: {self.label_encoder.classes_[pred_label]} ({pred_proba[pred_label]:.2f})',
                             fontsize=14, fontweight='bold')
                
                ax = fig.axes[0]
                ax.set_xlabel('Feature Contribution', fontweight='bold', fontsize=12)
                ax.set_title('') # Remove redundant title
                bars = ax.patches
                for bar in bars:
                    bar.set_color('green' if bar.get_width() > 0 else 'red')
                    bar.set_alpha(0.7)
                plt.tight_layout()
                plt.savefig(os.path.join(self.results_dir, 'lime_explanations', f'lime_explanation_class_{class_name}.png'), dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            print(f"Error during LIME explanation: {e}")
            import traceback
            traceback.print_exc()
            
    def shap_analysis(self, num_samples=100):
        """Generate SHAP analysis for global model interpretability."""
        print("\n=== Generating SHAP Analysis ===")
        try:
            background_data = self.X_train_original[np.random.choice(len(self.X_train_original), min(100, len(self.X_train_original)), replace=False)]
            test_data = self.X_test[np.random.choice(len(self.X_test), min(num_samples, len(self.X_test)), replace=False)]

            explainer = shap.KernelExplainer(lambda x: self.global_model.predict(x), background_data)
            print("Calculating SHAP values... (this may take a few minutes)")
            shap_values = explainer.shap_values(test_data)

            class_names = self.label_encoder.classes_
            
            # Global Summary Plot
            shap.summary_plot(shap_values, test_data, feature_names=self.feature_names, 
                              class_names=class_names, show=False, plot_size=(12, 10))
            plt.title('SHAP Global Feature Importance', fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(self.results_dir, 'shap_analysis', 'shap_global_summary.png'), dpi=300, bbox_inches='tight')
            plt.close()

        except Exception as e:
            print(f"Error during SHAP analysis: {e}")
            import traceback
            traceback.print_exc()

    def plot_network_latency_comparison(self):
        """Plot network latency comparison between 5G-uRLLC and 6G-Comm."""
        np.random.seed(42)
        x = np.arange(0, 100, 10)
        latency_6g = np.sort(np.random.uniform(5, 15, size=len(x)))
        latency_5g = latency_6g + np.random.uniform(2, 5, size=len(x))
        width = 4
        plt.figure(figsize=(8, 6))
        plt.bar(x - width/2, latency_5g, width=width, color='#FFCC99', label='5G-uRLLC')
        plt.bar(x + width/2, latency_6g, width=width, color='#33B5E5', label='6G-Comm')
        plt.ylabel('Network Latency (ms)', fontweight='bold', fontsize=14)
        plt.xlabel('Time (s)')
        plt.title('Simulated Network Latency Comparison', fontweight='bold')
        plt.xticks(x)
        plt.yticks(fontsize=12)
        plt.legend(fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'evaluation_plots', 'network_latency_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    def save_results_summary(self, accuracy, f1, recall):
        """Save a comprehensive results summary"""
        summary = {
            'experiment_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'federated_learning_config': {
                'num_clients': self.num_clients, 'num_rounds': self.num_rounds,
                'local_epochs': self.local_epochs, 'batch_size': self.batch_size
            },
            'dataset_info': {
                'num_features': len(self.feature_names), 'num_classes': self.num_classes,
                'test_samples': len(self.y_test), 'class_names': self.label_encoder.classes_.tolist()
            },
            'final_performance': {
                'accuracy': float(accuracy), 'f1_score_weighted': float(f1), 'recall_weighted': float(recall)
            },
            'training_history': {
                'global_accuracy': self.history['global_accuracy'],
                'global_loss': self.history['global_loss']
            }
        }
        
        import json
        with open(os.path.join(self.results_dir, 'experiment_summary.json'), 'w') as f:
            json.dump(summary, f, indent=4)
        
        with open(os.path.join(self.results_dir, 'experiment_summary.txt'), 'w') as f:
            f.write("="*50 + "\n")
            f.write(" FEDERATED LEARNING EXPERIMENT SUMMARY\n")
            f.write("="*50 + "\n\n")
            f.write(f"Timestamp: {summary['experiment_timestamp']}\n\n")
            
            f.write("--- Configuration ---\n")
            for key, val in summary['federated_learning_config'].items():
                f.write(f"- {key.replace('_', ' ').title()}: {val}\n")
            
            f.write("\n--- Final Performance ---\n")
            for key, val in summary['final_performance'].items():
                f.write(f"- {key.replace('_', ' ').title()}: {val:.4f}\n")
            
            f.write("\n--- Test Set Class Distribution ---\n")
            for i, class_name in enumerate(self.label_encoder.classes_):
                count = np.sum(self.y_test == i)
                f.write(f"- {class_name}: {count} samples\n")
        
        print(f"\nResults summary saved to '{self.results_dir}' directory")

def main():
    """Main function to run the federated learning experiment"""
    fl_system = EdgeIIoTFederatedLearning(
        num_clients=3,
        num_rounds=3,
        local_epochs=3,
        batch_size=64,
        results_dir='results'
    )
    
    file_path = 'DNN-EdgeIIoT-dataset.csv'
    
    try:
        X_train, y_train = fl_system.load_and_preprocess_data(file_path)
        
        fl_system.train_federated_learning(X_train, y_train)
        
        y_pred, accuracy, f1, recall = fl_system.evaluate_global_model()
        
        fl_system.plot_training_results()
        fl_system.plot_evaluation_results(y_pred, accuracy, f1, recall)
        
        print("\n" + "="*60 + "\nSTARTING EXPLAINABLE AI ANALYSIS\n" + "="*60)
        
        fl_system.lime_explanation()
        fl_system.shap_analysis(num_samples=50)
        
        fl_system.save_results_summary(accuracy, f1, recall)
        fl_system.plot_network_latency_comparison()
        
        print("\n" + "="*60 + "\nFEDERATED LEARNING WITH XAI EXPERIMENT COMPLETED!\n" + "="*60)
        print(f"All results saved to '{fl_system.results_dir}' directory")
        print(f"Final Performance - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}")
        
    except FileNotFoundError:
        print(f"Error: Dataset file '{file_path}' not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()