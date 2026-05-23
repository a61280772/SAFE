"""
CelebA Dataset Experiments with Fairness Intervention Methods.
Uses CelebA dataset and applies the attribute-factorized codebook approach.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import itertools
import sys
import atexit
import os

# Import shared modules
from shared_modules import (
    EmbeddingModel, PredictionHead, EmbeddingProjection, Codebook,
    GradientReversalLayer, LAFTRAdversary, ROADPerturbation, AdversarialLearningAdversary,
    train_phase1, train_phase2, train_group_dro, train_laftr, train_road,
    train_adversarial_learning, train_hsic, train_inlp, train_rlace, compute_hsic_loss,
    evaluate_model, calculate_group_fairness_metrics, calculate_trace_scatter_matrix,
    evaluate_information_leakage, evaluate_mutual_information,
    EMBEDDING_DIM, PROJECTION_DIM,
    construct_intersectional_groups, create_group_inclusion_split, compute_fairness_metrics,
    run_group_inclusion_experiment, print_comparison_summary,
    GroupCodebook, train_phase1_group, train_phase2_group, run_ablation_roar
)

class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open("experiment_results_celeba.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        self.terminal.flush()
        self.log.flush()    

sys.stdout = Logger()
sys.stderr = sys.stdout # Capture errors too
atexit.register(sys.stdout.log.flush)
atexit.register(sys.stdout.log.close)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# ==================== Data Loading ====================

def load_celeba_data(data_path=None, use_torchvision=True, download=True):
    """
    Loads CelebA data.
    
    Args:
        data_path: Path to CelebA data directory (if using local files)
        use_torchvision: Whether to use torchvision.datasets.CelebA (default: True)
        download: Whether to download if not available (only for torchvision)
    
    Returns:
        X: Features DataFrame (using attributes as features)
        y: Target labels (Attractive)
        sensitive_attributes: Sensitive attributes DataFrame
    """
    print("Loading CelebA data...")
    
    if use_torchvision:
        try:
            from torchvision.datasets import CelebA
            from torchvision import transforms
            
            # Load CelebA dataset
            transform = transforms.Compose([
                transforms.Resize(64),
                transforms.CenterCrop(64),
                transforms.ToTensor(),
            ])
            
            celeba_train = CelebA(root=data_path if data_path else './data', split='train', 
                                  target_type='attr', download=download, transform=transform)
            celeba_test = CelebA(root=data_path if data_path else './data', split='test', 
                                 target_type='attr', download=download, transform=transform)
            
            # Get attribute names
            attr_names = celeba_train.attr_names
            
            # Convert to pandas
            def celeba_to_dataframe(dataset, split_name):
                data = []
                for idx in range(len(dataset)):
                    img, attrs = dataset[idx]
                    attrs_dict = {attr_names[i]: int(attrs[i]) for i in range(len(attr_names))}
                    data.append(attrs_dict)
                df = pd.DataFrame(data)
                print(f"Loaded {len(df)} samples from {split_name}")
                return df
            
            train_df = celeba_to_dataframe(celeba_train, 'train')
            test_df = celeba_to_dataframe(celeba_test, 'test')
            
            # Combine for now, will split later
            full_df = pd.concat([train_df, test_df], ignore_index=True)
            
        except ImportError:
            print("torchvision not available or CelebA download failed. Trying to load from CSV...")
            use_torchvision = False
    
    if not use_torchvision:
        # Load from CSV (assumes a standard CelebA attribute CSV format)
        if data_path is None:
            data_path = './data'
        
        # Try to find the attribute file
        attr_file = os.path.join(data_path, 'list_attr_celeba.txt')
        if os.path.exists(attr_file):
            # Read CelebA attribute file
            # Format: first row is number of images, second row is attribute names, rest is data
            with open(attr_file, 'r') as f:
                lines = f.readlines()
            
            attr_names = lines[1].strip().split()
            data_lines = lines[2:]
            
            data = []
            for line in data_lines:
                parts = line.strip().split()
                image_id = parts[0]
                attrs = [int(x) for x in parts[1:]]
                attrs_dict = {attr_names[i]: attrs[i] for i in range(len(attr_names))}
                attrs_dict['image_id'] = image_id
                data.append(attrs_dict)
            
            full_df = pd.DataFrame(data)
            print(f"Loaded {len(full_df)} samples from CSV")
        else:
            raise FileNotFoundError(f"Could not find CelebA attribute file at {attr_file}")
    
    # Define target and sensitive attributes
    target_attr = 'Attractive'
    sensitive_attr_cols = ['Male', 'Young', 'Pale_Skin', 'Black_Hair']
    
    # Check if required attributes exist
    required_attrs = [target_attr] + sensitive_attr_cols
    missing_attrs = [attr for attr in required_attrs if attr not in full_df.columns]
    if missing_attrs:
        raise ValueError(f"Missing required attributes in CelebA data: {missing_attrs}")
    
    # Extract features (use other attributes as features, excluding target and sensitive)
    all_attrs = full_df.columns.tolist()
    feature_cols = [col for col in all_attrs if col not in required_attrs and col != 'image_id']
    
    X = full_df[feature_cols].copy()
    y = full_df[target_attr].copy()
    sensitive_attributes = full_df[sensitive_attr_cols].copy()
    
    print(f"Features shape: {X.shape}")
    print(f"Target distribution:\n{y.value_counts()}")
    print(f"Sensitive attributes:\n{sensitive_attributes.head()}")
    
    return X, y, sensitive_attributes

def preprocess_celeba_data(X, y, sensitive_attributes):
    """
    Preprocesses CelebA data for fairness experiments.
    
    Args:
        X: Features DataFrame
        y: Target labels
        sensitive_attributes: Sensitive attributes DataFrame
    
    Returns:
        X_processed: Processed features
        y_processed: Processed labels
        sensitive_attributes_processed: Processed sensitive attributes
    """
    print("\nPreprocessing CelebA data...")
    
    # All features in CelebA are binary (-1, 1), convert to (0, 1)
    X_processed = (X + 1) // 2
    sensitive_attributes_processed = (sensitive_attributes + 1) // 2
    
    # Convert target to binary (already -1, 1 in CelebA, convert to 0, 1)
    y_processed = (y + 1) // 2
    
    print(f"Processed features shape: {X_processed.shape}")
    print(f"Target distribution:\n{y_processed.value_counts()}")
    print(f"Sensitive attributes:\n{sensitive_attributes_processed.head()}")
    
    return X_processed, y_processed, sensitive_attributes_processed

# ==================== Sensitive Attribute Definition ====================

def define_sensitive_attributes(sensitive_attributes):
    """
    Defines and processes sensitive attributes for CelebA.
    
    Sensitive attributes:
    - Male: Gender (0: Female, 1: Male)
    - Young: Age (0: Not Young, 1: Young)
    - Pale_Skin: Skin tone (0: Not Pale, 1: Pale)
    - Black_Hair: Hair color (0: Not Black, 1: Black)
    
    Returns:
        sensitive_attributes_processed: Processed sensitive attributes
        attribute_mappings: Mapping of original values to indices
    """
    print("\nDefining sensitive attributes...")
    
    # Convert binary attributes to strings for consistency with group construction
    sensitive_attributes_processed = sensitive_attributes.copy()
    for col in sensitive_attributes_processed.columns:
        sensitive_attributes_processed[col] = sensitive_attributes_processed[col].map({0: f'Not_{col}', 1: col}).astype(str)
    
    selected_attrs = ['Male', 'Young', 'Pale_Skin', 'Black_Hair']
    
    print(f"Sensitive attributes shape: {sensitive_attributes_processed.shape}")
    print(f"Unique values per attribute:")
    for attr in selected_attrs:
        if attr in sensitive_attributes_processed.columns:
            print(f"  {attr}: {sensitive_attributes_processed[attr].nunique()} unique values")
            print(f"    Values: {sensitive_attributes_processed[attr].unique()}")
    
    return sensitive_attributes_processed

# ==================== Baseline Experiments Function ====================

def run_celeba_baseline_experiments(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test,
                                    attribute_values, device, train_head=True, n_roar_iterations=30):
    """
    Runs CelebA baseline experiments with ROAR and all other methods on the full test set.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_train: Training sensitive attributes
        X_test: Test features
        y_test: Test labels
        sensitive_test: Test sensitive attributes
        attribute_values: Dictionary of attribute values for codebook
        device: Device to use for training
        train_head: Whether to train the prediction head
        n_roar_iterations: Number of iterations for ROAR (default: 30)
    """
    print("\n" + "="*60)
    print("CELEBA BASELINE EXPERIMENTS")
    print("="*60)
    
    # Initialize comparison results dictionary
    comparison_results = {}
    
    group_definition = ['Male', 'Young', 'Pale_Skin', 'Black_Hair']
    
    # Reconstruct attribute indices for training data
    _, _, attribute_indices_train, _, valid_indices = construct_intersectional_groups(
        sensitive_train, group_definition, attribute_values
    )
    
    # Filter training data to only include valid samples
    X_train_filtered = X_train.iloc[valid_indices].reset_index(drop=True)
    y_train_filtered = y_train.iloc[valid_indices].reset_index(drop=True)
    
    # Create tensors
    X_train_tensor = torch.tensor(X_train_filtered.values, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_filtered.values, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.tensor(attribute_indices_train, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    input_dim = X_train_filtered.shape[1]
    criterion = nn.BCELoss()
    
    # Map column names for evaluation function compatibility
    sensitive_test_mapped = sensitive_test.copy()
    
    # ==================== ROAR Experiment ====================
    print("\n" + "="*60)
    print("ATTRIBUTE-FACTORIZED CODEBOOK (ROAR)")
    print("="*60)
    
    embedding_model = EmbeddingModel(input_dim).to(device)
    prediction_head = PredictionHead().to(device)
    embedding_projection = EmbeddingProjection().to(device)
    
    # Create codebook with sensitive attributes
    sensitive_attr_values = [attribute_values[attr] for attr in group_definition]
    codebook = Codebook(sensitive_attr_values, PROJECTION_DIM).to(device)
    
    for t in range(n_roar_iterations):
        print(f"\n\n--- Iteration {t+1}/{n_roar_iterations} ---")
        train_embedding_model = (t == 0)  # only train embedding model in first iteration
        for param in embedding_model.parameters(): param.requires_grad = train_embedding_model
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = True
        for param in codebook.parameters(): param.requires_grad = True

        param_groups = [
            {"params": embedding_model.parameters(), "lr": 1e-4} if train_embedding_model else None,
            {"params": prediction_head.parameters(), "lr": 1e-3} if train_head else None,
            {"params": embedding_projection.parameters(), "lr": 1e-3},
            {"params": codebook.parameters(), "lr": 5e-4},
        ]
        # Filter out None values
        param_groups = [pg for pg in param_groups if pg is not None]
        optimizer = optim.Adam(param_groups)
        
        # Phase 1: Metric learning with codebook
        print("\nTraining ROAR Phase 1 ...")
        train_phase1(
            embedding_model, prediction_head, embedding_projection, codebook,
            train_loader, criterion, optimizer, n_epochs=(40 if t == 0 else 20), alpha=(100.0 if t == 0 else 5.0), inter_group_margin=0.5,
            train_embedding_model=train_embedding_model
        ) #alpha=1.0 2.0
        if train_embedding_model:
            print("\n--- ROAR Phase 1 Evaluation ---")
            metrics_roar_phase1 = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 X_test_tensor, torch.empty(0), sensitive_test_mapped,
                 return_metrics=True)
            comparison_results['ROAR (Phase 1)'] = metrics_roar_phase1
        
        print("Training ROAR Phase 2 ...")
        for param in embedding_model.parameters(): param.requires_grad = True
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = False
        for param in codebook.parameters(): param.requires_grad = False

        param_groups_iter_2 = [
            {"params": embedding_model.parameters(), "lr": 1e-4},
            {"params": prediction_head.parameters(), "lr": 1e-4} if train_head else None, #5e-4
        ]
        # Filter out None values
        param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
        optimizer_iter_2 = optim.Adam(param_groups_iter_2)
        train_phase2(
            embedding_model, prediction_head, embedding_projection, codebook,
            train_loader, criterion, optimizer_iter_2, n_epochs=(40 if t == 0 else 20), beta=(100.0 if t == 0 else 30.0), #0.2 beta=2
            use_vae_disentanglement=True, input_dim=input_dim
        )
        if t < n_roar_iterations - 1:
            print(f"\n--- Iteration {t+1} ROAR Evaluation ---")
            metrics_roar_iteration = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                        X_test_tensor, y_test_tensor, sensitive_test_mapped,
                        X_test_tensor, torch.empty(0), sensitive_test_mapped,
                        return_metrics=True)
            comparison_results[f'ROAR (iteration {t+1})'] = metrics_roar_iteration
    
    print("\n--- Final ROAR Evaluation ---")
    metrics_roar_final = evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                    X_test_tensor, y_test_tensor, sensitive_test_mapped,
                    X_test_tensor, torch.empty(0), sensitive_test_mapped,
                    return_metrics=True)
    comparison_results['ROAR (Final)'] = metrics_roar_final
    
    # ==================== Baseline Experiments ====================    
    # --- ERM Baseline (Empirical Risk Minimization) ---
    print("\n" + "="*60)
    print("ERM BASELINE (Empirical Risk Minimization)")
    print("="*60)
    
    erm_embedding_model = EmbeddingModel(input_dim).to(device)
    erm_prediction_head = PredictionHead().to(device)
    
    erm_params = list(erm_embedding_model.parameters()) + list(erm_prediction_head.parameters() if train_head else [])
    erm_optimizer = optim.Adam(erm_params, lr=0.001)
    
    # Simple training loop for ERM baseline
    for epoch in range(20):
        erm_embedding_model.train()
        erm_prediction_head.train() if train_head else erm_prediction_head.eval()
        running_loss = 0.0
        
        for inputs, labels, _ in train_loader:
            erm_optimizer.zero_grad()
            embeddings = erm_embedding_model(inputs)
            outputs = erm_prediction_head(embeddings)
            loss = criterion(outputs, labels)
            loss.backward()
            erm_optimizer.step()
            running_loss += loss.item()
        
        print(f"Epoch {epoch+1}/20, ERM Loss: {running_loss/len(train_loader):.9f}")
    
    # Evaluate ERM baseline
    print("\n--- ERM Baseline Evaluation ---")
    metrics_erm = evaluate_model(erm_embedding_model, erm_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['ERM'] = metrics_erm
    
    # --- Group DRO Baseline ---
    print("\n" + "="*60)
    print("GROUP DRO BASELINE")
    print("="*60)
    
    group_dro_embedding_model = EmbeddingModel(input_dim).to(device)
    group_dro_prediction_head = PredictionHead().to(device)
    
    group_dro_optimizer = optim.Adam(
        list(group_dro_embedding_model.parameters()) +
        list(group_dro_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_group_dro(
        group_dro_embedding_model, group_dro_prediction_head, train_loader,
        criterion, group_dro_optimizer, n_epochs=20, eta=0.01, train_embedding_model=True
    )
    
    print("\n--- Group DRO Baseline Evaluation ---")
    metrics_group_dro = evaluate_model(group_dro_embedding_model, group_dro_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['Group DRO'] = metrics_group_dro
    
    # --- LAFTR Baseline ---
    print("\n" + "="*60)
    print("LAFTR BASELINE")
    print("="*60)
    
    laftr_embedding_model = EmbeddingModel(input_dim).to(device)
    laftr_task_head = PredictionHead().to(device)
    
    # Create adversaries for each sensitive attribute
    num_male_classes = len(attribute_values['Male'])
    num_young_classes = len(attribute_values['Young'])
    
    laftr_male_adversary = LAFTRAdversary(EMBEDDING_DIM, num_male_classes).to(device)
    laftr_young_adversary = LAFTRAdversary(EMBEDDING_DIM, num_young_classes).to(device)
    laftr_adversaries = [laftr_male_adversary, laftr_young_adversary]
    
    laftr_optimizer = optim.Adam(
        list(laftr_embedding_model.parameters()) +
        list(laftr_task_head.parameters() if train_head else []) +
        list(laftr_male_adversary.parameters()) +
        list(laftr_young_adversary.parameters()),
        lr=0.001
    )
    
    train_laftr(
        laftr_embedding_model, laftr_task_head, laftr_adversaries, train_loader,
        criterion, [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()], laftr_optimizer, n_epochs=20,
        lambda_adv=0.1, lambda_task=1.0, train_embedding_model=True
    )
    
    print("\n--- LAFTR Baseline Evaluation ---")
    metrics_laftr = evaluate_model(laftr_embedding_model, laftr_task_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['LAFTR'] = metrics_laftr
    
    # --- ROAD Baseline ---
    print("\n" + "="*60)
    print("ROAD BASELINE")
    print("="*60)
    
    road_embedding_model = EmbeddingModel(input_dim).to(device)
    road_prediction_head = PredictionHead().to(device)
    road_perturbation_net = ROADPerturbation(input_dim, perturbation_dim=32, epsilon=0.1).to(device)
    
    road_optimizer = optim.Adam(
        list(road_embedding_model.parameters()) +
        list(road_prediction_head.parameters() if train_head else []) +
        list(road_perturbation_net.parameters()),
        lr=0.001
    )
    
    train_road(
        road_embedding_model, road_prediction_head, road_perturbation_net, train_loader,
        criterion, road_optimizer, n_epochs=20, alpha=0.5, train_embedding_model=True
    )
    
    print("\n--- ROAD Baseline Evaluation ---")
    metrics_road = evaluate_model(road_embedding_model, road_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['ROAD'] = metrics_road
    
    # --- AdversarialLearning Baseline ---
    print("\n" + "="*60)
    print("ADVERSARIAL LEARNING BASELINE")
    print("="*60)
    
    adv_learning_embedding_model = EmbeddingModel(input_dim).to(device)
    adv_learning_task_head = PredictionHead().to(device)
    
    # Single adversary for all sensitive attributes
    adv_learning_adversary = AdversarialLearningAdversary(
        EMBEDDING_DIM,
        num_attributes=2,  # Male, Young
        num_classes_per_attr=[num_male_classes, num_young_classes]
    ).to(device)
    
    adv_learning_optimizer = optim.Adam(
        list(adv_learning_embedding_model.parameters()) +
        list(adv_learning_task_head.parameters() if train_head else []) +
        list(adv_learning_adversary.parameters()),
        lr=0.001
    )
    
    train_adversarial_learning(
        adv_learning_embedding_model, adv_learning_task_head, adv_learning_adversary,
        train_loader, criterion, nn.CrossEntropyLoss(), adv_learning_optimizer,
        n_epochs=20, alpha=1.0, train_embedding_model=True
    )
    
    print("\n--- AdversarialLearning Baseline Evaluation ---")
    metrics_adv_learning = evaluate_model(adv_learning_embedding_model, adv_learning_task_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['AdversarialLearning'] = metrics_adv_learning
    
    # --- HSIC Baseline ---
    print("\n" + "="*60)
    print("HSIC REGULARIZATION BASELINE")
    print("="*60)
    
    hsic_embedding_model = EmbeddingModel(input_dim).to(device)
    hsic_prediction_head = PredictionHead().to(device)
    
    hsic_optimizer = optim.Adam(
        list(hsic_embedding_model.parameters()) +
        list(hsic_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_hsic(
        hsic_embedding_model, hsic_prediction_head, train_loader,
        criterion, hsic_optimizer, n_epochs=20, lambda_hsic=0.1, train_embedding_model=True
    )
    
    print("\n--- HSIC Baseline Evaluation ---")
    metrics_hsic = evaluate_model(hsic_embedding_model, hsic_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['HSIC'] = metrics_hsic
    
    # ==================== INLP Baseline ====================
    print("\n--- INLP Baseline ---")
    inlp_embedding_model = EmbeddingModel(input_dim=input_dim).to(device)
    inlp_prediction_head = PredictionHead().to(device)
    inlp_optimizer = optim.Adam(
        list(inlp_embedding_model.parameters()) + 
        list(inlp_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_inlp(
        inlp_embedding_model, inlp_prediction_head, train_loader,
        criterion, inlp_optimizer, n_epochs=20, train_embedding_model=True, num_iterations=10
    )
    
    # Evaluate INLP baseline
    print("\n--- INLP Baseline Evaluation ---")
    metrics_inlp = evaluate_model(inlp_embedding_model, inlp_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['INLP'] = metrics_inlp
    
    # ==================== RLACE Baseline ====================
    print("\n--- RLACE Baseline ---")
    rlace_embedding_model = EmbeddingModel(input_dim=input_dim).to(device)
    rlace_prediction_head = PredictionHead().to(device)
    rlace_optimizer = optim.Adam(
        list(rlace_embedding_model.parameters()) + 
        list(rlace_prediction_head.parameters() if train_head else []),
        lr=0.001
    )
    
    train_rlace(
        rlace_embedding_model, rlace_prediction_head, train_loader,
        criterion, rlace_optimizer, n_epochs=20, train_embedding_model=True, num_iterations=10, alpha=100.0
    )
    
    # Evaluate RLACE baseline
    print("\n--- RLACE Baseline Evaluation ---")
    metrics_rlace = evaluate_model(rlace_embedding_model, rlace_prediction_head, None, None,
                 X_test_tensor, y_test_tensor, sensitive_test_mapped,
                 torch.empty(0), torch.empty(0), pd.DataFrame(),
                 return_metrics=True)
    comparison_results['RLACE'] = metrics_rlace
    
    # Print the comparison summary table
    print_comparison_summary(comparison_results)
    
    # ==================== Ablation Study: Group-Level Codebook ====================
    print("\n" + "="*60)
    print("ABLATION STUDY: Group-Level Codebook (No Attribute Factorization)")
    print("="*60)
    
    # Reconstruct group keys for ablation study
    group_keys, _, _, _, _ = construct_intersectional_groups(
        sensitive_train, group_definition, attribute_values
    )
    
    # Run ablation ROAR with group codebook
    ablation_metrics = run_ablation_roar(
        X_train, y_train, sensitive_train,
        X_test, y_test, sensitive_test,
        group_keys, attribute_values, device, group_definition,
        train_head=train_head, n_roar_iterations=10, n_epochs_phase1=40, n_epochs_phase2=20,
        alpha=100.0, beta=30.0, inter_group_margin=0.5
    )
    
    comparison_results['Ablation (Group Codebook)'] = ablation_metrics
    
    # Print updated comparison summary with ablation
    print("\n" + "="*60)
    print("COMPARISON SUMMARY (WITH ABLATION)")
    print("="*60)
    print_comparison_summary(comparison_results)
    
    print("\n" + "="*60)
    print("CELEBA BASELINE EXPERIMENTS COMPLETE")
    print("="*60)

# ==================== Main Function ====================

def main():
    # Load CelebA data
    # Note: torchvision download may fail due to Google Drive limits. 
    # Set use_torchvision=True only if you have the data already downloaded locally.
    # Otherwise, ensure you have the CelebA attribute file (list_attr_celeba.txt) in the data_path.
    X, y, sensitive_attributes = load_celeba_data(data_path='./data', use_torchvision=False, download=False)
    
    # Preprocess data
    X_processed, y_processed, sensitive_attributes_raw = preprocess_celeba_data(X, y, sensitive_attributes)
    
    # Define sensitive attributes
    sensitive_attributes_processed = define_sensitive_attributes(sensitive_attributes_raw)
    
    # Split data into train/test
    X_train, X_test, y_train, y_test, sensitive_train, sensitive_test = train_test_split(
        X_processed, y_processed, sensitive_attributes_processed, test_size=0.2, random_state=42
    )
    
    # Construct intersectional groups
    group_definition = ['Male', 'Young', 'Pale_Skin', 'Black_Hair']
    group_keys, group_to_idx, attribute_indices, attribute_values, valid_indices = construct_intersectional_groups(
        sensitive_train, group_definition
    )
    
    # Filter training data using valid indices
    X_train = X_train.iloc[valid_indices].reset_index(drop=True)
    y_train = y_train.iloc[valid_indices].reset_index(drop=True)
    sensitive_train = sensitive_train.iloc[valid_indices].reset_index(drop=True)
    
    # Filter out tiny groups to reduce noise
    min_group_size = 50  # Minimum group size threshold (configurable)
    sensitive_train['group_key'] = sensitive_train[group_definition].astype(str).agg('_'.join, axis=1)
    group_sizes = sensitive_train['group_key'].value_counts()
    valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
    
    initial_count = len(sensitive_train)
    valid_group_mask = sensitive_train['group_key'].isin(valid_groups)
    X_train = X_train[valid_group_mask].reset_index(drop=True)
    y_train = y_train[valid_group_mask].reset_index(drop=True)
    sensitive_train = sensitive_train[valid_group_mask].reset_index(drop=True)
    filtered_count = len(sensitive_train)
    if initial_count != filtered_count:
        print(f"Filtered out {initial_count - filtered_count} training samples from groups with size < {min_group_size}")
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Input dimension: {X_train.shape[1]}")
    print(f"Number of intersectional groups: {len(group_keys)}")
    
    # ==================== Baseline Experiments ====================
    # Run baseline experiments on full test set (similar to acs_income_experiments.py lines 1028-1279)
    # Set run_baseline_experiments to False to skip this and only run group inclusion experiment
    run_baseline_experiments = True  # Set to True to run baseline experiments
    n_roar_iterations_baseline = 10  #30. Number of ROAR iterations for baseline experiments 5
    
    if run_baseline_experiments:
        run_celeba_baseline_experiments(
            X_train, y_train, sensitive_train,
            X_test, y_test, sensitive_test,
            attribute_values, device, train_head=True, n_roar_iterations=n_roar_iterations_baseline
        )
    
    # ==================== Group Inclusion Experiment ====================
    # Configure target group and inclusion percentages
    # Example target group: Male_Young_Pale_Skin_Black_Hair
    target_group = 'Not_Male_Not_Young_Not_Pale_Skin_Black_Hair' #Not_Male_Not_Young_Not_Pale_Skin_Black_Hair
    inclusion_percentages = [0.0] #[0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    n_roar_iterations = 5  # Number of ROAR iterations for group inclusion experiment 5
    group_definition = ['Male', 'Young', 'Pale_Skin', 'Black_Hair']
    
    # Run experiment using shared function
    results, fairness_metrics = run_group_inclusion_experiment(
        X_train, y_train, sensitive_train,
        X_test, y_test, sensitive_test,
        target_group, inclusion_percentages,
        attribute_values, device, group_definition, train_head=True, n_roar_iterations=n_roar_iterations,
        use_group_inclusion_split=False
    )
    
    print("\n" + "="*60)
    print("CELEBA EXPERIMENTS COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()
