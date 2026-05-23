"""
Shared modules for fairness intervention experiments.
Contains model classes, training functions, evaluation functions, and baseline-specific components.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
import itertools

# Constants
EMBEDDING_DIM = 128
PROJECTION_DIM = 64 #64
MIN_GROUP_SIZE = 50 #50

# ==================== Model Classes ====================

class EmbeddingModel(nn.Module):
    """An MLP that creates a 128-dimensional embedding from the input features."""
    def __init__(self, input_dim):
        super(EmbeddingModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, EMBEDDING_DIM),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM) # Output embedding of size 128
        )
        self.inlp_projection = None
        self.rlace_projection = None

    def forward(self, x):
        embedding = self.layers(x)
        embedding = F.normalize(embedding, p=2, dim=1)
        
        # Apply INLP projection if available
        if self.inlp_projection is not None:
            embedding = embedding @ self.inlp_projection.T
        
        # Apply RLACE projection if available
        if self.rlace_projection is not None:
            embedding = embedding @ self.rlace_projection.T
        
        return embedding

class SensitiveAttributeClassifier(nn.Module):
    """A simple MLP to predict a sensitive attribute from embeddings."""
    def __init__(self, embedding_dim, num_classes):
        super(SensitiveAttributeClassifier, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.layers(x)

class PredictionHead(nn.Module):
    """An MLP that takes a 128-dimensional embedding and predicts income."""
    def __init__(self, embedding_dim=EMBEDDING_DIM):
        super(PredictionHead, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.layers(x)

class EmbeddingProjection(nn.Module):
    """An MLP that projects the 128D embedding to a 64D embedding."""
    def __init__(self, input_dim=EMBEDDING_DIM, output_dim=PROJECTION_DIM):
        super(EmbeddingProjection, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        projected_embedding = self.layers(x)
        return F.normalize(projected_embedding, p=2, dim=1)

class Codebook(nn.Module):
    """
    An attribute-factorized codebook that learns separate embeddings for each 
    sensitive attribute value and combines them through summation to represent
    intersectional groups. All exposed embeddings are L2-normalized.
    """
    def __init__(self, sensitive_attributes, embedding_dim):
        """
        Initializes the Codebook.
        Args:
            sensitive_attributes (list): A list of lists, where each inner list
                                       contains discrete values for a sensitive attribute.
                                       Example: [['Male', 'Female'], ['White', 'Black']]
            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        super(Codebook, self).__init__()
        self.sensitive_attributes = sensitive_attributes
        self.num_attributes = len(sensitive_attributes)
        self.embedding_dim = embedding_dim
        
        # Map discrete values to integer indices for nn.Embedding
        self.attribute_to_idx = {}
        for attr_idx, values in enumerate(sensitive_attributes):
            self.attribute_to_idx[attr_idx] = {value: i for i, value in enumerate(values)}
        
        # Create embedding layers, one for each sensitive attribute
        self.attribute_embeddings = nn.ModuleList()
        for attr_idx, values in enumerate(sensitive_attributes):
            embedding_layer = nn.Embedding(len(values), embedding_dim)
            nn.init.xavier_uniform_(embedding_layer.weight)
            self.attribute_embeddings.append(embedding_layer)
        
        # Store the original group keys for backward compatibility
        self._generate_all_group_keys()

    def _generate_all_group_keys(self):
        """Generate all possible group combinations for backward compatibility."""
        import itertools
        self.group_keys = []
        self.combination_to_idx = {}
        
        # Generate all combinations of attribute values
        for combination in itertools.product(*self.sensitive_attributes):
            group_key = '_'.join(combination)
            self.group_keys.append(group_key)
            self.combination_to_idx[combination] = len(self.group_keys) - 1

    def forward(self, attribute_indices):
        """
        Looks up and returns L2-normalized embeddings for the given attribute indices.
        Args:
            attribute_indices (Tensor): Shape (batch_size, num_attributes)
                                       Each column contains integer indices for that attribute.
        Returns:
            Tensor: A batch of L2-normalized combined embeddings.
        """
        batch_size = attribute_indices.shape[0]
        combined_embeddings = torch.zeros(batch_size, self.embedding_dim, 
                                         device=attribute_indices.device)
        
        # Sum embeddings from all attributes
        for attr_idx in range(self.num_attributes):
            attr_indices = attribute_indices[:, attr_idx]
            attr_embeddings = self.attribute_embeddings[attr_idx](attr_indices)
            combined_embeddings += attr_embeddings
        
        return F.normalize(combined_embeddings, p=2, dim=1)
    
    def get_all_embeddings(self):
        """
        Returns all L2-normalized embedding vectors for all possible group combinations.
        This method maintains backward compatibility with the original interface.
        """
        import itertools
        all_embeddings = []
        
        # Generate embeddings for all combinations
        for combination in itertools.product(*self.sensitive_attributes):
            # Convert combination to indices
            indices = torch.tensor([[self.attribute_to_idx[attr_idx][value] 
                                    for attr_idx, value in enumerate(combination)]])
            
            # Get combined embedding
            embedding = self.forward(indices)
            all_embeddings.append(embedding.squeeze(0))
        
        return torch.stack(all_embeddings, dim=0)

    def get_embedding_for_group(self, group_key):
        """
        Returns the L2-normalized embedding for a specific group key.
        Maintains backward compatibility with the original interface.
        """
        # Parse group key to get attribute values
        attributes = group_key.split('_')
        if len(attributes) != self.num_attributes:
            raise ValueError(f"Group key '{group_key}' does not match expected format")
        
        # Convert to indices
        indices = torch.tensor([[self.attribute_to_idx[attr_idx][value] 
                                for attr_idx, value in enumerate(attributes)]])
        
        return self.forward(indices).squeeze(0)

    def get_attribute_embedding(self, attr_idx, value):
        """
        Returns the L2-normalized embedding for a specific attribute value.
        This is a new method that provides access to individual attribute embeddings.
        """
        value_idx = self.attribute_to_idx[attr_idx][value]
        embedding = self.attribute_embeddings[attr_idx](torch.tensor(value_idx))
        return F.normalize(embedding, p=2, dim=0)

    def find_closest_embedding(self, input_embeddings):
        """
        Finds the closest codebook embedding for each embedding in the input batch.
        Args:
            input_embeddings (Tensor): A batch of L2-normalized embeddings.
        Returns:
            Tensor: The closest L2-normalized codebook vectors for each input embedding.
        """
        # Get L2-normalized codebook embeddings
        codebook_norm = self.get_all_embeddings()
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cos_sim = torch.matmul(input_embeddings, codebook_norm.t())
        
        # Find the index of the most similar codebook vector for each input
        closest_indices = torch.argmax(cos_sim, dim=1)
        
        # Retrieve the normalized closest embeddings using the indices
        return self.get_all_embeddings()[closest_indices]

    def find_closest_embedding_indices(self, input_embeddings):
        """
        Finds the index of the closest codebook embedding for each embedding in the input batch.
        Args:
            input_embeddings (Tensor): A batch of L2-normalized embeddings.
        Returns:
            Tensor: The indices of the closest codebook vectors for each input embedding.
        """
        # Get L2-normalized codebook embeddings
        codebook_norm = self.get_all_embeddings()
        
        # Calculate cosine similarity (dot product of normalized vectors)
        cos_sim = torch.matmul(input_embeddings, codebook_norm.t())
        
        # Find the index of the most similar codebook vector for each input
        return torch.argmax(cos_sim, dim=1)

# ==================== Baseline-Specific Classes ====================

class GradientReversalLayer(torch.autograd.Function):
    """Gradient Reversal Layer for adversarial training."""
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.clone()
    
    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None

class LAFTRAdversary(nn.Module):
    """Adversary head for LAFTR to predict sensitive attributes."""
    def __init__(self, embedding_dim, num_classes):
        super(LAFTRAdversary, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x, lambda_):
        x = GradientReversalLayer.apply(x, lambda_)
        return self.layers(x)

class ROADPerturbation(nn.Module):
    """Generates adversarial perturbations sensitive to sensitive attributes for ROAD."""
    def __init__(self, input_dim, perturbation_dim=32, epsilon=0.1):
        super(ROADPerturbation, self).__init__()
        self.perturbation_net = nn.Sequential(
            nn.Linear(input_dim, perturbation_dim),
            nn.ReLU(),
            nn.Linear(perturbation_dim, input_dim)
        )
        self.epsilon = epsilon
    
    def forward(self, x):
        # Generate perturbations
        perturbations = self.perturbation_net(x)
        # Clip to stay within epsilon budget
        perturbations = torch.clamp(perturbations, -self.epsilon, self.epsilon)
        return x + perturbations

class AdversarialLearningAdversary(nn.Module):
    """Adversary head for AdversarialLearning to predict sensitive attributes."""
    def __init__(self, embedding_dim, num_attributes, num_classes_per_attr):
        super(AdversarialLearningAdversary, self).__init__()
        # Single adversary for all attributes (different from LAFTR)
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, 64),
            nn.ReLU(),
            nn.Linear(64, sum(num_classes_per_attr))  # Total classes across all attributes
        )
        self.num_attributes = num_attributes
        self.num_classes_per_attr = num_classes_per_attr
        self.attr_offsets = [0]
        for i in range(num_attributes - 1):
            self.attr_offsets.append(self.attr_offsets[-1] + num_classes_per_attr[i])
    
    def forward(self, x, lambda_):
        x = GradientReversalLayer.apply(x, lambda_)
        logits = self.layers(x)
        # Split logits for each attribute
        attr_logits = []
        for i in range(self.num_attributes):
            start = self.attr_offsets[i]
            end = start + self.num_classes_per_attr[i]
            attr_logits.append(logits[:, start:end])
        return attr_logits

# ==================== Training Functions ====================

def train_phase1(embedding_model, prediction_head, embedding_projection, codebook, 
                 train_loader, criterion, optimizer, n_epochs, alpha, inter_group_margin,
                 train_embedding_model=True, use_dynamic_margin=True):
    """
    Phase 1 training: Metric learning with attribute-factorized codebook.
    Includes within-attribute distance regularization and cross-attribute orthogonality.
    
    Args:
        use_dynamic_margin: If True, gradually increase margin during training
    """
    print(f"\n--- Starting Phase 1: Training for Metric Learning ---")
    print(f"Starting training for {n_epochs} epochs...")
    if use_dynamic_margin:
        print(f"Using dynamic margin scheduling (0.1 -> {inter_group_margin})")
    
    # Early stopping variables
    loss_history = []
    patience_counter = 0
    patience = 50 #3 10
    min_improvement_fraction = 0.0002
    ablation_no_task = False #ablation for no prediction head training in phase 1
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval() # Keep dropout off if not training
        
        if ablation_no_task:
            prediction_head.eval()
        else:
            prediction_head.train() 
        embedding_projection.train()
        codebook.train()
        
        # Dynamic margin scheduling: gradually increase from 0 to inter_group_margin
        if use_dynamic_margin:
            current_margin = 0.1 + (inter_group_margin - 0.1) * ((epoch+1) / n_epochs)
        else:
            current_margin = inter_group_margin

        running_loss = 0.0
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            # --- Main Forward Pass ---
            embeddings = embedding_model(inputs)
            projected_embeddings = embedding_projection(embeddings)
            outputs = prediction_head(embeddings)
            
            # 1. Main prediction loss (on all samples)
            prediction_loss = criterion(outputs, labels)

            # --- Metric Learning Loss (on all samples with attribute indices) ---
            # All samples in the new DataLoader have valid attribute indices
            target_centroids = codebook(attribute_indices)
            pull_loss = F.mse_loss(projected_embeddings, target_centroids)
            # New factorized push loss with within-attribute distance and cross-attribute orthogonality
            within_attr_loss = 0.0
            cross_attr_loss = 0.0
            
            # Component 1: Within-attribute distance regularization
            for attr_idx in range(codebook.num_attributes):
                # Get all embeddings for this attribute
                attr_embeddings = codebook.attribute_embeddings[attr_idx].weight
                # Calculate pairwise distances within this attribute
                if len(attr_embeddings) > 1:
                    pairwise_dist = torch.pdist(attr_embeddings, p=2)
                    within_attr_loss += torch.relu(current_margin - pairwise_dist).mean()
            
            # Component 2: Cross-attribute orthogonality
            for i in range(codebook.num_attributes):
                for j in range(i + 1, codebook.num_attributes):
                    # Get embeddings for attribute i and j
                    emb_i = codebook.attribute_embeddings[i].weight  # Shape: [num_values_i, embedding_dim]
                    emb_j = codebook.attribute_embeddings[j].weight  # Shape: [num_values_j, embedding_dim]
                    
                    # Calculate all pairwise dot products between attributes i and j
                    # Result shape: [num_values_i, num_values_j]
                    dot_products = torch.matmul(emb_i, emb_j.t())
                    
                    # Enforce orthogonality (dot products should be 0)
                    cross_attr_loss += torch.abs(dot_products).mean()
            
            push_loss = within_attr_loss + cross_attr_loss
            #push_loss = cross_attr_loss #ablation of intra-attribute margin regularization
            #push_loss = within_attr_loss #ablation of inter-attribute orthogonality regularization
            
            # Weighted combination of pull and push losses
            # Use separate coefficients to balance different scales
            pull_weight = 1.0    # Weight for alignment loss
            push_weight = 0.1    # Weight for regularization loss (smaller due to different scale)
            
            iso_loss = pull_weight * pull_loss + push_weight * push_loss
            scale = 100
            iso_loss = scale * iso_loss
            
            # --- Combine Losses and Backpropagate ---
            if ablation_no_task:
                total_loss = alpha * iso_loss
            else:
                total_loss = prediction_loss + (alpha * iso_loss)
            #print(prediction_loss.item(), iso_loss.item())
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        
        # Early stopping logic - only check starting from epoch 3
        loss_history.append(epoch_loss)
        if epoch >= 3:
            # Calculate moving average of last 3 epochs
            avg_last_three = sum(loss_history[-3:]) / 3
            loss_decrease = avg_last_three - epoch_loss
            min_required_decrease = avg_last_three * min_improvement_fraction
            if loss_decrease < min_required_decrease:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}: loss decrease below threshold for {patience} consecutive epochs")
                    break
            else:
                patience_counter = 0

def train_phase2(embedding_model, prediction_head, embedding_projection, codebook,
                 train_loader, criterion, optimizer, *, n_epochs, beta, 
                 use_vae_disentanglement=True, input_dim=None):
    """
    Phase 2 training: Disentanglement training on desensitized embeddings.
    
    Args:
        use_vae_disentanglement: If True, adds VAE-style reconstruction loss for enhanced disentanglement
        input_dim: Input dimension for reconstruction decoder (required if use_vae_disentanglement=True)
    """
    print(f"\n--- Starting Phase 2 Training ---")
    print(f"Training for {n_epochs} epochs...")
    if use_vae_disentanglement:
        print(f"Using VAE-style disentanglement with reconstruction loss")
    
    # Early stopping variables
    loss_history = []
    patience_counter = 0
    patience = 10 #3
    min_improvement_fraction = 0.0002
    ablation_no_task = False #ablation for no prediction head training in phase 2
    
    # Freeze the embedding_projection and codebook
    for param in embedding_projection.parameters():
        param.requires_grad = False
    for param in codebook.parameters():
        param.requires_grad = False
    
    # Create reconstruction decoder if using VAE-style disentanglement
    if use_vae_disentanglement:
        if input_dim is None:
            raise ValueError("input_dim must be provided when use_vae_disentanglement=True")
        
        # Get device from model parameters
        device = next(embedding_model.parameters()).device
        
        # Simple decoder: projection_dim -> hidden -> input_dim
        decoder_hidden_dim = 256
        decoder = nn.Sequential(
            nn.Linear(PROJECTION_DIM, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, input_dim)
        ).to(device)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
        reconstruction_criterion = nn.MSELoss()
    else:
        decoder = None
        decoder_optimizer = None
        reconstruction_criterion = None
    
    for epoch in range(n_epochs):
        embedding_model.train()
        if ablation_no_task:
            prediction_head.eval()
        else:
            prediction_head.train()
        
        embedding_projection.eval()
        codebook.eval()
        #embedding_projection.train() #ABLATION
        #codebook.train() #ABLATION
        if decoder is not None:
            decoder.train()
        
        running_loss_phase2 = 0.0
        running_pred_loss = 0.0
        running_disent_loss = 0.0
        running_recon_loss = 0.0
        
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            if decoder_optimizer is not None:
                decoder_optimizer.zero_grad()
            
            # --- Forward Pass ---
            embeddings = embedding_model(inputs)
            projected_embeddings = embedding_projection(embeddings)
            outputs = prediction_head(embeddings)
            
            # --- Loss Calculation ---
            prediction_loss = criterion(outputs, labels)
            
            # Calculate mean code efficiently using attribute factorization
            mean_code = torch.zeros(codebook.embedding_dim, 
                                  device=projected_embeddings.device)
            
            for attr_idx in range(codebook.num_attributes):
                attr_embeddings = codebook.attribute_embeddings[attr_idx].weight
                attr_mean = attr_embeddings.mean(dim=0)
                mean_code += attr_mean
            
            mean_code = torch.nn.functional.normalize(mean_code, p=2, dim=0)
            
            # Disentanglement loss: distance between projected embeddings and mean code
            distances_to_mean = torch.cdist(projected_embeddings, mean_code.unsqueeze(0), p=2)
            disentanglement_loss = distances_to_mean.mean()
            
            # VAE-style reconstruction loss
            if use_vae_disentanglement and decoder is not None:
                reconstructed = decoder(projected_embeddings)
                reconstruction_loss = reconstruction_criterion(reconstructed, inputs)
            else:
                reconstruction_loss = torch.tensor(0.0, device=embeddings.device)
            
            # --- Combine Losses and Backpropagate ---
            #if epoch < 5 and running_loss_phase2 == 0:
            #    print(f"Prediction loss: {prediction_loss.item():.4f}, Disentanglement loss: {disentanglement_loss.item():.4f}, Reconstruction loss: {reconstruction_loss.item():.4f}")
            
            # Total loss with VAE-style components
            if use_vae_disentanglement:
                # Beta controls disentanglement strength, gamma controls reconstruction strength
                gamma = 0.1  # Weight for reconstruction loss
                if ablation_no_task:
                    total_loss_phase2 = (beta * disentanglement_loss) + (gamma * reconstruction_loss)
                else:
                    total_loss_phase2 = prediction_loss + (beta * disentanglement_loss) + (gamma * reconstruction_loss)
            else:
                if ablation_no_task:
                    total_loss_phase2 = beta * disentanglement_loss
                else:
                    total_loss_phase2 = prediction_loss + (beta * disentanglement_loss)
            
            total_loss_phase2.backward()
            optimizer.step()
            if decoder_optimizer is not None:
                decoder_optimizer.step()
            
            running_loss_phase2 += total_loss_phase2.item()
            running_pred_loss += prediction_loss.item()
            running_disent_loss += disentanglement_loss.item()
            running_recon_loss += reconstruction_loss.item()
        
        epoch_loss_phase2 = running_loss_phase2 / len(train_loader)
        epoch_pred_loss = running_pred_loss / len(train_loader)
        epoch_disent_loss = running_disent_loss / len(train_loader)
        epoch_recon_loss = running_recon_loss / len(train_loader)
        
        if use_vae_disentanglement:
            print(f"Epoch {epoch+1}/{n_epochs}, Total Loss: {epoch_loss_phase2:.4f}, Pred: {epoch_pred_loss:.4f}, Disent: {epoch_disent_loss:.4f}, Recon: {epoch_recon_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs}, Total Loss: {epoch_loss_phase2:.4f}, Pred: {epoch_pred_loss:.4f}, Disent: {epoch_disent_loss:.4f}")
        
        # Early stopping logic - only check starting from epoch 3
        loss_history.append(epoch_loss_phase2)
        if epoch >= 3:
            # Calculate moving average of last 3 epochs
            avg_last_three = sum(loss_history[-3:]) / 3
            loss_decrease = avg_last_three - epoch_loss_phase2
            min_required_decrease = avg_last_three * min_improvement_fraction
            if loss_decrease < min_required_decrease:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}: loss decrease below threshold for {patience} consecutive epochs")
                    break
            else:
                patience_counter = 0

    print(f"--- Phase 2 Training Complete ---")

# ==================== Ablation Study: Group-Level Codebook ====================

class GroupCodebook(nn.Module):
    """
    A group-level codebook that learns one embedding per intersectional group.
    This is an ablation study alternative to the attribute-factorized codebook.
    All exposed embeddings are L2-normalized.
    """
    def __init__(self, group_keys, embedding_dim):
        """
        Initializes the GroupCodebook.
        Args:
            group_keys (list): A list of group keys (e.g., ['White_Male_1', 'Black_Female_2'])
            embedding_dim (int): The dimensionality of the embedding vectors.
        """
        super(GroupCodebook, self).__init__()
        self.group_keys = group_keys
        self.num_groups = len(group_keys)
        self.embedding_dim = embedding_dim
        
        # Create a single embedding layer for all groups
        self.group_embeddings = nn.Embedding(self.num_groups, embedding_dim)
        nn.init.xavier_uniform_(self.group_embeddings.weight)
        
        # Map group keys to indices
        self.group_to_idx = {key: idx for idx, key in enumerate(group_keys)}
    
    def forward(self, group_indices):
        """
        Looks up and returns L2-normalized embeddings for the given group indices.
        Args:
            group_indices (Tensor): Shape (batch_size,) containing integer indices for groups.
        Returns:
            Tensor: A batch of L2-normalized group embeddings.
        """
        group_embeddings = self.group_embeddings(group_indices)
        return F.normalize(group_embeddings, p=2, dim=1)
    
    def get_all_embeddings(self):
        """
        Returns all L2-normalized embedding vectors for all groups.
        """
        all_embeddings = self.group_embeddings.weight
        return F.normalize(all_embeddings, p=2, dim=1)
    
    def get_embedding_for_group(self, group_key):
        """
        Returns the L2-normalized embedding for a specific group key.
        """
        if group_key not in self.group_to_idx:
            raise ValueError(f"Group key '{group_key}' not found in codebook")
        
        idx = self.group_to_idx[group_key]
        embedding = self.group_embeddings.weight[idx]
        return F.normalize(embedding.unsqueeze(0), p=2, dim=1).squeeze(0)

def train_phase1_group(embedding_model, prediction_head, embedding_projection, group_codebook, 
                       train_loader, criterion, optimizer, n_epochs, alpha, inter_group_margin,
                       train_embedding_model=True, use_dynamic_margin=True):
    """
    Phase 1 training for ablation: Metric learning with group-level codebook.
    This is the ablation version that uses one code per group instead of attribute factorization.
    
    Args:
        use_dynamic_margin: If True, gradually increase margin during training
    """
    print(f"\n--- Starting Phase 1 (Ablation): Training for Metric Learning with Group Codebook ---")
    print(f"Starting training for {n_epochs} epochs...")
    if use_dynamic_margin:
        print(f"Using dynamic margin scheduling (0.1 -> {inter_group_margin})")
    
    # Early stopping variables
    loss_history = []
    patience_counter = 0
    patience = 50
    min_improvement_fraction = 0.0002
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        
        prediction_head.train()
        embedding_projection.train()
        group_codebook.train()
        
        # Dynamic margin scheduling
        if use_dynamic_margin:
            current_margin = 0.1 + (inter_group_margin - 0.1) * ((epoch+1) / n_epochs)
        else:
            current_margin = inter_group_margin
        
        running_loss = 0.0
        for inputs, labels, group_indices in train_loader:
            optimizer.zero_grad()
            
            # --- Main Forward Pass ---
            embeddings = embedding_model(inputs)
            projected_embeddings = embedding_projection(embeddings)
            outputs = prediction_head(embeddings)
            
            # 1. Main prediction loss
            prediction_loss = criterion(outputs, labels)
            
            # --- Metric Learning Loss with Group Codebook ---
            # Get target centroids from group codebook
            target_centroids = group_codebook(group_indices)
            pull_loss = F.mse_loss(projected_embeddings, target_centroids)
            
            # Push loss: encourage separation between different group codes
            # Simple pairwise distance regularization
            all_group_embeddings = group_codebook.get_all_embeddings()
            pairwise_dist = torch.pdist(all_group_embeddings, p=2)
            push_loss = torch.relu(current_margin - pairwise_dist).mean()
            
            # Weighted combination
            pull_weight = 1.0
            push_weight = 0.1
            iso_loss = pull_weight * pull_loss + push_weight * push_loss
            scale = 100
            iso_loss = scale * iso_loss
            
            # --- Combine Losses ---
            total_loss = prediction_loss + (alpha * iso_loss)
            
            total_loss.backward()
            optimizer.step()
            running_loss += total_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
        
        # Early stopping logic
        loss_history.append(epoch_loss)
        if epoch >= 3:
            moving_avg = sum(loss_history[-3:]) / 3
            if epoch_loss > moving_avg * (1 - min_improvement_fraction):
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
    
    print(f"--- Phase 1 (Ablation) Training Complete ---")

def train_phase2_group(embedding_model, prediction_head, embedding_projection, group_codebook,
                       train_loader, criterion, optimizer, *, n_epochs, beta, 
                       use_vae_disentanglement=True, input_dim=None):
    """
    Phase 2 training for ablation: Disentanglement training with group-level codebook.
    This is the ablation version that uses one code per group instead of attribute factorization.
    
    Args:
        use_vae_disentanglement: If True, adds VAE-style reconstruction loss
        input_dim: Input dimension for reconstruction decoder
    """
    print(f"\n--- Starting Phase 2 (Ablation): Disentanglement Training with Group Codebook ---")
    print(f"Training for {n_epochs} epochs...")
    if use_vae_disentanglement:
        print(f"Using VAE-style disentanglement with reconstruction loss")
    
    # Early stopping variables
    loss_history = []
    patience_counter = 0
    patience = 10
    min_improvement_fraction = 0.0002
    
    # Freeze embedding_projection and group_codebook
    for param in embedding_projection.parameters():
        param.requires_grad = False
    for param in group_codebook.parameters():
        param.requires_grad = False
    
    # Create reconstruction decoder if using VAE-style disentanglement
    if use_vae_disentanglement:
        if input_dim is None:
            raise ValueError("input_dim must be provided when use_vae_disentanglement=True")
        
        device = next(embedding_model.parameters()).device
        decoder_hidden_dim = 256
        decoder = nn.Sequential(
            nn.Linear(PROJECTION_DIM, decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(decoder_hidden_dim, input_dim)
        ).to(device)
        decoder_optimizer = optim.Adam(decoder.parameters(), lr=1e-3)
        reconstruction_criterion = nn.MSELoss()
    else:
        decoder = None
        decoder_optimizer = None
        reconstruction_criterion = None
    
    for epoch in range(n_epochs):
        embedding_model.train()
        prediction_head.train()
        embedding_projection.eval()
        group_codebook.eval()
        if decoder is not None:
            decoder.train()
        
        running_loss_phase2 = 0.0
        running_pred_loss = 0.0
        running_disent_loss = 0.0
        running_recon_loss = 0.0
        
        for inputs, labels, group_indices in train_loader:
            optimizer.zero_grad()
            if decoder_optimizer is not None:
                decoder_optimizer.zero_grad()
            
            # --- Forward Pass ---
            embeddings = embedding_model(inputs)
            projected_embeddings = embedding_projection(embeddings)
            outputs = prediction_head(embeddings)
            
            # --- Loss Calculation ---
            prediction_loss = criterion(outputs, labels)
            
            # Calculate mean code from all group embeddings
            all_group_embeddings = group_codebook.get_all_embeddings()
            mean_code = all_group_embeddings.mean(dim=0)
            mean_code = F.normalize(mean_code, p=2, dim=0)
            
            # Disentanglement loss: distance to mean code
            distances_to_mean = torch.cdist(projected_embeddings, mean_code.unsqueeze(0), p=2)
            disentanglement_loss = distances_to_mean.mean()
            
            # VAE-style reconstruction loss
            if use_vae_disentanglement and decoder is not None:
                reconstructed = decoder(projected_embeddings)
                reconstruction_loss = reconstruction_criterion(reconstructed, inputs)
            else:
                reconstruction_loss = torch.tensor(0.0, device=embeddings.device)
            
            # --- Combine Losses ---
            if use_vae_disentanglement:
                gamma = 0.1
                total_loss_phase2 = prediction_loss + (beta * disentanglement_loss) + (gamma * reconstruction_loss)
            else:
                total_loss_phase2 = prediction_loss + (beta * disentanglement_loss)
            
            total_loss_phase2.backward()
            optimizer.step()
            if decoder_optimizer is not None:
                decoder_optimizer.step()
            
            running_loss_phase2 += total_loss_phase2.item()
            running_pred_loss += prediction_loss.item()
            running_disent_loss += disentanglement_loss.item()
            running_recon_loss += reconstruction_loss.item()
        
        epoch_loss_phase2 = running_loss_phase2 / len(train_loader)
        epoch_pred_loss = running_pred_loss / len(train_loader)
        epoch_disent_loss = running_disent_loss / len(train_loader)
        epoch_recon_loss = running_recon_loss / len(train_loader)
        
        if use_vae_disentanglement:
            print(f"Epoch {epoch+1}/{n_epochs}, Total Loss: {epoch_loss_phase2:.4f}, Pred: {epoch_pred_loss:.4f}, Disent: {epoch_disent_loss:.4f}, Recon: {epoch_recon_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}/{n_epochs}, Total Loss: {epoch_loss_phase2:.4f}, Pred: {epoch_pred_loss:.4f}, Disent: {epoch_disent_loss:.4f}")
        
        # Early stopping logic
        loss_history.append(epoch_loss_phase2)
        if epoch >= 3:
            moving_avg = sum(loss_history[-3:]) / 3
            if epoch_loss_phase2 > moving_avg * (1 - min_improvement_fraction):
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch+1}")
                    break
            else:
                patience_counter = 0
    
    print(f"--- Phase 2 (Ablation) Training Complete ---")

def run_ablation_roar(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test,
                      group_keys, attribute_values, device, group_definition, 
                      train_head=True, n_iterations=10, n_epochs_phase1=40, n_epochs_phase2=20,
                      alpha=100.0, beta=30.0, inter_group_margin=0.5):
    """
    Runs the ablation study ROAR experiment with group-level codebook (no attribute factorization).
    This is the ablation version that compares against the attribute-factorized codebook.
    
    Args:
        X_train, y_train, sensitive_train: Training data
        X_test, y_test, sensitive_test: Test data
        group_keys: List of all intersectional group keys
        attribute_values: Dictionary mapping attribute names to their values
        device: PyTorch device
        group_definition: List of attribute names for group construction
        train_head: Whether to train the prediction head
        n_iterations: Number of ROAR iterations
        n_epochs_phase1: Epochs for Phase 1 in first iteration
        n_epochs_phase2: Epochs for Phase 2
        alpha: Alpha parameter for Phase 1
        beta: Beta parameter for Phase 2
        inter_group_margin: Margin for metric learning
    
    Returns:
        Dictionary with final metrics
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Group-Level Codebook (No Attribute Factorization)")
    print("="*60)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train.values if hasattr(X_train, 'values') else X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train.values if hasattr(y_train, 'values') else y_train, dtype=torch.float32).reshape(-1, 1).to(device)
    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Create group indices for training data
    sensitive_train_copy = sensitive_train.copy()
    sensitive_train_copy['group_key'] = sensitive_train_copy[group_definition].astype(str).agg('_'.join, axis=1)
    group_to_idx = {key: idx for idx, key in enumerate(group_keys)}
    train_group_indices = sensitive_train_copy['group_key'].map(group_to_idx).fillna(-1).astype(int)
    valid_train_mask = train_group_indices != -1
    train_group_indices = train_group_indices[valid_train_mask].values
    X_train_tensor = X_train_tensor[valid_train_mask]
    y_train_tensor = y_train_tensor[valid_train_mask]
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.tensor(train_group_indices, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Initialize models
    input_dim = X_train_tensor.shape[1]
    embedding_model = EmbeddingModel(input_dim).to(device)
    prediction_head = PredictionHead().to(device)
    embedding_projection = EmbeddingProjection().to(device)
    group_codebook = GroupCodebook(group_keys, PROJECTION_DIM).to(device)
    
    criterion = nn.BCELoss()
    
    # Create sensitive attributes for evaluation
    sensitive_test_mapped = sensitive_test.copy()
    sensitive_test_mapped['group_key'] = sensitive_test_mapped[group_definition].astype(str).agg('_'.join, axis=1)
    
    for t in range(n_iterations):
        print(f"\n\n--- Ablation ROAR Iteration {t+1}/{n_iterations} ---")
        
        train_embedding_model = (t == 0)
        
        # Phase 1: Metric Learning
        for param in embedding_model.parameters(): param.requires_grad = train_embedding_model
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = True
        for param in group_codebook.parameters(): param.requires_grad = True
        
        param_groups = [
            {"params": embedding_model.parameters(), "lr": 1e-3} if train_embedding_model else None,
            {"params": prediction_head.parameters(), "lr": 1e-3} if train_head else None,
            {"params": embedding_projection.parameters(), "lr": 1e-3},
            {"params": group_codebook.parameters(), "lr": 5e-4},
        ]
        param_groups = [pg for pg in param_groups if pg is not None]
        optimizer = optim.Adam(param_groups)
        
        train_phase1_group(
            embedding_model, prediction_head, embedding_projection, group_codebook,
            train_loader, criterion, optimizer, 
            n_epochs=(n_epochs_phase1 if t == 0 else n_epochs_phase2), 
            alpha=(alpha if t == 0 else alpha/10), 
            inter_group_margin=inter_group_margin,
            train_embedding_model=train_embedding_model
        )
        
        # Phase 2: Disentanglement
        for param in embedding_model.parameters(): param.requires_grad = True
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = False
        for param in group_codebook.parameters(): param.requires_grad = False
        
        param_groups_iter_2 = [
            {"params": embedding_model.parameters(), "lr": 1e-4},
            {"params": prediction_head.parameters(), "lr": 1e-4} if train_head else None,
        ]
        param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
        optimizer_iter_2 = optim.Adam(param_groups_iter_2)
        
        train_phase2_group(
            embedding_model, prediction_head, embedding_projection, group_codebook,
            train_loader, criterion, optimizer_iter_2, 
            n_epochs=(n_epochs_phase1 if t == 0 else n_epochs_phase2), 
            beta=(beta if t == 0 else beta/3),
            use_vae_disentanglement=True, input_dim=input_dim
        )
    
    # Final evaluation
    print("\n--- Final Ablation ROAR Evaluation ---")
    metrics_final = evaluate_model(
        embedding_model, prediction_head, embedding_projection, group_codebook,
        X_test_tensor, y_test_tensor, sensitive_test_mapped,
        X_test_tensor, torch.empty(0), sensitive_test_mapped,
        return_metrics=True
    )
    
    return metrics_final

def run_ablation_group_inclusion(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test,
                                    target_group, group_keys, attribute_values, device,
                                    group_definition, train_head=True, n_roar_iterations=10,
                                    n_epochs_phase1=25, n_epochs_phase2=20,
                                    alpha=2.0, beta=2.0, inter_group_margin=0.5):
    """
    Runs ablation group inclusion experiment with group-level codebook (0% inclusion, no baselines).
    
    Args:
        X_train, y_train, sensitive_train: Training data
        X_test, y_test, sensitive_test: Test data
        target_group: The group key to exclude from training (0% inclusion)
        group_keys: List of all intersectional group keys
        attribute_values: Dictionary of attribute values
        device: PyTorch device
        group_definition: List of attributes for group construction
        train_head: Whether to train prediction head
        n_roar_iterations: Number of ROAR iterations
        n_epochs_phase1: Epochs for Phase 1 in first iteration
        n_epochs_phase2: Epochs for Phase 2
        alpha: Alpha parameter for Phase 1
        beta: Beta parameter for Phase 2
        inter_group_margin: Margin for metric learning
    
    Returns:
        results: Dictionary with accuracy and F1 on held-out target group
    """
    print("\n" + "="*60)
    print(f"ABLATION GROUP INCLUSION - Target Group: {target_group} (0% Inclusion)")
    print("="*60)
    
    results = {'Ablation_ROAR': {'accuracy': {}, 'f1': {}}}
    
    input_dim = X_train.shape[1]
    criterion = nn.BCELoss()
    
    # Create modified training data with 0% inclusion of target group
    X_train_mod, y_train_mod, sensitive_train_mod, train_group_mask = create_group_inclusion_split(
        X_train, y_train, sensitive_train, target_group, inclusion_percentage=0.0
    )
    
    print(f"Original training size: {len(X_train)}")
    print(f"Modified training size (excluding target group): {len(X_train_mod)}")
    print(f"Target group samples excluded: {train_group_mask.sum()}")
    
    # Verify that target group is really excluded from modified training data
    sensitive_train_copy = sensitive_train_mod.copy()
    sensitive_train_copy['group_key'] = sensitive_train_copy[group_definition].astype(str).agg('_'.join, axis=1)
    target_group_in_train = (sensitive_train_copy['group_key'] == target_group).sum()
    print(f"Verification: Target group samples in modified training data: {target_group_in_train}")
    if target_group_in_train > 0:
        print(f"ERROR: Target group {target_group} should be excluded but {target_group_in_train} samples found in training!")
        return results
    
    # Construct intersectional groups for training data
    group_keys_train, _, attribute_indices_train_mod, _, valid_indices = construct_intersectional_groups(
        sensitive_train_mod, group_definition, attribute_values
    )
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_mod.values if hasattr(X_train_mod, 'values') else X_train_mod, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train_mod.values if hasattr(y_train_mod, 'values') else y_train_mod, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Filter training data to only include valid samples
    X_train_tensor = X_train_tensor[valid_indices]
    y_train_tensor = y_train_tensor[valid_indices]
    
    # Convert attribute indices to group indices for GroupCodebook
    # GroupCodebook expects a single index per sample, not attribute indices
    # Use group_keys_train from the modified training data
    group_to_idx = {key: idx for idx, key in enumerate(group_keys_train)}
    
    # Convert numeric attribute indices back to group keys using attribute_values
    train_group_keys = []
    for row in attribute_indices_train_mod:
        # Convert numeric indices to actual attribute values
        attr_values_list = []
        for attr_idx, attr_name in enumerate(row):
            attr_values_list.append(attribute_values[group_definition[attr_idx]][attr_name])
        train_group_keys.append('_'.join(map(str, attr_values_list)))
    
    train_group_indices = np.array([group_to_idx[gk] for gk in train_group_keys])
    
    # Create DataLoader
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.tensor(train_group_indices, dtype=torch.long))
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    
    # Create test tensors
    X_test_tensor = torch.tensor(X_test.values if hasattr(X_test, 'values') else X_test, dtype=torch.float32).to(device)
    y_test_tensor = torch.tensor(y_test.values if hasattr(y_test, 'values') else y_test, dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Create group_key for test data
    sensitive_test_copy = sensitive_test.copy()
    sensitive_test_copy['group_key'] = sensitive_test_copy[group_definition].astype(str).agg('_'.join, axis=1)
    
    # Test on held-out target group
    test_group_mask = sensitive_test_copy['group_key'] == target_group
    X_test_heldout = X_test[test_group_mask.values]
    y_test_heldout = y_test[test_group_mask.values]
    
    if len(X_test_heldout) == 0:
        print(f"Warning: No test samples for target group {target_group}.")
        return results
    
    # Verify that test samples are indeed from target group
    test_heldout_copy = sensitive_test.iloc[test_group_mask.values].copy()
    test_heldout_copy['group_key'] = test_heldout_copy[group_definition].astype(str).agg('_'.join, axis=1)
    test_target_group_count = (test_heldout_copy['group_key'] == target_group).sum()
    print(f"Verification: Test samples from target group {target_group}: {test_target_group_count}/{len(X_test_heldout)}")
    if test_target_group_count != len(X_test_heldout):
        print(f"ERROR: Expected all {len(X_test_heldout)} test samples to be from {target_group}, but only {test_target_group_count} are!")
        return results
    
    X_test_heldout_tensor = torch.tensor(X_test_heldout.values if hasattr(X_test_heldout, 'values') else X_test_heldout, dtype=torch.float32).to(device)
    y_test_heldout_tensor = torch.tensor(y_test_heldout.values if hasattr(y_test_heldout, 'values') else y_test_heldout, dtype=torch.float32).reshape(-1, 1).to(device)
    
    print(f"Test samples (held-out target group): {len(X_test_heldout)}")
    
    # --- Ablation ROAR with Group Codebook ---
    print(f"\n--- Ablation ROAR ({n_roar_iterations} iteration(s)) ---")
    embedding_model = EmbeddingModel(input_dim).to(device)
    prediction_head = PredictionHead().to(device)
    embedding_projection = EmbeddingProjection().to(device)
    group_codebook = GroupCodebook(group_keys_train, PROJECTION_DIM).to(device)
    
    for iteration in range(n_roar_iterations):
        print(f"\n--- Ablation ROAR Iteration {iteration + 1}/{n_roar_iterations} ---")
        
        # Phase 1
        train_embedding_model = (iteration == 0)
        for param in embedding_model.parameters(): param.requires_grad = train_embedding_model
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = True
        for param in group_codebook.parameters(): param.requires_grad = True
        
        param_groups = [
            {"params": embedding_model.parameters(), "lr": 1e-3} if train_embedding_model else None,
            {"params": prediction_head.parameters(), "lr": 1e-3} if train_head else None,
            {"params": embedding_projection.parameters(), "lr": 1e-3},
            {"params": group_codebook.parameters(), "lr": 5e-4},
        ]
        param_groups = [pg for pg in param_groups if pg is not None]
        optimizer = optim.Adam(param_groups)
        
        train_phase1_group(embedding_model, prediction_head, embedding_projection, group_codebook,
                         train_loader, criterion, optimizer, 
                         n_epochs=(n_epochs_phase1 if iteration == 0 else n_epochs_phase2), 
                         alpha=alpha, 
                         inter_group_margin=inter_group_margin, train_embedding_model=train_embedding_model)
        
        # Phase 2
        for param in embedding_model.parameters(): param.requires_grad = True
        for param in prediction_head.parameters(): param.requires_grad = True if train_head else False
        for param in embedding_projection.parameters(): param.requires_grad = False
        for param in group_codebook.parameters(): param.requires_grad = False
        
        param_groups_iter_2 = [
            {"params": embedding_model.parameters(), "lr": 1e-4},
            {"params": prediction_head.parameters(), "lr": 1e-4} if train_head else None,
        ]
        param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
        optimizer_iter_2 = optim.Adam(param_groups_iter_2)
        
        train_phase2_group(embedding_model, prediction_head, embedding_projection, group_codebook,
                         train_loader, criterion, optimizer_iter_2, 
                         n_epochs=(n_epochs_phase1 if iteration == 0 else n_epochs_phase2), 
                         beta=beta,
                         use_vae_disentanglement=True, input_dim=input_dim)
    
    # Compute accuracy and F1 on held-out target group
    with torch.no_grad():
        embeddings = embedding_model(X_test_heldout_tensor)
        outputs = prediction_head(embeddings)
        predictions = (outputs > 0.5).float()
        accuracy = (predictions == y_test_heldout_tensor).float().mean()
        predictions_np = predictions.cpu().numpy()
        labels_np = y_test_heldout_tensor.cpu().numpy()
        f1 = f1_score(labels_np, predictions_np, zero_division=0)
        results['Ablation_ROAR']['accuracy'][0.0] = accuracy.item()
        results['Ablation_ROAR']['f1'][0.0] = f1
        print(f"Ablation ROAR Accuracy on held-out group: {accuracy.item():.4f}, F1: {f1:.4f}")
    
    # Compute fairness metrics
    fairness_metrics_result = compute_fairness_metrics(embedding_model, prediction_head, 
                                                      X_test_tensor, y_test_tensor, sensitive_test,
                                                      target_group, group_definition, attribute_values, device)
    
    print("\n" + "="*60)
    print("GROUP INCLUSION EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Target Group: {target_group}")
    print(f"Inclusion Percentage: 0% (excluded from training)")
    print(f"Ablation ROAR Accuracy: {results['Ablation_ROAR']['accuracy'][0.0]:.4f}")
    print(f"Ablation ROAR F1: {results['Ablation_ROAR']['f1'][0.0]:.4f}")
    print(f"Ablation ROAR - Accuracy Distance: {fairness_metrics_result['accuracy_distance']:.4f}")
    print(f"Ablation ROAR - F1 Distance: {fairness_metrics_result['f1_distance']:.4f}")
    print(f"Ablation ROAR - Embedding Distance Sq: {fairness_metrics_result['embedding_distance_sq']:.4f}")
    
    return results

def train_group_dro(embedding_model, prediction_head, train_loader,
                   criterion, optimizer, n_epochs, eta=0.01, train_embedding_model=True):
    """
    Trains models using Group DRO (Distributionally Robust Optimization).
    This method uses group-weighted loss updates to handle distribution shifts.
    """
    print(f"\n--- Starting Group DRO Training ---")
    print(f"Training for {n_epochs} epochs with eta={eta}...")
    
    # Initialize group weights uniformly
    num_groups = 4  # 4 intersectional groups
    group_weights = torch.ones(num_groups, device=next(embedding_model.parameters()).device) / num_groups
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        prediction_head.train()
        
        running_loss = 0.0
        
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = embedding_model(inputs)
            outputs = prediction_head(embeddings)
            
            # Compute per-sample losses
            sample_losses = F.binary_cross_entropy_with_logits(outputs, labels, reduction='none')
            
            # Compute group-wise losses
            group_losses_full = torch.zeros(num_groups, device=next(embedding_model.parameters()).device)
            for group_idx in range(num_groups):
                group_mask = (attribute_indices[:, 0] * 2 + attribute_indices[:, 1] == group_idx)
                if group_mask.sum() > 0:
                    group_loss = sample_losses[group_mask].mean()
                    group_losses_full[group_idx] = group_loss
            
            # Weighted loss
            valid_groups = group_losses_full > 0
            if valid_groups.sum() > 0:
                weighted_loss = (group_weights[valid_groups] * group_losses_full[valid_groups]).sum()
            else:
                weighted_loss = sample_losses.mean()
            
            # Backpropagate
            weighted_loss.backward()
            optimizer.step()
            
            # Update group weights (exponential moving average)
            if valid_groups.sum() > 0:
                # Only update weights for groups present in this batch
                group_weights_valid = group_weights[valid_groups]
                group_losses_valid = group_losses_full[valid_groups].detach()
                group_weights[valid_groups] = group_weights_valid * torch.exp(eta * group_losses_valid)
                group_weights = group_weights / group_weights.sum()
            
            running_loss += weighted_loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}, Group Weights: {group_weights.detach().cpu().numpy()}")
    
    print(f"--- Group DRO Training Complete ---")
    return group_weights

def train_laftr(embedding_model, task_head, adversaries, train_loader,
               task_criterion, adv_criterion, optimizer, n_epochs,
               lambda_adv=0.1, lambda_task=1.0, train_embedding_model=True,
               adv_attr_indices=None):
    """
    Trains models using LAFTR (Learning Adversarially Fair and Transferable Representations).
    This method uses gradient reversal to make representations fair.
    
    Args:
        adv_attr_indices: List of attribute indices to use for adversaries (e.g., [0, 1] for first 2 attributes)
                         If None, defaults to [1, 0] for backward compatibility with ACS Income
    """
    print(f"\n--- Starting LAFTR Training ---")
    print(f"Training for {n_epochs} epochs with lambda_adv={lambda_adv}, lambda_task={lambda_task}...")
    
    if adv_attr_indices is None:
        # Default for backward compatibility with ACS Income (sex, race)
        adv_attr_indices = [1, 0]
    
    # Lambda scheduling: gradually increase from 0 to lambda_adv
    lambda_schedule = lambda_adv * torch.linspace(0, 1, n_epochs)
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        task_head.train()
        for adversary in adversaries:
            adversary.train()
        
        running_task_loss = 0.0
        running_adv_loss = 0.0
        running_total_loss = 0.0
        
        current_lambda = lambda_schedule[epoch].item()
        
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = embedding_model(inputs)
            task_outputs = task_head(embeddings)
            
            # Task loss
            task_loss = task_criterion(task_outputs, labels)
            
            # Adversary losses
            # Use adv_attr_indices to select which attribute column to use for each adversary
            attr_labels = [attribute_indices[:, idx] for idx in adv_attr_indices]
            adv_losses = []
            for i, (adversary, attr_label) in enumerate(zip(adversaries, attr_labels)):
                adv_outputs = adversary(embeddings, current_lambda)
                # adv_criterion is a list of criterion functions, one for each adversary
                adv_loss = adv_criterion[i](adv_outputs, attr_label)
                adv_losses.append(adv_loss)
            
            total_adv_loss = sum(adv_losses)
            
            # Total loss
            total_loss = lambda_task * task_loss + current_lambda * total_adv_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            running_task_loss += task_loss.item()
            running_adv_loss += total_adv_loss.item()
            running_total_loss += total_loss.item()
        
        epoch_task_loss = running_task_loss / len(train_loader)
        epoch_adv_loss = running_adv_loss / len(train_loader)
        epoch_total_loss = running_total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Lambda: {current_lambda:.4f}")
        print(f"  Task Loss: {epoch_task_loss:.4f}, Adv Loss: {epoch_adv_loss:.4f}, Total Loss: {epoch_total_loss:.4f}")
    
    print(f"--- LAFTR Training Complete ---")

def train_road(embedding_model, prediction_head, perturbation_net, train_loader,
               criterion, optimizer, n_epochs, alpha=0.5, train_embedding_model=True):
    """
    Trains models using ROAD (Robust Optimization for Adversarial Debiasing).
    This method uses robust optimization with adversarial perturbations to handle
    distribution shifts and ensure fair performance across groups.
    """
    print(f"\n--- Starting ROAD Training ---")
    print(f"Training for {n_epochs} epochs with alpha={alpha}...")
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        prediction_head.train()
        perturbation_net.train()
        
        running_clean_loss = 0.0
        running_robust_loss = 0.0
        running_total_loss = 0.0
        
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            # Clean forward pass
            embeddings = embedding_model(inputs)
            clean_outputs = prediction_head(embeddings)
            clean_loss = criterion(clean_outputs, labels)
            
            # Adversarial perturbation forward pass
            # Generate perturbed inputs
            perturbed_inputs = perturbation_net(inputs)
            perturbed_embeddings = embedding_model(perturbed_inputs)
            perturbed_outputs = prediction_head(perturbed_embeddings)
            robust_loss = criterion(perturbed_outputs, labels)
            
            # Group-wise robust loss (worst-case across groups)
            group_losses = []
            for group_idx in range(4):  # 4 intersectional groups
                group_mask = (attribute_indices[:, 0] * 2 + attribute_indices[:, 1] == group_idx)
                if group_mask.sum() > 0:
                    group_outputs = perturbed_outputs[group_mask]
                    group_labels = labels[group_mask]
                    group_loss = criterion(group_outputs, group_labels)
                    group_losses.append(group_loss)
            
            if group_losses:
                worst_group_loss = torch.stack(group_losses).max()  # Worst-case
                robust_loss = robust_loss + 0.1 * worst_group_loss  # Add worst-group penalty
            
            # Total loss: weighted combination of clean and robust loss
            total_loss = (1 - alpha) * clean_loss + alpha * robust_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            running_clean_loss += clean_loss.item()
            running_robust_loss += robust_loss.item()
            running_total_loss += total_loss.item()
        
        epoch_clean_loss = running_clean_loss / len(train_loader)
        epoch_robust_loss = running_robust_loss / len(train_loader)
        epoch_total_loss = running_total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Clean Loss: {epoch_clean_loss:.4f}, Robust Loss: {epoch_robust_loss:.4f}, Total Loss: {epoch_total_loss:.4f}")
    
    print(f"--- ROAD Training Complete ---")

def train_adversarial_learning(embedding_model, task_head, adversary, train_loader,
                               task_criterion, adv_criterion, optimizer, n_epochs,
                               alpha=1.0, train_embedding_model=True, adv_attr_indices=None):
    """
    Trains models using AdversarialLearning (Mitigating Unwanted Biases with Adversarial Learning).
    This method removes sensitive information from representations using adversarial training.
    
    Args:
        adv_attr_indices: List of attribute indices to use for adversaries (e.g., [0, 1] for first 2 attributes)
                         If None, defaults to [1, 0] for backward compatibility with ACS Income
    """
    print(f"\n--- Starting AdversarialLearning Training ---")
    print(f"Training for {n_epochs} epochs with alpha={alpha}...")
    
    if adv_attr_indices is None:
        # Default for backward compatibility with ACS Income (sex, race)
        adv_attr_indices = [1, 0]
    
    # Lambda scheduling: gradually increase from 0 to 1
    lambda_schedule = torch.linspace(0, 1, n_epochs)
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        task_head.train()
        adversary.train()
        
        running_task_loss = 0.0
        running_adv_loss = 0.0
        running_total_loss = 0.0
        
        current_lambda = lambda_schedule[epoch].item()
        
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = embedding_model(inputs)
            task_outputs = task_head(embeddings)
            
            # Task loss
            task_loss = task_criterion(task_outputs, labels)
            
            # Adversary loss
            # Get attribute labels using adv_attr_indices
            attr_labels = [attribute_indices[:, idx] for idx in adv_attr_indices]
            
            # Adversary prediction with gradient reversal
            attr_logits = adversary(embeddings, current_lambda)
            
            # Compute adversary losses for each attribute
            adv_losses = []
            for i, (logits, attr_label) in enumerate(zip(attr_logits, attr_labels)):
                adv_loss = adv_criterion(logits, attr_label)
                adv_losses.append(adv_loss)
            
            total_adv_loss = sum(adv_losses)
            
            # Total loss
            total_loss = task_loss + alpha * total_adv_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            running_task_loss += task_loss.item()
            running_adv_loss += total_adv_loss.item()
            running_total_loss += total_loss.item()
        
        epoch_task_loss = running_task_loss / len(train_loader)
        epoch_adv_loss = running_adv_loss / len(train_loader)
        epoch_total_loss = running_total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs}, Lambda: {current_lambda:.4f}")
        print(f"  Task Loss: {epoch_task_loss:.4f}, Adv Loss: {epoch_adv_loss:.4f}, Total Loss: {epoch_total_loss:.4f}")
    
    print(f"--- AdversarialLearning Training Complete ---")

def compute_hsic_loss(embeddings, attribute_indices, lambda_hsic=0.1, num_attr_cols=None):
    """
    Computes HSIC loss to make embeddings independent of sensitive attributes.
    Uses linear kernel on embeddings and delta kernel on sensitive attributes.
    
    Args:
        num_attr_cols: Number of attribute columns to use (default: all columns)
    """
    batch_size = embeddings.shape[0]
    
    # Linear kernel on embeddings: K_X(i,j) = x_i^T x_j
    K_X = torch.matmul(embeddings, embeddings.t())
    
    # Delta kernel on sensitive attributes: K_Y(i,j) = 1 if same group, else 0
    # Compute intersectional group keys from all specified attribute columns
    if num_attr_cols is None:
        num_attr_cols = attribute_indices.shape[1]
    
    # Use the first num_attr_cols columns to compute group keys
    # This handles any number of attributes by encoding them as a tuple
    attr_subset = attribute_indices[:, :num_attr_cols]
    
    # Convert each row to a tuple for comparison
    # For efficiency, we can compute pairwise equality directly
    K_Y = torch.ones(batch_size, batch_size, device=embeddings.device)
    for i in range(num_attr_cols):
        K_Y = K_Y * (attr_subset[:, i].unsqueeze(0) == attr_subset[:, i].unsqueeze(1)).float()
    
    # Centering matrix: H = I - (1/n) * 11^T
    H = torch.eye(batch_size, device=embeddings.device) - (1.0 / batch_size) * torch.ones(batch_size, batch_size, device=embeddings.device)
    
    # HSIC = trace(K_X * H * K_Y * H) / (n-1)^2
    HSIC = torch.trace(torch.matmul(torch.matmul(K_X, H), torch.matmul(K_Y, H))) / ((batch_size - 1) ** 2)
    
    return lambda_hsic * HSIC

def train_hsic(embedding_model, prediction_head, train_loader, 
              criterion, optimizer, n_epochs, lambda_hsic=0.1, 
              train_embedding_model=True, num_attr_cols=None):
    """
    Trains models using HSIC regularization to make embeddings independent of sensitive attributes.
    This method uses linear kernel on embeddings and delta kernel on sensitive attributes.
    
    Args:
        num_attr_cols: Number of attribute columns to use for HSIC (default: all columns)
    """
    print(f"\n--- Starting HSIC Regularization Training ---")
    print(f"Training for {n_epochs} epochs with lambda_hsic={lambda_hsic}...")
    
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        prediction_head.train()
        
        running_task_loss = 0.0
        running_hsic_loss = 0.0
        running_total_loss = 0.0
        
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            # Forward pass
            embeddings = embedding_model(inputs)
            outputs = prediction_head(embeddings)
            
            # Task loss
            task_loss = criterion(outputs, labels)
            
            # HSIC loss
            hsic_loss = compute_hsic_loss(embeddings, attribute_indices, lambda_hsic, num_attr_cols)
            
            # Total loss
            total_loss = task_loss + hsic_loss
            
            # Backpropagate
            total_loss.backward()
            optimizer.step()
            
            running_task_loss += task_loss.item()
            running_hsic_loss += hsic_loss.item()
            running_total_loss += total_loss.item()
        
        epoch_task_loss = running_task_loss / len(train_loader)
        epoch_hsic_loss = running_hsic_loss / len(train_loader)
        epoch_total_loss = running_total_loss / len(train_loader)
        
        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"  Task Loss: {epoch_task_loss:.4f}, HSIC Loss: {epoch_hsic_loss:.4f}, Total Loss: {epoch_total_loss:.4f}")
    
    print(f"--- HSIC Regularization Training Complete ---")

def train_inlp(embedding_model, prediction_head, train_loader, criterion, optimizer, 
               n_epochs=20, train_embedding_model=True, num_iterations=10):
    """
    Trains models using INLP (Iterative Nullspace Projection) to remove sensitive information.
    INLP iteratively trains a linear classifier to predict sensitive attributes from embeddings,
    then projects the embeddings onto the nullspace of the classifier's weights.
    
    Args:
        num_iterations: Number of projection iterations (default: 10)
    """
    print(f"\n--- Starting INLP Training ---")
    print(f"Training for {n_epochs} epochs with {num_iterations} projection iterations...")
    
    device = next(embedding_model.parameters()).device
    
    # First, train the embedding model normally for a few epochs
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        prediction_head.train()
        
        running_loss = 0.0
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            embeddings = embedding_model(inputs)
            outputs = prediction_head(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
    
    # Now apply INLP projections
    print(f"\nApplying INLP projections...")
    embedding_model.eval()
    
    # Collect all embeddings and sensitive attributes
    all_embeddings = []
    all_sensitive = []
    
    with torch.no_grad():
        for inputs, labels, attribute_indices in train_loader:
            embeddings = embedding_model(inputs)
            all_embeddings.append(embeddings.cpu())
            # Use the first attribute column for INLP (can be extended for multiple)
            all_sensitive.append(attribute_indices[:, 0].cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_sensitive = torch.cat(all_sensitive, dim=0)
    
    # Compute projection matrix
    projection_matrix = torch.eye(all_embeddings.shape[1]).to(device)
    
    for iteration in range(num_iterations):
        # Train linear classifier to predict sensitive attribute
        from sklearn.linear_model import LogisticRegression
        clf = LogisticRegression(max_iter=1000, n_jobs=-1)
        clf.fit(all_embeddings.numpy(), all_sensitive.numpy())
        
        # Get weight vector
        w = torch.tensor(clf.coef_, dtype=torch.float32).to(device)
        w = w / (torch.norm(w) + 1e-8)
        
        # Compute projection onto nullspace
        P = torch.eye(w.shape[1]).to(device) - w.T @ w
        projection_matrix = P @ projection_matrix
        
        # Project embeddings
        all_embeddings = all_embeddings @ P.T
        
        print(f"  Iteration {iteration+1}/{num_iterations} complete")
    
    # Store projection matrix in embedding model for inference
    embedding_model.inlp_projection = projection_matrix
    
    print(f"--- INLP Training Complete ---")

def train_rlace(embedding_model, prediction_head, train_loader, criterion, optimizer,
                n_epochs=20, train_embedding_model=True, num_iterations=10, alpha=100.0):
    """
    Trains models using RLACE (Robust Linear Adversarial Concept Erasure) to remove sensitive information.
    RLACE finds a projection matrix that minimizes the mutual information between embeddings and sensitive attributes.
    
    Args:
        num_iterations: Number of optimization iterations for projection matrix (default: 10)
        alpha: Regularization parameter for RLACE (default: 100.0)
    """
    print(f"\n--- Starting RLACE Training ---")
    print(f"Training for {n_epochs} epochs with {num_iterations} projection iterations, alpha={alpha}...")
    
    device = next(embedding_model.parameters()).device
    
    # First, train the embedding model normally for a few epochs
    for epoch in range(n_epochs):
        if train_embedding_model:
            embedding_model.train()
        else:
            embedding_model.eval()
        prediction_head.train()
        
        running_loss = 0.0
        for inputs, labels, attribute_indices in train_loader:
            optimizer.zero_grad()
            
            embeddings = embedding_model(inputs)
            outputs = prediction_head(embeddings)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}")
    
    # Now apply RLACE projection
    print(f"\nApplying RLACE projection...")
    embedding_model.eval()
    
    # Collect all embeddings and sensitive attributes
    all_embeddings = []
    all_sensitive = []
    
    with torch.no_grad():
        for inputs, labels, attribute_indices in train_loader:
            embeddings = embedding_model(inputs)
            all_embeddings.append(embeddings.cpu())
            all_sensitive.append(attribute_indices[:, 0].cpu())
    
    all_embeddings = torch.cat(all_embeddings, dim=0)
    all_sensitive = torch.cat(all_sensitive, dim=0)
    
    # Compute covariance matrices
    X = all_embeddings - all_embeddings.mean(dim=0)
    Y = all_sensitive.float().unsqueeze(1) - all_sensitive.float().mean()
    
    C_xx = X.T @ X / X.shape[0]
    C_xy = X.T @ Y / X.shape[0]
    C_yy = Y.T @ Y / Y.shape[0]
    
    # Initialize projection matrix
    d = X.shape[1]
    P = torch.eye(d, dtype=torch.float32).to(device)
    P.requires_grad = True
    
    # Optimize projection matrix with lower learning rate and gradient clipping
    proj_optimizer = torch.optim.Adam([P], lr=0.001)
    
    for iteration in range(num_iterations):
        proj_optimizer.zero_grad()
        
        # Compute projected embeddings
        X_proj = X @ P.T
        
        # Compute mutual information proxy
        C_xx_proj = X_proj.T @ X_proj / X.shape[0]
        C_xy_proj = X_proj.T @ Y / X.shape[0]
        
        # RLACE objective: minimize correlation with sensitive attribute
        # while preserving information (proximity to identity)
        # Use more stable computation for MI loss
        try:
            C_yy_inv = torch.inverse(C_yy + 1e-5 * torch.eye(C_yy.shape[0]).to(device))
            mi_loss = torch.trace(C_xy_proj @ C_yy_inv @ C_xy_proj.T)
        except:
            # Fallback if inverse fails
            mi_loss = torch.norm(C_xy_proj)
        
        # Normalize preservation loss by matrix size to prevent explosion
        preserver_loss = torch.norm(P - torch.eye(d).to(device)) / d
        
        total_loss = mi_loss + alpha * preserver_loss
        
        total_loss.backward()
        
        # Gradient clipping to prevent explosion
        torch.nn.utils.clip_grad_norm_([P], max_norm=1.0)
        
        proj_optimizer.step()
        
        # Orthonormalize projection matrix
        U, S, V = torch.svd(P)
        P = U @ V.T
        P = P.detach().requires_grad_(True)
        
        print(f"  Iteration {iteration+1}/{num_iterations}, Loss: {total_loss.item():.4f}, MI: {mi_loss.item():.4f}, Pres: {preserver_loss.item():.4f}")
    
    # Store projection matrix in embedding model for inference
    embedding_model.rlace_projection = P.detach()
    
    print(f"--- RLACE Training Complete ---")

# ==================== Evaluation Functions ====================

def calculate_group_fairness_metrics(y_true, y_pred, sensitive_attributes, silent=False):
    """Calculates Demographic Parity and Equalized Odds for intersectional groups.
    
    Args:
        silent: If True, suppresses printing
    """
    if not silent:
        print("\n--- Calculating Group Fairness Metrics ---")
    
    # Convert to numpy if tensors and flatten to 1D
    if torch.is_tensor(y_true):
        y_true = y_true.detach().cpu().numpy().flatten()
    if torch.is_tensor(y_pred):
        y_pred = y_pred.detach().cpu().numpy().flatten()
    
    # Create group_key if not already present
    if 'group_key' not in sensitive_attributes.columns:
        # Use all available sensitive attributes to create intersectional group key
        # This adapts to different datasets with different numbers of attributes
        # Handle duplicate columns (e.g., both 'RACE' and 'race') by prioritizing uppercase
        attr_cols = [col for col in sensitive_attributes.columns if col != 'group_key']
        # Remove lowercase duplicates if uppercase version exists
        seen_upper = set()
        filtered_cols = []
        for col in attr_cols:
            upper_col = col.upper()
            if upper_col in seen_upper:
                continue  # Skip if we already have uppercase version
            if col.isupper():
                seen_upper.add(col)
                filtered_cols.append(col)
            elif upper_col not in seen_upper:
                # Keep lowercase only if uppercase version doesn't exist
                filtered_cols.append(col)
                seen_upper.add(upper_col)
        
        if len(filtered_cols) >= 2:
            sensitive_attributes['group_key'] = sensitive_attributes[filtered_cols].astype(str).agg('_'.join, axis=1)
        else:
            print("Error: Need at least two sensitive attribute columns to define groups.")
            return

    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group_key': sensitive_attributes['group_key']
    })

    # Filter groups with size >= MIN_GROUP_SIZE to avoid noise from tiny groups
    group_sizes = df['group_key'].value_counts()
    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index.tolist()
    df = df[df['group_key'].isin(valid_groups)]

    if df.empty:
        print(f"  - No intersectional groups with size >= {MIN_GROUP_SIZE} found to calculate fairness metrics.")
        return

    attribute = 'group_key'
    if not silent:
        print(f"Metrics for attribute: '{attribute}' (filtered for groups with size >= {MIN_GROUP_SIZE})")
    groups = df[attribute].unique()
    
    tprs = {}
    fprs = {}
    for group in sorted(groups):
        group_df = df[df[attribute] == group]
        # Calculate TPR, handling cases where there are no positive samples
        if (group_df['y_true'] == 1).sum() > 0:
            tpr = (group_df['y_pred'][group_df['y_true'] == 1] == 1).mean()
        else:
            tpr = float('nan')
        # Calculate FPR, handling cases where there are no negative samples
        if (group_df['y_true'] == 0).sum() > 0:
            fpr = (group_df['y_pred'][group_df['y_true'] == 0] == 1).mean()
        else:
            fpr = float('nan')
        
        tprs[group] = tpr
        fprs[group] = fpr
        if not silent:
            print(f"    - {group}: TPR={tpr:.4f}, FPR={fpr:.4f}")
    
    # Calculate P(Y=1|group) for each group with Laplace smoothing
    positive_probs = {}
    for group in groups:
        group_df = df[df[attribute] == group]
        # Laplace smoothing: add 1 to both numerator and denominator
        pos_count = (group_df['y_pred'] == 1).sum() + 1
        total_count = len(group_df) + 2  # +2 for binary outcome
        positive_probs[group] = pos_count / total_count
    
    # γ-Subgroup Fairness (γ-Equalized Odds per "Preventing Fairness Gerrymandering")
    if not silent:
        print("  - γ-Subgroup Fairness (γ-Equalized Odds):")
    
    # Calculate overall TPR and FPR
    overall_tp = (df['y_pred'] == 1).sum()
    overall_fp = ((df['y_pred'] == 1) & (df['y_true'] == 0)).sum()
    overall_positives = (df['y_true'] == 1).sum()
    overall_negatives = (df['y_true'] == 0).sum()
    
    overall_tpr = overall_tp / overall_positives if overall_positives > 0 else 0.0
    overall_fpr = overall_fp / overall_negatives if overall_negatives > 0 else 0.0
    
    # Calculate γ as the maximum of |TPR_g - TPR_overall| / P(g) and |FPR_g - FPR_overall| / P(g)
    gamma_subgroup_fairness = 0.0
    total_samples = len(df)
    
    for group in groups:
        group_df = df[df[attribute] == group]
        group_size = len(group_df)
        group_prob = group_size / total_samples
        
        # Calculate group TPR and FPR
        group_tp = (group_df['y_pred'] == 1).sum()
        group_fp = ((group_df['y_pred'] == 1) & (group_df['y_true'] == 0)).sum()
        group_positives = (group_df['y_true'] == 1).sum()
        group_negatives = (group_df['y_true'] == 0).sum()
        
        group_tpr = group_tp / group_positives if group_positives > 0 else 0.0
        group_fpr = group_fp / group_negatives if group_negatives > 0 else 0.0
        
        # Calculate γ for this group: |rate_g - rate_overall| * P(g)
        if group_prob > 0:
            gamma_tpr = abs(group_tpr - overall_tpr) * group_prob
            gamma_fpr = abs(group_fpr - overall_fpr) * group_prob
            gamma_subgroup_fairness = max(gamma_subgroup_fairness, gamma_tpr, gamma_fpr)
    
    if not silent:
        print(f"    - Maximum γ-Subgroup Fairness: {gamma_subgroup_fairness:.6f}")
    
    return gamma_subgroup_fairness

def calculate_hsic(embeddings, sensitive_attributes, attribute):
    """
    Calculates the Hilbert-Schmidt Independence Criterion (HSIC) between embeddings and a single sensitive attribute.
    Uses linear kernel on embeddings and delta kernel on sensitive attributes.
    Memory-efficient implementation that avoids constructing full kernel matrices.
    
    Args:
        embeddings: Tensor or numpy array of shape (n_samples, embedding_dim)
        sensitive_attributes: DataFrame with sensitive attributes
        attribute: Name of the attribute column to compute HSIC for
    
    Returns:
        hsic_value: HSIC value for this attribute
    """
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    n_samples = embeddings_np.shape[0]
    
    # Get attribute labels for delta kernel
    if attribute not in sensitive_attributes.columns:
        return 0.0
    
    attr_labels = sensitive_attributes[attribute].values
    
    # Center the embeddings
    embeddings_centered = embeddings_np - embeddings_np.mean(axis=0, keepdims=True)
    
    # Compute HSIC using memory-efficient formula:
    # HSIC = (1/(n-1)^2) * trace(K_X @ H @ K_Y @ H)
    # For linear kernel K_X = X @ X.T and delta kernel K_Y, this simplifies to:
    # HSIC = (1/(n-1)^2) * sum_{g} sum_{i,j in group_g} (x_i^T @ x_j)
    # where the sum is over all pairs in the same group
    
    # Compute sum of dot products within each group
    unique_groups = np.unique(attr_labels)
    hsic = 0.0
    
    for group in unique_groups:
        group_mask = attr_labels == group
        group_embeddings = embeddings_centered[group_mask]
        n_group = group_embeddings.shape[0]
        
        if n_group == 0:
            continue
        
        # Compute sum of pairwise dot products for this group
        # sum_{i,j in group} (x_i^T @ x_j) = ||sum_{i in group} x_i||^2
        group_sum = group_embeddings.sum(axis=0)
        group_dot_sum = np.dot(group_sum, group_sum)
        
        hsic += group_dot_sum
    
    hsic = hsic / ((n_samples - 1) ** 2)
    
    return hsic

def calculate_trace_scatter_matrix(embeddings, sensitive_attributes):
    """
    Calculates the trace of between-group scatter matrix using attribute factorization.
    For each sensitive attribute, calculates the expected L2 distance squared between
    the average embedding of each attribute value group and the overall average embedding,
    weighted by marginal empirical probabilities, then sums across all attributes.
    
    Args:
        embeddings: Tensor of shape (n_samples, embedding_dim)
        sensitive_attributes: DataFrame with sensitive attribute columns (e.g., 'sex', 'race')
    
    Returns:
        float: Trace of between-group scatter matrix
    """
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    n_samples = embeddings_np.shape[0]
    
    # Calculate overall average embedding
    overall_mean = embeddings_np.mean(axis=0)
    
    # Define attributes to process dynamically based on available columns
    # Filter out duplicate lowercase columns (e.g., both 'RACE' and 'race')
    attr_cols = [col for col in sensitive_attributes.columns if col != 'group_key']
    # Remove lowercase duplicates if uppercase version exists
    seen_upper = set()
    attributes = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue  # Skip if we already have uppercase version
        if col.isupper():
            seen_upper.add(col)
            attributes.append(col)
        elif upper_col not in seen_upper:
            # Keep lowercase only if uppercase version doesn't exist
            attributes.append(col)
            seen_upper.add(upper_col)
    
    total_trace = 0.0
    
    for attr in attributes:
        if attr not in sensitive_attributes.columns:
            continue
            
        # Get unique values for this attribute
        unique_values = sensitive_attributes[attr].unique()
        
        attr_trace = 0.0
        
        for value in unique_values:
            # Get samples with this attribute value
            mask = sensitive_attributes[attr] == value
            n_value = mask.sum()
            
            if n_value == 0:
                continue
            
            # Calculate marginal probability
            marginal_prob = n_value / n_samples
            
            # Calculate average embedding for this attribute value
            value_mean = embeddings_np[mask].mean(axis=0)
            
            # Calculate L2 distance squared
            distance_sq = np.sum((value_mean - overall_mean) ** 2)
            
            # Weight by marginal probability
            weighted_distance_sq = marginal_prob * distance_sq
            
            attr_trace += weighted_distance_sq
        
        print(f"  - Trace for attribute '{attr}': {attr_trace:.6f}")
        total_trace += attr_trace
    
    return total_trace

def evaluate_information_leakage(embeddings, sensitive_attributes, silent=False, min_group_size=None):
    """
    Evaluates how much sensitive attribute information is leaked in the embeddings.
    Trains a small MLP classifier to predict the intersectional group from embeddings.
    A prediction is correct only if all sensitive attributes are correct.
    
    Args:
        silent: If True, suppresses printing
        min_group_size: Minimum group size to include in evaluation (default: MIN_GROUP_SIZE)
    
    Returns:
        Dictionary with 'accuracy' and 'macro_f1' scores
    """
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    
    if min_group_size is None:
        min_group_size = MIN_GROUP_SIZE
    
    from sklearn.preprocessing import LabelEncoder
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score
    import torch.nn as nn
    import torch.optim as optim
    
    # Create combined group labels from all sensitive attributes
    # Each unique combination of attribute values becomes a distinct group
    group_labels = []
    for idx in range(len(sensitive_attributes)):
        # Create a string representation of all attribute values for this sample
        group_str = '_'.join([str(val) for val in sensitive_attributes.iloc[idx].values])
        group_labels.append(group_str)
    
    # Filter out groups with size < min_group_size
    group_series = pd.Series(group_labels)
    group_sizes = group_series.value_counts()
    valid_groups = group_sizes[group_sizes >= min_group_size].index.tolist()
    valid_mask = group_series.isin(valid_groups)
    
    initial_count = len(group_labels)
    embeddings = embeddings[valid_mask]
    group_labels = [group_labels[i] for i in range(len(group_labels)) if valid_mask[i]]
    filtered_count = len(group_labels)
    
    if not silent and initial_count != filtered_count:
        print(f"  - Filtered out {initial_count - filtered_count} samples from groups with size < {min_group_size}")
    
    # Encode group labels as integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(group_labels)
    num_classes = len(label_encoder.classes_)
    
    # Split into train/test for the leakage classifier
    # Use stratification to ensure all classes are represented in both splits
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)
    
    # Define a small MLP
    class LeakageMLP(nn.Module):
        def __init__(self, input_dim, num_classes, hidden_dim=128):
            super(LeakageMLP, self).__init__()
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, num_classes)
            )
        
        def forward(self, x):
            return self.network(x)
    
    # Train the MLP
    input_dim = embeddings.shape[1]
    model = LeakageMLP(input_dim, num_classes)
    
    # Compute class weights to handle imbalance (inverse frequency weighting)
    class_counts = np.bincount(y_train)
    class_weights = 1.0 / (class_counts + 1e-6)  # Add small epsilon to avoid division by zero
    class_weights = class_weights / class_weights.sum() * num_classes  # Normalize
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32)
    
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    
    # Training loop
    n_epochs = 50
    batch_size = 64
    n_samples = len(X_train_tensor)
    
    model.train()
    for epoch in range(n_epochs):
        for i in range(0, n_samples, batch_size):
            batch_end = min(i + batch_size, n_samples)
            X_batch = X_train_tensor[i:batch_end]
            y_batch = y_train_tensor[i:batch_end]
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
    
    # Evaluate on test set
    model.eval()
    with torch.no_grad():
        test_outputs = model(X_test_tensor)
        test_predictions = torch.argmax(test_outputs, dim=1).cpu().numpy()
        test_labels = y_test_tensor.cpu().numpy()
    
    # Calculate accuracy and macro F1
    accuracy = accuracy_score(test_labels, test_predictions)
    macro_f1 = f1_score(test_labels, test_predictions, average='macro', zero_division=0)
    
    if not silent:
        print(f"  - Group leakage accuracy: {accuracy:.4f}")
        print(f"  - Group leakage macro F1: {macro_f1:.4f}")
    
    return {'accuracy': accuracy, 'macro_f1': macro_f1}

class MINENetwork(nn.Module):
    """
    Mutual Information Neural Estimation (MINE) network.
    Estimates MI between embeddings and sensitive attributes using the Donsker-Varadhan representation.
    """
    def __init__(self, embedding_dim, num_groups, hidden_dim=256):
        super(MINENetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(embedding_dim + num_groups, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x, y):
        # Concatenate embeddings and one-hot encoded group labels
        joint = torch.cat([x, y], dim=1)
        return self.network(joint)

def evaluate_mutual_information(embeddings, sensitive_attributes, use_mine=True, n_mine_epochs=100, mine_lr=1e-3, silent=False):
    """
    Evaluates the mutual information between embeddings and sensitive attributes.
    A lower MI indicates better fairness.
    
    Args:
        use_mine: If True, uses MINE (neural estimation). If False, uses histogram-based estimation.
        n_mine_epochs: Number of training epochs for MINE
        mine_lr: Learning rate for MINE optimizer
        silent: If True, suppresses printing
    """
    if not silent:
        print("\n--- Evaluating Mutual Information ---")
        if use_mine:
            print("Using MINE (Mutual Information Neural Estimation)")
        else:
            print("Using histogram-based estimation")
    
    # Create the intersectional group key
    if 'group_key' not in sensitive_attributes.columns:
        # Use all available sensitive attributes to create intersectional group key
        # Handle duplicate columns (e.g., both 'RACE' and 'race') by prioritizing uppercase
        attr_cols = [col for col in sensitive_attributes.columns if col != 'group_key']
        # Remove lowercase duplicates if uppercase version exists
        seen_upper = set()
        filtered_cols = []
        for col in attr_cols:
            upper_col = col.upper()
            if upper_col in seen_upper:
                continue  # Skip if we already have uppercase version
            if col.isupper():
                seen_upper.add(col)
                filtered_cols.append(col)
            elif upper_col not in seen_upper:
                # Keep lowercase only if uppercase version doesn't exist
                filtered_cols.append(col)
                seen_upper.add(upper_col)
        
        if len(filtered_cols) >= 2:
            sensitive_attributes['group_key'] = sensitive_attributes[filtered_cols].astype(str).agg('_'.join, axis=1)
        else:
            print("Error: Need at least two sensitive attribute columns to define groups.")
            return
    
    # Use all groups with size >= MIN_GROUP_SIZE instead of hard-coded groups
    group_sizes = sensitive_attributes['group_key'].value_counts()
    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index.tolist()
    mask = sensitive_attributes['group_key'].isin(valid_groups)
    
    if torch.is_tensor(embeddings):
        embeddings_np = embeddings.detach().cpu().numpy()
    else:
        embeddings_np = np.array(embeddings)
    
    filtered_embeddings = embeddings_np[mask.values]
    filtered_attributes = sensitive_attributes[mask]

    if len(filtered_attributes) < 2: # Need at least 2 samples for MI calculation
        print("  - Not enough samples from the specified intersectional groups found to evaluate mutual information.")
        return

    # Prepare data
    labels = filtered_attributes['group_key'].astype('category').cat.codes.values
    num_groups = len(np.unique(labels))
    
    if use_mine:
        # --- MINE Implementation ---
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Convert to tensors
        X = torch.tensor(filtered_embeddings, dtype=torch.float32).to(device)
        y = torch.tensor(labels, dtype=torch.long).to(device)
        
        # One-hot encode labels
        y_onehot = F.one_hot(y, num_classes=num_groups).float()
        
        # Initialize MINE network
        embedding_dim = filtered_embeddings.shape[1]
        mine_net = MINENetwork(embedding_dim, num_groups).to(device)
        mine_optimizer = optim.Adam(mine_net.parameters(), lr=mine_lr)
        
        # Training loop for MINE
        mi_estimates = []
        for epoch in range(n_mine_epochs):
            # Shuffle to create negative samples
            idx = torch.randperm(X.size(0))
            X_shuffled = X[idx]
            
            # Forward pass
            joint = mine_net(X, y_onehot)
            marginal = mine_net(X_shuffled, y_onehot)
            
            # MINE loss: -E[T(x,y)] + log(E[exp(T(x',y))])
            # Using log-sum-exp trick for numerical stability
            mi_estimate = joint.mean() - torch.log(marginal.exp().mean() + 1e-8)
            mine_loss = -mi_estimate
            
            # Backward pass
            mine_optimizer.zero_grad()
            mine_loss.backward()
            mine_optimizer.step()
            
            mi_estimates.append(mi_estimate.item())
        
        # Use moving average of last 20 epochs as final estimate
        final_mi = np.mean(mi_estimates[-20:]) if len(mi_estimates) >= 20 else mi_estimates[-1]
        if not silent:
            print(f"  - Mutual Information (MINE): {final_mi:.4f}")
            print(f"  - MI trend (last 5 epochs): {mi_estimates[-5:]}")
        return final_mi
    else:
        # --- Original Histogram-based Implementation ---
        total_mi = 0.0
        n_samples = filtered_embeddings.shape[0]
        n_bins = int(np.sqrt(n_samples / 5)) # Sturges' rule variation for binning
        n_bins = max(n_bins, 2) # Ensure at least 2 bins
        
        for i in range(filtered_embeddings.shape[1]): # Iterate over each embedding dimension
            # Discretize the continuous embedding feature
            embedding_feature = filtered_embeddings[:, i]
            bins = np.linspace(embedding_feature.min(), embedding_feature.max(), n_bins + 1)
            binned_feature = np.digitize(embedding_feature, bins) - 1

            # Calculate joint and marginal probabilities
            joint_prob = np.histogram2d(binned_feature, labels, bins=[n_bins, len(np.unique(labels))])[0] / n_samples
            marginal_feature = np.sum(joint_prob, axis=1)
            marginal_labels = np.sum(joint_prob, axis=0)

            # Calculate mutual information for this feature
            mi = 0.0
            for row in range(n_bins):
                for col in range(len(np.unique(labels))):
                    if joint_prob[row, col] > 0 and marginal_feature[row] > 0 and marginal_labels[col] > 0:
                        mi += joint_prob[row, col] * np.log(joint_prob[row, col] / (marginal_feature[row] * marginal_labels[col]))
        
        total_mi += mi
    
    print(f"  - Total Mutual Information between embeddings and intersectional group: {total_mi:.6f}")

def evaluate_model(embedding_model, prediction_head, embedding_projection, codebook,
                  X_test_tensor, y_test_tensor, X_test_sensitive,
                  filtered_X_test_tensor, true_group_indices, filtered_X_test_sensitive,
                  return_metrics=False):
    """
    Evaluates the model on test data with fairness metrics.
    
    Args:
        return_metrics: If True, returns a dictionary of all metrics instead of printing
    
    Returns:
        If return_metrics=True: dict containing all metrics
        Otherwise: accuracy (for backward compatibility)
    """
    embedding_model.eval()
    prediction_head.eval()
    if embedding_projection:
        embedding_projection.eval()
    if codebook:
        codebook.eval()
    print_debug = True #not return_metrics

    # 1. Income Prediction Accuracy and F1
    with torch.no_grad():
        embeddings = embedding_model(X_test_tensor)
        y_pred = prediction_head(embeddings)
        predicted = (y_pred > 0.5).float()
        accuracy = (predicted == y_test_tensor).float().mean()
        
        # Calculate F1 score
        y_test_np = y_test_tensor.cpu().numpy().flatten()
        predicted_np = predicted.cpu().numpy().flatten()
        f1 = f1_score(y_test_np, predicted_np, zero_division=0)
        
        if not return_metrics:
            print(f"Accuracy on the test set: {accuracy.item():.4f}")
            print(f"F1 on the test set: {f1:.4f}")
    
    # --- Fairness Evaluations ---
    X_test_sensitive_eval = X_test_sensitive.copy()
    
    # Capture leakage scores
    leakage_scores = evaluate_information_leakage(embeddings, X_test_sensitive_eval.copy(), silent=return_metrics)
    leakage_acc = leakage_scores['accuracy'] if leakage_scores else 0.0
    leakage_f1 = leakage_scores['macro_f1'] if leakage_scores else 0.0
    
    # Capture DP and EO gaps
    gamma_subgroup_fairness = calculate_group_fairness_metrics(y_test_tensor, predicted, X_test_sensitive_eval.copy(), silent=return_metrics)
    
    # Capture mutual information
    mi = evaluate_mutual_information(embeddings, X_test_sensitive_eval.copy(), silent=return_metrics)
    
    # --- Silhouette Score ---
    if not return_metrics:
        print("\n--- Calculating Silhouette Score ---")
    # Create group labels for silhouette score
    if 'group_key' not in X_test_sensitive_eval.columns:
        attr_cols = [col for col in X_test_sensitive_eval.columns if col != 'group_key']
        seen_upper = set()
        filtered_cols = []
        for col in attr_cols:
            upper_col = col.upper()
            if upper_col in seen_upper:
                continue
            if col.isupper():
                seen_upper.add(col)
                filtered_cols.append(col)
            elif upper_col not in seen_upper:
                filtered_cols.append(col)
                seen_upper.add(upper_col)
        
        if len(filtered_cols) >= 2:
            X_test_sensitive_eval['group_key'] = X_test_sensitive_eval[filtered_cols].astype(str).agg('_'.join, axis=1)
        else:
            silhouette_score_val = 0.0
            if not return_metrics:
                print(f"  - Silhouette Score: {silhouette_score_val:.4f}")
    
    # Calculate silhouette score (after ensuring group_key exists)
    if 'group_key' in X_test_sensitive_eval.columns:
        from sklearn.metrics import silhouette_score
        embeddings_np = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
        group_labels = X_test_sensitive_eval['group_key'].values
        
        # Only calculate if we have at least 2 unique groups and enough samples
        unique_groups = len(np.unique(group_labels))
        if unique_groups >= 2 and len(group_labels) > 1:
            try:
                silhouette_score_val = silhouette_score(embeddings_np, group_labels)
            except:
                silhouette_score_val = 0.0
        else:
            silhouette_score_val = 0.0
        
        if not return_metrics:
            print(f"  - Silhouette Score: {silhouette_score_val:.4f}")
    else:
        silhouette_score_val = 0.0
        if not return_metrics:
            print(f"  - Silhouette Score: {silhouette_score_val:.4f}")
    
    # Create group_key dynamically based on available attributes
    if 'group_key' not in X_test_sensitive_eval.columns:
        # Use all available sensitive attributes to create intersectional group key
        # Handle duplicate columns (e.g., both 'RACE' and 'race') by prioritizing uppercase
        attr_cols = [col for col in X_test_sensitive_eval.columns if col != 'group_key']
        # Remove lowercase duplicates if uppercase version exists
        seen_upper = set()
        filtered_cols = []
        for col in attr_cols:
            upper_col = col.upper()
            if upper_col in seen_upper:
                continue  # Skip if we already have uppercase version
            if col.isupper():
                seen_upper.add(col)
                filtered_cols.append(col)
            elif upper_col not in seen_upper:
                # Keep lowercase only if uppercase version doesn't exist
                filtered_cols.append(col)
                seen_upper.add(upper_col)
        
        if len(filtered_cols) >= 2:
            X_test_sensitive_eval['group_key'] = X_test_sensitive_eval[filtered_cols].astype(str).agg('_'.join, axis=1)
        else:
            print("Error: Need at least two sensitive attribute columns to define groups.")
            if return_metrics:
                return None
            return
    
    # Use all groups with size >= MIN_GROUP_SIZE instead of hard-coded groups
    group_sizes = X_test_sensitive_eval['group_key'].value_counts()
    valid_groups = group_sizes[group_sizes >= MIN_GROUP_SIZE].index.tolist()
    
    group_accuracies = {}
    for group in valid_groups:
        group_mask = X_test_sensitive_eval['group_key'] == group
        # Convert pandas boolean mask to tensor mask for proper indexing
        group_mask_tensor = torch.tensor(group_mask.values, dtype=torch.bool, device=predicted.device)
        group_pred = predicted[group_mask_tensor]
        group_true = y_test_tensor[group_mask_tensor]
        group_acc = (group_pred == group_true).float().mean()
        group_accuracies[group] = group_acc.item()
        if not return_metrics:
            print(f"  {group}: {group_acc.item():.4f}")
    
    if group_accuracies:
        group_acc_values = list(group_accuracies.values())
        group_acc_variance = np.var(group_acc_values)
        if not return_metrics:
            print(f"Variance of Group Accuracy: {group_acc_variance:.6f}")
    else:
        group_acc_variance = 0.0
    
    # --- Trace of Between-Group Scatter Matrix ---
    if not return_metrics:
        print("\n--- Calculating Trace of Between-Group Scatter Matrix ---")
    
    # Calculate for original embeddings
    trace_scatter_original = calculate_trace_scatter_matrix(embeddings, X_test_sensitive_eval)
    if not return_metrics:
        print(f"  - Trace of scatter matrix (original embeddings): {trace_scatter_original:.6f}")
    if codebook and embedding_projection and filtered_X_test_tensor is not None and filtered_X_test_tensor.numel() > 0:
        with torch.no_grad():
            # Use filtered embeddings to match the size of true_group_indices
            filtered_embeddings = embedding_model(filtered_X_test_tensor)
            projected_embeddings = embedding_projection(filtered_embeddings)
            trace_scatter = calculate_trace_scatter_matrix(projected_embeddings, filtered_X_test_sensitive)
            if not return_metrics:
                print(f"Trace of Between-Group Scatter Matrix (projected embeddings): {trace_scatter:.4f}")
    
    # --- Hilbert-Schmidt Independence Criterion (HSIC) --- sum of marginal HSICs
    if not return_metrics:
        print("\n--- Calculating Hilbert-Schmidt Independence Criterion (HSIC) ---")
    
    # Get attributes to compute HSIC for (using same smart duplicate filtering)
    attr_cols = [col for col in X_test_sensitive_eval.columns if col != 'group_key']
    seen_upper = set()
    attributes = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue  # Skip if we already have uppercase version
        if col.isupper():
            seen_upper.add(col)
            attributes.append(col)
        elif upper_col not in seen_upper:
            # Keep lowercase only if uppercase version doesn't exist
            attributes.append(col)
            seen_upper.add(upper_col)
    
    # Calculate HSIC for original embeddings (sum of marginal HSICs)
    hsic_original = 0.0
    for attr in attributes:
        hsic_attr = calculate_hsic(embeddings, X_test_sensitive_eval, attr)
        hsic_original += hsic_attr
        if not return_metrics:
            print(f"  - HSIC for {attr}: {hsic_attr:.6f}")
    if not return_metrics:
        print(f"  - Total HSIC (original embeddings): {hsic_original:.6f}")
    
    if codebook and embedding_projection and filtered_X_test_tensor is not None and filtered_X_test_tensor.numel() > 0:
        with torch.no_grad():
            # Use filtered embeddings to match the size of true_group_indices
            filtered_embeddings = embedding_model(filtered_X_test_tensor)
            projected_embeddings = embedding_projection(filtered_embeddings)
            
            # Calculate HSIC for projected embeddings (sum of marginal HSICs)
            hsic_projected = 0.0
            for attr in attributes:
                hsic_attr = calculate_hsic(projected_embeddings, filtered_X_test_sensitive, attr)
                hsic_projected += hsic_attr
                if not return_metrics:
                    print(f"  - HSIC for {attr}: {hsic_attr:.6f}")
            if not return_metrics:
                print(f"  - Total HSIC (projected embeddings): {hsic_projected:.6f}")
    
    # Return metrics dictionary if requested
    if return_metrics:
        metrics = {
            'accuracy': accuracy.item(),
            'f1': f1,
            'leakage_f1': leakage_f1,
            'gamma_subgroup_fairness': gamma_subgroup_fairness,
            'mutual_information': mi if mi is not None else 0.0,
            'silhouette_score': silhouette_score_val if 'silhouette_score_val' in locals() else 0.0,
            'group_accuracy_variance': group_acc_variance,
            'trace_scatter_original': trace_scatter_original,
            'hsic_original': hsic_original
        }
        return metrics
    
    return accuracy.item()

def print_comparison_summary(results_dict):
    """
    Prints a summary table comparing ROAR and baseline methods.
    
    Args:
        results_dict: Dictionary with method names as keys and metrics dictionaries as values
                     Each metrics dict should contain: accuracy, f1, leakage_f1,
                     gamma_subgroup_fairness, mutual_information, silhouette_score,
                     group_accuracy_variance,
                     trace_scatter_original, hsic_original
    """
    print("\n" + "="*120)
    print("COMPARISON SUMMARY")
    print("="*120)
    
    # Define column headers
    headers = [
        "Method",
        "Accuracy",
        "F1",
        "Leakage F1",
        "γ-Subgroup",
        "MI",
        "Silhouette",
        "Acc Var",
        "Trace Scatter",
        "HSIC"
    ]
    
    # Format each row
    rows = []
    for method_name, metrics in results_dict.items():
        row = [
            method_name,
            f"{metrics.get('accuracy', 0):.4f}",
            f"{metrics.get('f1', 0):.4f}",
            f"{metrics.get('leakage_f1', 0):.4f}",
            f"{metrics.get('gamma_subgroup_fairness', 0):.4f}",
            f"{metrics.get('mutual_information', 0):.4f}",
            f"{metrics.get('silhouette_score', 0):.4f}",
            f"{metrics.get('group_accuracy_variance', 0):.6f}",
            f"{metrics.get('trace_scatter_original', 0):.6f}",
            f"{metrics.get('hsic_original', 0):.6f}"
        ]
        rows.append(row)
    
    # Calculate column widths
    col_widths = []
    for i, header in enumerate(headers):
        max_width = len(header)
        for row in rows:
            max_width = max(max_width, len(row[i]))
        col_widths.append(max_width + 2)
    
    # Print header
    header_line = ""
    for i, header in enumerate(headers):
        header_line += header.ljust(col_widths[i])
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in rows:
        row_line = ""
        for i, cell in enumerate(row):
            row_line += cell.ljust(col_widths[i])
        print(row_line)
    
    print("="*120)

# ==================== Shared Experiment Functions ====================

def compute_all_group_accuracies(embedding_model, prediction_head, X_test_tensor, y_test_tensor, 
                                  sensitive_test, group_definition, attribute_values, device):
    """
    Computes accuracy for each group in the test set.
    
    Returns:
        dict: Mapping from group_key to accuracy
    """
    # Create group_key for test data
    sensitive_test_copy = sensitive_test.copy()
    attr_cols = [col for col in sensitive_test_copy.columns if col != 'group_key']
    seen_upper = set()
    filtered_cols = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue
        if col.isupper():
            seen_upper.add(col)
            filtered_cols.append(col)
        elif upper_col not in seen_upper:
            filtered_cols.append(col)
            seen_upper.add(upper_col)
    sensitive_test_copy['group_key'] = sensitive_test_copy[filtered_cols].astype(str).agg('_'.join, axis=1)
    
    # Get unique groups
    unique_groups = sensitive_test_copy['group_key'].unique()
    group_accuracies = {}
    
    with torch.no_grad():
        embeddings = embedding_model(X_test_tensor)
        outputs = prediction_head(embeddings)
        predictions = (outputs > 0.5).float()
    
    for group in unique_groups:
        group_mask = sensitive_test_copy['group_key'] == group
        if group_mask.sum() == 0:
            continue
        # Convert pandas boolean mask to tensor mask for proper indexing
        group_mask_tensor = torch.tensor(group_mask.values, dtype=torch.bool, device=predictions.device)
        group_predictions = predictions[group_mask_tensor]
        group_labels = y_test_tensor[group_mask_tensor]
        accuracy = (group_predictions == group_labels).float().mean().item()
        group_accuracies[group] = accuracy
    
    return group_accuracies

def compute_all_group_embeddings(embedding_model, X_test_tensor, sensitive_test, group_definition, device):
    """
    Computes mean embedding for each group in the test set.
    
    Returns:
        dict: Mapping from group_key to mean embedding
    """
    # Create group_key for test data
    sensitive_test_copy = sensitive_test.copy()
    attr_cols = [col for col in sensitive_test_copy.columns if col != 'group_key']
    seen_upper = set()
    filtered_cols = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue
        if col.isupper():
            seen_upper.add(col)
            filtered_cols.append(col)
        elif upper_col not in seen_upper:
            filtered_cols.append(col)
            seen_upper.add(upper_col)
    sensitive_test_copy['group_key'] = sensitive_test_copy[filtered_cols].astype(str).agg('_'.join, axis=1)
    
    # Get unique groups
    unique_groups = sensitive_test_copy['group_key'].unique()
    group_embeddings = {}
    
    with torch.no_grad():
        embeddings = embedding_model(X_test_tensor)
    
    for group in unique_groups:
        group_mask = sensitive_test_copy['group_key'] == group
        if group_mask.sum() == 0:
            continue
        # Convert pandas boolean mask to tensor mask for proper indexing
        group_mask_tensor = torch.tensor(group_mask.values, dtype=torch.bool, device=embeddings.device)
        group_embs = embeddings[group_mask_tensor]
        group_mean_emb = group_embs.mean(dim=0)
        group_embeddings[group] = group_mean_emb
    
    return group_embeddings

def compute_all_group_f1_scores(embedding_model, prediction_head, X_test_tensor, y_test_tensor, 
                                 sensitive_test, group_definition, attribute_values, device):
    """
    Computes F1 score for each group in the test set.
    
    Returns:
        dict: Mapping from group_key to F1 score
    """
    # Create group_key for test data
    sensitive_test_copy = sensitive_test.copy()
    attr_cols = [col for col in sensitive_test_copy.columns if col != 'group_key']
    seen_upper = set()
    filtered_cols = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue
        if col.isupper():
            seen_upper.add(col)
            filtered_cols.append(col)
        elif upper_col not in seen_upper:
            filtered_cols.append(col)
            seen_upper.add(upper_col)
    sensitive_test_copy['group_key'] = sensitive_test_copy[filtered_cols].astype(str).agg('_'.join, axis=1)
    
    # Get unique groups
    unique_groups = sensitive_test_copy['group_key'].unique()
    group_f1_scores = {}
    
    with torch.no_grad():
        embeddings = embedding_model(X_test_tensor)
        outputs = prediction_head(embeddings)
        predictions = (outputs > 0.5).float()
    
    for group in unique_groups:
        group_mask = sensitive_test_copy['group_key'] == group
        if group_mask.sum() == 0:
            continue
        # Convert pandas boolean mask to tensor mask for proper indexing
        group_mask_tensor = torch.tensor(group_mask.values, dtype=torch.bool, device=predictions.device)
        group_predictions = predictions[group_mask_tensor].cpu().numpy()
        group_labels = y_test_tensor[group_mask_tensor].cpu().numpy()
        f1 = f1_score(group_labels, group_predictions, zero_division=0)
        group_f1_scores[group] = f1
    
    return group_f1_scores

def compute_fairness_metrics(embedding_model, prediction_head, X_test_tensor, y_test_tensor, 
                             sensitive_test, target_group, group_definition, attribute_values, device):
    """
    Computes fairness metrics for a given model.
    
    Args:
        embedding_model: The embedding model
        prediction_head: The prediction head
        X_test_tensor: Test features tensor
        y_test_tensor: Test labels tensor
        sensitive_test: Test sensitive attributes DataFrame
        target_group: The target group key
        group_definition: List of attributes for group construction
        attribute_values: Dictionary of attribute values
        device: Device to use
        
    Returns:
        dict: Dictionary with accuracy_distance, f1_distance, and embedding_distance_sq
    """
    group_accuracies = compute_all_group_accuracies(embedding_model, prediction_head, 
                                                     X_test_tensor, y_test_tensor, sensitive_test, 
                                                     group_definition, attribute_values, device)
    group_f1_scores = compute_all_group_f1_scores(embedding_model, prediction_head,
                                                  X_test_tensor, y_test_tensor, sensitive_test,
                                                  group_definition, attribute_values, device)
    group_embeddings = compute_all_group_embeddings(embedding_model, X_test_tensor, 
                                                      sensitive_test, group_definition, device)
    
    metrics = {}
    
    # Metric 1: Distance between target group accuracy and mean accuracy of all groups
    if target_group in group_accuracies:
        target_accuracy = group_accuracies[target_group]
        mean_accuracy = np.mean(list(group_accuracies.values()))
        accuracy_distance = abs(target_accuracy - mean_accuracy)
        metrics['accuracy_distance'] = accuracy_distance
    else:
        metrics['accuracy_distance'] = None
        print(f"Warning: Target group {target_group} not found in test set groups.")
    
    # Metric 2: Distance between target group F1 and mean F1 of all groups
    if target_group in group_f1_scores:
        target_f1 = group_f1_scores[target_group]
        mean_f1 = np.mean(list(group_f1_scores.values()))
        f1_distance = abs(target_f1 - mean_f1)
        metrics['f1_distance'] = f1_distance
    else:
        metrics['f1_distance'] = None
        print(f"Warning: Target group {target_group} not found in test set groups.")
    
    # Metric 3: Distance squared between target group embedding and mean embedding over all groups
    if target_group in group_embeddings:
        target_embedding = group_embeddings[target_group]
        all_embeddings = torch.stack(list(group_embeddings.values()))
        mean_embedding = all_embeddings.mean(dim=0)
        embedding_distance_sq = torch.sum((target_embedding - mean_embedding) ** 2).item()
        metrics['embedding_distance_sq'] = embedding_distance_sq
    else:
        metrics['embedding_distance_sq'] = None
        print(f"Warning: Target group {target_group} not found in test set embeddings.")
    
    return metrics

def run_group_inclusion_experiment(X_train, y_train, sensitive_train, X_test, y_test, sensitive_test,
                                    target_group, inclusion_percentages, attribute_values, device,
                                    group_definition, train_head=True, n_roar_iterations=1,
                                    use_group_inclusion_split=True, print_comparison_summary_table=True):
    """
    Runs group inclusion experiment with all methods at varying inclusion percentages.
    
    Args:
        X_train: Original training features
        y_train: Original training labels
        sensitive_train: Original training sensitive attributes
        X_test: Test features
        y_test: Test labels
        sensitive_test: Test sensitive attributes
        target_group: The group key to vary inclusion for
        inclusion_percentages: List of inclusion percentages to test (0.0 to 1.0)
        attribute_values: Dictionary of attribute values for codebook
        device: Device to use for training
        group_definition: List of attributes for group construction (e.g., ['RACE', 'SEX', 'EDUCATION'])
        train_head: Whether to train the prediction head
        n_roar_iterations: Number of iterations for ROAR's two phases (default: 1)
        use_group_inclusion_split: Whether to use create_group_inclusion_split to modify training data (default: True)
        print_comparison_summary_table: Whether to print comparison summary table at the end (default: True)
    
    Returns:
        results: Dictionary containing results for each method and percentage
        fairness_metrics: Dictionary containing fairness metrics (accuracy_distance, embedding_distance_sq)
    """
    print("\n" + "="*60)
    print(f"GROUP INCLUSION EXPERIMENT - Target Group: {target_group}")
    print("="*60)
    
    # Initialize comparison results dictionary for full test set evaluation
    comparison_results = {}
    
    results = {
        'ERM': {'f1': {}},
        'GroupDRO': {'f1': {}},
        'LAFTR': {'f1': {}},
        'ROAD': {'f1': {}},
        'AdversarialLearning': {'f1': {}},
        'HSIC': {'f1': {}},
        'INLP': {'f1': {}},
        'RLACE': {'f1': {}},
        'ROAR': {'f1': {}}
    }
    
    fairness_metrics = {
        'ERM': {'f1_distance': {}},
        'GroupDRO': {'f1_distance': {}},
        'LAFTR': {'f1_distance': {}},
        'ROAD': {'f1_distance': {}},
        'AdversarialLearning': { 'f1_distance': {}},
        'HSIC': {'f1_distance': {}},
        'INLP': { 'f1_distance': {}},
        'RLACE': {'f1_distance': {}},
        'ROAR': {'f1_distance': {}}
    }
    
    input_dim = X_train.shape[1]
    criterion = nn.BCELoss()
    
    # Create full test tensor for computing all-group metrics
    # Handle both pandas DataFrames and numpy arrays
    if hasattr(X_test, 'values'):
        X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
    else:
        X_test_tensor = torch.tensor(X_test.astype(np.float32), dtype=torch.float32).to(device)
    
    if hasattr(y_test, 'values'):
        y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1).to(device)
    else:
        y_test_tensor = torch.tensor(y_test.astype(np.float32), dtype=torch.float32).reshape(-1, 1).to(device)
    
    # Create group_key for test data
    sensitive_test_copy = sensitive_test.copy()
    attr_cols = [col for col in sensitive_test_copy.columns if col != 'group_key']
    seen_upper = set()
    filtered_cols = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue
        if col.isupper():
            seen_upper.add(col)
            filtered_cols.append(col)
        elif upper_col not in seen_upper:
            filtered_cols.append(col)
            seen_upper.add(upper_col)
    sensitive_test_copy['group_key'] = sensitive_test_copy[filtered_cols].astype(str).agg('_'.join, axis=1)
    
    for inclusion_pct in inclusion_percentages:
        print(f"\n\n--- Inclusion Percentage: {inclusion_pct * 100:.0f}% ---")
        
        if use_group_inclusion_split:
            # Create modified training data
            X_train_mod, y_train_mod, sensitive_train_mod, train_group_mask = create_group_inclusion_split(
                X_train, y_train, sensitive_train, target_group, inclusion_pct
            )
            
            print(f"Original training size: {len(X_train)}")
            print(f"Modified training size: {len(X_train_mod)}")
            print(f"Target group samples in training: {train_group_mask.sum()} (original: {train_group_mask.sum()})")
            print(f"Target group samples in modified training: {int(train_group_mask.sum() * inclusion_pct)}")
        else:
            # Use original training data
            X_train_mod = X_train
            y_train_mod = y_train
            sensitive_train_mod = sensitive_train
            print(f"Training size: {len(X_train)}")
        
        # Convert to tensors
        # Handle both pandas DataFrames and numpy arrays
        if hasattr(X_train_mod, 'values'):
            X_train_tensor = torch.tensor(X_train_mod.values, dtype=torch.float32).to(device)
        else:
            X_train_tensor = torch.tensor(X_train_mod.astype(np.float32), dtype=torch.float32).to(device)
        
        if hasattr(y_train_mod, 'values'):
            y_train_tensor = torch.tensor(y_train_mod.values, dtype=torch.float32).reshape(-1, 1).to(device)
        else:
            y_train_tensor = torch.tensor(y_train_mod.astype(np.float32), dtype=torch.float32).reshape(-1, 1).to(device)
        
        # Reconstruct attribute indices for modified training data
        _, _, attribute_indices_train_mod, _, valid_indices = construct_intersectional_groups(
            sensitive_train_mod, group_definition, attribute_values
        )
        
        # Filter training data to only include valid samples
        X_train_tensor = X_train_tensor[valid_indices]
        y_train_tensor = y_train_tensor[valid_indices]
        
        # Create DataLoader with group-aware oversampling for rare groups
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor, torch.tensor(attribute_indices_train_mod, dtype=torch.long))
        
        # Group-aware oversampling: compute group sizes and create weighted sampler
        # Convert attribute_indices to group keys for counting
        group_keys = ['_'.join(map(str, row)) for row in attribute_indices_train_mod]
        from collections import Counter
        group_counts = Counter(group_keys)
        
        # Compute weights inversely proportional to group size
        # w_i = 1 / (count_i + epsilon) to avoid division by zero
        epsilon = 1e-6
        sample_weights = torch.tensor([1.0 / (group_counts[g] + epsilon) for g in group_keys], dtype=torch.float32)
        
        # Normalize weights to sum to 1
        sample_weights = sample_weights / sample_weights.sum()
        
        # Create weighted sampler
        from torch.utils.data import WeightedRandomSampler
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        
        # Use sampler instead of shuffle=True
        train_loader = DataLoader(train_dataset, batch_size=128, sampler=sampler)
        
        # Test on held-out target group samples
        test_group_mask = sensitive_test_copy['group_key'] == target_group
        X_test_heldout = X_test[test_group_mask.values]
        y_test_heldout = y_test[test_group_mask.values]
        sensitive_test_heldout = sensitive_test.iloc[test_group_mask.values].reset_index(drop=True)
        
        print(f"Test samples (held-out target group): {len(X_test_heldout)}")
        
        if len(X_test_heldout) == 0:
            print(f"Warning: No test samples for target group {target_group}. Skipping this percentage.")
            continue
        
        # Handle both pandas DataFrames and numpy arrays
        if hasattr(X_test_heldout, 'values'):
            X_test_heldout_tensor = torch.tensor(X_test_heldout.values, dtype=torch.float32).to(device)
        else:
            X_test_heldout_tensor = torch.tensor(X_test_heldout.astype(np.float32), dtype=torch.float32).to(device)
        
        if hasattr(y_test_heldout, 'values'):
            y_test_heldout_tensor = torch.tensor(y_test_heldout.values, dtype=torch.float32).reshape(-1, 1).to(device)
        else:
            y_test_heldout_tensor = torch.tensor(y_test_heldout.astype(np.float32), dtype=torch.float32).reshape(-1, 1).to(device)
        
        # --- ROAR (with multiple iterations) ---
        print(f"\n--- ROAR ({n_roar_iterations} iteration(s)) ---")
        roar_embedding_model = EmbeddingModel(input_dim).to(device)
        roar_prediction_head = PredictionHead().to(device)
        roar_embedding_projection = EmbeddingProjection().to(device)
        # Convert attribute_values dict to list of lists for codebook
        roar_attribute_values_list = [attribute_values[attr] for attr in group_definition]
        roar_codebook = Codebook(roar_attribute_values_list, PROJECTION_DIM).to(device)
        
        for iteration in range(n_roar_iterations):
            print(f"\n--- ROAR Iteration {iteration + 1}/{n_roar_iterations} ---")
            
            # Phase 1: Metric Learning
            train_embedding_model = (iteration == 0) #only train embedding model in first iteration
            for param in roar_embedding_model.parameters(): param.requires_grad = train_embedding_model
            for param in roar_prediction_head.parameters(): param.requires_grad = True if train_head else False
            for param in roar_embedding_projection.parameters(): param.requires_grad = True
            for param in roar_codebook.parameters(): param.requires_grad = True
            param_groups = [
                {"params": roar_embedding_model.parameters(), "lr": 1e-3} if train_embedding_model else None, #1e-4
                {"params": roar_prediction_head.parameters(), "lr": 1e-3} if train_head else None,
                {"params": roar_embedding_projection.parameters(), "lr": 1e-3},
                {"params": roar_codebook.parameters(), "lr": 5e-4},
            ]
            param_groups = [pg for pg in param_groups if pg is not None]
            roar_optimizer = optim.Adam(param_groups)
            
            n_epochs = 25 #if iteration == 0 else 5
            # Only train embedding model in first iteration
            train_embedding_model = (iteration == 0)
            train_phase1(roar_embedding_model, roar_prediction_head, roar_embedding_projection, roar_codebook,
                        train_loader, criterion, roar_optimizer, n_epochs=n_epochs, alpha=2.0, inter_group_margin=0.5,
                        train_embedding_model=train_embedding_model) #alpha=1.0
            
            # Collect metrics after first iteration (ROAR after Phase 1) - only for first inclusion percentage
            # if iteration == 0 and inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            #     print(f"\n--- Evaluation after Phase 1 (Iteration 1) ---")
            #     metrics_roar_phase1 = evaluate_model(roar_embedding_model, roar_prediction_head, roar_embedding_projection, roar_codebook,
            #                                         X_test_tensor, y_test_tensor, sensitive_test_copy,
            #                                         X_test_tensor, torch.empty(0), sensitive_test_copy,
            #                                         return_metrics=True)
            #     comparison_results['ROAR (Phase 1)'] = metrics_roar_phase1
            
            # Phase 2: Prediction Training
            for param in roar_embedding_model.parameters(): param.requires_grad = True
            for param in roar_prediction_head.parameters(): param.requires_grad = True if train_head else False
            for param in roar_embedding_projection.parameters(): param.requires_grad = False
            for param in roar_codebook.parameters(): param.requires_grad = False
            param_groups_iter_2 = [
                {"params": roar_embedding_model.parameters(), "lr": 1e-4},
                {"params": roar_prediction_head.parameters(), "lr": 1e-4} if train_head else None, #5e-4
            ]
            param_groups_iter_2 = [pg for pg in param_groups_iter_2 if pg is not None]
            roar_optimizer_iter_2 = optim.Adam(param_groups_iter_2)
            
            train_phase2(roar_embedding_model, roar_prediction_head, roar_embedding_projection, roar_codebook,
                        train_loader, criterion, roar_optimizer_iter_2, n_epochs=n_epochs, beta=2.0, #0.2
                        use_vae_disentanglement=True, input_dim=input_dim)
        
        # Compute accuracy and F1 on held-out target group
        with torch.no_grad():
            embeddings = roar_embedding_model(X_test_heldout_tensor)
            outputs = roar_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            results['ROAR']['f1'][inclusion_pct] = f1
            print(f"ROAR Accuracy on held-out group: F1: {f1:.4f}")
        
        # Compute fairness metrics for ROAR
        roar_metrics = compute_fairness_metrics(roar_embedding_model, roar_prediction_head, 
                                                X_test_tensor, y_test_tensor, sensitive_test,
                                                target_group, group_definition, attribute_values, device)
        fairness_metrics['ROAR']['f1_distance'][inclusion_pct] = roar_metrics['f1_distance']
        print(f"ROAR - F1 Distance: {roar_metrics['f1_distance']:.4f}")
        
        # Collect ROAR final metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- ROAR Final Evaluation ---")
            metrics_roar_final = evaluate_model(roar_embedding_model, roar_prediction_head, roar_embedding_projection, roar_codebook,
                                                X_test_tensor, y_test_tensor, sensitive_test_copy,
                                                X_test_tensor, torch.empty(0), sensitive_test_copy,
                                                return_metrics=True)
            comparison_results['ROAR (Final)'] = metrics_roar_final

        # --- ERM Baseline ---
        print("\n--- ERM Baseline ---")
        erm_embedding_model = EmbeddingModel(input_dim).to(device)
        erm_prediction_head = PredictionHead().to(device)
        erm_optimizer = optim.Adam(
            list(erm_embedding_model.parameters()) +
            list(erm_prediction_head.parameters() if train_head else []),
            lr=0.001
        )
        
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

        with torch.no_grad():
            embeddings = erm_embedding_model(X_test_heldout_tensor)
            outputs = erm_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            results['ERM']['f1'][inclusion_pct] = f1
            print(f"ERM Accuracy on held-out group: F1: {f1:.4f}")
        
        # Compute fairness metrics
        erm_metrics = compute_fairness_metrics(erm_embedding_model, erm_prediction_head, 
                                                X_test_tensor, y_test_tensor, sensitive_test,
                                                target_group, group_definition, attribute_values, device)

        fairness_metrics['ERM']['f1_distance'][inclusion_pct] = erm_metrics['f1_distance']

        print(f"ERM - F1 Distance: {erm_metrics['f1_distance']:.4f}")
        
        # Collect ERM metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- ERM Evaluation ---")
            metrics_erm = evaluate_model(erm_embedding_model, erm_prediction_head, None, None,
                                        X_test_tensor, y_test_tensor, sensitive_test_copy,
                                        X_test_tensor, torch.empty(0), sensitive_test_copy,
                                        return_metrics=True)
            comparison_results['ERM'] = metrics_erm
        
        # --- Group DRO Baseline ---
        print("\n--- Group DRO Baseline ---")
        group_dro_embedding_model = EmbeddingModel(input_dim).to(device)
        group_dro_prediction_head = PredictionHead().to(device)
        group_dro_optimizer = optim.Adam(
            list(group_dro_embedding_model.parameters()) +
            list(group_dro_prediction_head.parameters() if train_head else []),
            lr=0.001
        )
        
        train_group_dro(group_dro_embedding_model, group_dro_prediction_head, train_loader, 
                       criterion, group_dro_optimizer, n_epochs=20, eta=0.01)
        
        with torch.no_grad():
            embeddings = group_dro_embedding_model(X_test_heldout_tensor)
            outputs = group_dro_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            
            results['GroupDRO']['f1'][inclusion_pct] = f1
            print(f"Group DRO Accuracy on held-out group:  F1: {f1:.4f}")
        
        # Compute fairness metrics
        group_dro_metrics = compute_fairness_metrics(group_dro_embedding_model, group_dro_prediction_head, 
                                                     X_test_tensor, y_test_tensor, sensitive_test,
                                                     target_group, group_definition, attribute_values, device)
        
        fairness_metrics['GroupDRO']['f1_distance'][inclusion_pct] = group_dro_metrics['f1_distance']
        
        print(f"GroupDRO -  F1 Distance: {group_dro_metrics['f1_distance']:.4f}")
        
        # Collect GroupDRO metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- Group DRO Evaluation ---")
            metrics_group_dro = evaluate_model(group_dro_embedding_model, group_dro_prediction_head, None, None,
                                              X_test_tensor, y_test_tensor, sensitive_test_copy,
                                              X_test_tensor, torch.empty(0), sensitive_test_copy,
                                              return_metrics=True)
            comparison_results['Group DRO'] = metrics_group_dro
        
        # --- LAFTR Baseline ---
        print("\n--- LAFTR Baseline ---")
        laftr_embedding_model = EmbeddingModel(input_dim).to(device)
        laftr_task_head = PredictionHead().to(device)
        
        # Determine number of classes for each attribute in group_definition
        # Use first 2 attributes for adversaries
        num_classes_per_attr = [len(attribute_values[attr]) for attr in group_definition[:2]]
        laftr_adversaries = [LAFTRAdversary(EMBEDDING_DIM, num_classes).to(device) for num_classes in num_classes_per_attr]
        laftr_optimizer = optim.Adam(
            list(laftr_embedding_model.parameters()) +
            list(laftr_task_head.parameters() if train_head else []) +
            list(p for adv in laftr_adversaries for p in adv.parameters()),
            lr=0.001
        )
        
        train_laftr(laftr_embedding_model, laftr_task_head, laftr_adversaries, train_loader,
                   criterion, [nn.CrossEntropyLoss() for _ in laftr_adversaries], laftr_optimizer, n_epochs=20,
                   lambda_adv=0.1, lambda_task=1.0, train_embedding_model=True, 
                   adv_attr_indices=list(range(len(num_classes_per_attr))))
        
        with torch.no_grad():
            embeddings = laftr_embedding_model(X_test_heldout_tensor)
            outputs = laftr_task_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            results['LAFTR']['f1'][inclusion_pct] = f1
            print(f"LAFTR Accuracy on held-out group: F1: {f1:.4f}")
        
        # Compute fairness metrics
        laftr_metrics = compute_fairness_metrics(laftr_embedding_model, laftr_task_head, 
                                                  X_test_tensor, y_test_tensor, sensitive_test,
                                                  target_group, group_definition, attribute_values, device)
        #fairness_metrics['LAFTR']['accuracy_distance'][inclusion_pct] = laftr_metrics['accuracy_distance']
        fairness_metrics['LAFTR']['f1_distance'][inclusion_pct] = laftr_metrics['f1_distance']
        #fairness_metrics['LAFTR']['embedding_distance_sq'][inclusion_pct] = laftr_metrics['embedding_distance_sq']
        print(f"LAFTR - F1 Distance: {laftr_metrics['f1_distance']:.4f}")
        
        # Collect LAFTR metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- LAFTR Evaluation ---")
            metrics_laftr = evaluate_model(laftr_embedding_model, laftr_task_head, None, None,
                                           X_test_tensor, y_test_tensor, sensitive_test_copy,
                                           X_test_tensor, torch.empty(0), sensitive_test_copy,
                                           return_metrics=True)
            comparison_results['LAFTR'] = metrics_laftr
        
        # --- ROAD Baseline ---
        print("\n--- ROAD Baseline ---")
        road_embedding_model = EmbeddingModel(input_dim).to(device)
        road_prediction_head = PredictionHead().to(device)
        road_perturbation_net = ROADPerturbation(input_dim, perturbation_dim=32, epsilon=0.1).to(device)
        road_optimizer = optim.Adam(
            list(road_embedding_model.parameters()) +
            list(road_prediction_head.parameters() if train_head else []) +
            list(road_perturbation_net.parameters()),
            lr=0.001
        )
        
        train_road(road_embedding_model, road_prediction_head, road_perturbation_net, train_loader,
                  criterion, road_optimizer, n_epochs=20, alpha=0.5, train_embedding_model=True)
        
        with torch.no_grad():
            embeddings = road_embedding_model(X_test_heldout_tensor)
            outputs = road_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            #results['ROAD']['accuracy'][inclusion_pct] = accuracy.item()
            results['ROAD']['f1'][inclusion_pct] = f1
            #print(f"ROAD Accuracy on held-out group: {accuracy.item():.4f}, F1: {f1:.4f}")
        
        # Compute fairness metrics
        road_metrics = compute_fairness_metrics(road_embedding_model, road_prediction_head, 
                                                 X_test_tensor, y_test_tensor, sensitive_test,
                                                 target_group, group_definition, attribute_values, device)
        #fairness_metrics['ROAD']['accuracy_distance'][inclusion_pct] = road_metrics['accuracy_distance']
        fairness_metrics['ROAD']['f1_distance'][inclusion_pct] = road_metrics['f1_distance']
        #fairness_metrics['ROAD']['embedding_distance_sq'][inclusion_pct] = road_metrics['embedding_distance_sq']
        #print(f"ROAD - Accuracy Distance: {road_metrics['accuracy_distance']:.4f}, F1 Distance: {road_metrics['f1_distance']:.4f}, Embedding Distance Sq: {road_metrics['embedding_distance_sq']:.4f}")
        
        # Collect ROAD metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- ROAD Evaluation ---")
            metrics_road = evaluate_model(road_embedding_model, road_prediction_head, None, None,
                                          X_test_tensor, y_test_tensor, sensitive_test_copy,
                                          X_test_tensor, torch.empty(0), sensitive_test_copy,
                                          return_metrics=True)
            comparison_results['ROAD'] = metrics_road
        
        # --- AdversarialLearning Baseline ---
        print("\n--- AdversarialLearning Baseline ---")
        adv_learning_embedding_model = EmbeddingModel(input_dim).to(device)
        adv_learning_task_head = PredictionHead().to(device)
        adv_learning_adversary = AdversarialLearningAdversary(
            EMBEDDING_DIM,
            num_attributes=len(num_classes_per_attr),
            num_classes_per_attr=num_classes_per_attr
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
            n_epochs=20, alpha=1.0, train_embedding_model=True,
            adv_attr_indices=list(range(len(num_classes_per_attr)))
        )
        
        with torch.no_grad():
            embeddings = adv_learning_embedding_model(X_test_heldout_tensor)
            outputs = adv_learning_task_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            #results['AdversarialLearning']['accuracy'][inclusion_pct] = accuracy.item()
            results['AdversarialLearning']['f1'][inclusion_pct] = f1
            #print(f"AdversarialLearning Accuracy on held-out group: {accuracy.item():.4f}, F1: {f1:.4f}")
        
        # Compute fairness metrics
        adv_metrics = compute_fairness_metrics(adv_learning_embedding_model, adv_learning_task_head, 
                                                X_test_tensor, y_test_tensor, sensitive_test,
                                                target_group, group_definition, attribute_values, device)
        #fairness_metrics['AdversarialLearning']['accuracy_distance'][inclusion_pct] = adv_metrics['accuracy_distance']
        fairness_metrics['AdversarialLearning']['f1_distance'][inclusion_pct] = adv_metrics['f1_distance']
        #fairness_metrics['AdversarialLearning']['embedding_distance_sq'][inclusion_pct] = adv_metrics['embedding_distance_sq']
        #print(f"AdversarialLearning - Accuracy Distance: {adv_metrics['accuracy_distance']:.4f}, F1 Distance: {adv_metrics['f1_distance']:.4f}, Embedding Distance Sq: {adv_metrics['embedding_distance_sq']:.4f}")
        
        # Collect AdversarialLearning metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- AdversarialLearning Evaluation ---")
            metrics_adv_learning = evaluate_model(adv_learning_embedding_model, adv_learning_task_head, None, None,
                                                 X_test_tensor, y_test_tensor, sensitive_test_copy,
                                                 X_test_tensor, torch.empty(0), sensitive_test_copy,
                                                 return_metrics=True)
            comparison_results['AdversarialLearning'] = metrics_adv_learning
        
        # --- HSIC Baseline ---
        print("\n--- HSIC Baseline ---")
        hsic_embedding_model = EmbeddingModel(input_dim).to(device)
        hsic_prediction_head = PredictionHead().to(device)
        hsic_optimizer = optim.Adam(
            list(hsic_embedding_model.parameters()) +
            list(hsic_prediction_head.parameters() if train_head else []),
            lr=0.001
        )
        
        train_hsic(hsic_embedding_model, hsic_prediction_head, train_loader,
                  criterion, hsic_optimizer, n_epochs=20, lambda_hsic=0.1,
                  num_attr_cols=len(num_classes_per_attr))
        
        with torch.no_grad():
            embeddings = hsic_embedding_model(X_test_heldout_tensor)
            outputs = hsic_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            #results['HSIC']['accuracy'][inclusion_pct] = accuracy.item()
            results['HSIC']['f1'][inclusion_pct] = f1
            #print(f"HSIC Accuracy on held-out group: {accuracy.item():.4f}, F1: {f1:.4f}")
        
        # Compute fairness metrics
        hsic_metrics = compute_fairness_metrics(hsic_embedding_model, hsic_prediction_head, 
                                                 X_test_tensor, y_test_tensor, sensitive_test,
                                                 target_group, group_definition, attribute_values, device)
        #fairness_metrics['HSIC']['accuracy_distance'][inclusion_pct] = hsic_metrics['accuracy_distance']
        fairness_metrics['HSIC']['f1_distance'][inclusion_pct] = hsic_metrics['f1_distance']
        #fairness_metrics['HSIC']['embedding_distance_sq'][inclusion_pct] = hsic_metrics['embedding_distance_sq']
        #print(f"HSIC - Accuracy Distance: {hsic_metrics['accuracy_distance']:.4f}, F1 Distance: {hsic_metrics['f1_distance']:.4f}, Embedding Distance Sq: {hsic_metrics['embedding_distance_sq']:.4f}")
        
        # Collect HSIC metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- HSIC Evaluation ---")
            metrics_hsic = evaluate_model(hsic_embedding_model, hsic_prediction_head, None, None,
                                          X_test_tensor, y_test_tensor, sensitive_test_copy,
                                          X_test_tensor, torch.empty(0), sensitive_test_copy,
                                          return_metrics=True)
            comparison_results['HSIC'] = metrics_hsic
        
        # --- INLP Baseline ---
        print("\n--- INLP Baseline ---")
        inlp_embedding_model = EmbeddingModel(input_dim).to(device)
        inlp_prediction_head = PredictionHead().to(device)
        inlp_optimizer = optim.Adam(
            list(inlp_embedding_model.parameters()) +
            list(inlp_prediction_head.parameters() if train_head else []),
            lr=0.001
        )
        
        train_inlp(inlp_embedding_model, inlp_prediction_head, train_loader,
                  criterion, inlp_optimizer, n_epochs=20, train_embedding_model=True, num_iterations=10)
        
        with torch.no_grad():
            embeddings = inlp_embedding_model(X_test_heldout_tensor)
            outputs = inlp_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            #results['INLP']['accuracy'][inclusion_pct] = accuracy.item()
            results['INLP']['f1'][inclusion_pct] = f1
            #print(f"INLP Accuracy on held-out group: {accuracy.item():.4f}, F1: {f1:.4f}")
        
        # Compute fairness metrics
        inlp_metrics = compute_fairness_metrics(inlp_embedding_model, inlp_prediction_head,
                                                 X_test_tensor, y_test_tensor, sensitive_test,
                                                 target_group, group_definition, attribute_values, device)
        #fairness_metrics['INLP']['accuracy_distance'][inclusion_pct] = inlp_metrics['accuracy_distance']
        fairness_metrics['INLP']['f1_distance'][inclusion_pct] = inlp_metrics['f1_distance']
        #fairness_metrics['INLP']['embedding_distance_sq'][inclusion_pct] = inlp_metrics['embedding_distance_sq']
        #print(f"INLP - Accuracy Distance: {inlp_metrics['accuracy_distance']:.4f}, F1 Distance: {inlp_metrics['f1_distance']:.4f}, Embedding Distance Sq: {inlp_metrics['embedding_distance_sq']:.4f}")
        
        # Collect INLP metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- INLP Evaluation ---")
            metrics_inlp = evaluate_model(inlp_embedding_model, inlp_prediction_head, None, None,
                                          X_test_tensor, y_test_tensor, sensitive_test_copy,
                                          X_test_tensor, torch.empty(0), sensitive_test_copy,
                                          return_metrics=True)
            comparison_results['INLP'] = metrics_inlp
        
        # --- RLACE Baseline ---
        print("\n--- RLACE Baseline ---")
        rlace_embedding_model = EmbeddingModel(input_dim).to(device)
        rlace_prediction_head = PredictionHead().to(device)
        rlace_optimizer = optim.Adam(
            list(rlace_embedding_model.parameters()) +
            list(rlace_prediction_head.parameters() if train_head else []),
            lr=0.001
        )
        
        train_rlace(rlace_embedding_model, rlace_prediction_head, train_loader,
                   criterion, rlace_optimizer, n_epochs=20, train_embedding_model=True, num_iterations=10, alpha=100.0)
        
        with torch.no_grad():
            embeddings = rlace_embedding_model(X_test_heldout_tensor)
            outputs = rlace_prediction_head(embeddings)
            predictions = (outputs > 0.5).float()
            accuracy = (predictions == y_test_heldout_tensor).float().mean()
            predictions_np = predictions.cpu().numpy()
            labels_np = y_test_heldout_tensor.cpu().numpy()
            f1 = f1_score(labels_np, predictions_np, zero_division=0)
            #results['RLACE']['accuracy'][inclusion_pct] = accuracy.item()
            results['RLACE']['f1'][inclusion_pct] = f1
            #print(f"RLACE Accuracy on held-out group: {accuracy.item():.4f}, F1: {f1:.4f}")
        
        # Compute fairness metrics
        rlace_metrics = compute_fairness_metrics(rlace_embedding_model, rlace_prediction_head,
                                                 X_test_tensor, y_test_tensor, sensitive_test,
                                                 target_group, group_definition, attribute_values, device)
        #fairness_metrics['RLACE']['accuracy_distance'][inclusion_pct] = rlace_metrics['accuracy_distance']
        fairness_metrics['RLACE']['f1_distance'][inclusion_pct] = rlace_metrics['f1_distance']
        #fairness_metrics['RLACE']['embedding_distance_sq'][inclusion_pct] = rlace_metrics['embedding_distance_sq']
        #print(f"RLACE - Accuracy Distance: {rlace_metrics['accuracy_distance']:.4f}, F1 Distance: {rlace_metrics['f1_distance']:.4f}, Embedding Distance Sq: {rlace_metrics['embedding_distance_sq']:.4f}")
        
        # Collect RLACE metrics - only for first inclusion percentage
        if inclusion_pct == inclusion_percentages[0] and print_comparison_summary_table:
            print(f"\n--- RLACE Evaluation ---")
            metrics_rlace = evaluate_model(rlace_embedding_model, rlace_prediction_head, None, None,
                                          X_test_tensor, y_test_tensor, sensitive_test_copy,
                                          X_test_tensor, torch.empty(0), sensitive_test_copy,
                                          return_metrics=True)
            comparison_results['RLACE'] = metrics_rlace
    
    # Print comparison summary table if requested
    if print_comparison_summary_table and comparison_results:
        print_comparison_summary(comparison_results)
    
    # Print summary
    print("\n" + "="*60)
    print("GROUP INCLUSION EXPERIMENT SUMMARY")
    print("="*60)
    print(f"Target Group: {target_group}")
   
    print("\nF1 on Held-Out Group Samples:")
    print(f"{'Method':<20} " + " ".join([f"{p*100:>5.0f}%" for p in inclusion_percentages]))
    for method in ['ROAR', 'ERM', 'GroupDRO', 'LAFTR', 'ROAD', 'AdversarialLearning', 'HSIC', 'INLP', 'RLACE']:
        f1s = [results[method]['f1'].get(p, float('nan')) for p in inclusion_percentages]
        print(f"{method:<20} " + " ".join([f"{f:>6.4f}" if not np.isnan(f) else "  N/A" for f in f1s]))
    
    print("\nFairness Metrics - F1 Distance:")
    print(f"{'Method':<20} " + " ".join([f"{p*100:>5.0f}%" for p in inclusion_percentages]))
    for method in ['ROAR', 'ERM', 'GroupDRO', 'LAFTR', 'ROAD', 'AdversarialLearning', 'HSIC', 'INLP', 'RLACE']:
        f1_distances = [fairness_metrics[method]['f1_distance'].get(p, float('nan')) for p in inclusion_percentages]
        print(f"{method:<20} " + " ".join([f"{d:>6.4f}" if d is not None and not np.isnan(d) else "  N/A" for d in f1_distances]))
    
    return results, fairness_metrics

def construct_intersectional_groups(sensitive_attributes, group_definition=None, attribute_values=None):
    """
    Constructs intersectional groups from sensitive attributes.
    
    Args:
        sensitive_attributes: DataFrame with sensitive attributes
        group_definition: List of attributes to use for group construction
                          If None, uses all attributes
        attribute_values: Pre-defined attribute values mapping. If None, creates from data.
    
    Returns:
        group_keys: List of all possible group keys
        group_to_idx: Mapping from group key to index
        attribute_indices: Array of attribute indices for each sample
        valid_indices: Indices of valid samples (those with all attribute values in the mapping)
    """
    print("\nConstructing intersectional groups...")
    
    if group_definition is None:
        group_definition = [col for col in sensitive_attributes.columns if col != 'group_key']
    
    # Get unique values for each attribute (use provided if available)
    if attribute_values is None:
        attribute_values = {}
        for attr in group_definition:
            attribute_values[attr] = sorted(sensitive_attributes[attr].unique())
    
    # Generate all possible combinations
    all_combinations = list(itertools.product(*[attribute_values[attr] for attr in group_definition]))
    
    # Create group keys
    group_keys = ['_'.join(map(str, combo)) for combo in all_combinations]
    group_to_idx = {key: idx for idx, key in enumerate(group_keys)}
    
    # Convert samples to attribute indices
    attribute_indices = []
    valid_indices = []
    for pos, (idx, row) in enumerate(sensitive_attributes[group_definition].iterrows()):
        indices = []
        valid = True
        for attr in group_definition:
            value = row[attr]
            if value in attribute_values[attr]:
                attr_idx = attribute_values[attr].index(value)
            else:
                valid = False
                break
            indices.append(attr_idx)
        if valid:
            attribute_indices.append(indices)
            valid_indices.append(pos)
    
    # Filter sensitive_attributes to only include valid samples
    sensitive_attributes = sensitive_attributes.iloc[valid_indices].reset_index(drop=True)
    
    attribute_indices = np.array(attribute_indices)
    valid_indices = np.array(valid_indices)
    
    print(f"Total number of intersectional groups: {len(group_keys)}")
    print(f"Attribute indices shape: {attribute_indices.shape}")
    print(f"Valid samples: {len(valid_indices)}/{len(sensitive_attributes)}")
    
    return group_keys, group_to_idx, attribute_indices, attribute_values, valid_indices

def create_group_inclusion_split(X_train, y_train, sensitive_train, target_group, inclusion_percentage):
    """
    Creates a modified training set where only a percentage of the target group is included.
    
    Args:
        X_train: Training features
        y_train: Training labels
        sensitive_train: Training sensitive attributes
        target_group: The group key to vary inclusion for (e.g., 'White_Male_3')
        inclusion_percentage: Percentage of target group to include in training (0.0 to 1.0)
    
    Returns:
        X_train_modified: Modified training features
        y_train_modified: Modified training labels
        sensitive_train_modified: Modified training sensitive attributes
        group_mask: Boolean mask indicating which samples belong to the target group
    """
    # Create group_key for training data
    attr_cols = [col for col in sensitive_train.columns if col != 'group_key']
    seen_upper = set()
    filtered_cols = []
    for col in attr_cols:
        upper_col = col.upper()
        if upper_col in seen_upper:
            continue
        if col.isupper():
            seen_upper.add(col)
            filtered_cols.append(col)
        elif upper_col not in seen_upper:
            filtered_cols.append(col)
            seen_upper.add(upper_col)
    
    sensitive_train_copy = sensitive_train.copy()
    sensitive_train_copy['group_key'] = sensitive_train_copy[filtered_cols].astype(str).agg('_'.join, axis=1)
    
    # Identify samples belonging to the target group
    group_mask = sensitive_train_copy['group_key'] == target_group
    group_indices = np.where(group_mask)[0]
    non_group_indices = np.where(~group_mask)[0]
    
    # Sample the specified percentage of the target group for training
    n_group_samples = len(group_indices)
    n_include = int(n_group_samples * inclusion_percentage)
    
    if n_include > 0:
        include_indices = np.random.choice(group_indices, size=n_include, replace=False)
    else:
        include_indices = np.array([])
    
    # Combine: all non-target group samples + sampled target group samples
    train_indices = np.concatenate([non_group_indices, include_indices])
    train_indices = train_indices.astype(int)  # Ensure integer type for indexing
    
    # Create modified training data
    # Handle both pandas DataFrames and numpy arrays
    if hasattr(X_train, 'iloc'):
        X_train_modified = X_train.iloc[train_indices].reset_index(drop=True)
    else:
        X_train_modified = X_train[train_indices]
    
    if hasattr(y_train, 'iloc'):
        y_train_modified = y_train.iloc[train_indices].reset_index(drop=True)
    else:
        y_train_modified = y_train[train_indices]
    
    if hasattr(sensitive_train, 'iloc'):
        sensitive_train_modified = sensitive_train.iloc[train_indices].reset_index(drop=True)
    else:
        sensitive_train_modified = sensitive_train[train_indices]
    
    return X_train_modified, y_train_modified, sensitive_train_modified, group_mask
