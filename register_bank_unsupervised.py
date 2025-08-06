import torch
import torch.nn as nn
import torch.nn.functional as F


class RegisterBankTransformerUnsupervised(nn.Module):
    """
    Transformer with register banks but NO supervised probe heads.
    Registers must learn useful representations purely from next-token prediction.
    """
    
    def __init__(self, vocab_size=1000, d_model=256, n_heads=8, n_layers=4, 
                 n_registers=3, register_dim=64):
        super().__init__()
        self.d_model = d_model
        self.n_registers = n_registers
        self.register_dim = register_dim
        
        # Core transformer components
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, 512, d_model) * 0.1)
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model, n_heads, 
                dim_feedforward=1024,
                dropout=0.1, 
                batch_first=True
            )
            for _ in range(n_layers)
        ])
        
        # Register bank components
        self.register_bank = nn.Parameter(torch.zeros(1, n_registers, register_dim))
        nn.init.normal_(self.register_bank, std=0.02)
        
        # Register interaction mechanisms
        self.register_gate = nn.Linear(d_model, n_registers)
        self.register_update = nn.ModuleList([
            nn.Linear(d_model + register_dim, register_dim) 
            for _ in range(n_registers)
        ])
        
        # Register read mechanism - inject register info back into sequence
        self.register_read = nn.Linear(register_dim * n_registers, d_model)
        
        # Output projection
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, input_ids, registers=None):
        B, L = input_ids.shape
        
        # Initialize registers if not provided (for cross-sequence persistence)
        if registers is None:
            registers = self.register_bank.expand(B, -1, -1)
        
        # Embed and encode position
        x = self.embedding(input_ids)
        x = x + self.pos_encoding[:, :L, :]
        
        # Process through transformer layers with register updates
        for i, layer in enumerate(self.layers):
            # On first layer, inject register information into the sequence
            if i == 0:
                # Read all registers and broadcast to sequence
                register_flat = registers.view(B, -1)  # [B, n_registers * register_dim]
                register_info = self.register_read(register_flat)  # [B, d_model]
                register_info = register_info.unsqueeze(1).expand(-1, L, -1)  # [B, L, d_model]
                x = x + 0.1 * register_info  # Gentle injection
            
            # Standard transformer layer
            x = layer(x)
            
            # Register update based on sequence information
            # Use mean pooling over sequence to get global context
            global_context = x.mean(dim=1)  # [B, d_model]
            
            # Gated update mechanism
            gate_logits = self.register_gate(global_context)  # [B, n_registers]
            gate = F.sigmoid(gate_logits)  # Use sigmoid for smooth gating
            
            # Update each register
            new_registers = []
            for j in range(self.n_registers):
                # Combine global context with current register
                combined = torch.cat([global_context, registers[:, j, :]], dim=-1)
                update = self.register_update[j](combined)
                
                # Gated update with residual connection
                new_reg = registers[:, j, :] + gate[:, j:j+1] * torch.tanh(update)
                new_registers.append(new_reg)
            
            registers = torch.stack(new_registers, dim=1)
        
        # Output predictions
        logits = self.output_proj(x)
        
        return logits, registers


class RegisterPersistentModel(nn.Module):
    """
    Wrapper that maintains register state across forward passes.
    This is crucial for testing true state persistence.
    """
    
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.register_states = {}  # Track registers per batch element
        
    def forward(self, input_ids, session_ids=None):
        """
        Forward pass with persistent registers.
        session_ids: tensor of shape [B] indicating which session each sequence belongs to
        """
        B = input_ids.shape[0]
        
        # Get or initialize registers for each session
        registers = []
        for i in range(B):
            session_id = session_ids[i].item() if session_ids is not None else i
            
            if session_id not in self.register_states:
                # Initialize new registers
                self.register_states[session_id] = self.base_model.register_bank.squeeze(0).clone()
            
            registers.append(self.register_states[session_id])
        
        registers = torch.stack(registers, dim=0)
        
        # Forward pass
        logits, new_registers = self.base_model(input_ids, registers)
        
        # Update stored registers
        for i in range(B):
            session_id = session_ids[i].item() if session_ids is not None else i
            self.register_states[session_id] = new_registers[i].detach()
        
        return logits
    
    def reset_session(self, session_id):
        """Reset registers for a specific session"""
        if session_id in self.register_states:
            del self.register_states[session_id]
    
    def reset_all_sessions(self):
        """Reset all register states"""
        self.register_states = {}