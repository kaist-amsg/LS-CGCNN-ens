from __future__ import print_function, division
import torch
import torch.nn as nn

class ConvLayer(nn.Module):
	def __init__(self,atom_fea_len,nbr_fea_len):
		super(ConvLayer,self).__init__()
		self.atom_fea_len = atom_fea_len    
		self.nbr_fea_len = nbr_fea_len
		self.fc_full = nn.Linear(2*self.atom_fea_len+self.nbr_fea_len,
                             2*self.atom_fea_len)
		self.sigmoid = nn.Sigmoid()
		self.softplus1 = nn.Tanh()
		self.bn1 = nn.BatchNorm1d(2*self.atom_fea_len)
		self.bn2 = nn.BatchNorm1d(self.atom_fea_len)
		self.softplus2 = nn.Tanh()

	def forward(self, atom_in_fea, nbr_fea, nbr_fea_idx):
		# TODO will there be problems with the index zero padding?        
		N, M = nbr_fea_idx.shape
		atom_nbr_fea = atom_in_fea[nbr_fea_idx,:]
		total_nbr_fea = torch.cat([atom_in_fea.unsqueeze(1).expand(N, M, self.atom_fea_len),
             									 atom_nbr_fea, nbr_fea], dim=2)

		total_gated_fea = self.fc_full(total_nbr_fea)
		total_gated_fea = self.bn1(total_gated_fea.view(-1, self.atom_fea_len*2)).view(N, M, self.atom_fea_len*2)
		nbr_filter,nbr_core = total_gated_fea.chunk(2,dim=2)
		nbr_filter = self.sigmoid(nbr_filter)
		nbr_core = self.softplus1(nbr_core)
		nbr_sumed = torch.sum(nbr_filter * nbr_core, dim=1)
		nbr_sumed = self.bn2(nbr_sumed)
		out = self.softplus2(atom_in_fea + nbr_sumed)
		return out


class CrystalGraphConvNet(nn.Module):
	def __init__(self, orig_atom_fea_len, nbr_fea_len,
           atom_fea_len=64, n_conv=3, h_fea_len=128, n_h=1,classification=False):

		super(CrystalGraphConvNet, self).__init__()
		self.classification = classification
		self.embedding = nn.Linear(orig_atom_fea_len, atom_fea_len)
		self.convs = nn.ModuleList([ConvLayer(atom_fea_len=atom_fea_len,
                        nbr_fea_len=nbr_fea_len) for _ in range(n_conv)])

		self.W3 = nn.Linear(atom_fea_len,atom_fea_len)
		self.attn_tanh = nn.Tanh()
		self.attn_sigmoid = nn.Sigmoid()

		self.conv_to_fc = nn.Linear(atom_fea_len, h_fea_len)
		self.conv_to_fc_softplus = nn.Softplus()
		if n_h > 1:
			self.fcs = nn.ModuleList([nn.Linear(h_fea_len, h_fea_len)
																for _ in range(n_h-1)])
			self.softpluses = nn.ModuleList([nn.Softplus()
																for _ in range(n_h-1)])
		self.fc_out = nn.Linear(h_fea_len,1)

	def forward(self, atom_fea, nbr_fea, nbr_fea_idx, crystal_atom_idx):
		N,_ = atom_fea.shape
		atom_fea = self.embedding(atom_fea)

		for conv_func in self.convs:
			atom_fea = conv_func(atom_fea, nbr_fea, nbr_fea_idx)
	
		v_ = self.pooling(atom_fea,crystal_atom_idx).view(N,-1)
		c = self.attn_tanh(self.W3(v_)).view(N,-1)
		a = self.attn_sigmoid(torch.sum(atom_fea*c,dim=1,keepdim=True))
		crys_fea = a*atom_fea
		crys_fea = self.global_pooling(crys_fea,crystal_atom_idx)
		attn_crys_fea = crys_fea
		crys_fea = self.conv_to_fc_softplus(self.conv_to_fc(crys_fea))

		if hasattr(self, 'fcs') and hasattr(self, 'softpluses'):
			for fc, softplus in zip(self.fcs, self.softpluses):
				crys_fea = softplus(fc(crys_fea))
			
		out = self.fc_out(crys_fea)
		return out,atom_fea

	def pooling(self, atom_fea, crystal_atom_idx):
		summed_fea = []
		_,fea_len = atom_fea.shape
		for idx_map in crystal_atom_idx:
			Natom_in_cell = len(idx_map)
			mapped = torch.mean(atom_fea[idx_map],dim=0,keepdim=True).expand(Natom_in_cell,fea_len)
			summed_fea.append(mapped)
		return torch.cat(summed_fea,dim=0)

	def global_pooling(self,atom_fea,crystal_atom_idx):
		summed_fea = [torch.mean(atom_fea[idx_map],dim=0,keepdim=True)
									for idx_map in crystal_atom_idx]
		return torch.cat(summed_fea,dim=0)

