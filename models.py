import torch
import torch.nn as nn
import math
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def create_padding_mask(length):
    seq = torch.eq(length, 0)
    return seq.unsqueeze(1).unsqueeze(2)

def create_look_ahead_mask(size):
    mask = (1 - torch.triu(torch.ones(size, size), diagonal=1)).bool()
    return mask

def positions(sequence, size):
    if not isinstance(sequence, int):
        raise ValueError(f"Invalid type for sequence. Expected int, got: {type(sequence)}")

    if not isinstance(size, int):
        raise ValueError(f"Invalid type for size. Expected int, got: {type(size)}")

    if sequence <= 0:
        raise ValueError("Sequence length must be positive.")

    if size <= 0:
        raise ValueError("Size of positional encoding must be positive.")

    batch_size = sequence
    pos = torch.arange(batch_size).float().unsqueeze(1)
    i = torch.arange(size).float().unsqueeze(0)
    angles = pos / 10000 ** (2 * (i // 2) / size)
    angles = angles.reshape(batch_size, size)

    pe = torch.zeros(batch_size, size)
    pe[:, 0::2] = torch.sin(angles[:, 0::2])
    pe[:, 1::2] = torch.cos(angles[:, 1::2])

    return pe

def accuracy(y_true, y_pred):
    y_pred_argmax = torch.argmax(y_pred, dim=-1)
    correct = (y_pred_argmax == y_true).float().sum()
    total = y_true.numel()
    return correct / total

class CustomSchedule(LambdaLR):
    def __init__(self, optimizer, d_model, warmup_steps=4000):
        self.d_model = torch.tensor(d_model, dtype=torch.float32)
        self.warmup_steps = warmup_steps
        super(CustomSchedule, self).__init__(optimizer, self.lr_lambda)

    def lr_lambda(self, step):
        step_float = torch.tensor(step, dtype=torch.float32)
        arg1 = torch.rsqrt(step_float)
        arg2 = step_float * (self.warmup_steps**-1.5)
        return (torch.rsqrt(self.d_model) * torch.minimum(arg1, arg2))


class Normalize(nn.Module):
  def __init__(self, scale: float, shift:float,):
    super(Normalize, self).__init__()
    self.scale = scale
    self.shift = shift

  def forward(self, x):
      #---Normalization---
      mean = torch.mean(x)
      deviation = torch.std(x)
      x = (x - mean) / deviation

      #---Scale---
      x = x * self.scale

      #---Shift---
      x = x + self.shift
      return x


"""
------
I'll implement this with the code later on.
This class comes from the main repo. Not sure how it work
with my modifications yet.
-------

class Embeddings(nn.Module):
    def __init__(self, vocab_size, d_model, max_len = 50):
        super(Embeddings, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(0.1)
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = self.create_positinal_encoding(max_len, self.d_model)
        self.dropout = nn.Dropout(0.1)
        
    def create_positinal_encoding(self, max_len, d_model):
        pe = torch.zeros(max_len, d_model).to(device)
        for pos in range(max_len):   # for each position of the word
            for i in range(0, d_model, 2):   # for each dimension of the each position
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)   # include the batch size
        return pe
        
    def forward(self, encoded_words):
        embedding = self.embed(encoded_words) * math.sqrt(self.d_model)
        embedding += self.pe[:, :embedding.size(1)]   # pe will automatically be expanded with the same batch size as encoded_words
        embedding = self.dropout(embedding)
        return embedding
"""


#---- Self Made Multi Head Attention Implementation ----
class Multi_Head_Attention(nn.Module):
    def __init__(self, embedding_size, head_num, batch_size, learning_rate):
        super(Multi_Head_Attention, self).__init__()

        self.embedding_size = embedding_size
        self.num_heads = head_num
        self.learning_rate = learning_rate
        self.head_size = embedding_size // self.num_heads

        self.head_size = int(self.head_size)

        self.w_q = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_k = nn.Linear(embedding_size, embedding_size, bias=False)
        self.w_v = nn.Linear(embedding_size, embedding_size, bias=False)

        self.w_o = nn.Linear(embedding_size, embedding_size, bias=False)

    def scaled_dot_product_attention(self, x, blank, mask=None):
        """
        'blank' is what they would normally call "decoder_input" but personally I rather use 'blank'
        like a 'BLANK piece of paper the model writes (predicts) it's output given the context'
        """
        batch, length, size = x.shape
        
        if blank is not None:
            b_batch, b_length, b_size = blank.shape
            q = self.w_q(blank).view(b_batch, b_length, self.num_heads, self.head_size).permute(0,2,1,3)
        else:
            q = self.w_q(x).view(batch, length, self.num_heads, self.head_size).permute(0,2,1,3)
        k = self.w_k(x).view(batch, length, self.num_heads, self.head_size).permute(0,2,1,3)
        v = self.w_v(x).view(batch, length, self.num_heads, self.head_size)

        q = q.permute(0, 2, 3, 1)
        k = k.permute(0, 2, 1, 3)

        distribution = torch.matmul(q, k)/ torch.sqrt(torch.tensor(self.embedding_size))

        if mask is not None:
            distribution.masked_fill(mask == 0, float('-inf'))

        v = v.transpose(2, 3)
        distribution = F.softmax(distribution, dim=-1)
        weight = torch.matmul(distribution, v)
        return weight

    def forward(self, x, blank=None, mask=None ):
        #shape: batch / length /embedding_size
        batch, length, size = x.shape

        product = self.scaled_dot_product_attention(x, blank, mask)

        product = product.permute(0,2,1,3)

        concat = product.reshape(batch, length, self.embedding_size)

        out = self.w_o(concat)

        return out


class Encoder_Layer(nn.Module):
    def __init__(self, embedding_size, head_num, batch_size, dropout, learning_rate):
        super(Encoder_Layer, self).__init__()

        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.fc2 = nn.Linear(embedding_size, embedding_size, bias=False)

        self.attention = Multi_Head_Attention(embedding_size, head_num, batch_size, learning_rate)

        self.norm = Normalize(0.4, 0.4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        attn = self.attention(x, None, None)

        attn_d = self.dropout(attn)
        attn_dn = self.norm(x + attn_d)

        attn_lin1 = self.fc1(attn_dn) # This part part downwards should the Feed Forward.
        attn_rel = F.relu(attn_lin1)  # The same for the decoder part. 

        attn_lin2 = self.fc2(attn_rel)
        attn_d = self.dropout(attn_lin2)
        out = self.norm(attn_dn + attn_d)

        return out

class Encoder(nn.Module):
    def __init__(self, embedding, embedding_size, head_num, batch_size, dropout, learning_rate, n_layers=2):
        super(Encoder, self).__init__()

        self.num_layers = n_layers
        self.encoder_layers = nn.ModuleList([Encoder_Layer(embedding_size, head_num, batch_size, dropout, learning_rate) for _ in range(n_layers)])

        self.embedding = embedding
        self.embedding_size = embedding_size

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # shape: batch / length / embedding_size
        batch, length, embedding_size = x.shape


        pos = positions(length, self.embedding_size).to(device)
        input_pos = pos + x

        input_d = self.dropout(input_pos)#.to(device)

        for encoder_layer in self.encoder_layers:
            output = encoder_layer(input_d, mask)

        return output

class Decoder_Layer(nn.Module):
    def __init__(self, embedding_size, head_num, batch_size, dropout, learning_rate):
        super(Decoder_Layer, self).__init__()

        self.embedding_size = embedding_size

        self.fc1 = nn.Linear(embedding_size, embedding_size, bias=False)
        self.fc2 = nn.Linear(embedding_size, embedding_size, bias=False)

        self.attention = Multi_Head_Attention(embedding_size, head_num, batch_size, learning_rate)

        self.norm = Normalize(0.4, 0.4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, enc_o, blank, ahead, mask):
        attn1 = self.attention(enc_o, None, mask)
        attn_n = self.norm(attn1 + enc_o)

        attn2 = self.attention(enc_o, blank, ahead)
        attn_d = self.dropout(attn2)
        attn_n2d = self.norm(attn_n + attn_d)

        attn_lin1 = self.fc1(attn_n2d)
        attn_rel = F.relu(attn_lin1)

        attn_lin2 = self.fc2(attn_rel)
        attn_d = self.dropout(attn_lin2)
        output = self.norm(attn_d + attn_n2d)

        return output

class Decoder(nn.Module):
    def __init__(self, embedding, embedding_size, head_num, batch_size, dropout, learning_rate, n_layers):
        super(Decoder, self).__init__()
        self.embedding = embedding
        self.embedding_size = embedding_size
        self.num_layers = n_layers

        self.decoder_layers = nn.ModuleList([Decoder_Layer(embedding_size, head_num, batch_size, dropout, learning_rate) for _ in range(n_layers)])

        self.dropout = nn.Dropout(dropout)

    def forward(self, decoder_input, enc_output, ahead=None, mask=None):
        # shape: batch/ length /embedding_size (GENERAL SHAPE)

        embedded = self.embedding(decoder_input.t()).to(device)

        batch, length, embedding_size = enc_output.shape

        batch, length, embedding_size = embedded.shape

        pos = positions(length, self.embedding_size).to(device)
        input_pos = pos + enc_output

        input_d = self.dropout(input_pos)#.to(device)
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(input_d, embedded, ahead, mask)

        return output

class Transformer(nn.Module):
    def __init__(self, embedding_size, head_num, batch_size, dropout, learning_rate, decoder_learning_ratio,
                 encoder_n_layers, decoder_n_layers, vocab_size, libra, task='train', loadFilename=None):
        super(Transformer, self).__init__()

        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.num_heads = head_num
        self.task = task

        self.fc = nn.Linear(embedding_size, vocab_size, bias=False)
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        self.decoder = Decoder(self.embedding, embedding_size, head_num, batch_size, dropout, learning_rate, encoder_n_layers)
        self.encoder = Encoder(self.embedding, embedding_size, head_num, batch_size, dropout, learning_rate, decoder_n_layers)

        if loadFilename:
            print("Set to: 'trained model'")
            self.checkpoint = torch.load(loadFilename, map_location=device)
            encoder_sd = self.checkpoint['en']
            decoder_sd = self.checkpoint['de']
            encoder_optimizer_sd = self.checkpoint['en_opt']
            decoder_optimizer_sd = self.checkpoint['de_opt']
            embedding_sd = self.checkpoint['embedding']
            libra.__dict__ = self.checkpoint['voc_dict']
            print("Loss: ",self.checkpoint["loss"])
            print("Time: ",self.checkpoint["time"])

        else:
            print("Set to: 'new model'")

        if loadFilename:
            self.embedding.load_state_dict(embedding_sd)

        if loadFilename:
            self.encoder.load_state_dict(encoder_sd)
            self.decoder.load_state_dict(decoder_sd)

        self.embedding = self.embedding.to(device)
        self.encoder = self.encoder.to(device)
        self.decoder = self.decoder.to(device)

        if task == "train":
            self.encoder.train()
            self.decoder.train()
        else:
            self.encoder.eval()
            self.decoder.eval()

        self.decoder_optimizer = optim.Adam(self.decoder.parameters(), lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
        self.encoder_optimizer = optim.Adam(self.encoder.parameters(), lr=learning_rate * decoder_learning_ratio, betas=(0.9, 0.98), eps=1e-9)

        self.decoder_scheduler = CustomSchedule(self.decoder_optimizer, embedding_size)
        self.encoder_scheduler = CustomSchedule(self.encoder_optimizer, embedding_size)

        if loadFilename:
            self.encoder_optimizer.load_state_dict(encoder_optimizer_sd)
            self.decoder_optimizer.load_state_dict(decoder_optimizer_sd)

    def forward(self, input_variable):
        # shape:  batch / length / embedding_size
        if len(input_variable.shape) == 3:
          batch, length, d_model = input_variable.size()
        else:
          input_variable = self.embedding(input_variable)
          batch, length, d_model = input_variable.size()

        enc_padding_mask = create_padding_mask(torch.LongTensor(length)).to(device)
        dec_padding_mask = create_padding_mask(torch.LongTensor(length)).to(device)
        look_ahead_mask  = create_look_ahead_mask(self.embedding_size // self.num_heads).to(device)

        decoder_input = torch.LongTensor([[SOS_token for _ in range(batch)]]).to(device)

        enc_output = self.encoder(input_variable, enc_padding_mask)
        dec_output = self.decoder(decoder_input, enc_output, look_ahead_mask, dec_padding_mask)

        output = self.fc(dec_output)
        decoder_output_probs = F.softmax(output, dim=-1)

        return decoder_output_probs
