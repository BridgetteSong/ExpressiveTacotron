
# CONFIG -----------------------------------------------------------------------------------------------------------#


from text import invalid_phonemes,  tokens

# Settings for all models
mel_type = ["lpc", "mel"][0]
n_mel_channels={"lpc": 20, "mel": 80}[mel_type]
model_type = ["non_attention", "attention"][1]

# Encoder parameters
num_chars = len(tokens)
use_skip=False
encoder_kernel_size=5
encoder_n_convolutions=3
encoder_embedding_dim=256*2
pos_embedding=32

# Duration parameters
duration_kernel_size=5

#############################################
# Reference Encoder Network Hyperparameters #
#############################################

speaker_encoder_type = ["GST","VAE", "GMVAE"][2]
expressive_encoder_type = ["GST","VAE", "GMVAE"][2]

spk_ids = {"spk_0": 0,
           "spk_1": 1,
           "spk_2": 2,
           "spk_3": 3,
           "spk_4": 4}

speaker_classes = len(spk_ids)
emotion_classes = speaker_classes

cat_lambda = 0.0
cat_incr = 0.01
cat_step = 1000
cat_step_after = 10
cat_max_step = 300000

kl_lambda = 0.00001
kl_incr = 0.000001
kl_step = 1000
kl_step_after = 500
kl_max_step = 300000

# reference_encoder
ref_enc_filters=[32, 32, 64, 64, 128, 128]
ref_enc_size=[3, 3]
ref_enc_strides=[2, 2]
ref_enc_pad=[1, 1]
ref_enc_gru_size=128

# Style Token Layer
token_num=10
num_heads=8

# embedding size
token_embedding_size=256
speaker_embedding_size=64
vae_size=32

# Decoder parameters
n_frames_per_step=1
decoder_rnn_dim=512*2
prenet_dims=[128*2, 128*2]
gate_threshold=0.5
max_decoder_steps=2000
p_attention_dropout=0.1
p_decoder_dropout=0.1

# Attention parameters
attention_mode=["GMM", "FA"][1]
attention_rnn_dim=512*2
attention_dim=256

# Location Layer parameters
attention_location_n_filters=32
attention_location_kernel_size=17

# GMM Attention parameters
delta_bias=1.0
sigma_biad=10.0
gmm_kernel=5

# Mel-post processing network parameters
postnet_embedding_dims=[512, 512, 512]
postnet_kernel_sizes=[5, 5, 5]
p_postnet_dropout=0.5

postnet_k=8
postnet_num_highways=4
post_projections=[256, n_mel_channels]

# Training parameters
adaption=False
adaption_id=0
distributed_run=True
cudnn_enabled=True
cudnn_benchmark=False
dist_backend="nccl"
dist_url="tcp://localhost:54321"
seed=1234
dynamic_loss_scaling=True
batch_size=32
learning_rate=1e-3
weight_decay=1e-6
training_steps=250_000
max_mel_len=1250
grad_clip_thresh=1.0
save_checkpoint_every_n_step=10_000
# ------------------------------------------------------------------------------------------------------------------#
