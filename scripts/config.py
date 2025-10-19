class Config:
    sample_rate = 16000
    n_fft = 512
    hop_length = 128
    win_length = 512

    hidden_dim = 256
    num_layers = 3
    dropout = 0.1

    batch_size = 4
    num_epochs = 100
    learning_rate = 1e-3
    gradient_clip = 5.0

    train_clean_path = "./data/wav_clean_train"
    train_noisy_path = "./data/wav_noisy_train"
    test_clean_path = "./data/wav_clean_test"
    test_noisy_path = "./data/wav_noisy_test"
    checkpoint_path = "./checkpoints"
    results_path = "./results"