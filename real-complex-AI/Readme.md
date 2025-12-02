# 📅 12-Week Real + Complex AI Implementation Challenge
## Week 1–2: Foundations (Classical ML)
### Datasets: Simple tabular & signal datasets
    1. Linear Regression → Boston Housing (scikit-learn) | Complex: synthetic sinusoidal dataset with complex noise.
    2. Logistic Regression → Breast Cancer Wisconsin | Complex: synthetic 2D spiral in complex plane.
    3. KNN (Classifier & Regressor) → Iris dataset | Complex: synthetic constellation points (QPSK).
    4. Naive Bayes → SMS Spam Dataset | Complex: Gaussian complex signals.
    5. Decision Trees → Titanic dataset | Complex: split using real/imag.
    6. Random Forests → Heart Disease Dataset | Complex: MRI small k-space data.
    7. Gradient Boosted Trees (XGBoost) → Adult Income | Complex: radio channel estimation synthetic.
    8. SVM (Linear + Kernel) → Digits dataset | Complex: Hermitian kernel on synthetic data.
    9. PCA → MNIST | Complex: apply to FFT-transformed MNIST images.
    10. LDA/QDA → Wine Dataset | Complex: classify on QAM modulated signals.
    11. k-Means → Fashion MNIST (flattened) | Complex: cluster constellation points.
    12. Gaussian Mixture Models → Iris | Complex: mixture of Gaussians on synthetic data.
    13. DBSCAN → 2D Moons dataset | Complex: group points in spiral.
    14. t-SNE/UMAP → CIFAR-10 features | Complex: visualize embedding of MRI spectra.
## Week 3–4: Core Deep Learning Blocks
### Datasets: Image & sequence basics
    15. Perceptron → Iris | Complex: XOR in complex plane.
    16. MLP → MNIST | Complex: same MNIST with FFT applied.
    17. Shallow Autoencoder → MNIST | Complex: denoising on k-space patches.
    18. Denoising Autoencoder → Fashion-MNIST | Complex: complex Gaussian noise removal.
    19. CNN (basic conv net) → CIFAR-10 | Complex: 2D Fourier-transformed images.
    20. Deep CNN (VGG-style) → Tiny ImageNet | Complex: spectrogram classification.
    21. RNN (vanilla) → IMDB Sentiment | Complex: synthetic time-series.
    22. LSTM → Shakespeare text | Complex: ECG signals in complex form.
    23. GRU → Stock price dataset | Complex: channel fading model.
    24. Bidirectional RNN → IMDB | Complex: speech spectrogram sequence.
    25. Conv1D for sequences → ECG dataset | Complex: radio signals.
    26. Complex Autoencoder → MRI reconstruction (fastMRI) small set.
    27. Complex CNN → Synthetic MRI data.
    28. Complex RNN/LSTM → QAM sequences.


## Week 5–6: Generative Models
### Datasets: Images & signals
    1. VAE → MNIST | Complex: synthetic k-space MRI.
    2. Conditional VAE → Fashion-MNIST | Complex: QPSK conditioned on labels.
    3. GAN (vanilla) → MNIST | Complex: generate sinusoidal waveforms.
    4. DCGAN → CIFAR-10 | Complex: MRI k-space patches.
    5. WGAN → CelebA | Complex: denoising k-space.
    6. CycleGAN → Horses ↔ Zebras dataset | Complex: frequency ↔ time domain mapping.
    7. Complex GAN → Synthetic constellation images.
    8. Complex DCGAN → MRI patches.
    9. PixelCNN → MNIST.
    10. RealNVP (flow) → Fashion-MNIST | Complex: amplitude-phase distribution modeling.
    11. Energy-based Model → CIFAR-10.
    12. Complex VAE → radio channel estimation dataset.

## Week 7–8: Modern Architectures
### Datasets: Medium-scale image + signal
    1. ResNet → CIFAR-10.
    2. DenseNet → CIFAR-100.
    3. U-Net → Carvana image segmentation.
    4. Complex U-Net → fastMRI dataset.
    5. Attention block → MNIST (toy).
    6. Transformer Encoder → IMDB.
    7. Transformer Decoder → WMT English-German translation.
    8. Vision Transformer (ViT) → CIFAR-10.
    9. Hybrid CNN+Transformer → Tiny ImageNet.
    10. Complex Transformer → MRI sequence data.
    11. MobileNet → CIFAR-100.
    12. EfficientNet → Flowers102.

## Week 9–10: Specialized & Hybrid Models
### Datasets: Graphs, speech, reinforcement learning
    1. Capsule Networks → MNIST.
    2. Siamese Network → Omniglot (few-shot).
    3. Triplet Network → Face dataset (LFW).
    4. GCN → Cora Citation Graph.
    5. GAT → Pubmed Graph.
    6. Spatio-Temporal GNN → Traffic dataset.
    7. Complex GNN → Graph of signals (synthetic).
    8. Seq2Seq + Attention → WMT14 Translation.
    9. Speech RNN → LibriSpeech dataset.
    10. Complex Spectrogram CNN → Audio spectrogram (UrbanSound8K).
    11. DQN → CartPole.
    12. Policy Gradient → MountainCar.

## Week 11–12: Advanced Topics & Applications
### Datasets: Cutting-edge tasks
    1. Diffusion Model (DDPM) → MNIST.
    2. Improved Diffusion (U-Net backbone) → CIFAR-10.
    3. StyleGAN → CelebA-HQ.
    4. SimCLR → CIFAR-10.
    5. BERT → WikiText-2.
    6. GPT-small → Shakespeare text.
    7. Complex Embeddings → WordNet embeddings.
    8. CLIP → Image-Text dataset (COCO small).
    9. Multi-modal Fusion → Image + audio dataset.
    10. Complex Fusion → Audio spectrogram + MRI.
    11. Federated Learning (FedAvg) → MNIST split across clients.
    12. MAML (Meta-Learning) → Omniglot few-shot.
    13. Neural ODE → Synthetic trajectories.
    14. Complex Neural ODE → Lorenz system (complex form).
    15. PINN → Solve PDE for heat equation.
    16. Complex PINN → Schrödinger equation.
    17. Final Complex Project → fastMRI or communications dataset.
    18. Wrap-up → Report: compare real vs complex architectures.




# Tips for Datasets
    • Tabular: UCI datasets, sklearn built-ins.
    • Images: MNIST, Fashion-MNIST, CIFAR-10/100, Tiny ImageNet, CelebA.
    • Text: IMDB, Shakespeare, WMT.
    • Graphs: Cora, Pubmed.
    • Complex Data:
        ○ Synthetic: sinusoidal, QPSK, QAM signals.
        ○ MRI: fastMRI.
        ○ Spectrograms: LibriSpeech, UrbanSound8K.
        ○ Radio signals: RML2016 dataset.


## This way, every day you:
    1. Implement real-valued model on dataset.
    2. Implement complex-valued version on synthetic/MRI/signal dataset.
    3. Compare performance + document.
