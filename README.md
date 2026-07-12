# Knowledge-distillation for Edge Devices
Transferring representations between Vision Transformers (ViTs) and
Convolutional Neural Networks (CNNs) remains an open challenge, driven by the
mismatch between global self-attention and local convolutional receptive
fields. The gap is acute in medical imaging, where CNNs are the only
computationally viable option at the edge yet lack the global context that many
diagnoses require. We present a cross-architecture knowledge distillation (KD)
framework in which a ViT-B/16 teacher shapes a lightweight EfficientNet-B0
student ($21.3\times$ fewer parameters, $42.4\times$ fewer MACs), and we
characterise the distilled student as a full deployment envelope rather than as
a single accuracy number. On CheXpert, logit-level distillation reaches a macro
AUROC of $0.8646$ over the five competition pathologies, recovering $99.7\%$ of
the teacher's $0.8675$ ceiling and improving the undistilled baseline by
$+0.069$ ($p<10^{-4}$, paired bootstrap). A complete precision study
(FP32/FP16/INT8) on an RTX~5070~Ti and a Jetson Orin Nano shows that FP16 is
lossless, that INT8 tolerance is a property of soft-logit supervision
\emph{specifically} rather than of distillation in general, and that blur and
quantization robustness are anti-correlated, mechanistically distinct axes. The
Jetson attains $\sim$$70$\,img/s/W at INT8 and hosts the entire accuracy--energy
Pareto frontier, with Logit-KD FP16 dominating at $22.9$\,mJ per image.
