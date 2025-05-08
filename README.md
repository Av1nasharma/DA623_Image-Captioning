# Image Captioning with CNNs and LSTMs

### By \[AVINASH SHARMA]

---
### Notebook along with Blog
[View Full Notebook along with Blog](https://github.com/Av1nasharma/DA623_Image-Captioning/blob/main/Image_Captioning.ipynb)

---

## Motivation

I chose the topic of **Image Captioning** because it is a classic yet actively evolving problem in **Multimodal AI**. It sits at the intersection of computer vision and natural language processing, demanding effective collaboration between both modalities. The challenge of converting visual inputs into descriptive language is not only technically stimulating but also highly applicable in areas like accessibility (e.g., generating alt text), content moderation, and autonomous systems.

---

## Connection with Multimodal Learning: A Short Historical Perspective

Multimodal learning refers to models that integrate and process data from multiple modalities, such as text, images, and audio. Image captioning began with **template-based approaches**, but with the success of **deep learning**, especially **CNNs for image encoding** and **RNNs (particularly LSTMs) for decoding**, the field advanced significantly.

### Notable Milestones:

* **2014**: Vinyals et al. proposed the Show and Tell model that used InceptionNet + LSTM.
* **2015–2017**: Attention mechanisms, such as in Show, Attend and Tell (Xu et al.), enhanced performance by focusing on salient parts of the image.
* **2020+**: Transformer-based models like ViLT, Oscar, and CLIP showed the power of large pretraining on multimodal data.
* **Now**: Vision-Language foundation models like **GPT-4V**, **BLIP-2**, and **Flamingo** are redefining the capabilities of multimodal understanding and generation.

Our project fits within the Vinyals-style encoder-decoder lineage and forms a stepping stone for understanding more complex models.

---

## Learning from This Work

By implementing image captioning using CNNs and LSTMs, I gained:

1. **Deeper understanding of encoder-decoder architectures**.
2. Practical exposure to **tokenization**, **sequence padding**, and **teacher forcing** during training.
3. Insights into **dataset preparation** (Flickr8k), including caption cleaning and vocabulary construction.
4. Experience with **evaluating generative models** using metrics like **BLEU** and visual inspections.

---

## Code & Experiments

We used the Flickr8k dataset, which includes over 8,000 images each paired with five human-generated captions.

### Key Components:

* **CNN Encoder**: Pre-trained VGG16 used to extract image features.
* **LSTM Decoder**: Trained to generate captions conditioned on the encoded image vector.
* **Tokenizer**: Used to convert words into indexed sequences.
* **Training Loop**: Teacher forcing used to speed up convergence.

#### Sample Caption Generation:

```python
image_path = 'example.jpg'
predicted_caption = generate_caption(image_path)
print("Predicted Caption:", predicted_caption)
```

#### Actual vs. Predicted Captions:

| True Caption             | Predicted Caption         |
| ------------------------ | ------------------------- |
| A boy is playing soccer. | A child kicks a ball.     |
| A man is riding a horse. | A person rides an animal. |

---

#### Training Loss and Validation Loss v/s Epochs
![image](https://github.com/user-attachments/assets/be2385a7-a980-4c44-a993-fe6a2d007902)

---

## Reflections

### What Surprised Me?

* The **variability of human captions** made exact matching a difficult metric.
* Simple models like CNN+LSTM still produce **semantically reasonable captions**.
* Generating captions on even a small dataset required substantial memory and training time.

### Scope for Improvement

* Replace LSTM with a **Transformer decoder** for better contextual understanding.
* Use **attention mechanisms** to improve word choice.
* Apply **BLEU or CIDEr metrics** for more nuanced evaluation.
* Experiment with **larger or pre-aligned datasets** like MS-COCO.

---

## References

* Vinyals et al., ["Show and Tell: A Neural Image Caption Generator"](https://arxiv.org/abs/1411.4555)
* Xu et al., ["Show, Attend and Tell"](https://arxiv.org/abs/1502.03044)
* Karpathy & Fei-Fei, ["Deep Visual-Semantic Alignments"](https://arxiv.org/abs/1412.2306)
* [Flickr8k Dataset](https://forms.illinois.edu/sec/1713398)
* TensorFlow & Keras documentation
* ChatGPT + Colab for interactive development

---

*This project serves as a foundational exploration into vision-language modeling and can be extended with modern transformer-based methods in future work.*
